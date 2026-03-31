"""
级联设备构网工具（cascade-only）：输出连续 Petri 固定拓扑与动态路由装配结果。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from solutions.A.construct.build_marks import build_marks_for_single_net
from solutions.A.construct.build_takt import build_takt_payload
from solutions.A.construct.build_topology import get_topology
from solutions.A.construct.preprocess_config import preprocess_chamber_runtime_blocks
from solutions.A.construct.build_route_queue import (
    build_token_route_queue,
    build_token_route_queue_multi,
)
from solutions.A.construct.route_compiler_single import (
    RobotSpec as CompiledRobotSpec,
    RouteIR,
    build_route_meta_from_route_ir,
    compile_route_stages,
    first_load_port_name,
)

BUFFER_NAMES: Set[str] = {"LLC", "LLD"}




def _build_route_source_target_transport(route_ir: RouteIR) -> Dict[Tuple[str, str], str]:
    """按当前 route 生成 (source,target)->transport 映射，用于动态选择 u 变迁。"""
    mapping: Dict[Tuple[str, str], str] = {}
    for hop in route_ir.transports:
        src_stage = route_ir.stages[hop.from_stage_idx]
        dst_stage = route_ir.stages[hop.to_stage_idx]
        transport_name = str(hop.transport_place).replace("d_", "")
        for src in src_stage.candidates:
            for dst in dst_stage.candidates:
                mapping[(str(src), str(dst))] = transport_name
    return mapping


def _lp_per_token_for_route(
    token_route_type_sequence: Sequence[int],
    wafer_type_to_subpath: Mapping[int, str],
    route_irs_by_name: Mapping[str, RouteIR],
) -> Tuple[str, ...]:
    out: List[str] = []
    for t in token_route_type_sequence:
        sp = wafer_type_to_subpath[int(t)]
        out.append(first_load_port_name(route_irs_by_name[sp]))
    return tuple(out)


def _merge_route_source_target_transport(
    route_irs: Sequence[RouteIR],
) -> Dict[Tuple[str, str], str]:
    merged: Dict[Tuple[str, str], str] = {}
    for route_ir in route_irs:
        current = _build_route_source_target_transport(route_ir)
        for key, val in current.items():
            if key in merged and merged[key] != val:
                raise ValueError(f"conflicting transport mapping for hop {key}: {merged[key]} vs {val}")
            merged[key] = val
    return merged


def _build_token_type_sequence(
    n_wafer: int,
    wafer_type_alloc_by_type: Mapping[int, int],
    lp_release_pattern_types: Sequence[int],
) -> List[int]:
    if n_wafer <= 0:
        return []
    if lp_release_pattern_types:
        pattern = [int(t) for t in lp_release_pattern_types if int(t) > 0]
        if not pattern:
            raise ValueError("lp_release_pattern_types must contain positive type ids")
        return [pattern[i % len(pattern)] for i in range(int(n_wafer))]

    weights = {int(k): max(0, int(v)) for k, v in wafer_type_alloc_by_type.items() if int(k) > 0}
    if not weights:
        return [1 for _ in range(int(n_wafer))]
    total = sum(weights.values())
    if total <= 0:
        first_type = sorted(weights.keys())[0]
        return [int(first_type) for _ in range(int(n_wafer))]
    base_counts: Dict[int, int] = {}
    fractions: List[Tuple[float, int]] = []
    assigned = 0
    for type_id, weight in sorted(weights.items()):
        raw = float(n_wafer) * float(weight) / float(total)
        cnt = int(raw)
        base_counts[type_id] = cnt
        assigned += cnt
        fractions.append((raw - cnt, type_id))
    rest = int(n_wafer) - assigned
    fractions.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    for i in range(rest):
        _, tid = fractions[i % len(fractions)]
        base_counts[tid] += 1

    sequence: List[int] = []
    rr_types = [tid for tid in sorted(base_counts.keys()) if base_counts[tid] > 0]
    while len(sequence) < int(n_wafer):
        progressed = False
        for tid in rr_types:
            if base_counts[tid] <= 0:
                continue
            sequence.append(int(tid))
            base_counts[tid] -= 1
            progressed = True
            if len(sequence) >= int(n_wafer):
                break
        if not progressed:
            break
    if len(sequence) < int(n_wafer):
        fill_type = rr_types[0] if rr_types else 1
        sequence.extend([int(fill_type)] * (int(n_wafer) - len(sequence)))
    return sequence[: int(n_wafer)]



def parse_route(
    stages: List[List[str]],
    buffer_names: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """
    从路线 stages 解析路由元数据。

    stages: 阶段序列，如 [["PM1"], ["PM3","PM4"], ["PM6"]] 表示 LP->PM1->[PM3,PM4]->PM6->LP_done
    buffer_names: 缓冲库所（如 LLC），不计入 chambers，但计入 timeline_chambers

    Returns:
        chambers, timeline_chambers, u_targets, step_map,
        release_station_aliases, release_chain_by_u, system_entry_places
    """
    buffer_names = buffer_names or BUFFER_NAMES

    def _is_buffer_stage(stage: List[str]) -> bool:
        return len(stage) == 1 and stage[0] in buffer_names

    # release_station_aliases: s1=stage[0], s2=stage[1], ...
    release_station_aliases: Dict[str, List[str]] = {}
    for i, stage in enumerate(stages):
        release_station_aliases[f"s{i + 1}"] = list(stage)

    # chambers: 按序展开，排除 buffer stage
    chamber_list: List[str] = []
    for stage in stages:
        if not _is_buffer_stage(stage):
            chamber_list.extend(stage)
    chambers = tuple(chamber_list)

    # timeline_chambers: chambers + 路径中的 buffer
    buffers_in_route = [
        s[0] for s in stages if _is_buffer_stage(s)
    ]
    timeline_chambers = chambers + tuple(buffers_in_route)

    # step_map: 按 stage 序分配，LP_done = len(stages)+1
    step_map: Dict[str, int] = {"LP_done": len(stages) + 1}
    step = 1
    for stage in stages:
        for place in stage:
            step_map[place] = step
        step += 1

    # u_targets: stage[i] 中每点 -> stage[i+1]；最后 stage -> [LP_done]
    u_targets: Dict[str, List[str]] = {}
    # LP -> stage[0]
    if stages:
        u_targets["LP"] = list(stages[0])

    for i, stage in enumerate(stages):
        next_stage = stages[i + 1] if i + 1 < len(stages) else ["LP_done"]
        for place in stage:
            u_targets[place] = list(next_stage)

    # system_entry_places: stage[0] 的 place 集合
    system_entry_places = set(stages[0]) if stages else set()

    # release_chain_by_u: 释放点 u_* 的下游 s_n 链
    # 释放点: u_LP（投放到 stage 0），以及每个 buffer 后 unload 的 u_*（投放到 buffer 的下一 stage）
    release_chain_by_u: Dict[str, List[str]] = {}
    if stages:
        # u_LP 投放 à stage 0，链为 s1 到 s_n
        release_chain_by_u["u_LP"] = [f"s{k}" for k in range(1, len(stages) + 1)]

    for i, stage in enumerate(stages):
        if _is_buffer_stage(stage):
            # buffer 如 LLC，u_LLC 投放 à 下一 stage，链为从 s_{i+2} 到 s_n
            buffer_name = stage[0]
            u_name = f"u_{buffer_name}"
            chain = [f"s{k}" for k in range(i + 2, len(stages) + 1)]
            if chain:
                release_chain_by_u[u_name] = chain

    # u_LLD: 从 LLD unload 投放到下一 stage（PM9/PM10 或 LP_done）
    for i, stage in enumerate(stages):
        if stage == ["LLD"] and i + 1 < len(stages):
            release_chain_by_u["u_LLD"] = [f"s{i + 2}"]  # s_{i+2} 对应 stages[i+1]
            break
        if stage == ["LLD"] and i + 1 >= len(stages):
            # route 3: LLD 为最后一 stage，无 s5
            break

    # 修正 release_chain_by_u：与现有一致
    # u_LP: [s1, s2]（cascade）或 [s1,s2] 或 [s1,s2,s3]（single）
    # u_LLC: [s3, s4]
    # u_LLD: [s5] 仅当存在 s5 时
    if len(stages) >= 2 and _is_buffer_stage(stages[1]):
        release_chain_by_u["u_LP"] = ["s1", "s2"]
    if len(stages) >= 4 and _is_buffer_stage(stages[1]):
        release_chain_by_u["u_LLC"] = ["s3", "s4"]
    if len(stages) >= 5:
        release_chain_by_u["u_LLD"] = ["s5"]

    return {
        "chambers": chambers,
        "timeline_chambers": timeline_chambers,
        "u_targets": u_targets,
        "step_map": step_map,
        "release_station_aliases": release_station_aliases,
        "release_chain_by_u": release_chain_by_u,
        "system_entry_places": system_entry_places,
    }


def build_net(n_wafer: int,
              ttime: int = 5,
              obs_config: Optional[Dict[str, Any]] = None,
              route_config: Optional[Mapping[str, Any]] = None,
              route_name: Optional[str] = None,
              n_wafer_route1: Optional[int] = None,
              n_wafer_route2: Optional[int] = None) -> Dict[str, object]:
    """
    构建 cascade-only 固定拓扑 Petri 网结构（route_config 驱动）。

    约束：
    - route_config 必填，schema 需包含 source/sink/chambers/robots/routes
    - route_name 用于选择具体路线

    返回 dict 除 pre/pst/m0/md/marks/id2p_name/id2t_name 等外，还包含预计算索引（供 get_enable_t/_fire 复用）：
    - pre_place_indices: List[np.ndarray]，pre_place_indices[t] 为变迁 t 的前置库所下标
    - pst_place_indices: List[np.ndarray]，pst_place_indices[t] 为变迁 t 的后置库所下标
    - transport_pre_place_idx: List[int]，transport_pre_place_idx[t] 为变迁 t 的运输位前置库所下标（无则为 -1）
    - process_time_map: Dict[str, int]，与 route_meta["chambers"] 一致的腔室工序时长（已含默认填充与取整到 5 秒）
    """
    source_cfg = dict(route_config.get("source") or {"name": "LP1"})
    sink_cfg = dict(route_config.get("sink") or {"name": "LP_done"})
    source_name = str(source_cfg.get("name", "LP1"))
    if source_name == "LP":
        source_name = "LP1"
    sink_name = str(sink_cfg.get("name", "LP_done"))
    if sink_name != "LP_done":
        raise ValueError("cascade fixed topology requires sink=LP_done")
    if source_name not in ("LP1", "LP2"):
        raise ValueError("cascade fixed topology requires source.name LP/LP1/LP2")
    selected_route_name = route_name

    chambers_cfg = dict(route_config.get("chambers") or {})
    chamber_kind_map: Dict[str, str] = {}
    for cname, cfg in chambers_cfg.items():
        chamber_kind_map[str(cname)] = str((cfg or {}).get("kind", "process"))

    robots_cfg_raw = dict(route_config.get("robots") or {})
    robots_cfg: Dict[str, CompiledRobotSpec] = {}
    for rb_name, rb in robots_cfg_raw.items():
        managed = tuple(
            "LP1" if str(x) == "LP" else str(x) for x in (rb or {}).get("managed_chambers", ())
        )
        transport_place = str((rb or {}).get("transport_place", str(rb_name))).replace("d_", "")
        robots_cfg[str(rb_name)] = CompiledRobotSpec(
            name=str(rb_name),
            managed_chambers=managed,
            transport_place=transport_place,
            priority=int((rb or {}).get("priority", 0)),
        )

    route_entry = dict((route_config.get("routes") or {}).get(selected_route_name) or {})
    subpaths_cfg_raw = route_entry.get("subpaths")
    route_irs_by_name: Dict[str, RouteIR] = {}
    if isinstance(subpaths_cfg_raw, Mapping) and subpaths_cfg_raw:
        subpath_items = list(subpaths_cfg_raw.items())
        for idx, (subpath_name, subpath_cfg) in enumerate(subpath_items):
            name = str(subpath_name)
            sub_raw = dict(subpath_cfg or {})
            sub_source = str(sub_raw.pop("source_name", "") or "").strip()
            if not sub_source:
                if len(subpath_items) == 2:
                    sub_source = "LP1" if idx == 0 else "LP2"
                else:
                    sub_source = source_name
            if sub_source == "LP":
                sub_source = "LP1"
            route_irs_by_name[name] = compile_route_stages(
                route_name=f"{selected_route_name}:{name}",
                route_cfg=sub_raw,
                source_name=sub_source,
                sink_name=sink_name,
                chamber_kind_map=chamber_kind_map,
                robots=robots_cfg,
            )
    else:
        route_irs_by_name["default"] = compile_route_stages(
            route_name=selected_route_name,
            route_cfg=route_entry,
            source_name=source_name,
            sink_name=sink_name,
            chamber_kind_map=chamber_kind_map,
            robots=robots_cfg,
        )
    default_subpath_name = next(iter(route_irs_by_name.keys()))
    route_ir = route_irs_by_name[default_subpath_name]

    preprocess_result = preprocess_chamber_runtime_blocks(
        route_ir=route_ir,
        route_config=route_config,
        source_name=source_name,
        sink_name=sink_name,
        route_name=selected_route_name,
    )
    chamber_blocks = preprocess_result.chamber_blocks
    buffer_names = preprocess_result.buffer_names
    if len(route_irs_by_name) >= 2:
        for subpath_name, subpath_ir in route_irs_by_name.items():
            for stage in subpath_ir.stages:
                is_process_stage = str(stage.stage_type) == "process"
                for chamber_name in stage.candidates:
                    if chamber_name in {"LP1", "LP2", sink_name}:
                        continue
                    block = chamber_blocks.get(str(chamber_name))
                    if block is None:
                        continue
                    if stage.stage_process_time is not None:
                        p_val = float(stage.stage_process_time)
                        if (not is_process_stage) or p_val > 0:
                            new_ptime = int(round(p_val))
                            if block.process_time > 0 and new_ptime > 0 and block.process_time != new_ptime:
                                raise ValueError(
                                    f"route {selected_route_name} subpath {subpath_name} has conflicting process_time "
                                    f"for {chamber_name}: {block.process_time} vs {new_ptime}"
                                )
                            block.process_time = int(new_ptime)
                    if stage.stage_cleaning_duration is not None:
                        block.cleaning_duration = int(stage.stage_cleaning_duration)
                    if stage.stage_cleaning_trigger_wafers is not None:
                        block.cleaning_trigger_wafers = int(stage.stage_cleaning_trigger_wafers)

    static_topology = get_topology()
    id2p_name = list(static_topology["id2p_name"])
    all_t_names = list(static_topology["id2t_name"])
    all_pre = np.array(static_topology["pre"], dtype=int)
    all_pst = np.array(static_topology["pst"], dtype=int)
    all_t_target_place = dict(static_topology["t_target_place"])
    transition_id: Dict[Tuple[str, str], str] = dict(static_topology["transition_id"])

    # ===== 根据路径信息动态选择变迁 =====
    route_ir_list = list(route_irs_by_name.values())
    route_source_target_transport = _merge_route_source_target_transport(route_ir_list)
    active_u_names: Set[str] = set()
    active_t_names: Set[str] = set()
    for (src, dst), transport in route_source_target_transport.items():
        active_u_names.add(f"u_{src}_{transport}")
        active_t_names.add(transition_id[(str(transport), str(dst))])

    # 筛选出路径中设计的变迁，对t_*变迁来说，记录其目标
    selected_t_indices: List[int] = []
    id2t_name: List[str] = []
    t_target_place: Dict[str, str] = {}
    for idx, t_name in enumerate(all_t_names):
        if t_name.startswith("u_") and t_name in active_u_names:
            selected_t_indices.append(idx)
            id2t_name.append(t_name)
            continue
        if t_name.startswith("t_") and t_name in active_t_names:
            selected_t_indices.append(idx)
            id2t_name.append(t_name)
            target = all_t_target_place.get(t_name)
            if target is not None:
                t_target_place[t_name] = str(target)

    if not id2t_name:
        raise ValueError("route produced empty transition set")

    pre = np.array(all_pre[:, selected_t_indices], dtype=int)
    pst = np.array(all_pst[:, selected_t_indices], dtype=int)
    t_count = len(id2t_name)

    # ======= 构造晶圆路由队列 ===========
    token_route_queue_templates: Dict[str, Tuple[object, ...]] = {}
    token_route_plan_templates: Dict[str, Any] = {}
    token_route_queue_by_type: Dict[int, Tuple[object, ...]] = {}
    token_route_type_sequence: List[int] = [1 for _ in range(int(n_wafer))]
    subpath_to_type: Dict[str, int] = {default_subpath_name: 1}
    wafer_type_to_subpath: Dict[int, str] = {1: default_subpath_name}
    wafer_type_alloc_by_type: Dict[int, int] = {1: int(n_wafer)}
    lp_release_pattern_types: Tuple[int, ...] = tuple()
    if len(route_irs_by_name) >= 2:
        multi_payload = build_token_route_queue_multi(
            route_irs=route_irs_by_name,
            id2t_name=id2t_name,
            t_target_place=t_target_place,
            wafer_type_alloc=dict(route_entry.get("wafer_type_alloc") or {}),
            lp_release_pattern=list(route_entry.get("lp_release_pattern") or []),
        )
        t_route_code_map = dict(multi_payload["t_route_code_map"])
        token_route_queue = tuple(multi_payload["token_route_queue_template"])
        token_plan = multi_payload["token_route_plan_template"]
        token_route_queue_templates = dict(multi_payload["token_route_queue_templates"])
        token_route_plan_templates = dict(multi_payload["token_route_plan_templates"])
        subpath_to_type = dict(multi_payload["subpath_to_type"])
        wafer_type_to_subpath = dict(multi_payload["wafer_type_to_subpath"])
        wafer_type_alloc_by_type = dict(multi_payload["wafer_type_alloc_by_type"])
        lp_release_pattern_types = tuple(multi_payload["lp_release_pattern_types"])
        token_route_queue_by_type = {
            int(type_id): tuple(token_route_queue_templates[subpath_name])
            for type_id, subpath_name in wafer_type_to_subpath.items()
        }
        token_route_type_sequence = _build_token_type_sequence(
            n_wafer=int(n_wafer),
            wafer_type_alloc_by_type=wafer_type_alloc_by_type,
            lp_release_pattern_types=lp_release_pattern_types,
        )
    else:
        _, t_route_code_map, token_route_queue, token_plan = build_token_route_queue(
            route_ir=route_ir,
            id2t_name=id2t_name,
            t_target_place=t_target_place,
        )
        token_route_queue_templates = {default_subpath_name: tuple(token_route_queue)}
        token_route_plan_templates = {default_subpath_name: token_plan}
        token_route_queue_by_type = {1: tuple(token_route_queue)}

    lp_per_token = _lp_per_token_for_route(
        token_route_type_sequence=token_route_type_sequence,
        wafer_type_to_subpath=wafer_type_to_subpath,
        route_irs_by_name=route_irs_by_name,
    )
    c_lp = Counter(lp_per_token)
    if n_wafer_route1 is not None and n_wafer_route2 is not None:
        if int(n_wafer_route1) + int(n_wafer_route2) != int(n_wafer):
            raise ValueError(
                f"n_wafer_route1+n_wafer_route2 must equal n_wafer: "
                f"{n_wafer_route1}+{n_wafer_route2}!={n_wafer}"
            )
        if c_lp.get("LP1", 0) != int(n_wafer_route1) or c_lp.get("LP2", 0) != int(n_wafer_route2):
            raise ValueError(
                f"LP token distribution {dict(c_lp)} does not match "
                f"n_wafer_route1/2=({n_wafer_route1},{n_wafer_route2})"
            )

    p_idx = {name: i for i, name in enumerate(id2p_name)}

    marks_result = build_marks_for_single_net(
        id2p_name=id2p_name,
        source_name=source_name,
        sink_name=sink_name,
        chamber_blocks=chamber_blocks,
        n_wafer=n_wafer,
        token_route_queue=token_route_queue,
        token_route_queue_by_type=token_route_queue_by_type,
        token_route_type_sequence=token_route_type_sequence,
        lp_per_token=lp_per_token,
        obs_config=obs_config,
        ttime=ttime,
    )
    m0 = marks_result.m0
    md = marks_result.md
    ptime = marks_result.ptime
    capacity = marks_result.capacity
    marks = marks_result.marks
    process_time_map_out = marks_result.process_time_map_out
    ttime_arr = np.array([ttime for _ in range(t_count)], dtype=int)

    route_meta = build_route_meta_from_route_ir(route_ir, buffer_names=buffer_names or BUFFER_NAMES)
    route_meta["route_stages"] = [list(stage.candidates) for stage in route_ir.stages[1:-1]]
    if len(route_irs_by_name) >= 2:
        for subpath_name, subpath_ir in route_irs_by_name.items():
            sub_meta = build_route_meta_from_route_ir(subpath_ir, buffer_names=buffer_names or BUFFER_NAMES)
            for src, tgts in (sub_meta.get("u_targets") or {}).items():
                arr = route_meta["u_targets"].setdefault(str(src), [])
                for dst in list(tgts):
                    if dst not in arr:
                        arr.append(dst)
            rcb = sub_meta.get("release_chain_by_u") or {}
            for k, v in rcb.items():
                route_meta["release_chain_by_u"][k] = list(v)
        route_meta["multi_subpath"] = True
    else:
        route_meta["multi_subpath"] = False
    has_repeat_syntax_reentry = any(stage.repeat_origin is not None for stage in route_ir.stages)
    route_meta["has_repeat_syntax_reentry"] = bool(has_repeat_syntax_reentry)
    route_meta["subpath_to_type"] = subpath_to_type
    route_meta["wafer_type_to_subpath"] = wafer_type_to_subpath
    route_meta["wafer_type_alloc_by_type"] = wafer_type_alloc_by_type
    route_meta["lp_release_pattern_types"] = list(lp_release_pattern_types)
    route_meta["takt_policy"] = str(route_entry.get("takt_policy", "") or "")
    route_meta["takt_stages_override"] = list(route_entry.get("takt_stages_override") or [])
    route_meta["default_subpath"] = str(default_subpath_name)
    route_meta["subpath_route_stages"] = {
        str(name): [list(stage.candidates) for stage in ir.stages[1:-1]]
        for name, ir in route_irs_by_name.items()
    }
    full_timeline_chambers = tuple(
        name for name in id2p_name
        if (name.startswith("PM") or name in {"LLC", "LLD"})
    )
    route_meta["chambers"] = full_timeline_chambers
    route_meta["timeline_chambers"] = full_timeline_chambers
    route_meta["wafer_type_to_load_port"] = {
        int(tid): first_load_port_name(route_irs_by_name[sp])
        for tid, sp in wafer_type_to_subpath.items()
    }
    route_meta["cleaning_duration_map"] = {
        str(name): int(block.cleaning_duration) for name, block in chamber_blocks.items()
    }
    route_meta["cleaning_trigger_wafers_map"] = {
        str(name): int(block.cleaning_trigger_wafers) for name, block in chamber_blocks.items()
    }
    route_meta["load_port_names"] = ("LP1", "LP2")
    aliases = dict(route_meta.get("release_station_aliases") or {})
    ordered_keys = sorted(
        [k for k in aliases.keys() if str(k).startswith("s")],
        key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 0,
    )
    route_stages = [list(aliases[k]) for k in ordered_keys]
    ctx = obs_config or {}
    takt_payload = build_takt_payload(
        route_stages=route_stages,
        base_proc_time_map=process_time_map_out,
        cleaning_enabled=bool(ctx.get("cleaning_enabled", False)),
        cleaning_duration=int(ctx.get("cleaning_duration", 150)),
        cleaning_duration_map=dict(ctx.get("cleaning_duration_map") or {}),
        cleaning_trigger_map=dict(ctx.get("cleaning_trigger_wafers_map") or {}),
        has_repeat_syntax_reentry=bool(has_repeat_syntax_reentry),
        multi_subpath=bool(route_meta.get("multi_subpath", False)),
        takt_policy=str(route_meta.get("takt_policy", "") or ""),
        wafer_type_to_subpath=wafer_type_to_subpath,
        subpath_route_stages=dict(route_meta.get("subpath_route_stages") or {}),
    )

    pre_place_indices: List[np.ndarray] = [np.flatnonzero(pre[:, t] > 0) for t in range(t_count)]
    pst_place_indices: List[np.ndarray] = [np.flatnonzero(pst[:, t] > 0) for t in range(t_count)]
    transport_pre_place_idx: List[int] = []
    for t in range(t_count):
        found = next(
            (int(idx) for idx in pre_place_indices[t] if id2p_name[int(idx)] in {"TM2", "TM3"}),
            -1,
        )
        transport_pre_place_idx.append(int(found))

    return {
        "m0": m0,
        "P": pre.shape[0],
        "T": pre.shape[1],
        "pre_place_indices": pre_place_indices,
        "pst_place_indices": pst_place_indices,
        "transport_pre_place_idx": transport_pre_place_idx,
        "ttime": ttime_arr,
        "capacity": capacity,
        "id2p_name": id2p_name,
        "id2t_name": id2t_name,
        "idle_idx": {"start": p_idx["LP1"], "end": p_idx[sink_name]},
        "marks": marks,
        "n_wafer": n_wafer,
        "n_wafer_route1": int(c_lp.get("LP1", 0)),
        "n_wafer_route2": int(c_lp.get("LP2", 0)),
        "route_meta": route_meta,
        "t_route_code_map": t_route_code_map,
        "t_target_place_map": t_target_place,
        "route_source_target_transport": route_source_target_transport,
        "token_route_queue_template": token_route_queue,
        "token_route_plan_template": token_plan,
        "token_route_queue_templates": token_route_queue_templates,
        "token_route_plan_templates": token_route_plan_templates,
        "token_route_queue_templates_by_type": token_route_queue_by_type,
        "token_route_type_sequence": token_route_type_sequence,
        "route_ir": route_ir,
        "process_time_map": process_time_map_out,
        "takt_payload": takt_payload,
        "fixed_topology": True,
    }
