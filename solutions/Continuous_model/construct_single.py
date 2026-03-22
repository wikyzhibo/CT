"""
级联设备构网工具（cascade-only）：输出连续 Petri 固定拓扑与动态路由装配结果。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
from solutions.Continuous_model.construct.build_marks import build_marks_for_single_net
from solutions.Continuous_model.construct.build_topology import get_topology
from solutions.Continuous_model.construct.preprocess_config import preprocess_chamber_runtime_blocks
from solutions.Continuous_model.construct.build_route_queue import build_token_route_queue
from solutions.Continuous_model.construct.route_compiler_single import (
    RobotSpec as CompiledRobotSpec,
    RouteIR,
    build_route_meta_from_route_ir,
    compile_route_stages,
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


def build_net(
    n_wafer: int,
    ttime: int = 5,
    robot_capacity: int = 1,
    process_time_map: Optional[Dict[str, int]] = None,
    route_code: int = 0,
    device_mode: str = "cascade",
    obs_config: Optional[Dict[str, Any]] = None,
    route_config: Optional[Mapping[str, Any]] = None,
    route_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    构建 cascade-only 固定拓扑 Petri 网结构（route_config 驱动）。

    约束：
    - route_config 必填，schema 需包含 source/sink/chambers/robots/routes
    - route_name 用于选择具体路线
    - route_code 仅用于兼容旧配置的路由别名选择
    - device_mode 必须是 cascade，single 路径已下线

    返回 dict 除 pre/pst/m0/md/marks/id2p_name/id2t_name 等外，还包含预计算索引（供 get_enable_t/_fire 复用）：
    - pre_place_indices: List[np.ndarray]，pre_place_indices[t] 为变迁 t 的前置库所下标
    - pst_place_indices: List[np.ndarray]，pst_place_indices[t] 为变迁 t 的后置库所下标
    - transport_pre_place_idx: List[int]，transport_pre_place_idx[t] 为变迁 t 的运输位前置库所下标（无则为 -1）
    - process_time_map: Dict[str, int]，与 route_meta["chambers"] 一致的腔室工序时长（已含默认填充与取整到 5 秒）
    """
    mode = str(device_mode).lower()
    route_code = int(route_code)
    if mode != "cascade":
        raise ValueError("build_net now supports cascade only")
    if route_config is None:
        raise ValueError("build_net requires route_config")
    process_time_map = dict(process_time_map or {})
    source_cfg = dict(route_config.get("source") or {"name": "LP"})
    sink_cfg = dict(route_config.get("sink") or {"name": "LP_done"})
    source_name = str(source_cfg.get("name", "LP"))
    sink_name = str(sink_cfg.get("name", "LP_done"))
    if source_name != "LP" or sink_name != "LP_done":
        raise ValueError(
            "cascade fixed topology requires source=LP and sink=LP_done"
        )
    selected_route_name = route_name

    chambers_cfg = dict(route_config.get("chambers") or {})
    chamber_kind_map: Dict[str, str] = {}
    for cname, cfg in chambers_cfg.items():
        chamber_kind_map[str(cname)] = str((cfg or {}).get("kind", "process"))

    robots_cfg_raw = dict(route_config.get("robots") or {})
    robots_cfg: Dict[str, CompiledRobotSpec] = {}
    for rb_name, rb in robots_cfg_raw.items():
        managed = tuple(str(x) for x in (rb or {}).get("managed_chambers", ()))
        transport_place = str((rb or {}).get("transport_place", str(rb_name))).replace("d_", "")
        robots_cfg[str(rb_name)] = CompiledRobotSpec(
            name=str(rb_name),
            managed_chambers=managed,
            transport_place=transport_place,
            priority=int((rb or {}).get("priority", 0)),
        )

    route_entry = dict((route_config.get("routes") or {}).get(selected_route_name) or {})
    route_ir = compile_route_stages(
        route_name=selected_route_name,
        route_cfg=route_entry,
        source_name=source_name,
        sink_name=sink_name,
        chamber_kind_map=chamber_kind_map,
        robots=robots_cfg,
    )

    ctx = obs_config or {}
    clean_dur_default = int(ctx.get("cleaning_duration", 150))
    clean_trig_default = int(ctx.get("cleaning_trigger_wafers", 5))
    preprocess_result = preprocess_chamber_runtime_blocks(
        route_ir=route_ir,
        route_config=route_config,
        process_time_map=process_time_map,
        source_name=source_name,
        sink_name=sink_name,
        route_name=selected_route_name,
        default_cleaning_duration=clean_dur_default,
        default_cleaning_trigger_wafers=clean_trig_default,
    )
    chamber_blocks = preprocess_result.chamber_blocks
    buffer_names = preprocess_result.buffer_names

    static_topology = get_topology()
    id2p_name = list(static_topology["id2p_name"])
    all_t_names = list(static_topology["id2t_name"])
    all_pre = np.array(static_topology["pre"], dtype=int)
    all_pst = np.array(static_topology["pst"], dtype=int)
    all_t_target_place = dict(static_topology["t_target_place"])
    transition_id: Dict[Tuple[str, str], str] = dict(static_topology["transition_id"])

    # ===== 根据路径信息动态选择变迁 =====
    route_source_target_transport = _build_route_source_target_transport(route_ir)
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
    _, t_route_code_map, token_route_queue, token_plan = build_token_route_queue(
        route_ir=route_ir,
        id2t_name=id2t_name,
        t_target_place=t_target_place,
    )

    p_idx = {name: i for i, name in enumerate(id2p_name)}

    marks_result = build_marks_for_single_net(
        id2p_name=id2p_name,
        source_name=source_name,
        sink_name=sink_name,
        chamber_blocks=chamber_blocks,
        n_wafer=n_wafer,
        token_route_queue=token_route_queue,
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
    full_timeline_chambers = tuple(
        name for name in id2p_name
        if (name.startswith("PM") or name in {"LLC", "LLD"})
    )
    route_meta["chambers"] = full_timeline_chambers
    route_meta["timeline_chambers"] = full_timeline_chambers

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
        "idle_idx": {"start": p_idx[source_name], "end": p_idx[sink_name]},
        "marks": marks,
        "n_wafer": n_wafer,
        "n_wafer_route1": n_wafer,
        "n_wafer_route2": 0,
        "single_route_code": route_code,
        "single_device_mode": mode,
        "route_meta": route_meta,
        "t_route_code_map": t_route_code_map,
        "t_target_place_map": t_target_place,
        "route_source_target_transport": route_source_target_transport,
        "token_route_queue_template": token_route_queue,
        "token_route_plan_template": token_plan,
        "route_ir": route_ir,
        "process_time_map": process_time_map_out,
        "fixed_topology": True,
    }
