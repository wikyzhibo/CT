"""
单设备构网工具：输出与现有连续 Petri 构网一致的结构化信息。
支持两种模板：
- single: 原单设备路径（可由 route_code 细分）
- cascade: 级联路径（route_code 可切换 1/2/3/4/5）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.pn import LL, PM, Place, SR, TM
from solutions.Continuous_model.route_compiler_single import (
    RobotSpec as CompiledRobotSpec,
    TokenRoutePlan,
    build_route_meta_from_route_ir,
    build_token_route_plan,
    compile_route_stages,
)


# 路线定义：stages 为阶段序列，每个阶段为单点或并行多点
ROUTE_SPECS: Dict[Tuple[str, int], List[List[str]]] = {
    ("single", 0): [["PM1"], ["PM3", "PM4"]],
    ("single", 1): [["PM1"], ["PM3", "PM4"], ["PM6"]],
    ("cascade", 1): [
        ["PM7", "PM8"],
        ["LLC"],
        ["PM1", "PM2", "PM3", "PM4"],
        ["LLD"],
        ["PM9", "PM10"],
    ],
    ("cascade", 2): [
        ["PM7", "PM8"],
        ["LLC"],
        ["PM1", "PM2"],
        ["LLD"],
        ["PM9", "PM10"],
    ],
    ("cascade", 3): [
        ["PM7", "PM8"],
        ["LLC"],
        ["PM1", "PM2"],
        ["LLD"],
    ],
    ("cascade", 4): [
        ["PM7"],
        ["PM8"],
        ["LLC"],
        ["LLD"],
    ],
    ("cascade", 5): [
        ["PM7", "PM8"],
        ["PM9", "PM10"],
    ],
}
BUFFER_NAMES: Set[str] = {"LLC"}

# 级联模式下固定 8 维 TM 去向 one-hot（逐目标编码，避免按目标组压缩导致信息丢失）
CASCADE_TM2_TARGET_ORDER: Tuple[str, ...] = (
    "PM7",
    "PM8",
    "PM9",
    "PM10",
    "LLC",
    "LLD",
    "LP_done",
    "LP",
)
CASCADE_TM3_TARGET_ORDER: Tuple[str, ...] = (
    "PM1",
    "PM2",
    "PM3",
    "PM4",
    "PM5",
    "PM6",
    "LLC",
    "LLD",
)
CASCADE_TM2_TARGET_ONEHOT: Dict[str, int] = {
    name: idx for idx, name in enumerate(CASCADE_TM2_TARGET_ORDER)
}
CASCADE_TM3_TARGET_ONEHOT: Dict[str, int] = {
    name: idx for idx, name in enumerate(CASCADE_TM3_TARGET_ORDER)
}

# t_* 路由码常量（仅用于 token 队头门控）
T_ROUTE_CODES_SINGLE: Dict[str, int] = {
    "t_PM1": 1,
    "t_PM3": 2,
    "t_PM4": 3,
    "t_PM6": 4,
    "t_LP_done": 5,
}
T_ROUTE_CODES_CASCADE: Dict[str, int] = {
    "t_PM7": 1,
    "t_PM8": 2,
    "t_LLC": 3,
    "t_PM1": 4,
    "t_PM2": 5,
    "t_PM3": 6,
    "t_PM4": 7,
    "t_LLD": 8,
    "t_PM9": 9,
    "t_PM10": 10,
    "t_LP_done": 11,
}


def _build_token_route_queue_and_t_code_map(
    mode: str,
    route_code: int,
) -> Tuple[Tuple[object, ...], Dict[str, int]]:
    """
    返回 token 路由队列模板与 t_* 路由码映射。
    约定：
    - 仅 t_* 需要路由门控，u_* 不做门控；
    - 但 token 每次 fire 都推进队头，因此 u_* 位置使用 -1 占位。
    """
    if mode == "cascade":
        t_codes = dict(T_ROUTE_CODES_CASCADE)
        if route_code == 1:
            queue = (
                -1, (t_codes["t_PM7"], t_codes["t_PM8"]),
                -1, t_codes["t_LLC"],
                -1, (t_codes["t_PM1"], t_codes["t_PM2"], t_codes["t_PM3"], t_codes["t_PM4"]),
                -1, t_codes["t_LLD"],
                -1, (t_codes["t_PM9"], t_codes["t_PM10"]),
                -1, t_codes["t_LP_done"],
            )
            valid_t = {"t_PM7", "t_PM8", "t_LLC", "t_PM1", "t_PM2", "t_PM3", "t_PM4", "t_LLD", "t_PM9", "t_PM10", "t_LP_done"}
        elif route_code == 2:
            queue = (
                -1, (t_codes["t_PM7"], t_codes["t_PM8"]),
                -1, t_codes["t_LLC"],
                -1, (t_codes["t_PM1"], t_codes["t_PM2"]),
                -1, t_codes["t_LLD"],
                -1, (t_codes["t_PM9"], t_codes["t_PM10"]),
                -1, t_codes["t_LP_done"],
            )
            valid_t = {"t_PM7", "t_PM8", "t_LLC", "t_PM1", "t_PM2", "t_LLD", "t_PM9", "t_PM10", "t_LP_done"}
        elif route_code == 4:
            cycle_unit = (
                t_codes["t_PM7"], -1,
                t_codes["t_PM8"], -1,
                t_codes["t_LLC"], -1,
                t_codes["t_LLD"], -1,
            )
            queue = (-1,) + cycle_unit * 5 + (t_codes["t_LP_done"],)
            valid_t = {"t_PM7", "t_PM8", "t_LLC", "t_LLD", "t_LP_done"}
        elif route_code == 5:
            queue = (
                -1, (t_codes["t_PM7"], t_codes["t_PM8"]),
                -1, (t_codes["t_PM9"], t_codes["t_PM10"]),
                -1, t_codes["t_LP_done"],
            )
            valid_t = {"t_PM7", "t_PM8", "t_PM9", "t_PM10", "t_LP_done"}
        else:
            queue = (
                -1, (t_codes["t_PM7"], t_codes["t_PM8"]),
                -1, t_codes["t_LLC"],
                -1, (t_codes["t_PM1"], t_codes["t_PM2"]),
                -1, t_codes["t_LLD"],
                -1, t_codes["t_LP_done"],
            )
            valid_t = {"t_PM7", "t_PM8", "t_LLC", "t_PM1", "t_PM2", "t_LLD", "t_LP_done"}
        return queue, {k: v for k, v in t_codes.items() if k in valid_t}

    t_codes = dict(T_ROUTE_CODES_SINGLE)
    if route_code == 1:
        queue = (
            -1, t_codes["t_PM1"],
            -1, (t_codes["t_PM3"], t_codes["t_PM4"]),
            -1, t_codes["t_PM6"],
            -1, t_codes["t_LP_done"],
        )
        valid_t = {"t_PM1", "t_PM3", "t_PM4", "t_PM6", "t_LP_done"}
    else:
        queue = (
            -1, t_codes["t_PM1"],
            -1, (t_codes["t_PM3"], t_codes["t_PM4"]),
            -1, t_codes["t_LP_done"],
        )
        valid_t = {"t_PM1", "t_PM3", "t_PM4", "t_LP_done"}
    return queue, {k: v for k, v in t_codes.items() if k in valid_t}


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


def _build_pre_color_from_route_queue(
    pre: np.ndarray,
    id2t_name: List[str],
    route_queue: Tuple[object, ...],
    t_route_code_map: Dict[str, int],
) -> np.ndarray:
    """
    从 token.route_queue 模板反推 pre_color（兼容字段）。
    """
    p_count, t_count = pre.shape
    max_where = max(1, len(route_queue)) + 1
    pre_color = np.zeros((p_count, t_count, max_where), dtype=int)
    for tid, t_name in enumerate(id2t_name):
        if t_name.startswith("u_"):
            pre_color[:, tid, :] = pre[:, tid][:, None]
            continue
        t_code = int(t_route_code_map.get(t_name, -1))
        allowed_where: List[int] = []
        for where_idx, gate in enumerate(route_queue):
            if gate == -1:
                continue
            if isinstance(gate, int):
                if gate == t_code:
                    allowed_where.append(where_idx)
            elif isinstance(gate, (tuple, list, set, frozenset)):
                if t_code in gate:
                    allowed_where.append(where_idx)
        for c in allowed_where:
            pre_color[:, tid, c] = pre[:, tid]
    return pre_color


def _legacy_route_sequence(mode: str, route_code: int) -> List[Dict[str, Any]]:
    """
    兼容层：由旧 route_code 生成新 sequence 结构（含 source/sink）。
    """
    if mode == "cascade":
        if route_code == 1:
            stages = [["LP"], ["PM7", "PM8"], ["LLC"], ["PM1", "PM2", "PM3", "PM4"], ["LLD"], ["PM9", "PM10"], ["LP_done"]]
        elif route_code == 2:
            stages = [["LP"], ["PM7", "PM8"], ["LLC"], ["PM1", "PM2"], ["LLD"], ["PM9", "PM10"], ["LP_done"]]
        elif route_code == 3:
            stages = [["LP"], ["PM7", "PM8"], ["LLC"], ["PM1", "PM2"], ["LLD"], ["LP_done"]]
        elif route_code == 4:
            return [
                {"stage": {"candidates": ["LP"]}},
                {
                    "repeat": {
                        "count": 5,
                        "sequence": [
                            {"stage": {"candidates": ["PM7"]}},
                            {"stage": {"candidates": ["PM8"]}},
                            {"stage": {"candidates": ["LLC"]}},
                            {"stage": {"candidates": ["LLD"]}},
                        ],
                    }
                },
                {"stage": {"candidates": ["LP_done"]}},
            ]
        else:
            stages = [["LP"], ["PM7", "PM8"], ["PM9", "PM10"], ["LP_done"]]
    else:
        if route_code == 1:
            stages = [["LP"], ["PM1"], ["PM3", "PM4"], ["PM6"], ["LP_done"]]
        else:
            stages = [["LP"], ["PM1"], ["PM3", "PM4"], ["LP_done"]]
    return [{"stage": {"candidates": s}} for s in stages]


def _legacy_route_config(
    mode: str,
    route_code: int,
    process_time_map: Mapping[str, int],
    robot_capacity: int,
    ttime: int,
) -> Dict[str, Any]:
    """
    兼容层：将旧 route_code 输入映射为配置驱动 schema。
    """
    legacy_process_defaults = {
        "single": {
            "PM1": int(process_time_map.get("PM1", 100)),
            "PM2": 0,
            "PM3": int(process_time_map.get("PM3", 300)),
            "PM4": int(process_time_map.get("PM4", 300)),
            "PM6": int(process_time_map.get("PM6", 300 if route_code == 1 else 0)),
        },
        "cascade": {
            "PM7": int(process_time_map.get("PM7", 70)),
            "PM8": int(process_time_map.get("PM8", 70)),
            "PM1": int(process_time_map.get("PM1", 600 if route_code == 1 else 300)),
            "PM2": int(process_time_map.get("PM2", 600 if route_code == 1 else 300)),
            "PM3": int(process_time_map.get("PM3", 600)),
            "PM4": int(process_time_map.get("PM4", 600)),
            "LLC": int(process_time_map.get("LLC", 0)),
            "LLD": int(process_time_map.get("LLD", 70)),
            "PM9": int(process_time_map.get("PM9", 200)),
            "PM10": int(process_time_map.get("PM10", 200)),
        },
    }
    pt_map = legacy_process_defaults["cascade" if mode == "cascade" else "single"]

    chambers: Dict[str, Dict[str, Any]] = {}
    for name, ptime in pt_map.items():
        if name in {"LLC", "LLD"}:
            kind = "buffer" if name == "LLC" else "loadlock"
            cls = "LL"
        elif name == "PM2" and mode != "cascade":
            kind = "buffer"
            cls = "Place"
        elif name == "PM6" and mode != "cascade" and route_code != 1:
            kind = "buffer"
            cls = "Place"
        else:
            kind = "process"
            cls = "PM"
        chambers[name] = {
            "kind": kind,
            "class": cls,
            "process_time": int(ptime),
            "capacity": 1,
            "cleaning_duration": 0,
            "cleaning_trigger_wafers": 0,
            "proc_rand_scale": 0.0,
        }

    if mode == "cascade":
        robots = {
            "TM2": {
                "transport_place": "d_TM2",
                "managed_chambers": ["LP", "PM7", "PM8", "LLD", "PM9", "PM10", "LP_done", "LLC"],
                "dwell_time": int(ttime),
                "capacity": int(robot_capacity),
                "priority": 10,
            },
            "TM3": {
                "transport_place": "d_TM3",
                "managed_chambers": ["LLC", "PM1", "PM2", "PM3", "PM4", "LLD"],
                "dwell_time": int(ttime),
                "capacity": int(robot_capacity),
                "priority": 20,
            },
        }
    else:
        robots = {
            "TM1": {
                "transport_place": "d_TM1",
                "managed_chambers": ["LP", "PM1", "PM2", "PM3", "PM4", "PM6", "LP_done"],
                "dwell_time": int(ttime),
                "capacity": int(robot_capacity),
                "priority": 10,
            }
        }

    route_name = f"legacy_{mode}_{route_code}"
    return {
        "version": 1,
        "source": {"name": "LP", "capacity": 1},
        "sink": {"name": "LP_done", "capacity": 1},
        "chambers": chambers,
        "robots": robots,
        "routes": {route_name: {"sequence": _legacy_route_sequence(mode=mode, route_code=route_code)}},
        "legacy": {
            "route_code_alias": {
                mode: {str(route_code): route_name}
            }
        },
    }


def _resolve_route_name(
    route_cfg: Mapping[str, Any],
    mode: str,
    route_code: int,
    route_name: Optional[str],
) -> str:
    routes = route_cfg.get("routes") or {}
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("route_config.routes must be a non-empty mapping")
    if route_name and route_name in routes:
        return str(route_name)
    legacy_alias = (
        route_cfg.get("legacy", {})
        .get("route_code_alias", {})
        .get(mode, {})
    )
    aliased = legacy_alias.get(str(route_code))
    if aliased and aliased in routes:
        return str(aliased)
    return str(next(iter(routes.keys())))


def _build_single_device_net_from_route_config(
    n_wafer: int,
    ttime: int,
    robot_capacity: int,
    process_time_map: Optional[Dict[str, int]],
    route_code: int,
    device_mode: str,
    obs_config: Optional[Dict[str, Any]],
    route_config: Mapping[str, Any],
    route_name: Optional[str],
) -> Dict[str, object]:
    process_time_map = process_time_map or {}
    mode = str(device_mode).lower()
    source_cfg = dict(route_config.get("source") or {"name": "LP", "capacity": max(1, n_wafer)})
    sink_cfg = dict(route_config.get("sink") or {"name": "LP_done", "capacity": max(1, n_wafer)})
    source_name = str(source_cfg.get("name", "LP"))
    sink_name = str(sink_cfg.get("name", "LP_done"))
    selected_route_name = _resolve_route_name(route_config, mode=mode, route_code=int(route_code), route_name=route_name)

    chambers_cfg = dict(route_config.get("chambers") or {})
    chamber_kind_map: Dict[str, str] = {}
    for cname, cfg in chambers_cfg.items():
        chamber_kind_map[str(cname)] = str((cfg or {}).get("kind", "process"))

    robots_cfg_raw = dict(route_config.get("robots") or {})
    robots_cfg: Dict[str, CompiledRobotSpec] = {}
    for rb_name, rb in robots_cfg_raw.items():
        managed = tuple(str(x) for x in (rb or {}).get("managed_chambers", ()))
        robots_cfg[str(rb_name)] = CompiledRobotSpec(
            name=str(rb_name),
            managed_chambers=managed,
            transport_place=str((rb or {}).get("transport_place", f"d_{rb_name}")),
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

    # 每条 route 的 stage 级参数（process/cleaning）覆盖 chamber 级默认值
    route_stage_proc_time: Dict[str, int] = {}
    route_stage_clean_dur: Dict[str, int] = {}
    route_stage_clean_trig: Dict[str, int] = {}
    route_stage_rand_scale: Dict[str, float] = {}

    def _set_consistent_int(target: Dict[str, int], name: str, value: int, field_name: str) -> None:
        if name in target and int(target[name]) != int(value):
            raise ValueError(
                f"route {selected_route_name} has conflicting {field_name} for {name}: "
                f"{target[name]} vs {value}"
            )
        target[name] = int(value)

    def _set_consistent_float(target: Dict[str, float], name: str, value: float, field_name: str) -> None:
        if name in target and float(target[name]) != float(value):
            raise ValueError(
                f"route {selected_route_name} has conflicting {field_name} for {name}: "
                f"{target[name]} vs {value}"
            )
        target[name] = float(value)

    for stage in route_ir.stages:
        is_process_stage = str(stage.stage_type) == "process"
        for chamber_name in stage.candidates:
            if chamber_name in {source_name, sink_name}:
                continue
            # 对 process stage，<=0 视为“未指定”，避免把未知工时覆盖成 0
            if stage.stage_process_time is not None:
                p_val = float(stage.stage_process_time)
                if (not is_process_stage) or p_val > 0:
                    _set_consistent_int(
                        route_stage_proc_time,
                        chamber_name,
                        int(round(p_val)),
                        "process_time",
                    )
            if stage.stage_cleaning_duration is not None:
                _set_consistent_int(
                    route_stage_clean_dur,
                    chamber_name,
                    int(stage.stage_cleaning_duration),
                    "cleaning_duration",
                )
            if stage.stage_cleaning_trigger_wafers is not None:
                _set_consistent_int(
                    route_stage_clean_trig,
                    chamber_name,
                    int(stage.stage_cleaning_trigger_wafers),
                    "cleaning_trigger_wafers",
                )
            if stage.stage_proc_rand_scale is not None:
                _set_consistent_float(
                    route_stage_rand_scale,
                    chamber_name,
                    float(stage.stage_proc_rand_scale),
                    "proc_rand_scale",
                )

    # Place 顺序：source -> route 中出现的 chambers -> sink -> transport places
    id2p_name: List[str] = []
    seen_places: Set[str] = set()

    def add_place_name(name: str) -> None:
        if name not in seen_places:
            seen_places.add(name)
            id2p_name.append(name)

    add_place_name(source_name)
    for stage in route_ir.stages[1:-1]:
        for c in stage.candidates:
            add_place_name(c)
    add_place_name(sink_name)

    # 兼容旧级联 route5/route4：即使未使用也保留 TM3 展示位
    for rb_name in (route_config.get("robots") or {}):
        rb_cfg = route_config["robots"][rb_name]
        add_place_name(str(rb_cfg.get("transport_place", f"d_{rb_name}")))

    # 统计每个 source 对应的 transport 集合；若同一 source 有多个 transport 则需分离 u_* 变迁
    src_to_transports: Dict[str, Set[str]] = {}
    for hop in route_ir.transports:
        src_stage = route_ir.stages[hop.from_stage_idx]
        d_place = hop.transport_place
        for src in src_stage.candidates:
            src_to_transports.setdefault(src, set()).add(d_place)

    # Transition 顺序：按 hop 顺序先 u_* 再 t_*
    id2t_name: List[str] = []
    seen_t: Set[str] = set()

    def add_transition(name: str) -> None:
        if name not in seen_t:
            seen_t.add(name)
            id2t_name.append(name)

    def u_transition_name(src: str, d_place: str) -> str:
        """当同一 source 有多个 transport 时使用 u_{src}_{d_place}，否则 u_{src}"""
        transports = src_to_transports.get(src, set())
        if len(transports) > 1:
            return f"u_{src}_{d_place}"
        return f"u_{src}"

    for hop in route_ir.transports:
        src_stage = route_ir.stages[hop.from_stage_idx]
        dst_stage = route_ir.stages[hop.to_stage_idx]
        d_place = hop.transport_place
        for src in src_stage.candidates:
            add_transition(u_transition_name(src, d_place))
        for dst in dst_stage.candidates:
            add_transition(f"t_{dst}")

    p_idx = {name: i for i, name in enumerate(id2p_name)}
    t_idx = {name: i for i, name in enumerate(id2t_name)}
    p_count, t_count = len(id2p_name), len(id2t_name)
    pre = np.zeros((p_count, t_count), dtype=int)
    pst = np.zeros((p_count, t_count), dtype=int)

    def add_arc(src: str, tr: str, dst: str) -> None:
        if src not in p_idx or dst not in p_idx or tr not in t_idx:
            return
        # 同名变迁在 repeat 路径中会被多次引用；同一库所-变迁弧应保持单位权重，
        # 不能因重复路径展开把权重累加到 >1（否则结构使能会错误要求多 token）。
        pre[p_idx[src], t_idx[tr]] = 1
        pst[p_idx[dst], t_idx[tr]] = 1

    for hop in route_ir.transports:
        src_stage = route_ir.stages[hop.from_stage_idx]
        dst_stage = route_ir.stages[hop.to_stage_idx]
        d_place = hop.transport_place
        for src in src_stage.candidates:
            add_arc(src, u_transition_name(src, d_place), d_place)
        for dst in dst_stage.candidates:
            add_arc(d_place, f"t_{dst}", dst)

    token_plan: TokenRoutePlan = build_token_route_plan(route_ir=route_ir, transition_names=id2t_name)
    token_route_queue = token_plan.route_queue_template
    t_route_code_map = dict(token_plan.transition_code_map)
    pre_color = _build_pre_color_from_route_queue(pre, id2t_name, token_route_queue, t_route_code_map)

    # 模块参数
    modules: Dict[str, SingleModuleSpec] = {}
    source_capacity = int(source_cfg.get("capacity", max(1, n_wafer)))
    sink_capacity = int(sink_cfg.get("capacity", max(1, n_wafer)))
    modules[source_name] = SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, source_capacity))
    modules[sink_name] = SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, sink_capacity))

    for name in id2p_name:
        if name in {source_name, sink_name}:
            continue
        if name.startswith("d_"):
            # transport
            rb_cfg = next(
                (dict(v) for v in (route_config.get("robots") or {}).values() if str(v.get("transport_place", "")) == name),
                {},
            )
            modules[name] = SingleModuleSpec(
                tokens=0,
                ptime=int(rb_cfg.get("dwell_time", ttime)),
                capacity=int(rb_cfg.get("capacity", robot_capacity)),
            )
            continue
        c_cfg = dict(chambers_cfg.get(name) or {})
        ptime = int(
            route_stage_proc_time.get(
                name,
                process_time_map.get(name, c_cfg.get("process_time", 0)),
            )
        )
        modules[name] = SingleModuleSpec(
            tokens=0,
            ptime=ptime,
            capacity=int(c_cfg.get("capacity", 1)),
        )

    m0 = np.array([modules[name].tokens for name in id2p_name], dtype=int)
    md = m0.copy()
    md[p_idx[source_name]] = 0
    md[p_idx[sink_name]] = n_wafer
    ptime = np.array([modules[name].ptime for name in id2p_name], dtype=int)
    capacity = np.array([modules[name].capacity for name in id2p_name], dtype=int)
    ttime_arr = np.array([ttime for _ in range(t_count)], dtype=int)

    # transport onehot map
    # 配置驱动路径默认使用逐目标编码；cascade 固定为 8 维目标字典。
    tm_target_onehot_map: Dict[str, Dict[str, int]] = {}
    for hop in route_ir.transports:
        tp = str(hop.transport_place)
        dst_candidates = tuple(route_ir.stages[hop.to_stage_idx].candidates)
        if mode == "cascade" and tp == "d_TM2":
            tm_target_onehot_map[tp] = dict(CASCADE_TM2_TARGET_ONEHOT)
            continue
        if mode == "cascade" and tp == "d_TM3":
            tm_target_onehot_map[tp] = dict(CASCADE_TM3_TARGET_ONEHOT)
            continue
        m = tm_target_onehot_map.setdefault(tp, {})
        for dst_name in dst_candidates:
            if dst_name not in m:
                m[dst_name] = len(m)

    ctx = obs_config or {}
    p_res = int(ctx.get("P_Residual_time", 15))
    d_res = int(ctx.get("D_Residual_time", 10))
    clean_dur_default = int(ctx.get("cleaning_duration", 150))
    clean_trig_default = int(ctx.get("cleaning_trigger_wafers", 5))
    cleaning_duration_map: Dict[str, int] = dict(ctx.get("cleaning_duration_map") or {})
    cleaning_trigger_wafers_map: Dict[str, int] = dict(ctx.get("cleaning_trigger_wafers_map") or {})
    scrap_clip = float(ctx.get("scrap_clip_threshold", 20.0))

    buffer_names = {
        name for name, cfg in chambers_cfg.items()
        if str((cfg or {}).get("kind", "")) == "buffer"
    }
    marks: List[Place] = []
    for name in id2p_name:
        spec = modules[name]
        if name == source_name or name == sink_name:
            ptype = 3
        elif name.startswith("d_"):
            ptype = 2
        else:
            kind = str((chambers_cfg.get(name) or {}).get("kind", "process"))
            ptype = 5 if kind in {"buffer", "loadlock"} else 1

        if obs_config is not None:
            if name == source_name:
                place = SR(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype, n_wafer=n_wafer)
            elif name == sink_name:
                place = SR(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
            elif name.startswith("d_"):
                tm_map = tm_target_onehot_map.get(name, {})
                if mode == "cascade" and name == "d_TM2":
                    tm_map = dict(CASCADE_TM2_TARGET_ONEHOT)
                elif mode == "cascade" and name == "d_TM3":
                    tm_map = dict(CASCADE_TM3_TARGET_ONEHOT)
                onehot_dim = max(tm_map.values(), default=-1) + 1 if tm_map else 0
                place = TM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    D_Residual_time=d_res,
                    target_onehot_map=tm_map,
                    onehot_dim=onehot_dim,
                )
            elif ptype == 5:
                place = LL(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
            elif ptype == 1:
                c_dur = int(
                    route_stage_clean_dur.get(
                        name,
                        cleaning_duration_map.get(
                            name,
                            (chambers_cfg.get(name) or {}).get("cleaning_duration", clean_dur_default),
                        ),
                    )
                )
                c_trig = int(
                    route_stage_clean_trig.get(
                        name,
                        cleaning_trigger_wafers_map.get(
                            name,
                            (chambers_cfg.get(name) or {}).get("cleaning_trigger_wafers", clean_trig_default),
                        ),
                    )
                )
                place = PM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    P_Residual_time=p_res,
                    cleaning_duration=max(1, c_dur),
                    cleaning_trigger_wafers=max(1, c_trig),
                    scrap_clip_threshold=scrap_clip,
                )
            else:
                place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
        else:
            place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)

        if name == source_name:
            for tok_id in range(n_wafer):
                place.append(
                    BasedToken(
                        enter_time=0,
                        token_id=tok_id,
                        route_type=1,
                        step=0,
                        where=0,
                        route_queue=token_route_queue,
                        route_head_idx=0,
                    )
                )
        marks.append(place)

    route_meta = build_route_meta_from_route_ir(route_ir, buffer_names=buffer_names or BUFFER_NAMES)

    pre_place_indices: List[np.ndarray] = [np.flatnonzero(pre[:, t] > 0) for t in range(t_count)]
    pst_place_indices: List[np.ndarray] = [np.flatnonzero(pst[:, t] > 0) for t in range(t_count)]
    transport_pre_place_idx: List[int] = []
    for t in range(t_count):
        found = next(
            (int(idx) for idx in pre_place_indices[t] if id2p_name[int(idx)].startswith("d_")),
            -1,
        )
        transport_pre_place_idx.append(int(found))

    return {
        "m0": m0,
        "md": md,
        "pre": pre,
        "pre_color": pre_color,
        "pst": pst,
        "pre_place_indices": pre_place_indices,
        "pst_place_indices": pst_place_indices,
        "transport_pre_place_idx": transport_pre_place_idx,
        "ptime": ptime,
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
        "token_route_queue_template": token_route_queue,
        "token_route_plan_template": token_plan,
        "route_ir": route_ir,
    }


@dataclass
class SingleModuleSpec:
    tokens: int = 0
    ptime: int = 0
    capacity: int = 1


def build_single_device_net(
    n_wafer: int,
    ttime: int = 5,
    robot_capacity: int = 1,
    process_time_map: Optional[Dict[str, int]] = None,
    route_code: int = 0,
    device_mode: str = "single",
    obs_config: Optional[Dict[str, Any]] = None,
    route_config: Optional[Mapping[str, Any]] = None,
    route_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    构建 Petri 网结构：
    - device_mode=single：通过 route_code 选择预置单设备路径
      - route_code=0: LP -> PM1 -> [PM3, PM4] -> LP_done
      - route_code=1: LP -> PM1 -> [PM3, PM4] -> PM6 -> LP_done
    - device_mode=cascade：通过 route_code 选择级联路径
      - route_code=1: LP -> PM7/8 -> LLC -> PM1/2/3/4 -> LLD -> PM9/10 -> LP_done
      - route_code=2: LP -> PM7/8 -> LLC -> PM1/2 -> LLD -> PM9/10 -> LP_done
      - route_code=3: LP -> PM7/8 -> LLC -> PM1/2 -> LLD -> LP_done
      - route_code=4: LP -> [PM7 -> PM8 -> LLC -> LLD] * 5 -> LP_done
      - route_code=5: LP -> PM7/8 -> PM9/10 -> LP_done

    route_config（可选）:
    - 提供配置驱动构网 schema（含 source/sink/chambers/robots/routes）
    - 提供后优先走编译链：parse/normalize -> route IR -> token route -> net build
    - 未提供时保留 legacy route_code 逻辑

    返回 dict 除 pre/pst/m0/md/marks/id2p_name/id2t_name 等外，还包含预计算索引（供 get_enable_t/_fire 复用）：
    - pre_place_indices: List[np.ndarray]，pre_place_indices[t] 为变迁 t 的前置库所下标
    - pst_place_indices: List[np.ndarray]，pst_place_indices[t] 为变迁 t 的后置库所下标
    - transport_pre_place_idx: List[int]，transport_pre_place_idx[t] 为变迁 t 的运输位前置库所下标（无则为 -1）
    """
    mode = str(device_mode).lower()
    if mode not in {"single", "cascade"}:
        mode = "single"
    robot_capacity = 2 if int(robot_capacity) == 2 else 1
    process_time_map = process_time_map or {}
    route_code = int(route_code)
    if route_config is not None:
        return _build_single_device_net_from_route_config(
            n_wafer=n_wafer,
            ttime=ttime,
            robot_capacity=robot_capacity,
            process_time_map=process_time_map,
            route_code=route_code,
            device_mode=mode,
            obs_config=obs_config,
            route_config=route_config,
            route_name=route_name,
        )
    if mode == "cascade":
        if route_code not in {1, 2, 3, 4, 5}:
            route_code = 1
        pm_stage1 = int(process_time_map.get("PM7", 70))
        lld_time = int(process_time_map.get("LLD", 70))
        pm_stage5 = int(process_time_map.get("PM9", 200))
        if route_code == 4:
            modules = {
                "LP": SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, n_wafer)),
                "PM7": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "PM8": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "LLC": SingleModuleSpec(tokens=0, ptime=0, capacity=1),
                "LLD": SingleModuleSpec(tokens=0, ptime=lld_time, capacity=1),
                "LP_done": SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, n_wafer)),
                "d_TM2": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
                "d_TM3": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
            }
            id2p_name = list(modules.keys())
            id2t_name = [
                "u_LP",
                "t_PM7",
                "u_PM7",
                "t_PM8",
                "u_PM8",
                "t_LLC",
                "u_LLC",
                "t_LLD",
                "u_LLD",
                "t_LP_done",
            ]
        elif route_code == 5:
            modules = {
                "LP": SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, n_wafer)),
                "PM7": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "PM8": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "PM9": SingleModuleSpec(tokens=0, ptime=pm_stage5, capacity=1),
                "PM10": SingleModuleSpec(tokens=0, ptime=pm_stage5, capacity=1),
                "LP_done": SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, n_wafer)),
                "d_TM2": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
                "d_TM3": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
            }
            id2p_name = list(modules.keys())
            id2t_name = [
                "u_LP",
                "t_PM7",
                "t_PM8",
                "u_PM7",
                "u_PM8",
                "t_PM9",
                "t_PM10",
                "u_PM9",
                "u_PM10",
                "t_LP_done",
            ]
        else:
            cascade_stage3 = ("PM1", "PM2", "PM3", "PM4") if route_code == 1 else ("PM1", "PM2")
            pm_stage3_default = 600 if route_code == 1 else 300
            modules = {
                "LP": SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, n_wafer)),
                "PM7": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "PM8": SingleModuleSpec(tokens=0, ptime=pm_stage1, capacity=1),
                "LLC": SingleModuleSpec(tokens=0, ptime=0, capacity=1),
                "LLD": SingleModuleSpec(tokens=0, ptime=lld_time, capacity=1),
                "LP_done": SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, n_wafer)),
                "d_TM2": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
                "d_TM3": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
            }
            if route_code != 3:
                modules.update(
                    {
                        "PM9": SingleModuleSpec(tokens=0, ptime=pm_stage5, capacity=1),
                        "PM10": SingleModuleSpec(tokens=0, ptime=pm_stage5, capacity=1),
                    }
                )
            for pm_name in cascade_stage3:
                pm_time = int(process_time_map.get(pm_name, pm_stage3_default))
                modules[pm_name] = SingleModuleSpec(tokens=0, ptime=pm_time, capacity=1)
            id2p_name = list(modules.keys())
            stage3_t = [f"t_{name}" for name in cascade_stage3]
            stage3_u = [f"u_{name}" for name in cascade_stage3]
            if route_code == 3:
                id2t_name = (
                    ["u_LP", "t_PM7", "t_PM8", "u_PM7", "u_PM8", "t_LLC", "u_LLC"]
                    + stage3_t
                    + stage3_u
                    + ["t_LLD", "u_LLD", "t_LP_done"]
                )
            else:
                id2t_name = (
                    ["u_LP", "t_PM7", "t_PM8", "u_PM7", "u_PM8", "t_LLC", "u_LLC"]
                    + stage3_t
                    + stage3_u
                    + ["t_LLD", "u_LLD", "t_PM9", "t_PM10", "u_PM9", "u_PM10", "t_LP_done"]
                )
    else:
        pm1_time = int(process_time_map.get("PM1", 100))
        pm3_time = int(process_time_map.get("PM3", 300))
        pm4_time = int(process_time_map.get("PM4", 300))
        pm6_time = int(process_time_map.get("PM6", 300 if route_code == 1 else 0))
        modules = {
            "LP": SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, n_wafer)),
            "PM1": SingleModuleSpec(tokens=0, ptime=pm1_time, capacity=1),
            "PM2": SingleModuleSpec(tokens=0, ptime=0, capacity=1),   # 展示腔体
            "PM3": SingleModuleSpec(tokens=0, ptime=pm3_time, capacity=1),
            "PM4": SingleModuleSpec(tokens=0, ptime=pm4_time, capacity=1),
            "PM6": SingleModuleSpec(tokens=0, ptime=pm6_time, capacity=1),
            "LP_done": SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, n_wafer)),
            # 关键约束：在 d_TM1 中停留 ttime 秒后，才允许进入目标腔室
            "d_TM1": SingleModuleSpec(tokens=0, ptime=ttime, capacity=robot_capacity),
        }
        id2p_name = list(modules.keys())
        if route_code == 1:
            id2t_name = [
                "u_LP", "t_PM1",
                "u_PM1", "t_PM3", "t_PM4",
                "u_PM3", "u_PM4", "t_PM6",
                "u_PM6", "t_LP_done",
            ]
        else:
            id2t_name = [
                "u_LP", "t_PM1",
                "u_PM1", "t_PM3", "t_PM4",
                "u_PM3", "u_PM4", "t_LP_done",
            ]

    p_idx = {name: i for i, name in enumerate(id2p_name)}
    t_idx = {name: i for i, name in enumerate(id2t_name)}
    P, T = len(id2p_name), len(id2t_name)

    pre = np.zeros((P, T), dtype=int)
    pst = np.zeros((P, T), dtype=int)

    def add_arc(src: str, tr: str, dst: str) -> None:
        pre[p_idx[src], t_idx[tr]] += 1
        pst[p_idx[dst], t_idx[tr]] += 1

    if mode == "cascade":
        add_arc("LP", "u_LP", "d_TM2")
        if route_code == 4:
            add_arc("d_TM2", "t_PM7", "PM7")
            add_arc("PM7", "u_PM7", "d_TM2")
            add_arc("d_TM2", "t_PM8", "PM8")
            add_arc("PM8", "u_PM8", "d_TM2")
            add_arc("d_TM2", "t_LLC", "LLC")
            add_arc("LLC", "u_LLC", "d_TM3")
            add_arc("d_TM3", "t_LLD", "LLD")
            add_arc("LLD", "u_LLD", "d_TM2")
            add_arc("d_TM2", "t_LP_done", "LP_done")
        elif route_code == 5:
            add_arc("d_TM2", "t_PM7", "PM7")
            add_arc("d_TM2", "t_PM8", "PM8")
            add_arc("PM7", "u_PM7", "d_TM2")
            add_arc("PM8", "u_PM8", "d_TM2")
            add_arc("d_TM2", "t_PM9", "PM9")
            add_arc("d_TM2", "t_PM10", "PM10")
            add_arc("PM9", "u_PM9", "d_TM2")
            add_arc("PM10", "u_PM10", "d_TM2")
            add_arc("d_TM2", "t_LP_done", "LP_done")
        else:
            # TM2: 负责 LP/PM7/PM8/LLD/PM9/PM10 取放，放 LLC/LP_done/PM7/PM8/PM9/PM10
            add_arc("d_TM2", "t_PM7", "PM7")
            add_arc("d_TM2", "t_PM8", "PM8")

            add_arc("PM7", "u_PM7", "d_TM2")
            add_arc("PM8", "u_PM8", "d_TM2")
            add_arc("d_TM2", "t_LLC", "LLC")

            # TM3: 负责 LLC/PM* 取放，放 LLD/PM*
            add_arc("LLC", "u_LLC", "d_TM3")
            for pm_name in cascade_stage3:
                add_arc("d_TM3", f"t_{pm_name}", pm_name)
                add_arc(pm_name, f"u_{pm_name}", "d_TM3")
            add_arc("d_TM3", "t_LLD", "LLD")

            add_arc("LLD", "u_LLD", "d_TM2")
            if route_code == 3:
                add_arc("d_TM2", "t_LP_done", "LP_done")
            else:
                add_arc("d_TM2", "t_PM9", "PM9")
                add_arc("d_TM2", "t_PM10", "PM10")
                add_arc("PM9", "u_PM9", "d_TM2")
                add_arc("PM10", "u_PM10", "d_TM2")
                add_arc("d_TM2", "t_LP_done", "LP_done")
    else:
        add_arc("LP", "u_LP", "d_TM1")
        add_arc("d_TM1", "t_PM1", "PM1")

        add_arc("PM1", "u_PM1", "d_TM1")
        add_arc("d_TM1", "t_PM3", "PM3")
        add_arc("d_TM1", "t_PM4", "PM4")

        add_arc("PM3", "u_PM3", "d_TM1")
        add_arc("PM4", "u_PM4", "d_TM1")
        if route_code == 1:
            add_arc("d_TM1", "t_PM6", "PM6")
            add_arc("PM6", "u_PM6", "d_TM1")
        add_arc("d_TM1", "t_LP_done", "LP_done")

    # color-aware 前置矩阵（兼容保留；路由门控已迁移到 token.route_queue）
    max_where = 13 if mode == "cascade" else (9 if route_code == 1 else 7)
    pre_color = np.zeros((P, T, max_where + 1), dtype=int)
    for t_name, tid in t_idx.items():
        if t_name.startswith("u_"):
            # u_* 不做 where 路由限制：所有 color 截面与二维 pre 一致（通配）
            pre_color[:, tid, :] = pre[:, tid][:, None]
            continue
        if mode == "cascade":
            if route_code == 4:
                if t_name == "t_PM7":
                    allowed_where = (1, 9)
                elif t_name == "t_PM8":
                    allowed_where = (3,)
                elif t_name == "t_LLC":
                    allowed_where = (5,)
                elif t_name == "t_LLD":
                    allowed_where = (7,)
                elif t_name == "t_LP_done":
                    allowed_where = (9,)
                else:
                    allowed_where = ()
            elif route_code == 5:
                if t_name in ("t_PM7", "t_PM8"):
                    allowed_where = (1,)
                elif t_name in ("t_PM9", "t_PM10"):
                    allowed_where = (3,)
                elif t_name == "t_LP_done":
                    allowed_where = (5,)
                else:
                    allowed_where = ()
            else:
                if t_name in ("t_PM7", "t_PM8"):
                    allowed_where = (1,)
                elif t_name == "t_LLC":
                    allowed_where = (3,)
                elif t_name in ("t_PM1", "t_PM2", "t_PM3", "t_PM4"):
                    allowed_where = (5,)
                elif t_name == "t_LLD":
                    allowed_where = (7,)
                elif t_name in ("t_PM9", "t_PM10"):
                    allowed_where = (9,)
                elif t_name == "t_LP_done":
                    allowed_where = (9,) if route_code == 3 else (11,)
                else:
                    allowed_where = ()
        else:
            if t_name == "t_PM1":
                allowed_where = (1,)
            elif t_name in ("t_PM3", "t_PM4"):
                allowed_where = (3,)
            elif t_name == "t_PM6":
                allowed_where = (5,)
            elif t_name == "t_LP_done":
                allowed_where = (7,) if route_code == 1 else (5,)
            else:
                allowed_where = ()
        for c in allowed_where:
            pre_color[:, tid, c] = pre[:, tid]

    m0 = np.array([modules[name].tokens for name in id2p_name], dtype=int)
    md = m0.copy()
    md[p_idx["LP"]] = 0
    md[p_idx["LP_done"]] = n_wafer

    ptime = np.array([modules[name].ptime for name in id2p_name], dtype=int)
    capacity = np.array([modules[name].capacity for name in id2p_name], dtype=int)
    ttime_arr = np.array([ttime for _ in range(T)], dtype=int)

    # 构建 TM one-hot 映射（按路线与设备模式）
    def _tm1_onehot_map() -> Dict[str, int]:
        m: Dict[str, int] = {"PM1": 0, "PM3": 1, "PM4": 1, "PM6": 2, "LP_done": 3}
        if route_code != 1:
            m.pop("PM6", None)
        return m

    def _tm2_onehot_map() -> Dict[str, int]:
        return dict(CASCADE_TM2_TARGET_ONEHOT)

    def _tm3_onehot_map() -> Dict[str, int]:
        return dict(CASCADE_TM3_TARGET_ONEHOT)

    ctx = obs_config or {}
    p_res = int(ctx.get("P_Residual_time", 15))
    d_res = int(ctx.get("D_Residual_time", 10))
    clean_dur_default = int(ctx.get("cleaning_duration", 150))
    clean_trig_default = int(ctx.get("cleaning_trigger_wafers", 5))
    cleaning_duration_map: Dict[str, int] = dict(ctx.get("cleaning_duration_map") or {})
    cleaning_trigger_wafers_map: Dict[str, int] = dict(ctx.get("cleaning_trigger_wafers_map") or {})
    scrap_clip = float(ctx.get("scrap_clip_threshold", 20.0))

    marks: List[Place] = []
    token_route_queue, t_route_code_map = _build_token_route_queue_and_t_code_map(
        mode=mode,
        route_code=route_code,
    )

    for name in id2p_name:
        spec = modules[name]
        if name.startswith("LP"):
            ptype = 3
        elif name.startswith("d_"):
            ptype = 2
        elif name in {"LLC", "LLD"} or ((mode != "cascade") and name == "PM2") or (name == "PM6" and route_code != 1):
            ptype = 5
        else:
            ptype = 1

        if obs_config is not None:
            if name == "LP":
                place = SR(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype, n_wafer=n_wafer)
            elif name == "LP_done":
                place = SR(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
            elif name == "d_TM1":
                place = TM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    D_Residual_time=d_res,
                    target_onehot_map=_tm1_onehot_map(),
                    onehot_dim=4,
                )
            elif name == "d_TM2":
                place = TM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    D_Residual_time=d_res,
                    target_onehot_map=_tm2_onehot_map(),
                    onehot_dim=8,
                )
            elif name == "d_TM3":
                place = TM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    D_Residual_time=d_res,
                    target_onehot_map=_tm3_onehot_map(),
                    onehot_dim=8,
                )
            elif name in {"LLC", "LLD"}:
                place = LL(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
            elif ptype == 1:
                c_dur = int(cleaning_duration_map.get(name, clean_dur_default))
                c_trig = int(cleaning_trigger_wafers_map.get(name, clean_trig_default))
                place = PM(
                    name=name,
                    capacity=spec.capacity,
                    processing_time=spec.ptime,
                    type=ptype,
                    P_Residual_time=p_res,
                    cleaning_duration=max(1, c_dur),
                    cleaning_trigger_wafers=max(1, c_trig),
                    scrap_clip_threshold=scrap_clip,
                )
            else:
                place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
        else:
            place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)

        if name == "LP":
            for tok_id in range(n_wafer):
                place.append(
                    BasedToken(
                        enter_time=0,
                        token_id=tok_id,
                        route_type=1,
                        step=0,
                        where=0,
                        route_queue=token_route_queue,
                        route_head_idx=0,
                    )
                )
        marks.append(place)

    # 从路线解析路由元数据
    route_key = (mode, route_code)
    stages = ROUTE_SPECS.get(route_key)
    if stages is None:
        stages = ROUTE_SPECS.get((mode, 1 if mode == "cascade" else 0), ROUTE_SPECS[("single", 0)])
    route_meta = parse_route(stages, BUFFER_NAMES)
    if mode == "cascade" and route_code == 4:
        # route4: LLD 出站可能回 PM7（前四轮）或去 LP_done（第五轮）。
        route_meta["u_targets"]["LLD"] = ["PM7", "LP_done"]

    # 预计算每个变迁的 pre/pst 库所索引与运输位索引，供运行时复用
    T = pre.shape[1]
    pre_place_indices: List[np.ndarray] = [
        np.flatnonzero(pre[:, t] > 0) for t in range(T)
    ]
    pst_place_indices: List[np.ndarray] = [
        np.flatnonzero(pst[:, t] > 0) for t in range(T)
    ]
    transport_pre_place_idx: List[int] = []
    for t in range(T):
        indices = pre_place_indices[t]
        found = next(
            (int(idx) for idx in indices if id2p_name[int(idx)].startswith("d_")),
            -1,
        )
        transport_pre_place_idx.append(int(found))

    return {
        "m0": m0,
        "md": md,
        "pre": pre,
        "pre_color": pre_color,
        "pst": pst,
        "pre_place_indices": pre_place_indices,
        "pst_place_indices": pst_place_indices,
        "transport_pre_place_idx": transport_pre_place_idx,
        "ptime": ptime,
        "ttime": ttime_arr,
        "capacity": capacity,
        "id2p_name": id2p_name,
        "id2t_name": id2t_name,
        "idle_idx": {"start": p_idx["LP"], "end": p_idx["LP_done"]},
        "marks": marks,
        "n_wafer": n_wafer,
        "n_wafer_route1": n_wafer,
        "n_wafer_route2": 0,
        "single_route_code": route_code,
        "single_device_mode": mode,
        "route_meta": route_meta,
        "t_route_code_map": t_route_code_map,
        "token_route_queue_template": token_route_queue,
    }
