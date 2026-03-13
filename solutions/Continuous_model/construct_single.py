"""
单设备构网工具：输出与现有连续 Petri 构网一致的结构化信息。
支持两种模板：
- single: 原单设备路径（可由 route_code 细分）
- cascade: 级联路径 LP->PM7/8->LLC->PM1/2/3/4->LLD->PM9/10->LP_done
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.pn import Place


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
}
BUFFER_NAMES: Set[str] = {"LLC"}


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
    """
    mode = str(device_mode).lower()
    if mode not in {"single", "cascade"}:
        mode = "single"
    robot_capacity = 2 if int(robot_capacity) == 2 else 1
    process_time_map = process_time_map or {}
    route_code = int(route_code)
    if mode == "cascade":
        if route_code not in {1, 2, 3}:
            route_code = 1
        cascade_stage3 = ("PM1", "PM2", "PM3", "PM4") if route_code == 1 else ("PM1", "PM2")
        pm_stage1 = int(process_time_map.get("PM7", 70))
        pm_stage3_default = 600 if route_code == 1 else 300
        pm_stage3 = int(process_time_map.get("PM1", pm_stage3_default))
        lld_time = int(process_time_map.get("LLD", 70))
        pm_stage5 = int(process_time_map.get("PM9", 200))
        modules: Dict[str, SingleModuleSpec] = {
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
        # TM2: 负责 LP/PM7/PM8/LLD/PM9/PM10 取放，放 LLC/LP_done/PM7/PM8/PM9/PM10
        add_arc("LP", "u_LP", "d_TM2")
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

    # color-aware 前置矩阵（color 维对应 token.where）
    max_where = 13 if mode == "cascade" else (9 if route_code == 1 else 7)
    pre_color = np.zeros((P, T, max_where + 1), dtype=int)
    for t_name, tid in t_idx.items():
        if t_name.startswith("u_"):
            # u_* 不做 where 路由限制：所有 color 截面与二维 pre 一致（通配）
            pre_color[:, tid, :] = pre[:, tid][:, None]
            continue
        if mode == "cascade":
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

    marks: List[Place] = []
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
        place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
        if name == "LP":
            for tok_id in range(n_wafer):
                place.append(BasedToken(enter_time=0, token_id=tok_id, route_type=1, step=0, where=0))
        marks.append(place)

    # 从路线解析路由元数据
    route_key = (mode, route_code)
    stages = ROUTE_SPECS.get(route_key)
    if stages is None:
        stages = ROUTE_SPECS.get((mode, 1 if mode == "cascade" else 0), ROUTE_SPECS[("single", 0)])
    route_meta = parse_route(stages, BUFFER_NAMES)

    return {
        "m0": m0,
        "md": md,
        "pre": pre,
        "pre_color": pre_color,
        "pst": pst,
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
    }
