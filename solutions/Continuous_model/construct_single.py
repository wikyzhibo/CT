"""
单设备构网工具：输出与现有连续 Petri 构网一致的结构化信息。
支持两种模板：
- single: 原单设备路径（可由 route_code 细分）
- cascade: 级联路径 LP->PM7/8->LLC->PM1/2/3/4->LLD->PM9/10->LP_done
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.pn import Place


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
    }
