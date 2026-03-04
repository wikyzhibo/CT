"""
单设备构网工具：输出与现有连续 Petri 构网一致的结构化信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.pn import Place


@dataclass
class SingleModuleSpec:
    tokens: int = 0
    ptime: int = 0
    capacity: int = 1


def build_single_device_net(n_wafer: int, ttime: int = 5) -> Dict[str, object]:
    """
    构建单设备 Petri 网结构：
    LP -> PM1 -> [PM3, PM4] -> PM5 -> LP_done
    并保留 PM2/PM6（无工艺边，仅展示）。
    """
    modules: Dict[str, SingleModuleSpec] = {
        "LP": SingleModuleSpec(tokens=n_wafer, ptime=0, capacity=max(1, n_wafer)),
        "PM1": SingleModuleSpec(tokens=0, ptime=100, capacity=1),
        "PM2": SingleModuleSpec(tokens=0, ptime=0, capacity=1),   # 展示腔体
        "PM3": SingleModuleSpec(tokens=0, ptime=300, capacity=1),
        "PM4": SingleModuleSpec(tokens=0, ptime=300, capacity=1),
        "PM5": SingleModuleSpec(tokens=0, ptime=100, capacity=1),
        "PM6": SingleModuleSpec(tokens=0, ptime=0, capacity=1),   # 展示腔体
        "LP_done": SingleModuleSpec(tokens=0, ptime=0, capacity=max(1, n_wafer)),
        "d_TM1": SingleModuleSpec(tokens=0, ptime=0, capacity=1),
    }

    id2p_name = list(modules.keys())
    id2t_name = [
        "u_LP_PM1", "t_PM1",
        "u_PM1_PM3", "t_PM3",
        "u_PM1_PM4", "t_PM4",
        "u_PM3_PM5", "u_PM4_PM5", "t_PM5",
        "u_PM5_LP_done", "t_LP_done",
    ]

    p_idx = {name: i for i, name in enumerate(id2p_name)}
    t_idx = {name: i for i, name in enumerate(id2t_name)}
    P, T = len(id2p_name), len(id2t_name)

    pre = np.zeros((P, T), dtype=int)
    pst = np.zeros((P, T), dtype=int)

    def add_arc(src: str, tr: str, dst: str) -> None:
        pre[p_idx[src], t_idx[tr]] += 1
        pst[p_idx[dst], t_idx[tr]] += 1

    add_arc("LP", "u_LP_PM1", "d_TM1")
    add_arc("d_TM1", "t_PM1", "PM1")

    add_arc("PM1", "u_PM1_PM3", "d_TM1")
    add_arc("d_TM1", "t_PM3", "PM3")

    add_arc("PM1", "u_PM1_PM4", "d_TM1")
    add_arc("d_TM1", "t_PM4", "PM4")

    add_arc("PM3", "u_PM3_PM5", "d_TM1")
    add_arc("PM4", "u_PM4_PM5", "d_TM1")
    add_arc("d_TM1", "t_PM5", "PM5")

    add_arc("PM5", "u_PM5_LP_done", "d_TM1")
    add_arc("d_TM1", "t_LP_done", "LP_done")

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
        elif name in {"PM2", "PM6"}:
            ptype = 5
        else:
            ptype = 1
        place = Place(name=name, capacity=spec.capacity, processing_time=spec.ptime, type=ptype)
        if name == "LP":
            for tok_id in range(n_wafer):
                place.append(BasedToken(enter_time=0, token_id=tok_id, route_type=1, step=0))
        marks.append(place)

    return {
        "m0": m0,
        "md": md,
        "pre": pre,
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
    }
