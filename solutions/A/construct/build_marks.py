"""
单设备构网的 marks 构造器：只消费预处理后的腔室块真源。
"""

from __future__ import annotations

from dataclasses import dataclass,field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from solutions.A.deprecated.pn import LL, PM, Place, SR, TM
from solutions.A.construct.preprocess_config import ChamberRuntimeBlock

SOURCE = 3
ROBOT = 2
CHAMBER = 1

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

@dataclass(slots=True)
class BasedToken:
    enter_time: int
    stay_time: int = 0
    token_id: int = -1
    machine: int = -1
    route_type: int = 0
    step: int = 0
    where: int = 0
    route_queue: Tuple[Any, ...] = ()
    route_head_idx: int = 0
    _target_place: Optional[str] = None
    _dst_level_targets: Optional[Tuple[str, ...]] = None
    _dst_level_full_on_pick: bool = False
    _place_idx: int = -1
    last_u_source: str = ""

    def clone(self):
        return BasedToken(
            enter_time=self.enter_time,
            stay_time=self.stay_time,
            token_id=self.token_id,
            machine=self.machine,
            route_type=self.route_type,
            step=self.step,
            where=self.where,
            route_queue=tuple(self.route_queue),
            route_head_idx=int(self.route_head_idx),
            last_u_source=str(self.last_u_source),
        )

@dataclass
class BuildMarksResult:
    marks: List[Place]
    m0: np.ndarray
    md: np.ndarray
    ptime: np.ndarray
    capacity: np.ndarray
    process_time_map_out: Dict[str, int]


def _add_token_to_source(n_wafer: int, place: Place, token_route_queue: Tuple[object, ...]) -> None:
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


def build_marks_for_single_net(
    *,
    id2p_name: Sequence[str],
    source_name: str,
    sink_name: str,
    chamber_blocks: Mapping[str, ChamberRuntimeBlock],
    n_wafer: int,
    token_route_queue: Tuple[object, ...],
    obs_config: Optional[Mapping[str, Any]],
    ttime: int,
) -> BuildMarksResult:
    p_idx = {name: i for i, name in enumerate(id2p_name)}
    ctx = dict(obs_config or {})
    p_res = int(ctx.get("P_Residual_time", 15))
    d_res = int(ctx.get("D_Residual_time", 10))
    scrap_clip = float(ctx.get("scrap_clip_threshold", 20.0))

    marks: List[Place] = []
    for name in id2p_name:
        if name == source_name or name == sink_name:
            ptype = SOURCE
        elif name in {"TM2", "TM3"}:
            ptype = ROBOT
        else:
            kind = chamber_blocks.get(name).kind if name in chamber_blocks else "process"
            ptype = 5 if kind in {"buffer", "loadlock"} else CHAMBER

        place_capacity = 100 if name in {source_name, sink_name} else 1
        if name == source_name or name == sink_name:
            proc_time = 0
        elif name in {"TM2", "TM3"}:
            proc_time = int(ttime)
        else:
            proc_time = int(chamber_blocks.get(name).process_time if name in chamber_blocks else 0)

        if obs_config is not None:
            if name == source_name:
                place = SR(name=name, capacity=100, processing_time=0, type=SOURCE, n_wafer=n_wafer)
            elif name == sink_name:
                place = SR(name=name, capacity=100, processing_time=0, type=SOURCE)
            elif name == "TM2":
                tm_map = dict(CASCADE_TM2_TARGET_ONEHOT)
                place = TM(
                    name=name,
                    capacity=place_capacity,
                    processing_time=proc_time,
                    type=ROBOT,
                    D_Residual_time=d_res,
                    target_onehot_map=tm_map,
                    onehot_dim=8,
                )
            elif name == "TM3":
                tm_map = dict(CASCADE_TM3_TARGET_ONEHOT)
                place = TM(
                    name=name,
                    capacity=place_capacity,
                    processing_time=proc_time,
                    type=ROBOT,
                    D_Residual_time=d_res,
                    target_onehot_map=tm_map,
                    onehot_dim=8,
                )
            elif ptype == 5:
                place = LL(name=name, capacity=place_capacity, processing_time=proc_time, type=ptype)
            elif ptype == CHAMBER:
                block = chamber_blocks.get(name)
                c_dur = int(block.cleaning_duration) if block is not None else 1
                c_trig = int(block.cleaning_trigger_wafers) if block is not None else 1
                place = PM(
                    name=name,
                    capacity=place_capacity,
                    processing_time=proc_time,
                    type=ptype,
                    P_Residual_time=p_res,
                    cleaning_duration=max(1, c_dur),
                    cleaning_trigger_wafers=max(1, c_trig),
                    scrap_clip_threshold=scrap_clip,
                )
            else:
                place = Place(name=name, capacity=place_capacity, processing_time=proc_time, type=ptype)
        else:
            place = Place(name=name, capacity=place_capacity, processing_time=proc_time, type=ptype)
        marks.append(place)

    m0 = np.zeros(len(id2p_name), dtype=int)
    md = m0.copy()
    md[p_idx[source_name]] = 0
    md[p_idx[sink_name]] = int(n_wafer)
    capacity = np.array([int(p.capacity) for p in marks], dtype=int)
    ptime = np.array([int(p.processing_time) for p in marks], dtype=int)

    source_place = marks[p_idx[source_name]]
    _add_token_to_source(n_wafer=n_wafer, place=source_place, token_route_queue=token_route_queue)

    timeline_names = [
        name for name in id2p_name
        if (str(name).startswith("PM") or str(name) in {"LLC", "LLD"})
    ]
    process_time_map_out = {
        name: int(chamber_blocks[name].process_time)
        for name in timeline_names
        if name in chamber_blocks
    }

    return BuildMarksResult(
        marks=marks,
        m0=m0,
        md=md,
        ptime=ptime,
        capacity=capacity,
        process_time_map_out=process_time_map_out,
    )
