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
    "LP1",
    "LP2",
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

CASCADE_TM2_ONEHOT_DIM: int = len(CASCADE_TM2_TARGET_ORDER)
CASCADE_TM3_ONEHOT_DIM: int = len(CASCADE_TM3_TARGET_ORDER)

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


def _add_tokens_to_load_ports(
    *,
    n_wafer: int,
    marks: List[Place],
    p_idx: Mapping[str, int],
    lp_per_token: Sequence[str],
    token_route_queue: Tuple[object, ...],
    token_route_queue_by_type: Optional[Mapping[int, Tuple[object, ...]]] = None,
    token_route_type_sequence: Optional[Sequence[int]] = None,
) -> None:
    if len(lp_per_token) != int(n_wafer):
        raise ValueError("lp_per_token length must equal n_wafer")
    if token_route_type_sequence is not None and len(token_route_type_sequence) != int(n_wafer):
        raise ValueError("token_route_type_sequence length must equal n_wafer")
    for tok_id in range(n_wafer):
        route_type = int(token_route_type_sequence[tok_id]) if token_route_type_sequence is not None else 1
        route_queue = token_route_queue
        if token_route_queue_by_type is not None:
            route_queue = tuple(token_route_queue_by_type.get(route_type) or token_route_queue)
        lp_name = str(lp_per_token[tok_id])
        place = marks[p_idx[lp_name]]
        place.append(
            BasedToken(
                enter_time=0,
                token_id=tok_id,
                route_type=route_type,
                step=0,
                where=0,
                route_queue=route_queue,
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
    token_route_queue_by_type: Optional[Mapping[int, Tuple[object, ...]]] = None,
    token_route_type_sequence: Optional[Sequence[int]] = None,
    lp_per_token: Optional[Sequence[str]] = None,
    p_residual_time: int = 15,
    d_residual_time: int = 10,
    scrap_clip_threshold: float = 20.0,
    ttime: int,
) -> BuildMarksResult:
    p_idx = {name: i for i, name in enumerate(id2p_name)}
    p_res = p_residual_time
    d_res = d_residual_time
    scrap_clip = scrap_clip_threshold

    load_ports = ("LP1", "LP2")
    if lp_per_token is None:
        lp_per_token = tuple(source_name for _ in range(int(n_wafer)))
    counts: Dict[str, int] = {}
    for lp_name in lp_per_token:
        s = str(lp_name)
        counts[s] = counts.get(s, 0) + 1

    marks: List[Place] = []
    for name in id2p_name:
        if name in load_ports or name == sink_name:
            ptype = SOURCE
        elif name in {"TM2", "TM3"}:
            ptype = ROBOT
        else:
            kind = chamber_blocks.get(name).kind if name in chamber_blocks else "process"
            ptype = 5 if kind in {"buffer", "loadlock"} else CHAMBER

        place_capacity = 100 if name in load_ports or name == sink_name else 1
        if name in load_ports or name == sink_name:
            proc_time = 0
        elif name in {"TM2", "TM3"}:
            proc_time = int(ttime)
        else:
            proc_time = int(chamber_blocks.get(name).process_time if name in chamber_blocks else 0)


        if name in load_ports:
            n_here = int(counts.get(name, 0))
            place = SR(
                name=name,
                capacity=100,
                processing_time=0,
                type=SOURCE,
                n_wafer=max(1, n_here),
            )
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
                onehot_dim=CASCADE_TM2_ONEHOT_DIM,
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
                onehot_dim=CASCADE_TM3_ONEHOT_DIM,
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

        marks.append(place)

    m0 = np.zeros(len(id2p_name), dtype=int)
    md = m0.copy()
    for lp in load_ports:
        if lp in p_idx:
            md[p_idx[lp]] = 0
    md[p_idx[sink_name]] = int(n_wafer)
    capacity = np.array([int(p.capacity) for p in marks], dtype=int)
    ptime = np.array([int(p.processing_time) for p in marks], dtype=int)

    _add_tokens_to_load_ports(
        n_wafer=n_wafer,
        marks=marks,
        p_idx=p_idx,
        lp_per_token=lp_per_token,
        token_route_queue=token_route_queue,
        token_route_queue_by_type=token_route_queue_by_type,
        token_route_type_sequence=token_route_type_sequence,
    )

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
