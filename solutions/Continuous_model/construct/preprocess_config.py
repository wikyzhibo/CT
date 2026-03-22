"""
配置预处理：将 route/chamber 配置归一为“每腔室一块”的运行时真源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from solutions.Continuous_model.helper_function import _preprocess_process_time_map as _hf_preprocess_process_time_map
from solutions.Continuous_model.construct.route_compiler_single import RouteIR

FIXED_TIMELINE_CHAMBERS: Tuple[str, ...] = tuple(
    [*(f"PM{i}" for i in range(1, 11)), "LLC", "LLD"]
)


@dataclass
class ChamberRuntimeBlock:
    name: str
    kind: str
    process_time: int
    cleaning_duration: int
    cleaning_trigger_wafers: int


@dataclass
class ChamberPreprocessResult:
    chamber_blocks: Dict[str, ChamberRuntimeBlock]
    buffer_names: Set[str]


def _route_ir_preprocess_chambers(
    route_ir: RouteIR,
    source_name: str,
    sink_name: str,
) -> Tuple[str, ...]:
    names = []
    for stage in route_ir.stages[1:-1]:
        if str(stage.stage_type) == "buffer":
            continue
        for chamber_name in stage.candidates:
            if chamber_name in {source_name, sink_name}:
                continue
            if chamber_name not in names:
                names.append(chamber_name)
    return tuple(names)


def _preprocess_process_time_map(
    process_time_map: Mapping[str, int],
    chambers: Tuple[str, ...],
    route_config: Mapping[str, Any],
) -> Dict[str, int]:
    raw = dict(process_time_map)
    cfg_ch = dict(route_config.get("chambers") or {})
    defaults = {
        name: int((spec or {}).get("process_time", 0))
        for name, spec in cfg_ch.items()
        if name in chambers
    }
    missing = sorted(c for c in chambers if c not in raw and c not in defaults)
    if missing:
        raise ValueError(f"missing default process times for chambers: {missing}")
    return dict(_hf_preprocess_process_time_map(raw, chambers, defaults))


def preprocess_chamber_runtime_blocks(
    *,
    route_ir: RouteIR,
    route_config: Mapping[str, Any],
    process_time_map: Mapping[str, int],
    source_name: str,
    sink_name: str,
    route_name: Optional[str],
    default_cleaning_duration: int,
    default_cleaning_trigger_wafers: int,
) -> ChamberPreprocessResult:
    chambers_cfg = dict(route_config.get("chambers") or {})
    blocks: Dict[str, ChamberRuntimeBlock] = {}
    for name, cfg in chambers_cfg.items():
        cfg_dict = dict(cfg or {})
        blocks[str(name)] = ChamberRuntimeBlock(
            name=str(name),
            kind=str(cfg_dict.get("kind", "process")),
            process_time=int(cfg_dict.get("process_time", 0)),
            cleaning_duration=int(cfg_dict.get("cleaning_duration", default_cleaning_duration)),
            cleaning_trigger_wafers=int(
                cfg_dict.get("cleaning_trigger_wafers", default_cleaning_trigger_wafers)
            ),
        )

    # 固定拓扑下，未在配置中声明的腔室也要占位，保持 process_time_map 键集合稳定。
    for name in FIXED_TIMELINE_CHAMBERS:
        if name in blocks:
            continue
        kind = "buffer" if name == "LLC" else ("loadlock" if name == "LLD" else "process")
        blocks[name] = ChamberRuntimeBlock(
            name=name,
            kind=kind,
            process_time=int(process_time_map.get(name, 0)),
            cleaning_duration=int(default_cleaning_duration),
            cleaning_trigger_wafers=int(default_cleaning_trigger_wafers),
        )

    route_stage_proc_time: Dict[str, int] = {}
    route_stage_clean_dur: Dict[str, int] = {}
    route_stage_clean_trig: Dict[str, int] = {}

    def _set_consistent_int(target: Dict[str, int], name: str, value: int, field_name: str) -> None:
        if name in target and int(target[name]) != int(value):
            raise ValueError(
                f"route {route_name} has conflicting {field_name} for {name}: "
                f"{target[name]} vs {value}"
            )
        target[name] = int(value)

    for stage in route_ir.stages:
        is_process_stage = str(stage.stage_type) == "process"
        for chamber_name in stage.candidates:
            if chamber_name in {source_name, sink_name}:
                continue
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

    ch_pre = _route_ir_preprocess_chambers(route_ir, source_name, sink_name)
    merged_for_preprocess: Dict[str, int] = dict(process_time_map)
    merged_for_preprocess.update(route_stage_proc_time)
    processed_pt = _preprocess_process_time_map(merged_for_preprocess, ch_pre, route_config)

    for name, block in blocks.items():
        if name in processed_pt:
            block.process_time = int(processed_pt[name])
        if name in route_stage_clean_dur:
            block.cleaning_duration = int(route_stage_clean_dur[name])
        if name in route_stage_clean_trig:
            block.cleaning_trigger_wafers = int(route_stage_clean_trig[name])

    buffer_names = {name for name, blk in blocks.items() if blk.kind == "buffer"}
    return ChamberPreprocessResult(chamber_blocks=blocks, buffer_names=buffer_names)
