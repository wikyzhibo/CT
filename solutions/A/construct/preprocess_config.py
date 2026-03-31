"""
配置预处理：将 route/chamber 配置归一为“每腔室一块”的运行时真源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from solutions.A.utils import _preprocess_process_time_map as _hf_preprocess_process_time_map
from solutions.A.construct.route_compiler_single import RouteIR

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
    # 按路线中间 stage 顺序展开腔室名（跳过 buffer stage、source/sink），供后续工时缺省与取整的腔室列表。
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
    # 对 chambers 中每个腔室：优先取 process_time_map 已有值，否则用 route_config.chambers[].process_time；再按 5s 取整（见 utils）。
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
    source_name: str,
    sink_name: str,
    route_name: Optional[str],
) -> ChamberPreprocessResult:
    # 契约：仅 route_config + route_ir；无外部默认清洗参数。供 model_builder.build_net 构网前生成唯一腔室真源。
    chambers_cfg = dict(route_config.get("chambers") or {})

    # 1) 扫描 route_ir.stages：收集 stage 级 process_time / cleaning_*，同腔室多处必须一致。
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

            clean_duration = stage.stage_cleaning_duration if stage.stage_cleaning_duration else 0
            _set_consistent_int(
                route_stage_clean_dur,
                chamber_name,
                clean_duration,
                "cleaning_duration",
            )

            trigger = stage.stage_cleaning_trigger_wafers if stage.stage_cleaning_trigger_wafers else 0
            _set_consistent_int(
                route_stage_clean_trig,
                chamber_name,
                trigger,
                "cleaning_trigger_wafers",
            )


    # 2) 路线腔室工序工时：stage 初值 + route_config.chambers 缺省 + 5s 取整。
    ch_pre = _route_ir_preprocess_chambers(route_ir, source_name, sink_name)
    merged_for_preprocess: Dict[str, int] = dict(route_stage_proc_time)
    processed_pt = _preprocess_process_time_map(merged_for_preprocess, ch_pre, route_config)

    # 3) 并集：配置声明腔室 + 固定拓扑占位（与 build_marks 键域一致）。
    all_names: Set[str] = set(chambers_cfg.keys()) | set(FIXED_TIMELINE_CHAMBERS)
    blocks: Dict[str, ChamberRuntimeBlock] = {}

    for name in sorted(all_names):
        spec = chambers_cfg.get(name)
        cfg_dict = dict(spec or {})
        if name in chambers_cfg:
            kind = str(cfg_dict.get("kind", "process"))
        else:
            kind = "buffer" if name in {"LLC", "LLD"} else "process"

        process_time = int(processed_pt.get(name, 0))
        c_dur = int(route_stage_clean_dur.get(name) or 0)
        c_trig = int(route_stage_clean_trig.get(name) or 0)

        blocks[name] = ChamberRuntimeBlock(
            name=name,
            kind=kind,
            process_time=process_time,
            cleaning_duration=c_dur,
            cleaning_trigger_wafers=c_trig,
        )

    buffer_names = {name for name, blk in blocks.items() if blk.kind == "buffer"}
    return ChamberPreprocessResult(chamber_blocks=blocks, buffer_names=buffer_names)
