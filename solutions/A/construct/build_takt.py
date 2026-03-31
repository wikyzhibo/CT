from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from solutions.A.construct.preprocess_config import ChamberRuntimeBlock
from solutions.A.takt_analysis import TAKT_HORIZON, analyze_cycle


def _normalize_cycle_values(raw_cycle: Sequence[Any]) -> Optional[List[float]]:
    values: List[float] = []
    for item in list(raw_cycle):
        value = float(item)
        if value < 0:
            raise ValueError("takt_cycle value must be >= 0")
        values.append(value)
    if not values:
        return None
    return values


def _build_takt_result_from_cycle(raw_cycle: Sequence[Any]) -> Optional[Dict[str, Any]]:
    cycle_values = _normalize_cycle_values(raw_cycle)
    if not cycle_values:
        return None
    return {
        "fast_takt": 0.0,
        "peak_slow_takts": [],
        "cycle_length": len(cycle_values),
        "cycle_takts": cycle_values,
    }


def _resolve_takt_cycle_override_by_type(
    *,
    takt_cycle: Any,
    multi_subpath: bool,
    all_types: Sequence[int],
    wafer_type_to_subpath: Mapping[int, str],
) -> Optional[Dict[int, Optional[Dict[str, Any]]]]:
    if takt_cycle is None:
        return None

    if isinstance(takt_cycle, Mapping):
        by_subpath_raw = takt_cycle.get("by_subpath")
        if isinstance(by_subpath_raw, Mapping):
            by_subpath = {str(k): v for k, v in dict(by_subpath_raw).items()}
            if not multi_subpath:
                return None
            result: Dict[int, Optional[Dict[str, Any]]] = {}
            for t_id in all_types:
                subpath = str(wafer_type_to_subpath.get(int(t_id), "") or "")
                if subpath not in by_subpath:
                    raise ValueError(f"missing takt_cycle.by_subpath entry for subpath: {subpath}")
                result[int(t_id)] = _build_takt_result_from_cycle(by_subpath[subpath])
            return result

        shared_raw = takt_cycle.get("shared")
        if isinstance(shared_raw, Sequence) and not isinstance(shared_raw, (str, bytes, bytearray)):
            shared_result = _build_takt_result_from_cycle(shared_raw)
            if multi_subpath:
                return {int(t_id): shared_result for t_id in all_types}
            return {1: shared_result}
        return None

    if isinstance(takt_cycle, Sequence) and not isinstance(takt_cycle, (str, bytes, bytearray)):
        shared_result = _build_takt_result_from_cycle(takt_cycle)
        if multi_subpath:
            return {int(t_id): shared_result for t_id in all_types}
        return {1: shared_result}

    raise ValueError("takt_cycle must be a sequence or mapping")


def _build_takt_stage(
    stage_idx: int,
    stage_places: Sequence[str],
    stage_process_time: Optional[int],
    base_proc_time_map: Mapping[str, int],
    cleaning_enabled: bool,
    chamber_blocks: Mapping[str, ChamberRuntimeBlock],
) -> Optional[Dict[str, Any]]:
    valid_places = [
        str(place)
        for place in list(stage_places or [])
        if int(base_proc_time_map.get(str(place), 0) or 0) > 0
    ]
    if not valid_places:
        return None

    if stage_process_time is not None and int(stage_process_time) > 0:
        base_p = int(stage_process_time)
    else:
        base_p = max(int(base_proc_time_map[place]) for place in valid_places)
    q: Optional[int] = None
    d = 0
    if cleaning_enabled:
        cleaning_candidates: List[Tuple[int, int, int, str]] = []
        for place in valid_places:
            block = chamber_blocks[str(place)]
            trigger = int(block.cleaning_trigger_wafers)
            if trigger <= 0:
                continue
            duration = int(block.cleaning_duration)
            score = int(base_p) + duration
            cleaning_candidates.append((score, trigger, duration, place))
        if cleaning_candidates:
            _, q, d, _ = max(cleaning_candidates, key=lambda item: (item[0], item[3]))

    return {
        "name": f"s{stage_idx + 1}",
        "p": int(base_p),
        "m": len(valid_places),
        "q": q,
        "d": int(d),
    }


def _compute_takt_result_from_stage_lists(
    route_stages: Sequence[Sequence[str]],
    route_stage_process_times: Optional[Sequence[int]],
    base_proc_time_map: Mapping[str, int],
    cleaning_enabled: bool,
    chamber_blocks: Mapping[str, ChamberRuntimeBlock],
) -> Optional[Dict[str, Any]]:
    analyzer_stages: List[Dict[str, Any]] = []
    for i, stage in enumerate(route_stages):
        if not stage:
            continue
        stage_ptime: Optional[int] = None
        if route_stage_process_times is not None and i < len(route_stage_process_times):
            stage_ptime = int(route_stage_process_times[i])
        stage_cfg = _build_takt_stage(
            stage_idx=i,
            stage_places=list(stage),
            stage_process_time=stage_ptime,
            base_proc_time_map=base_proc_time_map,
            cleaning_enabled=cleaning_enabled,
            chamber_blocks=chamber_blocks,
        )
        if stage_cfg is None:
            continue
        analyzer_stages.append(stage_cfg)
    if not analyzer_stages:
        return None
    try:
        return analyze_cycle(analyzer_stages, max_parts=10000)
    except Exception:
        return None


def _compute_takt_result(
    route_stages: Sequence[Sequence[str]],
    route_stage_process_times: Optional[Sequence[int]],
    base_proc_time_map: Mapping[str, int],
    cleaning_enabled: bool,
    chamber_blocks: Mapping[str, ChamberRuntimeBlock],
    has_repeat_syntax_reentry: bool,
) -> Optional[Dict[str, Any]]:
    if not route_stages:
        return None
    if has_repeat_syntax_reentry:
        horizon = int(TAKT_HORIZON)
        return {
            "fast_takt": 0.0,
            "peak_slow_takts": [],
            "cycle_length": horizon,
            "cycle_takts": [0.0 for _ in range(horizon)],
        }
    return _compute_takt_result_from_stage_lists(
        route_stages=route_stages,
        route_stage_process_times=route_stage_process_times,
        base_proc_time_map=base_proc_time_map,
        cleaning_enabled=cleaning_enabled,
        chamber_blocks=chamber_blocks,
    )


def build_takt_payload(
    *,
    route_stages: Sequence[Sequence[str]],
    route_stage_process_times: Sequence[int],
    base_proc_time_map: Mapping[str, int],
    cleaning_enabled: bool,
    chamber_blocks: Mapping[str, ChamberRuntimeBlock],
    has_repeat_syntax_reentry: bool,
    multi_subpath: bool,
    takt_policy: str,
    takt_cycle: Any,
    wafer_type_to_subpath: Mapping[int, str],
    subpath_route_stages: Mapping[str, Sequence[Sequence[str]]],
    subpath_route_stage_process_times: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    all_types = sorted(set(int(k) for k in wafer_type_to_subpath.keys()) or {1})
    takt_override_by_type = _resolve_takt_cycle_override_by_type(
        takt_cycle=takt_cycle,
        multi_subpath=multi_subpath,
        all_types=all_types,
        wafer_type_to_subpath=wafer_type_to_subpath,
    )
    if takt_override_by_type is not None:
        takt_result_default = takt_override_by_type.get(1)
        if takt_result_default is None and takt_override_by_type:
            takt_result_default = next(iter(takt_override_by_type.values()))
        return {
            "takt_result_by_type": takt_override_by_type,
            "takt_result_default": takt_result_default,
        }

    if not multi_subpath:
        takt_result_by_type: Dict[int, Optional[Dict[str, Any]]] = {
            1: _compute_takt_result(
                route_stages=route_stages,
                route_stage_process_times=route_stage_process_times,
                base_proc_time_map=base_proc_time_map,
                cleaning_enabled=cleaning_enabled,
                chamber_blocks=chamber_blocks,
                has_repeat_syntax_reentry=has_repeat_syntax_reentry,
            )
        }
    else:
        policy = str(takt_policy or "").strip().lower()
        if policy == "split_by_subpath":
            takt_result_by_type = {}
            for t_id in all_types:
                subpath = str(wafer_type_to_subpath.get(int(t_id), "") or "")
                stages = list(subpath_route_stages.get(subpath) or [])
                stage_ptimes = list(subpath_route_stage_process_times.get(subpath) or [])
                takt_result_by_type[int(t_id)] = _compute_takt_result_from_stage_lists(
                    route_stages=stages,
                    route_stage_process_times=stage_ptimes,
                    base_proc_time_map=base_proc_time_map,
                    cleaning_enabled=cleaning_enabled,
                    chamber_blocks=chamber_blocks,
                )
        else:
            shared = _compute_takt_result(
                route_stages=route_stages,
                route_stage_process_times=route_stage_process_times,
                base_proc_time_map=base_proc_time_map,
                cleaning_enabled=cleaning_enabled,
                chamber_blocks=chamber_blocks,
                has_repeat_syntax_reentry=has_repeat_syntax_reentry,
            )
            takt_result_by_type = {int(t_id): shared for t_id in all_types}

    takt_result_default = takt_result_by_type.get(1)
    if takt_result_default is None and takt_result_by_type:
        takt_result_default = next(iter(takt_result_by_type.values()))
    return {
        "takt_result_by_type": takt_result_by_type,
        "takt_result_default": takt_result_default,
    }
