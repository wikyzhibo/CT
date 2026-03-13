from typing import Dict, List, Tuple

def _round_to_nearest_five(value: float) -> int:
    rounded = int(round(float(value) / 5.0) * 5)
    return max(5, rounded)


def _normalize_wait_durations(durations) -> List[int]:
    """将输入的等待时间列表规范化为正整数列表，默认值为 [5]"""
    values: List[int] = []
    for raw in list(durations or []):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if not values:
        values = [5]
    return sorted(set(values))


def _preprocess_process_time_map(
    process_time_map: Dict[str, int],
    chambers: Tuple[str, ...],
    defaults: Dict[str, int],
) -> Dict[str, int]:
    """按腔室清单预处理工时映射并取整到 5 的倍数。"""
    processed: Dict[str, int] = {}
    for chamber in chambers:
        raw_value = process_time_map.get(chamber, defaults[chamber])
        processed[chamber] = _round_to_nearest_five(raw_value)
    return processed