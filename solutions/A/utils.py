from typing import Dict, Tuple

def _round_to_nearest_five(value: float) -> int:
    rounded = int(round(float(value) / 5.0) * 5)
    return max(5, rounded)



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