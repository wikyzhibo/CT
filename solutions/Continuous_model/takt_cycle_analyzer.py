"""
生产线节拍独特循环识别算法（独立模块）。

当前实现口径：
1) 先按工序构造“独立节拍周期”
2) 再按 LCM 周期逐位取 max，得到整线节拍周期
3) 慢节拍定义为 (p + d) / m（仅 q 不为 None 时生效）
"""

from __future__ import annotations

from math import gcd
from typing import Any, Dict, List, Optional


def _normalize_number(value: float) -> float | int:
    """若是整数值则返回 int，否则返回 float。"""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return float(value)


def _lcm(a: int, b: int) -> int:
    """最小公倍数。"""
    return abs(a * b) // gcd(a, b)


def _build_stage_takt_cycle(stage: Dict[str, Any], fast_takt: float) -> List[float]:
    """
    构造单工序节拍周期：
    - q is None: 恒为 [fast_takt]
    - q 有值: 周期长度 q*m+m，前 q*m 项为 fast_takt，后 m 项为 slow_takt=(p+d)/m
    """
    p = float(stage["p"])
    m = int(stage["m"])
    q_raw = stage.get("q")
    d = float(stage.get("d", 0))

    if q_raw is None:
        return [float(fast_takt)]

    q = int(q_raw)
    if q <= 0:
        raise ValueError("q 若提供，必须 > 0。")

    slow_takt = (p + d) / m
    fast_count = q * m
    return [float(fast_takt)] * fast_count + [float(slow_takt)] * m


def analyze_cycle(stages: List[Dict[str, Any]], max_parts: int = 10000) -> Dict[str, Any]:
    """
    计算产线节拍独特循环（工序周期叠加口径）。

    参数:
        stages: 每道工序配置列表，字段:
            - name: 工序名（可选）
            - p: 单件加工时间（>0）
            - m: 并行机器数（>=1）
            - q: 每台机器加工 q 件后维护（None 表示无维护）
            - d: 每次维护耗时（无维护可为 0）
        max_parts: 最大可接受循环长度；若 LCM 周期超限则抛异常

    返回:
        {
            "fast_takt": ...,
            "peak_slow_takts": [...],
            "cycle_length": ...,
            "cycle_takts": [...]
        }
    """
    if not stages:
        raise ValueError("stages 不能为空。")
    if max_parts <= 1:
        raise ValueError("max_parts 需要大于 1。")

    p_list: List[int] = []
    m_list: List[int] = []
    q_list: List[Optional[int]] = []
    d_list: List[int] = []

    for idx, stage in enumerate(stages):
        if "p" not in stage or "m" not in stage:
            raise ValueError(f"第 {idx} 道工序缺少必填字段 p/m。")
        p = int(stage["p"])
        m = int(stage["m"])
        q_raw = stage.get("q")
        q = None if q_raw is None else int(q_raw)
        d = int(stage.get("d", 0))

        if p <= 0:
            raise ValueError(f"第 {idx} 道工序 p 必须 > 0。")
        if m <= 0:
            raise ValueError(f"第 {idx} 道工序 m 必须 >= 1。")
        if q is not None and q <= 0:
            raise ValueError(f"第 {idx} 道工序 q 若提供，必须 > 0。")
        if d < 0:
            raise ValueError(f"第 {idx} 道工序 d 不能为负。")

        p_list.append(p)
        m_list.append(m)
        q_list.append(q)
        d_list.append(d)

    fast_takt_raw = max(p / m for p, m in zip(p_list, m_list))
    fast_takt = _normalize_number(fast_takt_raw)

    stage_cycles: List[List[float]] = [
        _build_stage_takt_cycle(stage, fast_takt_raw) for stage in stages
    ]
    stage_cycle_lens = [len(c) for c in stage_cycles]

    cycle_length = 1
    for length in stage_cycle_lens:
        cycle_length = _lcm(cycle_length, int(length))

    if cycle_length > max_parts:
        raise RuntimeError(
            f"在 max_parts={max_parts} 内未找到循环（LCM 周期长度为 {cycle_length}）。"
        )

    # 逐位叠加，整线节拍取各工序同位最大值
    cycle_takts_raw: List[float] = []
    for k in range(cycle_length):
        v = max(stage_cycle[k % len(stage_cycle)] for stage_cycle in stage_cycles)
        cycle_takts_raw.append(v)

    cycle_takts = [_normalize_number(v) for v in cycle_takts_raw]
    peak_slow_takts = sorted(
        {
            _normalize_number(v)
            for v in cycle_takts_raw
            if (v - fast_takt_raw) > 1e-9
        }
    )

    return {
        "fast_takt": fast_takt,
        "peak_slow_takts": peak_slow_takts,
        "cycle_length": cycle_length,
        "cycle_takts": cycle_takts,
    }


if __name__ == "__main__":
    stages = [
        {"name": "s1", "p": 100, "m": 1, "q": None, "d": 200},
        {"name": "s2", "p": 300, "m": 2, "q": 2, "d": 200},
        {"name": "s3", "p": 200, "m": 1, "q": None, "d": 0},
    ]

    result = analyze_cycle(stages, max_parts=10000)
    print("fast_takt =", result["fast_takt"])
    print("peak_slow_takts =", result["peak_slow_takts"])
    print("cycle_length =", result["cycle_length"])
    print("cycle_takts =", result["cycle_takts"])
