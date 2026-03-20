"""
生产线节拍独特循环识别算法（独立模块）。

当前口径：
1) 快节拍 w = max_i((p_i + 20) / m_i)
2) 单工序慢节拍按“p+d 回推 m-1 拍”迭代构造，并做 max(w, ...)
3) 多工序合并时，若同位出现来自不同工序的慢节拍冲突，
   取最大慢节拍所属工序，按同样回推规则重算当前拍
"""

from __future__ import annotations

from math import gcd
from typing import Any, Dict, List, Optional


TAKT_HORIZON = 100
DELIVERY_TIME = 40

def _normalize_number(value: float) -> float | int:
    """若是整数值则返回 int，否则返回 float。"""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return float(value)


def _lcm(a: int, b: int) -> int:
    """最小公倍数。"""
    return abs(a * b) // gcd(a, b)


def build_fixed_takt_result(interval: int, horizon: int = TAKT_HORIZON) -> Dict[str, Any]:
    """
    构造固定节拍结果，返回结构与 analyze_cycle 一致。
    用于需要手动指定固定发片节拍的场景（如 route4）。
    """
    fixed_interval = int(interval)
    if fixed_interval <= 0:
        raise ValueError("interval 必须 > 0。")
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon 必须 > 0。")
    takt_value = _normalize_number(float(fixed_interval))
    cycle_takts = [takt_value for _ in range(h)]
    return {
        "fast_takt": takt_value,
        "peak_slow_takts": [],
        "cycle_length": h,
        "cycle_takts": cycle_takts,
    }


def _build_stage_takt_cycle(
    stage: Dict[str, Any], fast_takt: float
) -> Dict[str, Any]:
    """
    构造单工序节拍序列（固定 100 拍）：
    - q is None: 100 个快节拍，无慢节拍位
    - q 有值:
      1) 前段：前 (q*m-1) 个拍子为快节拍
      2) 后续循环块：重复追加
         - q 个慢节拍：按“p+d 回推 m-1 拍”迭代构造，并做 max(fast_takt, ...)
         - q*(m-1) 个快节拍；若 m==1，则该段快节拍数量按口径取 (q-1)
      直到凑满 100 个拍子（截断到 100）
    """
    p = float(stage["p"])
    m = int(stage["m"])
    q_raw = stage.get("q")
    d = float(stage.get("d", 0))
    p_plus_d = p + d

    if q_raw is None:
        return {
            "cycle": [float(fast_takt)] * TAKT_HORIZON,
            "is_slow": [False] * TAKT_HORIZON,
            "m": m,
            "p_plus_d": p_plus_d,
        }

    q = int(q_raw)
    if q <= 0:
        raise ValueError("q 若提供，必须 > 0。")

    cycle: List[float] = []
    is_slow: List[bool] = []

    # 前段快拍：q*m-1
    fast_prefix = max(0, int(q) * int(m) - 1)
    if fast_prefix > 0:
        n = min(TAKT_HORIZON, fast_prefix)
        cycle.extend([float(fast_takt)] * n)
        is_slow.extend([False] * n)

    # 循环块：m 个慢拍 + (q-1)*m 个快拍
    fast_block = (int(q)-1) * int(m)
    if fast_block < 0:
        fast_block = 0

    while len(cycle) < TAKT_HORIZON:
        # m 个慢拍
        for _ in range(int(m)):
            if len(cycle) >= TAKT_HORIZON:
                break
            lookback_sum = sum(cycle[-(m - 1) :]) if m > 1 else 0.0
            slow_takt = max(float(fast_takt), float(p_plus_d - lookback_sum))
            cycle.append(slow_takt)
            is_slow.append(True)

        if len(cycle) >= TAKT_HORIZON:
            break

        # (q-1)*m 个快拍
        if fast_block > 0:
            n = min(TAKT_HORIZON - len(cycle), fast_block)
            cycle.extend([float(fast_takt)] * n)
            is_slow.extend([False] * n)

    return {
        "cycle": cycle,
        "is_slow": is_slow,
        "m": m,
        "p_plus_d": p_plus_d,
    }


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
    if max_parts < TAKT_HORIZON:
        raise ValueError(f"max_parts 需要大于等于 {TAKT_HORIZON}。")

    normalized_stages: List[Dict[str, Any]] = []

    for idx, stage in enumerate(stages):
        stage_index = idx + 1
        default_stage_name = f"s{stage_index}"
        if not isinstance(stage, dict):
            raise ValueError(
                f"[stage#{stage_index}:{default_stage_name}] "
                f"stage 类型错误，期望 dict，实际为 {type(stage).__name__}"
            )
        stage_name = str(stage.get("name", default_stage_name))
        stage_ctx = f"[stage#{stage_index}:{stage_name}]"
        if "p" not in stage or "m" not in stage:
            raise ValueError(f"{stage_ctx} 缺少必填字段 p/m。")
        try:
            p = int(stage["p"]) + DELIVERY_TIME
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage_ctx} p 非法: {stage.get('p')!r}。") from exc
        try:
            m = int(stage["m"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage_ctx} m 非法: {stage.get('m')!r}。") from exc
        q_raw = stage.get("q")
        if q_raw is None:
            q = None
        else:
            try:
                q = int(q_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{stage_ctx} q 非法: {q_raw!r}。") from exc
        try:
            d = int(stage.get("d", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage_ctx} d 非法: {stage.get('d', 0)!r}。") from exc

        if p <= 0:
            raise ValueError(f"{stage_ctx} p 必须 > 0。")
        if m <= 0:
            raise ValueError(f"{stage_ctx} m 必须 >= 1。")
        if q is not None and q <= 0:
            raise ValueError(f"{stage_ctx} q 若提供，必须 > 0。")
        if d < 0:
            raise ValueError(f"{stage_ctx} d 不能为负。")

        normalized_stages.append(
            {
                "name": stage_name,
                "p": p,
                "m": m,
                "q": q,
                "d": d,
            }
        )

    fast_takt_raw = max(float(s["p"]) / float(s["m"]) for s in normalized_stages)
    fast_takt = _normalize_number(fast_takt_raw)

    stage_infos: List[Dict[str, Any]] = [
        _build_stage_takt_cycle(stage, fast_takt_raw) for stage in normalized_stages
    ]
    # 固定窗口口径：直接合并 100 拍，不再计算 LCM 周期。
    cycle_length = TAKT_HORIZON

    # 从前往后合并：默认逐位取 max；出现慢节拍冲突时按冲突规则回推当前拍
    cycle_takts_raw: List[float] = []
    for k in range(cycle_length):
        values_at_k: List[float] = []
        slow_candidates: List[tuple[float, Dict[str, Any]]] = []

        for info in stage_infos:
            stage_cycle = info["cycle"]
            v = float(stage_cycle[k])
            values_at_k.append(v)
            if bool(info["is_slow"][k]):
                slow_candidates.append((v, info))

        current = max(values_at_k)
        if len(slow_candidates) >= 2:
            max_slow_value, owner = max(slow_candidates, key=lambda x: x[0])
            m_owner = int(owner["m"])
            lookback_sum = (
                sum(cycle_takts_raw[-(m_owner - 1) :]) if m_owner > 1 else 0.0
            )
            current = max(
                float(fast_takt_raw),
                float(owner["p_plus_d"]) - float(lookback_sum),
                float(max_slow_value),
            )

        cycle_takts_raw.append(current)

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
        {"name": "s1", "p": 70, "m": 2, "q": 13, "d": 300},
        {"name": "s2", "p": 300, "m": 2, "q": 13, "d": 300},
        {"name": "s3", "p": 70, "m": 1, "q": None, "d": 0},
    ]

    result = analyze_cycle(stages, max_parts=10000)
    print("fast_takt =", result["fast_takt"])
    print("peak_slow_takts =", result["peak_slow_takts"])
    print("cycle_length =", result["cycle_length"])
    print("cycle_takts =", result["cycle_takts"])
