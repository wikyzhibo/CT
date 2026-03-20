"""
级联模式变迁按钮：展示名映射与两列排布顺序（物理变迁名 / action_id 不变）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .algorithm_interface import ActionInfo

# 配置菜单可选路径（与 data/petri_configs/cascade_routes_1_star.json 中 routes 键一致）
CASCADE_ROUTE_OPTIONS: Tuple[str, ...] = tuple(f"1-{i}" for i in range(1, 7)) + tuple(f"2-{i}" for i in range(1, 5))

# 物理变迁名 -> 级联调试面板展示名
CASCADE_DISPLAY_NAMES: Dict[str, str] = {
    "t_LLC": "t_TM2_LLC",
    "u_LLC": "u_LLC_TM3",
    "t_LLD": "t_TM3_LLD",
    "u_LLD": "u_LLD_TM2",
}


def cascade_button_label(phys_name: str) -> str:
    return CASCADE_DISPLAY_NAMES.get(phys_name, phys_name)


def _pm_sort_key(name: str) -> Optional[int]:
    m = re.match(r"^[ut]_PM(\d+)$", name)
    return int(m.group(1)) if m else None


def build_cascade_transition_rows(
    ordered_names: List[str],
    by_name: Dict[str, ActionInfo],
    lld_targets: Optional[List[str]] = None,
) -> List[Tuple[Optional[ActionInfo], Optional[ActionInfo]]]:
    """
    生成 (左, 右) 行序列：左/右 可为 None（单行单钮时另一侧留白）。
    ordered_names: 与 net.id2t_name 顺序一致的去重变迁名列表。
    """
    consumed: set[str] = set()
    rows: List[Tuple[Optional[ActionInfo], Optional[ActionInfo]]] = []

    def pair(left_name: str, right_name: str) -> None:
        la = by_name.get(left_name)
        ra = by_name.get(right_name)
        if la is None and ra is None:
            return
        if la is not None:
            consumed.add(left_name)
        if ra is not None:
            consumed.add(right_name)
        rows.append((la, ra))

    # 1) LP 出入口
    pair("u_LP", "t_LP_done")
    # 2) LLC / LLD（TM2/TM3 语义展示名在 UI 层套）
    pair("t_LLC", "u_LLC")
    pair("t_LLD", "u_LLD")
    # 3) PM*：u_PMx 与 t_PMx 同行
    pm_nums = sorted(
        {_pm_sort_key(n) for n in by_name if _pm_sort_key(n) is not None}
    )
    for n in pm_nums:
        pair(f"u_PM{n}", f"t_PM{n}")

    # 4) 其余按 id2t 顺序各一行（每行一个按钮占左格，右格空）
    for name in ordered_names:
        if name in consumed:
            continue
        a = by_name.get(name)
        if a is None:
            continue
        consumed.add(name)
        rows.append((a, None))

    return rows


def u_lld_tooltip_extra(lld_targets: Optional[List[str]]) -> str:
    if not lld_targets:
        return ""
    return " 去向: " + ",".join(str(x) for x in lld_targets)
