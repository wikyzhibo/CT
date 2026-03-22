"""
级联模式变迁按钮：展示名映射；调试区按变迁顺序固定两列（每行最多两个按钮）。
"""

from __future__ import annotations

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


def build_transition_rows_two_columns(
    actions: List[ActionInfo],
) -> List[Tuple[Optional[ActionInfo], Optional[ActionInfo]]]:
    """按 `actions` 顺序每两个一排：(a0,a1), (a2,a3), … 奇数个时末行右侧为空。"""
    rows: List[Tuple[Optional[ActionInfo], Optional[ActionInfo]]] = []
    n = len(actions)
    for i in range(0, n, 2):
        left = actions[i]
        right = actions[i + 1] if i + 1 < n else None
        rows.append((left, right))
    return rows


def u_lld_tooltip_extra(lld_targets: Optional[List[str]]) -> str:
    if not lld_targets:
        return ""
    return " 去向: " + ",".join(str(x) for x in lld_targets)
