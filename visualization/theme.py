"""
PySide6 主题配色
从 Pygame viz.py 的 ColorTheme 迁移
"""

from dataclasses import dataclass
from typing import Tuple

from PySide6.QtGui import QColor


@dataclass
class ColorTheme:
    # 主色调
    primary: Tuple[int, int, int] = (0, 255, 65)
    secondary: Tuple[int, int, int] = (0, 143, 17)
    accent: Tuple[int, int, int] = (0, 200, 255)

    # 状态色
    success: Tuple[int, int, int] = (0, 255, 65)
    warning: Tuple[int, int, int] = (255, 184, 0)
    danger: Tuple[int, int, int] = (255, 51, 51)
    info: Tuple[int, int, int] = (59, 130, 246)

    # 背景层次（深蓝黑工业风）
    bg_deepest: Tuple[int, int, int] = (14, 18, 26)
    bg_fog: Tuple[int, int, int] = (22, 28, 38)
    bg_deep: Tuple[int, int, int] = (28, 35, 45)
    bg_surface: Tuple[int, int, int] = (40, 48, 60)
    bg_elevated: Tuple[int, int, int] = (55, 65, 78)

    # 边框
    border: Tuple[int, int, int] = (70, 85, 105)
    border_muted: Tuple[int, int, int] = (50, 62, 78)
    border_active: Tuple[int, int, int] = (0, 255, 65)

    # 文字
    text_primary: Tuple[int, int, int] = (230, 237, 243)
    text_secondary: Tuple[int, int, int] = (139, 148, 158)
    text_muted: Tuple[int, int, int] = (100, 110, 125)
    text_kpi: Tuple[int, int, int] = (0, 200, 255)
    accent_cyan: Tuple[int, int, int] = (0, 200, 255)
    complete_orange: Tuple[int, int, int] = (255, 140, 50)

    # 按钮分类颜色
    btn_transition: Tuple[int, int, int] = (0, 200, 255)
    btn_wait: Tuple[int, int, int] = (255, 184, 0)
    btn_random: Tuple[int, int, int] = (168, 85, 247)
    btn_model: Tuple[int, int, int] = (0, 255, 65)
    btn_auto: Tuple[int, int, int] = (255, 107, 53)
    btn_speed: Tuple[int, int, int] = (59, 130, 246)
    btn_reset: Tuple[int, int, int] = (255, 51, 51)
    btn_gantt: Tuple[int, int, int] = (59, 130, 246)
    btn_save: Tuple[int, int, int] = (0, 255, 65)
    btn_bug: Tuple[int, int, int] = (255, 184, 0)

    def qcolor(self, color: Tuple[int, int, int], alpha: int | None = None) -> QColor:
        if alpha is None:
            return QColor(*color)
        return QColor(color[0], color[1], color[2], alpha)

    def dim_color(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))

    def brighten_color(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        return (
            min(255, int(color[0] * (1 + factor))),
            min(255, int(color[1] * (1 + factor))),
            min(255, int(color[2] * (1 + factor))),
        )
