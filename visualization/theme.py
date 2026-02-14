"""
PySide6 主题配色 - 工业控制台风格

改进：
- 更清晰的背景层次对比
- 更柔和的晶圆状态色（降低饱和度）
- 更干净的边框层级
"""

from dataclasses import dataclass
from typing import Tuple

from PySide6.QtGui import QColor


@dataclass
class ColorTheme:
    # 主色调
    primary: Tuple[int, int, int] = (0, 230, 80)  # 降低亮度
    secondary: Tuple[int, int, int] = (0, 143, 17)
    accent: Tuple[int, int, int] = (0, 180, 230)  # 降低亮度

    # 状态色（降低饱和度，更工业化）
    success: Tuple[int, int, int] = (50, 205, 100)     # 柔和绿
    warning: Tuple[int, int, int] = (235, 170, 50)     # 柔和橙黄
    danger: Tuple[int, int, int] = (220, 80, 80)       # 柔和红
    info: Tuple[int, int, int] = (80, 140, 220)        # 柔和蓝

    # 背景层次（更强对比）
    bg_deepest: Tuple[int, int, int] = (10, 12, 18)    # 最深背景
    bg_fog: Tuple[int, int, int] = (18, 22, 30)        # 画布背景
    bg_deep: Tuple[int, int, int] = (24, 30, 40)       # IDLE 卡片
    bg_surface: Tuple[int, int, int] = (38, 46, 58)    # 普通卡片
    bg_elevated: Tuple[int, int, int] = (52, 62, 76)   # 激活卡片

    # 边框（更干净的层级）
    border: Tuple[int, int, int] = (60, 75, 95)        # 普通边框
    border_muted: Tuple[int, int, int] = (85, 105, 130)  # 淡边框
    border_active: Tuple[int, int, int] = (0, 200, 255) # 激活边框

    # 文字
    text_primary: Tuple[int, int, int] = (230, 235, 240)
    text_secondary: Tuple[int, int, int] = (145, 155, 165)
    text_muted: Tuple[int, int, int] = (90, 100, 115)
    text_kpi: Tuple[int, int, int] = (0, 180, 230)
    accent_cyan: Tuple[int, int, int] = (0, 180, 230)
    complete_orange: Tuple[int, int, int] = (235, 140, 60)

    # 按钮分类颜色（更柔和）
    btn_transition: Tuple[int, int, int] = (0, 170, 220)
    btn_wait: Tuple[int, int, int] = (220, 165, 40)
    btn_random: Tuple[int, int, int] = (145, 90, 210)
    btn_model: Tuple[int, int, int] = (50, 200, 90)
    btn_auto: Tuple[int, int, int] = (220, 100, 60)
    btn_speed: Tuple[int, int, int] = (70, 130, 210)
    btn_reset: Tuple[int, int, int] = (200, 70, 70)
    btn_gantt: Tuple[int, int, int] = (70, 130, 210)
    btn_save: Tuple[int, int, int] = (50, 200, 90)
    btn_bug: Tuple[int, int, int] = (220, 165, 40)

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
