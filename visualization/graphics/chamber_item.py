"""
腔室 QGraphicsItem - 卡片风格，IDLE/BUSY 区分，晶圆与进度环

改进：
- 晶圆文字采用 QFontMetrics 精确垂直居中
- 自动对比色文字（根据背景亮度选择黑/白）
- 高质量抗锯齿渲染
- 坐标整数对齐防抖动
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QDateTime, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QGraphicsItem

from ..algorithm_interface import ChamberState, WaferState
from ..theme import ColorTheme
from ..ui_params import ui_params
from .wafer_item import WaferItem


class ChamberItem(QGraphicsItem):
    """腔室卡片：IDLE 更暗/低对比，BUSY 高亮；晶圆带进度环，完成橙/报废红"""

    def __init__(self, chamber: ChamberState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.chamber = chamber
        self.theme = theme
        self._p = ui_params.chamber_item
        self._last_status: str | None = None
        self._flash_until: int = 0
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 创建子组件 WaferItem
        self.wafer_item = WaferItem(theme, parent=self)
        self.wafer_item.setVisible(False)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._p.w, self._p.h)

    def paint(self, painter: QPainter, option, widget) -> None:
        p = self._p
        # 启用高质量渲染
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # 坐标取整防止亚像素抖动
        rect = QRectF(
            int(p.inner_margin), 
            int(p.inner_margin), 
            int(p.w - p.inner_margin * 2), 
            int(p.h - p.inner_margin * 2)
        )

        now = QDateTime.currentMSecsSinceEpoch()

        is_idle = self.chamber.status == "idle"
        is_active = self.chamber.status == "active"
        
        if is_idle:
            bg = self.theme.bg_deep
            border = self.theme.border_muted
            grid_alpha = 40
        else:
            bg = self.theme.bg_elevated
            if is_active:
                border = self.theme.border_active
            else:
                border = self.theme.border_muted
            grid_alpha = 60

        # 1. 绘制阴影（更轻微）
        shadow_rect = rect.adjusted(-1, -1, 1, 1)
        shadow_color = QColor(0, 0, 0, 40)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(shadow_color))
        painter.drawRoundedRect(shadow_rect, p.corner_radius + 1, p.corner_radius + 1)

        # 2. 绘制背景
        painter.setBrush(QBrush(self.theme.qcolor(bg)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        # 3. 绘制边框
        pen_width = 2.0
        painter.setPen(QPen(self.theme.qcolor(border), pen_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        # 4. 绘制网格（更淡）
        grid_color = QColor(*self.theme.border_muted)
        grid_color.setAlpha(grid_alpha)
        painter.setPen(QPen(grid_color, 0.5))
        step = p.grid_step
        for x in range(int(rect.left()) + step, int(rect.right()), step):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(int(rect.top()) + step, int(rect.bottom()), step):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        # 5. 状态 LED（带发光）
        status_color = self._status_color()
        led_size = p.led_size
        led_x = int(rect.right() - led_size - 6)
        led_y = int(rect.top() + 6)
        led = QRectF(led_x, led_y, led_size, led_size)
        
        # 发光效果
        glow_rect = led.adjusted(-2, -2, 2, 2)
        glow_color = QColor(*status_color, 60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(glow_rect)
        
        # LED 本体
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led)

        # 6. 腔室名称
        title_font = QFont(p.font_family, p.name_font_pt, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(self.theme.qcolor(self.theme.accent_cyan))
        name_rect = QRectF(rect.left() + p.text_margin, rect.top() + 4, 
                          rect.width() - p.text_margin * 2 - led_size - 8, 20)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                        self.chamber.name)



        # 更新晶圆状态
        if self.chamber.wafers:
            self.wafer_item.set_wafer(self.chamber.wafers[0])
            self.wafer_item.setVisible(True)
            # 居中放置
            self.wafer_item.setPos(rect.center())
            
            # 多晶圆计数
            if len(self.chamber.wafers) > 1:
                painter.setPen(self.theme.qcolor(self.theme.text_muted))
                painter.setFont(QFont(p.font_family, p.extra_count_font_pt))
                painter.drawText(rect.adjusted(0, 0, -p.text_margin, -p.text_margin), 
                               Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, 
                               f"+{len(self.chamber.wafers) - 1}")
        else:
            self.wafer_item.setVisible(False)

    def _status_color(self):
        if self.chamber.status == "danger":
            return self.theme.danger
        if self.chamber.status == "warning":
            return self.theme.warning
        if self.chamber.status == "active":
            return self.theme.success
        return self.theme.text_muted



    def update_state(self, chamber: ChamberState) -> None:
        self.chamber = chamber
        self.update()
