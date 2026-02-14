"""
机械手 QGraphicsItem - 腔室式卡片：与腔室同尺寸、同布局

改进：
- IDLE 水印更淡（alpha更低、字号更小）
- 晶圆文字使用自动对比色
- 高质量抗锯齿渲染
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QDateTime, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QFont, QColor, QFontMetrics
from PySide6.QtWidgets import QGraphicsItem

from ..algorithm_interface import RobotState, WaferState
from ..theme import ColorTheme
from ..ui_params import ui_params
from .wafer_item import WaferItem


class RobotItem(QGraphicsItem):
    """机械手卡片：腔室式布局，名称置顶、状态灯右上、有晶圆时中心显示晶圆"""

    def __init__(self, robot: RobotState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.robot = robot
        self.theme = theme
        self._p = ui_params.robot_item
        self._ch = ui_params.chamber_item  # 晶圆绘制复用腔室参数
        self._last_busy: bool | None = None
        self._flash_until_ms: int = 0
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # 创建 WaferItem 子组件
        self.wafer_item = WaferItem(theme, parent=self)
        self.wafer_item.setVisible(False)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._p.w, self._p.h)

    def paint(self, painter: QPainter, option, widget) -> None:
        p = self._p
        ch = self._ch
        
        # 高质量渲染
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        
        # 坐标取整
        rect = QRectF(
            int(p.inner_margin), 
            int(p.inner_margin), 
            int(p.w - p.inner_margin * 2), 
            int(p.h - p.inner_margin * 2)
        )

        is_busy = self.robot.busy
        now_ms = QDateTime.currentMSecsSinceEpoch()
        flash = now_ms < self._flash_until_ms

        if is_busy or flash:
            bg = self.theme.bg_surface
            border = self.theme.accent_cyan if (is_busy or flash) else self.theme.border_muted
            grid_alpha = 50
        else:
            bg = self.theme.bg_deep
            border = self.theme.border_muted
            grid_alpha = 30

        pen_width = 2 if flash else 1.5
        painter.setPen(QPen(self.theme.qcolor(border), pen_width))
        painter.setBrush(QBrush(self.theme.qcolor(bg)))
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        # 网格更淡
        grid_color = QColor(*self.theme.border_muted)
        grid_color.setAlpha(grid_alpha)
        painter.setPen(QPen(grid_color, 0.5))
        step = p.grid_step
        for x in range(int(rect.left()), int(rect.right()), step):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(int(rect.top()), int(rect.bottom()), step):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        # LED
        status_color = self.theme.success if is_busy else self.theme.text_muted
        led = QRectF(int(rect.right() - p.led_size - 6), int(rect.top() + 6), p.led_size, p.led_size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led)

        # 名称
        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.title_font_pt, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(p.text_margin, 4, -p.text_margin, -4), 
                        Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, 
                        self.robot.name)

        if self.robot.wafers:
            self.wafer_item.set_wafer(self.robot.wafers[0])
            self.wafer_item.setVisible(True)
            self.wafer_item.setPos(rect.center())
            
            # 多晶圆计数
            if len(self.robot.wafers) > 1:
                painter.setPen(self.theme.qcolor(self.theme.text_muted))
                painter.setFont(QFont(ch.font_family, ch.extra_count_font_pt))
                painter.drawText(rect.adjusted(0, 0, -ch.text_margin, -ch.text_margin), 
                               Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, 
                               f"+{len(self.robot.wafers) - 1}")
        else:
            self.wafer_item.setVisible(False)
            self._draw_status_text(painter, rect)

    def _draw_status_text(self, painter: QPainter, rect: QRectF) -> None:
        """无晶圆时中心显示 IDLE / BUSY：做成更淡的水印"""
        p = self._p
        text = "BUSY" if self.robot.busy else "IDLE"
        
        if self.robot.busy:
            # BUSY 稍显眼
            painter.setPen(self.theme.qcolor(self.theme.text_secondary))
            painter.setFont(QFont(p.font_family, p.status_font_pt, QFont.Weight.DemiBold))
        else:
            # IDLE 做成极淡水印
            color = QColor(*self.theme.text_muted)
            color.setAlpha(80)  # 非常淡
            painter.setPen(color)
            painter.setFont(QFont(p.font_family, p.status_font_pt - 2))  # 更小
        
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def update_state(self, robot: RobotState) -> None:
        prev = self._last_busy
        self.robot = robot
        if prev is not None and robot.busy != prev:
            ms = self._p.flash_ms
            self._flash_until_ms = QDateTime.currentMSecsSinceEpoch() + ms
            QTimer.singleShot(ms + 50, self._clear_flash)
        self._last_busy = robot.busy
        self.update()

    def _clear_flash(self) -> None:
        self._flash_until_ms = 0
        self.update()
