"""
机械手 QGraphicsItem - 与腔室卡片一致风格
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QDateTime, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QGraphicsItem

from ..algorithm_interface import RobotState
from ..theme import ColorTheme
from ..ui_params import ui_params


class RobotItem(QGraphicsItem):
    """机械手卡片：与腔室统一视觉，标题置顶、状态灯在角；状态变化时边框闪烁"""

    def __init__(self, robot: RobotState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.robot = robot
        self.theme = theme
        self._p = ui_params.robot_item
        self._last_busy: bool | None = None
        self._flash_until_ms: int = 0
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._p.w, self._p.h)

    def paint(self, painter: QPainter, option, widget) -> None:
        p = self._p
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(p.inner_margin, p.inner_margin, p.w - p.inner_margin * 2, p.h - p.inner_margin * 2)

        is_busy = self.robot.busy
        now_ms = QDateTime.currentMSecsSinceEpoch()
        flash = now_ms < self._flash_until_ms
        bg = self.theme.bg_surface
        if flash:
            border = self.theme.accent_cyan
        else:
            border = self.theme.accent_cyan if is_busy else self.theme.border_muted
        pen_width = 2 if flash else 1
        painter.setPen(QPen(self.theme.qcolor(border), pen_width))
        painter.setBrush(QBrush(self.theme.qcolor(bg)))
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        step = p.grid_step
        grid_color = self.theme.dim_color(self.theme.border_muted, 0.7)
        painter.setPen(QPen(self.theme.qcolor(grid_color)))
        for x in range(int(rect.left()), int(rect.right()), step):
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
        for y in range(int(rect.top()), int(rect.bottom()), step):
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

        status_color = self.theme.success if self.robot.busy else self.theme.text_muted
        led = QRectF(rect.left() + p.led_offset, rect.top() + p.led_offset, p.led_size, p.led_size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led)

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.title_font_pt))
        painter.drawText(rect.adjusted(p.title_left_offset, 6, -p.text_margin, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.robot.name)

        painter.setPen(self.theme.qcolor(self.theme.text_secondary))
        painter.setFont(QFont(p.font_family, p.status_font_pt))
        status_text = "BUSY" if self.robot.busy else "IDLE"
        painter.drawText(rect.adjusted(p.text_margin, 32, -p.text_margin, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, status_text)

        painter.setPen(self.theme.qcolor(self.theme.text_muted))
        painter.setFont(QFont(p.font_family, p.wafers_font_pt))
        painter.drawText(rect.adjusted(p.text_margin, 52, -p.text_margin, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"Wafers: {len(self.robot.wafers)}")

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
