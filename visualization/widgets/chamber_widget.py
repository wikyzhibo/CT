"""
腔室组件
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from ..algorithm_interface import ChamberState
from ..theme import ColorTheme


class ChamberWidget(QWidget):
    """腔室显示组件"""

    def __init__(self, chamber: ChamberState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.chamber = chamber
        self.theme = theme
        self.setMinimumSize(130, 130)

    def update_state(self, chamber: ChamberState) -> None:
        self.chamber = chamber
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        self._draw_background(painter, rect)
        self._draw_status_led(painter, rect)
        self._draw_title(painter, rect)
        self._draw_wafer(painter, rect)

    def _draw_background(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(self.theme.bg_surface)))
        painter.drawRoundedRect(rect, 6, 6)

        # 网格纹理
        painter.setPen(QPen(self.theme.qcolor(self.theme.border_muted)))
        step = 12
        for x in range(int(rect.left()), int(rect.right()), step):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(int(rect.top()), int(rect.bottom()), step):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

    def _draw_status_led(self, painter: QPainter, rect: QRectF) -> None:
        status_color = self._get_status_color()
        led_rect = QRectF(rect.right() - 16, rect.top() + 6, 10, 10)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led_rect)

    def _draw_title(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        font = QFont("Consolas", 10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(6, 4, -6, -4), Qt.AlignTop | Qt.AlignLeft, self.chamber.name)

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        if not self.chamber.wafers:
            return

        wafer = self.chamber.wafers[0]
        center_x = rect.center().x()
        center_y = rect.center().y()
        radius = 20

        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(self._get_wafer_color(wafer))))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(
            QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2),
            Qt.AlignCenter,
            str(wafer.token_id),
        )

        if len(self.chamber.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(rect.adjusted(0, 0, -6, -6), Qt.AlignBottom | Qt.AlignRight, f"+{len(self.chamber.wafers) - 1}")

    def _get_status_color(self):
        if self.chamber.status == "danger":
            return self.theme.danger
        if self.chamber.status == "warning":
            return self.theme.warning
        if self.chamber.status == "active":
            return self.theme.success
        return self.theme.text_muted

    def _get_wafer_color(self, wafer) -> tuple[int, int, int]:
        if wafer.place_type == 1:
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:
            return self.theme.info
        return self.theme.secondary
