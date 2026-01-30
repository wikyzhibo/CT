"""
机械手组件
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from ..algorithm_interface import RobotState
from ..theme import ColorTheme


class RobotWidget(QWidget):
    """机械手状态显示组件"""

    def __init__(self, robot: RobotState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.robot = robot
        self.theme = theme
        self.setMinimumSize(140, 80)

    def update_state(self, robot: RobotState) -> None:
        self.robot = robot
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)

        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(self.theme.bg_surface)))
        painter.drawRoundedRect(rect, 6, 6)

        status_color = self.theme.success if self.robot.busy else self.theme.text_muted
        led_rect = QRectF(rect.left() + 6, rect.top() + 6, 10, 10)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led_rect)

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(rect.adjusted(20, 4, -6, -4), Qt.AlignTop | Qt.AlignLeft, self.robot.name)

        painter.setPen(self.theme.qcolor(self.theme.text_secondary))
        painter.setFont(QFont("Consolas", 9))
        status_text = "BUSY" if self.robot.busy else "IDLE"
        painter.drawText(rect.adjusted(6, 24, -6, -6), Qt.AlignLeft | Qt.AlignTop, status_text)

        painter.setPen(self.theme.qcolor(self.theme.text_muted))
        painter.setFont(QFont("Consolas", 8))
        count_text = f"Wafers: {len(self.robot.wafers)}"
        painter.drawText(rect.adjusted(6, 40, -6, -6), Qt.AlignLeft | Qt.AlignTop, count_text)
