"""
晶圆组件
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from ..algorithm_interface import WaferState
from ..theme import ColorTheme


class WaferWidget(QWidget):
    """晶圆显示组件"""

    def __init__(self, wafer: WaferState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.wafer = wafer
        self.theme = theme
        self.setMinimumSize(64, 64)

    def update_state(self, wafer: WaferState) -> None:
        self.wafer = wafer
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        radius = min(rect.width(), rect.height()) / 2

        fill_color = self._get_status_color()
        pen = QPen(self.theme.qcolor(self.theme.border))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.theme.qcolor(fill_color)))
        painter.drawEllipse(rect)

        # 进度环（仅加工腔室）
        if self.wafer.place_type == 1 and self.wafer.proc_time > 0:
            progress = min(1.0, max(0.0, self.wafer.stay_time / self.wafer.proc_time))
            ring_pen = QPen(self.theme.qcolor(self.theme.accent))
            ring_pen.setWidth(4)
            painter.setPen(ring_pen)
            ring_rect = QRectF(
                rect.x() + 4,
                rect.y() + 4,
                rect.width() - 8,
                rect.height() - 8,
            )
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))

        # Token ID (CAPTION - 10pt)
        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        font = QFont("Consolas", 10)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(self.wafer.token_id))

        # 状态文本（底部）(TINY - 9pt)
        painter.setPen(self.theme.qcolor(self.theme.text_muted))
        small_font = QFont("Consolas", 9)
        painter.setFont(small_font)
        status_text = f"{int(self.wafer.stay_time)}s"
        painter.drawText(0, rect.bottom() + 2, self.width(), 12, Qt.AlignCenter, status_text)

    def _get_status_color(self):
        if self.wafer.place_type == 1:
            if self.wafer.time_to_scrap <= 0:
                return self.theme.danger
            if self.wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if self.wafer.place_type == 2:
            return self.theme.info
        return self.theme.secondary
