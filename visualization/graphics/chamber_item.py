"""
腔室 QGraphicsItem - 卡片风格，IDLE/BUSY 区分，晶圆与进度环
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QDateTime, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PySide6.QtWidgets import QGraphicsItem

from ..algorithm_interface import ChamberState, WaferState
from ..theme import ColorTheme
from ..ui_params import ui_params


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

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._p.w, self._p.h)

    def paint(self, painter: QPainter, option, widget) -> None:
        p = self._p
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(p.inner_margin, p.inner_margin, p.w - p.inner_margin * 2, p.h - p.inner_margin * 2)

        now = QDateTime.currentMSecsSinceEpoch()
        flash = now < self._flash_until

        is_idle = self.chamber.status == "idle"
        if is_idle and not flash:
            bg = self.theme.bg_deep
            border = self.theme.border_muted
            grid_color = self.theme.dim_color(self.theme.border_muted, 0.6)
        else:
            bg = self.theme.bg_surface
            border = self.theme.accent_cyan if (not is_idle or flash) else self.theme.border_muted
            grid_color = self.theme.border_muted

        pen_width = 2 if flash else 1
        painter.setPen(QPen(self.theme.qcolor(border), pen_width))
        painter.setBrush(QBrush(self.theme.qcolor(bg)))
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        step = p.grid_step
        painter.setPen(QPen(self.theme.qcolor(grid_color)))
        for x in range(int(rect.left()), int(rect.right()), step):
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
        for y in range(int(rect.top()), int(rect.bottom()), step):
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

        status_color = self._status_color()
        led = QRectF(rect.right() - p.led_size - 6, rect.top() + 6, p.led_size, p.led_size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led)

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.name_font_pt))
        painter.drawText(rect.adjusted(p.text_margin, 4, -p.text_margin, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.chamber.name)

        if self.chamber.wafers:
            self._draw_wafer(painter, rect)

    def _status_color(self):
        if self.chamber.status == "danger":
            return self.theme.danger
        if self.chamber.status == "warning":
            return self.theme.warning
        if self.chamber.status == "active":
            return self.theme.success
        return self.theme.text_muted

    def _get_wafer_color(self, wafer: WaferState):
        if wafer.place_type == 1:
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if getattr(wafer, "proc_time", 0) and wafer.stay_time >= wafer.proc_time:
                return getattr(self.theme, "complete_orange", self.theme.warning)
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:
            return self.theme.info
        return self.theme.secondary

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        p = self._p
        wafer = self.chamber.wafers[0]
        cx = rect.center().x()
        cy = rect.center().y()
        r = p.wafer_radius

        fill = self._get_wafer_color(wafer)
        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(fill)))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            progress = min(1.0, max(0.0, wafer.stay_time / wafer.proc_time))
            ring_pen = QPen(self.theme.qcolor(self.theme.accent_cyan), p.progress_ring_width)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = r + p.progress_ring_offset
            ring_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.wafer_font_pt))
        painter.drawText(QRectF(cx - r, cy - r, r * 2, r * 2), Qt.AlignmentFlag.AlignCenter, str(wafer.token_id))

        if len(self.chamber.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(p.font_family, p.extra_count_font_pt))
            painter.drawText(rect.adjusted(0, 0, -p.text_margin, -p.text_margin), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, f"+{len(self.chamber.wafers) - 1}")

    def update_state(self, chamber: ChamberState) -> None:
        prev = self._last_status
        self.chamber = chamber
        if prev is not None and prev != chamber.status:
            ms = self._p.flash_ms
            self._flash_until = QDateTime.currentMSecsSinceEpoch() + ms
            QTimer.singleShot(ms + 50, self.update)
        self._last_status = chamber.status
        self.update()
