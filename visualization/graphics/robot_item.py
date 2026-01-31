"""
机械手 QGraphicsItem - 腔室式卡片：与腔室同尺寸、同布局（名称在上、晶圆居中）
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QDateTime, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QGraphicsItem

from ..algorithm_interface import RobotState, WaferState
from ..theme import ColorTheme
from ..ui_params import ui_params


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

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._p.w, self._p.h)

    def paint(self, painter: QPainter, option, widget) -> None:
        p = self._p
        ch = self._ch
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(p.inner_margin, p.inner_margin, p.w - p.inner_margin * 2, p.h - p.inner_margin * 2)

        is_busy = self.robot.busy
        now_ms = QDateTime.currentMSecsSinceEpoch()
        flash = now_ms < self._flash_until_ms

        if is_busy or flash:
            bg = self.theme.bg_surface
            border = self.theme.accent_cyan if (is_busy or flash) else self.theme.border_muted
            grid_color = self.theme.border_muted
        else:
            bg = self.theme.bg_deep
            border = self.theme.border_muted
            grid_color = self.theme.dim_color(self.theme.border_muted, 0.6)

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

        status_color = self.theme.success if is_busy else self.theme.text_muted
        led = QRectF(rect.right() - p.led_size - 6, rect.top() + 6, p.led_size, p.led_size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led)

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.title_font_pt))
        painter.drawText(rect.adjusted(p.text_margin, 4, -p.text_margin, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.robot.name)

        if self.robot.wafers:
            self._draw_wafer(painter, rect)
        else:
            self._draw_status_text(painter, rect)

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        """中心绘制主晶圆；机械手中显示编号+停留时间，配色 viz：正常绿、超时红。"""
        ch = self._ch
        wafer = self.robot.wafers[0]
        cx = rect.center().x()
        cy = rect.center().y()
        r = ch.wafer_radius

        fill = self._get_wafer_color(wafer)
        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(fill)))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            progress = min(1.0, max(0.0, wafer.stay_time / wafer.proc_time))
            ring_pen = QPen(self.theme.qcolor(self.theme.accent_cyan), ch.progress_ring_width)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = r + ch.progress_ring_offset
            ring_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))

        # 机械手晶圆：编号(上) + 停留时间(下)，配色 viz
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            remaining = max(0, int(wafer.proc_time - wafer.stay_time))
            if remaining > 0:
                painter.setPen(self.theme.qcolor(fill))
                painter.setFont(QFont(ch.font_family, ch.wafer_font_pt))
                painter.drawText(QRectF(cx - r, cy - r - 8, r * 2, 28), Qt.AlignmentFlag.AlignCenter, str(remaining))
            else:
                painter.setPen(self.theme.qcolor(fill))
                painter.setFont(QFont(ch.font_family, ch.wafer_font_pt))
                overtime = int(wafer.stay_time - wafer.proc_time)
                done_text = "SCRAP" if wafer.time_to_scrap <= 0 else f"+{overtime}s"
                painter.drawText(QRectF(cx - r, cy - r - 8, r * 2, 28), Qt.AlignmentFlag.AlignCenter, done_text)
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(ch.font_family, ch.wafer_id_font_pt))
            painter.drawText(QRectF(cx - r, cy - 4, r * 2, 24), Qt.AlignmentFlag.AlignCenter, f"#{wafer.token_id}")
        else:
            painter.setPen(self.theme.qcolor(self.theme.text_primary))
            painter.setFont(QFont(ch.font_family, ch.wafer_font_pt))
            painter.drawText(QRectF(cx - r, cy - r - 8, r * 2, 28), Qt.AlignmentFlag.AlignCenter, f"#{wafer.token_id}")
            time_color = fill if fill != self.theme.success else self.theme.text_primary
            painter.setPen(self.theme.qcolor(time_color))
            painter.setFont(QFont(ch.font_family, ch.wafer_id_font_pt))
            painter.drawText(QRectF(cx - r, cy - 4, r * 2, 24), Qt.AlignmentFlag.AlignCenter, f"{int(wafer.stay_time)}s")

        if len(self.robot.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(ch.font_family, ch.extra_count_font_pt))
            painter.drawText(rect.adjusted(0, 0, -ch.text_margin, -ch.text_margin), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, f"+{len(self.robot.wafers) - 1}")

    def _draw_status_text(self, painter: QPainter, rect: QRectF) -> None:
        """无晶圆时中心显示 IDLE / BUSY。"""
        p = self._p
        painter.setPen(self.theme.qcolor(self.theme.text_muted))
        painter.setFont(QFont(p.font_family, p.status_font_pt))
        text = "BUSY" if self.robot.busy else "IDLE"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _get_wafer_color(self, wafer: WaferState):
        """机械手晶圆配色对齐 viz：运输位 stay<7 绿，7–10 黄，≥10 红。"""
        if wafer.place_type == 1:
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if getattr(wafer, "proc_time", 0) and wafer.stay_time >= wafer.proc_time:
                return getattr(self.theme, "complete_orange", self.theme.warning)
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:  # 运输位：正常绿，超时红
            stay = int(wafer.stay_time)
            if stay >= 10:
                return self.theme.danger
            if stay >= 7:
                return self.theme.warning
            return self.theme.success
        return self.theme.secondary

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
