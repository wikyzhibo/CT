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
            self._draw_wafer(painter, rect)
        else:
            self._draw_status_text(painter, rect)

    def _get_contrast_color(self, bg_color: tuple) -> tuple:
        """根据背景亮度自动选择黑色或白色文字"""
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
        if luminance > 0.5:
            return (20, 20, 25)
        return (255, 255, 255)

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        """中心绘制主晶圆：精确两行布局"""
        ch = self._ch
        wafer = self.robot.wafers[0]
        cx = int(rect.center().x())
        cy = int(rect.center().y())
        r = ch.wafer_radius

        fill = self._get_wafer_color(wafer)
        text_color = self._get_contrast_color(fill)
        
        painter.setPen(QPen(self.theme.qcolor(self.theme.border), 1.5))
        painter.setBrush(QBrush(self.theme.qcolor(fill)))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # 进度环
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            progress = min(1.0, max(0.0, wafer.stay_time / wafer.proc_time))
            ring_pen = QPen(self.theme.qcolor(self.theme.accent_cyan), ch.progress_ring_width)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = r + ch.progress_ring_offset
            ring_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))

        # 字体和布局
        main_font = QFont(ch.font_family, ch.wafer_font_pt, QFont.Weight.Bold)
        sub_font = QFont(ch.font_family, ch.wafer_id_font_pt, QFont.Weight.DemiBold)
        
        main_fm = QFontMetrics(main_font)
        sub_fm = QFontMetrics(sub_font)
        
        main_height = main_fm.height()
        sub_height = sub_fm.height()
        total_height = main_height + sub_height - 4
        
        start_y = cy - total_height / 2

        # 确定内容
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            remaining = max(0, int(wafer.proc_time - wafer.stay_time))
            if remaining > 0:
                main_text = str(remaining)
            else:
                overtime = int(wafer.stay_time - wafer.proc_time)
                main_text = "SCRAP" if wafer.time_to_scrap <= 0 else f"+{overtime}"
            sub_text = f"#{wafer.token_id}"
        else:
            main_text = f"#{wafer.token_id}"
            sub_text = f"{int(wafer.stay_time)}s"

        # 主文本
        painter.setFont(main_font)
        painter.setPen(QColor(*text_color))
        main_rect = QRectF(cx - r, start_y, r * 2, main_height)
        painter.drawText(main_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, main_text)

        # 次文本
        painter.setFont(sub_font)
        sub_color = tuple(max(0, c - 30) for c in text_color) if text_color == (255, 255, 255) else text_color
        painter.setPen(QColor(*sub_color))
        sub_rect = QRectF(cx - r, start_y + main_height - 4, r * 2, sub_height)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, sub_text)

        if len(self.robot.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(ch.font_family, ch.extra_count_font_pt))
            painter.drawText(rect.adjusted(0, 0, -ch.text_margin, -ch.text_margin), 
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, 
                           f"+{len(self.robot.wafers) - 1}")

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
        if wafer.place_type == 2:
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
