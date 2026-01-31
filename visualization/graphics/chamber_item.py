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
        flash = now < self._flash_until

        is_idle = self.chamber.status == "idle"
        if is_idle and not flash:
            bg = self.theme.bg_deep
            border = self.theme.border_muted
            grid_alpha = 40  # 网格更淡
        else:
            bg = self.theme.bg_elevated
            border = self.theme.accent_cyan if (not is_idle or flash) else self.theme.border_muted
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
        pen_width = 2.5 if flash else 1.5
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

    def _get_contrast_color(self, bg_color: tuple) -> tuple:
        """根据背景亮度自动选择黑色或白色文字"""
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
        if luminance > 0.5:
            return (20, 20, 25)  # 深色文字
        return (255, 255, 255)  # 白色文字

    def _get_wafer_color(self, wafer: WaferState):
        """配色对齐 viz.py：加工腔 stay/proc 判断，运输位 stay 7/10 阈值。"""
        if wafer.place_type == 1:  # 加工腔室
            proc = getattr(wafer, "proc_time", 0) or 0
            stay = int(wafer.stay_time)
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if proc > 0 and stay >= proc + 20:
                return self.theme.danger
            if proc > 0 and stay >= proc:
                return self.theme.warning
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:  # 运输位
            stay = int(wafer.stay_time)
            if stay >= 10:
                return self.theme.danger
            if stay >= 7:
                return self.theme.warning
            return self.theme.success
        return self.theme.secondary

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        """晶圆绘制：精确两行布局，使用 QFontMetrics 计算真实高度"""
        p = self._p
        wafer = self.chamber.wafers[0]
        
        # 圆心坐标取整
        cx = int(rect.center().x())
        cy = int(rect.center().y())
        r = p.wafer_radius

        fill = self._get_wafer_color(wafer)
        text_color = self._get_contrast_color(fill)
        
        # 绘制晶圆圆形
        painter.setPen(QPen(self.theme.qcolor(self.theme.border), 1.5))
        painter.setBrush(QBrush(self.theme.qcolor(fill)))
        wafer_rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        painter.drawEllipse(wafer_rect)

        # 进度环
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            progress = min(1.0, max(0.0, wafer.stay_time / wafer.proc_time))
            ring_pen = QPen(self.theme.qcolor(self.theme.accent_cyan), p.progress_ring_width)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = r + p.progress_ring_offset
            ring_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))

        # 准备字体
        main_font = QFont(p.font_family, p.wafer_font_pt, QFont.Weight.Bold)
        sub_font = QFont(p.font_family, p.wafer_id_font_pt, QFont.Weight.DemiBold)
        
        main_fm = QFontMetrics(main_font)
        sub_fm = QFontMetrics(sub_font)
        
        main_height = main_fm.height()
        sub_height = sub_fm.height()
        total_height = main_height + sub_height - 4  # 减少行间距
        
        # 计算两行垂直居中的起始 y
        start_y = cy - total_height / 2

        # 确定显示内容
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

        # 绘制主文本（上半）
        painter.setFont(main_font)
        painter.setPen(QColor(*text_color))
        main_rect = QRectF(cx - r, start_y, r * 2, main_height)
        painter.drawText(main_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, main_text)

        # 绘制次文本（下半）
        painter.setFont(sub_font)
        # 次文本稍暗
        sub_color = tuple(max(0, c - 30) for c in text_color) if text_color == (255, 255, 255) else text_color
        painter.setPen(QColor(*sub_color))
        sub_rect = QRectF(cx - r, start_y + main_height - 4, r * 2, sub_height)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, sub_text)

        # 额外晶圆计数
        if len(self.chamber.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(p.font_family, p.extra_count_font_pt))
            painter.drawText(rect.adjusted(0, 0, -p.text_margin, -p.text_margin), 
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, 
                           f"+{len(self.chamber.wafers) - 1}")

    def update_state(self, chamber: ChamberState) -> None:
        prev = self._last_status
        self.chamber = chamber
        if prev is not None and prev != chamber.status:
            ms = self._p.flash_ms
            self._flash_until = QDateTime.currentMSecsSinceEpoch() + ms
            QTimer.singleShot(ms + 50, self.update)
        self._last_status = chamber.status
        self.update()
