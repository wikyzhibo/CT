"""
晶圆 QGraphicsObject - 独立组件，支持悬浮特效与动画
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, Property, QPropertyAnimation, QEasingCurve, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, 
    QRadialGradient, QLinearGradient, QGradient
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsDropShadowEffect

from ..algorithm_interface import WaferState
from ..theme import ColorTheme
from ..ui_params import ui_params


class WaferItem(QGraphicsObject):
    """
    独立晶圆组件，支持：
    1. 3D 球体质感（Radial Gradient）
    2. 悬浮阴影（DropShadowEffect）
    3. Hover 动画（上浮 + 阴影加深）
    """

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self._p = ui_params.chamber_item
        self.wafer: WaferState | None = None
        
        # 动画属性
        self._hover_offset = 0.0
        self._shadow_blur = 10.0
        self._shadow_alpha = 80
        
        # 启用接受悬停事件
        self.setAcceptHoverEvents(True)
        
        # 阴影特效
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(self._shadow_blur)
        self.shadow_effect.setOffset(0, 2)
        self.shadow_effect.setColor(QColor(0, 0, 0, self._shadow_alpha))
        self.setGraphicsEffect(self.shadow_effect)
        
        # 动画对象
        self.anim_offset = QPropertyAnimation(self, b"hover_offset", self)
        self.anim_offset.setDuration(200)
        self.anim_offset.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        self.anim_blur = QPropertyAnimation(self, b"shadow_blur", self)
        self.anim_blur.setDuration(200)

    def boundingRect(self) -> QRectF:
        r = self._p.wafer_radius
        # 考虑 hover 移动范围，稍微扩大包围盒
        return QRectF(-r, -r - 10, r * 2, r * 2 + 10)

    def set_wafer(self, wafer: WaferState | None) -> None:
        if self.wafer != wafer:
            self.wafer = wafer
            self.update()

    def paint(self, painter: QPainter, option, widget) -> None:
        if not self.wafer:
            return

        p = self._p
        r = p.wafer_radius
        
        # 应用悬浮偏移 (绘制坐标系下移)
        painter.save()
        painter.translate(0, -self._hover_offset)

        # 1. 绘制晶圆本体 (3D 质感)
        self._draw_body(painter, r)

        # 2. 绘制进度环 (如果有)
        has_ring = self._draw_progress_ring(painter, r)

        # 3. 绘制文字
        self._draw_text(painter, r)
        
        painter.restore()

    def _draw_body(self, painter: QPainter, r: float) -> None:
        fill_rgb = self._get_wafer_color(self.wafer)
        base_color = self.theme.qcolor(fill_rgb)
        
        # 3D 径向渐变
        gradient = QRadialGradient(QPointF(-r * 0.3, -r * 0.3), r * 1.5)
        
        # 高光点
        lighter = base_color.lighter(130)
        # 暗部
        darker = base_color.darker(120)
        
        gradient.setColorAt(0.0, lighter)
        gradient.setColorAt(0.5, base_color)
        gradient.setColorAt(1.0, darker)
        
        # 描边（微弱发光感）
        stroke_color = lighter
        stroke_color.setAlpha(180)
        
        painter.setPen(QPen(stroke_color, 1.0))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(0, 0), r, r)

    def _draw_progress_ring(self, painter: QPainter, r: float) -> bool:
        wafer = self.wafer
        p = self._p
        has_ring = False
        
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            progress = min(1.0, max(0.0, wafer.stay_time / wafer.proc_time))
            if progress < 1.0: # 完成后一般环会填满，这里保留逻辑
                has_ring = True
            
            ring_pen = QPen(self.theme.qcolor(self.theme.accent_cyan), p.progress_ring_width)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap) # 圆头更精致
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            ring_r = r + p.progress_ring_offset
            ring_rect = QRectF(-ring_r, -ring_r, ring_r * 2, ring_r * 2)
            
            # 绘制背景轨（可选，增加层次）
            # bg_pen = QPen(QColor(255, 255, 255, 30), p.progress_ring_width)
            # painter.setPen(bg_pen)
            # painter.drawEllipse(ring_rect)
            
            painter.setPen(ring_pen)
            painter.drawArc(ring_rect, 90 * 16, -int(360 * 16 * progress))
            return True
            
        return False

    def _draw_text(self, painter: QPainter, r: float) -> None:
        wafer = self.wafer
        p = self._p
        
        fill_rgb = self._get_wafer_color(wafer)
        text_color = self._get_contrast_color(fill_rgb)
        
        main_font = QFont(p.font_family, p.wafer_font_pt, QFont.Weight.Bold)
        sub_font = QFont(p.font_family, p.wafer_id_font_pt, QFont.Weight.DemiBold)
        
        main_fm = QFontMetrics(main_font)
        sub_fm = QFontMetrics(sub_font)
        
        main_height = main_fm.height()
        sub_height = sub_fm.height()
        total_height = main_height + sub_height - 4
        
        start_y = -total_height / 2
        
        # 内容逻辑
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
            
        # 绘制主文本
        painter.setFont(main_font)
        painter.setPen(QColor(*text_color))
        main_rect = QRectF(-r, start_y, r * 2, main_height)
        painter.drawText(main_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, main_text)
        
        # 绘制次文本
        painter.setFont(sub_font)
        sub_color = tuple(max(0, c - 30) for c in text_color) if text_color == (255, 255, 255) else text_color
        painter.setPen(QColor(*sub_color))
        sub_rect = QRectF(-r, start_y + main_height - 4, r * 2, sub_height)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, sub_text)

    # --- 辅助方法 ---

    def _get_wafer_color(self, wafer: WaferState):
        if wafer.place_type == 1:
            proc = getattr(wafer, "proc_time", 0) or 0
            stay = int(wafer.stay_time)
            if wafer.time_to_scrap <= 0: return self.theme.danger
            if proc > 0 and stay >= proc + 20: return self.theme.danger
            if proc > 0 and stay >= proc: return self.theme.warning
            if wafer.time_to_scrap <= 5: return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:
            stay = int(wafer.stay_time)
            if stay >= 10: return self.theme.danger
            if stay >= 7: return self.theme.warning
            return self.theme.success
        return self.theme.secondary

    def _get_contrast_color(self, bg_color: tuple) -> tuple:
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
        return (20, 20, 25) if luminance > 0.5 else (255, 255, 255)

    def _has_progress_ring(self) -> bool:
        if not self.wafer: return False
        return self.wafer.place_type == 1 and getattr(self.wafer, "proc_time", 0) > 0

    # --- 动画事件 ---

    def hoverEnterEvent(self, event) -> None:
        # 有进度环时动画幅度小一点，避免太花
        is_strong = not self._has_progress_ring()
        
        target_offset = 4.0 if is_strong else 2.0
        target_blur = 20.0 if is_strong else 15.0
        
        self.anim_offset.stop()
        self.anim_offset.setStartValue(self._hover_offset)
        self.anim_offset.setEndValue(target_offset)
        self.anim_offset.start()
        
        self.anim_blur.stop()
        self.anim_blur.setStartValue(self._shadow_blur)
        self.anim_blur.setEndValue(target_blur)
        self.anim_blur.start()
        
    def hoverLeaveEvent(self, event) -> None:
        self.anim_offset.stop()
        self.anim_offset.setStartValue(self._hover_offset)
        self.anim_offset.setEndValue(0.0)
        self.anim_offset.start()
        
        self.anim_blur.stop()
        self.anim_blur.setStartValue(self._shadow_blur)
        self.anim_blur.setEndValue(10.0)
        self.anim_blur.start()

    # --- Properties ---
    
    def get_hover_offset(self) -> float:
        return self._hover_offset
        
    def set_hover_offset(self, value: float):
        self._hover_offset = value
        self.update() # 触发重绘
        
    def get_shadow_blur(self) -> float:
        return self._shadow_blur
        
    def set_shadow_blur(self, value: float):
        self._shadow_blur = value
        self.shadow_effect.setBlurRadius(value)
        # 阴影越散，适当降低透明度防止太黑
        # alpha = max(40, 80 - (value - 10) * 2)
        # self.shadow_effect.setColor(QColor(0, 0, 0, int(alpha)))
        
    hover_offset = Property(float, get_hover_offset, set_hover_offset)
    shadow_blur = Property(float, get_shadow_blur, set_shadow_blur)
