"""
腔室组件 (ChamberWidget)

用于在 QWidget 布局中展示单个腔室的状态卡片，包含：
- 腔室名称
- 状态 LED（idle/active/warning/danger）
- 晶圆显示（含 token_id，多晶圆时显示 +N）
- 网格背景

与 ChamberItem (QGraphicsItem) 共用 ui_params.chamber_item 可调参数，便于统一风格。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from ..algorithm_interface import ChamberState
from ..theme import ColorTheme
from ..ui_params import ui_params


class ChamberWidget(QWidget):
    """腔室显示组件：以卡片形式展示腔室名称、状态灯、晶圆信息。"""

    def __init__(self, chamber: ChamberState, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.chamber = chamber
        self.theme = theme
        self._p = ui_params.chamber_item  # 与 ChamberItem 共用可调参数
        self.setMinimumSize(self._p.w, self._p.h)

    def update_state(self, chamber: ChamberState) -> None:
        """更新腔室状态并触发重绘。"""
        self.chamber = chamber
        self.update()

    def paintEvent(self, event) -> None:
        """绘制顺序：背景（圆角+网格）→ 状态灯 → 标题 → 晶圆。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        p = self._p
        rect = self.rect().adjusted(p.inner_margin, p.inner_margin, -p.inner_margin, -p.inner_margin)
        self._draw_background(painter, rect)
        self._draw_status_led(painter, rect)
        self._draw_title(painter, rect)
        self._draw_wafer(painter, rect)

    def _draw_background(self, painter: QPainter, rect: QRectF) -> None:
        """绘制圆角矩形背景和网格纹理。"""
        p = self._p
        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(self.theme.bg_surface)))
        painter.drawRoundedRect(rect, p.corner_radius, p.corner_radius)

        # 网格纹理（工业风）
        painter.setPen(QPen(self.theme.qcolor(self.theme.border_muted)))
        for x in range(int(rect.left()), int(rect.right()), p.grid_step):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(int(rect.top()), int(rect.bottom()), p.grid_step):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

    def _draw_status_led(self, painter: QPainter, rect: QRectF) -> None:
        """右上角状态灯：danger→红 / warning→黄 / active→绿 / idle→灰。"""
        p = self._p
        status_color = self._get_status_color()
        led_rect = QRectF(rect.right() - p.led_size - 6, rect.top() + 6, p.led_size, p.led_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.theme.qcolor(status_color)))
        painter.drawEllipse(led_rect)

    def _draw_title(self, painter: QPainter, rect: QRectF) -> None:
        """左上角绘制腔室名称。"""
        p = self._p
        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.name_font_pt))
        painter.drawText(rect.adjusted(p.text_margin, 4, -p.text_margin, -4), Qt.AlignTop | Qt.AlignLeft, self.chamber.name)

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        """中心绘制主晶圆；多晶圆时右下角显示 +N。"""
        if not self.chamber.wafers:
            return

        p = self._p
        wafer = self.chamber.wafers[0]
        cx, cy = rect.center().x(), rect.center().y()
        r = p.wafer_radius

        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(self.theme.qcolor(self._get_wafer_color(wafer))))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        painter.setPen(self.theme.qcolor(self.theme.text_primary))
        painter.setFont(QFont(p.font_family, p.wafer_font_pt))
        painter.drawText(QRectF(cx - r, cy - r, r * 2, r * 2), Qt.AlignCenter, str(wafer.token_id))

        if len(self.chamber.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(p.font_family, p.extra_count_font_pt))
            painter.drawText(rect.adjusted(0, 0, -p.text_margin, -p.text_margin), Qt.AlignBottom | Qt.AlignRight, f"+{len(self.chamber.wafers) - 1}")

    def _get_status_color(self):
        """按 chamber.status 返回对应主题色。"""
        if self.chamber.status == "danger":
            return self.theme.danger
        if self.chamber.status == "warning":
            return self.theme.warning
        if self.chamber.status == "active":
            return self.theme.success
        return self.theme.text_muted

    def _get_wafer_color(self, wafer) -> tuple[int, int, int]:
        """
        按 place_type 与 time_to_scrap 返回晶圆颜色：
        - place_type=1（加工位）：报废≤0→红，≤5→黄，否则绿
        - place_type=2（运输位）：蓝
        - 其他：灰
        """
        if wafer.place_type == 1:
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:
            return self.theme.info
        return self.theme.secondary
