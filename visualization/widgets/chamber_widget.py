"""
腔室组件 (ChamberWidget)

用于在 QWidget 布局中展示单个腔室的状态卡片。

显示内容:
- 腔室名称: 左上角
- 状态 LED: 右上角，idle→灰 / active→绿 / warning→黄 / danger→红
- 晶圆: 中心圆形，内部显示剩余加工时间 + 编号 (#token_id)
- 网格背景: 工业风纹理

晶圆信息规则（对齐 viz.py）:
- 加工腔 (place_type=1): 主数字=剩余时间，次行=#编号；完成显示 +Ns 或 SCRAP
- 运输位 (place_type=2): 主数字=#编号，次行=停留时间 Ns

与 ChamberItem (QGraphicsItem) 共用 ui_params.chamber_item 可调参数，便于统一风格。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QFont, QColor
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
        self._p = ui_params.chamber_item  # 尺寸、字号等，与 ChamberItem 共用
        self.setMinimumSize(self._p.w, self._p.h)

    def update_state(self, chamber: ChamberState) -> None:
        """更新腔室状态并触发重绘。"""
        self.chamber = chamber
        self.update()

    def paintEvent(self, event) -> None:
        """Qt 绘制入口，自下而上叠绘：背景 → LED → 标题 → 晶圆。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        p = self._p
        rect = self.rect().adjusted(p.inner_margin, p.inner_margin, -p.inner_margin, -p.inner_margin)  # 内容区
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
        painter.drawText(
            rect.adjusted(p.text_margin, 4, -p.text_margin, -4),
            Qt.AlignTop | Qt.AlignLeft,
            self.chamber.name,
        )

    @staticmethod
    def _pick_contrast_text(fill_qcolor: QColor) -> QColor:
        """根据填充色选择黑/白高对比文本色（避免绿色晶圆上字也发绿/看不清）。"""
        r, g, b, _ = fill_qcolor.getRgb()
        # 简单亮度估计（0~255）
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return QColor(0, 0, 0) if lum > 150 else QColor(255, 255, 255)

    def _draw_wafer(self, painter: QPainter, rect: QRectF) -> None:
        """绘制主晶圆：圆形 + 两行文字；多晶圆时右下角 +N。"""
        if not self.chamber.wafers:
            return

        p = self._p
        wafer = self.chamber.wafers[0]
        cx, cy = rect.center().x(), rect.center().y()
        r = p.wafer_radius

        fill_rgb = self._get_wafer_color(wafer)
        fill_qc = self.theme.qcolor(fill_rgb)

        # 晶圆圆形
        painter.setPen(QPen(self.theme.qcolor(self.theme.border)))
        painter.setBrush(QBrush(fill_qc))
        wafer_rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        painter.drawEllipse(wafer_rect)

        # 两行文字分区：严格上/下 50%（再加少量 padding，防挤/防漂）
        pad_y = max(2, int(r * 0.08))
        #top_rect = QRectF(wafer_rect.left(), wafer_rect.top() + pad_y, wafer_rect.width(), wafer_rect.height() * 0.50 - pad_y)
        bot_rect = QRectF(wafer_rect.left(), wafer_rect.center().y(), wafer_rect.width(), wafer_rect.height() * 0.50 - pad_y)

        # ✅ 文本颜色：按晶圆颜色自动取高对比色
        text_qc = self._pick_contrast_text(fill_qc)
        painter.setPen(text_qc)

        # 加工腔：上剩余时间，下 #id（完成时下显示 SCRAP / +Ns）
        if wafer.place_type == 1 and getattr(wafer, "proc_time", 0) > 0:
            remaining = max(0, int(wafer.proc_time - wafer.stay_time))

            #painter.setFont(QFont(p.font_family, p.wafer_font_pt))
            #painter.drawText(top_rect, Qt.AlignHCenter | Qt.AlignVCenter, str(remaining))

            painter.setFont(QFont(p.font_family, p.wafer_id_font_pt))
            if remaining == 0:
                overtime = max(0, int(wafer.stay_time - wafer.proc_time))
                done_text = "SCRAP" if wafer.time_to_scrap <= 0 else f"+{overtime}s"
                painter.drawText(bot_rect, Qt.AlignHCenter | Qt.AlignVCenter, done_text)
            else:
                painter.drawText(bot_rect, Qt.AlignHCenter | Qt.AlignVCenter, f"{remaining}#{wafer.token_id}")

        else:
            # 运输位等非加工：上 #id，下 Ns
            stay_s = int(wafer.stay_time)

            painter.setFont(QFont(p.font_family, p.wafer_font_pt))
            painter.drawText(top_rect, Qt.AlignHCenter | Qt.AlignVCenter, f"#{wafer.token_id}")

            painter.setFont(QFont(p.font_family, p.wafer_id_font_pt))
            painter.drawText(bot_rect, Qt.AlignHCenter | Qt.AlignVCenter, f"{stay_s}s")

        # 多晶圆显示 +N（放在卡片右下角）
        if len(self.chamber.wafers) > 1:
            painter.setPen(self.theme.qcolor(self.theme.text_muted))
            painter.setFont(QFont(p.font_family, p.extra_count_font_pt))
            painter.drawText(
                rect.adjusted(0, 0, -p.text_margin, -p.text_margin),
                Qt.AlignBottom | Qt.AlignRight,
                f"+{len(self.chamber.wafers) - 1}",
            )

    def _get_status_color(self) -> tuple[int, int, int]:
        """按 chamber.status 返回对应主题色。"""
        if self.chamber.status == "danger":
            return self.theme.danger
        if self.chamber.status == "warning":
            return self.theme.warning
        if self.chamber.status == "active":
            return self.theme.success
        return self.theme.text_muted

    def _get_wafer_color(self, wafer) -> tuple[int, int, int]:
        """按 place_type 与停留时间返回晶圆填充色，逻辑对齐 viz.py。"""
        if wafer.place_type == 1:  # 加工腔
            proc = getattr(wafer, "proc_time", 0) or 0
            stay = int(wafer.stay_time)
            if wafer.time_to_scrap <= 0:
                return self.theme.danger
            if proc > 0 and stay >= proc + 15:
                return self.theme.danger
            if proc > 0 and stay >= proc:
                return self.theme.warning
            if wafer.time_to_scrap <= 5:
                return self.theme.warning
            return self.theme.success
        if wafer.place_type == 2:  # 运输位：stay<7 绿，7–10 黄，≥10 红
            stay = int(wafer.stay_time)
            if stay >= 10:
                return self.theme.danger
            if stay >= 7:
                return self.theme.warning
            return self.theme.success
        return self.theme.secondary
