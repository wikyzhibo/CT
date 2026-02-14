"""
中心画布 - QGraphicsView 拓扑布局
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget

from ..algorithm_interface import StateInfo
from ..theme import ColorTheme
from ..ui_params import ui_params
from ..graphics import ChamberItem, RobotItem


class CenterCanvas(QGraphicsView):
    """中心区域：QGraphicsView + 腔室/机械手 Item，7×4 网格布局"""

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 使用更深的背景色以增强与腔室卡片的对比
        self.setBackgroundBrush(QBrush(self.theme.qcolor(self.theme.bg_deepest)))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.chambers: Dict[str, ChamberItem] = {}
        self.robots: Dict[str, RobotItem] = {}

    def update_state(self, state: StateInfo) -> None:
        if not self.chambers:
            self._build_layout(state)

        for chamber in state.chambers:
            item = self.chambers.get(chamber.name)
            if item:
                item.update_state(chamber)

        for name, robot in state.robot_states.items():
            item = self.robots.get(name)
            if item:
                item.update_state(robot)

    def _build_layout(self, state: StateInfo) -> None:
        cc = ui_params.center_canvas
        ch_item = ui_params.chamber_item
        r_item = ui_params.robot_item
        cw, ch = cc.cell_w, cc.cell_h

        # 腔室网格位置 (row, col)，7 行 × 4 列
        positions = {
            "PM3": (0, 1), "PM4": (0, 2),
            "PM2": (1, 0), "PM1": (2, 0),
            "PM5": (1, 3), "PM6": (2, 3),
            "LLC": (3, 1), "LLD": (3, 2),
            "PM8": (4, 0), "PM7": (5, 0),
            "PM9": (4, 3), "PM10": (5, 3),
            "LLA": (6, 1), "LLB": (6, 2),
        }

        for chamber in state.chambers:
            row, col = positions.get(chamber.name, (0, 0))
            item = ChamberItem(chamber, self.theme)
            self.chambers[chamber.name] = item
            self.scene.addItem(item)
            item.setPos(col * cw + (cw - ch_item.w) / 2, row * ch + (ch - ch_item.h) / 2)

        # 机械手置于所管辖腔室的几何中心
        TM2_CHAMBERS = ["LLA", "LLB", "PM7", "PM8", "PM9", "PM10", "LLC", "LLD"]
        TM3_CHAMBERS = ["LLC", "LLD", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
        robot_chambers = {"TM2": TM2_CHAMBERS, "TM3": TM3_CHAMBERS}

        for name, robot in state.robot_states.items():
            item = RobotItem(robot, self.theme)
            self.robots[name] = item
            self.scene.addItem(item)
            chamber_names = robot_chambers.get(name, [])
            if chamber_names:
                cx, cy = self._center_of_chambers(positions, chamber_names, cw, ch)
                item.setPos(cx - r_item.w / 2, cy - r_item.h / 2)

        self.scene.setSceneRect(0, 0, 4 * cw, 7 * ch)

    def _center_of_chambers(self, positions: dict, names: list, cw: float, ch: float) -> tuple[float, float]:
        """计算若干腔室网格中心的几何中心 (px)。"""
        xs, ys = [], []
        for name in names:
            row, col = positions.get(name, (0, 0))
            xs.append((col + 0.5) * cw)
            ys.append((row + 0.5) * ch)
        return sum(xs) / len(xs), sum(ys) / len(ys)
