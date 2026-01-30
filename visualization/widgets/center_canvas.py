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
        self.setBackgroundBrush(QBrush(self.theme.qcolor(self.theme.bg_fog)))
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
        positions = {
            "PM3": (0, 1),
            "PM4": (0, 2),
            "PM2": (1, 0),
            "PM1": (2, 0),
            "PM5": (1, 3),
            "PM6": (2, 3),
            "LLC": (3, 1),
            "LLD": (3, 2),
            "PM8": (4, 0),
            "PM7": (5, 0),
            "PM9": (4, 3),
            "PM10": (5, 3),
            "LLA": (6, 1),
            "LLB": (6, 2),
        }

        for chamber in state.chambers:
            row, col = positions.get(chamber.name, (0, 0))
            item = ChamberItem(chamber, self.theme)
            self.chambers[chamber.name] = item
            self.scene.addItem(item)
            item.setPos(col * cw + (cw - ch_item.w) / 2, row * ch + (ch - ch_item.h) / 2)

        for name, robot in state.robot_states.items():
            item = RobotItem(robot, self.theme)
            self.robots[name] = item
            self.scene.addItem(item)
            if name == "TM3":
                cx, cy = 1.5 * cw, 1.5 * ch
                item.setPos(cx - r_item.w / 2, cy - r_item.h / 2)
            elif name == "TM2":
                cx, cy = 1.5 * cw, 4.5 * ch
                item.setPos(cx - r_item.w / 2, cy - r_item.h / 2)

        self.scene.setSceneRect(0, 0, 4 * cw, 7 * ch)
