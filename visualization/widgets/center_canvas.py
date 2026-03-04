"""
中心画布 - QGraphicsView 拓扑布局
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget

from ..algorithm_interface import ChamberState, RobotState, StateInfo
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
        self._display_to_state: Dict[str, str] = {}
        self._robot_chambers: Dict[str, list[str]] = {}
        self._latest_state: StateInfo | None = None
        self._device_mode = "cascade"

    def set_device_mode(self, mode: str) -> None:
        """切换设备布局（仅 UI 占位）。"""
        if mode not in {"cascade", "single"}:
            return
        self._device_mode = mode
        if self._latest_state is not None:
            self._build_layout(self._latest_state)

    def update_state(self, state: StateInfo) -> None:
        self._latest_state = state
        if not self.chambers:
            self._build_layout(state)

        state_chambers = {c.name: c for c in state.chambers}
        for display_name, item in self.chambers.items():
            source_name = self._display_to_state.get(display_name, display_name)
            source = state_chambers.get(source_name)
            item.update_state(self._clone_chamber_state(display_name, source))

        for name, item in self.robots.items():
            robot = state.robot_states.get(name, RobotState(name=name, busy=False, wafers=[]))
            item.update_state(robot)

    def _build_layout(self, state: StateInfo) -> None:
        self.scene.clear()
        self.chambers.clear()
        self.robots.clear()

        cc = ui_params.center_canvas
        ch_item = ui_params.chamber_item
        r_item = ui_params.robot_item
        cw, ch = cc.cell_w, cc.cell_h

        positions, self._display_to_state, self._robot_chambers = self._layout_config_by_mode()
        state_chambers = {c.name: c for c in state.chambers}

        for display_name, (row, col) in positions.items():
            source_name = self._display_to_state.get(display_name, display_name)
            source = state_chambers.get(source_name)
            chamber_state = self._clone_chamber_state(display_name, source)
            item = ChamberItem(chamber_state, self.theme)
            self.chambers[display_name] = item
            self.scene.addItem(item)
            item.setPos(col * cw + (cw - ch_item.w) / 2, row * ch + (ch - ch_item.h) / 2)

        for name, chamber_names in self._robot_chambers.items():
            robot = state.robot_states.get(name, RobotState(name=name, busy=False, wafers=[]))
            item = RobotItem(robot, self.theme)
            self.robots[name] = item
            self.scene.addItem(item)
            if chamber_names:
                cx, cy = self._center_of_chambers(positions, chamber_names, cw, ch)
                item.setPos(cx - r_item.w / 2, cy - r_item.h / 2)

        row_count = max(pos[0] for pos in positions.values()) + 1 if positions else 1
        col_count = max(pos[1] for pos in positions.values()) + 1 if positions else 1
        self.scene.setSceneRect(0, 0, col_count * cw, row_count * ch)

    def _layout_config_by_mode(self) -> tuple[dict, dict, dict]:
        if self._device_mode == "single":
            # 单设备：LP / LP_done + 6 个处理腔室，保持与级联设备一致的网格风格
            positions = {
                "PM3": (0, 1), "PM4": (0, 2),
                "PM2": (1, 0), "PM1": (2, 0),
                "PM5": (1, 3), "PM6": (2, 3),
                "LP": (3, 1), "LP_done": (3, 2),
            }
            display_to_state = {name: name for name in positions}
            robot_chambers = {
                # 单设备仅保留一个机械手
                "TM2": ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LP", "LP_done"],
            }
            return positions, display_to_state, robot_chambers

        positions = {
            "PM3": (0, 1), "PM4": (0, 2),
            "PM2": (1, 0), "PM1": (2, 0),
            "PM5": (1, 3), "PM6": (2, 3),
            "LLC": (3, 1), "LLD": (3, 2),
            "PM8": (4, 0), "PM7": (5, 0),
            "PM9": (4, 3), "PM10": (5, 3),
            "LLA": (6, 1), "LLB": (6, 2),
        }
        display_to_state = {name: name for name in positions}
        robot_chambers = {
            "TM2": ["LLA", "LLB", "PM7", "PM8", "PM9", "PM10", "LLC", "LLD"],
            "TM3": ["LLC", "LLD", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"],
        }
        return positions, display_to_state, robot_chambers

    @staticmethod
    def _clone_chamber_state(display_name: str, source: ChamberState | None) -> ChamberState:
        if source is None:
            return ChamberState(
                name=display_name,
                place_idx=-1,
                capacity=1,
                wafers=[],
                proc_time=0.0,
                status="idle",
                chamber_type="processing",
            )
        return ChamberState(
            name=display_name,
            place_idx=source.place_idx,
            capacity=source.capacity,
            wafers=source.wafers,
            proc_time=source.proc_time,
            status=source.status,
            chamber_type=source.chamber_type,
        )

    def _center_of_chambers(self, positions: dict, names: list, cw: float, ch: float) -> tuple[float, float]:
        """计算若干腔室网格中心的几何中心 (px)。"""
        xs, ys = [], []
        for name in names:
            row, col = positions.get(name, (0, 0))
            xs.append((col + 0.5) * cw)
            ys.append((row + 0.5) * ch)
        return sum(xs) / len(xs), sum(ys) / len(ys)
