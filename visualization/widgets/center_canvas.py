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
        self._robot_display_to_state: Dict[str, str] = {}
        self._latest_state: StateInfo | None = None
        self._device_mode = "cascade"
        self._robot_capacity = 1

    def set_device_mode(self, mode: str) -> None:
        """切换设备布局（仅 UI 占位）。"""
        if mode not in {"cascade", "single"}:
            return
        self._device_mode = mode
        if self._latest_state is not None:
            self._build_layout(self._latest_state)

    def set_robot_capacity(self, capacity: int) -> None:
        """设置机械手容量（1/2），用于 UI 布局展示。"""
        self._robot_capacity = 2 if int(capacity) == 2 else 1
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
            chamber_state = self._clone_chamber_state(display_name, source)
            item.update_state(chamber_state)

        for name, item in self.robots.items():
            source_name = self._robot_display_to_state.get(name, name)
            source_robot = state.robot_states.get(source_name, RobotState(name=source_name, busy=False, wafers=[]))
            robot = RobotState(name=name, busy=source_robot.busy, wafers=source_robot.wafers)
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

        min_x = float("inf")
        min_y = float("inf")
        max_x = 0.0
        max_y = 0.0

        for display_name, (row, col) in positions.items():
            source_name = self._display_to_state.get(display_name, display_name)
            source = state_chambers.get(source_name)
            chamber_state = self._clone_chamber_state(display_name, source)
            item = ChamberItem(chamber_state, self.theme)
            self.chambers[display_name] = item
            self.scene.addItem(item)
            x, y = self._position_for_cell(row, col, cw, ch, ch_item.w, ch_item.h)
            item.setPos(x, y)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + ch_item.w)
            max_y = max(max_y, y + ch_item.h)

        robot_centers = self._robot_centers(positions, cw, ch)
        for name, chamber_names in self._robot_chambers.items():
            source_name = self._robot_display_to_state.get(name, name)
            source_robot = state.robot_states.get(source_name, RobotState(name=source_name, busy=False, wafers=[]))
            robot = RobotState(name=name, busy=source_robot.busy, wafers=source_robot.wafers)
            item = RobotItem(robot, self.theme)
            self.robots[name] = item
            self.scene.addItem(item)
            if chamber_names and name in robot_centers:
                cx, cy = robot_centers[name]
                item.setPos(cx - r_item.w / 2, cy - r_item.h / 2)
                min_x = min(min_x, cx - r_item.w / 2)
                min_y = min(min_y, cy - r_item.h / 2)
                max_x = max(max_x, cx + r_item.w / 2)
                max_y = max(max_y, cy + r_item.h / 2)

        padding = 24.0
        if min_x == float("inf"):
            min_x, min_y, max_x, max_y = 0.0, 0.0, float(cw), float(ch)
        self.scene.setSceneRect(min_x - padding, min_y - padding, (max_x - min_x) + padding * 2, (max_y - min_y) + padding * 2)

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
            if self._robot_capacity == 2:
                robot_chambers = {
                    "ARM1": ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LP", "LP_done"],
                    "ARM2": ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LP", "LP_done"],
                }
                self._robot_display_to_state = {"ARM1": "TM2", "ARM2": "TM3"}
            else:
                robot_chambers = {"ARM": ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LP", "LP_done"]}
                self._robot_display_to_state = {"ARM": "TM2"}
            # 容量=2 时在中心横向展示双机械手，ARM2 当前可能为 UI 占位
            if self._robot_capacity == 2:
                return positions, display_to_state, robot_chambers
            return positions, display_to_state, robot_chambers
        elif self._device_mode == "cascade":
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
            if self._robot_capacity == 2:
                robot_chambers = {
                    "TM2 ARM1": ["PM7", "PM8", "PM9", "PM10", "LLC", "LLD", "LLA", "LLB"],
                    "TM2 ARM2": ["PM7", "PM8", "PM9", "PM10", "LLC", "LLD", "LLA", "LLB"],
                    "TM3 ARM1": ["LLC", "LLD", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"],
                    "TM3 ARM2": ["LLC", "LLD", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"],
                }
                self._robot_display_to_state = {
                    "TM2 ARM1": "TM2",
                    "TM2 ARM2": "TM2",
                    "TM3 ARM1": "TM3",
                    "TM3 ARM2": "TM3",
                }
            else:
                robot_chambers = {
                    "TM2 ARM": ["PM7", "PM8", "PM9", "PM10", "LLC", "LLD", "LLA", "LLB"],
                    "TM3 ARM": ["LLC", "LLD", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"],
                }
                self._robot_display_to_state = {"TM2 ARM": "TM2", "TM3 ARM": "TM3"}
            return positions, display_to_state, robot_chambers

    @staticmethod
    def _clone_chamber_state(display_name: str, source: ChamberState | None) -> ChamberState:
        if source is None:
            is_disabled_placeholder = display_name in {"PM5"}
            return ChamberState(
                name=display_name,
                place_idx=-1,
                capacity=1,
                wafers=[],
                proc_time=0.0,
                status="disabled" if is_disabled_placeholder else "idle",
                chamber_type="disabled" if is_disabled_placeholder else "processing",
                cleaning_wafer_countdown=-1,
            )
        return ChamberState(
            name=display_name,
            place_idx=source.place_idx,
            capacity=source.capacity,
            wafers=source.wafers,
            proc_time=source.proc_time,
            status=source.status,
            chamber_type=source.chamber_type,
            cleaning_remaining=source.cleaning_remaining,
            inbound_blocked=source.inbound_blocked,
            cleaning_wafer_countdown=getattr(source, "cleaning_wafer_countdown", -1),
        )

    def _center_of_chambers(self, positions: dict, names: list, cw: float, ch: float) -> tuple[float, float]:
        """计算若干腔室网格中心的几何中心 (px)。"""
        xs, ys = [], []
        alias = {"LP": "LLA", "LP1": "LLA", "LP2": "LLA", "LP_done": "LLB"}
        for name in names:
            key = name
            if key not in positions:
                key = alias.get(name, name)
            if key not in positions:
                continue
            row, col = positions[key]
            cell_x, cell_y = self._position_for_cell(row, col, cw, ch, ui_params.chamber_item.w, ui_params.chamber_item.h)
            visual_x = cell_x + ui_params.chamber_item.w / 2
            visual_y = cell_y + ui_params.chamber_item.h / 2
            xs.append(visual_x)
            ys.append(visual_y)
        if not xs:
            for row, col in positions.values():
                cell_x, cell_y = self._position_for_cell(row, col, cw, ch, ui_params.chamber_item.w, ui_params.chamber_item.h)
                xs.append(cell_x + ui_params.chamber_item.w / 2)
                ys.append(cell_y + ui_params.chamber_item.h / 2)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _position_for_cell(self, row: int, col: int, cw: float, ch: float, item_w: float, item_h: float) -> tuple[float, float]:
        x = col * cw + (cw - item_w) / 2
        y = row * ch + (ch - item_h) / 2
        if self._device_mode == "single" and self._robot_capacity == 2:
            # 双机械手时轻微外扩四周腔室，避免中心拥挤
            x_shift_map = {0: -28.0, 1: -12.0, 2: 12.0, 3: 28.0}
            y_shift_map = {0: -8.0, 1: -2.0, 2: 2.0, 3: 8.0}
            x += x_shift_map.get(col, 0.0)
            y += y_shift_map.get(row, 0.0)
        elif self._device_mode == "cascade" and self._robot_capacity == 2:
            # 级联双臂时也做外扩，给中心四机械手留出空间
            x_shift_map = {0: -24.0, 1: -10.0, 2: 10.0, 3: 24.0}
            x += x_shift_map.get(col, 0.0)
            y += (float(row) - 3.0) * 2.5
        return x, y

    def _robot_centers(self, positions: dict, cw: float, ch: float) -> dict[str, tuple[float, float]]:
        centers: dict[str, tuple[float, float]] = {}
        if not self._robot_chambers:
            return centers

        if self._device_mode == "single" and self._robot_capacity == 2 and {"ARM1", "ARM2"}.issubset(self._robot_chambers.keys()):
            center_names = self._robot_chambers["ARM1"]
            cx, cy = self._center_of_chambers(positions, center_names, cw, ch)
            gap = 148.0
            centers["ARM1"] = (cx - gap / 2, cy)
            centers["ARM2"] = (cx + gap / 2, cy)
            return centers

        if (
            self._device_mode == "cascade"
            and self._robot_capacity == 2
            and {"TM2 ARM1", "TM2 ARM2", "TM3 ARM1", "TM3 ARM2"}.issubset(self._robot_chambers.keys())
        ):
            tm2_cx, tm2_cy = self._center_of_chambers(positions, self._robot_chambers["TM2 ARM1"], cw, ch)
            tm3_cx, tm3_cy = self._center_of_chambers(positions, self._robot_chambers["TM3 ARM1"], cw, ch)
            # 横排时使用基于卡片宽度的安全间距，避免双臂互相重叠
            arm_gap = float(ui_params.robot_item.w + 15)
            centers["TM2 ARM1"] = (tm2_cx - arm_gap / 2, tm2_cy)
            centers["TM2 ARM2"] = (tm2_cx + arm_gap / 2, tm2_cy)
            centers["TM3 ARM1"] = (tm3_cx - arm_gap / 2, tm3_cy)
            centers["TM3 ARM2"] = (tm3_cx + arm_gap / 2, tm3_cy)
            return centers

        if self._device_mode == "cascade" and self._robot_capacity == 1 and {"TM2 ARM", "TM3 ARM"}.issubset(self._robot_chambers.keys()):
            tm2_cx, tm2_cy = self._center_of_chambers(positions, self._robot_chambers["TM2 ARM"], cw, ch)
            tm3_cx, tm3_cy = self._center_of_chambers(positions, self._robot_chambers["TM3 ARM"], cw, ch)
            centers["TM2 ARM"] = (tm2_cx, tm2_cy)
            centers["TM3 ARM"] = (tm3_cx, tm3_cy)
            return centers

        for name, chamber_names in self._robot_chambers.items():
            if not chamber_names:
                continue
            centers[name] = self._center_of_chambers(positions, chamber_names, cw, ch)
        return centers
