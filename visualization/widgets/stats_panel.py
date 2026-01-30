"""
左侧统计面板
"""

from __future__ import annotations

from typing import Dict, Any, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QToolBox,
    QFrame,
)

from ..algorithm_interface import StateInfo
from ..theme import ColorTheme
from ..ui_params import ui_params


class StatsPanel(QScrollArea):
    """左侧统计面板 - 简化版本"""

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        p = ui_params.stats_panel
        #print("StatsPanelParams:", p)
        #print("label_font_pt:", p.label_font_pt, "kpi_font_pt:", p.kpi_font_pt)

        self.theme = theme
        p = ui_params.stats_panel

        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(p.layout_spacing)
        self.layout.setContentsMargins(*p.layout_margins)

        self.metrics_group = self._create_metrics_group()
        self.layout.addWidget(self.metrics_group)

        self.summary_toolbox = self._create_summary_toolbox()
        self.layout.addWidget(self.summary_toolbox)

        self.release_group = self._create_release_group()
        self.layout.addWidget(self.release_group)

        self.history_group = self._create_history_group()
        self.layout.addWidget(self.history_group)

        self.setWidget(content)
        self.setWidgetResizable(True)
        self.apply_params()

    def update_state(self, state: StateInfo, action_history: List[Dict[str, Any]], trend_data: Dict[str, List[float]] | None = None) -> None:
        self._update_metrics(state)
        self._update_summary(state)
        self._update_release_schedule(state)
        self._update_history(action_history)

    def update_reward(self, total_reward: float, detail: Dict[str, float]) -> None:
        self.reward_label.setText(f"REWARD: {total_reward:.2f}")
        # 指标全部展示：不过滤零值，按 key 排序便于对照
        detail_lines = [f"{k}: {v:+.2f}" for k, v in sorted(detail.items())]
        self.reward_detail.setText("\n".join(detail_lines) if detail_lines else "—")

    def update_step(self, step: int) -> None:
        self.step_label.setText(f"STEP: {step}")

    def _create_metrics_group(self) -> QGroupBox:
        p = ui_params.stats_panel
        group = QGroupBox("SYSTEM MONITOR")
        layout = QVBoxLayout(group)
        layout.setSpacing(p.group_spacing)

        label_font = QFont(p.font_family, p.label_font_pt)
        kpi_font = QFont(p.font_family, p.kpi_font_pt)

        self.time_label = QLabel("TIME: 0")
        self.time_label.setFont(kpi_font)
        self.time_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        layout.addWidget(self.time_label)

        progress_row = QWidget()
        progress_layout = QVBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_label = QLabel("PROGRESS: 0%")
        self.progress_label.setFont(label_font)
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(p.progress_bar_height)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_row)

        self.step_label = QLabel("STEP: 0")
        self.step_label.setFont(kpi_font)
        self.step_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        layout.addWidget(self.step_label)

        self.reward_label = QLabel("REWARD: 0.00")
        self.reward_label.setFont(kpi_font)
        self.reward_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        layout.addWidget(self.reward_label)

        self.reward_detail = QLabel("")
        self.reward_detail.setFont(QFont(p.font_family, p.reward_detail_font_pt))
        self.reward_detail.setAlignment(Qt.AlignTop)
        self.reward_detail.setWordWrap(True)
        layout.addWidget(self.reward_detail)

        return group

    def _create_summary_toolbox(self) -> QToolBox:
        p = ui_params.stats_panel
        toolbox = QToolBox()
        toolbox.setStyleSheet(f"QToolBox::tab {{ font-size: {p.toolbox_tab_font_pt}px; font-weight: bold; }}")
        self.system_summary_label = QLabel("")
        self.system_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.system_summary_label.setWordWrap(True)
        toolbox.addItem(self._wrap_in_frame(self.system_summary_label), "System")
        self.chambers_summary_label = QLabel("")
        self.chambers_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.chambers_summary_label.setWordWrap(True)
        toolbox.addItem(self._wrap_in_frame(self.chambers_summary_label), "Chambers")
        self.robots_summary_label = QLabel("")
        self.robots_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.robots_summary_label.setWordWrap(True)
        toolbox.addItem(self._wrap_in_frame(self.robots_summary_label), "Robots")
        return toolbox

    def _wrap_in_frame(self, widget: QWidget) -> QFrame:
        p = ui_params.stats_panel
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding, p.summary_frame_padding, p.summary_frame_padding)
        layout.addWidget(widget)
        return frame

    def _create_release_group(self) -> QGroupBox:
        p = ui_params.stats_panel
        group = QGroupBox("RELEASE TIME")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        self.release_text = QTextEdit()
        self.release_text.setReadOnly(True)
        self.release_text.setFont(QFont(p.font_family, p.release_font_pt))
        self.release_text.setMinimumHeight(p.release_min_height)
        layout.addWidget(self.release_text)
        return group

    def _create_history_group(self) -> QGroupBox:
        p = ui_params.stats_panel
        group = QGroupBox("HISTORY")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont(p.font_family, p.history_font_pt))
        self.history_text.setMinimumHeight(p.history_min_height)
        self.history_text.setStyleSheet("line-height: 1.3;")
        layout.addWidget(self.history_text)
        return group

    def _update_metrics(self, state: StateInfo) -> None:
        self.time_label.setText(f"TIME: {int(state.time)}")
        progress = 0
        if state.total_wafers > 0:
            progress = int((state.done_count / state.total_wafers) * 100)
        self.progress_label.setText(f"PROGRESS: {progress}% ({state.done_count}/{state.total_wafers} wafers)")
        self.progress_bar.setValue(progress)

    def _update_summary(self, state: StateInfo) -> None:
        # 指标全部展示：System 含 time / wafers / chambers，并展示 stats 中可文本化的项
        system_lines = [
            f"Time: {int(state.time)}",
            f"Wafers: {state.done_count}/{state.total_wafers}",
            f"Chambers: {len(state.chambers)}",
        ]
        for k, v in sorted(state.stats.items()):
            if k == "release_schedule":
                continue
            if isinstance(v, (int, float)):
                system_lines.append(f"{k}: {v}")
            elif isinstance(v, dict):
                system_lines.append(f"{k}: {len(v)} items")
            else:
                system_lines.append(f"{k}: {v}")
        self.system_summary_label.setText("\n".join(system_lines))
        chamber_lines = [f"{c.name}: {c.status}" for c in state.chambers]
        self.chambers_summary_label.setText("\n".join(chamber_lines) if chamber_lines else "—")
        robot_lines = [f"{name}: {'BUSY' if r.busy else 'IDLE'}" for name, r in state.robot_states.items()]
        self.robots_summary_label.setText("\n".join(robot_lines) if robot_lines else "—")

    def _update_release_schedule(self, state: StateInfo) -> None:
        schedule = state.stats.get("release_schedule", {})
        lines = []
        for place_name, items in schedule.items():
            if not items:
                continue
            pairs = ", ".join([f"{tid}->{rt}" for tid, rt in items])
            lines.append(f"{place_name}: {pairs}")
        self.release_text.setText("\n".join(lines))

    def _update_history(self, action_history: List[Dict[str, Any]]) -> None:
        n = ui_params.stats_panel.history_line_count
        lines = []
        for item in action_history[-n:]:
            lines.append(f"Step #{item['step']} - {item['action']} ({item['reward']:+.2f})")
        self.history_text.setText("\n".join(lines))

    def apply_params(self) -> None:
        p = ui_params.stats_panel

        # 主布局
        self.layout.setSpacing(p.layout_spacing)
        self.layout.setContentsMargins(*p.layout_margins)

        # 字体
        label_font = QFont(p.font_family, p.label_font_pt)
        kpi_font = QFont(p.font_family, p.kpi_font_pt)

        self.time_label.setFont(kpi_font)
        self.step_label.setFont(kpi_font)
        self.reward_label.setFont(kpi_font)
        self.progress_label.setFont(label_font)

        self.reward_detail.setFont(QFont(p.font_family, p.reward_detail_font_pt))

        # ToolBox tab：注意用 pt，不要用 px（否则高 DPI 下感觉不明显）
        self.summary_toolbox.setStyleSheet(
            f"QToolBox::tab {{ font-size: {p.toolbox_tab_font_pt}pt; font-weight: bold; }}"
        )

        # Summary 三页正文
        summary_font = QFont(p.font_family, p.summary_font_pt)
        self.system_summary_label.setFont(summary_font)
        self.chambers_summary_label.setFont(summary_font)
        self.robots_summary_label.setFont(summary_font)

        # RELEASE / HISTORY
        self.release_text.setFont(QFont(p.font_family, p.release_font_pt))
        self.release_text.setMinimumHeight(p.release_min_height)

        self.history_text.setFont(QFont(p.font_family, p.history_font_pt))
        self.history_text.setMinimumHeight(p.history_min_height)

        # Progress bar
        self.progress_bar.setMinimumHeight(p.progress_bar_height)

        # 触发布局重新计算
        self.metrics_group.adjustSize()
        self.summary_toolbox.adjustSize()
        self.release_group.adjustSize()
        self.history_group.adjustSize()
        self.widget().adjustSize()  # content
        self.updateGeometry()
        self.viewport().update()
