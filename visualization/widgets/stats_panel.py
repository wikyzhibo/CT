"""
左侧统计面板 (StatsPanel)

展示系统运行时的 KPI、进度、奖励、腔室/机械手状态、释放计划和动作历史。

区块结构（自上而下）:
- SYSTEM MONITOR: TIME / PROGRESS（进度条）/ STEP / REWARD / 奖励明细
- ToolBox: System / Chambers / Robots 三页可折叠摘要
- RELEASE TIME: 各库所的 token_id→release_time 映射
- HISTORY: 最近 N 步动作及奖励

布局与字号由 ui_params.stats_panel 控制。
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
    """左侧统计面板：KPI、进度、摘要、RELEASE TIME、HISTORY。"""

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        p = ui_params.stats_panel

        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(p.layout_spacing)
        self.layout.setContentsMargins(*p.layout_margins)

        self.metrics_group = self._create_metrics_group()  # TIME / PROGRESS / STEP / REWARD
        self.layout.addWidget(self.metrics_group)

        self.summary_toolbox = self._create_summary_toolbox()  # System / Chambers / Robots
        self.layout.addWidget(self.summary_toolbox)

        self.release_group = self._create_release_group()  # 释放计划
        self.layout.addWidget(self.release_group)

        self.history_group = self._create_history_group()  # 动作历史
        self.layout.addWidget(self.history_group)

        self.setWidget(content)
        self.setWidgetResizable(True)
        self.apply_params()

    def update_state(self, state: StateInfo, action_history: List[Dict[str, Any]], trend_data: Dict[str, List[float]] | None = None) -> None:
        """全量刷新：KPI、摘要、释放计划、历史。"""
        self._update_metrics(state)
        self._update_summary(state)
        self._update_release_schedule(state)
        self._update_history(action_history)

    def update_reward(self, total_reward: float, detail: Dict[str, float]) -> None:
        """单独刷新奖励：总奖励 + 明细（按 key 排序，不过滤零值）。"""
        self.reward_label.setText(f"REWARD: {total_reward:.2f}")
        detail_lines = [f"{k}: {v:+.2f}" for k, v in sorted(detail.items())]
        self.reward_detail.setText("\n".join(detail_lines) if detail_lines else "—")

    def update_step(self, step: int) -> None:
        """单独刷新步数。"""
        self.step_label.setText(f"STEP: {step}")

    def _create_metrics_group(self) -> QGroupBox:
        """创建 SYSTEM MONITOR 区块：TIME、进度条、STEP、REWARD、奖励明细。"""
        p = ui_params.stats_panel
        group = QGroupBox("SYSTEM MONITOR")
        layout = QVBoxLayout(group)
        layout.setSpacing(p.group_spacing)


        label_font = QFont(p.font_family, p.label_font_pt)
        kpi_font = QFont(p.font_family, p.kpi_font_pt)

        self.time_label = QLabel("TIME: 0")
        self.time_label.setFont(kpi_font)
        self.time_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.time_label.setObjectName("KpiLabel")
        layout.addWidget(self.time_label)
        print("TIME font:", self.time_label.font().pointSize(), self.time_label.font().family())

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
        self.progress_label.setObjectName("BigLabel")
        layout.addWidget(progress_row)

        self.step_label = QLabel("STEP: 0")
        self.step_label.setFont(kpi_font)
        self.step_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.step_label.setObjectName("KpiLabel")
        layout.addWidget(self.step_label)

        self.reward_label = QLabel("REWARD: 0.00")
        self.reward_label.setFont(kpi_font)
        self.reward_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.reward_label.setObjectName("KpiLabel")
        layout.addWidget(self.reward_label)

        self.reward_detail = QLabel("")
        self.reward_detail.setFont(QFont(p.font_family, p.reward_detail_font_pt))
        self.reward_detail.setAlignment(Qt.AlignTop)
        self.reward_detail.setWordWrap(True)
        self.reward_detail.setObjectName("DetailLabel")
        layout.addWidget(self.reward_detail)

        return group

    def _create_summary_toolbox(self) -> QToolBox:
        """创建 ToolBox：System / Chambers / Robots 三页可折叠摘要。"""
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
        """将控件包入带内边距的 Frame，用于 ToolBox 每页内容。"""
        p = ui_params.stats_panel
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding, p.summary_frame_padding, p.summary_frame_padding)
        layout.addWidget(widget)
        return frame

    def _create_release_group(self) -> QGroupBox:
        """创建 RELEASE TIME 区块：只读文本框，展示各库所 token_id→release_time。"""
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
        """创建 HISTORY 区块：只读文本框，展示最近 N 步动作及奖励。"""
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
        """更新 TIME、PROGRESS（百分比+进度条）、完成数/总数。"""
        self.time_label.setText(f"TIME: {int(state.time)}")
        progress = 0
        if state.total_wafers > 0:
            progress = int((state.done_count / state.total_wafers) * 100)
        self.progress_label.setText(f"PROGRESS: {progress}% ({state.done_count}/{state.total_wafers} wafers)")
        self.progress_bar.setValue(progress)

    def _update_summary(self, state: StateInfo) -> None:
        """更新 ToolBox 三页：System（含 stats）、Chambers、Robots。"""
        system_lines = [
            f"Time: {int(state.time)}",
            f"Wafers: {state.done_count}/{state.total_wafers}",
            f"Chambers: {len(state.chambers)}",
        ]
        for k, v in sorted(state.stats.items()):
            if k == "release_schedule":  # 单独在 RELEASE TIME 区块展示
                continue
            if isinstance(v, (int, float)):
                system_lines.append(f"{k}: {v}")
            elif isinstance(v, dict):
                system_lines.append(f"{k}: {len(v)} items")
            else:
                system_lines.append(f"{k}: {v}")
        self.system_summary_label.setText("\n".join(system_lines))
        chamber_lines = [f"{c.name}: {c.status}" for c in state.chambers]  # 腔室名: 状态
        self.chambers_summary_label.setText("\n".join(chamber_lines) if chamber_lines else "—")
        robot_lines = [f"{name}: {'BUSY' if r.busy else 'IDLE'}" for name, r in state.robot_states.items()]  # 机械手: 状态
        self.robots_summary_label.setText("\n".join(robot_lines) if robot_lines else "—")

    def _update_release_schedule(self, state: StateInfo) -> None:
        """从 state.stats['release_schedule'] 解析，格式 place_name: tid->rt, tid->rt。"""
        schedule = state.stats.get("release_schedule", {})
        lines = []
        for place_name, items in schedule.items():
            if not items:
                continue
            pairs = ", ".join([f"{tid}->{rt}" for tid, rt in items])  # token_id -> release_time
            lines.append(f"{place_name}: {pairs}")
        self.release_text.setText("\n".join(lines))

    def _update_history(self, action_history: List[Dict[str, Any]]) -> None:
        """取最近 N 条历史，格式 Step #N - action (reward)。"""
        n = ui_params.stats_panel.history_line_count
        lines = []
        for item in action_history[-n:]:
            lines.append(f"Step #{item['step']} - {item['action']} ({item['reward']:+.2f})")
        self.history_text.setText("\n".join(lines))

    def apply_params(self) -> None:
        """根据 ui_params 重新应用字号、间距、最小高度，并触发布局重算。"""
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

        # ToolBox 标签：用 pt 保证高 DPI 下缩放一致
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

        # 触发布局重算，确保字号/边距变化后正确渲染
        self.metrics_group.adjustSize()
        self.summary_toolbox.adjustSize()
        self.release_group.adjustSize()
        self.history_group.adjustSize()
        self.widget().adjustSize()  # content
        self.updateGeometry()
        self.viewport().update()
