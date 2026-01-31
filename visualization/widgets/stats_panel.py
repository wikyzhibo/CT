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
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
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
from .status_badge import StatusBadge, StatusDot


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
        """单独刷新奖励：总奖励 + 明细（按 key 排序，带颜色编码）。"""
        self.reward_label.setText(f"REWARD: {total_reward:.2f}")
        
        # 使用颜色编码的奖励明细
        detail_lines = []
        for k, v in sorted(detail.items()):
            color = self.theme.success if v >= 0 else self.theme.danger
            # 使用 HTML 富文本进行颜色编码
            detail_lines.append(f'<span style="color: rgb{color};">{k}: {v:+.2f}</span>')
        
        if detail_lines:
            self.reward_detail.setText("<br>".join(detail_lines))
        else:
            self.reward_detail.setText("—")

    def update_step(self, step: int) -> None:
        """单独刷新步数。"""
        self.step_label.setText(f"STEP: {step}")

    def _create_metrics_group(self) -> QGroupBox:
        """创建 SYSTEM MONITOR 区块：TIME、进度条、STEP、REWARD、奖励明细。"""
        p = ui_params.stats_panel
        group = QGroupBox("📊 SYSTEM MONITOR")
        # 直接设置标题样式以确保字号生效
        group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 18pt;
                font-weight: 700;
            }}
            QGroupBox::title {{
                font-size: 18pt;
                font-weight: 700;
            }}
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(p.group_spacing)

        label_font = QFont(p.font_family, p.label_font_pt)
        kpi_font = QFont(p.font_family, p.kpi_font_pt, QFont.Bold)

        # TIME KPI
        self.time_label = QLabel("TIME: 0")
        self.time_label.setFont(kpi_font)
        self.time_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.time_label.setObjectName("KpiLabel")
        layout.addWidget(self.time_label)

        # 添加间距
        layout.addSpacing(p.card_spacing)

        # PROGRESS 区块（带颜色编码）
        progress_row = QWidget()
        progress_layout = QVBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        
        self.progress_label = QLabel("PROGRESS: 0%")
        self.progress_label.setFont(label_font)
        self.progress_label.setObjectName("BigLabel")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(p.progress_bar_height)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_row)
        layout.addSpacing(p.card_spacing)

        # STEP KPI
        self.step_label = QLabel("STEP: 0")
        self.step_label.setFont(kpi_font)
        self.step_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.step_label.setObjectName("KpiLabel")
        layout.addWidget(self.step_label)
        layout.addSpacing(p.card_spacing)

        # REWARD KPI
        self.reward_label = QLabel("REWARD: 0.00")
        self.reward_label.setFont(kpi_font)
        self.reward_label.setStyleSheet(f"color: rgb{self.theme.text_kpi};")
        self.reward_label.setObjectName("KpiLabel")
        layout.addWidget(self.reward_label)

        # REWARD 明细（支持 HTML 富文本以实现颜色编码）
        self.reward_detail = QLabel("")
        self.reward_detail.setFont(QFont(p.font_family, p.reward_detail_font_pt))
        self.reward_detail.setAlignment(Qt.AlignTop)
        self.reward_detail.setWordWrap(True)
        self.reward_detail.setTextFormat(Qt.RichText)  # 启用富文本
        self.reward_detail.setObjectName("DetailLabel")
        layout.addWidget(self.reward_detail)

        return group

    def _create_summary_toolbox(self) -> QWidget:
        """创建摘要区域：System / Chambers / Robots 三个展开的区块。"""
        p = ui_params.stats_panel
        
        # 使用 QWidget 容器而不是 QToolBox
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(p.section_spacing)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # System 摘要
        system_group = QGroupBox("🖥️ System")
        system_group.setStyleSheet("QGroupBox { font-size: 18pt; font-weight: 700; } QGroupBox::title { font-size: 18pt; font-weight: 700; }")
        system_layout = QVBoxLayout(system_group)
        system_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding, 
                                        p.summary_frame_padding, p.summary_frame_padding)
        self.system_summary_label = QLabel("")
        self.system_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.system_summary_label.setWordWrap(True)
        system_layout.addWidget(self.system_summary_label)
        layout.addWidget(system_group)
        
        # Chambers 摘要
        chambers_group = QGroupBox("⚙️ Chambers")
        chambers_group.setStyleSheet("QGroupBox { font-size: 18pt; font-weight: 700; } QGroupBox::title { font-size: 18pt; font-weight: 700; }")
        chambers_layout = QVBoxLayout(chambers_group)
        chambers_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding,
                                          p.summary_frame_padding, p.summary_frame_padding)
        self.chambers_summary_label = QLabel("")
        self.chambers_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.chambers_summary_label.setWordWrap(True)
        chambers_layout.addWidget(self.chambers_summary_label)
        layout.addWidget(chambers_group)
        
        # Robots 摘要
        robots_group = QGroupBox("🤖 Robots")
        robots_group.setStyleSheet("QGroupBox { font-size: 18pt; font-weight: 700; } QGroupBox::title { font-size: 18pt; font-weight: 700; }")
        robots_layout = QVBoxLayout(robots_group)
        robots_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding,
                                        p.summary_frame_padding, p.summary_frame_padding)
        self.robots_summary_label = QLabel("")
        self.robots_summary_label.setFont(QFont(p.font_family, p.summary_font_pt))
        self.robots_summary_label.setWordWrap(True)
        robots_layout.addWidget(self.robots_summary_label)
        layout.addWidget(robots_group)
        
        return container

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
        group.setStyleSheet("QGroupBox { font-size: 18pt; font-weight: 700; } QGroupBox::title { font-size: 18pt; font-weight: 700; }")
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
        group.setStyleSheet("QGroupBox { font-size: 18pt; font-weight: 700; } QGroupBox::title { font-size: 18pt; font-weight: 700; }")
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
        """更新 TIME、PROGRESS（百分比+进度条，带颜色编码）、完成数/总数。"""
        self.time_label.setText(f"TIME: {int(state.time)}")
        progress = 0
        if state.total_wafers > 0:
            progress = int((state.done_count / state.total_wafers) * 100)
        self.progress_label.setText(f"PROGRESS: {progress}% ({state.done_count}/{state.total_wafers} wafers)")
        self.progress_bar.setValue(progress)
        
        # 根据进度动态设置颜色
        if progress < 30:
            color = self.theme.danger
        elif progress < 70:
            color = self.theme.warning
        else:
            color = self.theme.success
        
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid rgb{self.theme.border_muted};
                border-radius: 4px;
                text-align: center;
                background-color: rgb{self.theme.bg_deep};
                font-size: 11pt;
                min-height: {ui_params.stats_panel.progress_bar_height}px;
            }}
            QProgressBar::chunk {{
                background-color: rgb{color};
                border-radius: 3px;
            }}
        """)

    def _update_summary(self, state: StateInfo) -> None:
        """更新三个摘要区块：System（仅关键指标）、Chambers（分组统计）、Robots（停留时间）。"""
        
        # ========== System 区块：紧凑显示 ==========
        system_avg = state.stats.get("system_avg", 0.0)
        system_max = state.stats.get("system_max", 0)
        system_diff = state.stats.get("system_diff", 0.0)
        
        system_html = f"""
        <div style='line-height: 1.4;'>
            <p style='margin: 2px 0;'><span style='color: rgb{self.theme.text_secondary};'>Avg:</span> <span style='font-size: 15pt; font-weight: 700; color: rgb{self.theme.text_kpi};'>{system_avg:.1f}</span></p>
            <p style='margin: 2px 0;'><span style='color: rgb{self.theme.text_secondary};'>Max:</span> <span style='font-size: 15pt; font-weight: 700; color: rgb{self.theme.text_kpi};'>{system_max}</span></p>
            <p style='margin: 2px 0;'><span style='color: rgb{self.theme.text_secondary};'>Diff:</span> <span style='font-size: 15pt; font-weight: 700; color: rgb{self.theme.text_kpi};'>{system_diff:.1f}</span></p>
        </div>
        """
        self.system_summary_label.setText(system_html)
        self.system_summary_label.setTextFormat(Qt.RichText)
        
        # ========== Chambers 区块：3行紧凑显示 ==========
        chambers_data = state.stats.get("chambers", {})
        
        # 提取各组数据
        pm78_data = chambers_data.get("PM7/8", {})
        pm1234_data = chambers_data.get("PM1/2/3/4", {})
        pm910_data = chambers_data.get("PM9/10", {})
        
        def format_chamber_line(name: str, data: dict) -> str:
            """格式化腔室组为单行显示"""
            avg = data.get("avg", 0.0)
            max_time = data.get("max", 0)
            
            # 根据数值选择颜色
            if avg > 0:
                avg_color = self.theme.success if avg < 100 else (self.theme.warning if avg < 200 else self.theme.danger)
            else:
                avg_color = self.theme.text_muted
                
            return f"""<p style='margin: 2px 0;'><span style='color: rgb{self.theme.accent_cyan}; font-weight: 600;'>{name}:</span> <span style='color: rgb{self.theme.text_secondary};'>Avg</span> <span style='font-size: 14pt; font-weight: 700; color: rgb{avg_color};'>{avg:.1f}</span> <span style='color: rgb{self.theme.text_secondary};'>Max</span> <span style='font-size: 14pt; font-weight: 700; color: rgb{self.theme.text_kpi};'>{max_time}</span></p>"""
        
        chambers_html = f"""
        <div style='line-height: 1.4;'>
            {format_chamber_line("PM7/8", pm78_data)}
            {format_chamber_line("PM1-4", pm1234_data)}
            {format_chamber_line("PM9/10", pm910_data)}
        </div>
        """
        self.chambers_summary_label.setText(chambers_html)
        self.chambers_summary_label.setTextFormat(Qt.RichText)
        
        # ========== Robots 区块：2行紧凑显示 ==========
        transports_data = state.stats.get("transports", {})
        robot_avg = transports_data.get("avg", 0.0)
        robot_max = transports_data.get("max", 0)
        
        # 根据数值选择颜色
        if robot_avg > 0:
            robot_color = self.theme.success if robot_avg < 10 else (self.theme.warning if robot_avg < 20 else self.theme.danger)
        else:
            robot_color = self.theme.text_muted
        
        robots_html = f"""
        <div style='line-height: 1.4;'>
            <p style='margin: 2px 0;'><span style='color: rgb{self.theme.text_secondary};'>Avg:</span> <span style='font-size: 15pt; font-weight: 700; color: rgb{robot_color};'>{robot_avg:.1f}</span></p>
            <p style='margin: 2px 0;'><span style='color: rgb{self.theme.text_secondary};'>Max:</span> <span style='font-size: 15pt; font-weight: 700; color: rgb{self.theme.text_kpi};'>{robot_max}</span></p>
        </div>
        """
        self.robots_summary_label.setText(robots_html)
        self.robots_summary_label.setTextFormat(Qt.RichText)

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
        """取最近 N 条历史，格式 Step #N - action (reward)，带颜色编码。"""
        n = ui_params.stats_panel.history_line_count
        lines = []
        for item in action_history[-n:]:
            reward = item['reward']
            # 根据奖励值选择颜色
            if reward > 0:
                color_code = f"color: rgb{self.theme.success};"
            elif reward < 0:
                color_code = f"color: rgb{self.theme.danger};"
            else:
                color_code = f"color: rgb{self.theme.text_muted};"
            
            # 使用 HTML 格式化
            lines.append(
                f'Step #{item["step"]} - {item["action"]} '
                f'<span style="{color_code} font-weight: bold;">({reward:+.2f})</span>'
            )
        
        # 设置为 HTML 格式
        if lines:
            self.history_text.setHtml("<br>".join(lines))
        else:
            self.history_text.clear()

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

        # Summary 三页正文（ToolBox tab 样式由 main_window.py 全局 QSS 管理）
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
