"""
左侧统计面板 (StatsPanel) - 仪表盘布局

改进：
- REWARDS 面板固定高度，只显示非零项，按绝对值排序
- System 区块压缩为单行
- 统一背景色，消除条纹不一致
"""

from __future__ import annotations

from typing import Dict, Any, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QTextEdit,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QAbstractItemView,
)

from ..algorithm_interface import StateInfo
from ..theme import ColorTheme
from ..ui_params import ui_params


class KPICard(QFrame):
    """KPI 卡片组件"""
    
    def __init__(self, label: str, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("KPICard")
        
        p = ui_params.stats_panel
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(p.kpi_card_padding, p.kpi_card_padding, 
                                  p.kpi_card_padding, p.kpi_card_padding)
        layout.setSpacing(2)
        
        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("KPICardLabel")
        layout.addWidget(self.label_widget)
        
        self.value_widget = QLabel("0")
        self.value_widget.setObjectName("KPICardValue")
        layout.addWidget(self.value_widget)
        
        self.setFixedHeight(p.kpi_card_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    
    def set_value(self, value: str) -> None:
        self.value_widget.setText(value)
    
    def set_value_color(self, color: tuple) -> None:
        self.value_widget.setStyleSheet(f"color: rgb{color}; background: transparent;")


class SectionHeader(QLabel):
    """区块标题组件"""
    
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionHeader")


class MetricRow(QFrame):
    """指标行组件：统一背景，Key-Value 对齐"""
    
    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("MetricRow")
        
        p = ui_params.stats_panel
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(p.summary_frame_padding, 4, p.summary_frame_padding, 4)
        layout.setSpacing(p.summary_row_spacing)
        
        self.key_label = QLabel()
        self.key_label.setObjectName("MetricKey")
        layout.addWidget(self.key_label)
        
        self.value_label = QLabel()
        self.value_label.setObjectName("MetricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.value_label, 1)
    
    def set_data(self, key: str, value: str, value_color: tuple = None) -> None:
        self.key_label.setText(key)
        self.value_label.setText(value)
        if value_color:
            self.value_label.setStyleSheet(f"color: rgb{value_color}; background: transparent;")
        else:
            self.value_label.setStyleSheet("background: transparent;")


class StatsPanel(QWidget):
    """左侧统计面板：仪表盘布局，全部展开。"""

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("StatsPanelRoot")
        
        p = ui_params.stats_panel
        
        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.main_layout.setSpacing(p.section_spacing)
        self.main_layout.setContentsMargins(*p.layout_margins)
        
        # 1. KPI 网格区域
        self._create_kpi_section()
        
        # 2. 进度条区域
        self._create_progress_section()
        
        # 3. System 单行显示
        self._create_system_section()
        
        # 4. Chambers 摘要
        self._create_chambers_section()
        
        # 5. Robots 摘要
        self._create_robots_section()
        
        # 6. REWARDS 区块（固定高度，只显示非零项）
        self._create_rewards_section()
        
        # 7. RELEASE TIME 区域
        self._create_release_section()
        
        # 8. HISTORY 区域
        self._create_history_section()
        
        # 弹性空间
        self.main_layout.addStretch()
        
        # 应用样式
        self._apply_styles()

    def _create_kpi_section(self) -> None:
        p = ui_params.stats_panel
        
        header = SectionHeader("MONITOR")
        self.main_layout.addWidget(header)
        
        kpi_container = QWidget()
        kpi_layout = QGridLayout(kpi_container)
        kpi_layout.setSpacing(p.kpi_card_spacing)
        kpi_layout.setContentsMargins(0, 0, 0, 0)
        
        self.time_card = KPICard("TIME", self.theme)
        kpi_layout.addWidget(self.time_card, 0, 0)
        
        self.step_card = KPICard("STEP", self.theme)
        kpi_layout.addWidget(self.step_card, 0, 1)
        
        self.reward_card = KPICard("REWARD", self.theme)
        kpi_layout.addWidget(self.reward_card, 1, 0, 1, 2)
        
        self.main_layout.addWidget(kpi_container)

    def _create_progress_section(self) -> None:
        p = ui_params.stats_panel
        
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(p.progress_label_spacing)
        
        self.progress_label = QLabel("PROGRESS: 0%")
        self.progress_label.setObjectName("ProgressLabel")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("MainProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(p.progress_bar_height)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.main_layout.addWidget(progress_container)

    def _create_system_section(self) -> None:
        """System 区块：单行显示 Avg | Max | Diff"""
        p = ui_params.stats_panel
        
        header = SectionHeader("SYSTEM")
        self.main_layout.addWidget(header)
        
        self.system_row = QFrame()
        self.system_row.setObjectName("MetricRow")
        row_layout = QHBoxLayout(self.system_row)
        row_layout.setContentsMargins(p.summary_frame_padding, 6, p.summary_frame_padding, 6)
        row_layout.setSpacing(12)
        
        # Avg
        avg_container = QWidget()
        avg_layout = QHBoxLayout(avg_container)
        avg_layout.setContentsMargins(0, 0, 0, 0)
        avg_layout.setSpacing(4)
        avg_key = QLabel("Avg")
        avg_key.setObjectName("MetricKey")
        self.sys_avg_value = QLabel("0.0")
        self.sys_avg_value.setObjectName("MetricValue")
        avg_layout.addWidget(avg_key)
        avg_layout.addWidget(self.sys_avg_value)
        row_layout.addWidget(avg_container)
        
        sep1 = QLabel("|")
        sep1.setObjectName("MetricSeparator")
        row_layout.addWidget(sep1)
        
        # Max
        max_container = QWidget()
        max_layout = QHBoxLayout(max_container)
        max_layout.setContentsMargins(0, 0, 0, 0)
        max_layout.setSpacing(4)
        max_key = QLabel("Max")
        max_key.setObjectName("MetricKey")
        self.sys_max_value = QLabel("0")
        self.sys_max_value.setObjectName("MetricValue")
        max_layout.addWidget(max_key)
        max_layout.addWidget(self.sys_max_value)
        row_layout.addWidget(max_container)
        
        sep2 = QLabel("|")
        sep2.setObjectName("MetricSeparator")
        row_layout.addWidget(sep2)
        
        # Diff
        diff_container = QWidget()
        diff_layout = QHBoxLayout(diff_container)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        diff_layout.setSpacing(4)
        diff_key = QLabel("Diff")
        diff_key.setObjectName("MetricKey")
        self.sys_diff_value = QLabel("0.0")
        self.sys_diff_value.setObjectName("MetricValue")
        diff_layout.addWidget(diff_key)
        diff_layout.addWidget(self.sys_diff_value)
        row_layout.addWidget(diff_container)
        
        row_layout.addStretch()
        
        self.main_layout.addWidget(self.system_row)

    def _create_chambers_section(self) -> None:
        header = SectionHeader("CHAMBERS")
        self.main_layout.addWidget(header)
        
        self.pm78_row = MetricRow(self.theme)
        self.pm1234_row = MetricRow(self.theme)
        self.pm910_row = MetricRow(self.theme)
        
        self.main_layout.addWidget(self.pm78_row)
        self.main_layout.addWidget(self.pm1234_row)
        self.main_layout.addWidget(self.pm910_row)

    def _create_robots_section(self) -> None:
        header = SectionHeader("ROBOTS")
        self.main_layout.addWidget(header)
        
        self.robot_avg_row = MetricRow(self.theme)
        self.robot_max_row = MetricRow(self.theme)
        
        self.main_layout.addWidget(self.robot_avg_row)
        self.main_layout.addWidget(self.robot_max_row)

    def _create_rewards_section(self) -> None:
        """REWARDS 区块：固定高度，只显示非零项，按绝对值排序"""
        p = ui_params.stats_panel
        
        header = SectionHeader("REWARDS")
        self.main_layout.addWidget(header)
        
        # 固定高度容器
        self.rewards_container = QFrame()
        self.rewards_container.setObjectName("RewardsContainer")
        self.rewards_container.setFixedHeight(p.rewards_fixed_height)
        
        self.rewards_layout = QVBoxLayout(self.rewards_container)
        self.rewards_layout.setContentsMargins(p.summary_frame_padding, 6, 
                                               p.summary_frame_padding, 6)
        self.rewards_layout.setSpacing(p.rewards_item_spacing)
        self.rewards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.main_layout.addWidget(self.rewards_container)

    def _create_release_section(self) -> None:
        p = ui_params.stats_panel
        
        header = SectionHeader("RELEASE")
        self.main_layout.addWidget(header)
        
        self.release_text = QTextEdit()
        self.release_text.setObjectName("ReleaseText")
        self.release_text.setReadOnly(True)
        self.release_text.setFixedHeight(p.release_fixed_height)
        self.release_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.release_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.main_layout.addWidget(self.release_text)

    def _create_history_section(self) -> None:
        p = ui_params.stats_panel
        
        header = SectionHeader("HISTORY")
        self.main_layout.addWidget(header)
        
        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        self.history_list.setFixedHeight(p.history_fixed_height)
        self.history_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.history_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history_list.setWordWrap(False)
        self.history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        
        self.main_layout.addWidget(self.history_list)

    def _apply_styles(self) -> None:
        t = self.theme
        p = ui_params.stats_panel
        
        qss = f"""
        #StatsPanelRoot {{
            background-color: transparent;
        }}
        
        #SectionHeader {{
            font-size: {p.section_title_font_pt}pt;
            font-weight: 700;
            color: rgb{t.accent_cyan};
            padding: 2px 0;
            margin: 0;
            background: transparent;
        }}
        
        #KPICard {{
            background-color: rgba{(*t.bg_surface, 0.4)};
            border: 1px solid rgb{t.border_muted};
            border-radius: {p.kpi_card_radius}px;
        }}
        #KPICardLabel {{
            font-size: {p.kpi_label_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_secondary};
            background-color: transparent;
        }}
        #KPICardValue {{
            font-size: {p.kpi_value_font_pt}pt;
            font-weight: 700;
            color: rgb{t.text_kpi};
            background-color: transparent;
        }}
        
        #ProgressLabel {{
            font-size: {p.label_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_primary};
            background-color: transparent;
        }}
        #MainProgressBar {{
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            background-color: rgb{t.bg_deep};
        }}
        #MainProgressBar::chunk {{
            background-color: rgb{t.accent_cyan};
            border-radius: 3px;
        }}
        
        #MetricRow, #RewardsContainer {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
        }}
        #MetricKey {{
            font-size: {p.summary_key_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_muted};
            background-color: transparent;
        }}
        #MetricValue {{
            font-size: {p.summary_value_font_pt}pt;
            font-weight: 700;
            color: rgb{t.text_kpi};
            background-color: transparent;
        }}
        #MetricSeparator {{
            font-size: {p.summary_key_font_pt}pt;
            color: rgb{t.border_muted};
            background-color: transparent;
        }}
        
        #ReleaseText {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            color: rgb{t.text_secondary};
            font-family: "{p.font_family}";
            font-size: {p.release_font_pt}pt;
            padding: 6px;
        }}
        
        #HistoryList {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            outline: none;
        }}
        #HistoryList::item {{
            color: rgb{t.text_secondary};
            font-family: "{p.font_family}";
            font-size: {p.history_font_pt}pt;
            padding: 2px 6px;
            border: none;
            background-color: transparent;
        }}
        
        .RewardItem {{
            font-size: {p.rewards_item_font_pt}pt;
            background: transparent;
        }}
        """
        self.setStyleSheet(qss)

    # ===== 公共更新接口 =====

    def update_state(self, state: StateInfo, action_history: List[Dict[str, Any]], 
                     trend_data: Dict[str, List[float]] | None = None) -> None:
        self._update_metrics(state)
        self._update_summary(state)
        self._update_release_schedule(state)
        self._update_history(action_history)

    def update_reward(self, total_reward: float, detail: Dict[str, float]) -> None:
        """更新奖励：总奖励 + 明细（只显示非零项，按绝对值排序）"""
        # 更新总奖励
        if total_reward > 0:
            color = self.theme.success
        elif total_reward < 0:
            color = self.theme.danger
        else:
            color = self.theme.text_kpi
        self.reward_card.set_value(f"{total_reward:.2f}")
        self.reward_card.set_value_color(color)
        
        # 清理旧的明细
        while self.rewards_layout.count():
            item = self.rewards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 过滤非零项并按绝对值排序
        non_zero_items = [(k, v) for k, v in detail.items() if abs(v) > 0.001]
        sorted_items = sorted(non_zero_items, key=lambda x: abs(x[1]), reverse=True)
        
        p = ui_params.stats_panel
        max_items = p.rewards_max_items
        
        # 显示前 N 项
        display_items = sorted_items[:max_items]
        remaining = len(sorted_items) - max_items
        
        for name, value in display_items:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            
            # 名称
            name_label = QLabel(name)
            name_label.setStyleSheet(f"""
                font-size: {p.rewards_item_font_pt}pt;
                color: rgb{self.theme.text_secondary};
                background: transparent;
            """)
            row_layout.addWidget(name_label)
            
            row_layout.addStretch()
            
            # 值（降低饱和度）
            if value > 0:
                base_color = self.theme.success
            else:
                base_color = self.theme.danger
            # 降低饱和度
            value_color = tuple(int(c * 0.75 + 60) for c in base_color)
            
            value_label = QLabel(f"{value:+.2f}")
            value_label.setStyleSheet(f"""
                font-size: {p.rewards_item_font_pt}pt;
                font-weight: 600;
                color: rgb{value_color};
                background: transparent;
            """)
            row_layout.addWidget(value_label)
            
            self.rewards_layout.addWidget(row)
        
        # 显示剩余项数
        if remaining > 0:
            more_label = QLabel(f"+{remaining} more...")
            more_label.setStyleSheet(f"""
                font-size: {p.rewards_item_font_pt - 1}pt;
                color: rgb{self.theme.text_muted};
                background: transparent;
                font-style: italic;
            """)
            self.rewards_layout.addWidget(more_label)
        
        # 如果没有非零项
        if not display_items:
            empty_label = QLabel("—")
            empty_label.setStyleSheet(f"""
                font-size: {p.rewards_item_font_pt}pt;
                color: rgb{self.theme.text_muted};
                background: transparent;
            """)
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rewards_layout.addWidget(empty_label)

    def update_step(self, step: int) -> None:
        self.step_card.set_value(str(step))

    def _update_metrics(self, state: StateInfo) -> None:
        self.time_card.set_value(str(int(state.time)))
        
        progress = 0
        if state.total_wafers > 0:
            progress = int((state.done_count / state.total_wafers) * 100)
        self.progress_label.setText(f"PROGRESS: {progress}% ({state.done_count}/{state.total_wafers})")
        self.progress_bar.setValue(progress)
        
        if progress < 30:
            color = self.theme.danger
        elif progress < 70:
            color = self.theme.warning
        else:
            color = self.theme.success
        
        self.progress_bar.setStyleSheet(f"""
            #MainProgressBar {{
                border: 1px solid rgb{self.theme.border_muted};
                border-radius: 4px;
                background-color: rgb{self.theme.bg_deep};
            }}
            #MainProgressBar::chunk {{
                background-color: rgb{color};
                border-radius: 3px;
            }}
        """)

    def _update_summary(self, state: StateInfo) -> None:
        system_avg = state.stats.get("system_avg", 0.0)
        system_max = state.stats.get("system_max", 0)
        system_diff = state.stats.get("system_diff", 0.0)
        
        self.sys_avg_value.setText(f"{system_avg:.1f}")
        self.sys_max_value.setText(str(system_max))
        self.sys_diff_value.setText(f"{system_diff:.1f}")
        
        chambers_data = state.stats.get("chambers", {})
        
        pm78 = chambers_data.get("PM7/8", {})
        pm1234 = chambers_data.get("PM1/2/3/4", {})
        pm910 = chambers_data.get("PM9/10", {})
        
        def format_chamber(data: dict) -> str:
            avg = data.get("avg", 0.0)
            max_time = data.get("max", 0)
            return f"Avg {avg:.1f}  Max {max_time}"
        
        self.pm78_row.set_data("PM7/8", format_chamber(pm78))
        self.pm1234_row.set_data("PM1-4", format_chamber(pm1234))
        self.pm910_row.set_data("PM9/10", format_chamber(pm910))
        
        transports = state.stats.get("transports", {})
        robot_avg = transports.get("avg", 0.0)
        robot_max = transports.get("max", 0)
        
        color = None
        if robot_avg > 0:
            color = self.theme.success if robot_avg < 10 else (
                self.theme.warning if robot_avg < 20 else self.theme.danger)
        
        self.robot_avg_row.set_data("Avg", f"{robot_avg:.1f}", color)
        self.robot_max_row.set_data("Max", str(robot_max))

    def _update_release_schedule(self, state: StateInfo) -> None:
        schedule = state.stats.get("release_schedule", {})
        lines = []
        for place_name, items in schedule.items():
            if not items:
                continue
            pairs = ", ".join([f"{tid}→{rt}" for tid, rt in items])
            lines.append(f"{place_name}: {pairs}")
        self.release_text.setText("\n".join(lines) if lines else "—")

    def _update_history(self, action_history: List[Dict[str, Any]]) -> None:
        n = ui_params.stats_panel.history_line_count
        
        self.history_list.clear()
        
        for item in action_history[-n:]:
            reward = item['reward']
            if reward > 0:
                color = self.theme.success
            elif reward < 0:
                color = self.theme.danger
            else:
                color = self.theme.text_muted
            
            list_item = QListWidgetItem(f"#{item['step']} {item['action']} ({reward:+.2f})")
            list_item.setForeground(QColor(*color))
            self.history_list.addItem(list_item)

    def apply_params(self) -> None:
        self._apply_styles()
