"""
左侧统计面板 (StatsPanel) - 仪表盘布局

展示系统运行时的 KPI、进度、摘要、释放计划和动作历史。
采用全部展开的仪表盘布局，不使用折叠/Accordion。

区块结构（自上而下）:
- KPI 网格: TIME / STEP / REWARD（2×2 卡片布局）
- PROGRESS: 进度条独立行
- 摘要区块: System / Chambers / Robots（Key-Value 两列对齐）
- RELEASE TIME: 固定高度 QTextEdit，内部滚动
- HISTORY: 固定高度 QTextEdit，内部滚动

布局参数由 ui_params.stats_panel 控制。
"""

from __future__ import annotations

from typing import Dict, Any, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QFrame,
    QTextEdit,
    QProgressBar,
    QSizePolicy,
)

from ..algorithm_interface import StateInfo
from ..theme import ColorTheme
from ..ui_params import ui_params


class KPICard(QFrame):
    """KPI 卡片组件：带背景、圆角、内边距的数值显示卡片"""
    
    def __init__(self, label: str, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("KPICard")
        
        p = ui_params.stats_panel
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(p.kpi_card_padding, p.kpi_card_padding, 
                                  p.kpi_card_padding, p.kpi_card_padding)
        layout.setSpacing(2)
        
        # 标签
        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("KPICardLabel")
        layout.addWidget(self.label_widget)
        
        # 数值
        self.value_widget = QLabel("0")
        self.value_widget.setObjectName("KPICardValue")
        layout.addWidget(self.value_widget)
        
        self.setFixedHeight(p.kpi_card_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    def set_value(self, value: str) -> None:
        """更新数值显示"""
        self.value_widget.setText(value)
    
    def set_value_color(self, color: tuple) -> None:
        """设置数值颜色"""
        self.value_widget.setStyleSheet(f"color: rgb{color};")


class SummaryRow(QWidget):
    """摘要行组件：Key-Value 两列对齐显示"""
    
    def __init__(self, key: str, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        
        p = ui_params.stats_panel
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Key 列
        self.key_label = QLabel(key)
        self.key_label.setObjectName("SummaryKey")
        self.key_label.setFixedWidth(p.summary_key_width)
        self.key_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.key_label)
        
        # Value 列
        self.value_label = QLabel("—")
        self.value_label.setObjectName("SummaryValue")
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.value_label, 1)
    
    def set_value(self, value: str, color: tuple = None) -> None:
        """更新数值"""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: rgb{color};")


class SectionHeader(QLabel):
    """区块标题组件"""
    
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionHeader")


class StatsPanel(QWidget):
    """左侧统计面板：仪表盘布局，全部展开，不折叠。"""

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("StatsPanelRoot")
        
        p = ui_params.stats_panel
        
        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setSpacing(p.section_spacing)
        self.main_layout.setContentsMargins(*p.layout_margins)
        
        # 1. KPI 网格区域
        self._create_kpi_section()
        
        # 2. 进度条区域
        self._create_progress_section()
        
        # 3. 摘要区域（System / Chambers / Robots）
        self._create_summary_section()
        
        # 4. RELEASE TIME 区域
        self._create_release_section()
        
        # 5. HISTORY 区域
        self._create_history_section()
        
        # 弹性空间
        self.main_layout.addStretch()
        
        # 应用样式
        self._apply_styles()

    def _create_kpi_section(self) -> None:
        """创建 KPI 网格区域：TIME / STEP / REWARD"""
        p = ui_params.stats_panel
        
        # 区块标题
        header = SectionHeader("MONITOR")
        self.main_layout.addWidget(header)
        
        # KPI 网格容器
        kpi_container = QWidget()
        kpi_layout = QGridLayout(kpi_container)
        kpi_layout.setSpacing(p.kpi_card_spacing)
        kpi_layout.setContentsMargins(0, 0, 0, 0)
        
        # TIME 卡片
        self.time_card = KPICard("TIME", self.theme)
        kpi_layout.addWidget(self.time_card, 0, 0)
        
        # STEP 卡片
        self.step_card = KPICard("STEP", self.theme)
        kpi_layout.addWidget(self.step_card, 0, 1)
        
        # REWARD 卡片（跨两列）
        self.reward_card = KPICard("REWARD", self.theme)
        kpi_layout.addWidget(self.reward_card, 1, 0, 1, 2)
        
        self.main_layout.addWidget(kpi_container)
        
        # REWARD 明细
        self.reward_detail = QLabel()
        self.reward_detail.setObjectName("RewardDetail")
        self.reward_detail.setWordWrap(True)
        self.reward_detail.setTextFormat(Qt.RichText)
        self.main_layout.addWidget(self.reward_detail)

    def _create_progress_section(self) -> None:
        """创建进度条区域"""
        p = ui_params.stats_panel
        
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(p.progress_label_spacing)
        
        # 进度标签
        self.progress_label = QLabel("PROGRESS: 0%")
        self.progress_label.setObjectName("ProgressLabel")
        progress_layout.addWidget(self.progress_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("MainProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(p.progress_bar_height)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.main_layout.addWidget(progress_container)

    def _create_summary_section(self) -> None:
        """创建摘要区域：System / Chambers / Robots"""
        p = ui_params.stats_panel
        
        # ===== System 区块 =====
        system_header = SectionHeader("SYSTEM")
        self.main_layout.addWidget(system_header)
        
        system_container = QFrame()
        system_container.setObjectName("SummaryFrame")
        system_layout = QVBoxLayout(system_container)
        system_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding,
                                         p.summary_frame_padding, p.summary_frame_padding)
        system_layout.setSpacing(p.summary_row_spacing)
        
        self.sys_avg_row = SummaryRow("Avg", self.theme)
        self.sys_max_row = SummaryRow("Max", self.theme)
        self.sys_diff_row = SummaryRow("Diff", self.theme)
        
        system_layout.addWidget(self.sys_avg_row)
        system_layout.addWidget(self.sys_max_row)
        system_layout.addWidget(self.sys_diff_row)
        
        self.main_layout.addWidget(system_container)
        
        # ===== Chambers 区块 =====
        chambers_header = SectionHeader("CHAMBERS")
        self.main_layout.addWidget(chambers_header)
        
        chambers_container = QFrame()
        chambers_container.setObjectName("SummaryFrame")
        chambers_layout = QVBoxLayout(chambers_container)
        chambers_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding,
                                           p.summary_frame_padding, p.summary_frame_padding)
        chambers_layout.setSpacing(p.summary_row_spacing)
        
        self.pm78_row = SummaryRow("PM7/8", self.theme)
        self.pm1234_row = SummaryRow("PM1-4", self.theme)
        self.pm910_row = SummaryRow("PM9/10", self.theme)
        
        chambers_layout.addWidget(self.pm78_row)
        chambers_layout.addWidget(self.pm1234_row)
        chambers_layout.addWidget(self.pm910_row)
        
        self.main_layout.addWidget(chambers_container)
        
        # ===== Robots 区块 =====
        robots_header = SectionHeader("ROBOTS")
        self.main_layout.addWidget(robots_header)
        
        robots_container = QFrame()
        robots_container.setObjectName("SummaryFrame")
        robots_layout = QVBoxLayout(robots_container)
        robots_layout.setContentsMargins(p.summary_frame_padding, p.summary_frame_padding,
                                         p.summary_frame_padding, p.summary_frame_padding)
        robots_layout.setSpacing(p.summary_row_spacing)
        
        self.robot_avg_row = SummaryRow("Avg", self.theme)
        self.robot_max_row = SummaryRow("Max", self.theme)
        
        robots_layout.addWidget(self.robot_avg_row)
        robots_layout.addWidget(self.robot_max_row)
        
        self.main_layout.addWidget(robots_container)

    def _create_release_section(self) -> None:
        """创建 RELEASE TIME 区域：固定高度，内部滚动"""
        p = ui_params.stats_panel
        
        header = SectionHeader("RELEASE")
        self.main_layout.addWidget(header)
        
        self.release_text = QTextEdit()
        self.release_text.setObjectName("ReleaseText")
        self.release_text.setReadOnly(True)
        self.release_text.setFixedHeight(p.release_fixed_height)
        self.release_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.release_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.main_layout.addWidget(self.release_text)

    def _create_history_section(self) -> None:
        """创建 HISTORY 区域：固定高度，内部滚动"""
        p = ui_params.stats_panel
        
        header = SectionHeader("HISTORY")
        self.main_layout.addWidget(header)
        
        self.history_text = QTextEdit()
        self.history_text.setObjectName("HistoryText")
        self.history_text.setReadOnly(True)
        self.history_text.setFixedHeight(p.history_fixed_height)
        self.history_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.main_layout.addWidget(self.history_text)

    def _apply_styles(self) -> None:
        """应用组件内部样式（覆盖全局 QSS）"""
        t = self.theme
        p = ui_params.stats_panel
        
        qss = f"""
        /* ===== StatsPanel 根容器 ===== */
        #StatsPanelRoot {{
            background-color: transparent;
        }}
        
        /* ===== 区块标题 ===== */
        #SectionHeader {{
            font-size: {p.section_title_font_pt}pt;
            font-weight: 700;
            color: rgb{t.accent_cyan};
            padding: 2px 0;
            margin: 0;
        }}
        
        /* ===== KPI 卡片 ===== */
        #KPICard {{
            background-color: rgba{(*t.bg_surface, 0.5)};
            border: 1px solid rgb{t.border_muted};
            border-radius: {p.kpi_card_radius}px;
        }}
        #KPICardLabel {{
            font-size: {p.kpi_label_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_secondary};
        }}
        #KPICardValue {{
            font-size: {p.kpi_value_font_pt}pt;
            font-weight: 700;
            color: rgb{t.text_kpi};
        }}
        
        /* ===== 进度条 ===== */
        #ProgressLabel {{
            font-size: {p.label_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_primary};
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
        
        /* ===== 摘要区域 ===== */
        #SummaryFrame {{
            background-color: rgba{(*t.bg_surface, 0.3)};
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
        }}
        #SummaryKey {{
            font-size: {p.summary_key_font_pt}pt;
            font-weight: 600;
            color: rgb{t.text_secondary};
        }}
        #SummaryValue {{
            font-size: {p.summary_value_font_pt}pt;
            font-weight: 700;
            color: rgb{t.text_kpi};
        }}
        
        /* ===== Reward 明细 ===== */
        #RewardDetail {{
            font-size: {p.reward_detail_font_pt}pt;
            color: rgb{t.text_muted};
            padding: 4px 0;
        }}
        
        /* ===== 文本框 ===== */
        #ReleaseText, #HistoryText {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            color: rgb{t.text_secondary};
            font-family: "{p.font_family}";
            font-size: {p.release_font_pt}pt;
            padding: 6px;
        }}
        """
        self.setStyleSheet(qss)

    # ===== 公共更新接口（保持不变） =====

    def update_state(self, state: StateInfo, action_history: List[Dict[str, Any]], 
                     trend_data: Dict[str, List[float]] | None = None) -> None:
        """全量刷新：KPI、摘要、释放计划、历史。"""
        self._update_metrics(state)
        self._update_summary(state)
        self._update_release_schedule(state)
        self._update_history(action_history)

    def update_reward(self, total_reward: float, detail: Dict[str, float]) -> None:
        """单独刷新奖励：总奖励 + 明细（带颜色编码）。"""
        self.reward_card.set_value(f"{total_reward:.2f}")
        
        # 明细显示
        detail_parts = []
        for k, v in sorted(detail.items()):
            color = self.theme.success if v >= 0 else self.theme.danger
            detail_parts.append(f'<span style="color: rgb{color};">{k}: {v:+.2f}</span>')
        
        if detail_parts:
            self.reward_detail.setText(" | ".join(detail_parts))
        else:
            self.reward_detail.setText("")

    def update_step(self, step: int) -> None:
        """单独刷新步数。"""
        self.step_card.set_value(str(step))

    def _update_metrics(self, state: StateInfo) -> None:
        """更新 KPI 卡片和进度条"""
        # TIME
        self.time_card.set_value(str(int(state.time)))
        
        # PROGRESS
        progress = 0
        if state.total_wafers > 0:
            progress = int((state.done_count / state.total_wafers) * 100)
        self.progress_label.setText(f"PROGRESS: {progress}% ({state.done_count}/{state.total_wafers})")
        self.progress_bar.setValue(progress)
        
        # 进度条颜色
        if progress < 30:
            color = self.theme.danger
        elif progress < 70:
            color = self.theme.warning
        else:
            color = self.theme.success
        
        p = ui_params.stats_panel
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
        """更新摘要区域：System / Chambers / Robots"""
        
        # ===== System =====
        system_avg = state.stats.get("system_avg", 0.0)
        system_max = state.stats.get("system_max", 0)
        system_diff = state.stats.get("system_diff", 0.0)
        
        self.sys_avg_row.set_value(f"{system_avg:.1f}")
        self.sys_max_row.set_value(str(system_max))
        self.sys_diff_row.set_value(f"{system_diff:.1f}")
        
        # ===== Chambers =====
        chambers_data = state.stats.get("chambers", {})
        
        pm78 = chambers_data.get("PM7/8", {})
        pm1234 = chambers_data.get("PM1/2/3/4", {})
        pm910 = chambers_data.get("PM9/10", {})
        
        def format_chamber(data: dict) -> str:
            avg = data.get("avg", 0.0)
            max_time = data.get("max", 0)
            return f"Avg {avg:.1f}  Max {max_time}"
        
        self.pm78_row.set_value(format_chamber(pm78))
        self.pm1234_row.set_value(format_chamber(pm1234))
        self.pm910_row.set_value(format_chamber(pm910))
        
        # ===== Robots =====
        transports = state.stats.get("transports", {})
        robot_avg = transports.get("avg", 0.0)
        robot_max = transports.get("max", 0)
        
        # 颜色编码
        if robot_avg > 0:
            color = self.theme.success if robot_avg < 10 else (
                self.theme.warning if robot_avg < 20 else self.theme.danger)
        else:
            color = None
        
        self.robot_avg_row.set_value(f"{robot_avg:.1f}", color)
        self.robot_max_row.set_value(str(robot_max))

    def _update_release_schedule(self, state: StateInfo) -> None:
        """更新 RELEASE TIME 区域"""
        schedule = state.stats.get("release_schedule", {})
        lines = []
        for place_name, items in schedule.items():
            if not items:
                continue
            pairs = ", ".join([f"{tid}→{rt}" for tid, rt in items])
            lines.append(f"{place_name}: {pairs}")
        self.release_text.setText("\n".join(lines) if lines else "—")

    def _update_history(self, action_history: List[Dict[str, Any]]) -> None:
        """更新 HISTORY 区域"""
        n = ui_params.stats_panel.history_line_count
        lines = []
        for item in action_history[-n:]:
            reward = item['reward']
            if reward > 0:
                color = f"color: rgb{self.theme.success};"
            elif reward < 0:
                color = f"color: rgb{self.theme.danger};"
            else:
                color = f"color: rgb{self.theme.text_muted};"
            
            lines.append(
                f'#{item["step"]} {item["action"]} '
                f'<span style="{color} font-weight: bold;">({reward:+.2f})</span>'
            )
        
        if lines:
            self.history_text.setHtml("<br>".join(lines))
        else:
            self.history_text.clear()

    def apply_params(self) -> None:
        """根据 ui_params 重新应用样式（热更新用）"""
        self._apply_styles()
