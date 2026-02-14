"""
右侧控制面板

改进：
- 禁用 Transition 按钮更淡、边框弱化
- 启用 Transition 按钮更突出
- 速度按钮选中态高亮
- Auto 模式 ON 状态更明显
- 无模型时禁用 Model Step / Auto Mode
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout

from ..algorithm_interface import ActionInfo
from ..theme import ColorTheme
from ..ui_params import ui_params


class ControlPanel(QWidget):
    """控制按钮面板"""

    action_clicked = Signal(int)
    # random_clicked removed
    model_step_clicked = Signal()
    model_auto_toggled = Signal(bool)
    model_b_auto_toggled = Signal(bool)
    verify_planb_clicked = Signal()
    reset_clicked = Signal()
    gantt_clicked = Signal()
    speed_changed = Signal(float)

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ControlPanelRoot")
        p = ui_params.control_panel
        self.theme = theme
        self._current_speed = 1.0
        self._model_enabled = False  # 是否有模型

        self.buttons: List[QPushButton] = []
        self.speed_buttons: List[QPushButton] = []
        self.auto_active = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title = QLabel("TRANSITIONS")
        self.title.setObjectName("PanelTitle")
        layout.addWidget(self.title)

        self.transition_group = QVBoxLayout()
        self.transition_group.setSpacing(4)
        layout.addLayout(self.transition_group)

        layout.addSpacing(p.spacing_after_transitions)
        control_title = QLabel("CONTROL")
        control_title.setObjectName("PanelTitle")
        layout.addWidget(control_title)
        layout.addSpacing(p.spacing_after_control_title)

        self.wait_button = QPushButton("WAIT (W)")
        self.wait_button.setObjectName("WaitButton")
        self.wait_button.clicked.connect(lambda: self.action_clicked.emit(-100))
        layout.addWidget(self.wait_button)

        # Random button removed

        self.model_button = QPushButton("Model A Step (M)")
        self.model_button.setObjectName("ModelButton")
        self.model_button.setEnabled(False)  # 默认禁用
        self.model_button.clicked.connect(self.model_step_clicked.emit)
        layout.addWidget(self.model_button)

        self.model_auto_button = QPushButton("Auto Mode (A)")
        self.model_auto_button.setObjectName("AutoButton")
        self.model_auto_button.setEnabled(False)  # 默认禁用
        self.model_auto_button.clicked.connect(self._toggle_auto)
        layout.addWidget(self.model_auto_button)
        
        # Verify PlanB Button (Renamed to Model B Step)
        self.verify_button = QPushButton("Model B Step (V)")
        self.verify_button.setObjectName("VerifyButton")
        self.verify_button.clicked.connect(self.verify_planb_clicked.emit)
        layout.addWidget(self.verify_button)

        # New Auto Model B Button
        self.model_b_auto_button = QPushButton("Auto Model B (B)")
        self.model_b_auto_button.setObjectName("ModelBAutoButton")
        self.model_b_auto_button.clicked.connect(self._toggle_model_b_auto)
        layout.addWidget(self.model_b_auto_button)

        layout.addSpacing(p.spacing_before_speed)

        # 速度标签
        speed_label = QLabel("SPEED")
        speed_label.setObjectName("PanelTitle")
        layout.addWidget(speed_label)

        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(4)
        for speed in p.speed_options:
            btn = QPushButton(f"{speed}x")
            btn.setObjectName("SpeedButton")
            btn.clicked.connect(lambda _=False, s=speed: self._set_speed(s))
            speed_layout.addWidget(btn)
            self.speed_buttons.append(btn)
        layout.addLayout(speed_layout)
        
        # 默认选中 1x
        self._update_speed_highlight()

        layout.addSpacing(p.spacing_before_reset)

        self.reset_button = QPushButton("Reset (Space)")
        self.reset_button.setObjectName("ResetButton")
        self.reset_button.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_button)

        layout.addSpacing(p.spacing_before_reset)

        # UTILS Section
        utils_title = QLabel("UTILS")
        utils_title.setObjectName("PanelTitle")
        layout.addWidget(utils_title)
        
        self.gantt_button = QPushButton("Draw Gantt")
        self.gantt_button.setObjectName("GanttButton")
        self.gantt_button.clicked.connect(self.gantt_clicked.emit)
        layout.addWidget(self.gantt_button)
        
        # 应用样式
        self._apply_styles()

    def _apply_styles(self) -> None:
        t = self.theme
        p = ui_params.control_panel
        
        qss = f"""
        #ControlPanelRoot {{
            background-color: transparent;
        }}
        
        #PanelTitle {{
            font-size: {p.title_font_size_px}px;
            font-weight: 700;
            color: rgb{t.accent_cyan};
            padding: 4px 0;
        }}
        
        /* 基础按钮样式 */
        QPushButton {{
            background-color: rgb{t.bg_surface};
            color: rgb{t.text_primary};
            border: 1px solid rgb{t.border};
            border-radius: 5px;
            padding: {p.button_padding_v}px {p.button_padding_h}px;
            font-size: {p.button_font_size_px}px;
            min-height: {p.button_min_height}px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: rgb{t.bg_elevated};
            border-color: rgb{t.accent_cyan};
        }}
        QPushButton:disabled {{
            color: rgb{t.text_muted};
            border-color: rgba{(*t.border_muted, 0.5)};
            background-color: rgb{t.bg_deep};
        }}
        
        /* Transition 按钮 - 启用态更突出 */
        #TransitionButton {{
            font-size: {p.transition_button_font_size_px}px;
            padding: 4px 8px;
            min-height: 28px;
        }}
        #TransitionButton:enabled {{
            border-color: rgb{t.btn_transition};
            color: rgb{t.text_primary};
        }}
        #TransitionButton:disabled {{
            color: rgba{(*t.text_muted, 0.6)};
            border-color: rgba{(*t.border_muted, 0.3)};
            background-color: rgba{(*t.bg_deep, 0.5)};
        }}
        
        /* 控制按钮分类 */
        #WaitButton {{
            border-color: rgb{t.btn_wait};
        }}
        #WaitButton:hover {{
            background-color: rgba{(*t.btn_wait, 0.2)};
        }}
        
        /* RandomButton removed */
        
        #ModelButton {{
            border-color: rgb{t.btn_model};
        }}
        #ModelButton:hover {{
            background-color: rgba{(*t.btn_model, 0.2)};
        }}
        
        #AutoButton {{
            border-color: rgb{t.btn_auto};
        }}
        #AutoButton:hover {{
            background-color: rgba{(*t.btn_auto, 0.2)};
        }}
        
        #ModelBAutoButton {{
            border-color: rgb{t.btn_transition};
        }}
        #ModelBAutoButton:hover {{
            background-color: rgba{(*t.btn_transition, 0.2)};
        }}
        
        #SpeedButton {{
            min-height: 26px;
            padding: 3px 6px;
            font-size: 11px;
        }}
        
        #ResetButton {{
            border-color: rgb{t.btn_reset};
        }}
        #ResetButton:hover {{
            background-color: rgba{(*t.btn_reset, 0.2)};
        }}
        
        #VerifyButton {{
            border-color: rgb{t.btn_random};
        }}
        #VerifyButton:hover {{
            background-color: rgba{(*t.btn_random, 0.2)};
        }}

        #GanttButton {{
            border-color: rgb{t.btn_gantt};
        }}
        #GanttButton:hover {{
            background-color: rgba{(*t.btn_gantt, 0.2)};
        }}
        """
        self.setStyleSheet(qss)

    def set_model_enabled(self, enabled: bool) -> None:
        """设置是否有模型可用，控制 Model Step / Auto Mode 按钮启用状态"""
        self._model_enabled = enabled
        self.model_button.setEnabled(enabled)
        # Auto 按钮：有模型时启用，但如果当前 auto_active 则保持当前状态
        if not enabled and self.auto_active:
            # 没有模型但 auto 正在运行，需要停止
            self.set_auto_active(False)
        self.model_auto_button.setEnabled(enabled)

    def update_actions(self, actions: List[ActionInfo]) -> None:
        # 清理旧按钮
        while self.transition_group.count():
            item = self.transition_group.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.buttons.clear()
        for action in actions:
            if action.action_id < 0:
                continue
            if action.action_name == "WAIT":
                continue
            btn = QPushButton(action.action_name)
            btn.setEnabled(action.enabled)
            tip = (action.description or "Condition not met") if not action.enabled else "Click to execute"
            btn.setToolTip(tip)
            btn.setObjectName("TransitionButton")
            btn.clicked.connect(lambda _=False, a=action.action_id: self.action_clicked.emit(a))
            self.transition_group.addWidget(btn)
            self.buttons.append(btn)

    def set_auto_active(self, active: bool) -> None:
        self.auto_active = active
        if active:
            self.model_auto_button.setText("◉ AUTO ON")
            self.model_auto_button.setStyleSheet(f"""
                background-color: rgba{(*self.theme.btn_auto, 0.3)};
                border: 2px solid rgb{self.theme.btn_auto};
                color: rgb{self.theme.text_primary};
                font-weight: 700;
            """)
            self.model_auto_button.setText("Auto Mode (A)")
            self.model_auto_button.setStyleSheet("")

    def set_model_b_active(self, active: bool) -> None:
        # Programmatic setter for Model B Auto button
        self.model_b_auto_button.setProperty("active", active)
        if active:
            self.model_b_auto_button.setText("◉ AUTO B ON")
            self.model_b_auto_button.setStyleSheet(f"""
                background-color: rgba{(*self.theme.btn_transition, 0.3)};
                border: 2px solid rgb{self.theme.btn_transition};
                color: rgb{self.theme.text_primary};
                font-weight: 700;
            """)
        else:
            self.model_b_auto_button.setText("Auto Model B (B)")
            self.model_b_auto_button.setStyleSheet("")

    def _toggle_auto(self) -> None:
        if not self._model_enabled:
            return  # 没有模型，不能切换
        self.auto_active = not self.auto_active
        self.set_auto_active(self.auto_active)
        self.model_auto_toggled.emit(self.auto_active)

    def _toggle_model_b_auto(self) -> None:
        # User interaction toggle
        is_checked = self.model_b_auto_button.property("active") or False
        new_state = not is_checked
        # Signal handler will perform logic and likely call set_model_b_active(True/False)
        # But we set visually here for immediate feedback? 
        # Better: let main window drive state via signal to avoid desync
        # However, for responsiveness we emit toggle.
        self.model_b_auto_toggled.emit(new_state)

    def _set_speed(self, speed: float) -> None:
        self._current_speed = speed
        self._update_speed_highlight()
        self.speed_changed.emit(float(speed))

    def _update_speed_highlight(self) -> None:
        """高亮当前速度按钮"""
        p = ui_params.control_panel
        for i, btn in enumerate(self.speed_buttons):
            speed = p.speed_options[i]
            if speed == self._current_speed:
                btn.setStyleSheet(f"""
                    background-color: rgba{(*self.theme.btn_speed, 0.3)};
                    border: 2px solid rgb{self.theme.btn_speed};
                    color: rgb{self.theme.text_primary};
                    font-weight: 700;
                """)
            else:
                btn.setStyleSheet("")
