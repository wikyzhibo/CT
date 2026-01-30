"""
右侧控制面板
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
    random_clicked = Signal()
    model_step_clicked = Signal()
    model_auto_toggled = Signal(bool)
    reset_clicked = Signal()
    speed_changed = Signal(float)

    def __init__(self, theme: ColorTheme, parent=None) -> None:
        super().__init__(parent)
        p = ui_params.control_panel

        self.setStyleSheet(f"""
        QPushButton {{
            font-size: {p.button_font_size_px}px;
            padding: {p.button_padding_v}px {p.button_padding_h}px;
            min-height: {p.button_min_height}px;
        }}

        QPushButton#TransitionButton {{
            font-size: {p.transition_button_font_size_px}px;
        }}
        """)

        self.theme = theme

        self.buttons: List[QPushButton] = []
        self.speed_buttons: List[QPushButton] = []
        self.auto_active = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        self.title = QLabel("TRANSITIONS")
        self.title.setStyleSheet(f"font-size: {p.title_font_size_px}px; font-weight: bold; color: rgb{self.theme.accent_cyan};")
        layout.addWidget(self.title)

        self.transition_group = QVBoxLayout()
        layout.addLayout(self.transition_group)

        layout.addSpacing(p.spacing_after_transitions)
        control_title = QLabel("CONTROL")
        control_title.setStyleSheet(f"font-size: {p.title_font_size_px}px; font-weight: bold; color: rgb{self.theme.accent_cyan};")
        layout.addWidget(control_title)
        layout.addSpacing(p.spacing_after_control_title)

        self.wait_button = QPushButton("WAIT (W)")
        self.wait_button.clicked.connect(lambda: self.action_clicked.emit(-100))
        layout.addWidget(self.wait_button)

        self.random_button = QPushButton("Random (R)")
        self.random_button.clicked.connect(self.random_clicked.emit)
        layout.addWidget(self.random_button)

        self.model_button = QPushButton("Model(one step) (M)")
        self.model_button.clicked.connect(self.model_step_clicked.emit)
        layout.addWidget(self.model_button)

        self.model_auto_button = QPushButton("Model(auto) (A)")
        self.model_auto_button.clicked.connect(self._toggle_auto)
        layout.addWidget(self.model_auto_button)

        layout.addSpacing(p.spacing_before_speed)

        speed_layout = QHBoxLayout()
        for speed in p.speed_options:
            btn = QPushButton(f"{speed}x")
            btn.clicked.connect(lambda _=False, s=speed: self._set_speed(s))
            speed_layout.addWidget(btn)
            self.speed_buttons.append(btn)
        layout.addLayout(speed_layout)

        layout.addSpacing(p.spacing_before_reset)

        self.reset_button = QPushButton("Reset (Space)")
        self.reset_button.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_button)

    def update_actions(self, actions: List[ActionInfo]) -> None:
        # 清理旧的变迁按钮
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
            tip = (action.description or "当前条件不满足") if not action.enabled else "点击执行"
            btn.setToolTip(tip)
            btn.setObjectName("TransitionButton")
            btn.clicked.connect(lambda _=False, a=action.action_id: self.action_clicked.emit(a))
            self.transition_group.addWidget(btn)
            self.buttons.append(btn)

    def set_auto_active(self, active: bool) -> None:
        self.auto_active = active
        text = "Model(auto) ON" if active else "Model(auto) (A)"
        self.model_auto_button.setText(text)

    def _toggle_auto(self) -> None:
        self.auto_active = not self.auto_active
        self.set_auto_active(self.auto_active)
        self.model_auto_toggled.emit(self.auto_active)

    def _set_speed(self, speed: float) -> None:
        self.speed_changed.emit(float(speed))
