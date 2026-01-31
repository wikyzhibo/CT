"""
主窗口 - 三栏布局
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QLinearGradient, QPalette, QBrush
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from .theme import ColorTheme
from .ui_params import ui_params
from .viewmodel import PetriViewModel
from .widgets.stats_panel import StatsPanel
from .widgets.center_canvas import CenterCanvas
from .widgets.control_panel import ControlPanel


class PetriMainWindow(QMainWindow):
    """主窗口 - 三栏布局"""

    def __init__(self, viewmodel: PetriViewModel):
        super().__init__()
        self.viewmodel = viewmodel
        self.theme = ColorTheme()
        self._model_handler = None
        p = ui_params.main_window

        self.setWindowTitle("晶圆加工控制台 - PySide6")
        self.setGeometry(p.initial_x, p.initial_y, p.initial_width, p.initial_height)

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(p.central_spacing)

        self.left_panel = StatsPanel(self.theme)
        self.left_panel.setFixedWidth(p.left_panel_width)
        main_layout.addWidget(self.left_panel)

        self.center_canvas = CenterCanvas(self.theme)
        main_layout.addWidget(self.center_canvas, stretch=1)

        self.right_panel = ControlPanel(self.theme)
        self.right_panel.setFixedWidth(p.right_panel_width)
        main_layout.addWidget(self.right_panel)

        self.setCentralWidget(central_widget)
        self._set_background_gradient(central_widget)

        self._connect_signals()
        self._apply_stylesheet()

    def set_model_handler(self, handler) -> None:
        """设置模型动作获取器"""
        self._model_handler = handler

    def _connect_signals(self) -> None:
        self.viewmodel.state_updated.connect(self._on_state_updated)
        self.viewmodel.reward_updated.connect(self.left_panel.update_reward)
        self.viewmodel.step_updated.connect(self.left_panel.update_step)
        self.viewmodel.done_changed.connect(self._on_done_changed)
        self.viewmodel.auto_mode_changed.connect(self.right_panel.set_auto_active)

        self.right_panel.action_clicked.connect(self._on_action_clicked)
        self.right_panel.random_clicked.connect(self._on_random_clicked)
        self.right_panel.model_step_clicked.connect(self._on_model_step_clicked)
        self.right_panel.model_auto_toggled.connect(self.viewmodel.set_auto_mode)
        self.right_panel.reset_clicked.connect(self.viewmodel.reset)
        self.right_panel.speed_changed.connect(self._on_speed_changed)

    def _on_state_updated(self, state) -> None:
        self.center_canvas.update_state(state)
        self.left_panel.update_state(state, self.viewmodel.action_history, self.viewmodel.trend_data)
        self.right_panel.update_actions(state.enabled_actions)

    def _on_done_changed(self, done: bool) -> None:
        if done:
            self.right_panel.set_auto_active(False)

    def _on_action_clicked(self, action_id: int) -> None:
        if action_id == -100:
            action_id = self.viewmodel.adapter.action_space_size
        self.viewmodel.execute_action(action_id)

    def _on_random_clicked(self) -> None:
        actions = self.viewmodel.adapter.get_enabled_actions()
        enabled = [a.action_id for a in actions if a.enabled]
        if not enabled:
            return
        import random
        action = random.choice(enabled)
        self.viewmodel.execute_action(action)

    def _on_model_step_clicked(self) -> None:
        if self._model_handler is None:
            return
        action = self._model_handler()
        if action is not None:
            self.viewmodel.execute_action(action)

    def _on_speed_changed(self, speed: float) -> None:
        self.viewmodel.set_auto_speed(1.0 / max(1.0, speed))

    def _set_background_gradient(self, widget: QWidget) -> None:
        grad = QLinearGradient(0, 0, 0, 1)
        grad.setColorAt(0, self.theme.qcolor(self.theme.bg_deepest))
        grad.setColorAt(1, self.theme.qcolor(self.theme.bg_fog))
        widget.setAutoFillBackground(True)
        pal = widget.palette()
        pal.setBrush(QPalette.Window, QBrush(grad))
        widget.setPalette(pal)

    def _apply_stylesheet(self) -> None:
        t = self.theme

        # 读取可调参数（避免写死）
        sp = ui_params.stats_panel
        cp = ui_params.control_panel

        qss = f"""
        QMainWindow {{
            background-color: rgb{t.bg_deepest};
        }}

        QWidget {{
            color: rgb{t.text_primary};
            background-color: rgb{t.bg_deepest};
        }}

        /* -------- GroupBox -------- */
        QGroupBox {{
            border: 1px solid rgb{t.border_muted};
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px 14px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            font-size: 15px;
            font-weight: bold;
            color: rgb{t.accent_cyan};
        }}

        /* -------- 全局默认文字（别锁死关键组件） -------- */
        QLabel {{
            color: rgb{t.text_primary};
            font-size: 13px;   /* 默认小字，关键的用 ID 覆盖 */
        }}

        /* -------- StatsPanel：KPI 强制覆盖（必须更具体才压得住全局） -------- */
        QLabel#KpiLabel {{
            font-size: {sp.kpi_font_pt}px;
            font-weight: 800;
            color: rgb{t.text_kpi};
        }}
        QLabel#BigLabel {{
            font-size: {sp.label_font_pt}px;
            font-weight: 800;
            color: rgb{t.text_primary};
        }}
        QLabel#DetailLabel {{
            font-size: {sp.reward_detail_font_pt}px;
            color: rgb{t.text_muted};
        }}

        /* -------- Buttons -------- */
        QPushButton {{
            background-color: rgb{t.bg_surface};
            color: rgb{t.text_primary};
            border: 1px solid rgb{t.border};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: {getattr(cp, "button_font_size_px", 13)}px;
        }}
        QPushButton:hover {{
            background-color: rgb{t.bg_elevated};
            border-color: rgb{t.accent_cyan};
        }}
        QPushButton:disabled {{
            color: rgb{t.text_muted};
            border-color: rgb{t.border_muted};
            background-color: rgb{t.bg_deep};
        }}
        QPushButton#TransitionButton:enabled {{
            border-color: rgb{t.accent_cyan};
            font-size: {getattr(cp, "transition_button_font_size_px", getattr(cp, "button_font_size_px", 13))}px;
        }}

        /* -------- TextEdit -------- */
        QTextEdit {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border};
            color: rgb{t.text_secondary};
            font-size: 12px;
        }}

        /* -------- ProgressBar -------- */
        QProgressBar {{
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            text-align: center;
            background-color: rgb{t.bg_deep};
            font-size: 12px;
        }}
        QProgressBar::chunk {{
            background-color: rgb{t.accent_cyan};
            border-radius: 3px;
        }}
        """
        self.setStyleSheet(qss)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_W:
            self._on_action_clicked(self.viewmodel.adapter.action_space_size)
        elif key == Qt.Key_R:
            self._on_random_clicked()
        elif key == Qt.Key_M:
            self._on_model_step_clicked()
        elif key == Qt.Key_A:
            self.viewmodel.set_auto_mode(not self.viewmodel.auto_mode)
        elif key == Qt.Key_Space:
            self.viewmodel.reset()
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
