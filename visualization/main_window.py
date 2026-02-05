"""
主窗口 - 三栏布局 + 无边框 + 圆角阴影

改进：
1) 无模型时禁用 Model Step/Auto Mode
2) Reset 时停止 Auto 模式
3) 无边框窗口支持拖动（智能判断可拖动区域）
4) 圆角 + 外阴影效果
"""

from __future__ import annotations

from pathlib import Path
import json
import os
from datetime import datetime

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import (
    QLinearGradient, QPalette, QBrush, QIcon, QMouseEvent, 
    QPainter, QColor, QPainterPath, QRegion
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QApplication,
    QGraphicsDropShadowEffect, QPushButton, QLineEdit, QTextEdit,
    QScrollBar, QListWidget, QComboBox, QSpinBox, QSlider, QMessageBox
)

from .theme import ColorTheme
from .ui_params import ui_params
from .viewmodel import PetriViewModel
from .widgets.stats_panel import StatsPanel
from .widgets.center_canvas import CenterCanvas
from .widgets.control_panel import ControlPanel


class RoundedContainer(QWidget):
    """圆角容器，用于绘制圆角背景"""
    
    def __init__(self, theme: ColorTheme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.radius = ui_params.main_window.window_corner_radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制圆角背景
        path = QPainterPath()
        path.addRoundedRect(self.rect(), self.radius, self.radius)
        
        # 使用渐变背景
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, self.theme.qcolor(self.theme.bg_deepest))
        grad.setColorAt(1, self.theme.qcolor(self.theme.bg_fog))
        
        painter.fillPath(path, QBrush(grad))
        
        # 绘制边框
        painter.setPen(QColor(*self.theme.border_muted))
        painter.drawPath(path)


class PetriMainWindow(QMainWindow):
    """主窗口 - 三栏布局 + 圆角阴影"""

    # 不可拖动的控件类型
    NON_DRAGGABLE_WIDGETS = (
        QPushButton, QLineEdit, QTextEdit, QScrollBar, 
        QListWidget, QComboBox, QSpinBox, QSlider
    )

    def __init__(self, viewmodel: PetriViewModel):
        super().__init__()
        self.viewmodel = viewmodel
        self.theme = ColorTheme()
        self._model_handler = None
        self._concurrent_model_handler = None  # 双动作模型处理器
        
        # 验证模式状态
        self._verification_active = False
        self._verification_sequence = []
        self._verification_index = 0
        
        # Auto Mode State Tracking ('A' or 'B' or None)
        self._current_auto_mode = None
        
        self._drag_pos: QPoint | None = None
        p = ui_params.main_window

        self.setWindowTitle("晶圆加工控制台")
        
        # 无边框 + 透明背景（为阴影留空）
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # 窗口大小（包含阴影边距）
        margin = p.shadow_margin
        self.setGeometry(
            p.initial_x - margin, 
            p.initial_y - margin, 
            p.initial_width + margin * 2, 
            p.initial_height + margin * 2
        )

        # 外层容器（用于阴影）
        outer_widget = QWidget()
        outer_widget.setStyleSheet("background: transparent;")
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(margin, margin, margin, margin)
        outer_layout.setSpacing(0)
        
        # 圆角内容容器
        self.content_container = RoundedContainer(self.theme)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self.content_container)
        shadow.setBlurRadius(p.shadow_blur_radius)
        shadow.setOffset(p.shadow_offset, p.shadow_offset)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.content_container.setGraphicsEffect(shadow)
        
        outer_layout.addWidget(self.content_container)
        
        # 内容布局
        content_layout = QHBoxLayout(self.content_container)
        content_layout.setSpacing(p.central_spacing)
        content_layout.setContentsMargins(12, 12, 12, 12)

        self.left_panel = StatsPanel(self.theme)
        self.left_panel.setFixedWidth(p.left_panel_width)
        content_layout.addWidget(self.left_panel)

        self.center_canvas = CenterCanvas(self.theme)
        content_layout.addWidget(self.center_canvas, stretch=1)

        self.right_panel = ControlPanel(self.theme)
        self.right_panel.setFixedWidth(p.right_panel_width)
        content_layout.addWidget(self.right_panel)

        self.setCentralWidget(outer_widget)

        self._connect_signals()
        self._apply_stylesheet()
        
        # 初始化时禁用模型按钮
        self._update_model_buttons_state()

    def set_model_handler(self, handler) -> None:
        """设置模型动作获取器（单动作模型）"""
        self._model_handler = handler
        self._concurrent_model_handler = None  # 清除并发处理器
        self.viewmodel.set_agent_callback(handler)
        self._update_model_buttons_state()

    def set_concurrent_model_handler(self, handler) -> None:
        """设置并发模型动作获取器（双动作模型）
        
        handler: 调用时返回 (a1, a2) 的函数
        """
        self._concurrent_model_handler = handler
        self._model_handler = None  # 清除单动作处理器
        
        # 创建 auto 模式的回调包装器
        def concurrent_callback():
            if self._concurrent_model_handler is None:
                return None
            a1, a2 = self._concurrent_model_handler()
            self.viewmodel.execute_concurrent_action(a1, a2)
            return None  # 已经执行，不需要返回动作
        
        self.viewmodel.set_agent_callback(concurrent_callback)
        self._update_model_buttons_state()

    def _update_model_buttons_state(self) -> None:
        """根据模型加载状态更新按钮启用状态"""
        has_model = self._model_handler is not None or self._concurrent_model_handler is not None
        self.right_panel.set_model_enabled(has_model)

    def _connect_signals(self) -> None:
        self.viewmodel.state_updated.connect(self._on_state_updated)
        self.viewmodel.reward_updated.connect(self.left_panel.update_reward)
        self.viewmodel.step_updated.connect(self.left_panel.update_step)
        self.viewmodel.done_changed.connect(self._on_done_changed)
        self.viewmodel.done_changed.connect(self._on_done_changed)
        # self.viewmodel.auto_mode_changed.connect(self.right_panel.set_auto_active)
        self.viewmodel.auto_mode_changed.connect(self._on_auto_mode_changed_ui)

        self.right_panel.action_clicked.connect(self._on_action_clicked)
        # self.right_panel.random_clicked.connect(self._on_random_clicked)  # Removed
        self.right_panel.model_step_clicked.connect(self._on_model_step_clicked)
        self.right_panel.model_auto_toggled.connect(self._on_model_auto_toggled)
        self.right_panel.model_b_auto_toggled.connect(self._on_model_b_auto_toggled)
        self.right_panel.reset_clicked.connect(self._on_reset_clicked)
        self.right_panel.speed_changed.connect(self._on_speed_changed)
        self.right_panel.speed_changed.connect(self._on_speed_changed)
        self.right_panel.verify_planb_clicked.connect(self._on_verify_planb_clicked)
        self.right_panel.gantt_clicked.connect(self._on_gantt_clicked)

    def _on_state_updated(self, state) -> None:
        self.center_canvas.update_state(state)
        self.left_panel.update_state(state, self.viewmodel.action_history, self.viewmodel.trend_data)
        self.right_panel.update_actions(state.enabled_actions)

    def _on_done_changed(self, done: bool) -> None:
        if done:
            self.right_panel.set_auto_active(False)

    def _on_action_clicked(self, action_id: int) -> None:
        if self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        if action_id == -100:
            action_id = self.viewmodel.adapter.action_space_size
        self.viewmodel.execute_action(action_id)

    # _on_random_clicked removed

    def _on_model_step_clicked(self) -> None:
        if self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        # 优先使用并发模型
        if self._concurrent_model_handler is not None:
            a1, a2 = self._concurrent_model_handler()
            self.viewmodel.execute_concurrent_action(a1, a2)
            return
        # 降级到单动作模型
        if self._model_handler is not None:
            action = self._model_handler()
            if action is not None:
                self.viewmodel.execute_action(action)

    def _on_model_auto_toggled(self, enabled: bool) -> None:
        if enabled and self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        has_model = self._model_handler is not None or self._concurrent_model_handler is not None
        if enabled and not has_model:
            return
        self.viewmodel.set_auto_mode(enabled)

    def _on_reset_clicked(self) -> None:
        """先停止 Auto 模式，再重置环境"""
        # 重置验证模式
        self._stop_verification_mode()
        self.right_panel.verify_button.setEnabled(True)

        if self.viewmodel.auto_mode:
            self.viewmodel.set_auto_mode(False)
        self.viewmodel.reset()

    def _on_speed_changed(self, speed: float) -> None:
        self.viewmodel.set_auto_speed(1.0 / max(1.0, speed))

    def _apply_stylesheet(self) -> None:
        t = self.theme
        sp = ui_params.stats_panel
        cp = ui_params.control_panel

        qss = f"""
        QWidget {{
            color: rgb{t.text_primary};
        }}

        /* -------- Buttons -------- */
        QPushButton {{
            background-color: rgb{t.bg_surface};
            color: rgb{t.text_primary};
            border: 1px solid rgb{t.border};
            border-radius: 6px;
            padding: {cp.button_padding_v}px {cp.button_padding_h}px;
            font-size: {cp.button_font_size_px}px;
            min-height: {cp.button_min_height}px;
            font-weight: 500;
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
            font-size: {cp.transition_button_font_size_px}px;
        }}

        /* -------- TextEdit -------- */
        QTextEdit {{
            background-color: rgb{t.bg_deep};
            border: 1px solid rgb{t.border};
            border-radius: 4px;
            color: rgb{t.text_secondary};
            font-family: "{sp.font_family}";
            font-size: {sp.release_font_pt}pt;
            padding: 8px;
            selection-background-color: rgb{t.accent_cyan};
        }}

        /* -------- ProgressBar -------- */
        QProgressBar {{
            border: 1px solid rgb{t.border_muted};
            border-radius: 4px;
            text-align: center;
            background-color: rgb{t.bg_deep};
            font-size: {sp.label_font_pt}pt;
            min-height: {sp.progress_bar_height}px;
            font-weight: 600;
        }}
        QProgressBar::chunk {{
            background-color: rgb{t.accent_cyan};
            border-radius: 3px;
        }}
        
        /* -------- ScrollBar -------- */
        QScrollBar:vertical {{
            background-color: rgb{t.bg_deep};
            width: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background-color: rgb{t.bg_elevated};
            border-radius: 5px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: rgb{t.border};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """
        self.setStyleSheet(qss)

    # ===== 无边框窗口拖动（智能判断） =====
    
    def _is_draggable_pos(self, pos: QPoint) -> bool:
        """判断点击位置是否可拖动
        
        规则：
        1. 必须在内容容器内
        2. 不能在交互控件上
        """
        # 转换到全局坐标再到内容容器坐标
        global_pos = self.mapToGlobal(pos)
        container_pos = self.content_container.mapFromGlobal(global_pos)
        
        # 检查是否在容器内
        if not self.content_container.rect().contains(container_pos):
            return False
        
        # 检查点击位置的控件
        child = self.childAt(pos)
        if child is None:
            return True  # 空白区域可拖动
        
        # 遍历父控件检查是否是交互控件
        widget = child
        while widget is not None and widget != self:
            if isinstance(widget, self.NON_DRAGGABLE_WIDGETS):
                return False
            # 检查 QGraphicsView（画布也不拖动）
            if widget.__class__.__name__ == 'QGraphicsView':
                return False
            widget = widget.parent()
        
        return True

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_draggable_pos(event.position().toPoint()):
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_W:
            self._on_action_clicked(self.viewmodel.adapter.action_space_size)
        # elif key == Qt.Key.Key_R: self._on_random_clicked() # Removed
        elif key == Qt.Key.Key_M:
            self._on_model_step_clicked()
        elif key == Qt.Key.Key_A:
            if self._model_handler is not None:
                self.viewmodel.set_auto_mode(not self.viewmodel.auto_mode)
        elif key == Qt.Key.Key_B:
            self.right_panel.model_b_auto_button.click()
        elif key == Qt.Key.Key_Space:
            self._on_reset_clicked()
        elif key == Qt.Key.Key_V:
            self._on_verify_planb_clicked()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _ensure_verification_initialized(self) -> bool:
        """Ensure verification sequence is loaded and ready"""
        if self._verification_active:
            return True
        
        try:
            seq_path = Path("solutions/Td_petri/planB_sequence.json")
            if not seq_path.exists():
                    QMessageBox.warning(self, "Error", f"{seq_path} not found! Run generation script first.")
                    return False
            
            with open(seq_path, "r") as f:
                self._verification_sequence = json.load(f)
            
            if not self._verification_sequence:
                QMessageBox.warning(self, "Error", "Sequence is empty!")
                return False
            
            # Reset environment implicitly? User might want to run from current state.
            # But usually verification assumes initial state.
            # Previous logic called _on_reset_clicked().
            self._on_reset_clicked()
            
            self._verification_active = True
            self._verification_index = 0
            
            self.right_panel.verify_button.setText("Next Step")
            self.right_panel.verify_button.setStyleSheet(f"border-color: rgb{self.theme.accent_cyan}; background-color: rgba{(*self.theme.accent_cyan, 0.2)};")
            print(f"Verification started. Sequence length: {len(self._verification_sequence)}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start verification: {e}")
            self._stop_verification_mode()
            return False

    def _get_verification_step_action(self):
        """Get next action from verification sequence"""
        if not self._ensure_verification_initialized():
             return None
             
        if self._verification_index >= len(self._verification_sequence):
             QMessageBox.information(self, "Done", "Verification Sequence Completed!")
             self.viewmodel.set_auto_mode(False) # Stop auto if running
             return None
             
        item = self._verification_sequence[self._verification_index]
        actions = item.get("actions", [None, None])
        tm2_name, tm3_name = actions
        
        try:
            all_transitions = self.viewmodel.adapter.net.id2t_name
            
            if tm2_name and tm2_name != "WAIT":
                a1 = all_transitions.index(tm2_name)
            else:
                a1 = -1
                
            if tm3_name and tm3_name != "WAIT":
                a2 = all_transitions.index(tm3_name)
            else:
                a2 = -1
            
            self._verification_index += 1
            return (a1, a2)
            
        except ValueError as e:
            print(f"Error mapping transition: {e}")
            self.viewmodel.set_auto_mode(False)
            return None

    def _on_verify_planb_clicked(self) -> None:
        """Handle Verify PlanB button click (Manual Step)"""
        if self._current_auto_mode == 'B' and self.viewmodel.auto_mode:
             # If auto is running, maybe pause?
             self.viewmodel.set_auto_mode(False)
             return

        action_pair = self._get_verification_step_action()
        if action_pair:
             self.viewmodel.execute_concurrent_action(*action_pair)

    def _on_model_b_auto_toggled(self, enabled: bool) -> None:
        """Handle Auto Model B toggle"""
        if enabled:
            # Switch to B-mode
            self._current_auto_mode = 'B'
            
            # Use a lambda wrapper to call our verification provider
            # The provider returns (a1, a2) or None
            def b_auto_cb():
                res = self._get_verification_step_action()
                if res:
                    self.viewmodel.execute_concurrent_action(*res)
                return None # Wrapper executed action, return None to main loop
            
            self.viewmodel.set_agent_callback(b_auto_cb)
            self.viewmodel.set_auto_mode(True)
        else:
            if self._current_auto_mode == 'B':
                self.viewmodel.set_auto_mode(False)
                self._current_auto_mode = None

    def _on_model_auto_toggled(self, enabled: bool) -> None:
        """Handle Auto Mode A toggle"""
        if enabled:
            # Switch to A-mode
            self._current_auto_mode = 'A'
            
            # Restoration logic for A:
            if self._concurrent_model_handler:
                 self.set_concurrent_model_handler(self._concurrent_model_handler) # Re-register callback
            elif self._model_handler:
                 self.set_model_handler(self._model_handler)
            
            self.viewmodel.set_auto_mode(True)
        else:
             if self._current_auto_mode == 'A':
                self.viewmodel.set_auto_mode(False)
                self._current_auto_mode = None

    def _on_auto_mode_changed_ui(self, active: bool) -> None:
        """Update UI buttons based on auto mode state"""
        # If viewmodel stops auto (active=False), turn off buttons
        if not active:
            self.right_panel.set_auto_active(False)
            self.right_panel.set_model_b_active(False)
            self._current_auto_mode = None
        else:
            # If active, ensure correct button is lit
            if self._current_auto_mode == 'B':
                self.right_panel.set_model_b_active(True)
                self.right_panel.set_auto_active(False)
            elif self._current_auto_mode == 'A':
                self.right_panel.set_auto_active(True)
                self.right_panel.set_model_b_active(False)

    def _stop_verification_mode(self) -> None:
        """Exit verification mode and reset button state"""
        if self._verification_active:
            self._verification_active = False
            self.right_panel.verify_button.setText("Model B Step (V)")
            self.right_panel.verify_button.setStyleSheet("") 

    def _on_gantt_clicked(self) -> None:
        """Handle Draw Gantt button click"""
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"continuous_gantt_{timestamp}.png"
            output_path = results_dir / filename
            
            success = self.viewmodel.render_gantt(str(output_path))
            
            if success:
                print(f"✓ Gantt chart saved to: {output_path}")
                # Optional: Show success message in status bar or small popup if desired
                # QMessageBox.information(self, "Success", f"Gantt chart saved to:\n{output_path}")
            else:
                print("✗ Failed to generate Gantt chart")
                QMessageBox.warning(self, "Error", "Failed to generate Gantt chart. Check console for details.")
                
        except Exception as e:
            print(f"✗ Error generating Gantt chart: {e}")
            QMessageBox.critical(self, "Error", f"Error generating Gantt chart:\n{e}") 

