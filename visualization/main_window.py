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
    QAction, QActionGroup, QLinearGradient, QPalette, QBrush, QIcon, QMouseEvent,
    QPainter, QColor, QPainterPath, QRegion
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QApplication,
    QPushButton, QLineEdit, QTextEdit,
    QScrollBar, QListWidget, QComboBox, QSpinBox, QSlider, QMessageBox,
    QFileDialog, QLabel, QDialog, QDialogButtonBox, QFormLayout, QFrame,
)

from .algorithm_interface import ActionInfo
from .theme import ColorTheme
from .widgets.route_path_display import format_route_path_html
from .widgets.transition_labels import CASCADE_ROUTE_OPTIONS
from .ui_params import ui_params
from .viewmodel import PetriViewModel
from .widgets.stats_panel import StatsPanel
from .widgets.center_canvas import CenterCanvas
from .widgets.control_panel import ControlPanel
from results.paths import action_sequence_path, gantt_output_path

class RoundedContainer(QWidget):
    """主内容容器，用于绘制背景"""
    
    def __init__(self, theme: ColorTheme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.radius = ui_params.main_window.window_corner_radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制直角背景
        path = QPainterPath()
        path.addRect(self.rect())
        
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

    def __init__(self, viewmodel: PetriViewModel, *, debug: bool = False):
        super().__init__()
        self.viewmodel = viewmodel
        self.theme = ColorTheme()
        self._model_handler = None
        self._concurrent_model_handler = None  # 三动作模型处理器
        
        # 验证模式状态
        self._verification_active = False
        self._verification_sequence = []
        self._verification_index = 0
        self._verification_env_overrides: dict | None = None
        
        # Auto Mode State Tracking ('A' or 'B' or None)
        self._current_auto_mode = None
        self._device_mode = "single"
        self._robot_capacity = 1
        self._wafer_count_route1: int | None = None
        self._wafer_count_route2: int | None = None
        self._cleaning_options: dict[str, bool] = {
            "idle_80": False,
            "switch_process": False,
            "pm_5_wafers": False,
        }
        self._model_path_override: Path | None = None
        self._model_apply_callback = None
        self._action_sequence_default_dir = action_sequence_path("tmp").parent
        self._action_sequence_path: Path | None = None
        self._adapter_factory = None
        self._cascade_route_name: str | None = None
        self._concurrent_runtime = False

        self._drag_pos: QPoint | None = None
        p = ui_params.main_window

        self.setWindowTitle("晶圆加工控制台")
        
        # 使用系统窗口边框，保留系统最小化/最大化/关闭按钮
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setGeometry(p.initial_x, p.initial_y, p.initial_width, p.initial_height)

        # 内容容器
        self.content_container = RoundedContainer(self.theme)
        content_layout = QHBoxLayout(self.content_container)
        content_layout.setSpacing(p.central_spacing)
        content_layout.setContentsMargins(12, 12, 12, 12)

        self.left_panel = StatsPanel(self.theme)
        self.left_panel.setFixedWidth(p.left_panel_width)
        content_layout.addWidget(self.left_panel)

        self._center_column = QWidget()
        center_col_layout = QVBoxLayout(self._center_column)
        center_col_layout.setContentsMargins(0, 0, 0, 0)
        center_col_layout.setSpacing(8)

        self.route_banner_frame = QFrame()
        self.route_banner_frame.setObjectName("RouteBannerFrame")
        route_inner = QVBoxLayout(self.route_banner_frame)
        route_inner.setContentsMargins(10, 8, 10, 8)
        self.route_path_label = QLabel("")
        self.route_path_label.setObjectName("RouteBannerPath")
        self.route_path_label.setWordWrap(True)
        self.route_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        route_inner.addWidget(self.route_path_label)

        self.center_canvas = CenterCanvas(self.theme)
        self.center_canvas.set_device_mode(self._device_mode)
        self.center_canvas.set_robot_capacity(self._robot_capacity)

        center_col_layout.addWidget(self.route_banner_frame)
        center_col_layout.addWidget(self.center_canvas, stretch=1)
        content_layout.addWidget(self._center_column, stretch=1)

        self.right_panel = ControlPanel(self.theme)
        self.right_panel.set_debug_mode(debug)
        self.right_panel.setFixedWidth(p.right_panel_width)
        content_layout.addWidget(self.right_panel)

        self.setCentralWidget(self.content_container)
        self._create_menu_bar()

        self._connect_signals()
        self._apply_stylesheet()
        self._action_config_cascade_route.setEnabled(self._device_mode == "cascade")
        self._refresh_status_message()
        
        # 初始化时禁用模型按钮
        self._update_model_buttons_state()

    def set_model_handler(self, handler) -> None:
        """设置模型动作获取器（单动作模型）"""
        self._model_handler = handler
        self._concurrent_model_handler = None  # 清除并发处理器
        self.viewmodel.set_agent_callback(handler)
        self._update_model_buttons_state()

    def set_concurrent_model_handler(self, handler) -> None:
        """设置并发模型动作获取器。"""
        self._concurrent_model_handler = handler
        self._model_handler = None  # 清除单动作处理器
        
        # 创建 auto 模式的回调包装器
        def concurrent_callback():
            if self._concurrent_model_handler is None:
                return None
            actions = tuple(self._concurrent_model_handler())
            self.viewmodel.execute_concurrent_action(*actions)
            return None  # 已经执行，不需要返回动作
        
        self.viewmodel.set_agent_callback(concurrent_callback)
        self._update_model_buttons_state()

    def _update_model_buttons_state(self) -> None:
        """根据模型加载状态更新按钮启用状态"""
        has_model = self._model_handler is not None or self._concurrent_model_handler is not None
        self.right_panel.set_model_enabled(has_model)

    def _create_menu_bar(self) -> None:
        """创建菜单栏：设备/配置/回放。"""
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        # 设备菜单（仅 UI 占位）
        device_menu = menu_bar.addMenu("设备")
        device_group = QActionGroup(self)
        device_group.setExclusive(True)
        self._action_device_cascade = QAction("级联设备", self, checkable=True)
        self._action_device_single = QAction("单设备", self, checkable=True)
        self._action_device_single.setChecked(True)
        device_group.addAction(self._action_device_cascade)
        device_group.addAction(self._action_device_single)
        device_menu.addAction(self._action_device_cascade)
        device_menu.addAction(self._action_device_single)
        self._action_device_cascade.triggered.connect(lambda: self._set_device_mode("cascade"))
        self._action_device_single.triggered.connect(lambda: self._set_device_mode("single"))

        # 配置菜单
        config_menu = menu_bar.addMenu("配置")
        self._action_config_wafer = QAction("设置晶圆数量", self)
        arm_mode_menu = config_menu.addMenu("机械手模式")
        arm_mode_group = QActionGroup(self)
        arm_mode_group.setExclusive(True)
        self._action_arm_single = QAction("Single Arm", self, checkable=True)
        self._action_arm_dual = QAction("Dual Arm", self, checkable=True)
        self._action_arm_single.setChecked(True)
        arm_mode_group.addAction(self._action_arm_single)
        arm_mode_group.addAction(self._action_arm_dual)
        arm_mode_menu.addAction(self._action_arm_single)
        arm_mode_menu.addAction(self._action_arm_dual)
        config_menu.addAction(self._action_config_wafer)
        self._action_config_wafer.triggered.connect(self._set_wafer_count)
        self._action_config_cascade_route = QAction("路径", self)
        config_menu.addAction(self._action_config_cascade_route)
        self._action_config_cascade_route.triggered.connect(self._open_cascade_route_dialog)
        self._action_arm_single.triggered.connect(lambda: self._set_robot_capacity(1))
        self._action_arm_dual.triggered.connect(lambda: self._set_robot_capacity(2))
        cleaning_menu = config_menu.addMenu("清洁")
        self._action_clean_idle_80 = QAction("闲置时间超过80s (清洗时间30s)", self, checkable=True)
        self._action_clean_switch = QAction("工艺切换 (清洗时间200s)", self, checkable=True)
        self._action_clean_pm5 = QAction("PM加工5片晶圆 (清洗时间300s)", self, checkable=True)
        cleaning_menu.addAction(self._action_clean_idle_80)
        cleaning_menu.addAction(self._action_clean_switch)
        cleaning_menu.addAction(self._action_clean_pm5)
        self._action_clean_idle_80.toggled.connect(lambda checked: self._on_cleaning_option_toggled("idle_80", checked))
        self._action_clean_switch.toggled.connect(lambda checked: self._on_cleaning_option_toggled("switch_process", checked))
        self._action_clean_pm5.toggled.connect(lambda checked: self._on_cleaning_option_toggled("pm_5_wafers", checked))

        # 回放菜单（动作序列 JSON）
        replay_menu = menu_bar.addMenu("回放")
        self._action_pick_model = QAction("选择模型文件", self)
        self._action_pick_sequence = QAction("选择动作序列 JSON", self)
        self._action_reset_sequence = QAction("恢复默认序列路径", self)
        replay_menu.addAction(self._action_pick_model)
        replay_menu.addAction(self._action_pick_sequence)
        replay_menu.addAction(self._action_reset_sequence)
        self._action_pick_model.triggered.connect(self._pick_model_file)
        self._action_pick_sequence.triggered.connect(self._pick_action_sequence_json)
        self._action_reset_sequence.triggered.connect(self._reset_default_action_sequence)

    def _pick_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            str(Path.cwd()),
            "Model Files (*.pt *.pth);;All Files (*)",
        )
        if not path:
            return
        self._model_path_override = Path(path)
        # 立即按当前设备模式尝试加载模型
        if self._model_apply_callback is not None:
            ok, msg = self._model_apply_callback(str(self._model_path_override), self._device_mode)
            if not ok:
                QMessageBox.warning(self, "模型加载失败", msg)
        self._refresh_status_message()

    def _clear_model_file(self) -> None:
        self._model_path_override = None
        self._refresh_status_message()

    def _set_device_mode(self, mode: str) -> None:
        if mode not in {"cascade", "single"}:
            return
        if mode == self._device_mode:
            return
        old_mode = self._device_mode
        # 运行时切换底层 env/adapter
        if self._adapter_factory is not None:
            try:
                new_adapter = self._adapter_factory(mode, self._robot_capacity)
                self.apply_runtime_adapter(
                    new_adapter,
                    mode,
                    concurrent_runtime=False,
                    reset=True,
                )
            except Exception as e:
                # 切换失败时保持原模式与菜单勾选，避免 UI 与后端状态错位
                self._action_device_cascade.blockSignals(True)
                self._action_device_single.blockSignals(True)
                self._action_device_cascade.setChecked(old_mode == "cascade")
                self._action_device_single.setChecked(old_mode == "single")
                self._action_device_cascade.blockSignals(False)
                self._action_device_single.blockSignals(False)
                QMessageBox.warning(self, "设备切换失败", f"无法切换到 {mode}: {e}")
                self._refresh_status_message()
                return
        else:
            self._device_mode = mode
            self._concurrent_runtime = False
            self.center_canvas.set_device_mode(mode)
            self._action_config_cascade_route.setEnabled(mode == "cascade")
            self._update_route_banner_text()
        self._model_handler = None
        self._concurrent_model_handler = None
        self._update_model_buttons_state()
        # 兼容策略：若已选择模型，则按新模式重载；成功保留，失败提示并失效
        if self._model_path_override is not None and self._model_apply_callback is not None:
            ok, msg = self._model_apply_callback(str(self._model_path_override), mode)
            if not ok:
                self.set_model_handler(None)
                QMessageBox.warning(self, "模型与设备模式不兼容", msg)
        self._refresh_status_message()

    def set_adapter_factory(self, factory) -> None:
        """注入设备模式 -> adapter 的构造函数。"""
        self._adapter_factory = factory

    def set_model_apply_callback(self, callback) -> None:
        """注入模型应用器：callback(model_path, device_mode) -> (ok, message)。"""
        self._model_apply_callback = callback

    def apply_runtime_adapter(
        self,
        adapter,
        mode: str,
        *,
        concurrent_runtime: bool,
        reset: bool = True,
    ) -> None:
        self.viewmodel.replace_adapter(adapter, reset=reset)
        self._device_mode = mode
        self._concurrent_runtime = bool(concurrent_runtime)
        self.center_canvas.set_device_mode(mode)
        self._action_device_cascade.blockSignals(True)
        self._action_device_single.blockSignals(True)
        self._action_device_cascade.setChecked(mode == "cascade")
        self._action_device_single.setChecked(mode == "single")
        self._action_device_cascade.blockSignals(False)
        self._action_device_single.blockSignals(False)
        self._action_config_cascade_route.setEnabled(mode == "cascade")
        self._update_route_banner_text()
        self._refresh_status_message()

    def _build_cascade_transition_actions(self) -> tuple[list[ActionInfo], list[str] | None]:
        """级联：优先按 env mask 构建；并发适配器则回退到 adapter 输出。"""
        adapter = self.viewmodel.adapter
        env = adapter.env
        net = adapter.net
        if hasattr(env, "_mask"):
            t_count = int(net.T)
            mask = env._mask()
            actions: list[ActionInfo] = []
            for i in range(t_count):
                name = str(net.id2t_name[i])
                ok = bool(mask[i])
                actions.append(
                    ActionInfo(
                        action_id=i,
                        action_name=name,
                        enabled=ok,
                        description="" if ok else "当前条件不满足",
                    )
                )
        else:
            actions = [
                action
                for action in adapter.get_enabled_actions()
                if action.action_id >= 0 and not str(action.action_name).upper().startswith("WAIT")
            ]
        ut = getattr(net, "_u_targets", None) or {}
        raw = ut.get("LLD", [])
        lld_targets = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else None
        return actions, lld_targets

    def _route_path_plain_from_net(self) -> str:
        """仅返回配置中的 path 字符串，不重复展示路线键名（如 1-6）。"""
        net = getattr(self.viewmodel.adapter, "net", None)
        if net is None:
            return ""
        name = getattr(net, "single_route_name", None) or ""
        cfg = getattr(net, "single_route_config", None) or {}
        routes = cfg.get("routes") or {}
        if name and isinstance(routes, dict) and name in routes:
            entry = routes[name]
            if isinstance(entry, dict):
                path = entry.get("path")
                if path:
                    return str(path).strip()
        return ""

    def _update_route_banner_text(self) -> None:
        if self._device_mode != "cascade":
            self.route_banner_frame.hide()
            return
        self.route_banner_frame.show()
        plain = self._route_path_plain_from_net()
        fallback = "（无路径描述：非配置驱动拓扑或缺少 routes.path）"
        ff = ui_params.stats_panel.font_family
        if plain:
            self.route_path_label.setTextFormat(Qt.TextFormat.RichText)
            self.route_path_label.setText(
                format_route_path_html(plain, self.theme, font_family=ff, font_size_px=15)
            )
            return
        self.route_path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.route_path_label.setText(fallback)

    def _open_cascade_route_dialog(self) -> None:
        if self._device_mode != "cascade":
            QMessageBox.information(self, "路径", "请先切换到「级联设备」模式后再选择路径。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("级联路径")
        form = QFormLayout(dialog)
        combo = QComboBox(dialog)
        for n in CASCADE_ROUTE_OPTIONS:
            combo.addItem(n)
        current = self._cascade_route_name or getattr(self.viewmodel.adapter.net, "single_route_name", "") or ""
        idx = combo.findText(str(current))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        form.addRow("路线键名", combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog
        )
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._cascade_route_name = combo.currentText()
        self._rebuild_adapter_preserve_device("cascade")

    def _rebuild_adapter_preserve_device(self, mode: str) -> None:
        if self._adapter_factory is None:
            return
        try:
            overrides = {"runtime_mode": "concurrent"} if self._concurrent_runtime else None
            new_adapter = self._adapter_factory(mode, self._robot_capacity, overrides)
            self.apply_runtime_adapter(
                new_adapter,
                mode,
                concurrent_runtime=self._concurrent_runtime,
                reset=True,
            )
        except Exception as e:
            QMessageBox.warning(self, "路径切换失败", str(e))
            return
        self._model_handler = None
        self._concurrent_model_handler = None
        self._update_model_buttons_state()
        if self._model_path_override is not None and self._model_apply_callback is not None:
            ok, msg = self._model_apply_callback(str(self._model_path_override), mode)
            if not ok:
                self.set_model_handler(None)
                QMessageBox.warning(self, "模型与当前拓扑不兼容", msg)
        self._update_route_banner_text()
        self._refresh_status_message()

    def _set_wafer_count(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("晶圆数量")
        form = QFormLayout(dialog)
        form.addRow(QLabel("设置两种晶圆数量（当前仅 UI 占位）"))

        route1_spin = QSpinBox(dialog)
        route1_spin.setRange(0, 2000)
        route1_spin.setValue(self._wafer_count_route1 if self._wafer_count_route1 is not None else 8)
        route2_spin = QSpinBox(dialog)
        route2_spin.setRange(0, 2000)
        route2_spin.setValue(self._wafer_count_route2 if self._wafer_count_route2 is not None else 10)

        form.addRow("路线1晶圆数量", route1_spin)
        form.addRow("路线2晶圆数量", route2_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._wafer_count_route1 = int(route1_spin.value())
        self._wafer_count_route2 = int(route2_spin.value())
        self._refresh_status_message()

    def _set_robot_capacity(self, capacity: int) -> None:
        self._robot_capacity = 2 if int(capacity) == 2 else 1
        self.center_canvas.set_robot_capacity(self._robot_capacity)
        if self._adapter_factory is not None:
            try:
                overrides = {"runtime_mode": "concurrent"} if self._concurrent_runtime else None
                new_adapter = self._adapter_factory(self._device_mode, self._robot_capacity, overrides)
                self.apply_runtime_adapter(
                    new_adapter,
                    self._device_mode,
                    concurrent_runtime=self._concurrent_runtime,
                    reset=True,
                )
                self._model_handler = None
                self._concurrent_model_handler = None
                self._update_model_buttons_state()
                if self._model_path_override is not None and self._model_apply_callback is not None:
                    ok, msg = self._model_apply_callback(str(self._model_path_override), self._device_mode)
                    if not ok:
                        self.set_model_handler(None)
                        QMessageBox.warning(self, "模型与机械手模式不兼容", msg)
            except Exception as e:
                QMessageBox.warning(self, "机械手模式切换失败", f"无法切换到容量 {self._robot_capacity}: {e}")
        self._refresh_status_message()

    def _pick_action_sequence_json(self) -> None:
        if self._action_sequence_path is not None:
            start_dir = self._action_sequence_path.parent
        else:
            start_dir = self._action_sequence_default_dir
        if not start_dir.exists():
            start_dir = Path.cwd()
        path, _ = QFileDialog.getOpenFileName(self, "选择动作序列 JSON", str(start_dir), "JSON Files (*.json)")
        if not path:
            return
        self._action_sequence_path = Path(path)
        self._verification_env_overrides = None
        self._refresh_status_message()

    def _reset_default_action_sequence(self) -> None:
        self._action_sequence_path = None
        self._verification_env_overrides = None
        self._refresh_status_message()

    def _refresh_status_message(self) -> None:
        mode_text = "级联设备" if self._device_mode == "cascade" else "单设备"
        runtime_text = "并发三动作" if self._concurrent_runtime else "单动作"
        if self._wafer_count_route1 is None or self._wafer_count_route2 is None:
            wafers_text = "未设置"
        else:
            wafers_text = f"R1={self._wafer_count_route1}, R2={self._wafer_count_route2}"
        arm_mode_text = "Dual Arm" if self._robot_capacity == 2 else "Single Arm"
        model_text = str(self._model_path_override) if self._model_path_override else "未选择"
        if self._action_sequence_path is None:
            seq_text = f"未选择（默认目录: {self._action_sequence_default_dir}）"
        else:
            seq_text = str(self._action_sequence_path)
        clean_labels = []
        if self._cleaning_options["idle_80"]:
            clean_labels.append("idle80/30s")
        if self._cleaning_options["switch_process"]:
            clean_labels.append("switch/200s")
        if self._cleaning_options["pm_5_wafers"]:
            clean_labels.append("pm5/300s")
        clean_text = ",".join(clean_labels) if clean_labels else "关闭"
        self.statusBar().showMessage(
            f"设备: {mode_text} | 运行时: {runtime_text} | 机械手模式: {arm_mode_text} | 晶圆数量: {wafers_text}（仅UI占位） | 清洁: {clean_text} | 模型: {model_text} | 动作序列: {seq_text}"
        )

    def _on_cleaning_option_toggled(self, key: str, checked: bool) -> None:
        if key not in self._cleaning_options:
            return
        self._cleaning_options[key] = bool(checked)
        self._refresh_status_message()

    def _connect_signals(self) -> None:
        self.viewmodel.state_updated.connect(self._on_state_updated)
        self.viewmodel.reward_updated.connect(self.left_panel.update_reward)
        self.viewmodel.step_updated.connect(self.left_panel.update_step)
        self.viewmodel.done_changed.connect(self._on_done_changed)
        self.viewmodel.done_changed.connect(self._on_done_changed)
        # self.viewmodel.auto_mode_changed.connect(self.right_panel.set_auto_active)
        self.viewmodel.auto_mode_changed.connect(self._on_auto_mode_changed_ui)

        self.right_panel.action_clicked.connect(self._on_action_clicked)
        self.right_panel.wait_clicked.connect(self._on_wait_clicked)
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
        if (self._wafer_count_route1 is None or self._wafer_count_route2 is None) and state.total_wafers > 0:
            total = int(state.total_wafers)
            self._wafer_count_route1 = total // 2
            self._wafer_count_route2 = total - self._wafer_count_route1
            self._refresh_status_message()
        self.center_canvas.update_state(state)
        self.left_panel.update_state(state, self.viewmodel.action_history, self.viewmodel.trend_data)
        if self._device_mode == "cascade":
            sn = getattr(self.viewmodel.adapter.net, "single_route_name", None)
            if sn:
                self._cascade_route_name = str(sn)
            cascade_actions, lld_targets = self._build_cascade_transition_actions()
            self.right_panel.update_actions(
                cascade_actions, device_mode="cascade", lld_targets=lld_targets
            )
            if hasattr(getattr(self.viewmodel.adapter, "env", None), "tm2_wait_action"):
                self.right_panel.set_wait_durations([5], enabled_map={5: True})
        else:
            actions_for_panel = state.enabled_actions
            # 单设备模式：展示“单设备全集变迁”，并通过 enabled 状态区分可点/禁用。
            actions_for_panel = [
                a for a in state.enabled_actions
                if a.action_id >= 0 and not str(a.action_name).upper().startswith("WAIT")
            ]
            self.right_panel.update_actions(actions_for_panel, device_mode="single")
        self._update_route_banner_text()
        if self._device_mode == "single" and hasattr(self.viewmodel.adapter, "env"):
            wait_durations = list(getattr(self.viewmodel.adapter.env, "wait_durations", [5]))
            wait_enabled_map = {}
            for action in state.enabled_actions:
                action_name = str(getattr(action, "action_name", ""))
                if not action_name.upper().startswith("WAIT_"):
                    continue
                wait_text = action_name.removeprefix("WAIT_").removesuffix("s")
                try:
                    wait_duration = int(wait_text)
                except ValueError:
                    continue
                wait_enabled_map[wait_duration] = bool(getattr(action, "enabled", False))
            self.right_panel.set_wait_durations(wait_durations, enabled_map=wait_enabled_map)

    def _on_done_changed(self, done: bool) -> None:
        if done:
            self.right_panel.set_auto_active(False)

    def _on_action_clicked(self, action_id: int) -> None:
        if self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        self.viewmodel.execute_action(action_id)

    def _on_wait_clicked(self, wait_duration: int) -> None:
        if self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        action_id = self.viewmodel.adapter.action_space_size
        if hasattr(self.viewmodel.adapter, "env"):
            env = self.viewmodel.adapter.env
            for idx in range(int(getattr(env, "n_actions", 0))):
                parse_fn = getattr(env, "parse_wait_action", None)
                if parse_fn is None:
                    continue
                parsed = parse_fn(idx)
                if parsed == int(wait_duration):
                    action_id = int(idx)
                    break
        self.viewmodel.execute_action(int(action_id))

    # _on_random_clicked removed

    def _on_model_step_clicked(self) -> None:
        if self._verification_active:
            self._stop_verification_mode()
            self.right_panel.verify_button.setEnabled(False)

        # 优先使用并发模型
        if self._concurrent_model_handler is not None:
            actions = tuple(self._concurrent_model_handler())
            self.viewmodel.execute_concurrent_action(*actions)
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

        QMenuBar {{
            background-color: rgb{t.bg_surface};
            border: 1px solid rgb{t.border_muted};
            font-size: 16px;
            font-weight: 600;
            min-height: 32px;
            padding: 2px 8px;
        }}
        QMenuBar::item {{
            background-color: rgb{t.bg_surface};
            padding: 6px 14px;
            margin-right: 4px;
        }}
        QMenuBar::item:selected {{
            background-color: rgb{t.bg_elevated};
        }}
        QMenu {{
            background-color: rgb{t.bg_surface};
            border: 1px solid rgb{t.border};
            font-size: 14px;
            font-weight: 500;
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 24px 6px 12px;
        }}
        QMenu::item:selected {{
            background-color: rgb{t.bg_elevated};
            color: rgb{t.accent_cyan};
        }}

        QMessageBox {{
            background-color: rgb{t.bg_surface};
        }}
        QMessageBox QLabel {{
            color: rgb{t.text_primary};
            background-color: transparent;
        }}
        QMessageBox QPushButton {{
            min-width: 88px;
            padding: 5px 14px;
            color: rgb{t.text_primary};
            background-color: rgb{t.bg_elevated};
            border: 1px solid rgb{t.border};
        }}
        QMessageBox QPushButton:hover {{
            border-color: rgb{t.accent_cyan};
            background-color: rgb{t.bg_surface};
        }}

        #RouteBannerFrame {{
            background-color: rgb{t.bg_surface};
            border: 1px solid rgb{t.border_muted};
        }}
        #RouteBannerPath {{
            background-color: transparent;
            font-size: 15px;
            font-weight: 700;
        }}

        /* -------- Buttons -------- */
        QPushButton {{
            background-color: rgb{t.bg_surface};
            color: rgb{t.text_primary};
            border: 1px solid rgb{t.border};
            border-radius: 0px;
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
            border-radius: 0px;
            color: rgb{t.text_secondary};
            font-family: "{sp.font_family}";
            font-size: {sp.label_font_pt}pt;
            padding: 8px;
            selection-background-color: rgb{t.accent_cyan};
        }}

        /* -------- ProgressBar -------- */
        QProgressBar {{
            border: 1px solid rgb{t.border_muted};
            border-radius: 0px;
            text-align: center;
            background-color: rgb{t.bg_deep};
            font-size: {sp.label_font_pt}pt;
            min-height: {sp.progress_bar_height}px;
            font-weight: 600;
        }}
        QProgressBar::chunk {{
            background-color: rgb{t.accent_cyan};
            border-radius: 0px;
        }}
        
        /* -------- ScrollBar -------- */
        QScrollBar:vertical {{
            background-color: rgb{t.bg_deep};
            width: 10px;
            border-radius: 0px;
        }}
        QScrollBar::handle:vertical {{
            background-color: rgb{t.bg_elevated};
            border-radius: 0px;
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
            default_wait = 5
            if self._device_mode == "single" and hasattr(self.viewmodel.adapter, "env"):
                wait_durations = list(getattr(self.viewmodel.adapter.env, "wait_durations", [5]))
                if wait_durations:
                    default_wait = int(min(wait_durations))
            self._on_wait_clicked(default_wait)
        # elif key == Qt.Key.Key_R: self._on_random_clicked() # Removed
        elif key == Qt.Key.Key_M:
            self._on_model_step_clicked()
        elif key == Qt.Key.Key_A:
            if self._model_handler is not None or self._concurrent_model_handler is not None:
                self.right_panel.model_auto_button.click()
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
            seq_path = self._action_sequence_path
            if seq_path is None:
                QMessageBox.warning(self, "Error", "未选择动作序列 JSON，请先在回放菜单中选择文件。")
                return False
            if not seq_path.exists():
                    QMessageBox.warning(self, "Error", f"{seq_path} not found! Run generation script first.")
                    return False
            
            with open(seq_path, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
            self._verification_sequence, self._verification_env_overrides = self._parse_verification_payload(raw_payload)
            
            if not self._verification_sequence:
                QMessageBox.warning(self, "Error", "Sequence is empty!")
                return False

            if not self._apply_verification_env_overrides_if_needed():
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

    def _parse_verification_payload(self, payload):
        """
        兼容两种格式：
        1) 旧版：直接是 list[step]
        2) 新版：{"sequence": [...], "replay_env_overrides": {...}, ...}
        """
        if isinstance(payload, list):
            overrides = {
                "runtime_mode": "concurrent" if self._sequence_looks_concurrent(payload) else "single"
            }
            return payload, overrides
        if isinstance(payload, dict):
            seq = payload.get("sequence", [])
            overrides = dict(payload.get("replay_env_overrides", None) or {})
            runtime_mode = str(
                payload.get("device_mode", "") or overrides.get("runtime_mode", "")
            ).lower()
            if runtime_mode == "concurrent" or self._sequence_looks_concurrent(seq):
                overrides["runtime_mode"] = "concurrent"
            elif overrides:
                overrides["runtime_mode"] = "single"
            if not overrides:
                overrides = None
            return seq, overrides
        raise ValueError("动作序列 JSON 格式不合法，需为数组或包含 sequence 字段的对象")

    def _sequence_looks_concurrent(self, sequence) -> bool:
        if not isinstance(sequence, list) or not sequence:
            return False
        for item in sequence[:5]:
            if not isinstance(item, dict):
                continue
            if any(key in item for key in ("action_tm1", "action_tm2", "action_tm3")):
                return True
            actions = item.get("actions", None)
            if not isinstance(actions, list) or len(actions) < 2:
                continue
            non_wait = [
                str(name)
                for name in actions[:3]
                if isinstance(name, str) and not str(name).upper().startswith("WAIT")
            ]
            if len(non_wait) >= 2:
                return True
        return False

    def _apply_verification_env_overrides_if_needed(self) -> bool:
        """
        若序列携带环境覆盖参数，回放前先重建环境，确保导出序列与回放环境一致。
        """
        overrides = self._verification_env_overrides
        if not overrides:
            return True
        if self._adapter_factory is None:
            return False
        try:
            runtime_mode = str(dict(overrides).get("runtime_mode", "")).lower()
            target_mode = "cascade" if runtime_mode == "concurrent" else self._device_mode
            new_adapter = self._adapter_factory(target_mode, self._robot_capacity, overrides)
            self.apply_runtime_adapter(
                new_adapter,
                target_mode,
                concurrent_runtime=(runtime_mode == "concurrent"),
                reset=True,
            )
            self._model_handler = None
            self._concurrent_model_handler = None
            self._update_model_buttons_state()
            return True
        except Exception as e:
            QMessageBox.warning(self, "回放环境构建失败", f"无法应用动作序列携带的环境参数: {e}")
            return False

    def _resolve_wait_or_transition(
        self, action_name: str | None, all_transitions: list
    ) -> int:
        """
        将动作名解析为 adapter 可用的索引。
        - 非 WAIT 变迁：返回 id2t_name 中的索引
        - "WAIT"：返回 -1（adapter 用 action_space_size，即最小 wait）
        - "WAIT_Xs"：返回对应档位的 action 索引，确保 WAIT_50s 等正确执行
        """
        if not action_name or not str(action_name).strip():
            return -1
        s = str(action_name).strip()
        if not s.upper().startswith("WAIT"):
            try:
                return all_transitions.index(action_name)
            except ValueError:
                return -1
        if s.upper() == "WAIT":
            return -1
        # WAIT_5s / WAIT_50s 等：解析时长并查找匹配的 action 索引
        if s.startswith("WAIT_") and s.endswith("s"):
            wait_text = s.removeprefix("WAIT_").removesuffix("s")
            try:
                wait_duration = int(wait_text)
                if hasattr(self.viewmodel.adapter, "env"):
                    env = self.viewmodel.adapter.env
                    for idx in range(int(getattr(env, "n_actions", 0))):
                        parse_fn = getattr(env, "parse_wait_action", None)
                        if parse_fn and parse_fn(idx) == wait_duration:
                            return int(idx)
            except ValueError:
                pass
        return -1

    def _get_verification_step_action(self):
        """Get next action from verification sequence"""
        if not self._ensure_verification_initialized():
             return None
             
        if self._verification_index >= len(self._verification_sequence):
             QMessageBox.information(self, "Done", "Verification Sequence Completed!")
             self.viewmodel.set_auto_mode(False) # Stop auto if running
             return None
             
        try:
            all_transitions = self.viewmodel.adapter.net.id2t_name
            item = self._verification_sequence[self._verification_index]

            if self._device_mode == "single":
                action_name = item.get("action", None)
                if action_name is None:
                    actions = item.get("actions", [])
                    if isinstance(actions, list) and actions:
                        action_name = actions[0]
                if action_name is None:
                    raise ValueError("当前为单设备模式，序列缺少 action（或 actions[0]）字段")

                resolved = self._resolve_wait_or_transition(action_name, all_transitions)
                action_id = (
                    int(resolved)
                    if resolved >= 0
                    else int(self.viewmodel.adapter.action_space_size)
                )
                self._verification_index += 1
                return action_id

            actions = item.get("actions", None)
            if not isinstance(actions, list) or len(actions) < 2:
                raise ValueError("当前为级联设备模式，序列需提供 actions=[tm1, tm2, tm3] 或旧格式 [tm2, tm3]")

            if len(actions) >= 3:
                tm1_name, tm2_name, tm3_name = actions[:3]
            else:
                tm1_name, tm2_name, tm3_name = "WAIT", actions[0], actions[1]
            a1 = self._resolve_wait_or_transition(tm1_name, all_transitions)
            a2 = self._resolve_wait_or_transition(tm2_name, all_transitions)
            a3 = self._resolve_wait_or_transition(tm3_name, all_transitions)

            self._verification_index += 1
            return (a1, a2, a3)

        except ValueError as e:
            QMessageBox.warning(self, "回放序列不匹配", str(e))
            print(f"Error mapping transition: {e}")
            self.viewmodel.set_auto_mode(False)
            return None

    def _on_verify_planb_clicked(self) -> None:
        """Handle Verify PlanB button click (Manual Step)"""
        if self._current_auto_mode == 'B' and self.viewmodel.auto_mode:
             # If auto is running, maybe pause?
             self.viewmodel.set_auto_mode(False)
             return

        replay_action = self._get_verification_step_action()
        if replay_action is None:
            return
        if self._device_mode == "single":
            self.viewmodel.execute_action(int(replay_action))
        else:
            self.viewmodel.execute_concurrent_action(*replay_action)

    def _on_model_b_auto_toggled(self, enabled: bool) -> None:
        """Handle Auto Model B toggle"""
        if enabled:
            # Switch to B-mode
            self._current_auto_mode = 'B'
            
            # Use a lambda wrapper to call our verification provider
            # The provider returns action(int) / (a1, a2) / None
            def b_auto_cb():
                res = self._get_verification_step_action()
                if res is None:
                    return None
                if self._device_mode == "single":
                    self.viewmodel.execute_action(int(res))
                else:
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
            output_path = gantt_output_path("ui_gantt.png")
            
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

