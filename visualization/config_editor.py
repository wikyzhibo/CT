"""
配置编辑器 - 编辑 PetriEnvConfig JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
)


class ConfigEditor(QWidget):
    """Petri 配置编辑器（JSON）"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Petri 配置编辑器")
        self.setMinimumWidth(480)

        self.path_edit = QLineEdit()
        self.path_edit.setText(str(Path("data/petri_configs/phase2_config.json")))

        self.fields: Dict[str, Any] = {}

        root = QVBoxLayout(self)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("配置文件"))
        path_row.addWidget(self.path_edit)
        load_btn = QPushButton("加载")
        save_btn = QPushButton("保存")
        load_btn.clicked.connect(self.load_config)
        save_btn.clicked.connect(self.save_config)
        path_row.addWidget(load_btn)
        path_row.addWidget(save_btn)
        root.addLayout(path_row)

        self.basic_group = QGroupBox("基础与腔室参数")
        self.reward_group = QGroupBox("奖励参数")
        root.addWidget(self.basic_group)
        root.addWidget(self.reward_group)

        self._build_basic_form()
        self._build_reward_form()

    def _build_basic_form(self) -> None:
        layout = QFormLayout(self.basic_group)

        self.fields["n_wafer"] = self._spin(0, 1000, 8)
        self.fields["n_wafer_route1"] = self._spin(0, 1000, 8)
        self.fields["n_wafer_route2"] = self._spin(0, 1000, 0)
        self.fields["max_wafers_in_system"] = self._spin(0, 1000, 8)
        self.fields["training_phase"] = self._spin(1, 2, 2)
        self.fields["stop_on_scrap"] = self._check(True)

        # 时间/腔室相关参数
        self.fields["D_Residual_time"] = self._spin(0, 1000, 10)
        self.fields["P_Residual_time"] = self._spin(0, 1000, 15)
        self.fields["T_transport"] = self._spin(0, 1000, 5)
        self.fields["T_load"] = self._spin(0, 1000, 5)
        self.fields["T_pm1_to_pm2"] = self._spin(0, 1000, 15)

        # 其他参数
        self.fields["MAX_WAIT_STEP"] = self._spin(0, 1000, 20)
        self.fields["idle_timeout"] = self._spin(0, 10000, 700)
        self.fields["idle_penalty"] = self._spin(0, 100000, 1000)

        layout.addRow("n_wafer", self.fields["n_wafer"])
        layout.addRow("n_wafer_route1", self.fields["n_wafer_route1"])
        layout.addRow("n_wafer_route2", self.fields["n_wafer_route2"])
        layout.addRow("max_wafers_in_system", self.fields["max_wafers_in_system"])
        layout.addRow("training_phase", self.fields["training_phase"])
        layout.addRow("stop_on_scrap", self.fields["stop_on_scrap"])
        layout.addRow("D_Residual_time", self.fields["D_Residual_time"])
        layout.addRow("P_Residual_time", self.fields["P_Residual_time"])
        layout.addRow("T_transport", self.fields["T_transport"])
        layout.addRow("T_load", self.fields["T_load"])
        layout.addRow("T_pm1_to_pm2", self.fields["T_pm1_to_pm2"])
        layout.addRow("MAX_WAIT_STEP", self.fields["MAX_WAIT_STEP"])
        layout.addRow("idle_timeout", self.fields["idle_timeout"])
        layout.addRow("idle_penalty", self.fields["idle_penalty"])

    def _build_reward_form(self) -> None:
        layout = QFormLayout(self.reward_group)

        self.fields["R_done"] = self._spin(-100000, 100000, 100)
        self.fields["R_scrap"] = self._spin(-100000, 100000, 500)
        self.fields["c_time"] = self._spin(0, 100, 2)
        self.fields["c_congest"] = self._spin(0, 10000, 50)
        self.fields["c_release_violation"] = self._spin(0, 10000, 10)
        self.fields["T_warn"] = self._spin(0, 1000, 10)
        self.fields["T_safe"] = self._spin(0, 1000, 15)
        self.fields["a_warn"] = self._double(0, 10, 0.0)
        self.fields["b_safe"] = self._double(0, 10, 0.5)

        layout.addRow("R_done", self.fields["R_done"])
        layout.addRow("R_scrap", self.fields["R_scrap"])
        layout.addRow("c_time", self.fields["c_time"])
        layout.addRow("c_congest", self.fields["c_congest"])
        layout.addRow("c_release_violation", self.fields["c_release_violation"])
        layout.addRow("T_warn", self.fields["T_warn"])
        layout.addRow("a_warn", self.fields["a_warn"])
        layout.addRow("T_safe", self.fields["T_safe"])
        layout.addRow("b_safe", self.fields["b_safe"])

        # reward_config 开关
        self.fields["reward_config.proc_reward"] = self._check(True)
        self.fields["reward_config.safe_reward"] = self._check(True)
        self.fields["reward_config.penalty"] = self._check(True)
        self.fields["reward_config.warn_penalty"] = self._check(True)
        self.fields["reward_config.transport_penalty"] = self._check(True)
        self.fields["reward_config.congestion_penalty"] = self._check(False)
        self.fields["reward_config.time_cost"] = self._check(True)
        self.fields["reward_config.release_violation_penalty"] = self._check(True)

        layout.addRow("proc_reward", self.fields["reward_config.proc_reward"])
        layout.addRow("safe_reward", self.fields["reward_config.safe_reward"])
        layout.addRow("penalty", self.fields["reward_config.penalty"])
        layout.addRow("warn_penalty", self.fields["reward_config.warn_penalty"])
        layout.addRow("transport_penalty", self.fields["reward_config.transport_penalty"])
        layout.addRow("congestion_penalty", self.fields["reward_config.congestion_penalty"])
        layout.addRow("time_cost", self.fields["reward_config.time_cost"])
        layout.addRow("release_violation_penalty", self.fields["reward_config.release_violation_penalty"])

    def load_config(self) -> None:
        path = Path(self.path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "加载失败", f"找不到配置文件: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        for key, widget in self.fields.items():
            value = self._get_value(data, key)
            if value is None:
                continue
            self._set_widget_value(widget, value)

    def save_config(self) -> None:
        path = Path(self.path_edit.text())
        data: Dict[str, Any] = {}
        for key, widget in self.fields.items():
            value = self._get_widget_value(widget)
            self._set_value(data, key, value)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "保存成功", f"已保存到 {path}")

    def _spin(self, min_val: int, max_val: int, default: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(min_val, max_val)
        box.setValue(default)
        return box

    def _double(self, min_val: float, max_val: float, default: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(min_val, max_val)
        box.setValue(default)
        return box

    def _check(self, default: bool) -> QCheckBox:
        box = QCheckBox()
        box.setChecked(default)
        return box

    def _get_widget_value(self, widget):
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def _set_widget_value(self, widget, value):
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def _get_value(self, data: Dict[str, Any], key: str):
        parts = key.split(".")
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _set_value(self, data: Dict[str, Any], key: str, value: Any) -> None:
        parts = key.split(".")
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
