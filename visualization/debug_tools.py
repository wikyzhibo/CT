"""
调试工具 - 断点/单步执行/状态检查器
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QSpinBox,
    QMessageBox,
)

from .viewmodel import PetriViewModel


class DebugTools(QWidget):
    """调试工具面板"""

    def __init__(self, viewmodel: PetriViewModel, parent=None) -> None:
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.setWindowTitle("调试工具")
        self.setMinimumWidth(520)

        self.breakpoint_step = None

        root = QVBoxLayout(self)

        # 断点控制
        bp_row = QHBoxLayout()
        bp_row.addWidget(QLabel("断点(step):"))
        self.bp_input = QSpinBox()
        self.bp_input.setRange(0, 1000000)
        self.bp_input.setValue(0)
        bp_row.addWidget(self.bp_input)
        self.bp_set_btn = QPushButton("设置断点")
        self.bp_clear_btn = QPushButton("清除断点")
        self.bp_set_btn.clicked.connect(self._set_breakpoint)
        self.bp_clear_btn.clicked.connect(self._clear_breakpoint)
        bp_row.addWidget(self.bp_set_btn)
        bp_row.addWidget(self.bp_clear_btn)
        root.addLayout(bp_row)

        # 单步执行
        step_row = QHBoxLayout()
        self.step_btn = QPushButton("单步执行")
        self.step_btn.clicked.connect(self._step_once)
        step_row.addWidget(self.step_btn)
        root.addLayout(step_row)

        # 状态检查
        self.state_text = QTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMinimumHeight(240)
        root.addWidget(QLabel("状态检查器"))
        root.addWidget(self.state_text)

        self.viewmodel.step_updated.connect(self._on_step_updated)
        self.viewmodel.state_updated.connect(self._refresh_state)

    def _set_breakpoint(self) -> None:
        self.breakpoint_step = int(self.bp_input.value())
        QMessageBox.information(self, "断点已设置", f"断点: step {self.breakpoint_step}")

    def _clear_breakpoint(self) -> None:
        self.breakpoint_step = None
        QMessageBox.information(self, "断点已清除", "断点已清除")

    def _step_once(self) -> None:
        actions = self.viewmodel.adapter.get_enabled_actions()
        enabled = [a.action_id for a in actions if a.enabled]
        if not enabled:
            QMessageBox.warning(self, "无动作", "当前无可用动作")
            return
        self.viewmodel.execute_action(enabled[0])

    def _on_step_updated(self, step: int) -> None:
        if self.breakpoint_step is not None and step >= self.breakpoint_step:
            self.viewmodel.set_auto_mode(False)

    def _refresh_state(self, state) -> None:
        state_dict = self._to_dict(state)
        self.state_text.setText(self._format_dict(state_dict))

    def _to_dict(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: self._to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_dict(v) for v in obj]
        return obj

    def _format_dict(self, data: Dict[str, Any]) -> str:
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)
