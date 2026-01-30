"""
数据导出工具 - 甘特图/统计数据/动作序列
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox

from .viewmodel import PetriViewModel


class ExportTools(QWidget):
    """导出工具面板"""

    def __init__(self, viewmodel: PetriViewModel, parent=None) -> None:
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.setWindowTitle("数据导出")
        self.setMinimumWidth(360)

        root = QVBoxLayout(self)
        self.btn_gantt = QPushButton("导出甘特图 (PNG)")
        self.btn_stats = QPushButton("导出统计数据 (CSV)")
        self.btn_actions = QPushButton("导出动作序列 (JSON)")

        self.btn_gantt.clicked.connect(self.export_gantt)
        self.btn_stats.clicked.connect(self.export_stats)
        self.btn_actions.clicked.connect(self.export_actions)

        root.addWidget(self.btn_gantt)
        root.addWidget(self.btn_stats)
        root.addWidget(self.btn_actions)

    def export_gantt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存甘特图", "gantt.png", "PNG Files (*.png)")
        if not path:
            return
        ok = self.viewmodel.adapter.render_gantt(path)
        if ok:
            QMessageBox.information(self, "导出成功", f"已保存: {path}")
        else:
            QMessageBox.warning(self, "导出失败", "甘特图导出失败")

    def export_stats(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存统计数据", "stats.csv", "CSV Files (*.csv)")
        if not path:
            return
        stats = self._build_stats()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in stats.items():
                writer.writerow([key, value])
        QMessageBox.information(self, "导出成功", f"已保存: {path}")

    def export_actions(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存动作序列", "actions.json", "JSON Files (*.json)")
        if not path:
            return
        actions = self.viewmodel.adapter.export_action_sequence()
        Path(path).write_text(json.dumps(actions, indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "导出成功", f"已保存: {path}")

    def _build_stats(self) -> Dict[str, Any]:
        state = self.viewmodel.adapter.get_current_state()
        return {
            "time": state.time,
            "done_count": state.done_count,
            "total_wafers": state.total_wafers,
            "progress": state.done_count / max(1, state.total_wafers),
            "throughput": self.viewmodel.trend_data.get("throughput", [])[-1:] or [0],
            "avg_stay_time": self.viewmodel.trend_data.get("avg_stay_time", [])[-1:] or [0],
            "utilization": self.viewmodel.trend_data.get("utilization", [])[-1:] or [0],
        }
