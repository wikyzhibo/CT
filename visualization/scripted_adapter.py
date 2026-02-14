"""
脚本动作适配器 - 支持从文件加载动作序列
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class ScriptedActions:
    """动作序列管理"""

    def __init__(self, actions_path: Optional[str] = None) -> None:
        self.actions: List[str | int] = []
        self.index = 0
        if actions_path:
            self.load(actions_path)

    def load(self, actions_path: str) -> None:
        path = Path(actions_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到动作文件: {actions_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("动作文件必须是 JSON list")
        self.actions = data
        self.index = 0

    def has_next(self) -> bool:
        return self.index < len(self.actions)

    def next_action(self):
        if not self.has_next():
            return None
        action = self.actions[self.index]
        self.index += 1
        return action
