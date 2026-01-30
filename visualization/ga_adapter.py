"""
遗传算法适配器 - 使用脚本动作序列回放
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

from .algorithm_interface import AlgorithmAdapter, ActionInfo, StateInfo
from .pdr_adapter import PDRAdapter
from .scripted_adapter import ScriptedActions


class GAAdapter(AlgorithmAdapter):
    """遗传算法动作回放适配器"""

    def __init__(self, actions_path: Optional[str] = None) -> None:
        self.base = PDRAdapter()
        self.scripted = ScriptedActions(actions_path)

    def reset(self) -> StateInfo:
        return self.base.reset()

    def step(self, action: int) -> Tuple[StateInfo, float, bool, Dict]:
        if self.scripted.has_next():
            scripted = self.scripted.next_action()
            action = self._resolve_action(scripted)
        return self.base.step(action)

    def get_action_name(self, action: int) -> str:
        return self.base.get_action_name(action)

    def get_enabled_actions(self) -> List[ActionInfo]:
        return self.base.get_enabled_actions()

    def get_reward_breakdown(self) -> Dict[str, float]:
        return {}

    @property
    def action_space_size(self) -> int:
        return self.base.action_space_size

    def get_current_state(self) -> StateInfo:
        return self.base.get_current_state()

    def export_action_sequence(self) -> List[Dict[str, Any]]:
        return self.base.export_action_sequence()

    def _resolve_action(self, action: Any) -> int:
        if isinstance(action, int):
            return action
        if isinstance(action, str):
            if action in self.base.net.id2t_name:
                return self.base.net.id2t_name.index(action)
        return 0
