"""
PDR 算法适配器
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

from data.config.params_N8 import params_N8
from solutions.PDR.net import Petri

from .algorithm_interface import AlgorithmAdapter, ActionInfo, StateInfo


class PDRAdapter(AlgorithmAdapter):
    """PDR 调度适配器（简化版）"""

    def __init__(self) -> None:
        params = dict(params_N8)
        self.net = Petri(
            with_controller=True,
            with_capacity_controller=True,
            with_zhiliu_controller=True,
            **params,
        )
        self._last_actions: List[Dict[str, Any]] = []

    def reset(self) -> StateInfo:
        self.net.reset()
        return self._build_state()

    def step(self, action: int) -> Tuple[StateInfo, float, bool, Dict]:
        info = self.net.step(action)
        done = bool(info.get("finish", False) or info.get("deadlock", False))
        self._last_actions.append({
            "step": len(self._last_actions) + 1,
            "action": self.get_action_name(action),
            "time": info.get("time"),
            "deadlock": info.get("deadlock"),
        })
        return self._build_state(), 0.0, done, info

    def get_action_name(self, action: int) -> str:
        if 0 <= action < len(self.net.id2t_name):
            return self.net.id2t_name[action]
        return f"UNKNOWN_{action}"

    def get_enabled_actions(self) -> List[ActionInfo]:
        mask = self.net.mask_t(self.net.m, self.net.marks)
        actions: List[ActionInfo] = []
        for i, name in enumerate(self.net.id2t_name):
            actions.append(ActionInfo(action_id=i, action_name=name, enabled=bool(mask[i])))
        return actions

    def get_reward_breakdown(self) -> Dict[str, float]:
        return {}

    @property
    def action_space_size(self) -> int:
        return int(self.net.T)

    def get_current_state(self) -> StateInfo:
        return self._build_state()

    def export_action_sequence(self) -> List[Dict[str, Any]]:
        return list(self._last_actions)

    def _build_state(self) -> StateInfo:
        return StateInfo(
            time=float(getattr(self.net, "time", 0)),
            chambers=[],
            transport_buffers=[],
            start_buffers=[],
            end_buffers=[],
            robot_states={},
            enabled_actions=self.get_enabled_actions(),
            done_count=0,
            total_wafers=int(getattr(self.net, "n_wafer", 0) or 0),
            stats={},
        )
