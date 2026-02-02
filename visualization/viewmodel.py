"""
ViewModel - 统一状态管理与交互逻辑
"""

from __future__ import annotations

from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from .algorithm_interface import AlgorithmAdapter, StateInfo


class PetriViewModel(QObject):
    """Petri 网可视化 ViewModel"""

    state_updated = Signal(object)
    reward_updated = Signal(float, dict)
    step_updated = Signal(int)
    done_changed = Signal(bool)
    auto_mode_changed = Signal(bool)

    def __init__(self, adapter: AlgorithmAdapter) -> None:
        super().__init__()
        self.adapter = adapter

        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.auto_mode = False
        self.auto_speed = 0.25

        self.action_history: List[Dict[str, Any]] = []
        self.trend_data: Dict[str, List[float]] = {
            "throughput": [],
            "avg_stay_time": [],
            "utilization": [],
        }

        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self._auto_step)

    @Slot()
    def reset(self) -> None:
        state = self.adapter.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.action_history.clear()
        for key in self.trend_data:
            self.trend_data[key].clear()

        self.state_updated.emit(state)
        self.step_updated.emit(0)
        self.reward_updated.emit(0.0, {})
        self.done_changed.emit(False)

    @Slot(int)
    def execute_action(self, action: int) -> None:
        if self.done:
            return

        state, reward, done, _info = self.adapter.step(action)

        self.step_count += 1
        self.total_reward += reward
        self.last_reward = reward
        self.done = done

        self.action_history.append({
            "step": self.step_count,
            "action": self.adapter.get_action_name(action),
            "reward": reward,
            "detail": self.adapter.get_reward_breakdown(),
        })

        self._update_trends(state)

        self.state_updated.emit(state)
        self.step_updated.emit(self.step_count)
        self.reward_updated.emit(self.total_reward, self.adapter.get_reward_breakdown())
        if done:
            self.done_changed.emit(True)
            self.set_auto_mode(False)

    @Slot(int, int)
    def execute_concurrent_action(self, a1: int, a2: int) -> None:
        """执行双机械手并发动作。
        
        Args:
            a1: TM2 的变迁索引，-1 表示 WAIT
            a2: TM3 的变迁索引，-1 表示 WAIT
        """
        if self.done:
            return

        state, reward, done, _info = self.adapter.step_concurrent(a1, a2)

        self.step_count += 1
        self.total_reward += reward
        self.last_reward = reward
        self.done = done

        # 记录双动作历史
        a1_name = "WAIT" if a1 == -1 else self.adapter.get_action_name(a1)
        a2_name = "WAIT" if a2 == -1 else self.adapter.get_action_name(a2)
        self.action_history.append({
            "step": self.step_count,
            "action": f"TM2:{a1_name} | TM3:{a2_name}",
            "reward": reward,
            "detail": self.adapter.get_reward_breakdown(),
        })

        self._update_trends(state)

        self.state_updated.emit(state)
        self.step_updated.emit(self.step_count)
        self.reward_updated.emit(self.total_reward, self.adapter.get_reward_breakdown())
        if done:
            self.done_changed.emit(True)
            self.set_auto_mode(False)

    @Slot(bool)
    def set_auto_mode(self, enabled: bool) -> None:
        self.auto_mode = enabled
        if enabled:
            self.auto_timer.start(int(self.auto_speed * 1000))
        else:
            self.auto_timer.stop()
        self.auto_mode_changed.emit(enabled)

    @Slot(float)
    def set_auto_speed(self, speed: float) -> None:
        self.auto_speed = float(speed)
        if self.auto_mode:
            self.auto_timer.start(int(self.auto_speed * 1000))

    def set_agent_callback(self, callback) -> None:
        """设置自动模式下的智能体回调"""
        self.agent_callback = callback

    @Slot()
    def _auto_step(self) -> None:
        # 优先使用智能体回调
        if hasattr(self, 'agent_callback') and self.agent_callback:
            try:
                action = self.agent_callback()
                if action is not None:
                    self.execute_action(action)
                return
            except Exception as e:
                print(f"Agent callback error: {e}")
                self.set_auto_mode(False)
                return

        # 降级到随机游走
        actions = self.adapter.get_enabled_actions()
        enabled = [a.action_id for a in actions if a.enabled]
        if not enabled:
            return
        import random
        action = random.choice(enabled)
        self.execute_action(action)

    def _update_trends(self, state: StateInfo) -> None:
        throughput = state.done_count
        total_stay = 0.0
        wafer_count = 0
        for chamber in state.chambers + state.transport_buffers:
            for wafer in chamber.wafers:
                total_stay += wafer.stay_time
                wafer_count += 1

        avg_stay = total_stay / max(1, wafer_count)
        active_chambers = sum(1 for c in state.chambers if c.chamber_type == "processing")
        busy_chambers = sum(1 for c in state.chambers if c.wafers)
        utilization = busy_chambers / max(1, active_chambers)

        self._append_trend("throughput", float(throughput))
        self._append_trend("avg_stay_time", float(avg_stay))
        self._append_trend("utilization", float(utilization))

    def _append_trend(self, key: str, value: float) -> None:
        self.trend_data[key].append(value)
        max_points = 30
        if len(self.trend_data[key]) > max_points:
            self.trend_data[key] = self.trend_data[key][-max_points:]
