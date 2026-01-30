"""
可视化适配器冒烟测试（无 GUI）
"""

from __future__ import annotations

from solutions.PPO.enviroment import Env_PN

from .petri_adapter import PetriAdapter


def run_smoke() -> None:
    env = Env_PN(detailed_reward=True)
    adapter = PetriAdapter(env)
    state = adapter.reset()
    print("time:", state.time)
    print("actions:", len(state.enabled_actions))

    enabled = [a.action_id for a in state.enabled_actions if a.enabled]
    if not enabled:
        print("no enabled actions")
        return

    action = enabled[0]
    state, reward, done, _info = adapter.step(action)
    print("step reward:", reward)
    print("done:", done)
    print("chambers:", len(state.chambers))


if __name__ == "__main__":
    run_smoke()
