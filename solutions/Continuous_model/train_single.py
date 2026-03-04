"""
单设备训练入口（占位实现）。

说明：
- 该文件仅提供单设备环境构建入口，后续可按现有 PPO 训练脚本扩展。
"""

from __future__ import annotations

from solutions.Continuous_model.env_single import Env_PN_Single


def build_single_env() -> Env_PN_Single:
    return Env_PN_Single(detailed_reward=True)


if __name__ == "__main__":
    env = build_single_env()
    td = env.reset()
    print("single env ready, obs shape:", td["observation"].shape)
