"""
Deprecated compatibility module.

请改用 `solutions.A.rl_env.Env_PN_Concurrent`。
本文件仅保留导入兼容，避免历史调用路径中断。
"""

from solutions.A.rl_env import Env_PN_Concurrent

__all__ = ["Env_PN_Concurrent"]
