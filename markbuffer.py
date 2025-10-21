import torch
import numpy as np

def _as_bool(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if x.dtype != torch.bool:
        x = x != 0
    return x.view(-1)  # [T]

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def collect_markings_env(batch_td, keep_deadlock_mid=False):
    """
    适配单环境（非并行）SyncDataCollector 批次：
      输入：一个 batch TensorDict（含 next.observation / next.terminated / (可选) next.deadlock）
      输出：
        live_X  : List[np.ndarray]，每个元素是成功 episode 的整段 [L, F]
        dead_X  : np.ndarray [N_dead, F]，死锁终止标识集合
        risky_X : List[np.ndarray]（可选），死锁 episode 的中间标识序列
    逻辑：
      - 扫描 t=0..T-1，把 next.observation[t] 依次压入当前轨迹
      - 若 next.terminated[t] 为 True：
          * 若死锁 -> 记录终止 obs[t] 到 dead_X，清空当前轨迹（可选：把过程放 risky）
          * 否则   -> 把当前轨迹作为 live_X，清空当前轨迹
    """
    nxt = batch_td.get("next", default=None)
    if nxt is None:
        raise RuntimeError("batch missing 'next'")

    obs = nxt.get("observation")         # [T, F]
    term = nxt.get("terminated")         # [T] 或 [T,1]
    dead = nxt.get("deadlock", None)     # 可能没有
    am   = nxt.get("action_mask", None)  # 兜底判断死锁

    if obs is None or term is None:
        return [], np.zeros((0,)), []

    T = obs.shape[0]
    F = obs.shape[-1]

    term = _as_bool(term)                # [T]
    if dead is not None:
        dead = _as_bool(dead)            # [T]
    else:
        # fallback: 以 action_mask 全 False + terminated 为死锁
        if am is not None:
            no_legal = (am.sum(dim=-1) == 0).view(-1)  # [T]
            dead = term & no_legal
        else:
            dead = torch.zeros_like(term)

    live_X, dead_list, risky_X = [], [], []
    traj = []  # 当前未结束的活标识序列

    for t in range(T):
        obs_t = obs[t]                   # [F]
        traj.append(_to_np(obs_t).astype(np.float32))

        if term[t]:
            if dead[t]:
                # 死锁：记录终止标识
                dead_list.append(_to_np(obs_t).astype(np.float32))
                if keep_deadlock_mid and len(traj) > 0:
                    risky_X.append(np.stack(traj, axis=0))
                traj.clear()
            else:
                # 成功：整段作为 live
                if len(traj) > 190:
                    live_X.append(np.stack(traj, axis=0))  # [L, F]
                traj.clear()

    dead_X = np.stack(dead_list, axis=0) if len(dead_list) else np.zeros((0, F), dtype=np.float32)
    return live_X, dead_X, risky_X
