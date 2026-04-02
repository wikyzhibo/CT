"""
单设备单动作 PPO 训练脚本（Ultra collector + Batched PPO update）。
"""

from __future__ import annotations

import copy
import json
import os
from collections import defaultdict, deque
from datetime import datetime
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensordict import TensorDict
from config.training.training_config import PPOTrainingConfig
from pathlib import Path
from results.paths import (
    action_sequence_path,
    gantt_output_path,
    model_output_path,
    safe_name,
    training_log_output_path,
)

import numpy as np

from solutions.A.rl_env import (
    Env_PN_Single, Env_PN_Concurrent,
    FastEnvWrapper, VectorEnv,
    FastEnvWrapper_Concurrent, VectorEnv_Concurrent,
)
from solutions.A.eval.plot_train_metrics import plot_metrics
from solutions.A.eval.export_inference_sequence import rollout_and_export
from solutions.model.network import MaskedPolicyHead, DualHeadPolicyNet


class SingleActionPolicyModule(nn.Module):
    """
    单动作策略模块包装器。
    与并发训练的保存风格保持一致：保存 policy_module.state_dict()。
    """

    def __init__(self, backbone: MaskedPolicyHead):
        super().__init__()
        self.backbone = backbone

    def forward(self, observation_f):
        return self.backbone(observation_f)


class DualActionPolicyModule(nn.Module):
    """
    双动作策略模块包装器（TM2/TM3 双头）。
    """

    def __init__(self, backbone: DualHeadPolicyNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, observation_f):
        return self.backbone(observation_f)


@torch.no_grad()
def _sample_actions_masked(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    高速采样：不创建 distribution 对象，直接 softmax+multinomial。
    """
    # 极端情况下若 mask 全 False，回退到动作 0，避免 softmax NaN。
    valid = mask.any(dim=-1, keepdim=True)
    if not torch.all(valid):
        mask = mask.clone()
        mask[~valid.expand_as(mask)] = False
        mask[~valid.squeeze(-1), 0] = True
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    actions = torch.multinomial(probs, 1).squeeze(-1)
    selected = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1).clamp_min_(1e-12)
    log_probs = torch.log(selected)
    return actions, log_probs


def _masked_logprob_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    更新阶段 fused 计算：
    logits + mask -> log_prob(action), entropy
    避免构造 MaskedCategorical 对象。
    """
    valid = mask.any(dim=-1, keepdim=True)
    if not torch.all(valid):
        mask = mask.clone()
        mask[~valid.expand_as(mask)] = False
        mask[~valid.squeeze(-1), 0] = True
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return action_log_prob, entropy


def _collect_rollout_concurrent(
    env: Env_PN_Concurrent,
    policy_module: DualActionPolicyModule,
    n_steps: int,
    device: str,
) -> TensorDict:
    """
    采集双动作并发 rollout。
    """
    data = {
        "observation_f": [],
        "action_mask_tm2": [],
        "action_mask_tm3": [],
        "action_tm2": [],
        "action_tm3": [],
        "log_prob_tm2": [],
        "log_prob_tm3": [],
        "reward": [],
        "done": [],
        "finish": [],
        "scrap": [],
        "time": [],
        "next_observation_f": [],
    }

    td = env.reset()
    policy_module.eval()
    for _ in range(int(n_steps)):
        obs_f = td["observation"].unsqueeze(0).to(device).float()
        mask_tm2 = td["action_mask_tm2"].unsqueeze(0).to(device).bool()
        mask_tm3 = td["action_mask_tm3"].unsqueeze(0).to(device).bool()

        with torch.no_grad():
            out = policy_module(obs_f)
            logits_tm2 = out["logits_tm2"]
            logits_tm3 = out["logits_tm3"]
            a_tm2, lp_tm2 = _sample_actions_masked(logits_tm2, mask_tm2)
            a_tm3, lp_tm3 = _sample_actions_masked(logits_tm3, mask_tm3)

        data["observation_f"].append(td["observation"].float())
        data["action_mask_tm2"].append(td["action_mask_tm2"])
        data["action_mask_tm3"].append(td["action_mask_tm3"])
        data["action_tm2"].append(a_tm2.squeeze(0).cpu())
        data["action_tm3"].append(a_tm3.squeeze(0).cpu())
        data["log_prob_tm2"].append(lp_tm2.squeeze(0).cpu())
        data["log_prob_tm3"].append(lp_tm3.squeeze(0).cpu())

        step_td = td.clone()
        step_td["action_tm2"] = a_tm2.squeeze(0).cpu()
        step_td["action_tm3"] = a_tm3.squeeze(0).cpu()
        td_next = env.step(step_td)

        if "next" in td_next.keys() and "reward" in td_next["next"].keys():
            reward = td_next["next", "reward"]
            next_obs = td_next["next", "observation"]
            terminated = (
                td_next["next", "terminated"]
                if "terminated" in td_next["next"].keys()
                else td_next["terminated"]
            )
            finish = (
                td_next["next", "finish"]
                if "finish" in td_next["next"].keys()
                else td_next.get("finish", torch.tensor(False))
            )
            scrap = (
                td_next["next", "scrap"]
                if "scrap" in td_next["next"].keys()
                else td_next.get("scrap", torch.tensor(False))
            )
            time_val = (
                td_next["next", "time"]
                if "time" in td_next["next"].keys()
                else td_next.get("time", torch.tensor([0]))
            )
        else:
            reward = td_next["reward"]
            next_obs = td_next["observation"]
            terminated = td_next["terminated"]
            finish = td_next.get("finish", torch.tensor(False))
            scrap = td_next.get("scrap", torch.tensor(False))
            time_val = td_next.get("time", torch.tensor([0]))

        data["reward"].append(reward)
        data["done"].append(terminated)
        data["finish"].append(finish)
        data["scrap"].append(scrap)
        data["time"].append(time_val)
        data["next_observation_f"].append(next_obs.float())

        if bool(terminated.item()):
            td = env.reset()
        else:
            if "next" in td_next.keys():
                td = td_next["next"].clone()
            else:
                td = td_next.clone()

    return TensorDict(
        {k: torch.stack(v) for k, v in data.items()},
        batch_size=[int(n_steps)],
    )


def _train_concurrent(
    config: PPOTrainingConfig,
    checkpoint_path: str | None = None,
    artifact_dir: str | Path | None = None,
    rollout_n_envs: int = 1,
):
    """
    双机械手并发 PPO 训练（Ultra collector + 并行环境 rollout）。
    """
    print("[Concurrent PPO Training]", flush=True)
    print(config, flush=True)

    torch.manual_seed(config.seed)
    device = config.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"  警告: 请求 {device} 但 CUDA 不可用，回退到 cpu", flush=True)
            device = "cpu"
        else:
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    use_cuda_update = str(device).startswith("cuda")
    amp_enabled = use_cuda_update
    amp_dtype = torch.bfloat16 if (use_cuda_update and torch.cuda.is_bf16_supported()) else torch.float16
    scaler: torch.cuda.amp.GradScaler | None = None
    if amp_enabled and (amp_dtype == torch.float16):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    rollout_n_envs = max(1, int(rollout_n_envs))
    rollout_device = "cpu"

    # 从一个探针环境读取动作/观测维度
    _probe_env = Env_PN_Concurrent(device="cpu")
    n_obs = _probe_env.observation_spec["observation"].shape[0]
    n_actions_tm2 = int(_probe_env.n_actions_tm2)
    n_actions_tm3 = int(_probe_env.n_actions_tm3)
    if n_actions_tm2 < 2 or n_actions_tm3 < 2:
        raise ValueError("并发动作空间非法：TM2/TM3 均需包含变迁+WAIT")
    del _probe_env

    print(f"  计算设备: {device}", flush=True)
    print(f"  环境类型: Env_PN_Concurrent", flush=True)
    print(
        f"  Rollout Collector: ultra_concurrent "
        f"(n_envs={rollout_n_envs}, rollout_device={rollout_device})",
        flush=True,
    )
    print(f"  观测维度: {n_obs}", flush=True)
    print(f"  TM2 动作空间: {n_actions_tm2}", flush=True)
    print(f"  TM3 动作空间: {n_actions_tm3}", flush=True)
    print("  并发 WAIT 规则: 单档 WAIT(5s)", flush=True)

    backbone = DualHeadPolicyNet(
        n_obs=n_obs,
        n_hidden=config.n_hidden,
        n_actions_tm2=n_actions_tm2,
        n_actions_tm3=n_actions_tm3,
        n_layers=config.n_layer,
    ).to(device)
    policy_module = DualActionPolicyModule(backbone).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"从 checkpoint 加载: {checkpoint_path}", flush=True)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        policy_module.load_state_dict(state_dict)

    value_net = nn.Sequential(
        nn.Linear(n_obs, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, 1),
    ).to(device)

    train_params = list(backbone.parameters()) + list(value_net.parameters())
    optim = Adam(train_params, lr=config.lr, foreach=bool(use_cuda_update))

    policy_train = _maybe_compile_model(backbone, enabled=use_cuda_update)
    value_train = _maybe_compile_model(value_net, enabled=use_cuda_update)
    rollout_policy = copy.deepcopy(backbone).to("cpu").eval() if use_cuda_update else backbone

    ultra_state: dict[str, Any] = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = model_output_path(f"concurrent_{timestamp}").with_suffix("")
    backup_dir.mkdir(parents=True, exist_ok=True)

    run_name = "train_concurrent"
    if artifact_dir is not None:
        run_name = safe_name(Path(str(artifact_dir)).name, "train_concurrent")
    best_model_path = model_output_path("CT_concurrent_best.pt")
    run_best_model_path = model_output_path(f"{run_name}_best.pt")
    run_final_model_path = model_output_path(f"{run_name}_final.pt")

    best_reward = float("-inf")
    log = defaultdict(list)

    training_start = perf_counter()
    batch_time_window: deque[float] = deque(maxlen=10)
    total_env_steps = 0

    for batch_idx in range(config.total_batch):
        t_batch = perf_counter()

        backbone.eval()
        if use_cuda_update:
            with torch.no_grad():
                rollout_policy.load_state_dict(backbone.state_dict(), strict=True)
            policy_for_rollout = rollout_policy
        else:
            policy_for_rollout = backbone

        steps_per_env = max(1, (int(config.frames_per_batch) + rollout_n_envs - 1) // rollout_n_envs)
        rollout_cpu, ultra_state = collect_rollout_ultra_concurrent(
            env_fn=lambda: Env_PN_Concurrent(device="cpu"),
            policy=policy_for_rollout,
            n_steps=steps_per_env,
            n_envs=rollout_n_envs,
            rollout_device=rollout_device,
            state=ultra_state,
            pin_memory=use_cuda_update,
        )

        t_rollout_end = perf_counter()

        available_frames = int(rollout_cpu["actions_tm2"].numel())
        used_frames = min(int(config.frames_per_batch), available_frames)
        total_env_steps += used_frames

        non_blocking = bool(use_cuda_update)
        rollout_dev = {
            "obs": rollout_cpu["obs"].to(device, non_blocking=non_blocking),
            "next_obs": rollout_cpu["next_obs"].to(device, non_blocking=non_blocking),
            "mask_tm2": rollout_cpu["mask_tm2"].to(device, non_blocking=non_blocking),
            "mask_tm3": rollout_cpu["mask_tm3"].to(device, non_blocking=non_blocking),
            "actions_tm2": rollout_cpu["actions_tm2"].to(device, non_blocking=non_blocking),
            "actions_tm3": rollout_cpu["actions_tm3"].to(device, non_blocking=non_blocking),
            "log_probs_tm2": rollout_cpu["log_probs_tm2"].to(device, non_blocking=non_blocking),
            "log_probs_tm3": rollout_cpu["log_probs_tm3"].to(device, non_blocking=non_blocking),
            "rewards": rollout_cpu["rewards"].to(device, non_blocking=non_blocking),
            "dones": rollout_cpu["dones"].to(device, non_blocking=non_blocking),
            "finish": rollout_cpu["finish"].to(device, non_blocking=non_blocking),
            "scrap": rollout_cpu["scrap"].to(device, non_blocking=non_blocking),
            "time": rollout_cpu["time"].to(device, non_blocking=non_blocking),
        }

        policy_train.train()
        value_train.train()
        with torch.no_grad():
            advantages, returns = _compute_gae_and_returns(
                value_net=value_train,
                obs=rollout_dev["obs"],
                next_obs=rollout_dev["next_obs"],
                rewards=rollout_dev["rewards"],
                dones=rollout_dev["dones"],
                gamma=float(config.gamma),
                gae_lambda=float(config.gae_lambda),
            )

        update_batch = _flatten_rollout_concurrent(
            rollout=rollout_dev,
            advantages=advantages,
            returns=returns,
            max_frames=used_frames,
        )

        update_stats = _ppo_update_batched_concurrent(
            policy_backbone=policy_train,
            value_net=value_train,
            optim=optim,
            train_params=train_params,
            batch=update_batch,
            num_epochs=int(config.num_epochs),
            clip_epsilon=float(config.clip_epsilon),
            entropy_coef=float(config.entropy_start),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )

        t_update_end = perf_counter()

        rollout_time = t_rollout_end - t_batch
        update_time = t_update_end - t_rollout_end
        batch_time = t_update_end - t_batch
        steps_per_sec = used_frames / rollout_time if rollout_time > 0 else 0.0
        batch_time_window.append(batch_time)
        remaining = config.total_batch - (batch_idx + 1)
        eta_s = remaining * (sum(batch_time_window) / len(batch_time_window))
        eta_str = f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"

        flat_rewards = rollout_cpu["rewards"].reshape(-1)[:used_frames]
        flat_finish = rollout_cpu["finish"].reshape(-1).bool()[:used_frames]
        flat_scrap = rollout_cpu["scrap"].reshape(-1).bool()[:used_frames]
        flat_time = rollout_cpu["time"].reshape(-1)[:used_frames]

        ep_reward = float(flat_rewards.sum().item())
        finish_count = int(flat_finish.sum().item())
        scrap_count = int(flat_scrap.sum().item())
        avg_makespan = float(flat_time[flat_finish].float().mean().item()) if finish_count > 0 else 0.0

        log["reward"].append(ep_reward)
        log["finish"].append(finish_count)
        log["scrap"].append(scrap_count)
        log["makespan"].append(avg_makespan)
        log["deadlock"].append(0)
        log["rollout_time"].append(rollout_time)
        log["update_time"].append(update_time)
        log["steps_per_sec"].append(steps_per_sec)
        log["frames"].append(used_frames)
        log["policy_loss"].append(update_stats["policy_loss"])
        log["value_loss"].append(update_stats["value_loss"])
        log["entropy"].append(update_stats["entropy"])

        print(
            f"batch {batch_idx+1:04d} | reward={ep_reward:.2f} | finish={finish_count} "
            f"| scrap={scrap_count} | makespan={avg_makespan:.1f} "
            f"| rollout={rollout_time:.2f}s update={update_time:.2f}s "
            f"| steps/s={steps_per_sec:.0f} | p={update_stats['policy_loss']:.4f} "
            f"v={update_stats['value_loss']:.4f} ent={update_stats['entropy']:.4f} ETA={eta_str}",
            flush=True,
        )

        if ep_reward > best_reward and finish_count > 0:
            best_reward = ep_reward
            torch.save(policy_module.state_dict(), best_model_path)
            torch.save(policy_module.state_dict(), run_best_model_path)
            torch.save(policy_module.state_dict(), backup_dir / "CT_concurrent_best.pt")
            print(f"  -> New best model! reward={ep_reward:.2f}", flush=True)

    total_training_time = perf_counter() - training_start
    total_rollout = sum(log["rollout_time"])
    total_update = sum(log["update_time"])
    avg_rollout = total_rollout / len(log["rollout_time"]) if log["rollout_time"] else 0.0
    avg_update = total_update / len(log["update_time"]) if log["update_time"] else 0.0
    avg_batch = avg_rollout + avg_update
    overall_sps = (total_env_steps / total_rollout) if total_rollout > 0 else 0.0
    rollout_pct = (avg_rollout / avg_batch * 100) if avg_batch > 0 else 0.0
    update_pct = (avg_update / avg_batch * 100) if avg_batch > 0 else 0.0

    print("\n[Concurrent Training Summary]", flush=True)
    print(f"  总训练时间: {total_training_time:.1f}s ({total_training_time / 60:.1f}m)", flush=True)
    print(f"  总 env steps: {total_env_steps}", flush=True)
    print(
        f"  平均 batch: {avg_batch:.2f}s "
        f"(rollout={avg_rollout:.2f}s [{rollout_pct:.0f}%] "
        f"| update={avg_update:.2f}s [{update_pct:.0f}%])",
        flush=True,
    )
    print(f"  平均 steps/sec: {overall_sps:.0f}", flush=True)
    print(f"  Best reward: {best_reward:.2f}", flush=True)

    torch.save(policy_module.state_dict(), backup_dir / "CT_concurrent_final.pt")
    torch.save(policy_module.state_dict(), run_final_model_path)
    print(f"Best model: {best_model_path}", flush=True)

    training_log_path = training_log_output_path(f"{run_name}_training_log.json")
    training_metrics_path = training_log_output_path(f"{run_name}_training_metrics.json")
    metrics_png = training_log_output_path(f"{run_name}_training_metrics_plot.png").with_suffix(".png")
    _save_training_log_json(log, training_log_path)
    _save_training_metrics_json(log, training_metrics_path)
    if log.get("reward"):
        plot_metrics(training_metrics_path, metrics_png, route_label="路径 concurrent")
        print(f"[artifact] metrics_plot={metrics_png}", flush=True)
    route_label = _artifact_route_label(Env_PN_Concurrent(device="cpu"))
    _postprocess_training_artifacts(
        best_model_path,
        run_name,
        config,
        route_label,
        concurrent=True,
    )
    return log, policy_module


def _alloc_rollout_buffers(
    n_steps: int,
    n_envs: int,
    obs_dim: int,
    n_actions: int,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    alloc_kwargs: dict[str, Any] = {"pin_memory": True} if pin_memory else {}
    return {
        "obs": torch.empty((n_steps, n_envs, obs_dim), dtype=torch.float32, **alloc_kwargs),
        "next_obs": torch.empty((n_steps, n_envs, obs_dim), dtype=torch.float32, **alloc_kwargs),
        "actions": torch.empty((n_steps, n_envs), dtype=torch.int64, **alloc_kwargs),
        "log_probs": torch.empty((n_steps, n_envs), dtype=torch.float32, **alloc_kwargs),
        "rewards": torch.empty((n_steps, n_envs), dtype=torch.float32, **alloc_kwargs),
        "dones": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "action_mask": torch.empty((n_steps, n_envs, n_actions), dtype=torch.bool, **alloc_kwargs),
        "finish": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "scrap": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "deadlock": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "time": torch.empty((n_steps, n_envs), dtype=torch.int64, **alloc_kwargs),
    }


def _alloc_rollout_buffers_concurrent(
    n_steps: int,
    n_envs: int,
    obs_dim: int,
    n_tm2: int,
    n_tm3: int,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    alloc_kwargs: dict[str, Any] = {"pin_memory": True} if pin_memory else {}
    return {
        "obs": torch.empty((n_steps, n_envs, obs_dim), dtype=torch.float32, **alloc_kwargs),
        "next_obs": torch.empty((n_steps, n_envs, obs_dim), dtype=torch.float32, **alloc_kwargs),
        "mask_tm2": torch.empty((n_steps, n_envs, n_tm2), dtype=torch.bool, **alloc_kwargs),
        "mask_tm3": torch.empty((n_steps, n_envs, n_tm3), dtype=torch.bool, **alloc_kwargs),
        "actions_tm2": torch.empty((n_steps, n_envs), dtype=torch.int64, **alloc_kwargs),
        "actions_tm3": torch.empty((n_steps, n_envs), dtype=torch.int64, **alloc_kwargs),
        "log_probs_tm2": torch.empty((n_steps, n_envs), dtype=torch.float32, **alloc_kwargs),
        "log_probs_tm3": torch.empty((n_steps, n_envs), dtype=torch.float32, **alloc_kwargs),
        "rewards": torch.empty((n_steps, n_envs), dtype=torch.float32, **alloc_kwargs),
        "dones": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "finish": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "scrap": torch.empty((n_steps, n_envs), dtype=torch.bool, **alloc_kwargs),
        "time": torch.empty((n_steps, n_envs), dtype=torch.int64, **alloc_kwargs),
    }


@torch.no_grad()
def collect_rollout_ultra_concurrent(
    env_fn,
    policy,
    n_steps,
    n_envs: int = 1,
    rollout_device: str = "cpu",
    state: dict[str, Any] | None = None,
    pin_memory: bool = False,
):
    """
    并发双动作高速 rollout（镜像 collect_rollout_ultra）：
    - 全程 CPU rollout + tensor buffer 预分配
    - 支持跨 batch 续采样（state 持有 env/obs/mask）
    - env_fn() 必须返回 Env_PN_Concurrent 实例
    """
    if str(rollout_device) != "cpu":
        raise ValueError("collect_rollout_ultra_concurrent 仅支持 rollout_device='cpu'")
    if state is None:
        state = {}

    n_envs = max(1, int(n_envs))
    if (state.get("n_envs") != n_envs) or ("env" not in state):
        if n_envs <= 1:
            env: FastEnvWrapper_Concurrent | VectorEnv_Concurrent = FastEnvWrapper_Concurrent(env_fn())
            obs_np, info = env.reset()
            obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
            mask_tm2 = torch.from_numpy(np.asarray(info["action_mask_tm2"], dtype=np.bool_)).unsqueeze(0)
            mask_tm3 = torch.from_numpy(np.asarray(info["action_mask_tm3"], dtype=np.bool_)).unsqueeze(0)
        else:
            env = VectorEnv_Concurrent(env_fn=env_fn, n_envs=n_envs)
            obs_np, info = env.reset()
            obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
            mask_tm2 = torch.from_numpy(np.asarray(info["action_mask_tm2"], dtype=np.bool_))
            mask_tm3 = torch.from_numpy(np.asarray(info["action_mask_tm3"], dtype=np.bool_))
        state["env"] = env
        state["obs"] = obs
        state["mask_tm2"] = mask_tm2
        state["mask_tm3"] = mask_tm3
        state["n_envs"] = n_envs

    env = state["env"]
    obs = state["obs"]
    mask_tm2 = state["mask_tm2"]
    mask_tm3 = state["mask_tm3"]
    obs_dim = int(obs.shape[-1])
    n_tm2 = int(mask_tm2.shape[-1])
    n_tm3 = int(mask_tm3.shape[-1])

    buffers = _alloc_rollout_buffers_concurrent(
        n_steps=int(n_steps),
        n_envs=n_envs,
        obs_dim=obs_dim,
        n_tm2=n_tm2,
        n_tm3=n_tm3,
        pin_memory=pin_memory,
    )

    policy = policy.to("cpu").eval()
    step_fn = env.step

    for t in range(int(n_steps)):
        buffers["obs"][t].copy_(obs)
        buffers["mask_tm2"][t].copy_(mask_tm2)
        buffers["mask_tm3"][t].copy_(mask_tm3)

        out = policy(obs)
        actions_tm2, log_probs_tm2 = _sample_actions_masked(out["logits_tm2"], mask_tm2)
        actions_tm3, log_probs_tm3 = _sample_actions_masked(out["logits_tm3"], mask_tm3)

        if n_envs <= 1:
            next_obs_np, rew, done, info = step_fn(int(actions_tm2[0].item()), int(actions_tm3[0].item()))
            next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)).unsqueeze(0)
            next_mask_tm2 = torch.from_numpy(np.asarray(info["action_mask_tm2"], dtype=np.bool_)).unsqueeze(0)
            next_mask_tm3 = torch.from_numpy(np.asarray(info["action_mask_tm3"], dtype=np.bool_)).unsqueeze(0)
            buffers["actions_tm2"][t, 0] = actions_tm2[0]
            buffers["actions_tm3"][t, 0] = actions_tm3[0]
            buffers["log_probs_tm2"][t, 0] = log_probs_tm2[0]
            buffers["log_probs_tm3"][t, 0] = log_probs_tm3[0]
            buffers["rewards"][t, 0] = float(rew)
            buffers["dones"][t, 0] = bool(done)
            buffers["finish"][t, 0] = bool(info.get("finish", False))
            buffers["scrap"][t, 0] = bool(info.get("scrap", False))
            buffers["time"][t, 0] = int(info.get("time", 0))
        else:
            next_obs_np, rewards_np, dones_np, info = step_fn(actions_tm2.numpy(), actions_tm3.numpy())
            next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32))
            next_mask_tm2 = torch.from_numpy(np.asarray(info["action_mask_tm2"], dtype=np.bool_))
            next_mask_tm3 = torch.from_numpy(np.asarray(info["action_mask_tm3"], dtype=np.bool_))
            buffers["actions_tm2"][t].copy_(actions_tm2)
            buffers["actions_tm3"][t].copy_(actions_tm3)
            buffers["log_probs_tm2"][t].copy_(log_probs_tm2)
            buffers["log_probs_tm3"][t].copy_(log_probs_tm3)
            buffers["rewards"][t].copy_(torch.from_numpy(np.asarray(rewards_np, dtype=np.float32)))
            buffers["dones"][t].copy_(torch.from_numpy(np.asarray(dones_np, dtype=np.bool_)))
            buffers["finish"][t].copy_(torch.from_numpy(np.asarray(info["finish"], dtype=np.bool_)))
            buffers["scrap"][t].copy_(torch.from_numpy(np.asarray(info["scrap"], dtype=np.bool_)))
            buffers["time"][t].copy_(torch.from_numpy(np.asarray(info["time"], dtype=np.int64)))

        buffers["next_obs"][t].copy_(next_obs)
        obs = next_obs
        mask_tm2 = next_mask_tm2
        mask_tm3 = next_mask_tm3

    state["obs"] = obs
    state["mask_tm2"] = mask_tm2
    state["mask_tm3"] = mask_tm3
    for key, value in list(buffers.items()):
        if isinstance(value, torch.Tensor):
            buffers[key] = value.contiguous()
    return buffers, state


def _flatten_rollout_concurrent(
    rollout: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    t_size, n_envs = rollout["actions_tm2"].shape
    flat = t_size * n_envs
    keep = min(int(max_frames), flat)
    obs_dim = rollout["obs"].shape[-1]
    n_tm2 = rollout["mask_tm2"].shape[-1]
    n_tm3 = rollout["mask_tm3"].shape[-1]
    return {
        "obs": rollout["obs"].reshape(flat, obs_dim)[:keep].contiguous(),
        "next_obs": rollout["next_obs"].reshape(flat, obs_dim)[:keep].contiguous(),
        "mask_tm2": rollout["mask_tm2"].reshape(flat, n_tm2)[:keep].contiguous(),
        "mask_tm3": rollout["mask_tm3"].reshape(flat, n_tm3)[:keep].contiguous(),
        "actions_tm2": rollout["actions_tm2"].reshape(flat)[:keep].contiguous(),
        "actions_tm3": rollout["actions_tm3"].reshape(flat)[:keep].contiguous(),
        "old_log_probs": (rollout["log_probs_tm2"] + rollout["log_probs_tm3"]).reshape(flat)[:keep].contiguous(),
        "rewards": rollout["rewards"].reshape(flat)[:keep].contiguous(),
        "dones": rollout["dones"].reshape(flat)[:keep].contiguous(),
        "finish": rollout["finish"].reshape(flat)[:keep].contiguous(),
        "scrap": rollout["scrap"].reshape(flat)[:keep].contiguous(),
        "time": rollout["time"].reshape(flat)[:keep].contiguous(),
        "advantages": advantages.reshape(flat)[:keep].contiguous(),
        "returns": returns.reshape(flat)[:keep].contiguous(),
    }


def _ppo_update_batched_concurrent(
    policy_backbone: nn.Module,
    value_net: nn.Module,
    optim: torch.optim.Optimizer,
    train_params: list[torch.nn.Parameter],
    batch: dict[str, torch.Tensor],
    num_epochs: int,
    clip_epsilon: float,
    entropy_coef: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    """
    并发双动作 Batched PPO 更新：
    联合 log_prob = log_prob_tm2 + log_prob_tm3，其余与单动作版本一致。
    """
    obs = batch["obs"]
    mask_tm2 = batch["mask_tm2"].bool()
    mask_tm3 = batch["mask_tm3"].bool()
    actions_tm2 = batch["actions_tm2"].long()
    actions_tm3 = batch["actions_tm3"].long()
    old_log_probs = batch["old_log_probs"].to(torch.float32)
    returns = batch["returns"].to(torch.float32)
    advantages = batch["advantages"].to(torch.float32)

    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp_min_(1e-6)
    advantages = (advantages - adv_mean) / adv_std

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(int(num_epochs)):
        perm = torch.randperm(obs.shape[0], device=obs.device)
        obs_e = obs.index_select(0, perm)
        mask_tm2_e = mask_tm2.index_select(0, perm)
        mask_tm3_e = mask_tm3.index_select(0, perm)
        act_tm2_e = actions_tm2.index_select(0, perm)
        act_tm3_e = actions_tm3.index_select(0, perm)
        old_lp_e = old_log_probs.index_select(0, perm)
        adv_e = advantages.index_select(0, perm)
        ret_e = returns.index_select(0, perm)

        with torch.autocast(
            device_type="cuda" if obs.is_cuda else "cpu",
            dtype=amp_dtype,
            enabled=amp_enabled and obs.is_cuda,
        ):
            out = policy_backbone(obs_e)
            value_pred = value_net(obs_e).squeeze(-1)

        new_lp_tm2, ent_tm2 = _masked_logprob_entropy(out["logits_tm2"].to(torch.float32), mask_tm2_e, act_tm2_e)
        new_lp_tm3, ent_tm3 = _masked_logprob_entropy(out["logits_tm3"].to(torch.float32), mask_tm3_e, act_tm3_e)
        new_log_prob = new_lp_tm2 + new_lp_tm3
        entropy = (ent_tm2 + ent_tm3).mean()

        ratio = torch.exp(new_log_prob - old_lp_e)
        surr1 = ratio * adv_e
        surr2 = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * adv_e
        policy_loss = -torch.minimum(surr1, surr2).mean()
        value_loss = 0.5 * (value_pred.to(torch.float32) - ret_e).pow(2).mean()
        loss = policy_loss + value_loss - float(entropy_coef) * entropy

        optim.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optim.step()

        last_policy_loss = float(policy_loss.detach().item())
        last_value_loss = float(value_loss.detach().item())
        last_entropy = float(entropy.detach().item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }


@torch.no_grad()
def collect_rollout_ultra(
    env_fn,
    policy,
    n_steps,
    n_envs=8,
    rollout_device="cpu",
    state: dict[str, Any] | None = None,
    pin_memory: bool = False,
):
    """
    工业级高速 rollout：
    - 全程 CPU rollout + tensor buffer
    - 纯 tensor 预分配 buffer
    - 可跨 batch 续采样（state 持有 env/obs/mask，不强制 reset）
    """
    if str(rollout_device) != "cpu":
        raise ValueError("collect_rollout_ultra 当前仅支持 rollout_device='cpu'")
    if state is None:
        state = {}

    n_envs = max(1, int(n_envs))
    if (state.get("n_envs") != n_envs) or ("env" not in state):
        if n_envs <= 1:
            env = FastEnvWrapper(env_fn())
            obs_np, info = env.reset()
            obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
            mask = torch.from_numpy(np.asarray(info["action_mask"], dtype=np.bool_)).unsqueeze(0)
        else:
            env = VectorEnv(env_fn=env_fn, n_envs=n_envs)
            obs_np, info = env.reset()
            obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
            mask = torch.from_numpy(np.asarray(info["action_mask"], dtype=np.bool_))
        state["env"] = env
        state["obs"] = obs
        state["mask"] = mask
        state["n_envs"] = n_envs

    env = state["env"]
    obs = state["obs"]
    mask = state["mask"]
    obs_dim = int(obs.shape[-1])
    n_act = int(mask.shape[-1])

    buffers = _alloc_rollout_buffers(
        n_steps=int(n_steps),
        n_envs=n_envs,
        obs_dim=obs_dim,
        n_actions=n_act,
        pin_memory=pin_memory,
    )

    policy = policy.to("cpu").eval()
    policy_fn = policy.forward
    step_fn = env.step

    for t in range(int(n_steps)):
        buffers["obs"][t].copy_(obs)
        buffers["action_mask"][t].copy_(mask)
        logits = policy_fn(obs)
        actions, log_probs = _sample_actions_masked(logits, mask)

        if n_envs <= 1:
            next_obs_np, rew, done, info = step_fn(int(actions[0].item()))
            next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)).unsqueeze(0)
            next_mask = torch.from_numpy(np.asarray(info["action_mask"], dtype=np.bool_)).unsqueeze(0)
            buffers["actions"][t, 0] = actions[0]
            buffers["log_probs"][t, 0] = log_probs[0]
            buffers["rewards"][t, 0] = float(rew)
            buffers["dones"][t, 0] = bool(done)
            buffers["finish"][t, 0] = bool(info.get("finish", False))
            buffers["scrap"][t, 0] = bool(info.get("scrap", False))
            buffers["deadlock"][t, 0] = bool(info.get("deadlock", False))
            buffers["time"][t, 0] = int(info.get("time", 0))
        else:
            next_obs_np, rewards_np, dones_np, info = step_fn(actions.numpy())
            next_obs = torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32))
            next_mask = torch.from_numpy(np.asarray(info["action_mask"], dtype=np.bool_))
            buffers["actions"][t].copy_(actions)
            buffers["log_probs"][t].copy_(log_probs)
            buffers["rewards"][t].copy_(torch.from_numpy(np.asarray(rewards_np, dtype=np.float32)))
            buffers["dones"][t].copy_(torch.from_numpy(np.asarray(dones_np, dtype=np.bool_)))
            buffers["finish"][t].copy_(torch.from_numpy(np.asarray(info.get("finish"), dtype=np.bool_)))
            buffers["scrap"][t].copy_(torch.from_numpy(np.asarray(info.get("scrap"), dtype=np.bool_)))
            buffers["deadlock"][t].copy_(torch.from_numpy(np.asarray(info.get("deadlock"), dtype=np.bool_)))
            buffers["time"][t].copy_(torch.from_numpy(np.asarray(info.get("time"), dtype=np.int64)))

        buffers["next_obs"][t].copy_(next_obs)
        obs = next_obs
        mask = next_mask

    state["obs"] = obs
    state["mask"] = mask
    buffers["n_actions"] = int(n_act)
    for key, value in list(buffers.items()):
        if isinstance(value, torch.Tensor):
            buffers[key] = value.contiguous()
    return buffers, state


def _gae_scan_impl(delta: torch.Tensor, not_done: torch.Tensor, gamma_lambda: float) -> torch.Tensor:
    """
    GAE 递推核心（按 [T, N] 扫描）。
    使用 torch.compile 包装后可显著减少 Python loop 开销。
    """
    t_size = int(delta.shape[0])
    n_envs = int(delta.shape[1])
    adv = torch.empty_like(delta)
    gae = torch.zeros((n_envs,), device=delta.device, dtype=delta.dtype)
    gl = torch.tensor(gamma_lambda, device=delta.device, dtype=delta.dtype)
    for t in range(t_size - 1, -1, -1):
        gae = delta[t] + gl * not_done[t] * gae
        adv[t] = gae
    return adv


if hasattr(torch, "compile"):
    try:
        _gae_scan_compiled = torch.compile(_gae_scan_impl, mode="reduce-overhead")
    except Exception:
        _gae_scan_compiled = _gae_scan_impl
else:
    _gae_scan_compiled = _gae_scan_impl

_gae_compile_failed = False


def _gae_scan(delta: torch.Tensor, not_done: torch.Tensor, gamma_lambda: float) -> torch.Tensor:
    global _gae_compile_failed
    # CPU 上直接走 eager，避免 torch.compile 首次编译导致本地训练“卡住”。
    if delta.device.type != "cuda":
        return _gae_scan_impl(delta, not_done, gamma_lambda)
    if _gae_compile_failed:
        return _gae_scan_impl(delta, not_done, gamma_lambda)
    try:
        return _gae_scan_compiled(delta, not_done, gamma_lambda)
    except Exception:
        _gae_compile_failed = True
        return _gae_scan_impl(delta, not_done, gamma_lambda)


@torch.no_grad()
def _compute_gae_and_returns(
    value_net: nn.Module,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    向量化 value 评估 + compile GAE:
    - obs/next_obs shape: [T, N, D]
    - rewards/dones shape: [T, N]
    """
    t_size, n_envs, obs_dim = obs.shape
    flat = t_size * n_envs
    obs_cat = torch.cat(
        [
            obs.reshape(flat, obs_dim),
            next_obs.reshape(flat, obs_dim),
        ],
        dim=0,
    )
    values_cat = value_net(obs_cat).squeeze(-1).to(torch.float32)
    values = values_cat[:flat].reshape(t_size, n_envs)
    next_values = values_cat[flat:].reshape(t_size, n_envs)
    not_done = 1.0 - dones.to(torch.float32)
    delta = rewards.to(torch.float32) + float(gamma) * next_values * not_done - values
    advantages = _gae_scan(delta, not_done, float(gamma) * float(gae_lambda))
    returns = advantages + values
    return advantages, returns


def _flatten_rollout_for_update(
    rollout: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    obs = rollout["obs"]
    next_obs = rollout["next_obs"]
    actions = rollout["actions"]
    log_probs = rollout["log_probs"]
    action_mask = rollout["action_mask"]
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    finish = rollout["finish"]
    scrap = rollout["scrap"]
    deadlock = rollout["deadlock"]
    time_arr = rollout["time"]

    t_size, n_envs = actions.shape
    flat = t_size * n_envs
    keep = min(int(max_frames), flat)

    return {
        "obs": obs.reshape(flat, -1)[:keep].contiguous(),
        "next_obs": next_obs.reshape(flat, -1)[:keep].contiguous(),
        "actions": actions.reshape(flat)[:keep].contiguous(),
        "old_log_probs": log_probs.reshape(flat)[:keep].contiguous(),
        "action_mask": action_mask.reshape(flat, action_mask.shape[-1])[:keep].contiguous(),
        "rewards": rewards.reshape(flat)[:keep].contiguous(),
        "dones": dones.reshape(flat)[:keep].contiguous(),
        "finish": finish.reshape(flat)[:keep].contiguous(),
        "scrap": scrap.reshape(flat)[:keep].contiguous(),
        "deadlock": deadlock.reshape(flat)[:keep].contiguous(),
        "time": time_arr.reshape(flat)[:keep].contiguous(),
        "advantages": advantages.reshape(flat)[:keep].contiguous(),
        "returns": returns.reshape(flat)[:keep].contiguous(),
    }


def _ppo_update_batched(
    policy_backbone: nn.Module,
    value_net: nn.Module,
    optim: torch.optim.Optimizer,
    train_params: list[torch.nn.Parameter],
    batch: dict[str, torch.Tensor],
    num_epochs: int,
    clip_epsilon: float,
    entropy_coef: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    """
    Batched PPO 更新：
    - 每个 epoch 单次大 batch forward（无 minibatch Python 内层循环）
    - 用 index_select + contiguous 保持访存连续
    """
    obs = batch["obs"]
    action_mask = batch["action_mask"].bool()
    actions = batch["actions"].long()
    old_log_probs = batch["old_log_probs"].to(torch.float32)
    returns = batch["returns"].to(torch.float32)
    advantages = batch["advantages"].to(torch.float32)

    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp_min_(1e-6)
    advantages = (advantages - adv_mean) / adv_std

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(int(num_epochs)):
        perm = torch.randperm(obs.shape[0], device=obs.device)
        obs_e = obs.index_select(0, perm)
        mask_e = action_mask.index_select(0, perm)
        act_e = actions.index_select(0, perm)
        old_lp_e = old_log_probs.index_select(0, perm)
        adv_e = advantages.index_select(0, perm)
        ret_e = returns.index_select(0, perm)

        with torch.autocast(
            device_type="cuda" if obs.is_cuda else "cpu",
            dtype=amp_dtype,
            enabled=amp_enabled and obs.is_cuda,
        ):
            logits = policy_backbone(obs_e)
            value_pred = value_net(obs_e).squeeze(-1)

        new_log_prob, entropy = _masked_logprob_entropy(logits.to(torch.float32), mask_e, act_e)
        ratio = torch.exp(new_log_prob - old_lp_e)
        surr1 = ratio * adv_e
        surr2 = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * adv_e
        policy_loss = -torch.minimum(surr1, surr2).mean()
        value_loss = 0.5 * (value_pred.to(torch.float32) - ret_e).pow(2).mean()
        entropy_mean = entropy.mean()
        loss = policy_loss + value_loss - float(entropy_coef) * entropy_mean

        optim.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optim.step()

        last_policy_loss = float(policy_loss.detach().item())
        last_value_loss = float(value_loss.detach().item())
        last_entropy = float(entropy_mean.detach().item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }


def _maybe_compile_model(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception:
        return model


def _save_training_log_json(log: defaultdict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: list(v) for k, v in log.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _artifact_route_label(env: Any) -> str | None:
    net = getattr(env, "net", env)
    rn = getattr(net, "single_route_name", None)
    if rn is None or str(rn).strip() == "":
        return None
    return f"路径 {rn}"


def _save_training_metrics_json(log: dict[str, list] | defaultdict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "reward": list(log.get("reward", [])),
        "makespan": list(log.get("makespan", [])),
        "finish": list(log.get("finish", [])),
        "scrap": list(log.get("scrap", [])),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _postprocess_training_artifacts(
    best_model_path: Path,
    run_name: str,
    config: PPOTrainingConfig,
    route_label: str | None,
    concurrent: bool = False,
) -> None:
    if not best_model_path.is_file():
        print("[artifact] 无 best 模型，跳过序列导出", flush=True)
        return
    safe_run_name = safe_name(
        run_name,
        "train_concurrent_run" if concurrent else "train_single_run",
    )
    seq_name = Path(action_sequence_path(safe_run_name)).stem
    gantt_png = gantt_output_path(f"{safe_run_name}_gantt.png")
    out = rollout_and_export(
        model_path=best_model_path,
        seed=int(config.seed),
        out_name=seq_name,
        concurrent=concurrent,
        retry=10,
        gantt_png_path=gantt_png,
        gantt_title_suffix=route_label,
    )
    seq_path = out["action_series_path"]
    print(f"[artifact] seq={seq_path}", flush=True)
    print(f"[artifact] gantt={gantt_png}", flush=True)


def train_single(
    config: PPOTrainingConfig | None = None,
    checkpoint_path: str | None = None,
    device_mode: str = "single",
    rollout_n_envs: int = 1,
    artifact_dir: str | Path | None = None,
    concurrent: bool = True,
):
    assert config is not None, "training config must be provided"

    if bool(concurrent):
        return _train_concurrent(
            config=config,
            checkpoint_path=checkpoint_path,
            artifact_dir=artifact_dir,
            rollout_n_envs=rollout_n_envs,
        )

    print("[Single PPO Training]", flush=True)
    print(config, flush=True)

    torch.manual_seed(config.seed)
    device = config.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"  警告: 请求 {device} 但 CUDA 不可用，回退到 cpu", flush=True)
            device = "cpu"
        else:
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    use_cuda_update = str(device).startswith("cuda")
    amp_enabled = use_cuda_update
    amp_dtype = torch.bfloat16 if (use_cuda_update and torch.cuda.is_bf16_supported()) else torch.float16
    scaler: torch.cuda.amp.GradScaler | None = None
    if amp_enabled and (amp_dtype == torch.float16):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    env = Env_PN_Single(device="cpu")
    # 仅保留 ultra 模式：rollout 固定在 CPU。
    rollout_device = "cpu"
    rollout_n_envs = max(1, int(rollout_n_envs))

    print(f"  计算设备: {device}", flush=True)
    print(f"  环境类型: {env.__class__.__name__}", flush=True)
    print(
        f"  Rollout Collector: ultra "
        f"(n_envs={rollout_n_envs}, rollout_device={rollout_device})",
        flush=True,
    )

    n_obs = env.observation_spec["observation"].shape[0]
    n_actions = env.n_actions
    print(f"  观测维度: {n_obs}", flush=True)
    print(f"  动作空间: {n_actions}", flush=True)

    policy_backbone = MaskedPolicyHead(
        hidden=config.n_hidden,
        n_obs=n_obs,
        n_actions=n_actions,
        n_layers=config.n_layer,
    ).to(device)
    policy_module = SingleActionPolicyModule(policy_backbone).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"从 checkpoint 加载: {checkpoint_path}", flush=True)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        try:
            policy_module.load_state_dict(state_dict)
        except RuntimeError:
            policy_backbone.load_state_dict(state_dict)

    value_net = nn.Sequential(
        nn.Linear(n_obs, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, 1),
    ).to(device)

    train_params = list(policy_backbone.parameters()) + list(value_net.parameters())
    optim = Adam(train_params, lr=config.lr, foreach=bool(use_cuda_update))

    # 仅编译 update 侧模型；rollout 侧始终使用 CPU 模型。
    policy_train = _maybe_compile_model(policy_backbone, enabled=use_cuda_update)
    value_train = _maybe_compile_model(value_net, enabled=use_cuda_update)

    rollout_policy = copy.deepcopy(policy_backbone).to("cpu").eval() if use_cuda_update else policy_backbone
    ultra_state: dict[str, Any] = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = model_output_path(f"single_{timestamp}").with_suffix("")
    backup_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = "CT_single"
    run_name = "train_single"
    if artifact_dir is not None:
        run_name = safe_name(Path(str(artifact_dir)).name, "train_single")
    best_model_path = model_output_path("CT_single_best.pt")
    run_best_model_path = model_output_path(f"{run_name}_best.pt")
    run_final_model_path = model_output_path(f"{run_name}_final.pt")
    best_reward = float("-inf")
    log = defaultdict(list)

    training_start = perf_counter()
    batch_time_window: deque[float] = deque(maxlen=10)
    total_env_steps = 0

    for batch_idx in range(config.total_batch):
        t_batch = perf_counter()

        # CPU rollout 前同步参数（一次性复制，避免 step 内频繁 .to()）。
        policy_backbone.eval()
        if use_cuda_update:
            with torch.no_grad():
                rollout_policy.load_state_dict(policy_backbone.state_dict(), strict=True)
            policy_for_rollout = rollout_policy
        else:
            policy_for_rollout = policy_backbone

        # Ultra rollout：每个环境采样 steps_per_env 步，再拉平成 frames_per_batch。
        steps_per_env = max(1, (int(config.frames_per_batch) + rollout_n_envs - 1) // rollout_n_envs)
        rollout_cpu, ultra_state = collect_rollout_ultra(
            env_fn=lambda: Env_PN_Single(device="cpu"),
            policy=policy_for_rollout,
            n_steps=steps_per_env,
            n_envs=rollout_n_envs,
            rollout_device=rollout_device,
            state=ultra_state,
            pin_memory=use_cuda_update,
        )

        t_rollout_end = perf_counter()

        available_frames = int(rollout_cpu["actions"].numel())
        used_frames = min(int(config.frames_per_batch), available_frames)
        total_env_steps += used_frames

        # CPU->GPU 单次搬运，更新阶段完全在训练设备。
        non_blocking = bool(use_cuda_update)
        rollout_dev = {
            "obs": rollout_cpu["obs"].to(device, non_blocking=non_blocking),
            "next_obs": rollout_cpu["next_obs"].to(device, non_blocking=non_blocking),
            "actions": rollout_cpu["actions"].to(device, non_blocking=non_blocking),
            "log_probs": rollout_cpu["log_probs"].to(device, non_blocking=non_blocking),
            "rewards": rollout_cpu["rewards"].to(device, non_blocking=non_blocking),
            "dones": rollout_cpu["dones"].to(device, non_blocking=non_blocking),
            "action_mask": rollout_cpu["action_mask"].to(device, non_blocking=non_blocking),
            "finish": rollout_cpu["finish"].to(device, non_blocking=non_blocking),
            "scrap": rollout_cpu["scrap"].to(device, non_blocking=non_blocking),
            "deadlock": rollout_cpu["deadlock"].to(device, non_blocking=non_blocking),
            "time": rollout_cpu["time"].to(device, non_blocking=non_blocking),
        }

        policy_train.train()
        value_train.train()
        with torch.no_grad():
            advantages, returns = _compute_gae_and_returns(
                value_net=value_train,
                obs=rollout_dev["obs"],
                next_obs=rollout_dev["next_obs"],
                rewards=rollout_dev["rewards"],
                dones=rollout_dev["dones"],
                gamma=float(config.gamma),
                gae_lambda=float(config.gae_lambda),
            )

        update_batch = _flatten_rollout_for_update(
            rollout=rollout_dev,
            advantages=advantages,
            returns=returns,
            max_frames=used_frames,
        )

        update_stats = _ppo_update_batched(
            policy_backbone=policy_train,
            value_net=value_train,
            optim=optim,
            train_params=train_params,
            batch=update_batch,
            num_epochs=int(config.num_epochs),
            clip_epsilon=float(config.clip_epsilon),
            entropy_coef=float(config.entropy_start),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )

        t_update_end = perf_counter()

        rollout_time = t_rollout_end - t_batch
        update_time = t_update_end - t_rollout_end
        batch_time = t_update_end - t_batch
        steps_per_sec = used_frames / rollout_time if rollout_time > 0 else 0.0
        batch_time_window.append(batch_time)
        remaining = config.total_batch - (batch_idx + 1)
        eta_s = remaining * (sum(batch_time_window) / len(batch_time_window))
        eta_str = f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"

        # 统计指标使用 CPU rollout，避免额外 GPU 同步。
        flat_rewards = rollout_cpu["rewards"].reshape(-1)[:used_frames]
        flat_finish = rollout_cpu["finish"].reshape(-1).bool()[:used_frames]
        flat_scrap = rollout_cpu["scrap"].reshape(-1).bool()[:used_frames]
        flat_deadlock = rollout_cpu["deadlock"].reshape(-1).bool()[:used_frames]
        flat_time = rollout_cpu["time"].reshape(-1)[:used_frames]

        ep_reward = float(flat_rewards.sum().item())
        finish_count = int(flat_finish.sum().item())
        scrap_count = int(flat_scrap.sum().item())
        deadlock_count = int(flat_deadlock.sum().item())
        avg_makespan = float(flat_time[flat_finish].float().mean().item()) if finish_count > 0 else 0.0

        log["reward"].append(ep_reward)
        log["finish"].append(finish_count)
        log["scrap"].append(scrap_count)
        log["deadlock"].append(deadlock_count)
        log["makespan"].append(avg_makespan)
        log["rollout_time"].append(rollout_time)
        log["update_time"].append(update_time)
        log["steps_per_sec"].append(steps_per_sec)
        log["frames"].append(used_frames)
        log["policy_loss"].append(update_stats["policy_loss"])
        log["value_loss"].append(update_stats["value_loss"])
        log["entropy"].append(update_stats["entropy"])

        print(
            f"batch {batch_idx+1:04d} | reward={ep_reward:.2f} | finish={finish_count} "
            f"| scrap={scrap_count} | deadlock={deadlock_count} | makespan={avg_makespan:.1f} "
            f"| rollout={rollout_time:.2f}s update={update_time:.2f}s "
            f"| steps/s={steps_per_sec:.0f} | p={update_stats['policy_loss']:.4f} "
            f"v={update_stats['value_loss']:.4f} ent={update_stats['entropy']:.4f} ETA={eta_str}",
            flush=True,
        )

        if ep_reward > best_reward and finish_count > 0:
            best_reward = ep_reward
            torch.save(policy_module.state_dict(), best_model_path)
            torch.save(policy_module.state_dict(), run_best_model_path)
            backup_path = os.path.join(str(backup_dir), f"{model_prefix}_best.pt")
            torch.save(policy_module.state_dict(), backup_path)
            print(f"  -> New best model! reward={ep_reward:.2f}", flush=True)

    total_training_time = perf_counter() - training_start
    total_rollout = sum(log["rollout_time"])
    total_update = sum(log["update_time"])
    avg_rollout = total_rollout / len(log["rollout_time"]) if log["rollout_time"] else 0.0
    avg_update = total_update / len(log["update_time"]) if log["update_time"] else 0.0
    avg_batch = avg_rollout + avg_update
    overall_sps = (total_env_steps / total_rollout) if total_rollout > 0 else 0.0
    rollout_pct = (avg_rollout / avg_batch * 100) if avg_batch > 0 else 0.0
    update_pct = (avg_update / avg_batch * 100) if avg_batch > 0 else 0.0

    print("\n[Training Summary]", flush=True)
    print(f"  总训练时间: {total_training_time:.1f}s ({total_training_time / 60:.1f}m)", flush=True)
    print(f"  总 env steps: {total_env_steps}", flush=True)
    print(
        f"  平均 batch: {avg_batch:.2f}s "
        f"(rollout={avg_rollout:.2f}s [{rollout_pct:.0f}%] "
        f"| update={avg_update:.2f}s [{update_pct:.0f}%])",
        flush=True,
    )
    print(f"  平均 steps/sec: {overall_sps:.0f}", flush=True)
    print(f"  Best reward: {best_reward:.2f}", flush=True)

    final_path = os.path.join(str(backup_dir), f"{model_prefix}_final.pt")
    torch.save(policy_module.state_dict(), final_path)
    torch.save(policy_module.state_dict(), run_final_model_path)
    print(f"Final model: {final_path}", flush=True)
    print(f"Best model: {best_model_path}", flush=True)
    route_label = _artifact_route_label(env)
    training_log_path = training_log_output_path(f"{run_name}_training_log.json")
    training_metrics_path = training_log_output_path(f"{run_name}_training_metrics.json")
    metrics_png = training_log_output_path(f"{run_name}_training_metrics_plot.png").with_suffix(".png")
    _save_training_log_json(log, training_log_path)
    _save_training_metrics_json(log, training_metrics_path)
    if log.get("reward"):
        plot_metrics(
            training_metrics_path,
            metrics_png,
            route_label=route_label,
        )
        print(f"[artifact] metrics_plot={metrics_png}", flush=True)
    _postprocess_training_artifacts(best_model_path, run_name, config, route_label, concurrent=False)
    return log, policy_backbone


def build_single_env() -> Env_PN_Single:
    return Env_PN_Single()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 训练（默认并发双动作，可切换单动作）")
    parser.add_argument("--device", type=str, default="single", choices=["single", "cascade"], help="设备模式（single/cascade）")
    parser.add_argument("--compute-device", type=str, default=None, help="计算设备：cpu / cuda / cuda:0；未指定时：有 CUDA 则用 cuda，否则用配置文件中的 device")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--rollout-n-envs", type=int, default=1, help="ultra collector 并行环境数")
    parser.add_argument("--concurrent", dest="concurrent", action="store_true", default=True, help="启用并发双动作训练（默认开启）")
    parser.add_argument("--no-concurrent", dest="concurrent", action="store_false", help="关闭并发，回退到单动作训练")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="运行名称（用于 results 下文件名前缀），例如 exp_001",
    )
    args = parser.parse_args()

    root = Path(__file__).parents[2]
    path = root / "config" / "training" / "s_train.yaml"
    cfg = PPOTrainingConfig.load(path)
    if args.compute_device is not None:
        cfg.device = args.compute_device.strip()
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    print(f"从 {path} 加载配置", flush=True)
    if args.checkpoint is not None:
        checkpoint_path = model_output_path(args.checkpoint)
    else:
        checkpoint_path = None

    train_single(
        config=cfg,
        checkpoint_path=checkpoint_path,
        device_mode=args.device,
        rollout_n_envs=args.rollout_n_envs,
        artifact_dir=args.artifact_dir,
        concurrent=bool(args.concurrent),
    )
