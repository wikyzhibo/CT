from __future__ import annotations

import os
import json
from math import ceil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict


import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.optim import Adam

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import ParallelEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from solutions.B.clustertool_config import ClusterToolCfg
from solutions.B.training_config import TrainingConfig
from solutions.B.Env import Env
from solutions.B.ppo_models import MaskedPolicyHead


def build_policy_actor(
    obs_dim: int,
    n_actions: int,
    n_hidden: int,
    n_layer: int,
    device: str,
) -> ProbabilisticActor:
    policy_backbone = MaskedPolicyHead(
        hidden=n_hidden,
        n_obs=obs_dim,
        n_actions=n_actions,
        n_layers=n_layer,
    )
    td_module = TensorDictModule(policy_backbone, in_keys=["observation_f"], out_keys=["logits"])
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)
    return policy


def _build_parallel_env(
    num_envs: int,
    device: str,
    base_seed: int,
    clustertool_cfg: ClusterToolCfg,
) -> ParallelEnv:
    def _make_env(rank: int) -> Callable[[], Env]:
        return lambda: Env(
            device=device,
            seed=base_seed + rank,
            clustertool_cfg=clustertool_cfg,
        )

    return ParallelEnv(num_workers=num_envs, create_env_fn=[_make_env(i) for i in range(num_envs)])


def collect_rollout(
    env: ParallelEnv,
    policy: ProbabilisticActor,
    frames_per_batch: int,
) -> TensorDict:
    current_td = env.reset()
    num_envs = int(current_td.batch_size[0])
    rollout_steps = ceil(frames_per_batch / max(1, num_envs))
    transitions = []
    for _ in range(rollout_steps):
        policy_td = policy(current_td)
        step_td = policy_td.clone()
        if step_td.get("action").ndim == 1:
            step_td.set("action", step_td.get("action").unsqueeze(-1))
        stepped_td = env.step(step_td)
        transition = stepped_td.select("observation_f", "action_mask", "time", "next").detach().clone()
        transition.set("action", policy_td.get("action").detach().clone())
        transition.set("action_log_prob", policy_td.get("action_log_prob").detach().clone())
        transitions.append(transition)

        next_td = stepped_td.get("next").clone()
        done_mask = next_td.get("done")
        if done_mask.any():
            reset_mask = TensorDict({"_reset": done_mask.clone()}, batch_size=next_td.batch_size)
            reset_td = env.reset(reset_mask)
            done_idx = done_mask.squeeze(-1)
            for key in ("observation_f", "action_mask", "time", "done", "finish", "scrap", "terminated"):
                patched = next_td.get(key).clone()
                patched[done_idx] = reset_td.get(key).clone()[done_idx]
                next_td.set(key, patched)

        current_td = next_td

    rollout_td = torch.stack(transitions, dim=0).reshape(-1)
    if rollout_td.numel() > frames_per_batch:
        rollout_td = rollout_td[:frames_per_batch]
    return rollout_td


def _training_metrics_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "training_metrics.json"


def train(
    training_config_path: str | None = None,
    cluster_tool_config_path: str | None = None,
):
    training_cfg = TrainingConfig.load(training_config_path)
    cluster_tool_cfg = ClusterToolCfg.load(cluster_tool_config_path)
    device = training_cfg.device

    torch.manual_seed(training_cfg.seed)
    env = _build_parallel_env(
        num_envs=training_cfg.num_envs,
        device=device,
        base_seed=training_cfg.seed,
        clustertool_cfg=cluster_tool_cfg,
    )

    project_root = os.path.dirname(os.path.dirname(__file__))
    saved_models_dir = os.path.join(project_root, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(saved_models_dir, f"pdr_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    best_model_path = os.path.join(saved_models_dir, "CT_pdr_best.pt")
    latest_model_path = os.path.join(saved_models_dir, "CT_pdr_latest.pt")

    obs_dim = int(env.observation_spec["observation_f"].shape[-1])
    n_actions = int(env.action_spec.space.n)
    policy = build_policy_actor(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_hidden=training_cfg.n_hidden,
        n_layer=training_cfg.n_layer,
        device=device,
    )
    value_module = ValueOperator(
        module=nn.Sequential(
            nn.Linear(obs_dim, training_cfg.n_hidden),
            nn.ReLU(),
            nn.Linear(training_cfg.n_hidden, training_cfg.n_hidden),
            nn.ReLU(),
            nn.Linear(training_cfg.n_hidden, training_cfg.n_hidden),
            nn.ReLU(),
            nn.Linear(training_cfg.n_hidden, 1),
        ),
        in_keys=["observation_f"],
    ).to(device)
    optim = Adam(list(policy.parameters()) + list(value_module.parameters()), lr=training_cfg.lr)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=training_cfg.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    gae = GAE(gamma=training_cfg.gamma, lmbda=training_cfg.gae_lambda, value_network=value_module)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=training_cfg.clip_epsilon,
        entropy_coeff=training_cfg.entropy_start,
        critic_coeff=training_cfg.critic_coeff,
        normalize_advantage=True,
    )

    frame_count = 0
    best_reward = float("-inf")
    log = defaultdict(list)
    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx in range(training_cfg.total_batch):
            tensordict_data = collect_rollout(
                env=env,
                policy=policy,
                frames_per_batch=training_cfg.frames_per_batch,
            )
            frac = min(1.0, batch_idx / max(1, training_cfg.total_batch))
            entropy_coeff = training_cfg.entropy_start + (training_cfg.entropy_end - training_cfg.entropy_start) * frac
            loss_module.entropy_coeff.copy_(
                torch.tensor(
                    entropy_coeff,
                    device=loss_module.entropy_coeff.device,
                    dtype=loss_module.entropy_coeff.dtype,
                )
            )

            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))
            replay_buffer.extend(tensordict_data)

            for _ in range(training_cfg.num_epochs):
                for _ in range(training_cfg.frames_per_batch // training_cfg.sub_batch_size):
                    subdata = replay_buffer.sample(training_cfg.sub_batch_size).to(device)
                    loss_vals = loss_module(subdata)
                    loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_module.parameters()),
                        max_norm=training_cfg.grad_clip_norm,
                    )
                    optim.step()

            frame_count += int(tensordict_data.numel())
            batch_reward = float(tensordict_data["next", "reward"].sum().item())
            finish_mask = tensordict_data["next", "finish"]
            scrap_mask = tensordict_data["next", "scrap"]
            finish_count = int(finish_mask.sum().item())
            scrap_count = int(scrap_mask.sum().item())
            finished_times = tensordict_data["next", "time"][finish_mask]
            mean_makespan = (
                float(finished_times.float().mean().item())
                if finish_count > 0
                else 0.0
            )

            print(
                f"batch {batch_idx + 1:04d} | frames={frame_count} | "
                f"sum_reward={batch_reward:.2f} | finish={finish_count} | "
                f"scrap={scrap_count} | makespan={mean_makespan:.2f}",
                flush=True,
            )
            log["reward"].append(batch_reward)
            log["finish"].append(finish_count)
            log["scrap"].append(scrap_count)
            log["makespan"].append(mean_makespan)

            if batch_reward > best_reward and finish_count > 0:
                best_reward = batch_reward
                torch.save(policy.state_dict(), best_model_path)
                torch.save(policy.state_dict(), os.path.join(backup_dir, "CT_pdr_best.pt"))
                print(f"  -> New best model saved! reward={batch_reward:.2f}", flush=True)

    env.close()
    torch.save(policy.state_dict(), os.path.join(backup_dir, "CT_pdr_final.pt"))
    torch.save(policy.state_dict(), latest_model_path)
    metrics_path = _training_metrics_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "reward": list(log["reward"]),
        "finish": list(log["finish"]),
        "scrap": list(log["scrap"]),
        "makespan": list(log["makespan"]),
        "meta": {
            "candidate_k": int(cluster_tool_cfg.candidate_k),
            "search_depth": int(cluster_tool_cfg.search_depth),
            "seed": int(training_cfg.seed),
            "total_batch": int(training_cfg.total_batch),
            "timestamp": timestamp,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Training done. Best reward: {best_reward:.2f}", flush=True)
    print(f"Best model: {best_model_path}", flush=True)
    print(f"Latest model: {latest_model_path}", flush=True)
    print(f"Backup folder: {backup_dir}", flush=True)
    print(f"Training metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    train()


