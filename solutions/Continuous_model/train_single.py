"""
单设备单动作 PPO 训练脚本（两阶段 release 追责回填）。
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torchrl.modules import MaskedCategorical
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from data.ppo_configs.training_config import PPOTrainingConfig
from solutions.Continuous_model.env_single import Env_PN_Single
from solutions.PPO.network.models import MaskedPolicyHead
from pathlib import Path

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


def _extract_step_result(td_next):
    if "next" in td_next.keys():
        reward = td_next["next", "reward"]
        terminated = td_next["next", "terminated"]
        finish = td_next["next", "finish"] if "finish" in td_next["next"].keys() else td_next["next", "terminated"]
        scrap = td_next["next", "scrap"] if "scrap" in td_next["next"].keys() else torch.tensor(False)
        deadlock = td_next["next", "deadlock"] if "deadlock" in td_next["next"].keys() else torch.tensor(False)
        time_t = td_next["next", "time"]
        next_obs = td_next["next", "observation"]
    else:
        reward = td_next["reward"]
        terminated = td_next["terminated"]
        finish = td_next.get("finish", td_next["terminated"])
        scrap = td_next.get("scrap", torch.tensor(False))
        deadlock = td_next.get("deadlock", torch.tensor(False))
        time_t = td_next["time"]
        next_obs = td_next["observation"]
    return reward, terminated, finish, scrap, deadlock, time_t, next_obs


def _to_next_state(td_next):
    return td_next["next"].clone() if "next" in td_next.keys() else td_next.clone()


def collect_rollout_single(
    env: Env_PN_Single,
    policy_backbone: MaskedPolicyHead,
    n_steps: int,
    device: str = "cpu",
    blame_enabled: bool = False,
):
    """
    两阶段采样：
    1) 收集轨迹（step 不施加 release 惩罚）
    2) 若 blame_enabled，episode 结束后 blame_release_violations 回填奖励
    """
    data = {
        "observation": [],
        "observation_f": [],
        "action_mask": [],
        "action": [],
        "log_prob": [],
        "reward": [],
        "done": [],
        "finish": [],
        "scrap": [],
        "deadlock": [],
        "time": [],
        "next_observation": [],
        "next_observation_f": [],
    }

    fire_log_ranges = []
    td = env.reset()
    second_pass_events = 0

    for _ in range(n_steps):
        obs = td["observation"].unsqueeze(0).to(device)
        obs_f = obs.float()
        mask = td["action_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = policy_backbone(obs_f)
            dist = MaskedCategorical(logits=logits, mask=mask.bool())
            action = dist.sample()
            log_prob = dist.log_prob(action)

        data["observation"].append(td["observation"])
        data["observation_f"].append(td["observation"].float())
        data["action_mask"].append(td["action_mask"])
        data["action"].append(action.squeeze(0).cpu())
        data["log_prob"].append(log_prob.squeeze(0).cpu())

        fire_start = len(env.net.fire_log)

        step_td = td.clone()
        step_td["action"] = action.squeeze(0).cpu()
        td_next = env.step(step_td)

        fire_end = len(env.net.fire_log)
        fire_log_ranges.append((fire_start, fire_end))

        reward, terminated, finish, scrap, deadlock, time_t, next_obs = _extract_step_result(td_next)
        data["reward"].append(reward.cpu())
        data["done"].append(terminated.cpu())
        data["finish"].append(finish.cpu())
        data["scrap"].append(scrap.cpu())
        data["deadlock"].append(deadlock.cpu())
        data["time"].append(time_t.cpu())
        data["next_observation"].append(next_obs.cpu())
        data["next_observation_f"].append(next_obs.float().cpu())

        if bool(terminated.item() if hasattr(terminated, "item") else terminated):
            if blame_enabled:
                blame = env.net.blame_release_violations()
                if blame:
                    for fire_idx, penalty in blame.items():
                        for step_idx, (fstart, fend) in enumerate(fire_log_ranges):
                            if fstart <= fire_idx < fend:
                                data["reward"][step_idx] = data["reward"][step_idx] - float(penalty)
                                second_pass_events += 1
                                break
            td = env.reset()
            fire_log_ranges = []
        else:
            td = _to_next_state(td_next)

    rollout = TensorDict({k: torch.stack(v) for k, v in data.items()}, batch_size=[n_steps])
    return rollout, second_pass_events


def train_single(
    config: PPOTrainingConfig | None = None,
    checkpoint_path: str | None = None,
    device_mode: str = "single",
    proc_time_rand_enabled: bool | None = None,
    proc_time_rand_scale_map: dict[str, dict[str, float]] | None = None,
    blame_enabled: bool = False,
):
    assert config is not None, "training config must be provided"

    print("[Single PPO Training]")
    print(config)

    torch.manual_seed(config.seed)
    device = config.device
    env = Env_PN_Single(
        detailed_reward=True,
        device_mode=device_mode,
        proc_time_rand_enabled=proc_time_rand_enabled,
        proc_time_rand_scale_map=proc_time_rand_scale_map,
    )
    print(f"  环境类型: {env.__class__.__name__}")

    n_obs = env.observation_spec["observation"].shape[0]
    n_actions = env.n_actions
    print(f"  观测维度: {n_obs}")
    print(f"  动作空间: {n_actions}")

    policy_backbone = MaskedPolicyHead(
        hidden=config.n_hidden,
        n_obs=n_obs,
        n_actions=n_actions,
        n_layers=config.n_layer,
    ).to(device)
    policy_module = SingleActionPolicyModule(policy_backbone).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"从 checkpoint 加载: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        try:
            # 新格式：与并发训练一致，保存 policy_module.state_dict()
            policy_module.load_state_dict(state_dict)
        except RuntimeError:
            # 兼容旧格式：仅 backbone.state_dict()
            policy_backbone.load_state_dict(state_dict)

    value_net = nn.Sequential(
        nn.Linear(n_obs, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, 1),
    ).to(device)

    optim = Adam(list(policy_backbone.parameters()) + list(value_net.parameters()), lr=config.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    backup_dir = os.path.join(saved_models_dir, f"single_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    model_prefix = "CT_single"
    best_model_path = Path(__file__).resolve().parents[2] / "models" / "tmp.pt"
    best_reward = float("-inf")
    log = defaultdict(list)

    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx in range(config.total_batch):
            rollout, second_pass_events = collect_rollout_single(
                env, policy_backbone, config.frames_per_batch, device=device, blame_enabled=blame_enabled
            )
            rollout = rollout.to(device)

            with torch.no_grad():
                values = value_net(rollout["observation_f"]).squeeze(-1)
                next_values = value_net(rollout["next_observation_f"]).squeeze(-1)
                rewards = rollout["reward"].squeeze(-1)
                dones = rollout["done"].float().squeeze(-1)
                delta = rewards + config.gamma * next_values * (1 - dones) - values
                advantages = torch.zeros_like(rewards)
                gae = 0.0
                for t in reversed(range(len(rewards))):
                    gae = delta[t] + config.gamma * config.gae_lambda * (1 - dones[t]) * gae
                    advantages[t] = gae
                returns = advantages + values

            rollout["advantage"] = advantages
            rollout["value_target"] = returns

            for _ in range(config.num_epochs):
                indices = torch.randperm(len(rollout))
                for start in range(0, len(rollout), config.sub_batch_size):
                    end = start + config.sub_batch_size
                    batch = rollout[indices[start:end]]

                    obs_f = batch["observation_f"]
                    mask = batch["action_mask"].bool()
                    old_action = batch["action"]
                    old_log_prob = batch["log_prob"]
                    adv = batch["advantage"]
                    ret = batch["value_target"]

                    logits = policy_backbone(obs_f)
                    dist = MaskedCategorical(logits=logits, mask=mask)
                    new_log_prob = dist.log_prob(old_action)

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                    surr1 = ratio * adv_norm
                    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * adv_norm
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred = value_net(obs_f).squeeze(-1)
                    value_loss = 0.5 * (value_pred - ret).pow(2).mean()
                    entropy_loss = -config.entropy_start * dist.entropy().mean()
                    loss = policy_loss + value_loss + entropy_loss

                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(policy_backbone.parameters()) + list(value_net.parameters()), max_norm=1.0)
                    optim.step()

            ep_reward = rollout["reward"].sum().item()
            finish_count = rollout["finish"].sum().item()
            scrap_count = rollout["scrap"].sum().item()
            deadlock_count = rollout["deadlock"].sum().item()

            finish_mask = rollout["finish"].squeeze(-1).bool()
            if finish_mask.any():
                finish_times = rollout["time"][finish_mask].squeeze(-1).float()
                avg_makespan = finish_times.mean().item()
            else:
                avg_makespan = 0.0

            log["reward"].append(ep_reward)
            log["finish"].append(finish_count)
            log["scrap"].append(scrap_count)
            log["deadlock"].append(deadlock_count)
            log["makespan"].append(avg_makespan)
            log["second_pass_events"].append(second_pass_events)

            print(
                f"batch {batch_idx+1:04d} | reward={ep_reward:.2f} | finish={int(finish_count)} "
                f"| scrap={int(scrap_count)} | deadlock={int(deadlock_count)} "
                f"| makespan={avg_makespan:.1f} | second_pass={second_pass_events}"
            )

            if ep_reward > best_reward and finish_count > 0:
                best_reward = ep_reward
                torch.save(policy_module.state_dict(), best_model_path)
                backup_path = os.path.join(backup_dir, f"{model_prefix}_best.pt")
                torch.save(policy_module.state_dict(), backup_path)
                print(f"  -> New best model! reward={ep_reward:.2f}")

    print(f"\nTraining done. Best reward: {best_reward:.2f}")
    final_path = os.path.join(backup_dir, f"{model_prefix}_final.pt")
    torch.save(policy_module.state_dict(), final_path)
    print(f"Final model: {final_path}")
    print(f"Best model: {best_model_path}")
    return log, policy_backbone


def build_single_env() -> Env_PN_Single:
    return Env_PN_Single(detailed_reward=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单设备单动作 PPO 训练（两阶段 release 回填）")
    parser.add_argument("--device", type=str, default="single", choices=["single", "cascade"], help="设备模式")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--proc-time-rand-enabled", action="store_true", help="开启单设备工序时间随机扰动")
    parser.add_argument("--blame", action="store_true", help="开启二次追责（episode 结束后 blame_release_violations 回填惩罚）")
    args = parser.parse_args()

    root = Path(__file__).parents[2]
    path = root / "data" / "ppo_configs" / "s_train.json"
    cfg = PPOTrainingConfig.load(path)
    print(f"从 {path} 加载配置")
    if args.checkpoint is not None:
        checkpoint_path = root / "models" / args.checkpoint
    else:
        checkpoint_path = None
    train_single(
        config=cfg,
        checkpoint_path=checkpoint_path,
        device_mode=args.device,
        proc_time_rand_enabled=True if args.proc_time_rand_enabled else None,
        blame_enabled=args.blame,
    )
