"""
单设备单动作 PPO 训练脚本（两阶段 release 追责回填）。
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from datetime import datetime
from time import perf_counter

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


def collect_rollout_single(
    env: Env_PN_Single,
    policy_backbone: MaskedPolicyHead,
    n_steps: int,
    device: str = "cpu",
    blame_enabled: bool = False,
):
    """
    Rollout 采样（预分配 tensor 版本）。
    blame_enabled=False 时跳过 fire_log_ranges 追踪。
    """
    obs_dim = env.observation_spec["observation"].shape[0]
    n_act = env.n_actions
    on_cpu = (device == "cpu")

    obs_buf = torch.zeros(n_steps, obs_dim, dtype=torch.float32)
    mask_buf = torch.zeros(n_steps, n_act, dtype=torch.bool)
    action_buf = torch.zeros(n_steps, dtype=torch.int64)
    log_prob_buf = torch.zeros(n_steps, dtype=torch.float32)
    reward_buf = torch.zeros(n_steps, 1, dtype=torch.float32)
    done_buf = torch.zeros(n_steps, dtype=torch.bool)
    finish_buf = torch.zeros(n_steps, dtype=torch.bool)
    scrap_buf = torch.zeros(n_steps, dtype=torch.bool)
    deadlock_buf = torch.zeros(n_steps, dtype=torch.bool)
    time_buf = torch.zeros(n_steps, 1, dtype=torch.int64)
    next_obs_buf = torch.zeros(n_steps, obs_dim, dtype=torch.float32)

    _FALSE = torch.tensor(False)
    fire_log_ranges: list | None = [] if blame_enabled else None
    td = env.reset()
    second_pass_events = 0

    for i in range(n_steps):
        obs = td["observation"].unsqueeze(0).to(device)
        mask = td["action_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = policy_backbone(obs.float())
            dist = MaskedCategorical(logits=logits, mask=mask.bool())
            action = dist.sample()
            log_prob = dist.log_prob(action)

        obs_buf[i] = td["observation"]
        mask_buf[i] = td["action_mask"]
        a = action.squeeze(0)
        lp = log_prob.squeeze(0)
        if not on_cpu:
            a = a.cpu()
            lp = lp.cpu()
        action_buf[i] = a
        log_prob_buf[i] = lp

        if blame_enabled:
            fire_start = len(env.net.fire_log)

        step_td = td.clone()
        step_td["action"] = a
        td_next = env.step(step_td)

        if blame_enabled:
            fire_end = len(env.net.fire_log)
            fire_log_ranges.append((fire_start, fire_end))

        nxt = td_next["next"]
        reward = nxt["reward"]
        terminated = nxt["terminated"]
        if on_cpu:
            reward_buf[i] = reward
            done_buf[i] = terminated
            finish_buf[i] = nxt.get("finish", terminated)
            scrap_buf[i] = nxt.get("scrap", _FALSE)
            deadlock_buf[i] = nxt.get("deadlock", _FALSE)
            time_buf[i] = nxt["time"]
            next_obs_buf[i] = nxt["observation"]
        else:
            reward_buf[i] = reward.cpu()
            done_buf[i] = terminated.cpu()
            finish_buf[i] = nxt.get("finish", terminated).cpu()
            scrap_buf[i] = nxt.get("scrap", _FALSE).cpu()
            deadlock_buf[i] = nxt.get("deadlock", _FALSE).cpu()
            time_buf[i] = nxt["time"].cpu()
            next_obs_buf[i] = nxt["observation"].cpu()

        if bool(terminated.item() if hasattr(terminated, "item") else terminated):
            if blame_enabled:
                blame = env.net.blame_release_violations()
                if blame:
                    for fire_idx, penalty in blame.items():
                        for step_idx, (fstart, fend) in enumerate(fire_log_ranges):
                            if fstart <= fire_idx < fend:
                                reward_buf[step_idx] -= float(penalty)
                                second_pass_events += 1
                                break
                fire_log_ranges.clear()
            td = env.reset()
        else:
            td = nxt.clone()

    rollout = TensorDict({
        "observation": obs_buf,
        "observation_f": obs_buf.float(),
        "action_mask": mask_buf,
        "action": action_buf,
        "log_prob": log_prob_buf,
        "reward": reward_buf,
        "done": done_buf,
        "finish": finish_buf,
        "scrap": scrap_buf,
        "deadlock": deadlock_buf,
        "time": time_buf,
        "next_observation": next_obs_buf,
        "next_observation_f": next_obs_buf.float(),
    }, batch_size=[n_steps])
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
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"  警告: 请求 {device} 但 CUDA 不可用，回退到 cpu", flush=True)
            device = "cpu"
        else:
            torch.cuda.manual_seed_all(config.seed)
    env = Env_PN_Single(
        device="cpu",
        detailed_reward=True,
        device_mode=device_mode,
        proc_time_rand_enabled=proc_time_rand_enabled,
        proc_time_rand_scale_map=proc_time_rand_scale_map,
    )
    print(f"  计算设备: {device}", flush=True)
    print(f"  环境类型: {env.__class__.__name__}", flush=True)

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

    training_start = perf_counter()
    batch_time_window: deque[float] = deque(maxlen=10)

    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx in range(config.total_batch):
            t_batch = perf_counter()

            rollout, second_pass_events = collect_rollout_single(
                env, policy_backbone, config.frames_per_batch, device=device, blame_enabled=blame_enabled
            )
            t_rollout_end = perf_counter()

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

            t_update_end = perf_counter()

            rollout_time = t_rollout_end - t_batch
            update_time = t_update_end - t_rollout_end
            batch_time = t_update_end - t_batch
            steps_per_sec = config.frames_per_batch / rollout_time if rollout_time > 0 else 0.0
            batch_time_window.append(batch_time)
            remaining = config.total_batch - (batch_idx + 1)
            eta_s = remaining * (sum(batch_time_window) / len(batch_time_window))
            eta_str = f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"

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
            log["rollout_time"].append(rollout_time)
            log["update_time"].append(update_time)
            log["steps_per_sec"].append(steps_per_sec)

            print(
                f"batch {batch_idx+1:04d} | reward={ep_reward:.2f} | finish={int(finish_count)} "
                f"| scrap={int(scrap_count)} | deadlock={int(deadlock_count)} "
                f"| makespan={avg_makespan:.1f}"
                f" | rollout={rollout_time:.2f}s update={update_time:.2f}s"
                f" | steps/s={steps_per_sec:.0f} ETA={eta_str}",
                flush=True,
            )

            if ep_reward > best_reward and finish_count > 0:
                best_reward = ep_reward
                torch.save(policy_module.state_dict(), best_model_path)
                backup_path = os.path.join(backup_dir, f"{model_prefix}_best.pt")
                torch.save(policy_module.state_dict(), backup_path)
                print(f"  -> New best model! reward={ep_reward:.2f}", flush=True)

    total_training_time = perf_counter() - training_start
    total_steps = config.total_batch * config.frames_per_batch
    total_rollout = sum(log["rollout_time"])
    total_update = sum(log["update_time"])
    avg_rollout = total_rollout / len(log["rollout_time"]) if log["rollout_time"] else 0.0
    avg_update = total_update / len(log["update_time"]) if log["update_time"] else 0.0
    avg_batch = avg_rollout + avg_update
    overall_sps = total_steps / total_rollout if total_rollout > 0 else 0.0
    rollout_pct = (avg_rollout / avg_batch * 100) if avg_batch > 0 else 0
    update_pct = (avg_update / avg_batch * 100) if avg_batch > 0 else 0

    print(f"\n[Training Summary]", flush=True)
    print(f"  总训练时间: {total_training_time:.1f}s ({total_training_time / 60:.1f}m)", flush=True)
    print(f"  总 env steps: {total_steps}", flush=True)
    print(
        f"  平均 batch: {avg_batch:.2f}s "
        f"(rollout={avg_rollout:.2f}s [{rollout_pct:.0f}%] "
        f"| update={avg_update:.2f}s [{update_pct:.0f}%])",
        flush=True,
    )
    print(f"  平均 steps/sec: {overall_sps:.0f}", flush=True)
    print(f"  Best reward: {best_reward:.2f}", flush=True)

    step_profile = env.net.get_step_profile_summary()
    if int(step_profile.get("count", 0)) > 0:
        print(
            f"[Step Time Profile] steps={int(step_profile['count'])} "
            f"| total={float(step_profile['total_ms']):.2f}ms "
            f"| avg_step={float(step_profile['avg_ms']):.4f}ms "
            f"| steps_per_sec={float(step_profile.get('steps_per_sec', 0.0)):.2f}"
        )
        ordered_segments = [
            ("get_enable_t", "get_enable_t"),
            ("fire", "_fire"),
            ("build_obs", "build_obs"),
            ("advance_and_reward", "advance+reward"),
            ("next_event_delta", "next_event"),
            ("other", "other"),
        ]
        for key, label in ordered_segments:
            seg = step_profile["segments"][key]
            print(
                f"  {label:<12} total={float(seg['total_ms']):.2f}ms "
                f"| avg={float(seg['avg_ms']):.4f}ms "
                f"| ratio={float(seg['ratio_pct']):.2f}%"
            )
    final_path = os.path.join(backup_dir, f"{model_prefix}_final.pt")
    torch.save(policy_module.state_dict(), final_path)
    print(f"Final model: {final_path}", flush=True)
    print(f"Best model: {best_model_path}", flush=True)
    return log, policy_backbone


def build_single_env() -> Env_PN_Single:
    return Env_PN_Single(detailed_reward=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单设备单动作 PPO 训练（两阶段 release 回填）")
    parser.add_argument("--device", type=str, default="single", choices=["single", "cascade"], help="设备模式（single/cascade）")
    parser.add_argument("--compute-device", type=str, default=None, help="计算设备：cpu / cuda / cuda:0；未指定时：有 CUDA 则用 cuda，否则用配置文件中的 device")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--proc-time-rand-enabled", action="store_true", help="开启单设备工序时间随机扰动")
    parser.add_argument("--blame", action="store_true", help="开启二次追责（episode 结束后 blame_release_violations 回填惩罚）")
    args = parser.parse_args()

    root = Path(__file__).parents[2]
    path = root / "data" / "ppo_configs" / "s_train.json"
    cfg = PPOTrainingConfig.load(path)
    if args.compute_device is not None:
        cfg.device = args.compute_device.strip()
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    print(f"从 {path} 加载配置", flush=True)
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
