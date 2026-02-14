"""
双机械手并发动作 PPO 训练脚本

支持 TM2/TM3 双动作空间的 PPO 训练。
使用 Env_PN_Concurrent 环境和 DualHeadPolicyNet 网络。
"""
import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MaskedCategorical

from solutions.PPO.enviroment import Env_PN_Concurrent
from solutions.PPO.network.models import DualHeadPolicyNet
from data.ppo_configs.training_config import PPOTrainingConfig


class DualActionPolicyModule(nn.Module):
    """
    包装 DualHeadPolicyNet，输出双动作的 TensorDict 格式。
    """
    def __init__(self, backbone: DualHeadPolicyNet):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, observation_f, action_mask_tm2, action_mask_tm3):
        """
        Args:
            observation_f: 观测 (batch, obs_dim)
            action_mask_tm2: TM2 动作掩码 (batch, n_actions_tm2)
            action_mask_tm3: TM3 动作掩码 (batch, n_actions_tm3)
        
        Returns:
            action_tm2, action_tm3, log_prob_tm2, log_prob_tm3
        """
        out = self.backbone(observation_f)
        logits_tm2 = out["logits_tm2"]
        logits_tm3 = out["logits_tm3"]
        
        # 创建 masked 分布
        dist_tm2 = MaskedCategorical(logits=logits_tm2, mask=action_mask_tm2.bool())
        dist_tm3 = MaskedCategorical(logits=logits_tm3, mask=action_mask_tm3.bool())
        
        # 采样动作
        action_tm2 = dist_tm2.sample()
        action_tm3 = dist_tm3.sample()
        
        # 计算 log_prob
        log_prob_tm2 = dist_tm2.log_prob(action_tm2)
        log_prob_tm3 = dist_tm3.log_prob(action_tm3)
        
        return action_tm2, action_tm3, log_prob_tm2, log_prob_tm3, logits_tm2, logits_tm3


def collect_rollout(env: Env_PN_Concurrent, policy_module: DualActionPolicyModule, 
                    n_steps: int, device: str = "cpu"):
    """
    收集一轮 rollout 数据。
    
    Returns:
        TensorDict 包含轨迹数据
    """
    data = {
        "observation": [],
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
        "next_observation": [],
        "next_observation_f": [],
    }
    
    td = env.reset()
    
    for _ in range(n_steps):
        obs = td["observation"].unsqueeze(0).to(device)
        obs_f = obs.float()
        mask_tm2 = td["action_mask_tm2"].unsqueeze(0).to(device)
        mask_tm3 = td["action_mask_tm3"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            a1, a2, lp1, lp2, _, _ = policy_module(obs_f, mask_tm2, mask_tm3)
        
        # 存储当前状态
        data["observation"].append(td["observation"])
        data["observation_f"].append(td["observation"].float())
        data["action_mask_tm2"].append(td["action_mask_tm2"])
        data["action_mask_tm3"].append(td["action_mask_tm3"])
        data["action_tm2"].append(a1.squeeze(0))
        data["action_tm3"].append(a2.squeeze(0))
        data["log_prob_tm2"].append(lp1.squeeze(0))
        data["log_prob_tm3"].append(lp2.squeeze(0))
        
        # 构造 step 输入
        step_td = td.clone()
        step_td["action_tm2"] = a1.squeeze(0).cpu()
        step_td["action_tm3"] = a2.squeeze(0).cpu()
        
        # 执行动作
        td_next = env.step(step_td)
        
        # reward 在 next 键下或直接在顶层
        if "next" in td_next.keys() and "reward" in td_next["next"].keys():
            reward = td_next["next", "reward"]
            next_obs = td_next["next", "observation"]
            terminated = td_next["next", "terminated"] if "terminated" in td_next["next"].keys() else td_next["terminated"]
            finish = td_next["next", "finish"] if "finish" in td_next["next"].keys() else td_next.get("finish", torch.tensor(False))
            scrap = td_next["next", "scrap"] if "scrap" in td_next["next"].keys() else td_next.get("scrap", torch.tensor(False))
            time_val = td_next["next", "time"] if "time" in td_next["next"].keys() else td_next.get("time", torch.tensor([0]))
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
        data["next_observation"].append(next_obs)
        data["next_observation_f"].append(next_obs.float())
        
        if terminated.item():
            td = env.reset()
        else:
            # 从 next 中提取下一状态
            if "next" in td_next.keys():
                td = td_next["next"].clone()
            else:
                td = td_next.clone()
    
    # 转换为 TensorDict
    return TensorDict({
        k: torch.stack(v) for k, v in data.items()
    }, batch_size=[n_steps])


def train_concurrent(
    config: PPOTrainingConfig = None,
    training_phase: int = 2,
    config_path: str = None,
    checkpoint_path: str = None,
):
    """
    双机械手并发动作 PPO 训练。
    
    Args:
        config: 训练配置（优先使用）
        training_phase: 训练阶段
        config_path: 配置文件路径（如果config为None则从此加载）
        checkpoint_path: checkpoint文件路径，用于继续训练或微调
    
    Returns:
        log: 训练日志
        policy_module: 训练好的策略
    """
    # 加载配置
    if config is None:
        if config_path is not None and os.path.exists(config_path):
            config = PPOTrainingConfig.load(config_path)
            print(f"从 {config_path} 加载配置")
        else:
            config = PPOTrainingConfig()
            print("使用默认配置")
    
    print(f"[Concurrent PPO Training - Phase {training_phase}]")
    print(config)
    
    torch.manual_seed(config.seed)
    device = config.device
    
    # 创建环境
    env = Env_PN_Concurrent(training_phase=training_phase)
    
    # 网络参数
    n_obs = env.observation_spec["observation"].shape[0]
    n_actions_tm2 = env.n_actions_tm2
    n_actions_tm3 = env.n_actions_tm3
    
    print(f"  观测维度: {n_obs}")
    print(f"  TM2 动作空间: {n_actions_tm2}")
    print(f"  TM3 动作空间: {n_actions_tm3}")
    
    # 创建策略网络
    backbone = DualHeadPolicyNet(
        n_obs=n_obs,
        n_hidden=config.n_hidden,
        n_actions_tm2=n_actions_tm2,
        n_actions_tm3=n_actions_tm3,
        n_layers=config.n_layer,
    ).to(device)
    
    policy_module = DualActionPolicyModule(backbone).to(device)
    
    # 加载 checkpoint（如果提供）
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"从 checkpoint 加载: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        policy_module.load_state_dict(state_dict)
        print("Checkpoint 加载成功，继续训练...")
    
    # 价值网络
    value_net = nn.Sequential(
        nn.Linear(n_obs, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
        nn.Linear(config.n_hidden, 1),
    ).to(device)
    
    # 优化器
    optim = Adam(
        list(policy_module.parameters()) + list(value_net.parameters()),
        lr=config.lr
    )
    
    # 保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    backup_dir = os.path.join(saved_models_dir, f"concurrent_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    
    model_prefix = f"CT_concurrent_phase{training_phase}"
    best_model_path = os.path.join(saved_models_dir, f"{model_prefix}_best.pt")
    best_reward = float('-inf')
    
    log = defaultdict(list)
    
    # 训练循环
    for batch_idx in range(config.total_batch):
        # 收集 rollout
        rollout = collect_rollout(env, policy_module, config.frames_per_batch, device)
        rollout = rollout.to(device)
        
        # 计算 GAE
        with torch.no_grad():
            values = value_net(rollout["observation_f"]).squeeze(-1)
            next_values = value_net(rollout["next_observation_f"]).squeeze(-1)
            
            rewards = rollout["reward"].squeeze(-1)
            dones = rollout["done"].float().squeeze(-1)
            
            # TD error
            delta = rewards + config.gamma * next_values * (1 - dones) - values
            
            # GAE
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = delta[t] + config.gamma * config.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            
            returns = advantages + values
        
        rollout["advantage"] = advantages
        rollout["value_target"] = returns
        
        # PPO 更新
        for _ in range(config.num_epochs):
            # Mini-batch 更新
            indices = torch.randperm(len(rollout))
            for start in range(0, len(rollout), config.sub_batch_size):
                end = start + config.sub_batch_size
                batch_indices = indices[start:end]
                batch = rollout[batch_indices]
                
                obs_f = batch["observation_f"]
                mask_tm2 = batch["action_mask_tm2"]
                mask_tm3 = batch["action_mask_tm3"]
                old_a1 = batch["action_tm2"]
                old_a2 = batch["action_tm3"]
                old_lp1 = batch["log_prob_tm2"]
                old_lp2 = batch["log_prob_tm3"]
                adv = batch["advantage"]
                ret = batch["value_target"]
                
                # 前向传播
                out = policy_module.backbone(obs_f)
                dist_tm2 = MaskedCategorical(logits=out["logits_tm2"], mask=mask_tm2.bool())
                dist_tm3 = MaskedCategorical(logits=out["logits_tm3"], mask=mask_tm3.bool())
                
                new_lp1 = dist_tm2.log_prob(old_a1)
                new_lp2 = dist_tm3.log_prob(old_a2)
                
                # 联合 log_prob
                old_log_prob = old_lp1 + old_lp2
                new_log_prob = new_lp1 + new_lp2
                
                # PPO ratio
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 标准化 advantage
                adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                # Clipped loss
                surr1 = ratio * adv_normalized
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * adv_normalized
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值 loss
                values_pred = value_net(obs_f).squeeze(-1)
                value_loss = 0.5 * (values_pred - ret).pow(2).mean()
                
                # 熵奖励
                entropy = dist_tm2.entropy().mean() + dist_tm3.entropy().mean()
                entropy_loss = -config.entropy_start * entropy
                
                loss = policy_loss + value_loss + entropy_loss
                
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy_module.parameters()) + list(value_net.parameters()),
                    max_norm=1.0
                )
                optim.step()
        
        # 统计
        ep_reward = rollout["reward"].sum().item()
        finish_count = rollout["finish"].sum().item()  # 成功完成
        scrap_count = rollout["scrap"].sum().item()    # 报废截断
        
        # 计算成功完成的平均 makespan
        finish_mask = rollout["finish"].squeeze(-1).bool()
        if finish_mask.any():
            finish_times = rollout["time"][finish_mask].squeeze(-1).float()
            avg_makespan = finish_times.mean().item()
        else:
            avg_makespan = 0.0
        
        log["reward"].append(ep_reward)
        log["finish"].append(finish_count)
        log["scrap"].append(scrap_count)
        log["makespan"].append(avg_makespan)
        
        if batch_idx % 1 == 0:
            print(f"batch {batch_idx+1:04d} | reward={ep_reward:.2f} | finish={int(finish_count)} | scrap={int(scrap_count)} | makespan={avg_makespan:.1f}")
        
        # 保存最佳模型
        if ep_reward > best_reward and finish_count > 0:
            best_reward = ep_reward
            torch.save(policy_module.state_dict(), best_model_path)
            backup_path = os.path.join(backup_dir, f"{model_prefix}_best.pt")
            torch.save(policy_module.state_dict(), backup_path)
            print(f"  -> New best model! reward={ep_reward:.2f}")
    
    print(f"\nTraining done. Best reward: {best_reward:.2f}")
    
    # 保存最终模型
    final_path = os.path.join(backup_dir, f"{model_prefix}_final.pt")
    torch.save(policy_module.state_dict(), final_path)
    print(f"Final model: {final_path}")
    print(f"Best model: {best_model_path}")
    
    return log, policy_module


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="双机械手并发动作 PPO 训练")
    parser.add_argument("--config", type=str, 
                       default="data/ppo_configs/concurrent.json",
                       help="训练配置文件路径")
    parser.add_argument("--phase", type=int, default=2, 
                       help="训练阶段 (1 or 2)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="checkpoint模型路径，用于继续训练或微调")
    args = parser.parse_args()
    
    # 从配置文件加载
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
        config = None
    else:
        config = PPOTrainingConfig.load(config_path)
    
    train_concurrent(
        config=config, 
        training_phase=args.phase, 
        config_path=config_path,
        checkpoint_path=args.checkpoint
    )
