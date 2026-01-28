import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#from IPython.core.pylabtools import figsize
from torch.optim import Adam
from tensordict.nn import TensorDictModule

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MaskedCategorical
from collections import defaultdict
from torchrl.collectors import SyncDataCollector
import warnings

#from CT_env import init_marks_from_m
from solutions.PPO.network.models import MaskedPolicyHead
from solutions.PPO.behavior_clone import pretrain_bc,build_expert_buffer
from data.ppo_configs.training_config import PPOTrainingConfig



# --------- 训练主程序（PPO + 行为克隆 CE，按 λ 衰减） ----------
def train(
    env,
    eval_env,
    config: PPOTrainingConfig = None,
    checkpoint_path: str = None,
    config_path: str = None,
):
    """
    PPO 训练主程序。
    
    Args:
        env: 训练环境
        eval_env: 评估环境
        config: PPOTrainingConfig 配置对象
        checkpoint_path: checkpoint 文件路径，用于加载预训练模型继续训练
        config_path: 配置文件路径，用于加载配置（如果config为None）
    
    Returns:
        log: 训练日志字典
        policy: 训练好的策略网络
    """
    # 加载或使用默认配置
    if config is None:
        if config_path is not None and os.path.exists(config_path):
            config = PPOTrainingConfig.load(config_path)
            print(f"从 {config_path} 加载配置")
        else:
            config = PPOTrainingConfig()
            print("使用默认配置")
    
    # 打印配置信息
    print(config)
    
    # 保存本次训练使用的配置
    saved_configs_dir = os.path.join("data", "ppo_configs", "training_runs")
    os.makedirs(saved_configs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(saved_configs_dir, f"config_phase{config.training_phase}_{timestamp}.json")
    config.save(config_save_path)
    
    torch.manual_seed(config.seed)
    
    # 创建保存目录
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # 创建带时间戳的备份文件夹
    backup_dir = os.path.join(saved_models_dir, timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    # 初始化最佳奖励追踪
    best_reward = float('-inf')
    best_model_path = os.path.join(saved_models_dir, f"CT_phase{config.training_phase}_best.pt")
    
    print(f"\n[Training Phase {config.training_phase}]")
    if config.training_phase == 1:
        print("  -> 仅考虑报废惩罚（加工腔室超时）")
    else:
        print("  -> 完整奖励（加工腔室超时 + 运输位超时）")
    print(f"  -> Backup folder: {backup_dir}")

    # 策略与价值网络
    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]

    policy_backbone = MaskedPolicyHead(
        hidden=config.n_hidden,
        n_obs=n_m, 
        n_actions=n_actions,
        n_layers=config.n_layer
    )
    td_module = TensorDictModule(
        policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(config.device)

    # 加载 checkpoint（如果提供）
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("Checkpoint loaded successfully.")

    if config.with_pretrain:
        obs, actions, mask = env.net.collect_expert_data()
        rb = build_expert_buffer(obs, mask, actions)
        policy = pretrain_bc(policy, td_module, rb, device=config.device)
        env.net.reset()
    else:
        rb = None

    value_module = ValueOperator(
        module=nn.Sequential(
            nn.Linear(n_m, config.n_hidden), nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_hidden), nn.ReLU(),
            nn.Linear(config.n_hidden, 1),
        ),
        in_keys=["observation_f"],
    ).to(config.device)

    optim = Adam(list(policy.parameters()) + list(value_module.parameters()), lr=config.lr)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.frames_per_batch * config.total_batch,
        device=config.device,
    )

    # relaybuffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # GAE + PPO
    gae = GAE(gamma=config.gamma, lmbda=config.gae_lambda, value_network=value_module)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_start,
        critic_coeff=0.5,
        normalize_advantage=True,
    )

    frame_count = 0
    log = defaultdict(list)

    with set_exploration_type(ExplorationType.MODE):
        for batch_idx, tensordict_data in enumerate(collector):

            frac = min(1.0, batch_idx / config.total_batch)
            val = config.entropy_start + (config.entropy_end - config.entropy_start) * frac

            loss_module.entropy_coeff.copy_(
                torch.tensor(val, device=loss_module.entropy_coeff.device, dtype=loss_module.entropy_coeff.dtype)
            )

            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))

            replay_buffer.extend(tensordict_data)

            n_deadlock = 0
            for _ in range(config.num_epochs):
                for _ in range(config.frames_per_batch // config.sub_batch_size):
                    subdata = replay_buffer.sample(config.sub_batch_size).to(config.device)

                    loss_vals = loss_module(subdata)
                    ppo_loss = (loss_vals["loss_objective"]
                                + loss_vals["loss_critic"]
                                + loss_vals["loss_entropy"])

                    # ---- 加 BC 正则（如果提供专家数据）----
                    if rb is not None:
                        # 使用配置中的BC权重
                        if batch_idx < config.bc_switch_batch:
                            lambda_bc = config.bc_weight_early
                        else:
                            lambda_bc = config.bc_weight_late

                        if lambda_bc > 0:
                            exp_batch = rb.sample(config.sub_batch_size).to(config.device)
                            td_module(exp_batch)
                            logits = exp_batch["logits"]
                            mask = exp_batch["action_mask"].bool()

                            # 2) 构造 masked 分布
                            dist = MaskedCategorical(logits=logits, mask=mask)

                            # 3) BC loss = -log pi(a_expert | s)
                            expert_action = exp_batch["action_exp"].long()
                            bc_loss = -(dist.log_prob(expert_action)).mean()
                            loss = ppo_loss + lambda_bc * bc_loss
                        else:
                            loss = ppo_loss
                    else:
                        loss = ppo_loss

                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_module.parameters()), max_norm=1.0)
                    optim.step()

            # 统计
            frame_count += int(tensordict_data.numel())
            ep_ret = tensordict_data["next","reward"].sum().item()
            makespan = tensordict_data["next", "time"][
                tensordict_data["next", "finish"]
            ]
            mean_makespan = makespan.float().mean()
            finish_times = len(makespan)

            if batch_idx % 1 ==0:
                print(f"batch {batch_idx+1:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}| circle={finish_times}|makespan={mean_makespan:.2f}")

                n_deadlock=0
            log['reward'].append(ep_ret)
            
            # 检查是否是最佳模型
            if ep_ret > best_reward and finish_times>0:
                best_reward = ep_ret
                torch.save(policy.state_dict(), best_model_path)
                # 同时备份到时间戳文件夹
                backup_best_path = os.path.join(backup_dir, f"CT_phase{config.training_phase}_best.pt")
                torch.save(policy.state_dict(), backup_best_path)
                print(f"  -> New best model saved! reward={ep_ret:.2f}")

    print(f"\nTraining done. Best reward: {best_reward:.2f}")
    
    # 保存最终模型到时间戳备份文件夹
    final_model_path = os.path.join(backup_dir, f"CT_phase{config.training_phase}_final.pt")
    torch.save(policy.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # 保存 latest 模型（覆盖）
    latest_model_path = os.path.join(saved_models_dir, f"CT_phase{config.training_phase}_latest.pt")
    torch.save(policy.state_dict(), latest_model_path)
    print(f"Latest model saved to: {latest_model_path}")
    
    print(f"Best model: {best_model_path}")
    print(f"Backup folder: {backup_dir}")
    
    return log, policy



