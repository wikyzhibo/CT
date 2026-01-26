import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector

from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.modules import GRUModule
from torchrl.modules.utils import get_primers_from_module

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# ===== 你原本的超参（可按需保留/调整）=====
n_hidden = 256
total_batch = 100
num_epochs = 20

# 这里改成“按轨迹 minibatch”
traj_minibatch_size = 8          # 每次更新用多少条 episode（单环境建议 4~16）
max_ep_len = 60                  # 你说平均 60 步；如果环境有 max_step，优先用 env 的那个
episodes_per_batch = 20          # 每个 batch 采集多少条 episode（越大越稳，但更慢）

gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
lr = 1e-4
entropy_start = 0.02
entropy_end = 0.01

# 如果你还要 BC 预训练/正则，保留你的接口
from behavior_clone import pretrain_bc, build_expert_buffer


def rnn_ppo_train(
    env,
    eval_env,
    device="cpu",
    seed=42,
    with_pretrain=False,
):
    torch.manual_seed(seed)

    # --------- 环境 / 维度 ---------
    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]

    # --------- (1) 定义 RNN Actor / Critic 模块（用不同的 recurrent_state key，避免覆盖） ---------
    actor_gru = GRUModule(
        input_size=n_m,
        hidden_size=n_hidden,
        in_keys=["observation_f", "recurrent_state_pi"],
        out_keys=["rnn_out_pi", ("next", "recurrent_state_pi")],
    )
    critic_gru = GRUModule(
        input_size=n_m,
        hidden_size=n_hidden,
        in_keys=["observation_f", "recurrent_state_v"],
        out_keys=["rnn_out_v", ("next", "recurrent_state_v")],
    )

    actor_head = TensorDictModule(
        nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        ),
        in_keys=["rnn_out_pi"],
        out_keys=["logits"],
    )
    critic_head = TensorDictModule(
        nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 1),
        ),
        in_keys=["rnn_out_v"],
        out_keys=["state_value"],
    )

    # 组合成 tensordict modules
    actor_td_module = TensorDictSequential(actor_gru, actor_head)
    critic_td_module = TensorDictSequential(critic_gru, critic_head)

    # --------- (2) 给 env 自动补齐 RNN 隐状态（primer）---------
    # get_primers_from_module 会扫描子模块（GRU/LSTM 等）并生成 TensorDictPrimer transform
    primers = get_primers_from_module(nn.ModuleList([actor_gru, critic_gru]))

    # 用 TransformedEnv 包一层，然后 append primer
    env = TransformedEnv(env)
    env.append_transform(primers)

    eval_env = TransformedEnv(eval_env)
    eval_env.append_transform(primers)

    # --------- (3) ProbabilisticActor + MaskedCategorical（你原来的 mask 保持不变）---------
    policy = ProbabilisticActor(
        module=actor_td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)

    value_module = critic_td_module.to(device)

    # --------- (4) 可选：BC 预训练（注意：专家 buffer 里需要补齐 recurrent_state_pi）---------
    if with_pretrain:
        obs, actions, mask = env.base_env.net.collect_expert_data() if hasattr(env, "base_env") else env.net.collect_expert_data()
        rb = build_expert_buffer(obs, mask, actions)

        # 让 BC 时也有 recurrent_state_pi
        # 假设专家数据是单步 [B, ...]：recurrent_state shape 常见为 [B, 1, hidden]
        def _add_rnn_state_to_exp(td):
            B = td.batch_size[0]
            td.set("recurrent_state_pi", torch.zeros(B, 1, n_hidden, device=td.device))
            return td

        # 你的 pretrain_bc 里如果会直接 sample tensordict，可以在里面或这里加；这里示意最小改法：
        policy = pretrain_bc(policy, actor_td_module, rb, device=device, preprocess_fn=_add_rnn_state_to_exp)

        # 重置环境内部网络（你原代码有 env.net.reset()）
        if hasattr(env, "base_env") and hasattr(env.base_env, "net"):
            env.base_env.net.reset()
        elif hasattr(env, "net"):
            env.net.reset()
    else:
        rb = None

    # --------- (5) 优化器 / GAE / PPO loss ---------
    optim = Adam(list(policy.parameters()) + list(value_module.parameters()), lr=lr)

    gae = GAE(gamma=gamma, lmbda=gae_lambda, value_network=value_module)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_start,
        critic_coeff=0.5,
        normalize_advantage=True,
    )

    # --------- (6) Collector：按 episode 切轨迹 ---------
    frames_per_batch = episodes_per_batch * max_ep_len

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * total_batch,
        device=device,
        split_trajs=True,   # 关键：按轨迹切分，方便 RNN 训练
    )

    frame_count = 0
    log = defaultdict(list)

    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx, tensordict_data in enumerate(collector):
            # --- 熵系数线性退火 ---
            frac = min(1.0, batch_idx / total_batch)
            ent = entropy_start + (entropy_end - entropy_start) * frac
            loss_module.entropy_coeff.copy_(
                torch.tensor(ent, device=loss_module.entropy_coeff.device, dtype=loss_module.entropy_coeff.dtype)
            )

            # tensordict_data 现在是 [N_traj, T]（T 会 padding；会带一个有效位 mask）:contentReference[oaicite:3]{index=3}
            tensordict_data = tensordict_data.to(device)

            # --- GAE（需要让 GRU 以多步模式运行）---
            with actor_gru.set_recurrent_mode(), critic_gru.set_recurrent_mode():
                gae_out = gae(tensordict_data)
            tensordict_data.set("advantage", gae_out.get("advantage"))
            tensordict_data.set("value_target", gae_out.get("value_target"))

            # --- PPO 多 epoch：按“轨迹”做 minibatch（不打碎序列）---
            n_traj = tensordict_data.batch_size[0]
            for _ in range(num_epochs):
                perm = torch.randperm(n_traj, device=device)
                for start in range(0, n_traj, traj_minibatch_size):
                    idx = perm[start:start + traj_minibatch_size]
                    subdata = tensordict_data[idx]

                    with actor_gru.set_recurrent_mode(), critic_gru.set_recurrent_mode():
                        loss_vals = loss_module(subdata)
                        ppo_loss = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                        )

                        # ---- BC 正则（可选，保持你原逻辑；这里只给最小兼容示意）----
                        if rb is not None:
                            # 你原来的 lambda 逻辑
                            if batch_idx < 100:
                                lambda_bc = 2.0
                            else:
                                lambda_bc = 0.1

                            if lambda_bc > 0:
                                exp_batch = rb.sample(traj_minibatch_size).to(device)
                                # 给专家 batch 补 recurrent_state_pi
                                B = exp_batch.batch_size[0]
                                exp_batch.set("recurrent_state_pi", torch.zeros(B, 1, n_hidden, device=device))

                                actor_td_module(exp_batch)
                                logits = exp_batch["logits"]
                                mask = exp_batch["action_mask"].bool()
                                dist = MaskedCategorical(logits=logits, mask=mask)
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

            # --------- 统计 / 打印 ---------
            # reward: [N_traj, T, 1] 或 [N_traj, T]，这里直接 sum 全部有效步（padding 的 mask 由 split_trajs 产生）
            reward = tensordict_data.get(("next", "reward"))
            valid_mask = tensordict_data.get("mask")  # [N_traj, T] bool
            if reward.dim() == valid_mask.dim() + 1:
                valid_mask_r = valid_mask.unsqueeze(-1)
            else:
                valid_mask_r = valid_mask

            ep_ret = (reward * valid_mask_r).sum().item()
            frame_count += int(valid_mask.sum().item())

            # 你原先 makespan 的逻辑我保留一个更稳的版本：用 finish 过滤
            if ("next", "finish") in tensordict_data.keys(True) and ("next", "time") in tensordict_data.keys(True):
                finish = tensordict_data.get(("next", "finish"))
                tval = tensordict_data.get(("next", "time"))
                # 只统计有效步
                finish = finish & valid_mask_r
                makespan = tval[finish]
                mean_makespan = makespan.float().mean().item() if makespan.numel() else float("nan")
                finish_times = int(finish.sum().item())
            else:
                mean_makespan = float("nan")
                finish_times = 0

            if batch_idx % 1 == 0:
                print(
                    f"batch {batch_idx+1:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}"
                    f"| circle={finish_times}| makespan={mean_makespan:.2f}"
                )

            log["reward"].append(ep_ret)

    print("Training done.")

    # --------- 保存模型 ---------
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(saved_models_dir, f"CT_RNN_{timestamp}.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    latest_model_path = os.path.join(saved_models_dir, "CT_RNN_latest.pt")
    torch.save(policy.state_dict(), latest_model_path)
    print(f"Latest model saved to: {latest_model_path}")

    return log, policy
