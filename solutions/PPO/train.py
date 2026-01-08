import os
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
from behavior_clone import pretrain_bc,build_expert_buffer

warnings.filterwarnings(
    "ignore",
    message=".*size_average.*reduce.*",
    category=UserWarning,
)

n_hidden = 256
total_batch=200

sub_batch_size = 400
num_epochs= 5
frames_per_batch= sub_batch_size * (num_epochs+1)

gamma=0.99
gae_lambda=0.95
clip_epsilon=0.2
lr=1e-4
entropy_start = 0.015
entropy_end   = 0.001


def save_dedup_csv(new_data, filename, append=True):
    """将新数据写入 CSV 文件；append=True 时会读取旧文件并去重"""
    if new_data.size == 0:
        return 0
    if append:
        try:
            old = pd.read_csv(filename, header=None).to_numpy()
            merged = np.concatenate([old, new_data], axis=0)
        except FileNotFoundError:
            merged = new_data
    else:
        merged = new_data
    # 去重
    merged = np.unique(merged, axis=0)
    pd.DataFrame(merged).to_csv(filename, index=False, header=False)
    return len(merged)



# --------- 训练主程序（PPO + 行为克隆 CE，按 λ 衰减） ----------
def train(
    env,
    eval_env,
    device="cpu",
    seed=42,
    with_pretrain=False,
):
    torch.manual_seed(seed)

    # 策略与价值网络
    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]


    policy_backbone = MaskedPolicyHead(hidden=n_hidden,n_obs=n_m, n_actions=n_actions)
    td_module = TensorDictModule(
        policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)

    if with_pretrain:
        obs, actions, mask = env.net.collect_expert_data()
        rb = build_expert_buffer(obs, mask, actions)
        policy = pretrain_bc(policy, td_module, rb, device=device)
        env.net.reset()
    else:
        rb = None

    value_module = ValueOperator(
        module=nn.Sequential(
            nn.Linear(n_m,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,1),
        ),
        in_keys=["observation_f"],
    ).to(device)

    optim = Adam(list(policy.parameters()) + list(value_module.parameters()), lr=lr)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * total_batch,
        device=device,
    )

    # relaybuffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # GAE + PPO
    gae = GAE(gamma=gamma, lmbda=gae_lambda, value_network=value_module)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_start,
        critic_coeff=0.5,
        normalize_advantage=True,
    )

    frame_count = 0
    log = defaultdict(list)

    # ====== (B) PPO阶段：可选 BC 正则项衰减 ======
    lambda_bc0 = 1.0  # 初始 BC 权重（可调 0.1~5）
    bc_decay_batches = 200  # 多少个 batch 衰减到 0（可调）



    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx, tensordict_data in enumerate(collector):

            frac = min(1.0, batch_idx / total_batch)
            val = entropy_start + (entropy_end - entropy_start) * frac

            loss_module.entropy_coeff.copy_(
                torch.tensor(val, device=loss_module.entropy_coeff.device, dtype=loss_module.entropy_coeff.dtype)
            )

            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))

            replay_buffer.extend(tensordict_data)

            n_deadlock = 0
            for _ in range(num_epochs):
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size).to(device)

                    loss_vals = loss_module(subdata)
                    ppo_loss = (loss_vals["loss_objective"]
                                + loss_vals["loss_critic"]
                                + loss_vals["loss_entropy"])

                    # ---- 加 BC 正则（如果提供专家数据）----
                    if rb is not None:
                        # 线性衰减：batch_idx=0 -> lambda_bc0, 到 bc_decay_batches 后变 0
                        frac = max(0.0, 3.0 - batch_idx / float(bc_decay_batches))
                        #lambda_bc = lambda_bc0 * frac
                        if batch_idx < 100:
                            lambda_bc = 2.0
                        else:
                            lambda_bc = 0.1

                        if lambda_bc > 0:
                            exp_batch = rb.sample(sub_batch_size).to(device)
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
            #if batch_idx > 150 and batch_idx % 10 == 0:
            #    ev(eval_env,batch_idx,policy)


            frame_count += int(tensordict_data.numel())
            ep_ret = tensordict_data["next","reward"].sum().item()
            overtime = tensordict_data["next","overtime"].sum().item()
            makespan = tensordict_data["next", "time"][
                tensordict_data["next", "finish"]
            ]
            mean_makespan = makespan.float().mean()
            finish_times = len(makespan)

            #n_deadlock += tensordict_data["next","deadlock_type"].sum().item()

            if batch_idx % 1 ==0:
                print(f"batch {batch_idx+1:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}| circle={finish_times}|makespan={mean_makespan:.2f}")

                n_deadlock=0
            #if batch_idx % 3 == 0 and batch_idx >0:
            #    makespan = ev(eval_env,policy=policy)
            #    log['makespan'].append(makespan)
            log['reward'].append(ep_ret)
            #log['makespan'].append(makespan)


    print("Training done.")
    # 确保 saved_models 文件夹存在
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # 保存模型，使用带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(saved_models_dir, f"CT_{timestamp}.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # 同时保存一个最新的模型副本
    latest_model_path = os.path.join(saved_models_dir, "CT_latest.pt")
    torch.save(policy.state_dict(), latest_model_path)
    print(f"Latest model saved to: {latest_model_path}")
    
    return log,policy



