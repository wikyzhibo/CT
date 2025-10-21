import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.core.pylabtools import figsize
from torch.optim import Adam
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (Compose, StepCounter, DTypeCastTransform,
                          TransformedEnv,ActionMask,RewardSum)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MaskedCategorical
from collections import defaultdict
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*size_average.*reduce.*",
    category=UserWarning,
)
from sklearn.ensemble import RandomForestClassifier
import joblib

from CT_env import CT
from markbuffer import collect_markings_env

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

# --------------- 策略头 ---------------
class MaskedPolicyHead(nn.Module):
    def __init__(self, obs_dim=None, hidden=128, n_actions=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, obs):
        logits = self.net(obs.to(torch.float32))
        return logits

# --------- 训练主程序（PPO + 行为克隆 CE，按 λ 衰减） ----------
def train(
    env,
    total_batch=30,
    frames_per_batch=1024,
    live_ratio = 0.5,
    sub_batch_size = 64,
    num_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    lr=3e-4,
    device="cpu",
    seed=42,
):
    torch.manual_seed(seed)

    # 策略与价值网络
    n_actions = 16
    n_m = 21
    n_hidden = 256
    policy_backbone = MaskedPolicyHead(obs_dim=n_m, hidden=n_hidden, n_actions=n_actions)
    td_module = TensorDictModule(
        policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys = ["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)

    value_module = ValueOperator(
        module=nn.Sequential(
            nn.Linear(n_m, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 1),
        ),
        in_keys=["observation_f"],
    ).to(device)

    optim = Adam(list(policy.parameters()) + list(value_module.parameters()), lr=lr)

    # 收集器
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * total_batch,
        device=device,
        split_trajs=False,
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
        entropy_coeff=0.04,
        critic_coeff=0.5,
        normalize_advantage=True,
    )

    frame_count = 0
    log = defaultdict(list)

    live_repo = []  # List of [L, F]
    dead_repo = []  # List of [F]
    live_count = 0
    dead_count = 0

    LIVE_THRESHOLD = 10000
    DEAD_THRESHOLD = 100
    APPEND_MODE = True
    n_deadlock = 0
    with ((set_exploration_type(ExplorationType.RANDOM))):
        for batch_idx, tensordict_data in enumerate(collector):


            data_view = tensordict_data.reshape(-1).to("cpu")

            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))

            '''
            live_X, dead_X, _ = collect_markings_env(tensordict_data, keep_deadlock_mid=False)
            if live_X:
                live_repo.extend(live_X)
            if dead_X.size:
                dead_repo.append(dead_X)

            # 检查数量阈值
            live_count = sum(len(x) for x in live_repo)
            dead_count = sum(x.shape[0] for x in dead_repo)

            if live_count >= LIVE_THRESHOLD or dead_count >= DEAD_THRESHOLD:
                print(f"[SAVE TRIGGER] batch={batch_idx + 1}, live={live_count}, dead={dead_count}")

                # 合并、去重、写入
                if live_repo:
                    live_all = np.concatenate(live_repo, axis=0)
                    live_all = np.unique(live_all, axis=0)
                    n_live = save_dedup_csv(live_all, "live_markings.csv", append=APPEND_MODE)
                    print(f"  → 写入 live_markings.csv （共 {n_live} 条去重后记录）")
                    live_repo.clear()

                if dead_repo:
                    dead_all = np.concatenate(dead_repo, axis=0)
                    dead_all = np.unique(dead_all, axis=0)
                    n_dead = save_dedup_csv(dead_all, "dead_markings.csv", append=APPEND_MODE)
                    print(f"  → 写入 dead_markings.csv （共 {n_dead} 条去重后记录）")
                    dead_repo.clear()

                live_count = 0
                dead_count = 0
            '''

            for _ in range(num_epochs):
                replay_buffer.extend(data_view)

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    # PPO 损失
                    loss_vals = loss_module(subdata)
                    loss_value =  loss_vals["loss_objective"] + loss_vals["loss_critic"
                                    ] + loss_vals["loss_entropy"]

                    optim.zero_grad()
                    loss_value.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters())+list(value_module.parameters()), max_norm=1.0)
                    optim.step()

            # 统计

            frame_count += int(tensordict_data.numel())
            ep_ret = tensordict_data["next","reward"].sum().item()
            n_deadlock += tensordict_data["next","deadlock_type"].sum().item()
            if batch_idx % 1 ==0 and batch_idx != 0:
                print(f"batch {batch_idx:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}| deadlock={n_deadlock}")
                log['deadlock'].append(n_deadlock)
                n_deadlock = 0
            log['reward'].append(ep_ret)

    print("Training done.")
    torch.save(policy, "CT.pt")
    return log

def ev(env, policy=None):
    if policy is None:
        policy = torch.load("CT.pt",weights_only=False)
    out = env.rollout(300, policy)
    actions = out["action"].tolist()
    time = out["time"].squeeze().tolist()
    time.append(out["next","time"][-1].item())
    out = time.pop(0)
    df = pd.DataFrame({"time":time, "actions":actions})
    df.to_csv("out.csv")

    print("makspan:",time[-1])




import matplotlib.pyplot as plt
if __name__ == "__main__":
    # 环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = CT(device=device)

    transform = Compose([ActionMask(),
                         DTypeCastTransform(dtype_in=torch.int64,dtype_out=torch.float32,
                                            in_keys="observation",out_keys="observation_f"),])
    env = TransformedEnv(base_env,transform)
    print("="*40,"\n observation_spec:",env.observation_spec)
    print("="*40,"\n action_spec:",env.action_spec)
    print("="*40,"\n done_spec:",env.done_spec)

    check_env_specs(env)

    mode = 1
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if mode == 0:
        start_time = time.time()
        log = train(
            env,
            frames_per_batch=256,
            total_batch=100,
            live_ratio=0.8,
            gamma=0.99,
            sub_batch_size=64,
            num_epochs=10,
            device="cpu",
        )
        print("training time:",time.time()-start_time)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(log["reward"][::10],)
        ax[0].set_ylim(-10000, 0)
        ax[1].plot(log["deadlock"])
        plt.show()

        ev(env)

    elif mode == 1:
        for _ in range(10):
            ev(env)

