import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#from IPython.core.pylabtools import figsize
from torch.optim import Adam
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
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

#from CT_env import init_marks_from_m
from solutions.PPO.data_collector import DeadlockSafeCollector
from solutions.PPO.enviroment import CT

warnings.filterwarnings(
    "ignore",
    message=".*size_average.*reduce.*",
    category=UserWarning,
)

n_hidden = 256
total_batch=300
frames_per_batch=64 * 4
sub_batch_size = 64
num_epochs=5
gamma=0.99
gae_lambda=0.95
clip_epsilon=0.2
lr=1e-4
entropy_coeff = 0.01




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
    def __init__(self, hidden=128,n_obs=None, n_actions=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs,hidden), nn.ReLU(),
            nn.Linear(hidden,hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden,n_actions)
        )
    def forward(self, obs):
        logits = self.net(obs.to(torch.float32))
        return logits

# --------- 训练主程序（PPO + 行为克隆 CE，按 λ 衰减） ----------
def train(
    env,
    device="cpu",
    seed=42,
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

    # 收集器：自定义版本，记录死锁终止后继续采样
    collector = DeadlockSafeCollector(
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
        entropy_coeff=entropy_coeff,
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

    #base_env = CT(device=device)
    #transform = Compose([ActionMask(),
    #                     DTypeCastTransform(dtype_in=torch.int64, dtype_out=torch.float32,
    #                                        in_keys="observation", out_keys="observation_f"), ])
    #eval_env = TransformedEnv(base_env, transform)

    with ((set_exploration_type(ExplorationType.RANDOM))):
        for batch_idx, tensordict_data in enumerate(collector):
            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))

            data_view = tensordict_data.reshape(-1).to("cpu")
            #replay_buffer.empty()
            replay_buffer.extend(tensordict_data)

            for _ in range(num_epochs):
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
            #if batch_idx > 150 and batch_idx % 10 == 0:
            #    ev(eval_env,batch_idx,policy)


            frame_count += int(tensordict_data.numel())
            ep_ret = tensordict_data["next","reward"].sum().item()
            makespan = torch.where(
                tensordict_data["next", "finish"],
                tensordict_data["next", "time"],
                100000
            ).min().item()

            n_deadlock += tensordict_data["next","deadlock_type"].sum().item()
            if batch_idx % 1 ==0 and batch_idx != 0:
                print(f"batch {batch_idx:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}| deadlock={n_deadlock}| makespan={makespan}")
                log['deadlock'].append(n_deadlock)
                n_deadlock = 0
            log['reward'].append(ep_ret)
            log['makespan'].append(makespan)


    print("Training done.")
    torch.save(policy, "../../CT.pt")
    return log

def ev(env,epoch, policy=None):
    if policy is None:
        policy = torch.load("../../CT.pt", weights_only=False)
    policy.eval()
    with torch.no_grad():
        out = env.rollout(300, policy)
    if out["next","finish"].sum().item() <= 0:
        return
    actions = out["action"].tolist()
    time = out["time"].squeeze().tolist()
    time.append(out["next","time"][-1].item())
    time.pop(0)
    name = mapping(actions)
    df = pd.DataFrame({"actions":actions,"transition":name,"time":time, },index=None)
    fname = f"gantt/batch{epoch}.csv" if epoch is not None else "gantt/ev.csv"
    df.to_csv(fname)
    #makespan = torch.where(out["next","finish"], out["next","time"]).tolist()
    #print("makspan:",out["next","time"][-1])

def mapping(path):
    transition = ["u1","t1","u2","t2","u3","t3",
                  "u4","t4","u5","t5","u6","t6",
                  "u7","t7","u8","t8"]
    s = []
    for k in path:
        s.append(transition[k])
    return s

import matplotlib.pyplot as plt
if __name__ == "__main__":
    '''
    params_N2 = {'path': 'Net/N2.txt',
                 'n_wafer': 13,
                 'process_time': [20, 10, 2790, 10, 104, 20, 2790],
                 'capacity': {'pm': [2, 2, 4, 2, 1, 1, 1],
                              'bm': [2, 2],
                              'robot': [1, 1, 1], },
                 'controller': {'bm1': {'p': ['d1', 'p1', 'd6', 'p7'],
                                        't': ['u1', 't1', 'u6', 't6']},
                                'bm2': {'p': ['d2', 'p2', 'd4', 'p4'],
                                        't': ['u2', 't2', 'u4', 't4']},
                                'f': [('p3', 4, 'u3'), ('p5', 1, 'u5'),
                                      ('p7', 1, 'u8'), ('p5', 1, 'u9')]}}
    '''
    params_N6 = {'path': r'C:\Users\khand\OneDrive\code\dqn\CT\Net\N6.txt',
                 'n_wafer': 37,
                 'process_time': [8, 20, 70, 0, 300, 70, 20],
                 'capacity': {'pm': [1, 2, 2, 2, 2, 2, 2],
                              'bm': [2, 2],
                              'robot': [1, 2, 2]},
                 'controller': {'bm1': {'p': ['d2', 'p2', 'd7', 'p7'],
                                        't': ['u1', 't2', 'u6', 't7']},
                                'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                        't': ['u3', 't4', 'u5', 't6']},
                                'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                                      ('p5', 2, 'u4')]}
                 }

    # 环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = CT(device=device,**params_N6)

    transform = Compose([ActionMask(),
                         DTypeCastTransform(dtype_in=torch.int64,dtype_out=torch.float32,
                                            in_keys="observation",out_keys="observation_f"),])
    env = TransformedEnv(base_env,transform)

    #print("="*40,"\n observation_spec:",env.observation_spec)
    #print("="*40,"\n action_spec:",env.action_spec)
    #print("="*40,"\n done_spec:",env.done_spec)

    check_env_specs(env)

    mode = 0
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if mode == 0:
        start_time = time.time()
        log = train(
            env,
            device="cpu",
        )
        print("training time:",time.time()-start_time)
        fig, ax = plt.subplots(1,2,figsize=(12,8))
        ax[0].plot(log["makespan"],)
        ax[0].set_ylim(9000, 20000)
        ax[1].plot(log["deadlock"])
        plt.show()

        #ev(env)

    elif mode == 1:
        for _ in range(1):
            print("1")
            #ev(env)

