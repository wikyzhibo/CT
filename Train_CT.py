import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


from CT_env import CT


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
    n_m = 20
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
        entropy_coeff=0.02,
        critic_coeff=0.5,
        normalize_advantage=True,
    )

    # λ 衰减（从 1.0 到 0.1）
    def lambda_bc_schedule(progress):  # progress ∈ [0,1]
        return 1.0 - 0.9 * progress

    frame_count = 0
    log = defaultdict(list)
    with set_exploration_type(ExplorationType.RANDOM):
        for batch_idx, tensordict_data in enumerate(collector):
            data_view = tensordict_data.reshape(-1).to("cpu")

            gae_vals = gae(tensordict_data)
            tensordict_data.set("advantage", gae_vals.get("advantage"))
            tensordict_data.set("value_target", gae_vals.get("value_target"))


            for _ in range(num_epochs):
                replay_buffer.extend(data_view)

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    # PPO 损失
                    loss_vals = loss_module(subdata)

                    # λ 衰减（依据进度）
                    progress = min(1.0, 1 / total_batch)
                    lambda_bc = lambda_bc_schedule(progress)

                    #loss = loss_ppo["loss_objective"] + loss_ppo["loss_critic"] + loss_ppo["loss_entropy"] \
                    #       + lambda_bc * bc_loss
                    loss_value =  loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

                    optim.zero_grad()
                    loss_value.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters())+list(value_module.parameters()), max_norm=1.0)
                    optim.step()

            # 统计

            frame_count += int(tensordict_data.numel())
            ep_ret = tensordict_data["next","reward"].sum().item()
            print(f"batch {batch_idx:04d} | frames={frame_count} | sum_reward={ep_ret:.2f}")
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

    #check_env_specs(env)
    mode = 0
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if mode == 0:
        log = train(
            env,
            frames_per_batch=258,
            total_batch=900,
            live_ratio=0.8,
            gamma=0.99,
            sub_batch_size=64,
            num_epochs=10,
            device="cpu",
        )
        plt.plot(log["reward"])
        plt.show()
        ev(env)

    else:
        ev(env)

