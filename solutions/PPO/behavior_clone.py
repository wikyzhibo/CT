from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch.optim import Adam
import torch.nn as nn
from torchrl.modules import MaskedCategorical, ProbabilisticActor
from tensordict.nn import TensorDictModule
import torch
from tensordict import TensorDict

#from solutions.PDR.net import Petri
from solutions.PPO.network.models import MaskedPolicyHead

import numpy as np
from collections import defaultdict

def obs_to_key(o):
    """
    将观测转换为可哈希的字典键（tuple）。

    说明:
    - 接受 numpy.ndarray 或 torch.Tensor。
    - 对观测做 flatten 并转换为 tuple，用于在 state->action 映射中作为键。
    - 注意：flatten 顺序会影响键的值（默认按行主序 C-order）。

    参数:
    - o: np.ndarray 或 torch.Tensor，形状例如 (n_obs,) 或 (H, W, ...)

    返回:
    - tuple: 可哈希的键，用于 dict 索引。
    """
    # 如果是 numpy
    if isinstance(o, np.ndarray):
        return tuple(o.flatten().tolist())
    # 如果是 torch tensor
    else:
        return tuple(o.flatten().cpu().numpy().tolist())

def build_expert_buffer(expert_obs_f, expert_mask, expert_act, expert_return=None):
    """
    将专家数据构造为 torchrl ReplayBuffer。

    说明:
    - 构造一个 TensorDict，包含键 "observation_f", "action_mask", "action_exp"。
    - 使用 LazyTensorStorage + RandomSampler 构造 ReplayBuffer，并将单条 TensorDict 扩展到 buffer 中。

    参数:
    - expert_obs_f: array-like, 形状 [N, n_obs]（N 为样本数）
    - expert_mask:  array-like, 形状 [N, n_actions]，布尔或可转换为 bool 的矩阵
    - expert_act:   array-like, 形状 [N] 或 [N,1]，表示专家动作的整数索引
    - expert_return: 可选，array-like, [N]，用于保存每条样本的回报（如果有）

    返回:
    - ReplayBuffer 已填充专家数据，可直接 sample(batch_size)
    """
    # expert_obs_f: np/torch [N, n_obs]
    # expert_mask:  np/torch [N, n_actions]
    # expert_act:   np/torch [N] or [N,1]
    td = TensorDict(
        {
            "observation_f": torch.as_tensor(expert_obs_f, dtype=torch.float32),
            "action_mask":   torch.as_tensor(expert_mask, dtype=torch.bool),
            "action_exp":        torch.as_tensor(expert_act, dtype=torch.long),
        },
        batch_size=[len(expert_act)]
    )
    if expert_return is not None:
        td.set("return", torch.as_tensor(expert_return, dtype=torch.float32))

    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=td.numel()),
        sampler=RandomSampler(),
    )
    rb.extend(td)
    return rb


def pretrain_bc(
    policy,           # ProbabilisticActor
    td_module,        # TensorDictModule: outputs "logits"
    expert_buffer,
    device,
    bc_steps=2000,
    batch_size=64,
    lr=1e-4,
):
    """
    使用行为克隆 (Behavior Cloning) 对特征提取/策略 head 进行预训练。

    说明/流程:
    1. 在每一步从 expert_buffer 采样一个 batch（期望包含 keys: "observation_f", "action_mask", "action_exp"）。
    2. 通过 td_module 将 observation_f 映射为 logits（写入 batch["logits"]）。
    3. 使用 MaskedCategorical(mask=action_mask) 构造分布，并计算专家动作的负对数似然作为 BC loss。
    4. 仅优化 td_module 的参数（作为 backbone / logits 生成网络），并做梯度裁剪。

    参数:
    - policy: ProbabilisticActor（在本函数中仅用于接口一致性）
    - td_module: TensorDictModule，输入 "observation_f"，输出 "logits"
    - expert_buffer: ReplayBuffer，包含专家数据
    - device: torch.device
    - bc_steps, batch_size, lr: 训练超参

    返回:
    - policy: 原始 policy（td_module 已被训练以改善 logits）
    """
    policy.train()
    td_module.train()
    # 优化 backbone(td_module) 的参数（只训练特征提取/输出 logits 的模块）
    optim = Adam(list(td_module.parameters()), lr=lr)

    for step in range(bc_steps):
        batch = expert_buffer.sample(batch_size).to(device)

        # 你这里用的 observation key 是 "observation_f"
        # 专家动作 key 假设是 "action_exp"
        expert_action = batch["action_exp"].long().view(-1)  # 保证形状为 [B]

        # 1) 只跑 backbone，拿 logits（不要用 policy 去采样 action）
        td_module(batch)                       # batch["logits"] 会被写入
        logits = batch["logits"]               # [B, n_actions]
        mask = batch["action_mask"].bool()     # [B, n_actions]

        # 2) 构造 masked 分布
        dist = MaskedCategorical(logits=logits, mask=mask)

        # 3) BC loss = -log pi(a_expert | s)
        # log_prob 对于离散 action 期望 expert_action 形状为 [B] 或 [B,1]
        loss = -(dist.log_prob(expert_action)).mean()

        optim.zero_grad()
        loss.backward()
        # 修正 grad clipping 的参数对象（之前对 policy.parameters()，应对 td_module.parameters()）
        nn.utils.clip_grad_norm_(td_module.parameters(), 1.0)
        optim.step()

        if step % 300 == 0:
            print(f"[BC] step={step}, loss={loss.item():.4f}")

    return policy


def main():
    """
    入口函数 —— 执行以下步骤（导航）:
    1) 初始化环境与参数
    2) collect_expert_data: 收集专家轨迹（obs, actions, mask）
    3) 预处理: 将观测压平并去重，构造 state->actions 与 state->mask 的映射
    4) 构造专家 ReplayBuffer（build_expert_buffer）
    5) 构建模型（backbone + ProbabilisticActor）
    6) 运行 pretrain_bc 做行为克隆预训练

    注意:
    - 本函数中的预处理实现为示例，可能需要根据实际数据格式调整（比如 mask 的合并策略）。
    """
    params_N6 = {'path': r'C:\Users\khand\OneDrive\code\dqn\CT\Net\N6.txt',
                 'n_wafer': 8,
                 'process_time': [8, 20, 70, 0, 300, 70, 20],
                 'capacity': {'pm': [1, 2, 2, 2, 2, 2, 2],
                              'bm': [2, 2],
                              'robot': [1, 2, 2]},
                 'capacity_xianzhi': {'s1': 'u1'},
                 'controller': {'bm1': {'p': ['d2', 'p2', 'd7', 'p7'],
                                        't': ['u1', 't2', 'u6', 't7']},
                                'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                        't': ['u3', 't4', 'u5', 't6']},
                                'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                                      ('p5', 2, 'u4')]}
                 }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Petri(**params_N6)

    # 2) 数据收集
    obs,actions,mask = env.collect_expert_data()

    # 3) 预处理：低维表示/压平等
    from enviroment import impress_m
    tmp = [impress_m(o,env.idle_idx) for o in obs]
    #tmp = [low_dim(o,env.low_dim_idx) for o in tmp]

    # 4) 构造 state -> actions / mask（去重）
    state2actions = defaultdict(set)
    state2mask = defaultdict(set)

    for o, a, m in zip(tmp, actions,mask):
        key = obs_to_key(o)
        state2actions[key].add(int(a))
        #state2mask[key].add(m)

    # 打印统计信息，便于调试
    conflict_states = {
        k: v for k, v in state2actions.items()
        if len(v) > 1
    }

    print(f'总状态数量: {len(state2actions)}')
    print(f"冲突状态数量: {len(conflict_states)}")

    for i, (k, v) in enumerate(conflict_states.items()):
        print(f"state {i}: actions = {v}")

    # 5) 将去重的状态/动作整理为数组，准备构造 ReplayBuffer
    new_obs = np.array([k for k in state2actions.keys()])
    new_actions = []
    new_mask = []
    for k in state2actions.values():
        new_actions.append(list(k)[0])  # 取第一个动作作为代表
    for k in state2mask.values():
        new_actions.append(list(k)[0])

    rb = build_expert_buffer(new_obs, new_mask, new_actions)

    # 6) 构建模型并预训练
    policy_backbone = MaskedPolicyHead(hidden=128, n_obs=17, n_actions=env.T)
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

    pretrain_bc(policy, td_module, rb, device=device)


if __name__ == "__main__":
    main()