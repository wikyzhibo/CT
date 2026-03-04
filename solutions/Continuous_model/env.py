"""
连续时间 Petri 网 RL 环境：双机械手并发动作的 TorchRL 封装。

本模块定义 Env_PN_Concurrent，将 solutions.Continuous_model.pn 中的 Petri 网封装为
兼容 TorchRL 的 EnvBase 环境。为保持兼容，solutions.PPO.enviroment 会 re-export 此类。
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri


class Env_PN_Concurrent(EnvBase):
    """
    支持双机械手并发动作的 Petri 网环境。

    动作空间：
    - action_tm2: TM2 的动作（10 个变迁 + 1 个 WAIT = 11 个）
    - action_tm3: TM3 的动作（4 个变迁 + 1 个 WAIT = 5 个）

    观测空间：
    - observation: 晶圆状态观测
    - action_mask_tm2: TM2 可用动作掩码
    - action_mask_tm3: TM3 可用动作掩码
    """
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False

    # TM2 控制的变迁名称（与 pn.py 中的 TM2_TRANSITIONS 对应）
    TM2_TRANSITION_NAMES = [
        "u_LP1_s1", "u_LP2_s1", "u_s1_s2", "u_s1_s5", "u_s4_s5", "u_s5_LP_done",
        "t_s1", "t_s2", "t_s5", "t_LP_done"
    ]
    # TM3 控制的变迁名称
    TM3_TRANSITION_NAMES = [
        "u_s2_s3", "u_s3_s4", "t_s3", "t_s4"
    ]

    def __init__(self, device='cpu', seed=None, detailed_reward: bool = False, training_phase: int = 2,
                 reward_config: Optional[Dict[str, int]] = None):
        super().__init__(device=device)
        self.training_phase = training_phase

        # 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        config_path = os.path.join(project_root, "data", "petri_configs", f"phase{training_phase}_config.json")
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        config = PetriEnvConfig.load(config_path)
        self.net = Petri(config=config)

        # 构建变迁名称到索引的映射
        self._t_name_to_idx = {name: i for i, name in enumerate(self.net.id2t_name)}

        # TM2 动作空间：10 个变迁 + 1 个 WAIT
        self.tm2_transition_indices = [self._t_name_to_idx[name] for name in self.TM2_TRANSITION_NAMES
                                       if name in self._t_name_to_idx]
        self.n_actions_tm2 = len(self.tm2_transition_indices) + 1  # +1 for WAIT
        self.tm2_wait_action = len(self.tm2_transition_indices)  # WAIT 是最后一个动作

        # TM3 动作空间：4 个变迁 + 1 个 WAIT
        self.tm3_transition_indices = [self._t_name_to_idx[name] for name in self.TM3_TRANSITION_NAMES
                                       if name in self._t_name_to_idx]
        self.n_actions_tm3 = len(self.tm3_transition_indices) + 1
        self.tm3_wait_action = len(self.tm3_transition_indices)

        self.detailed_reward = detailed_reward
        self.n_wafer = config.n_wafer

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        obs_dim = 12 * 6  # 与 Env_PN 相同
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.int64, device=self.device),
            action_mask_tm2=Binary(n=self.n_actions_tm2, dtype=torch.bool),
            action_mask_tm3=Binary(n=self.n_actions_tm3, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        # 双动作空间
        self.action_spec = Composite(
            action_tm2=Categorical(n=self.n_actions_tm2, shape=(1,), dtype=torch.int64),
            action_tm3=Categorical(n=self.n_actions_tm3, shape=(1,), dtype=torch.int64),
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            finish=Unbounded(shape=(1,), dtype=torch.bool),
            scrap=Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_obs(self):
        """复用 Env_PN 的观测构建逻辑"""
        MAX_WAFERS = 12
        processing_wafers = {}
        lp1_wafers = []
        lp2_wafers = []

        for p_idx, place in enumerate(self.net.marks):
            if place.type == 4:
                continue
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue
                stay_time = int(tok.stay_time)
                color = getattr(tok, 'color', 0)
                if place.type == 1:
                    time_to_scrap = 20 - (stay_time - place.processing_time)
                elif place.type == 2:
                    time_to_scrap = 10 - stay_time
                else:
                    time_to_scrap = -1
                wafer_tuple = (tok.token_id, p_idx, place.type, stay_time, time_to_scrap, color)
                if place.type in (1, 2, 5):
                    processing_wafers[tok.token_id] = wafer_tuple
                elif place.type == 3:
                    if place.name == "LP1":
                        lp1_wafers.append(wafer_tuple)
                    elif place.name == "LP2":
                        lp2_wafers.append(wafer_tuple)

        selected_wafers = list(processing_wafers.values())
        if len(selected_wafers) < MAX_WAFERS and len(lp1_wafers) > 0:
            selected_wafers.append(lp1_wafers[0])
        if len(selected_wafers) < MAX_WAFERS and len(lp2_wafers) > 0:
            selected_wafers.append(lp2_wafers[0])
        selected_wafers.sort(key=lambda x: x[0])

        obs = []
        for i in range(MAX_WAFERS):
            if i < len(selected_wafers):
                obs.extend(selected_wafers[i])
            else:
                obs.extend([0, 0, 0, 0, 0, 0])
        return np.array(obs, dtype=np.int64)

    def _build_action_masks(self):
        """构建 TM2/TM3 各自的动作掩码"""
        tm2_enabled, tm3_enabled = self.net.get_enable_t()
        tm2_enabled_set = set(tm2_enabled)
        tm3_enabled_set = set(tm3_enabled)

        # TM2 掩码
        mask_tm2 = np.zeros(self.n_actions_tm2, dtype=bool)
        for i, t_idx in enumerate(self.tm2_transition_indices):
            mask_tm2[i] = (t_idx in tm2_enabled_set)
        mask_tm2[self.tm2_wait_action] = True  # WAIT 始终可用

        # TM3 掩码
        mask_tm3 = np.zeros(self.n_actions_tm3, dtype=bool)
        for i, t_idx in enumerate(self.tm3_transition_indices):
            mask_tm3[i] = (t_idx in tm3_enabled_set)
        mask_tm3[self.tm3_wait_action] = True

        return mask_tm2, mask_tm3

    def _reset(self, td_params):
        self.net.reset()
        obs = self._build_obs()
        mask_tm2, mask_tm3 = self._build_action_masks()
        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask_tm2": torch.as_tensor(mask_tm2, dtype=torch.bool),
            "action_mask_tm3": torch.as_tensor(mask_tm3, dtype=torch.bool),
            "time": torch.tensor([self.net.time], dtype=torch.int64),
        })

    def _step(self, tensordict=None):
        action_tm2 = tensordict["action_tm2"].item()
        action_tm3 = tensordict["action_tm3"].item()

        # 将环境动作索引转换为 Petri 网变迁索引
        if action_tm2 == self.tm2_wait_action:
            a1 = None
        else:
            a1 = self.tm2_transition_indices[action_tm2]

        if action_tm3 == self.tm3_wait_action:
            a2 = None
        else:
            a2 = self.tm3_transition_indices[action_tm3]

        # 调用并发 step
        done, reward_result, scrap = self.net.step_concurrent(
            a1=a1, a2=a2,
            with_reward=True, detailed_reward=self.detailed_reward
        )

        if isinstance(reward_result, dict):
            reward = reward_result.get('total', 0)
        else:
            reward = reward_result

        obs = self._build_obs()
        mask_tm2, mask_tm3 = self._build_action_masks()
        time = self.net.time
        terminated = bool(done)
        finish = done and not scrap

        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask_tm2": torch.as_tensor(mask_tm2, dtype=torch.bool),
            "action_mask_tm3": torch.as_tensor(mask_tm3, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "scrap": torch.tensor(scrap, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
        }, batch_size=[])

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng
