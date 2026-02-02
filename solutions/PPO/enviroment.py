import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
#from solutions.PDR.net import Petri
import copy
from typing import Dict, Optional
import os

#from solutions.v2.net_v2 import PetriNet
#from solutions.v3.net_v3 import PetriV3
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Continuous_model.pn import Petri
#from data.config.params_N7 import params_N7
from data.petri_configs.env_config import PetriEnvConfig

def impress_m(m, idle):
    m_new = m.copy()
    for i in idle.values():
        if m[i] > 0:
            m_new[i] = 1
    return m_new

def low_dim(m, idx):
    return np.array([m[i] for i in idx])

class CT_v2(EnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False


    def __init__(self, device='cpu', seed=None):

        super().__init__(device=device)

        #self.net = PetriV3(with_controller=True)
        self.net = TimedPetri()
        self.n_actions = self.net.T


        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        obs_dim = self.net.obs_dim
        act_dim = self.net.A

        # 使用 self.n_actions，按 allow_idle 决定动作数量
        self.observation_spec = Composite(
            observation=Bounded(low=0, high=76, shape=(obs_dim,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=act_dim,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            overtime=Bounded(low=0, high=4, shape=(1,), dtype=torch.int64, device=self.device),
            finish= Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_state_td(self, obs, action_mask, time):


        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
        })

    def _reset(self,td_params):
        obs, mask = self.net.reset()

        return self._build_state_td(obs, mask, time=0)

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        last_time = tensordict["time"].item()

        mask_next, new_obs, time, finish, reward1 = self.net.step(action)


        delta_time = time - last_time
        if delta_time > 0:
            r_time = -1 * delta_time
        else:
            r_time = 0
        #reward = reward1 * 1000
        reward = r_time
        #reward = int(-delta_dense*1000)

        terminated = finish

        assert len(mask_next) == 16, f"mask_next length={len(mask_next)}, expected 16"
        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "overtime": torch.tensor(0, dtype=torch.int64),
        }, batch_size=[])

        return out

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng


class Env_PN(EnvBase):
    # 连续系统的PPO环境
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False

    def __init__(self, device='cpu', seed=None, detailed_reward: bool = False, training_phase: int = 2,
                 reward_config: Optional[Dict[str, int]] = None):
        """
        初始化连续 Petri 网环境。

        Args:
            device: 计算设备
            seed: 随机种子
            detailed_reward: 是否使用详细奖励模式
            training_phase: 训练阶段
                - 1: 仅考虑报废惩罚（加工腔室超时）
                - 2: 完整奖励（加工腔室超时 + 运输位超时）
            reward_config: 奖励开关配置字典，1=启用，0=禁用
                - 'proc_reward': 加工奖励
                - 'safe_reward': 安全裕量奖励
                - 'penalty': 加工腔室超时惩罚
                - 'warn_penalty': 预警惩罚
                - 'transport_penalty': 运输位超时惩罚
                - 'congestion_penalty': 堵塞预测惩罚
                - 'time_cost': 时间成本
        """
        super().__init__(device=device)
        self.training_phase = training_phase
        
        # 使用相对路径（跨平台兼容）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        config_path = os.path.join(project_root, "data", "petri_configs", f"phase{training_phase}_config.json")
        config_path = os.path.abspath(config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        config = PetriEnvConfig.load(config_path)

        self.net = Petri(config=config)
        self.n_actions = self.net.T + 1  # wait action at index net.T
        self.detailed_reward = detailed_reward  # 是否使用详细奖励模式

        # wafer 数量（从 LP 初始 token 数获取）
        self.n_wafer = config.n_wafer

        # 计算加工腔室数量（用于增强观测）
        self.chamber_indices = [
            i for i, place in enumerate(self.net.marks) if place.type == 1
        ]
        self.n_chambers = len(self.chamber_indices)

        # 存储最近一次的详细奖励信息（用于调试）
        self.last_reward_detail = None
        self.last_scrap_info = None

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        # 观测维度 = n_wafer * 6 (每个 wafer 的六元组)
        # 六元组: (token_id, place_idx, place_type, stay_time, time_to_scrap, color)
        # 双路线：最多 9 个晶圆同时可见（s1:2 + s2:1 + s3:4 + s4:1 + s5:2 + LP1:1 + LP2:1）
        obs_dim = 12 * 6
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=self.n_actions, shape=(1,), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            finish=Unbounded(shape=(1,), dtype=torch.bool),
            scrap=Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_state_td(self, obs, action_mask, time):
        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
        })

    def _build_obs(self):
        """
        构建观测向量：只展示机械手可操作的 wafer 信息
        每个 wafer 包含: (token_id, place_idx, place_type, stay_time, time_to_scrap, color)

        选择逻辑：
        1. 优先收集所有在加工腔室（type=1）、运输位（type=2）和无驻留约束腔室（type=5）中的 wafer
        2. 如果不满 MAX_WAFERS 个，分别从 LP1 和 LP2（type=3）中各取队首的 1 个 wafer
        3. 不够则用全零的 6 元组补齐
        4. 按 token_id 升序排列后输出

        观测维度 = 9 * 6 = 54
        """
        MAX_WAFERS = 12  # 双路线：s1:2 + s2:1 + s3:4 + s4:1 + s5:2 + LP1:1 + LP2:1

        # 收集加工区（type=1, 2, 5）的 wafer 信息
        processing_wafers = {}  # token_id -> (token_id, place_idx, place_type, stay_time, time_to_scrap, color)
        lp1_wafers = []  # LP1 中的 wafer 列表，按队列顺序（队首在前）
        lp2_wafers = []  # LP2 中的 wafer 列表，按队列顺序（队首在前）

        for p_idx, place in enumerate(self.net.marks):
            # 跳过资源库所（type=4 且名称以 r_ 开头）和终点（LP_done）
            if place.type == 4:
                continue

            for tok in place.tokens:
                if tok.token_id < 0:
                    continue  # 跳过无效 token

                stay_time = int(tok.stay_time)
                color = getattr(tok, 'color', 0)  # 获取晶圆颜色

                # 计算距离报废时间
                if place.type == 1:  # 加工腔室（有驻留约束）
                    time_to_scrap = 20 - (stay_time - place.processing_time)
                elif place.type == 2:  # 运输位
                    time_to_scrap = 10 - stay_time
                elif place.type == 5:  # 无驻留约束腔室（s2/s4）
                    time_to_scrap = -1  # 无报废风险
                else:  # LP1/LP2 等
                    time_to_scrap = -1  # 无报废风险

                wafer_tuple = (tok.token_id, p_idx, place.type, stay_time, time_to_scrap, color)

                if place.type in (1, 2, 5):  # 加工腔室、运输位或无驻留约束腔室
                    processing_wafers[tok.token_id] = wafer_tuple
                elif place.type == 3:  # LP1 或 LP2
                    if place.name == "LP1":
                        lp1_wafers.append(wafer_tuple)
                    elif place.name == "LP2":
                        lp2_wafers.append(wafer_tuple)

        # 选择要展示的 wafer
        selected_wafers = list(processing_wafers.values())

        # 如果加工区 wafer 不满 MAX_WAFERS 个，从 LP1 取 1 个队首 wafer
        if len(selected_wafers) < MAX_WAFERS and len(lp1_wafers) > 0:
            selected_wafers.append(lp1_wafers[0])

        # 如果仍不满 MAX_WAFERS 个，从 LP2 取 1 个队首 wafer
        if len(selected_wafers) < MAX_WAFERS and len(lp2_wafers) > 0:
            selected_wafers.append(lp2_wafers[0])

        # 按 token_id 升序排列
        selected_wafers.sort(key=lambda x: x[0])

        # 构建观测向量（6 元组）
        obs = []
        for i in range(MAX_WAFERS):
            if i < len(selected_wafers):
                tid, p_idx, p_type, stay, scrap_time, color = selected_wafers[i]
                obs.extend([tid, p_idx, p_type, stay, scrap_time, color])
            else:
                # 不够则用全零补齐
                obs.extend([0, 0, 0, 0, 0, 0])

        return np.array(obs, dtype=np.int64)

    def _reset(self, td_params):
        self.net.reset()
        obs = self._build_obs()
        action_mask = self.net.get_enable_t()
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[action_mask] = True
        mask[self.net.T] = True  # wait always enabled
        return self._build_state_td(obs, mask, time=self.net.time)

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        last_time = tensordict["time"].item()

        if action == self.net.T:
            done, reward_result, scrap = self.net.step(
                wait=True, with_reward=True, detailed_reward=self.detailed_reward
            )
        else:
            done, reward_result, scrap = self.net.step(
                t=action, with_reward=True, detailed_reward=self.detailed_reward
            )

        # 处理详细奖励模式
        if self.detailed_reward and isinstance(reward_result, dict):
            self.last_reward_detail = reward_result
            self.last_scrap_info = reward_result.get('scrap_info')
            reward = reward_result.get('total', 0)
        else:
            self.last_reward_detail = None
            self.last_scrap_info = None
            reward = reward_result

        obs = self._build_obs()
        action_mask = self.net.get_enable_t()
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[action_mask] = True
        mask[self.net.T] = True

        time = self.net.time
        terminated = bool(done)
        # finish 表示正常完成（所有 wafer 到达终点），scrap 表示报废
        finish = done and not scrap

        out = TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "scrap": torch.tensor(scrap, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
        }, batch_size=[])

        return out

    def get_scrap_info(self):
        """获取最近一次报废的详细信息（仅在 detailed_reward=True 时有效）"""
        return self.last_scrap_info

    def get_reward_detail(self):
        """获取最近一次的详细奖励分解（仅在 detailed_reward=True 时有效）"""
        return self.last_reward_detail

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng


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
        
        from data.petri_configs.env_config import PetriEnvConfig
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
        tm2_enabled, tm3_enabled = self.net.get_enable_t_by_robot()
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



class CT(EnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False

    def __init__(self, device='cpu', seed=None, allow_idle: bool = True, generation: int = 0, **kwargs):

        super().__init__(device=device)
        # 将训练分段（代）信号传入 Petri

        net = PetriNet(with_controller=True,
                       with_capacity_controller=True,
                       with_zhiliu_controller=False,
                       **params_N7)
        # 保存是否允许空转，并计算动作个数与空转动作索引（若允许空转，则空转为索引 self.net.T）
        self.allow_idle = allow_idle
        self.n_actions = self.net.T + (1 if self.allow_idle else 0)
        self.idle_action = self.net.T if self.allow_idle else None

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.idle_count = 0

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T
        #in_dim = len(self.net.low_dim_idx) + self.net.n_wafer
        in_dim = len(self.net.low_dim_idx)

        # 使用 self.n_actions，按 allow_idle 决定动作数量
        self.observation_spec = Composite(
            observation=Bounded(low=0, high=76, shape=(in_dim,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            #phi_s = Unbounded(shape=(1,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=self.n_actions,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            deadlock_type=Bounded(low=0,high=4,shape=(1,), dtype=torch.int64, device=self.device),
            overtime=Bounded(low=0, high=4, shape=(1,), dtype=torch.int64, device=self.device),
            finish= Unbounded(shape=(1,), dtype=torch.bool),
            #truncated =Unbounded(shape=(), dtype=torch.bool),
        )

    def _build_state_td(self, obs, action_mask, time):

        obs = impress_m(obs, self.net.idle_idx)
        obs = low_dim(obs, self.net.low_dim_idx)
        #obs = np.concatenate([obs, np.zeros(self.net.n_wafer)])

        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
        })

    def _snapshot(self):
        """Capture Petri net state for backtracking."""
        return {
            "m": self.net.m.copy(),
            "marks": [copy.deepcopy(mk) for mk in self.net.marks],
            "time": self.net.time,
        }

    def _restore(self, snapshot):
        self.net.m = snapshot["m"].copy()
        self.net.marks = [copy.deepcopy(mk) for mk in snapshot["marks"]]
        self.net.time = snapshot["time"]
        obs = self.net.m.copy()
        action_mask = self.net.mask_t(obs, self.net.marks)
        return self._build_state_td(obs, action_mask, self.net.time)

    def _reset(self,td_params):
        self.net.reset()
        obs = self.net.m0.copy()
        mark = self.net.marks
        action_mask = self.net.mask_t(obs, mark)
        # 仅当允许空转时追加空转位
        if self.allow_idle:
            action_mask = np.concatenate([action_mask,[True]])
        # 超时记录已移入 Petri.net 内部（self.net._overtime_record）

        return self._build_state_td(obs, action_mask, time=0)

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        time = tensordict["time"].item()

        info = self.net.step(action)
        new_time = info["time"]
        finish = info["finish"]
        deadlock = info["deadlock"]
        mask_next = info["mask"]
        time_violation_type = info['time_violation_type']
        new_obs = info["m"]
        new_obs = impress_m(new_obs, self.net.idle_idx)
        new_obs = low_dim(new_obs, self.net.low_dim_idx)

        match time_violation_type:
            case 1: #驻留时间约束
                deadlock_type = 1
                r_over = -5000
            case 2:
                deadlock_type = 0
                r_over = -100
            case _:
                deadlock_type = 0
                r_over = 0

        # --- 死锁惩罚 ---
        if deadlock:
            r_dead = -10000.
            deadlock_type = 1
        else:
            r_dead = 0.
            deadlock_type = 0

        r_time = -1 * (new_time - time)
        reward = r_dead + int(r_time) + r_over
        if finish:
            reward += 1

        terminated = bool(deadlock_type or finish)

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next, dtype=torch.bool),
            "time": torch.tensor([new_time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "deadlock_type": torch.tensor(deadlock_type, dtype=torch.int64),
            "overtime": torch.tensor(1 if time_violation_type == 1 else 0, dtype=torch.int64),
        }, batch_size=[])

        return out

    def _render(self):
        pass

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng

class CT2(EnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False


    def __init__(self, device='cpu', seed=None, allow_idle: bool = True, generation: int = 0, **kwargs):

        super().__init__(device=device)
        # 将训练分段（代）信号传入 Petri
        self.generation = generation
        self.net = Petri(with_controller=True,
                         with_capacity_controller=True,
                         with_zhiliu_controller=True,
                         generation=self.generation,
                         **kwargs)
        # 保存是否允许空转，并计算动作个数与空转动作索引（若允许空转，则空转为索引 self.net.T）
        self.n_actions = self.net.T

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T
        #in_dim = len(self.net.low_dim_idx) + self.net.n_wafer
        in_dim = len(self.net.low_dim_idx)

        # 使用 self.n_actions，按 allow_idle 决定动作数量
        self.observation_spec = Composite(
            observation=Bounded(low=0, high=50, shape=(in_dim,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            #phi_s = Unbounded(shape=(1,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=self.n_actions,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            deadlock_type=Bounded(low=0,high=4,shape=(1,), dtype=torch.int64, device=self.device),
            overtime=Bounded(low=0, high=4, shape=(1,), dtype=torch.int64, device=self.device),
            finish= Unbounded(shape=(1,), dtype=torch.bool),
            #truncated =Unbounded(shape=(), dtype=torch.bool),
        )

    def _build_state_td(self, obs, action_mask, time):

        obs = impress_m(obs, self.net.idle_idx)
        obs = low_dim(obs, self.net.low_dim_idx)
        #obs = np.concatenate([obs, np.zeros(self.net.n_wafer)])

        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
        })

    def _reset(self,td_params):
        self.net.reset()
        obs = self.net.m0.copy()
        mark = self.net.marks
        action_mask = self.net.mask_t(obs, mark)
        return self._build_state_td(obs, action_mask, time=0)

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        time = tensordict["time"].item()
        info = self.net.step(action)
        new_time = info["time"]
        finish = info["finish"]
        deadlock = info["deadlock"]
        mask_next = info["mask"]
        time_violation_type = info['time_violation_type']
        new_obs = info["m"]
        finish_a_wafer = info["finish_a_wafer"]
        new_obs = impress_m(new_obs, self.net.idle_idx)
        new_obs = low_dim(new_obs, self.net.low_dim_idx)

        r_over = 0
        deadlock_type = 0
        match time_violation_type:
            case 1: #违反驻留时间约束
                deadlock_type = 1
                r_over = -5000
            case 2: #违反运输时间约束
                deadlock_type = 0
                r_over = -100
            case _:
                deadlock_type = 0

        # ---- time shaping ----
        dt = max(0, new_time - time)

        # penalty per second (tune)
        alpha = 0.5
        r_time = -alpha * dt

        if finish_a_wafer:
            r_output = 200
        else:
            r_output = 0

        reward = r_output + r_over + r_time

        if new_time > 3600 * 24:
            terminated = True
        else:
            terminated = False

        if finish:
            terminated = True

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next, dtype=torch.bool),
            "time": torch.tensor([new_time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "deadlock_type": torch.tensor(deadlock_type, dtype=torch.int64),
            "overtime": torch.tensor(1 if time_violation_type == 1 else 0, dtype=torch.int64),
        }, batch_size=[])

        return out

    def _render(self):
        pass

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng