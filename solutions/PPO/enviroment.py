import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
#from solutions.PDR.net import Petri
import copy
from typing import Dict, Optional
import os

# Legacy imports removed - v2 and v3 have been deprecated
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

        # 观测特征开关
        self.obs_features = config.obs_features or {}
        self.obs_dim = self._calc_obs_dim()

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

    def _calc_obs_dim(self):
        """根据启用的观测特征计算 obs 维度"""
        MAX_WAFERS = 12
        per_wafer = 6  # 基础 6 元组
        if self.obs_features.get("wafer_progress"):
            per_wafer += 2  # step + remaining_steps
        dim = MAX_WAFERS * per_wafer
        if self.obs_features.get("robot_status"):
            dim += 4  # tm2_busy, tm3_busy, tm2_free_in, tm3_free_in
        if self.obs_features.get("chamber_occupancy"):
            dim += 6  # s1_occ, s3_occ, s5_occ, lp1_rem, lp2_rem, done_count
        if self.obs_features.get("release_times"):
            dim += 3  # s1_rel, s3_rel, s5_rel
        if self.obs_features.get("global_progress"):
            dim += 3  # done_pct, time_pct, entered_count
        if self.obs_features.get("urgency_summary"):
            dim += 2  # n_critical, min_scrap_time
        return dim

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self.obs_dim,), dtype=torch.int64, device=self.device),
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
        构建观测向量。

        基础 per-wafer 特征 (6 元组):
            (token_id, place_idx, place_type, stay_time, time_to_scrap, color)

        可选增强特征（通过 obs_features 配置开关控制）:
            robot_status:      TM2/TM3 忙碌 + 最早空闲时间（4 标量）
            chamber_occupancy: s1/s3/s5/LP1/LP2/LP_done 占用数（6 标量）
            release_times:     s1/s3/s5 最早释放时间差（3 标量）
            wafer_progress:    per-wafer step + remaining_steps（12×2=24）
            global_progress:   done_pct + time_pct + entered_count（3 标量）
            urgency_summary:   n_critical + min_scrap_time（2 标量）

        选择逻辑：
        1. 优先收集加工区（type=1/2/5）wafer
        2. 不满 MAX_WAFERS 则从 LP1/LP2 各取 1 个队首
        3. 全零补齐，按 token_id 升序排列
        """
        MAX_WAFERS = 12
        include_progress = self.obs_features.get("wafer_progress", False)

        # ── 收集 wafer 信息 ──
        processing_wafers = {}  # token_id -> tuple
        lp1_wafers = []
        lp2_wafers = []
        scrap_times_for_urgency = []  # 用于 urgency_summary

        for p_idx, place in enumerate(self.net.marks):
            if place.type == 4:
                continue

            for tok in place.tokens:
                if tok.token_id < 0:
                    continue

                stay_time = int(tok.stay_time)
                color = getattr(tok, 'color', 0)

                # 计算距离报废时间
                if place.type == 1:
                    time_to_scrap = 20 - (stay_time - place.processing_time)
                elif place.type == 2:
                    time_to_scrap = 10 - stay_time
                else:
                    time_to_scrap = -1

                # 基础 6 元组
                wafer_tuple = [tok.token_id, p_idx, place.type, stay_time, time_to_scrap, color]

                # 可选：per-wafer 工序进度
                if include_progress:
                    step = getattr(tok, 'step', 0)
                    route_type = getattr(tok, 'route_type', 1)
                    route_cfg = self.net.ROUTE_CONFIG.get(route_type, [])
                    remaining = max(0, len(route_cfg) - step)
                    wafer_tuple.extend([step, remaining])

                if place.type in (1, 2, 5):
                    processing_wafers[tok.token_id] = tuple(wafer_tuple)
                    if time_to_scrap >= 0:
                        scrap_times_for_urgency.append(time_to_scrap)
                elif place.type == 3:
                    if place.name == "LP1":
                        lp1_wafers.append(tuple(wafer_tuple))
                    elif place.name == "LP2":
                        lp2_wafers.append(tuple(wafer_tuple))

        selected_wafers = list(processing_wafers.values())
        if len(selected_wafers) < MAX_WAFERS and len(lp1_wafers) > 0:
            selected_wafers.append(lp1_wafers[0])
        if len(selected_wafers) < MAX_WAFERS and len(lp2_wafers) > 0:
            selected_wafers.append(lp2_wafers[0])
        selected_wafers.sort(key=lambda x: x[0])

        # ── 拼接 obs 向量 ──
        obs = []

        # 全局特征（前缀）
        if self.obs_features.get("robot_status"):
            obs.extend(self._obs_robot_status())
        if self.obs_features.get("chamber_occupancy"):
            obs.extend(self._obs_chamber_occupancy())
        if self.obs_features.get("release_times"):
            obs.extend(self._obs_release_times())
        if self.obs_features.get("global_progress"):
            obs.extend(self._obs_global_progress())
        if self.obs_features.get("urgency_summary"):
            obs.extend(self._obs_urgency_summary(scrap_times_for_urgency))

        # Per-wafer 特征
        per_wafer_dim = 8 if include_progress else 6
        for i in range(MAX_WAFERS):
            if i < len(selected_wafers):
                obs.extend(selected_wafers[i])
            else:
                obs.extend([0] * per_wafer_dim)

        return np.array(obs, dtype=np.int64)

    # ========== 观测辅助方法 ==========

    def _obs_robot_status(self):
        """
        机械手状态：[tm2_busy, tm3_busy, tm2_free_in, tm3_free_in]
        busy: 1=忙碌（资源库所为空），0=空闲
        free_in: 距最早空闲的时间差（秒），空闲时为 0
        """
        net = self.net
        t_now = net.time

        # TM2 忙碌状态
        tm2_busy = 0
        if "r_TM2" in net.id2p_name:
            tm2_idx = net.id2p_name.index("r_TM2")
            if net.m[tm2_idx] == 0:
                tm2_busy = 1

        # TM3 忙碌状态
        tm3_busy = 0
        if "r_TM3" in net.id2p_name:
            tm3_idx = net.id2p_name.index("r_TM3")
            if net.m[tm3_idx] == 0:
                tm3_busy = 1

        # 最早空闲时间（从时间轴预估）
        def earliest_free(timeline):
            if not timeline:
                return 0
            # 找最早结束时间 >= t_now 的区间
            max_end = 0
            for itv in timeline:
                if itv.end > t_now:
                    max_end = max(max_end, itv.end)
            return max(0, max_end - t_now)

        tm2_free_in = earliest_free(net._tm2_timeline)
        tm3_free_in = earliest_free(net._tm3_timeline)

        return [tm2_busy, tm3_busy, int(tm2_free_in), int(tm3_free_in)]

    def _obs_chamber_occupancy(self):
        """
        腔室占用数：[s1_occ, s3_occ, s5_occ, lp1_rem, lp2_rem, done_count]
        直接返回 token 数量（整数），智能体可推断负载状况
        """
        net = self.net

        def count_tokens(name):
            if name not in net.id2p_name:
                return 0
            idx = net.id2p_name.index(name)
            return len(net.marks[idx].tokens)

        return [
            count_tokens("s1"),
            count_tokens("s3"),
            count_tokens("s5"),
            count_tokens("LP1"),
            count_tokens("LP2"),
            count_tokens("LP_done"),
        ]

    def _obs_release_times(self):
        """
        腔室最早释放时间差：[s1_rel_delta, s3_rel_delta, s5_rel_delta]
        = earliest_release - current_time，无预约则 -1
        """
        net = self.net
        t_now = net.time
        result = []
        for name in ["s1", "s3", "s5"]:
            if name not in net.id2p_name:
                result.append(-1)
                continue
            idx = net.id2p_name.index(name)
            place = net.marks[idx]
            # release_schedule 是 deque[(token_id, release_time), ...]
            if not place.release_schedule:
                result.append(-1)
            else:
                earliest = min(rt for _, rt in place.release_schedule)
                result.append(max(0, earliest - t_now))
        return result

    def _obs_global_progress(self):
        """
        全局进度指标：[done_pct, time_pct, entered_count]
        百分比取整保持 int64（0~100）
        """
        net = self.net
        n_wafer = getattr(net, 'n_wafer', 12)
        max_time = getattr(net, 'MAX_TIME', 7000)
        done_count = getattr(net, 'done_count', 0)
        entered = getattr(net, 'entered_wafer_count', 0)

        done_pct = int(done_count * 100 / max(1, n_wafer))
        time_pct = int(net.time * 100 / max(1, max_time))
        return [done_pct, time_pct, entered]

    def _obs_urgency_summary(self, scrap_times):
        """
        紧急程度汇总：[n_critical, min_scrap_time]
        n_critical: time_to_scrap < 5 的 wafer 数
        min_scrap_time: 所有有报废风险 wafer 中最小的 time_to_scrap，无则 -1
        """
        if not scrap_times:
            return [0, -1]
        n_critical = sum(1 for t in scrap_times if t < 5)
        min_scrap = min(scrap_times)
        return [n_critical, min_scrap]

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

        # 观测特征开关（复用 Env_PN 的逻辑）
        self.obs_features = config.obs_features or {}
        self.obs_dim = Env_PN._calc_obs_dim(self)
        
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self.obs_dim,), dtype=torch.int64, device=self.device),
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
        return Env_PN._build_obs(self)

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

