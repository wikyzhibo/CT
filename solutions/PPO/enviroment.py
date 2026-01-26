import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
#from solutions.PDR.net import Petri
import copy

from solutions.v2.net_v2 import PetriNet
from data.config.params_N7 import params_N7

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

        self.net = PetriNet(with_controller=True,
                       with_capacity_controller=True,
                       with_zhiliu_controller=False,
                       **params_N7)
        self.n_actions = self.net.T


        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        # 使用 self.n_actions，按 allow_idle 决定动作数量
        self.observation_spec = Composite(
            observation=Bounded(low=0, high=76, shape=(n_p,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=self.n_actions,shape=(1,),dtype=torch.int64)
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

        mask_next, new_obs, time, qtime_violation, over_time, finish = self.net.step(action)

        r_over = 0
        if qtime_violation:
            r_over = -100
        if over_time:
            r_over = -5000

        delta_time = time - last_time
        if delta_time > 0:
            r_time = -1 * delta_time
        else:
            r_time = 0

        reward = int(r_time) + r_over

        terminated = finish

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "overtime": torch.tensor(qtime_violation+over_time, dtype=torch.int64),
        }, batch_size=[])

        return out

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng

<<<<<<< Updated upstream
=======
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
        if training_phase == 1:
            config = PetriEnvConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\petri_configs\phase1_config.json")
        else:
            config = PetriEnvConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\petri_configs\phase2_config.json")

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
        # 观测维度 = n_wafer * 5 (每个 wafer 的五元组)
        # 五元组: (token_id, place_idx, place_type, stay_time, time_to_scrap)
        # 新六步路线：最多 9 个晶圆同时可见（s1:2 + s2:1 + s3:2 + s4:1 + s5:2 + LP:1）
        obs_dim = 7 * 5
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
        每个 wafer 包含: (token_id, place_idx, place_type, stay_time, time_to_scrap)
        
        选择逻辑：
        1. 优先收集所有在加工腔室（type=1）和运输位（type=2）中的 wafer
        2. 如果不满 3 个，从 LP（type=3）中取队首的 1 个 wafer
        3. 不够 3 个则用全零的 5 元组补齐
        4. 按 token_id 升序排列后输出
        
        观测维度 = 9 * 5 = 45
        """
        MAX_WAFERS = 7  # 新六步路线最多 9 个晶圆同时可见
        
        # 收集加工区（type=1, 2）的 wafer 信息
        processing_wafers = {}  # token_id -> (place_idx, place_type, stay_time, time_to_scrap)
        lp_wafers = []  # LP 中的 wafer 列表，按队列顺序（队首在前）
        
        for p_idx, place in enumerate(self.net.marks):
            # 跳过资源库所（type=4 且名称以 r_ 开头）和终点（LP_done）
            if place.type == 4:
                continue
            
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue  # 跳过无效 token
                
                stay_time = int(tok.stay_time)
                
                # 计算距离报废时间
                if place.type == 1:  # 加工腔室
                    time_to_scrap = 20 - (stay_time - place.processing_time)
                elif place.type == 2:  # 运输位
                    time_to_scrap = 10 - stay_time
                else:  # LP 等
                    time_to_scrap = -1  # 无报废风险
                
                wafer_tuple = (tok.token_id, p_idx, place.type, stay_time, time_to_scrap)
                
                if place.type in (1, 2):  # 加工腔室或运输位
                    processing_wafers[tok.token_id] = wafer_tuple
                elif place.type == 3:  # LP
                    lp_wafers.append(wafer_tuple)
        
        # 选择要展示的 wafer
        selected_wafers = list(processing_wafers.values())
        
        # 如果加工区 wafer 不满 MAX_WAFERS 个，从 LP 取 1 个队首 wafer
        if len(selected_wafers) < MAX_WAFERS and len(lp_wafers) > 0:
            selected_wafers.append(lp_wafers[0])
        
        # 按 token_id 升序排列
        selected_wafers.sort(key=lambda x: x[0])
        
        # 构建观测向量
        obs = []
        for i in range(MAX_WAFERS):
            if i < len(selected_wafers):
                tid, p_idx, p_type, stay, scrap_time = selected_wafers[i]
                obs.extend([tid, p_idx, p_type, stay, scrap_time])
            else:
                # 不够 3 个则用全零补齐
                obs.extend([0, 0, 0, 0, 0])
        
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





>>>>>>> Stashed changes
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