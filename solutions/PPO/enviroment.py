import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
#from solutions.PDR.net import Petri
import copy

from solutions.v2.net_v2 import PetriNet
from solutions.v3.net_v3 import PetriV3
from solutions.Td_petri.tdpn import TimedPetri
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

        '''
        r_over = 0
        if qtime_violation:
            r_over = -100
        if over_time:
            r_over = -5000
        '''


        delta_time = time - last_time
        if delta_time > 0:
            r_time = -1 * delta_time
        else:
            r_time = 0
        reward = reward1 * 100
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