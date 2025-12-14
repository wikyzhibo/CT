import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
from solutions.pdr.net import Petri

class CT(EnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False


    def __init__(self,device='cpu',seed=None,**kwargs):

        super().__init__(device=device)
        self.net = Petri(**kwargs)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        self.observation_spec = Composite(
            observation=Bounded(low=0, high=50, shape=(n_p,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=n_t, dtype=torch.bool),
            #phi_s = Unbounded(shape=(1,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=n_t,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            deadlock_type=Bounded(low=0,high=4,shape=(1,), dtype=torch.int64, device=self.device),
            finish= Unbounded(shape=(1,), dtype=torch.bool),
            #truncated =Unbounded(shape=(), dtype=torch.bool),
        )

    def _reset(self,td_params):
        self.net.reset()
        obs = self.net.m0.copy()
        action_mask = self.net.mask_t(obs)
        out = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.int64),
                          "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                          #"phi_s": torch.tensor([phi_s], dtype=torch.float32),
                          "time": torch.tensor([0], dtype=torch.int64),
                          })
        return out

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        time = tensordict["time"].item()

        #Petri网子类交互
        info = self.net.step(action,update=True)
        new_time = info["time"]
        finish = info["finish"]
        new_obs = info['m']
        mask_next = info['mask']

        # ---死锁大惩罚 ---
        if info['deadlock']:
            r_dead = -10000.
            # 死锁标识
            deadlock_type = 1
        else:
            r_dead = 0.
            #好标识
            deadlock_type = 0

        r_time = - 0.1 * (new_time - time)
        reward =  r_dead + int(r_time)
        terminated = bool(deadlock_type or finish)

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next,dtype=torch.bool),
            "time":torch.tensor([new_time], dtype=torch.int64),
            "finish":torch.tensor(finish, dtype=torch.bool),
            #"phi_s": torch.tensor([phi_sp],dtype=torch.float32),
            "reward": torch.tensor([reward],dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "deadlock_type": torch.tensor(deadlock_type,dtype=torch.int64),
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


    def __init__(self, device='cpu',seed=None):

        super().__init__(device=device)
        self.net = Petri(csv_path='../../N1p21t16.CSV', n_t=16)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        self.observation_spec = Composite(
            observation=Bounded(low=0, high=20, shape=(5,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=5, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=5,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            deadlock_type=Bounded(low=0,high=4,shape=(1,), dtype=torch.int64, device=self.device),
            finish= Unbounded(shape=(1,), dtype=torch.bool),
            #truncated =Unbounded(shape=(), dtype=torch.bool),
        )

    def _reset(self,td_params):
        self.net.reset()
        obs = self.net.m0.copy()
        obs = self._encode(obs)
        action_mask = self._gen_mask(obs)
        out = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.int64),
                          "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                          "time": torch.tensor([0], dtype=torch.int64),
                          })
        return out

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        time = tensordict["time"].item()
        s = [[0, 1, 2, 3, 4, 5],
              [0, 1, 2, 15],
              np.arange(6, 14),
              [14, 9, 10, 11, 12, 13]]
        s = s[action]
        for t in s:
            new_obs, _, finish ,deadlock, _, info = self.net.step(t)
            if deadlock:
                break
        new_time = info["time"]

        # ---死锁大惩罚 ---
        if deadlock:
            r_dead = -10000.
            # 死锁标识
            deadlock_type = 1
        else:
            r_dead = 0.
            #好标识
            deadlock_type = 0

        r_time = - 0.2 * (new_time - time)
        reward =  r_dead + int(r_time)
        terminated = bool(finish or deadlock_type)
        new_obs = self._encode(new_obs)
        mask_next = self._gen_mask(new_obs)

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next,dtype=torch.bool),
            "time":torch.tensor([new_time], dtype=torch.int64),
            "finish":torch.tensor(finish, dtype=torch.bool),
            "reward": torch.tensor([reward],dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "deadlock_type": torch.tensor(deadlock_type,dtype=torch.int64),
        }, batch_size=[])
        return out

    def _encode(self,obs):
        return [obs[i] for i in [0,6,10,14,15]]

    def _gen_mask(self,obs):
        k = [100,4,1,100,1]
        # t1: LP->PM3
        # t2:LP->PM1
        # t3:PM3->PM2
        # t4:PM1->PM2
        # t5:PM2->LP
        pre = [[1,1, 0, 0, 0],
               [0,  0,1, 0, 0],
               [0,  0, 0, 0,1],
               [0,  0, 0, 0,0],
               [0,  0, 0,1, 0]]
        pst =  [[0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0]]
        pre = np.array(pre)
        pst = np.array(pst)
        mask = np.zeros(5, dtype=np.bool)
        for t in range(5):
            if (obs + pst[:,t] <= k).all() and (obs >= pre[:,t]).all():
                mask[t] = True
        return mask


    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng