import numpy as np
from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
from net import Petri

class CT(EnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False


    def __init__(self, device='cpu',seed=None):

        super().__init__(device=device)
        self.net = Petri(csv_path='N1p21t16.CSV', n_t=16)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        self.observation_spec = Composite(
            observation=Bounded(low=0, high=20, shape=(n_p-2,), dtype=torch.int64, device=self.device),
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
        #phi_s = self.net.residual_process_time(self.net.m)
        out = TensorDict({"observation": torch.as_tensor(np.delete(obs,[0,14]), dtype=torch.int64),
                          "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                          #"phi_s": torch.tensor([phi_s], dtype=torch.float32),
                          "time": torch.tensor([0], dtype=torch.int64),
                          })
        return out

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        #phi_s = tensordict["phi_s"].item()
        time = tensordict["time"].item()
        #if action == 15 or action == 14:
            #print("pause")
        #Petri网子类交互
        new_obs, mask_next, finish ,deadlock, _, info = self.net.step(action)
        new_time = info["time"]

        #r1_t_ids = [0,1,2,3,4,13,14,15]
        #r2_t_ids = [5,6,7,8,9,10,11,12,13]
        #r1_mask = mask_next[r1_t_ids]
        #r2_mask = mask_next[r2_t_ids]
        #r1_src = new_obs[17].item()
        #r2_src = new_obs[18].item()

        # ---死锁大惩罚 ---
        if deadlock:
            r_dead = -10000.
            # 死锁标识
            deadlock_type = 1
        #elif (r1_src==0 and r1_mask.sum()== 0)  or (r2_src==0 and r2_mask.sum() == 0):
        #    r_dead = -10000.
            # 坏标识类型1
        #    deadlock_type = 1
        else:
            r_dead = 0.
            #好标识
            deadlock_type = 0

        # --- PBRS：Φ(s)-Φ(s') ---
        #r_pbrs = 0.1 * (phi_s - phi_sp)  # 越减少越正

        # -- 时间增加惩罚
        r_time = - 0.2 * (new_time - time)

        reward =  r_dead + int(r_time)
        terminated = bool(finish or deadlock_type)

        out = TensorDict({
            "observation": torch.as_tensor(np.delete(new_obs,[0,14]), dtype=torch.int64),
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