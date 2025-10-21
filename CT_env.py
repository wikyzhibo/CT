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
        self.net = Petri(csv_path='N4p20t16.csv')
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        n_p = self.net.P
        n_t = self.net.T

        self.observation_spec = Composite(
            observation=Bounded(low=0, high=10, shape=(n_p,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=n_t, dtype=torch.bool),
            phi_s = Unbounded(shape=(1,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Categorical(n=n_t,shape=(1,),dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            #truncated =Unbounded(shape=(), dtype=torch.bool),
        )

    def _reset(self,td_params):
        self.net.reset()
        obs = self.net.m0.copy()
        action_mask = self.net.mask_t(obs)
        phi_s = self.net.residual_process_time(self.net.m)
        out = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.int64),
                          "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                          "phi_s": torch.tensor([phi_s], dtype=torch.float32),
                          "time": torch.tensor([0], dtype=torch.int64),
                          })
        return out

    def _step(self, tensordict=None):
        action = tensordict["action"].item()
        phi_s = tensordict["phi_s"].item()
        time = tensordict["time"].item()

        #Petri网子类交互
        new_obs, mask_next, done,deadlock, phi_sp, info = self.net.step(action)
        new_time = info["time"]

        # --- PBRS：Φ(s)-Φ(s') ---
        r_pbrs = 0.1 * (phi_s - phi_sp)  # 越减少越正

        # -- 时间增加惩罚
        r_time = - 0.1 * (new_time - time)

        # --- 每步惩罚 + 死锁大惩罚 ---
        c_dead = 10000.
        if deadlock:
            r_dead = -c_dead
        else:
            r_dead = 0.0

        reward =  r_dead + int(r_time)
        terminated = bool(done or deadlock)
        if 0:
            print("obs", new_obs)
            print("reward", reward)

        out = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next,dtype=torch.bool),
            "time":torch.tensor([new_time], dtype=torch.int64),
            "phi_s": torch.tensor([phi_sp],dtype=torch.float32),
            "reward": torch.tensor([reward],dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
        }, batch_size=[])
        return out

    def _render(self):
        pass

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng