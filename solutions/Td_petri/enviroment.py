from torchrl.data import Bounded, Unbounded, Categorical, Composite,Binary
from torchrl.envs import EnvBase
import torch
from tensordict import TensorDict
from solutions.Td_petri.tdpn import TimedPetri

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
