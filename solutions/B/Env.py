import numpy as np
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase
import torch

from .clustertool_config import ClusterToolCfg
from .core import ClusterTool


class Env(EnvBase):
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False

    def __init__(
        self,
        device: str = "cpu",
        seed: int | None = None,
        n_wafer: int | None = None,
        ttime: int | None = None,
        search_depth: int | None = None,
        candidate_k: int | None = None,
        takt_cycle: list[int] | None = None,
        clustertool_cfg: ClusterToolCfg | None = None,
    ) -> None:
        super().__init__(device=device)

        cfg = clustertool_cfg if clustertool_cfg is not None else ClusterToolCfg.load()

        self.candidate_k = int(candidate_k if candidate_k is not None else cfg.candidate_k)
        resolved_n_wafer = int(
            n_wafer if n_wafer is not None else int(cfg.n_wafer1) + int(cfg.n_wafer2)
        )
        resolved_ttime = int(ttime if ttime is not None else cfg.ttime)
        resolved_search_depth = int(search_depth if search_depth is not None else cfg.search_depth)
        resolved_takt_cycle = takt_cycle if takt_cycle is not None else cfg.takt_cycle
        self.net = ClusterTool()
        self._out_time = torch.zeros(1, dtype=torch.int64)
        self._out_reward = torch.zeros(1, dtype=torch.float32)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(int(seed))

    def _make_spec(self) -> None:
        obs_dim = int(self.net.obs_dim)
        n_act = int(self.candidate_k)
        self.observation_spec = Composite(
            observation_f=Unbounded(shape=(obs_dim,), dtype=torch.float32, device=self.device),
            action_mask=Binary(n=n_act, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=(),
        )
        self.action_spec = Categorical(n=n_act, shape=(1,), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool),
            finish=Unbounded(shape=(1,), dtype=torch.bool),
            scrap=Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_state_td(self, obs: np.ndarray, action_mask: np.ndarray, time_v: int) -> TensorDict:
        self._out_time[0] = int(time_v)
        return TensorDict(
            {
                "observation_f": torch.as_tensor(obs, dtype=torch.float32),
                "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                "time": self._out_time.clone(),
            },
            batch_size=[],
        )

    def _reset(self, td_params: TensorDict | None = None) -> TensorDict:
        self.net.reset()
        prepared = self.net.prepare_train_candidates()
        obs = self.net.get_obs()
        return self._build_state_td(
            obs=obs,
            action_mask=np.asarray(prepared["action_mask"], dtype=bool),
            time_v=int(self.net._cur_clock),
        )

    def _step(self, tensordict: TensorDict | None = None) -> TensorDict:
        action_idx = int(tensordict["action"].item())
        obs, reward, terminated, next_mask, info = self.net.step(action_idx=action_idx)
        finish = bool(info["finish"])
        scrap = bool(info["scrap"])
        done = bool(terminated) or finish or scrap

        self._out_time[0] = int(info["time"])
        self._out_reward[0] = float(reward)
        out = TensorDict(
            {
                "observation_f": torch.as_tensor(obs, dtype=torch.float32),
                "action_mask": torch.as_tensor(next_mask, dtype=torch.bool),
                "time": self._out_time.clone(),
                "finish": torch.tensor(finish, dtype=torch.bool),
                "scrap": torch.tensor(scrap, dtype=torch.bool),
                "reward": self._out_reward.clone(),
                "done": torch.tensor(done, dtype=torch.bool),
            },
            batch_size=[],
        )
        return out

    def _set_seed(self, seed: int | None) -> None:
        self.rng = torch.manual_seed(seed)
