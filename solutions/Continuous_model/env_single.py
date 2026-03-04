"""
单设备单动作 TorchRL 环境封装。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import PetriSingleDevice


class Env_PN_Single(EnvBase):
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False

    def __init__(
        self,
        device: str = "cpu",
        seed=None,
        detailed_reward: bool = False,
        training_phase: int = 2,
        reward_config: Optional[Dict[str, int]] = None,
    ):
        super().__init__(device=device)
        self.training_phase = training_phase
        self.detailed_reward = detailed_reward

        config = PetriEnvConfig(training_phase=training_phase)
        if reward_config:
            config.reward_config.update(reward_config)

        self.net = PetriSingleDevice(config=config)
        self.n_actions = self.net.T + 1  # 最后一个是 WAIT
        self.n_wafer = config.n_wafer
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        obs_dim = 12 * 6
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.int64, device=self.device),
            action_mask=Binary(n=self.n_actions, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=(),
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
        return TensorDict(
            {
                "observation": torch.as_tensor(obs, dtype=torch.int64),
                "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                "time": torch.tensor([time], dtype=torch.int64),
            }
        )

    def _build_obs(self):
        max_wafers = 12
        wafers = []
        for p_idx, place in enumerate(self.net.marks):
            if place.name in {"PM2", "PM6"}:
                continue
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue
                stay_time = int(getattr(tok, "stay_time", 0))
                place_type = int(place.type)
                if place_type == 1:
                    time_to_scrap = place.processing_time + int(getattr(self.net, "P_Residual_time", 0)) - stay_time
                elif place_type == 2:
                    time_to_scrap = int(getattr(self.net, "D_Residual_time", 0)) - stay_time
                else:
                    time_to_scrap = -1
                wafers.append((tok.token_id, p_idx, place_type, stay_time, int(time_to_scrap), 0))
        wafers.sort(key=lambda x: x[0])
        obs = []
        for i in range(max_wafers):
            if i < len(wafers):
                obs.extend(wafers[i])
            else:
                obs.extend([0, 0, 0, 0, 0, 0])
        return np.array(obs, dtype=np.int64)

    def _mask(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        enabled = self.net.get_enable_t()
        mask[enabled] = True
        mask[self.net.T] = True
        return mask

    def _reset(self, td_params):
        self.net.reset()
        return self._build_state_td(self._build_obs(), self._mask(), self.net.time)

    def _step(self, tensordict=None):
        action = int(tensordict["action"].item())
        if action == self.net.T:
            done, reward_result, scrap = self.net.step(wait=True, with_reward=True, detailed_reward=self.detailed_reward)
        else:
            done, reward_result, scrap = self.net.step(t=action, with_reward=True, detailed_reward=self.detailed_reward)
        reward = reward_result.get("total", 0.0) if isinstance(reward_result, dict) else float(reward_result)
        return TensorDict(
            {
                "observation": torch.as_tensor(self._build_obs(), dtype=torch.int64),
                "action_mask": torch.as_tensor(self._mask(), dtype=torch.bool),
                "time": torch.tensor([self.net.time], dtype=torch.int64),
                "finish": torch.tensor(bool(done and not scrap), dtype=torch.bool),
                "scrap": torch.tensor(bool(scrap), dtype=torch.bool),
                "reward": torch.tensor([float(reward)], dtype=torch.float32),
                "terminated": torch.tensor(bool(done), dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng
