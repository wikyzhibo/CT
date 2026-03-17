"""
单设备单动作 TorchRL 环境封装。

训练时请使用 device="cpu"：环境内部为 NumPy/CPU，_step 返回的 TensorDict 均在 CPU，
由 train_single 在 CPU 上采集 rollout 后再 .to(device) 送入 GPU 计算。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import ClusterTool
from pathlib import Path

_TRUE_T = torch.tensor(True)
_FALSE_T = torch.tensor(False)


class Env_PN_Single(EnvBase):
    """
    单设备 Petri 网 RL 环境。训练时应传入 device="cpu"（与 train_single 约定一致）。
    """
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False

    def __init__(
        self,
        device: str = "cpu",
        seed=None,
        eval_mode: bool = False,
        device_mode: str = "single",
        reward_config: Optional[Dict[str, int]] = None,
        robot_capacity: int = 1,
        route_code: Optional[int] = None,
        process_time_map: Optional[Dict[str, int]] = None,
        proc_time_rand_enabled: Optional[bool] = None,
        proc_time_rand_scale_map: Optional[Dict[str, Dict[str, float]]] = None,
        detailed_reward: bool = False,  # 已弃用，保留以兼容旧调用，不再使用
    ):
        super().__init__(device=device)
        self.eval_mode = eval_mode

        dir = Path(__file__).parents[2] / "data" / "petri_configs"
        mode_name = str(device_mode).lower()
        if mode_name not in {"single", "cascade"}:
            mode_name = "single"
        path = dir / ("cascade.json" if mode_name == "cascade" else "single.json")
        config = PetriEnvConfig().load(path=path)
        if reward_config:
            config.reward_config.update(reward_config)
        config.device_mode = str(device_mode).lower()
        config.single_robot_capacity = 2 if int(robot_capacity) == 2 else 1
        if route_code is not None:
            config.route_code = int(route_code)
        if process_time_map is not None:
            config.process_time_map = {
                str(chamber): int(value) for chamber, value in dict(process_time_map).items()
            }
        if proc_time_rand_enabled is not None:
            config.proc_rand_enabled = bool(proc_time_rand_enabled)
        if proc_time_rand_scale_map is not None:
            config.proc_time_rand_scale_map = {
                str(chamber): {"min": float(bounds.get("min", 1.0)), "max": float(bounds.get("max", 1.0))}
                for chamber, bounds in dict(proc_time_rand_scale_map).items()
            }

        self.net = ClusterTool(config=config)
        # 与 pn_single 保持同一份 wait 档位来源，避免 env/net 两处规则漂移。
        self.wait_durations = list(getattr(self.net, "wait_durations", [5]))
        self.action_catalog = self._build_action_catalog()
        self.n_actions = len(self.action_catalog)
        self.wait_action_start = int(self.net.T)
        self.wait_action_indices = list(range(self.wait_action_start, self.n_actions))
        self.n_wafer = config.n_wafer
        self._make_spec()
        self._last_action_enable_info: dict = {}
        self._last_reward_detail: dict = {}
        self._out_time = torch.zeros(1, dtype=torch.int64)
        self._out_reward = torch.zeros(1, dtype=torch.float32)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def _normalize_wait_durations(durations) -> List[int]:
        values: List[int] = []
        for raw in list(durations or []):
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        if not values:
            values = [5]
        return sorted(set(values))

    def _build_action_catalog(self) -> List[Tuple[str, int]]:
        # 多档 wait 统一在动作目录里维护，避免在 if-else 中散落硬编码。
        catalog: List[Tuple[str, int]] = [("transition", int(t)) for t in range(int(self.net.T))]
        catalog.extend(("wait", int(duration)) for duration in self.wait_durations)
        return catalog

    def _decode_action(self, action: int) -> Tuple[str, int]:
        if action < 0 or action >= len(self.action_catalog):
            raise IndexError(f"invalid action index: {action}")
        return self.action_catalog[int(action)]

    def parse_wait_action(self, action: int) -> Optional[int]:
        kind, value = self._decode_action(action)
        if kind != "wait":
            return None
        return int(value)

    def get_action_name(self, action: int) -> str:
        kind, value = self._decode_action(action)
        if kind == "wait":
            return f"WAIT_{int(value)}s"
        return self.net.id2t_name[int(value)]

    def _make_spec(self):
        obs_dim = self.net.get_obs_dim()
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.float32, device=self.device),
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
            deadlock=Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_state_td(self, obs, action_mask, time):
        self._out_time[0] = time
        return TensorDict(
            {
                "observation": torch.from_numpy(obs),
                "action_mask": torch.from_numpy(action_mask),
                "time": self._out_time.clone(),
            }
        )

    def _mask(self):
        return self.net.get_action_mask(
            wait_action_start=self.wait_action_start,
            n_actions=self.n_actions,
        )

    def _reset(self, td_params):
        self.net.reset()
        self._last_action_enable_info = {}
        self._last_reward_detail = {}
        if self.eval_mode:
            self.net.eval()
        else:
            self.net.train()
        return self._build_state_td(self.net.get_obs(), self._mask(), self.net.time)

    def _step(self, tensordict=None):
        action = int(tensordict["action"].item())
        if self.eval_mode:
            self._last_action_enable_info = self.net.get_enable_actions_with_reasons(
                wait_action_start=self.wait_action_start
            )
        wait_duration = self.parse_wait_action(action)
        use_detailed_reward = self.eval_mode
        if wait_duration is not None:
            done, reward_result, scrap, action_mask, obs = self.net.step(
                wait_duration=int(wait_duration), detailed_reward=use_detailed_reward
            )
        else:
            _, transition_idx = self._decode_action(action)
            done, reward_result, scrap, action_mask, obs = self.net.step(
                a1=int(transition_idx), detailed_reward=use_detailed_reward
            )
        deadlock = bool(getattr(self.net, "_last_deadlock", False))
        reward = float(reward_result) if not isinstance(reward_result, dict) else float(reward_result.get("total", 0.0))
        if self.eval_mode and isinstance(reward_result, dict):
            detail = {}
            for k, v in reward_result.items():
                if isinstance(v, (int, float)):
                    detail[k] = float(v)
                elif isinstance(v, dict) and k == "scrap_info":
                    detail[k] = v
            self._last_reward_detail = detail

        self._out_time[0] = self.net.time
        self._out_reward[0] = reward
        out = TensorDict(
            {
                "observation": torch.from_numpy(obs),
                "action_mask": torch.from_numpy(action_mask),
                "time": self._out_time.clone(),
                "finish": _TRUE_T if (done and not scrap and not deadlock) else _FALSE_T,
                "scrap": _TRUE_T if scrap else _FALSE_T,
                "deadlock": _TRUE_T if deadlock else _FALSE_T,
                "reward": self._out_reward.clone(),
                "terminated": _TRUE_T if done else _FALSE_T,
            },
            batch_size=[],
        )
        return out

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng
