"""
单设备单动作 TorchRL 环境封装。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import PetriSingleDevice
from pathlib import Path

class Env_PN_Single(EnvBase):
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False
    SCRAP_CLIP_THRESHOLD = 20.0

    def __init__(
        self,
        device: str = "cpu",
        seed=None,
        detailed_reward: bool = False,
        training_phase: int = 2,
        reward_config: Optional[Dict[str, int]] = None,
        robot_capacity: int = 1,
        route_code: Optional[int] = None,
        process_time_map: Optional[Dict[str, int]] = None,
        proc_time_rand_enabled: Optional[bool] = None,
        proc_time_rand_scale_map: Optional[Dict[str, Dict[str, float]]] = None,
        proc_time_rand_min_scale: Optional[float] = None,
        proc_time_rand_max_scale: Optional[float] = None,
    ):
        super().__init__(device=device)
        self.training_phase = training_phase
        self.detailed_reward = detailed_reward

        dir = Path(__file__).parents[2] / "data" / "petri_configs"
        path = dir / "single.json"
        config = PetriEnvConfig().load(path=path)
        if reward_config:
            config.reward_config.update(reward_config)
        config.single_robot_capacity = 2 if int(robot_capacity) == 2 else 1
        if route_code is not None:
            config.single_route_code = int(route_code)
        if process_time_map is not None:
            config.single_process_time_map = {
                str(chamber): int(value) for chamber, value in dict(process_time_map).items()
            }
        if proc_time_rand_enabled is not None:
            config.single_proc_time_rand_enabled = bool(proc_time_rand_enabled)
        if proc_time_rand_scale_map is not None:
            config.single_proc_time_rand_scale_map = {
                str(chamber): {"min": float(bounds.get("min", 1.0)), "max": float(bounds.get("max", 1.0))}
                for chamber, bounds in dict(proc_time_rand_scale_map).items()
            }
        if proc_time_rand_min_scale is not None:
            config.single_proc_time_rand_min_scale = float(proc_time_rand_min_scale)
        if proc_time_rand_max_scale is not None:
            config.single_proc_time_rand_max_scale = float(proc_time_rand_max_scale)

        self.net = PetriSingleDevice(config=config)
        # 与 pn_single 保持同一份 wait 档位来源，避免 env/net 两处规则漂移。
        self.wait_durations = list(getattr(self.net, "wait_durations", [5]))
        self.action_catalog = self._build_action_catalog()
        self.n_actions = len(self.action_catalog)
        self.wait_action_start = int(self.net.T)
        self.wait_action_indices = list(range(self.wait_action_start, self.n_actions))
        self.n_wafer = config.n_wafer
        self._make_spec()
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
        pm_names = self._get_place_obs_pm_names()
        # LP(1) + TM(4) + 每个 PM 的 9 维特征（route 感知）
        obs_dim = 1 + 4 + 9 * len(pm_names)
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
        return TensorDict(
            {
                "observation": torch.as_tensor(obs, dtype=torch.float32),
                "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                "time": torch.tensor([time], dtype=torch.int64),
            }
        )

    def _get_place_obs_pm_names(self) -> List[str]:
        # 观测 PM 列表与 pn_single 路线配置保持同源，避免 route 变更后维度漂移。
        candidates = tuple(getattr(self.net, "_single_process_chambers", ("PM1", "PM3", "PM4")))
        observed = [name for name in candidates if name in {"PM1", "PM3", "PM4", "PM6"}]
        if not observed:
            observed = ["PM1", "PM3", "PM4"]
        return observed

    def _extract_place_features(self, place_name: str) -> List[float]:
        net_place_name = "d_TM1" if place_name == "TM" else place_name
        place = self.net._get_place(net_place_name)
        if place_name == "LP":
            remaining = float(len(place.tokens))
            denom = max(1.0, float(self.n_wafer))
            remaining_norm = float(np.clip(remaining / denom, 0.0, 1.0))
            return [remaining_norm]
        if place_name == "TM":
            return self._extract_tm_features(place)
        if place_name in {"PM1", "PM3", "PM4", "PM6"}:
            return self._extract_pm_features(place)
        return []

    def _extract_tm_features(self, place) -> List[float]:
        has_wafer = len(place.tokens) > 0
        dwell_time = max(1.0, float(getattr(place, "processing_time", self.net.T_transport)))
        penalty_time = max(1.0, float(getattr(self.net, "D_Residual_time", 10)))

        if not has_wafer:
            return [0.0, 0.0, 0.0, 1.0]

        stay_time = float(getattr(place.head(), "stay_time", 0))
        transport_complete = 1.0 if stay_time >= dwell_time else 0.0
        wafer_stay_over_long = 1.0 if stay_time > penalty_time else 0.0

        tm_norm_denom = max(dwell_time, penalty_time) * 2.0
        wafer_stay_time_norm = float(np.clip(stay_time / tm_norm_denom, 0.0, 1.0))
        distance_to_penalty = max(0.0, penalty_time - stay_time)
        distance_to_penalty_norm = float(np.clip(distance_to_penalty / penalty_time, 0.0, 1.0))
        return [
            transport_complete,
            wafer_stay_over_long,
            wafer_stay_time_norm,
            distance_to_penalty_norm,
        ]

    def _extract_pm_features(self, place) -> List[float]:
        has_wafer = len(place.tokens) > 0
        proc_time = max(1.0, float(getattr(place, "processing_time", 0)))
        p_residual = max(1.0, float(getattr(self.net, "P_Residual_time", 15)))
        clean_duration = max(1.0, float(getattr(self.net, "single_cleaning_duration", 150)))
        clean_trigger_runs = max(1.0, float(getattr(self.net, "single_cleaning_trigger_wafers", 2)))

        occupied = 1.0 if has_wafer else 0.0
        processing = 0.0
        done_waiting_pick = 0.0
        remaining_process_time_norm = 0.0
        wafer_stay_time_norm = 0.0
        wafer_time_to_scrap_norm = 0.0

        if has_wafer:
            stay_time = float(getattr(place.head(), "stay_time", 0))
            processing = 1.0 if stay_time < proc_time else 0.0
            done_waiting_pick = 1.0 if stay_time >= proc_time else 0.0
            remaining_proc = max(0.0, proc_time - stay_time)
            remaining_process_time_norm = float(np.clip(remaining_proc / proc_time, 0.0, 1.0))
            wafer_stay_time_norm = float(np.clip(stay_time / proc_time, 0.0, 1.0))
            time_to_scrap = max(0.0, proc_time + p_residual - stay_time)
            wafer_time_to_scrap_norm = float(
                np.clip(time_to_scrap, 0.0, self.SCRAP_CLIP_THRESHOLD) / self.SCRAP_CLIP_THRESHOLD
            )

        is_cleaning = 1.0 if bool(getattr(place, "is_cleaning", False)) else 0.0
        clean_remaining = max(0.0, float(getattr(place, "cleaning_remaining", 0)))
        clean_remaining_time_norm = float(np.clip(clean_remaining / clean_duration, 0.0, 1.0))

        processed_count = max(0.0, float(getattr(place, "processed_wafer_count", 0)))
        remaining_runs = max(0.0, clean_trigger_runs - processed_count)
        remaining_runs_before_clean_norm = float(np.clip(remaining_runs / clean_trigger_runs, 0.0, 1.0))

        return [
            occupied,
            processing,
            done_waiting_pick,
            remaining_process_time_norm,
            wafer_stay_time_norm,
            wafer_time_to_scrap_norm,
            is_cleaning,
            clean_remaining_time_norm,
            remaining_runs_before_clean_norm,
        ]

    def _build_obs(self):
        obs: List[float] = []
        obs.extend(self._extract_place_features("LP"))
        obs.extend(self._extract_place_features("TM"))
        for pm_name in self._get_place_obs_pm_names():
            obs.extend(self._extract_place_features(pm_name))
        return np.array(obs, dtype=np.float32)

    def get_enable_t(self) -> List[int]:
        return [
            int(a)
            for a in self.net.get_enable_actions(wait_action_start=self.wait_action_start)
        ]

    def _mask(self):
        return self.net.get_action_mask(
            wait_action_start=self.wait_action_start,
            n_actions=self.n_actions,
        )

    def _reset(self, td_params):
        self.net.reset()
        return self._build_state_td(self._build_obs(), self._mask(), self.net.time)

    def _step(self, tensordict=None):
        action = int(tensordict["action"].item())
        wait_duration = self.parse_wait_action(action)
        if wait_duration is not None:
            done, reward_result, scrap = self.net.step(detailed_reward=self.detailed_reward,wait_duration=int(wait_duration))
        else:
            _, transition_idx = self._decode_action(action)
            done, reward_result, scrap = self.net.step(a1=int(transition_idx), detailed_reward=self.detailed_reward)
        deadlock = bool(getattr(self.net, "_last_deadlock", False))
        reward = reward_result.get("total", 0.0) if isinstance(reward_result, dict) else float(reward_result)
        return TensorDict(
            {
                "observation": torch.as_tensor(self._build_obs(), dtype=torch.float32),
                "action_mask": torch.as_tensor(self._mask(), dtype=torch.bool),
                "time": torch.tensor([self.net.time], dtype=torch.int64),
                "finish": torch.tensor(bool(done and not scrap and not deadlock), dtype=torch.bool),
                "scrap": torch.tensor(bool(scrap), dtype=torch.bool),
                "deadlock": torch.tensor(bool(deadlock), dtype=torch.bool),
                "reward": torch.tensor([float(reward)], dtype=torch.float32),
                "terminated": torch.tensor(bool(done), dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng
