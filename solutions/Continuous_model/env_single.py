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
    MAX_WAFERS = 4
    SCRAP_CLIP_THRESHOLD = 20.0

    def __init__(
        self,
        device: str = "cpu",
        seed=None,
        detailed_reward: bool = False,
        training_phase: int = 2,
        reward_config: Optional[Dict[str, int]] = None,
        robot_capacity: int = 1,
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
        self.n_actions = self.net.T + 1  # 最后一个是 WAIT
        self.n_wafer = config.n_wafer
        self.valid_pairs = self._build_valid_place_where_pairs()
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.valid_pairs)}
        self.pair_dim = len(self.valid_pairs)
        # present(1) + one-hot(pair) + status(4) + remaining_processing + time_to_scrap
        self.wafer_feat_dim = self.pair_dim + 7
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _build_valid_place_where_pairs(self) -> List[Tuple[int, int]]:
        """
        基于单设备工艺规则静态枚举合法 (place_idx, where)：
        LP(0) -> d_TM1(1) -> PM1(2) -> d_TM1(3) -> PM3/PM4(4) -> d_TM1(5) -> LP_done(6)
        """
        place_idx = {name: idx for idx, name in enumerate(self.net.id2p_name)}
        pair_names = [
            ("LP", 0),
            ("d_TM1", 1),
            ("PM1", 2),
            ("d_TM1", 3),
            ("PM3", 4),
            ("PM4", 4),
            ("d_TM1", 5),
            ("LP_done", 6),
        ]
        valid_pairs: List[Tuple[int, int]] = []
        for name, where in pair_names:
            if name not in place_idx:
                continue
            valid_pairs.append((int(place_idx[name]), int(where)))
        return valid_pairs

    def _make_spec(self):
        cleaning_feat_dim = 6  # PM1, PM3, PM4 各 3 维：is_cleaning / clean_remaining_time_norm / remaining_runs_before_clean_norm
        obs_dim = int(self.MAX_WAFERS) * int(self.wafer_feat_dim) + int(cleaning_feat_dim)
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

    def _build_status_one_hot(self, place_type: int, processing_time: int, stay_time: int) -> List[float]:
        # status: processing / done_waiting_pick / moving / waiting
        status = [0.0, 0.0, 0.0, 0.0]
        if place_type == 1:
            if stay_time < max(0, int(processing_time)):
                status[0] = 1.0
            else:
                status[1] = 1.0
        elif place_type == 2:
            if stay_time > int(getattr(self.net, "D_Residual_time", 0)):
                status[3] = 1.0
            else:
                status[2] = 1.0
        return status

    def _build_obs_features(
        self, p_idx: int, where: int, place_type: int, processing_time: int, stay_time: int
    ) -> List[float]:
        one_hot = [0.0] * int(self.pair_dim)
        pair_idx = self.pair_to_idx.get((int(p_idx), int(where)))
        if pair_idx is not None:
            one_hot[int(pair_idx)] = 1.0

        # 剩余加工量归一化，0 表示可取（加工完成）
        if place_type == 1 and int(processing_time) > 0:
            remaining_processing = max(0.0, float(int(processing_time) - int(stay_time)))
            remaining_processing_norm = remaining_processing / float(int(processing_time))
        else:
            remaining_processing_norm = 0.0

        # 运输位按用户要求固定为 1；其余位置按“先裁剪再归一化”
        if place_type == 2:
            time_to_scrap_norm = 1.0
        else:
            if place_type == 1:
                raw_time_to_scrap = max(
                    0.0,
                    float(int(processing_time) + int(getattr(self.net, "P_Residual_time", 0)) - int(stay_time)),
                )
            else:
                raw_time_to_scrap = self.SCRAP_CLIP_THRESHOLD
            clipped = min(raw_time_to_scrap, self.SCRAP_CLIP_THRESHOLD)
            time_to_scrap_norm = clipped / self.SCRAP_CLIP_THRESHOLD

        status_one_hot = self._build_status_one_hot(place_type, processing_time, stay_time)
        return [1.0, *one_hot, *status_one_hot, float(remaining_processing_norm), float(time_to_scrap_norm)]

    def _build_obs(self):
        max_wafers = int(self.MAX_WAFERS)
        wafer_feat_dim = int(self.wafer_feat_dim)
        wafers = []
        for p_idx, place in enumerate(self.net.marks):
            if place.name in {"PM2", "PM5", "PM6"}:
                continue
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue
                stay_time = int(getattr(tok, "stay_time", 0))
                place_type = int(place.type)
                token_id = int(getattr(tok, "token_id", -1))
                where = int(getattr(tok, "where", 0))
                wafers.append((token_id, int(p_idx), where, stay_time, place_type, int(place.processing_time)))
        wafers.sort(key=lambda x: x[0])
        obs: List[float] = []
        for i in range(max_wafers):
            if i < len(wafers):
                _, p_idx, where, stay_time, place_type, processing_time = wafers[i]
                obs.extend(
                    self._build_obs_features(
                        p_idx=int(p_idx),
                        where=int(where),
                        place_type=int(place_type),
                        processing_time=int(processing_time),
                        stay_time=int(stay_time),
                    )
                )
            else:
                obs.extend([0.0] * wafer_feat_dim)
        clean_duration = 150.0
        clean_trigger_runs = 2.0
        for chamber_name in ("PM3", "PM4"):
            place = self.net._get_place(chamber_name)
            is_cleaning = 1.0 if bool(getattr(place, "is_cleaning", False)) else 0.0
            clean_remaining_time = max(0.0, float(getattr(place, "cleaning_remaining", 0)))
            clean_remaining_time_norm = float(np.clip(clean_remaining_time / max(50,clean_duration), 0.0, 1.0))

            processed_count = max(0.0, float(getattr(place, "processed_wafer_count", 0)))
            remaining_runs = clean_trigger_runs - processed_count
            remaining_runs_before_clean_norm = float(np.clip(remaining_runs / clean_trigger_runs, 0.0, 1.0))

            obs.extend([float(is_cleaning), clean_remaining_time_norm, remaining_runs_before_clean_norm])
        return np.array(obs, dtype=np.float32)

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


class Env_PN_Single_PlaceObs(Env_PN_Single):
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False
    SCRAP_CLIP_THRESHOLD = 20.0

    def _make_spec(self):
        # LP(1) + TM(4) + PM1/PM3/PM4(各9)
        obs_dim = 1 + 4 + 9 * 3
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
        if place_name in {"PM1", "PM3", "PM4"}:
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
        for pm_name in ("PM1", "PM3", "PM4"):
            obs.extend(self._extract_place_features(pm_name))
        return np.array(obs, dtype=np.float32)
