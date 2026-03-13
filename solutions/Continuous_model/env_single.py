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
from solutions.Continuous_model.pn_single import ClusterTool
from pathlib import Path

class Env_PN_Single(EnvBase):
    metadata = {"render.modes": ["human", "rgb_array"], "reder_fps": 30}
    batch_locked = False
    SCRAP_CLIP_THRESHOLD = 20.0

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
        chamber_names = self._get_place_obs_pm_names()
        chamber_feature_dim = sum(self._get_place_obs_feature_dim(name) for name in chamber_names)
        # LP(1) + TM + 每个腔室特征（route 感知）
        # single: TM=8 维（4 维时间 + 4 维去向 one-hot）；cascade: TM=14 维（TM2 8 维 + TM3 6 维）
        is_cascade = getattr(self.net, "single_device_mode", "single") == "cascade"
        tm_dim = 14 if is_cascade else 8
        obs_dim = 1 + tm_dim + chamber_feature_dim
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
        # 观测腔室列表与 pn_single 路线配置保持同源，避免 route 变更后维度漂移。
        candidates = list(self.net.chambers)
        is_cascade = getattr(self.net, "single_device_mode", "single") == "cascade"
        if is_cascade and "LLC" not in candidates:
            candidates.append("LLC")
        observed: List[str] = []
        seen = set()
        for name in candidates:
            if not (name.startswith("PM") or name in {"LLC", "LLD"}):
                continue
            if name in seen:
                continue
            observed.append(name)
            seen.add(name)
        if not observed:
            observed = ["PM1", "PM3", "PM4"]
        return observed

    def _get_place_obs_feature_dim(self, place_name: str) -> int:
        if place_name in {"LLC", "LLD"}:
            return 4
        return 9

    def _extract_place_features(self, place_name: str) -> List[float]:
        if place_name == "TM":
            if getattr(self.net, "single_device_mode", "single") == "cascade":
                return self._extract_tm_cascade_features()
            return self._extract_tm_single_features()
        place = self.net._get_place(place_name)
        if place_name == "LP":
            remaining = float(len(place.tokens))
            denom = max(1.0, float(self.n_wafer))
            remaining_norm = float(np.clip(remaining / denom, 0.0, 1.0))
            return [remaining_norm]
        if place_name in {"LLC", "LLD"}:
            return self._extract_ll_features(place)
        if place_name.startswith("PM"):
            return self._extract_pm_features(place)
        return []

    def _extract_tm_features(self) -> List[float]:
        transport_names = [name for name in self.net.id2p_name if name.startswith("d_")]
        penalty_time = max(1.0, float(getattr(self.net, "D_Residual_time", 10)))
        if not transport_names:
            return [0.0, 0.0, 0.0, 1.0]

        has_wafer = False
        any_complete = 0.0
        any_over_long = 0.0
        max_stay_norm = 0.0
        min_distance_norm = 1.0

        for name in transport_names:
            place = self.net._get_place(name)
            if len(place.tokens) == 0:
                continue
            has_wafer = True
            dwell_time = max(1.0, float(getattr(place, "processing_time", self.net.T_transport)))
            stay_time = float(getattr(place.head(), "stay_time", 0))
            any_complete = max(any_complete, 1.0 if stay_time >= dwell_time else 0.0)
            any_over_long = max(any_over_long, 1.0 if stay_time > penalty_time else 0.0)
            tm_norm_denom = max(dwell_time, penalty_time) * 2.0
            max_stay_norm = max(max_stay_norm, float(np.clip(stay_time / tm_norm_denom, 0.0, 1.0)))
            distance_to_penalty = max(0.0, penalty_time - stay_time)
            min_distance_norm = min(
                min_distance_norm,
                float(np.clip(distance_to_penalty / penalty_time, 0.0, 1.0)),
            )

        if not has_wafer:
            return [0.0, 0.0, 0.0, 1.0]

        return [any_complete, any_over_long, max_stay_norm, min_distance_norm]

    def _extract_wafer_dest_onehot_tm2(self) -> List[float]:
        """d_TM2 队首晶圆去向 one-hot（4 类）：PM7/PM8 -> 0, LLC -> 1, PM9/PM10 -> 2, LP_done -> 3"""
        onehot = [0.0, 0.0, 0.0, 0.0]
        if "d_TM2" not in self.net.id2p_name:
            return onehot
        place = self.net._get_place("d_TM2")
        if len(place.tokens) == 0:
            return onehot
        target = getattr(place.head(), "_target_place", None)
        if target is None:
            return onehot
        if target in {"PM7", "PM8"}:
            onehot[0] = 1.0
        elif target == "LLC":
            onehot[1] = 1.0
        elif target in {"PM9", "PM10"}:
            onehot[2] = 1.0
        elif target == "LP_done":
            onehot[3] = 1.0
        return onehot

    def _extract_wafer_dest_onehot_tm3(self) -> List[float]:
        """d_TM3 队首晶圆去向 one-hot（2 类）：PM1/2/3/4 -> 0, LLD -> 1"""
        onehot = [0.0, 0.0]
        if "d_TM3" not in self.net.id2p_name:
            return onehot
        place = self.net._get_place("d_TM3")
        if len(place.tokens) == 0:
            return onehot
        target = getattr(place.head(), "_target_place", None)
        if target is None:
            return onehot
        if target in {"PM1", "PM2", "PM3", "PM4"}:
            onehot[0] = 1.0
        elif target == "LLD":
            onehot[1] = 1.0
        return onehot

    def _extract_wafer_dest_onehot_tm1(self) -> List[float]:
        """d_TM1 队首晶圆去向 one-hot（4 类）：PM1 -> 0, PM3/PM4 -> 1, PM6 -> 2, LP_done -> 3"""
        onehot = [0.0, 0.0, 0.0, 0.0]
        if "d_TM1" not in self.net.id2p_name:
            return onehot
        place = self.net._get_place("d_TM1")
        if len(place.tokens) == 0:
            return onehot
        target = getattr(place.head(), "_target_place", None)
        if target is None:
            return onehot
        if target == "PM1":
            onehot[0] = 1.0
        elif target in {"PM3", "PM4"}:
            onehot[1] = 1.0
        elif target == "PM6":
            onehot[2] = 1.0
        elif target == "LP_done":
            onehot[3] = 1.0
        return onehot

    def _extract_tm_features_single_place(self, place_name: str) -> List[float]:
        """单个运输位（d_TM2 或 d_TM3）的 4 维时间特征"""
        penalty_time = max(1.0, float(getattr(self.net, "D_Residual_time", 10)))
        if place_name not in self.net.id2p_name:
            return [0.0, 0.0, 0.0, 1.0]
        place = self.net._get_place(place_name)
        if len(place.tokens) == 0:
            return [0.0, 0.0, 0.0, 1.0]
        dwell_time = max(1.0, float(getattr(place, "processing_time", self.net.T_transport)))
        stay_time = float(getattr(place.head(), "stay_time", 0))
        any_complete = 1.0 if stay_time >= dwell_time else 0.0
        any_over_long = 1.0 if stay_time > penalty_time else 0.0
        tm_norm_denom = max(dwell_time, penalty_time) * 2.0
        max_stay_norm = float(np.clip(stay_time / tm_norm_denom, 0.0, 1.0))
        distance_to_penalty = max(0.0, penalty_time - stay_time)
        min_distance_norm = float(np.clip(distance_to_penalty / penalty_time, 0.0, 1.0))
        return [any_complete, any_over_long, max_stay_norm, min_distance_norm]

    def _extract_tm_cascade_features(self) -> List[float]:
        """cascade 模式：TM2 块（4+4）+ TM3 块（4+2）共 14 维"""
        # TM2: 4 维时间 + 4 维去向 one-hot
        tm2_time = self._extract_tm_features_single_place("d_TM2")
        tm2_onehot = self._extract_wafer_dest_onehot_tm2()
        # TM3: 4 维时间 + 2 维去向 one-hot
        tm3_time = self._extract_tm_features_single_place("d_TM3")
        tm3_onehot = self._extract_wafer_dest_onehot_tm3()
        return tm2_time + tm2_onehot + tm3_time + tm3_onehot

    def _extract_tm_single_features(self) -> List[float]:
        """single 模式：d_TM1 的 4 维时间 + 4 维去向 one-hot 共 8 维"""
        tm1_time = self._extract_tm_features_single_place("d_TM1")
        tm1_onehot = self._extract_wafer_dest_onehot_tm1()
        return tm1_time + tm1_onehot

    def _extract_pm_features(self, place) -> List[float]:
        has_wafer = len(place.tokens) > 0
        proc_time = max(1.0, float(getattr(place, "processing_time", 0)))
        p_residual = max(1.0, float(getattr(self.net, "P_Residual_time", 15)))
        clean_duration = self.net.cleaning_duration
        clean_trigger_runs = self.net.cleaning_trigger_wafers

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

    def _extract_ll_features(self, place) -> List[float]:
        has_wafer = len(place.tokens) > 0
        raw_proc_time = float(getattr(place, "processing_time", 0))
        proc_time = max(1.0, raw_proc_time)

        occupied = 1.0 if has_wafer else 0.0
        processing = 0.0
        done_waiting_pick = 0.0
        remaining_process_time_norm = 0.0

        if has_wafer:
            stay_time = float(getattr(place.head(), "stay_time", 0))
            if raw_proc_time <= 0.0:
                # LLC 是 0 秒驻留位，落片后可视为已完成待取。
                processing = 0.0
                done_waiting_pick = 1.0
                remaining_process_time_norm = 0.0
            else:
                processing = 1.0 if stay_time < proc_time else 0.0
                done_waiting_pick = 1.0 if stay_time >= proc_time else 0.0
                remaining_proc = max(0.0, proc_time - stay_time)
                remaining_process_time_norm = float(np.clip(remaining_proc / proc_time, 0.0, 1.0))

        return [
            occupied,
            processing,
            done_waiting_pick,
            remaining_process_time_norm,
        ]

    def _build_obs(self):
        obs: List[float] = []
        obs.extend(self._extract_place_features("LP"))
        obs.extend(self._extract_place_features("TM"))
        for pm_name in self._get_place_obs_pm_names():
            obs.extend(self._extract_place_features(pm_name))
        return np.array(obs, dtype=np.float32)

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
        return self._build_state_td(self._build_obs(), self._mask(), self.net.time)

    def _step(self, tensordict=None):
        action = int(tensordict["action"].item())
        if self.eval_mode:
            self._last_action_enable_info = self.net.get_enable_actions_with_reasons(
                wait_action_start=self.wait_action_start
            )
        wait_duration = self.parse_wait_action(action)
        use_detailed_reward = self.eval_mode
        if wait_duration is not None:
            done, reward_result, scrap, action_mask = self.net.step(
                wait_duration=int(wait_duration), detailed_reward=use_detailed_reward
            )
        else:
            _, transition_idx = self._decode_action(action)
            done, reward_result, scrap, action_mask = self.net.step(
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

        out = TensorDict(
            {
                "observation": torch.as_tensor(self._build_obs(), dtype=torch.float32),
                "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
                "time": torch.tensor([self.net.time], dtype=torch.int64),
                "finish": torch.tensor(bool(done and not scrap and not deadlock), dtype=torch.bool),
                "scrap": torch.tensor(bool(scrap), dtype=torch.bool),
                "deadlock": torch.tensor(bool(deadlock), dtype=torch.bool),
                "reward": torch.tensor([float(reward)], dtype=torch.float32),
                "terminated": torch.tensor(bool(done), dtype=torch.bool),
            },
            batch_size=[],
        )
        return out

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng
