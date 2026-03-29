"""
级联设备单动作 TorchRL 环境封装（cascade-only）。

训练时请使用 device="cpu"：环境内部为 NumPy/CPU，_step 返回的 TensorDict 均在 CPU，
由 train_single 在 CPU 上采集 rollout 后再 .to(device) 送入 GPU 计算。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from config.cluster_tool.env_config import PetriEnvConfig
from solutions.A.petri_net import ClusterTool
from solutions.A.deprecated.pn import Petri
from pathlib import Path

_TRUE_T = torch.tensor(True)
_FALSE_T = torch.tensor(False)

try:
    from numba import njit
except Exception:  # pragma: no cover - numba 可选依赖
    njit = None


def _step_core_numpy(
    action: int,
    wait_action_start: int,
    wait_durations: np.ndarray,
) -> Tuple[bool, int, int]:
    """
    纯 numpy 数值路径：解析动作类型与参数。
    返回: (is_wait, transition_idx, wait_duration)
    """
    a = int(action)
    start = int(wait_action_start)
    if a >= start:
        idx = a - start
        if idx < 0 or idx >= int(wait_durations.shape[0]):
            return True, -1, int(wait_durations[0]) if wait_durations.size > 0 else 5
        return True, -1, int(wait_durations[idx])
    return False, a, 0


if njit is not None:
    @njit(cache=True)
    def step_core_numba(
        action: int,
        wait_action_start: int,
        wait_durations: np.ndarray,
    ) -> Tuple[np.bool_, np.int64, np.int64]:
        a = int(action)
        start = int(wait_action_start)
        if a >= start:
            idx = a - start
            if idx < 0 or idx >= wait_durations.shape[0]:
                fallback = 5
                if wait_durations.shape[0] > 0:
                    fallback = int(wait_durations[0])
                return True, -1, fallback
            return True, -1, int(wait_durations[idx])
        return False, a, 0
else:
    step_core_numba = _step_core_numpy


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
        device_mode: str = "cascade",
        robot_capacity: int = 1,
        route_code: Optional[int] = None,
        single_route_config: Optional[Dict[str, Any]] = None,
        single_route_name: Optional[str] = None,
        process_time_map: Optional[Dict[str, int]] = None,
        detailed_reward: bool = False,  # 已弃用，保留以兼容旧调用，不再使用
    ):
        super().__init__(device=device)
        self.eval_mode = eval_mode

        dir = Path(__file__).parents[2] / "config" / "cluster_tool"
        mode_name = str(device_mode).lower()
        if mode_name != "cascade":
            raise ValueError("Env_PN_Single now supports cascade mode only")
        path = dir / "cascade.yaml"
        config = PetriEnvConfig.load(path)
        config.device_mode = str(device_mode).lower()
        config.single_robot_capacity = 2 if int(robot_capacity) == 2 else 1
        if route_code is not None:
            config.route_code = int(route_code)
        if single_route_config is not None:
            config.single_route_config = dict(single_route_config)
        if single_route_name is not None:
            config.single_route_name = str(single_route_name)
        if process_time_map is not None:
            config.process_time_map = {
                str(chamber): int(value) for chamber, value in dict(process_time_map).items()
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
        self._last_reward_detail = {}
        if self.eval_mode:
            self.net.eval()
        else:
            self.net.train()
        return self._build_state_td(self.net.get_obs(), self._mask(), self.net.time)

    def _step(self, tensordict=None):
        action = int(tensordict["action"].item())
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


class Env_PN_Concurrent(EnvBase):
    """
    双机械手并发动作环境（TM2/TM3 双动作）。
    WAIT 固定单档 5s。
    """

    metadata = {'render.modes': ['human', 'rgb_array'], "reder_fps": 30}
    batch_locked = False

    TM2_TRANSITION_NAMES = [
        "u_LP1_s1", "u_LP2_s1", "u_s1_s2", "u_s1_s5", "u_s4_s5", "u_s5_LP_done",
        "t_s1", "t_s2", "t_s5", "t_LP_done"
    ]
    TM3_TRANSITION_NAMES = [
        "u_s2_s3", "u_s3_s4", "t_s3", "t_s4"
    ]

    def __init__(self, device='cpu', seed=None, detailed_reward: bool = False):
        super().__init__(device=device)
        config = PetriEnvConfig()
        # 并发模式强约束：只保留 WAIT=5。
        config.wait_durations = [5]
        self.net = Petri(config=config)

        self._t_name_to_idx = {name: i for i, name in enumerate(self.net.id2t_name)}

        self.tm2_transition_indices = [self._t_name_to_idx[name] for name in self.TM2_TRANSITION_NAMES
                                       if name in self._t_name_to_idx]
        self.n_actions_tm2 = len(self.tm2_transition_indices) + 1
        self.tm2_wait_action = len(self.tm2_transition_indices)

        self.tm3_transition_indices = [self._t_name_to_idx[name] for name in self.TM3_TRANSITION_NAMES
                                       if name in self._t_name_to_idx]
        self.n_actions_tm3 = len(self.tm3_transition_indices) + 1
        self.tm3_wait_action = len(self.tm3_transition_indices)

        self.detailed_reward = detailed_reward
        self.n_wafer = config.n_wafer
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        obs_dim = 12 * 6
        self.observation_spec = Composite(
            observation=Unbounded(shape=(obs_dim,), dtype=torch.int64, device=self.device),
            action_mask_tm2=Binary(n=self.n_actions_tm2, dtype=torch.bool),
            action_mask_tm3=Binary(n=self.n_actions_tm3, dtype=torch.bool),
            time=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            shape=()
        )
        self.action_spec = Composite(
            action_tm2=Categorical(n=self.n_actions_tm2, shape=(1,), dtype=torch.int64),
            action_tm3=Categorical(n=self.n_actions_tm3, shape=(1,), dtype=torch.int64),
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.state_spec = Composite(shape=())
        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            finish=Unbounded(shape=(1,), dtype=torch.bool),
            scrap=Unbounded(shape=(1,), dtype=torch.bool),
        )

    def _build_obs(self):
        max_wafers = 12
        processing_wafers = {}
        lp1_wafers = []
        lp2_wafers = []

        for p_idx, place in enumerate(self.net.marks):
            if place.type == 4:
                continue
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue
                stay_time = int(tok.stay_time)
                color = getattr(tok, 'color', 0)
                if place.type == 1:
                    time_to_scrap = 20 - (stay_time - place.processing_time)
                elif place.type == 2:
                    time_to_scrap = 10 - stay_time
                else:
                    time_to_scrap = -1
                wafer_tuple = (tok.token_id, p_idx, place.type, stay_time, time_to_scrap, color)
                if place.type in (1, 2, 5):
                    processing_wafers[tok.token_id] = wafer_tuple
                elif place.type == 3:
                    if place.name == "LP1":
                        lp1_wafers.append(wafer_tuple)
                    elif place.name == "LP2":
                        lp2_wafers.append(wafer_tuple)

        selected_wafers = list(processing_wafers.values())
        if len(selected_wafers) < max_wafers and len(lp1_wafers) > 0:
            selected_wafers.append(lp1_wafers[0])
        if len(selected_wafers) < max_wafers and len(lp2_wafers) > 0:
            selected_wafers.append(lp2_wafers[0])
        selected_wafers.sort(key=lambda x: x[0])

        obs = []
        for i in range(max_wafers):
            if i < len(selected_wafers):
                obs.extend(selected_wafers[i])
            else:
                obs.extend([0, 0, 0, 0, 0, 0])
        return np.array(obs, dtype=np.int64)

    def _build_action_masks(self):
        tm2_enabled, tm3_enabled = self.net.get_enable_t()
        tm2_enabled_set = set(tm2_enabled)
        tm3_enabled_set = set(tm3_enabled)

        mask_tm2 = np.zeros(self.n_actions_tm2, dtype=bool)
        for i, t_idx in enumerate(self.tm2_transition_indices):
            mask_tm2[i] = (t_idx in tm2_enabled_set)
        mask_tm2[self.tm2_wait_action] = True

        mask_tm3 = np.zeros(self.n_actions_tm3, dtype=bool)
        for i, t_idx in enumerate(self.tm3_transition_indices):
            mask_tm3[i] = (t_idx in tm3_enabled_set)
        mask_tm3[self.tm3_wait_action] = True

        return mask_tm2, mask_tm3

    def _reset(self, td_params):
        self.net.reset()
        obs = self._build_obs()
        mask_tm2, mask_tm3 = self._build_action_masks()
        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask_tm2": torch.as_tensor(mask_tm2, dtype=torch.bool),
            "action_mask_tm3": torch.as_tensor(mask_tm3, dtype=torch.bool),
            "time": torch.tensor([self.net.time], dtype=torch.int64),
        })

    def _step(self, tensordict=None):
        action_tm2 = tensordict["action_tm2"].item()
        action_tm3 = tensordict["action_tm3"].item()

        a1 = None if action_tm2 == self.tm2_wait_action else self.tm2_transition_indices[action_tm2]
        a2 = None if action_tm3 == self.tm3_wait_action else self.tm3_transition_indices[action_tm3]

        done, reward_result, scrap = self.net.step(
            a1=a1, a2=a2,
            with_reward=True, detailed_reward=self.detailed_reward
        )

        reward = reward_result.get('total', 0) if isinstance(reward_result, dict) else reward_result
        obs = self._build_obs()
        mask_tm2, mask_tm3 = self._build_action_masks()
        time = self.net.time
        terminated = bool(done)
        finish = done and not scrap

        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask_tm2": torch.as_tensor(mask_tm2, dtype=torch.bool),
            "action_mask_tm3": torch.as_tensor(mask_tm3, dtype=torch.bool),
            "time": torch.tensor([time], dtype=torch.int64),
            "finish": torch.tensor(finish, dtype=torch.bool),
            "scrap": torch.tensor(scrap, dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
        }, batch_size=[])

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng


class FastEnvWrapper:
    """
    高性能 CPU rollout 接口适配器。
    统一输出:
      reset() -> (obs, info)
      step(action) -> (obs, reward, done, info)
    """

    def __init__(self, env: Any):
        self.env = env
        self._last_obs_np: Optional[np.ndarray] = None
        self._last_mask_np: Optional[np.ndarray] = None

        if hasattr(env, "wait_action_start"):
            self.wait_action_start = int(env.wait_action_start)
            self.wait_durations = np.asarray(getattr(env, "wait_durations", [5]), dtype=np.int64)
        elif hasattr(env, "net"):
            self.wait_action_start = int(getattr(env.net, "T", 0))
            self.wait_durations = np.asarray(getattr(env.net, "wait_durations", [5]), dtype=np.int64)
        else:
            self.wait_action_start = 0
            self.wait_durations = np.asarray([5], dtype=np.int64)

    def _as_numpy(self, x: Any, dtype: np.dtype | None = None) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(dtype, copy=False) if dtype is not None else x
        if torch.is_tensor(x):
            arr = x.detach().cpu().numpy()
            return arr.astype(dtype, copy=False) if dtype is not None else arr
        arr = np.asarray(x)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def _extract_obs_mask(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(state, TensorDict):
            obs = self._as_numpy(state["observation"], np.float32)
            mask = self._as_numpy(state["action_mask"], np.bool_)
            return obs, mask
        if isinstance(state, dict):
            obs = self._as_numpy(state["observation"], np.float32)
            mask = self._as_numpy(state["action_mask"], np.bool_)
            return obs, mask
        if isinstance(state, tuple) and len(state) >= 2:
            obs = self._as_numpy(state[0], np.float32)
            mask = self._as_numpy(state[1], np.bool_)
            return obs, mask
        raise TypeError(f"Unsupported reset/step return type: {type(state)}")

    def _fast_step_single_env(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if isinstance(self.env, Env_PN_Single):
            is_wait, transition_idx, wait_duration = step_core_numba(
                int(action),
                self.wait_action_start,
                self.wait_durations,
            )
            detailed = bool(getattr(self.env, "eval_mode", False))
            if bool(is_wait):
                done, reward_result, scrap, action_mask, obs = self.env.net.step(
                    wait_duration=int(wait_duration),
                    detailed_reward=detailed,
                )
            else:
                done, reward_result, scrap, action_mask, obs = self.env.net.step(
                    a1=int(transition_idx),
                    detailed_reward=detailed,
                )
            reward = float(reward_result) if not isinstance(reward_result, dict) else float(reward_result.get("total", 0.0))
            deadlock = bool(getattr(self.env.net, "_last_deadlock", False))
            last_scan = getattr(self.env.net, "_last_state_scan", None)
            scan_scrap = bool(last_scan.get("is_scrap", False)) if isinstance(last_scan, dict) else False
            scrap_flag = bool(scrap) or scan_scrap
            finish_flag = bool(done and not scrap_flag and not deadlock)
            terminated = bool(done) or scrap_flag or finish_flag
            info: Dict[str, Any] = {
                "action_mask": np.asarray(action_mask, dtype=np.bool_),
                "scrap": scrap_flag,
                "deadlock": deadlock,
                "finish": finish_flag,
                "terminated": terminated,
                "time": int(getattr(self.env.net, "time", 0)),
            }
            return np.asarray(obs, dtype=np.float32), reward, terminated, info

        # 兼容旧接口环境（TensorDict step）
        if isinstance(getattr(self, "_cached_td", None), TensorDict):
            td = self._cached_td.clone()
            td["action"] = torch.tensor(int(action), dtype=torch.int64)
            td_next = self.env.step(td)
            nxt = td_next["next"]
            obs = self._as_numpy(nxt["observation"], np.float32)
            reward = float(nxt["reward"].item() if torch.is_tensor(nxt["reward"]) else nxt["reward"])
            done = bool(nxt["terminated"].item() if torch.is_tensor(nxt["terminated"]) else nxt["terminated"])
            mask = self._as_numpy(nxt["action_mask"], np.bool_)
            info = {"action_mask": mask, "terminated": done}
            self._cached_td = nxt.clone()
            return obs, reward, done, info

        raise TypeError("Unsupported env type for FastEnvWrapper.step")

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if isinstance(self.env, Env_PN_Single):
            self.env.net.reset()
            obs = np.asarray(self.env.net.get_obs(), dtype=np.float32)
            mask = np.asarray(
                self.env.net.get_action_mask(
                    wait_action_start=int(self.wait_action_start),
                    n_actions=int(self.wait_action_start + len(self.wait_durations)),
                ),
                dtype=np.bool_,
            )
            self._last_obs_np = obs
            self._last_mask_np = mask
            return obs, {"action_mask": mask, "time": int(self.env.net.time)}

        td = self.env.reset()
        self._cached_td = td.clone() if isinstance(td, TensorDict) else td
        obs, mask = self._extract_obs_mask(td)
        self._last_obs_np = obs
        self._last_mask_np = mask
        return obs, {"action_mask": mask}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._fast_step_single_env(int(action))
        terminated = bool(done) or bool(info.get("scrap", False)) or bool(info.get("finish", False))
        self._last_obs_np = obs
        self._last_mask_np = np.asarray(info.get("action_mask"), dtype=np.bool_)
        if terminated:
            terminal_obs = np.asarray(obs, dtype=np.float32).copy()
            obs, reset_info = self.reset()
            info["terminal_observation"] = terminal_obs
            info["auto_reset"] = True
            info["terminated"] = True
            info["action_mask"] = reset_info["action_mask"]
        else:
            info["terminated"] = False
        return obs, reward, terminated, info


class FastEnvWrapper_Concurrent:
    """
    并发双动作环境高性能 CPU rollout 适配器。
    统一输出:
      reset() -> (obs, info)  info 含 action_mask_tm2, action_mask_tm3, time
      step(action_tm2, action_tm3) -> (obs, reward, done, info)
    终止时自动 reset，info 中掩码反映 reset 后状态。
    """

    def __init__(self, env: Env_PN_Concurrent):
        self.env = env

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.net.reset()
        obs = np.asarray(self.env._build_obs(), dtype=np.float32)
        mask_tm2, mask_tm3 = self.env._build_action_masks()
        return obs, {
            "action_mask_tm2": mask_tm2.astype(np.bool_),
            "action_mask_tm3": mask_tm3.astype(np.bool_),
            "time": int(self.env.net.time),
        }

    def step(self, action_tm2: int, action_tm3: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        env = self.env
        a1 = None if action_tm2 == env.tm2_wait_action else env.tm2_transition_indices[int(action_tm2)]
        a2 = None if action_tm3 == env.tm3_wait_action else env.tm3_transition_indices[int(action_tm3)]

        done, reward_result, scrap = env.net.step(a1=a1, a2=a2, with_reward=True, detailed_reward=False)
        reward = float(reward_result) if not isinstance(reward_result, dict) else float(reward_result.get("total", 0.0))
        scrap = bool(scrap)
        done = bool(done)
        terminated = done or scrap
        finish = done and not scrap

        obs = np.asarray(env._build_obs(), dtype=np.float32)
        mask_tm2_arr, mask_tm3_arr = env._build_action_masks()
        info: Dict[str, Any] = {
            "action_mask_tm2": mask_tm2_arr.astype(np.bool_),
            "action_mask_tm3": mask_tm3_arr.astype(np.bool_),
            "finish": finish,
            "scrap": scrap,
            "time": int(env.net.time),
        }

        if terminated:
            obs, reset_info = self.reset()
            info["action_mask_tm2"] = reset_info["action_mask_tm2"]
            info["action_mask_tm3"] = reset_info["action_mask_tm3"]

        return obs, reward, terminated, info


class VectorEnv_Concurrent:
    """
    并发双动作轻量级多环境并行容器（进程内，多实例）。
    """

    def __init__(self, env_fn: Callable[[], Any], n_envs: int):
        self.n_envs = int(n_envs)
        self.envs: List[FastEnvWrapper_Concurrent] = [FastEnvWrapper_Concurrent(env_fn()) for _ in range(self.n_envs)]
        obs0, info0 = self.envs[0].reset()
        self.obs_dim = int(np.asarray(obs0).shape[-1])
        self.n_tm2 = int(np.asarray(info0["action_mask_tm2"]).shape[-1])
        self.n_tm3 = int(np.asarray(info0["action_mask_tm3"]).shape[-1])
        self._obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        self._mask_tm2 = np.zeros((self.n_envs, self.n_tm2), dtype=np.bool_)
        self._mask_tm3 = np.zeros((self.n_envs, self.n_tm3), dtype=np.bool_)
        self._obs[0] = np.asarray(obs0, dtype=np.float32)
        self._mask_tm2[0] = np.asarray(info0["action_mask_tm2"], dtype=np.bool_)
        self._mask_tm3[0] = np.asarray(info0["action_mask_tm3"], dtype=np.bool_)
        for i in range(1, self.n_envs):
            obs_i, info_i = self.envs[i].reset()
            self._obs[i] = np.asarray(obs_i, dtype=np.float32)
            self._mask_tm2[i] = np.asarray(info_i["action_mask_tm2"], dtype=np.bool_)
            self._mask_tm3[i] = np.asarray(info_i["action_mask_tm3"], dtype=np.bool_)

    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            self._obs[i] = np.asarray(obs, dtype=np.float32)
            self._mask_tm2[i] = np.asarray(info["action_mask_tm2"], dtype=np.bool_)
            self._mask_tm3[i] = np.asarray(info["action_mask_tm3"], dtype=np.bool_)
        return self._obs.copy(), {
            "action_mask_tm2": self._mask_tm2.copy(),
            "action_mask_tm3": self._mask_tm3.copy(),
        }

    def step(
        self,
        actions_tm2: np.ndarray,
        actions_tm3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        acts_tm2 = np.asarray(actions_tm2, dtype=np.int64)
        acts_tm3 = np.asarray(actions_tm3, dtype=np.int64)
        rewards = np.zeros((self.n_envs,), dtype=np.float32)
        dones = np.zeros((self.n_envs,), dtype=np.bool_)
        finish = np.zeros((self.n_envs,), dtype=np.bool_)
        scrap = np.zeros((self.n_envs,), dtype=np.bool_)
        time_arr = np.zeros((self.n_envs,), dtype=np.int64)
        for i in range(self.n_envs):
            obs_i, rew_i, done_i, info_i = self.envs[i].step(int(acts_tm2[i]), int(acts_tm3[i]))
            self._obs[i] = np.asarray(obs_i, dtype=np.float32)
            self._mask_tm2[i] = np.asarray(info_i["action_mask_tm2"], dtype=np.bool_)
            self._mask_tm3[i] = np.asarray(info_i["action_mask_tm3"], dtype=np.bool_)
            rewards[i] = float(rew_i)
            dones[i] = bool(done_i)
            finish[i] = bool(info_i.get("finish", False))
            scrap[i] = bool(info_i.get("scrap", False))
            time_arr[i] = int(info_i.get("time", 0))
        return self._obs.copy(), rewards, dones, {
            "action_mask_tm2": self._mask_tm2.copy(),
            "action_mask_tm3": self._mask_tm3.copy(),
            "finish": finish,
            "scrap": scrap,
            "time": time_arr,
        }


class VectorEnv:
    """
    轻量级多环境并行容器（进程内，多实例）。
    """

    def __init__(self, env_fn: Callable[[], Any], n_envs: int):
        self.n_envs = int(n_envs)
        self.envs: List[FastEnvWrapper] = [FastEnvWrapper(env_fn()) for _ in range(self.n_envs)]
        obs0, info0 = self.envs[0].reset()
        self.obs_dim = int(np.asarray(obs0).shape[-1])
        self.action_dim = int(np.asarray(info0["action_mask"]).shape[-1])
        self._obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        self._mask = np.zeros((self.n_envs, self.action_dim), dtype=np.bool_)
        self._obs[0] = np.asarray(obs0, dtype=np.float32)
        self._mask[0] = np.asarray(info0["action_mask"], dtype=np.bool_)
        for i in range(1, self.n_envs):
            obs_i, info_i = self.envs[i].reset()
            self._obs[i] = np.asarray(obs_i, dtype=np.float32)
            self._mask[i] = np.asarray(info_i["action_mask"], dtype=np.bool_)

    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            self._obs[i] = np.asarray(obs, dtype=np.float32)
            self._mask[i] = np.asarray(info["action_mask"], dtype=np.bool_)
        return self._obs.copy(), {"action_mask": self._mask.copy()}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        acts = np.asarray(actions, dtype=np.int64)
        rewards = np.zeros((self.n_envs,), dtype=np.float32)
        dones = np.zeros((self.n_envs,), dtype=np.bool_)
        finish = np.zeros((self.n_envs,), dtype=np.bool_)
        scrap = np.zeros((self.n_envs,), dtype=np.bool_)
        deadlock = np.zeros((self.n_envs,), dtype=np.bool_)
        time_arr = np.zeros((self.n_envs,), dtype=np.int64)
        for i in range(self.n_envs):
            obs_i, rew_i, done_i, info_i = self.envs[i].step(int(acts[i]))
            self._obs[i] = np.asarray(obs_i, dtype=np.float32)
            self._mask[i] = np.asarray(info_i["action_mask"], dtype=np.bool_)
            rewards[i] = float(rew_i)
            dones[i] = bool(done_i)
            finish[i] = bool(info_i.get("finish", False))
            scrap[i] = bool(info_i.get("scrap", False))
            deadlock[i] = bool(info_i.get("deadlock", False))
            time_arr[i] = int(info_i.get("time", 0))
        return self._obs.copy(), rewards, dones, {
            "action_mask": self._mask.copy(),
            "finish": finish,
            "scrap": scrap,
            "deadlock": deadlock,
            "time": time_arr,
        }
