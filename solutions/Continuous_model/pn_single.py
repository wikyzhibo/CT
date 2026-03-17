"""
单设备 Petri 网（构网驱动、单机械手、单动作）。
执行链：construct_single -> _get_enable_t -> step -> calc_reward
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from solutions.Continuous_model.helper_function import (
    _normalize_wait_durations,
    _preprocess_process_time_map,
    _round_to_nearest_five,
)

import numpy as np

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.construct_single import (
    BUFFER_NAMES,
    ROUTE_SPECS,
    build_single_device_net,
    parse_route,
)
from solutions.Continuous_model.pn import Place
from solutions.Continuous_model.takt_cycle_analyzer import analyze_cycle

CHAMBER = 1
DELIVERY_ROBOT = 2
SOURCE = 3

# 动作不使能原因的人性化描述（用于 Markdown 报告）
REASON_DESC: Dict[str, str] = {
    "pre_color_mismatch": "前置颜色/路径约束不满足",
    "insufficient_tokens": "库所 token 不足",
    "capacity_exceeded": "容量超限",
    "locked_by_arm2_head": "双臂模式：来源被队首晶圆目标锁定",
    "no_receiving_target": "下游无可接收腔室",
    "wrong_destination": "与队首晶圆目标不一致",
    "process_not_ready": "腔室加工未完成",
    "target_cleaning": "目标腔室清洗中",
    "dwell_time_not_met": "运输位停留时间未满足",
    "has_ready_wafer_restrict_wait": "有待取晶圆，仅允许短等待",
    "takt_release_limit": "节拍限制：距上次发片间隔未达当前周期节拍",
}


class ClusterTool:
    def __init__(self, config: PetriEnvConfig = None) -> None:
        assert config is not None, "config must be provided"
        self.config = config

        self.MAX_TIME = config.MAX_TIME
        self.n_wafer = int(config.n_wafer)
        
        self.done_event_reward = int(config.done_event_reward)
        self.finish_event_reward = self.done_event_reward * 6
        self.scrap_event_penalty = int(config.scrap_event_penalty)
        self.idle_event_penalty = float(config.idle_event_penalty)
        self.release_event_penalty = float(config.release_event_penalty)
        
        self.warn_coef_penalty = int(config.warn_coef_penalty)
        self.processing_coef_reward = float(config.processing_coef_reward)
        self.transport_overtime_coef_penalty = float(config.transport_overtime_coef_penalty)
        self.time_coef_penalty = float(config.time_coef_penalty)
        
        self.P_Residual_time = int(config.P_Residual_time)
        self.D_Residual_time = int(config.D_Residual_time)
        self.T_transport = int(config.T_transport)
        self.T_load = int(config.T_load)
        
        self.stop_on_scrap = bool(config.stop_on_scrap)
        
        self.reward_config = dict(config.reward_config)
        # 临时执行策略：固定单臂，不启用双臂分支。
        self.robot_capacity = 1

        self.cleaning_enabled = bool(config.cleaning_enabled)
        self.cleaning_targets = set(config.cleaning_targets)
        self.cleaning_trigger_wafers = config.cleaning_trigger_wafers
        self.cleaning_duration = max(0, int(config.cleaning_duration))
        self._cleaning_trigger_map: Dict[str, int] = dict(
            config.cleaning_trigger_wafers_map
        ) if getattr(config, "cleaning_trigger_wafers_map", None) else {
            c: config.cleaning_trigger_wafers for c in config.cleaning_targets
        }
        self._cleaning_duration_map: Dict[str, int] = dict(
            config.cleaning_duration_map
        ) if getattr(config, "cleaning_duration_map", None) else {
            c: max(0, int(config.cleaning_duration)) for c in config.cleaning_targets
        }
        self.wait_durations = _normalize_wait_durations(config.wait_durations)
        
        self.device_mode = config.device_mode
        self.route_code = config.route_code
        self.single_device_mode = self.device_mode
        self.single_route_code = self.route_code

        route_key = (self.device_mode, self.route_code)
        stages = ROUTE_SPECS.get(route_key) or ROUTE_SPECS.get(
            (self.device_mode, 1 if self.device_mode == "cascade" else 0),
            ROUTE_SPECS[("single", 0)],
        )
        route_meta = parse_route(stages, BUFFER_NAMES)
        self._route_stages: List[List[str]] = list(stages)  # 用于节拍分析器
        self.chambers = tuple(route_meta["chambers"])
        self._timeline_chambers = tuple(route_meta["timeline_chambers"])
        self._u_targets = dict(route_meta["u_targets"])
        self._step_map = dict(route_meta["step_map"])
        self._release_station_aliases = dict(route_meta["release_station_aliases"])
        self._release_chain_by_u = dict(route_meta["release_chain_by_u"])
        self._system_entry_places = set(route_meta["system_entry_places"])
        self._ready_chambers = tuple(route_meta["chambers"])
        self._single_process_chambers = self.chambers

        self.proc_rand_enabled = bool(config.proc_rand_enabled)
        self._proc_rand_scale_map = dict(config.proc_time_rand_scale_map or {})
        raw = dict(config.process_time_map or {})
        self._base_proc_time_map = self._preprocess_process_time_map(raw)
        self._episode_proc_time_map: Dict[str, int] = {}

        obs_config = {
            "P_Residual_time": self.P_Residual_time,
            "D_Residual_time": self.D_Residual_time,
            "cleaning_duration": self.cleaning_duration,
            "cleaning_trigger_wafers": self.cleaning_trigger_wafers,
            "cleaning_duration_map": self._cleaning_duration_map,
            "cleaning_trigger_wafers_map": self._cleaning_trigger_map,
            "scrap_clip_threshold": 20.0,
        }
        info = build_single_device_net(
            n_wafer=self.n_wafer,
            ttime=max(1, self.T_transport),
            robot_capacity=self.robot_capacity,
            process_time_map=self._base_proc_time_map,
            route_code=self.route_code,
            device_mode=self.device_mode,
            obs_config=obs_config,
        )
        self.pre: np.ndarray = info["pre"]
        self.pre_color: np.ndarray = info.get("pre_color", self.pre[:, :, None])
        self.pst: np.ndarray = info["pst"]
        self.net: np.ndarray = self.pst - self.pre
        self.m0: np.ndarray = info["m0"]
        self.m: np.ndarray = self.m0.copy()
        self.md: np.ndarray = info["md"]
        self.ptime: np.ndarray = info["ptime"]
        self.k: np.ndarray = info["capacity"]
        self.id2p_name: List[str] = info["id2p_name"]
        self.id2t_name: List[str] = info["id2t_name"]
        self._t_route_code_map: Dict[str, int] = dict(info.get("t_route_code_map") or {})
        self._t_route_code_by_idx: List[int] = [
            int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name
        ]
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.ttime = int(np.max(info["ttime"])) if len(info["ttime"]) > 0 else 5

        # 预计算的 pre/pst 库所索引与运输位索引（构网返回或本地计算以兼容旧版）
        self._pre_place_indices: List[np.ndarray] = info.get("pre_place_indices")
        self._pst_place_indices: List[np.ndarray] = info.get("pst_place_indices")
        self._transport_pre_place_idx: List[int] = info.get("transport_pre_place_idx")
        if (
            self._pre_place_indices is None
            or self._pst_place_indices is None
            or self._transport_pre_place_idx is None
        ):
            self._pre_place_indices = [
                np.flatnonzero(self.pre[:, t] > 0) for t in range(self.T)
            ]
            self._pst_place_indices = [
                np.flatnonzero(self.pst[:, t] > 0) for t in range(self.T)
            ]
            self._transport_pre_place_idx = []
            for t in range(self.T):
                indices = self._pre_place_indices[t]
                found = next(
                    (
                        int(idx)
                        for idx in indices
                        if self.id2p_name[int(idx)].startswith("d_")
                    ),
                    -1,
                )
                self._transport_pre_place_idx.append(found)

        self.ori_marks: List[Place] = info["marks"]
        self.marks: List[Place] = self._clone_marks(self.ori_marks)
        # 临时执行策略：除 LP/LP_done 外全部按 unit-capacity 运行。
        for idx, p in enumerate(self.marks):
            if p.name not in {"LP", "LP_done"}:
                p.capacity = 1
                self.k[idx] = 1
        self._refresh_episode_proc_time()

        self.time = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 50
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log: List[Dict[str, Any]] = []
        self.enable_statistics = True
        self._per_wafer_reward = 0.0
        self._token_stats: Dict[int, Dict[str, Any]] = {}
        self._qtime_violated_tokens: Set[int] = set()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        self._cascade_round_robin_pairs: Dict[str, Tuple[str, ...]] = {}
        self._cascade_round_robin_next: Dict[str, str] = {}
        self._single_round_robin_pairs: Dict[str, Tuple[str, str]] = {}
        self._single_round_robin_next: Dict[str, str] = {}
        self._u_transition_by_source: Dict[str, int] = {}
        self._t_transitions_by_transport: Dict[str, List[int]] = {}
        self._token_pool: List[BasedToken] = []
        if self.device_mode == "cascade":
            self._cascade_round_robin_pairs["LP"] = ("PM7", "PM8")
            if self.route_code == 1:
                self._cascade_round_robin_pairs["LLC"] = ("PM1", "PM2", "PM3", "PM4")
            if self.route_code in {2, 3}:
                self._cascade_round_robin_pairs["LLC"] = ("PM1", "PM2")
            if self.route_code == 2:
                self._cascade_round_robin_pairs["LLD"] = ("PM9", "PM10")
            self._cascade_round_robin_next = {
                source: pair[0] for source, pair in self._cascade_round_robin_pairs.items()
            }
        else:
            for source, targets in self._u_targets.items():
                if len(targets) == 2:
                    pair = (str(targets[0]), str(targets[1]))
                    self._single_round_robin_pairs[source] = pair
            self._single_round_robin_next = {
                source: pair[0] for source, pair in self._single_round_robin_pairs.items()
            }
        self._last_deadlock = False
        self._chamber_timeline: Dict[str, list] = {name: [] for name in self._timeline_chambers}
        self._chamber_active: Dict[str, Dict[int, int]] = {name: {} for name in self._timeline_chambers}
        self._init_cleaning_state()
        self._takt_result: Optional[Dict[str, Any]] = self._compute_takt_result()
        self._last_u_LP_fire_time: int = 0
        self._u_LP_release_count: int = 0
        self._training = True
        self._step_profile = {
            "count": 0,
            "total_s": 0.0,
            "get_enable_t_s": 0.0,
            "fire_s": 0.0,
            "build_obs_s": 0.0,
            "advance_and_reward_s": 0.0,
            "next_event_delta_s": 0.0,
            "other_s": 0.0,
        }
        self._last_state_scan: Dict[str, Any] = {}
        self._place_by_name: Dict[str, Place] = {}
        self._obs_place_names: List[str] = []
        self._obs_places: List[Place] = []
        self._obs_offsets: List[int] = []
        self._obs_specs: List[Dict[str, Any]] = []
        self._obs_dim: int = 0
        self._obs_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._obs_return_copy: bool = True
        self._rebuild_place_cache()
        self._init_obs_cache()
        self._build_transition_index()
        self._rebuild_token_pool()

    def train(self):
        """训练模式"""
        self._training = True

    def eval(self):
        """评估模式"""
        self._training = False

    def step(self, a1=None, detailed_reward: bool = False, wait_duration: Optional[int] = None):
        """
        单设备一步推进入口。
        - 非 wait：按既有 ttime 发射变迁
        - wait：支持多档等待，并用关键事件截断，避免一次跨越多个决策点
        返回：(done, reward_result, scrap, action_mask, obs)
        """
        DONE = False
        SCRAPE = False
        step_start = perf_counter()
        advance_and_reward_s = 0.0
        fire_s = 0.0
        get_enable_t_s = 0.0
        build_obs_s = 0.0
        next_event_delta_s = 0.0

        self._last_deadlock = False
        # ======超出最大运行时间直接判定为超时结束，避免“原地等待”导致的无效步骤======
        if self.time >= self.MAX_TIME:
            timeout_reward = {"total": -100.0, "timeout": True} if detailed_reward else -100.0
            DONE = True
            SCRAPE = True
            t_mask = perf_counter()
            action_mask = self.get_action_mask(
                wait_action_start=int(self.T),
                n_actions=int(self.T + len(self.wait_durations)),
            )
            get_enable_t_s += perf_counter() - t_mask
            t_obs = perf_counter()
            obs = self.get_obs()
            build_obs_s += perf_counter() - t_obs
            self._record_step_profile(
                total_s=perf_counter() - step_start,
                get_enable_t_s=get_enable_t_s,
                fire_s=fire_s,
                build_obs_s=build_obs_s,
                advance_and_reward_s=advance_and_reward_s,
                next_event_delta_s=next_event_delta_s,
            )
            return DONE, timeout_reward, SCRAPE, action_mask, obs

        action = a1
        do_wait = (wait_duration is not None) or action is None
        scan_info: Dict[str, Any] = {}

        t1 = self.time

        # wait 动作逻辑
        if do_wait:
            requested_wait = int(wait_duration)
            assert requested_wait > 0, "wait_duration should be non-positive"
            lp_done_count = len(self._get_place("LP_done").tokens)
            episode_finished = lp_done_count >= self.n_wafer
            if episode_finished and requested_wait > 5:
                # 仅当 episode 全部完成（所有晶圆均在 LP_done）时才截断 wait，避免长时间空等。
                requested_wait = 5
                wait_reason = "episode_finished_cap_5s"
                actual_dt = requested_wait
            elif requested_wait == 5:
                # 最小 wait 固定推进 5s，不做事件截断。
                wait_reason = "fixed_5s"
                actual_dt = requested_wait
            else:
                t_next_event = perf_counter()
                next_event_delta = self.get_next_event_delta()
                next_event_delta_s += perf_counter() - t_next_event
                if next_event_delta is None:
                    # 没有可预见关键事件时，按请求时长推进，避免“原地空转”。
                    actual_dt = requested_wait
                    wait_reason = "no_future_event"
                else:
                    # 关键规则：wait 只推进到“下一个事件或请求时长”中更早者。
                    if int(next_event_delta) <= 0:
                        # next_event_delta 可能为 0（关键事件就在当前时刻），
                        # 此时若继续按 min() 会得到 0 并触发断言，导致 wait 分支崩溃。
                        actual_dt = min(requested_wait, 5)
                        wait_reason = "immediate_event_fallback_5s"
                    else:
                        actual_dt = min(requested_wait, next_event_delta)
                        wait_reason = "next_event_capped" if int(next_event_delta) < requested_wait else "requested_duration"

            assert  actual_dt > 0, f"actual_dt ({actual_dt}) should be positive"

            t2 = t1 + actual_dt
            t_ar = perf_counter()
            reward_result, scan_info = self._advance_and_compute_reward(
                actual_dt, t1, t2, detailed=detailed_reward,
            )
            advance_and_reward_s += perf_counter() - t_ar
            self._consecutive_wait_time += (t2 - t1)

            # 停滞惩罚
            if self._consecutive_wait_time >= self.idle_timeout and not self._idle_penalty_applied:
                self._idle_penalty_applied = True
                if detailed_reward:
                    reward_result["idle_timeout_penalty"] = -float(self.idle_event_penalty)
                    reward_result["total"] -= float(self.idle_event_penalty)
                else:
                    reward_result -= float(self.idle_event_penalty)
        else:
            self._consecutive_wait_time = 0

            t2 = t1 + self.ttime
            t_ar = perf_counter()
            reward_result, scan_info = self._advance_and_compute_reward(
                self.ttime, t1, t2, detailed=detailed_reward,
            )
            advance_and_reward_s += perf_counter() - t_ar
            t_fire = perf_counter()
            log_entry = self._fire(int(action), start_time=t1, end_time=t2)
            fire_s += perf_counter() - t_fire
            self.fire_log.append(log_entry)

            if self._per_wafer_reward > 0:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] += self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(self._get_place("LP_done").tokens) >= self.n_wafer
        # ====== 违反驻留时间约束（已在 _advance_and_compute_reward 中检测）======
        scan = scan_info if isinstance(scan_info, dict) else {}
        is_scrap = bool(scan.get("is_scrap", False))
        scrap_info = scan.get("scrap_info")
        if "is_scrap" not in scan:
            is_scrap, scrap_info = self._check_scrap()
        if is_scrap:
            self.scrap_count += 1
            self.resident_violation_count += 1
            if detailed_reward:
                reward_result["scrap_penalty"] -= float(self.scrap_event_penalty)
                reward_result["total"] -= float(self.scrap_event_penalty)
                reward_result["scrap_info"] = scrap_info
            else:
                reward_result -= float(self.scrap_event_penalty)
            if self.stop_on_scrap:
                DONE = True
                SCRAPE = True
                t_mask = perf_counter()
                action_mask = self.get_action_mask(
                    wait_action_start=int(self.T),
                    n_actions=int(self.T + len(self.wait_durations)),
                )
                get_enable_t_s += perf_counter() - t_mask
                t_obs = perf_counter()
                obs = self.get_obs()
                build_obs_s += perf_counter() - t_obs
                self._record_step_profile(
                    total_s=perf_counter() - step_start,
                    get_enable_t_s=get_enable_t_s,
                    fire_s=fire_s,
                    build_obs_s=build_obs_s,
                    advance_and_reward_s=advance_and_reward_s,
                    next_event_delta_s=next_event_delta_s,
                )
                return DONE, reward_result, SCRAPE, action_mask, obs

        # ===== 任务完成奖励 ======
        if finish:
            if detailed_reward:
                reward_result["finish_bonus"] += float(self.finish_event_reward)
                reward_result["total"] += float(self.finish_event_reward)
            else:
                reward_result += float(self.finish_event_reward)
            SCRAPE = False
        t_mask = perf_counter()
        action_mask = self.get_action_mask(
            wait_action_start=int(self.T),
            n_actions=int(self.T + len(self.wait_durations)),
        )
        get_enable_t_s += perf_counter() - t_mask
        t_obs = perf_counter()
        obs = self.get_obs()
        build_obs_s += perf_counter() - t_obs
        self._record_step_profile(
            total_s=perf_counter() - step_start,
            get_enable_t_s=get_enable_t_s,
            fire_s=fire_s,
            build_obs_s=build_obs_s,
            advance_and_reward_s=advance_and_reward_s,
            next_event_delta_s=next_event_delta_s,
        )
        return bool(finish), reward_result, SCRAPE, action_mask, obs

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self._refresh_episode_proc_time()
        self._rebuild_place_cache()
        self._init_obs_cache()
        self.time = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log.clear()
        self._per_wafer_reward = 0.0
        self._token_stats = {}
        self._qtime_violated_tokens.clear()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        if self.device_mode == "cascade":
            self._cascade_round_robin_next = {
                source: pair[0] for source, pair in self._cascade_round_robin_pairs.items()
            }
        else:
            self._single_round_robin_next = {
                source: pair[0] for source, pair in self._single_round_robin_pairs.items()
            }
        self._last_deadlock = False
        self._chamber_timeline = {name: [] for name in self._timeline_chambers}
        self._chamber_active = {name: {} for name in self._timeline_chambers}
        self._init_cleaning_state()
        self._takt_result = self._compute_takt_result()
        self._last_u_LP_fire_time = 0
        self._u_LP_release_count = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        for idx, p in enumerate(self.marks):
            if p.name not in {"LP", "LP_done"}:
                p.capacity = 1
                self.k[idx] = 1
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._rebuild_token_pool()
        self._last_state_scan = {}
        return None, self._get_enable_t()

    def _get_obs_place_order(self) -> List[str]:
        """返回观测顺序：LP + 运输位 + 腔室。"""
        tm_names = ["d_TM2", "d_TM3"] if self.device_mode == "cascade" else ["d_TM1"]
        tm_names = [n for n in tm_names if n in self.id2p_name]
        candidates = list(self.chambers)
        if self.device_mode == "cascade" and "LLC" not in candidates:
            candidates.append("LLC")
        chambers: List[str] = []
        seen: Set[str] = set()
        for name in candidates:
            if not (name.startswith("PM") or name in {"LLC", "LLD"}):
                continue
            if name in seen:
                continue
            chambers.append(name)
            seen.add(name)
        if not chambers:
            chambers = ["PM1", "PM3", "PM4"]
        return ["LP"] + tm_names + chambers

    def _rebuild_place_cache(self) -> None:
        self._place_by_name = {p.name: p for p in self.marks}

    def _init_obs_cache(self) -> None:
        order = self._get_obs_place_order()
        obs_names = [name for name in order if name in self._place_by_name]
        obs_places = [self._place_by_name[name] for name in obs_names]
        offsets: List[int] = []
        specs: List[Dict[str, Any]] = []
        cursor = 0
        for place in obs_places:
            dim = int(place.get_obs_dim())
            offsets.append(cursor)
            specs.append({"name": place.name, "offset": cursor, "dim": dim})
            cursor += dim
        self._obs_place_names = obs_names
        self._obs_places = obs_places
        self._obs_offsets = offsets
        self._obs_specs = specs
        self._obs_dim = int(cursor)
        self._obs_buffer = np.zeros(self._obs_dim, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        if self._obs_dim == 0:
            return np.zeros(0, dtype=np.float32)
        buffer = self._obs_buffer
        for place, offset in zip(self._obs_places, self._obs_offsets):
            place.write_obs(buffer, offset)
        if self._obs_return_copy:
            return buffer.copy()
        return buffer

    def get_obs_dim(self) -> int:
        return int(self._obs_dim)

    def _record_step_profile(
        self,
        total_s: float,
        get_enable_t_s: float,
        fire_s: float,
        build_obs_s: float,
        advance_and_reward_s: float,
        next_event_delta_s: float = 0.0,
    ) -> None:
        tracked = (
            get_enable_t_s
            + fire_s
            + build_obs_s
            + advance_and_reward_s
            + next_event_delta_s
        )
        other_s = max(0.0, float(total_s) - float(tracked))
        self._step_profile["count"] += 1
        self._step_profile["total_s"] += float(total_s)
        self._step_profile["get_enable_t_s"] += float(get_enable_t_s)
        self._step_profile["fire_s"] += float(fire_s)
        self._step_profile["build_obs_s"] += float(build_obs_s)
        self._step_profile["advance_and_reward_s"] += float(advance_and_reward_s)
        self._step_profile["next_event_delta_s"] += float(next_event_delta_s)
        self._step_profile["other_s"] += float(other_s)

    def get_step_profile_summary(self) -> Dict[str, Any]:
        count = int(self._step_profile.get("count", 0))
        total_s = float(self._step_profile.get("total_s", 0.0))
        summary: Dict[str, Any] = {
            "count": count,
            "total_ms": total_s * 1000.0,
            "avg_ms": (total_s * 1000.0 / count) if count > 0 else 0.0,
            "steps_per_sec": (count / total_s) if total_s > 0 else 0.0,
            "segments": {},
        }
        keys = (
            "get_enable_t",
            "fire",
            "build_obs",
            "advance_and_reward",
            "next_event_delta",
            "other",
        )
        for key in keys:
            segment_total_s = float(self._step_profile.get(f"{key}_s", 0.0))
            ratio_pct = (segment_total_s / total_s * 100.0) if total_s > 0 else 0.0
            summary["segments"][key] = {
                "total_ms": segment_total_s * 1000.0,
                "avg_ms": (segment_total_s * 1000.0 / count) if count > 0 else 0.0,
                "ratio_pct": ratio_pct,
            }
        return summary

    def calc_reward(self, t1: int, t2: int, detailed: bool = False):
        dt = max(0, t2 - t1)
        parts = {
            "time_cost": 0.0,
            "proc_reward": 0.0,
            "safe_reward": 0.0,
            "warn_penalty": 0.0,
            "penalty": 0.0,
            "wafer_done_bonus": 0.0,
            "finish_bonus": 0.0,
            "scrap_penalty": 0.0,
        }

        # 时间惩罚：每步按 dt 线性惩罚，鼓励更快完成
        if self.reward_config.get("time_cost", 1):
            parts["time_cost"] = -float(dt * self.time_coef_penalty)

        # 加工奖励：每个加工位上每个晶圆根据加工进度给予奖励
        if self.reward_config.get("proc_reward", 1):
            for p in self.marks:
                if p.type != CHAMBER or len(p.tokens) == 0 or p.processing_time <= 0:
                    continue
                remain = max(0, p.processing_time - int(p.head().stay_time))
                progress = min(dt, remain)
                parts["proc_reward"] += self.processing_coef_reward * float(progress)

        # 与 pn.py 对齐：运输位(type=2, 单设备主要是 d_TM1)超过 D_Residual_time 后按超时秒数惩罚
        if self.reward_config.get("transport_penalty", 1):
            for p in self.marks:
                if p.type != DELIVERY_ROBOT or len(p.tokens) == 0:
                    continue
                for tok in p.tokens:
                    deadline = int(tok.enter_time) + int(self.D_Residual_time)
                    over_start = max(int(t1), deadline)
                    if int(t2) > over_start:
                        parts["penalty"] -= float(int(t2) - over_start) * float(self.transport_overtime_coef_penalty)

        # 驻留警告惩罚
        for p in self.marks:
            if p.type != CHAMBER or len(p.tokens) == 0:
                continue
            left = p.processing_time + self.P_Residual_time - p.head().stay_time
            if self.reward_config.get("warn_penalty", 1) and left <= self.P_Residual_time:
                parts["warn_penalty"] -= int(self.warn_coef_penalty) * dt

        total = sum(parts.values())
        if detailed:
            parts["total"] = float(total)
            return parts
        return float(total)

    def _advance_and_compute_reward(
        self,
        dt: int,
        t1: int,
        t2: int,
        detailed: bool = False,
    ) -> tuple:
        """
        单次 marks 遍历完成：reward 计算 + stay_time 推进 + 清洗/idle 推进 + scrap/qtime 检测。
        排序：reward 使用推进前 stay_time，scrap/qtime 使用推进后 stay_time。
        返回 (reward_result, scan_info)。
        """
        safe_dt = max(0, int(dt))
        self.time += safe_dt

        parts = {
            "time_cost": 0.0,
            "proc_reward": 0.0,
            "safe_reward": 0.0,
            "warn_penalty": 0.0,
            "penalty": 0.0,
            "wafer_done_bonus": 0.0,
            "finish_bonus": 0.0,
            "scrap_penalty": 0.0,
        }

        do_time_cost = bool(self.reward_config.get("time_cost", 1))
        do_proc_reward = bool(self.reward_config.get("proc_reward", 1))
        do_transport_penalty = bool(self.reward_config.get("transport_penalty", 1))
        do_warn_penalty = bool(self.reward_config.get("warn_penalty", 1))

        if do_time_cost:
            parts["time_cost"] = -float(safe_dt * self.time_coef_penalty)

        is_scrap = False
        scrap_info: Optional[Dict[str, Any]] = None
        qtime_new_violations: List[int] = []
        qtime_limit = int(self.D_Residual_time)
        p_residual = int(self.P_Residual_time)
        proc_coef = self.processing_coef_reward
        transport_coef = self.transport_overtime_coef_penalty
        warn_coef = int(self.warn_coef_penalty)
        check_qtime = not self._training

        for p in self.marks:
            p_type = p.type
            has_tok = len(p.tokens) > 0

            # --- A: reward (PRE-advance stay_time) ---
            if has_tok and p_type == CHAMBER:
                head = p.tokens[0]
                head_stay = int(head.stay_time)
                proc_time = int(p.processing_time)
                if do_proc_reward and proc_time > 0:
                    remain = proc_time - head_stay
                    if remain < 0:
                        remain = 0
                    progress = safe_dt if safe_dt < remain else remain
                    parts["proc_reward"] += proc_coef * float(progress)
                if do_warn_penalty:
                    left = proc_time + p_residual - head_stay
                    if left <= p_residual:
                        parts["warn_penalty"] -= warn_coef * safe_dt
            elif has_tok and p_type == DELIVERY_ROBOT and do_transport_penalty:
                for tok in p.tokens:
                    deadline = int(tok.enter_time) + int(self.D_Residual_time)
                    over_start = int(t1) if int(t1) > deadline else deadline
                    if int(t2) > over_start:
                        parts["penalty"] -= float(int(t2) - over_start) * float(transport_coef)

            # --- B: update stay_times ---
            if p_type != SOURCE and safe_dt > 0:
                for tok in p.tokens:
                    tok.stay_time += safe_dt

            # --- C: advance cleaning / idle (PM only) ---
            if safe_dt > 0 and p.name.startswith("PM"):
                if not has_tok:
                    p.idle_time += safe_dt
                else:
                    p.idle_time = 0
                if p.is_cleaning:
                    remaining_clean = int(getattr(p, "cleaning_remaining", 0)) - safe_dt
                    if remaining_clean < 0:
                        remaining_clean = 0
                    p.cleaning_remaining = remaining_clean
                    if remaining_clean == 0:
                        p.is_cleaning = False
                        p.cleaning_reason = ""
                        self.fire_log.append({
                            "event_type": "cleaning_end",
                            "time": int(self.time),
                            "chamber": p.name,
                            "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                        })

            # --- D: scrap / qtime detection (POST-advance stay_time) ---
            if has_tok and (not is_scrap) and p_type == CHAMBER:
                head = p.tokens[0]
                remaining_proc = int(p.processing_time) - int(head.stay_time)
                if remaining_proc < -p_residual:
                    overtime = -remaining_proc - p_residual
                    scrap_info = {
                        "token_id": int(getattr(head, "token_id", -1)),
                        "place": p.name,
                        "stay_time": int(head.stay_time),
                        "proc_time": int(p.processing_time),
                        "overtime": int(overtime),
                        "type": "resident",
                    }
                    is_scrap = True
            if check_qtime and has_tok and p_type == DELIVERY_ROBOT:
                for tok in p.tokens:
                    token_id = int(getattr(tok, "token_id", -1))
                    if token_id < 0 or token_id in self._qtime_violated_tokens:
                        continue
                    if int(tok.stay_time) > qtime_limit:
                        qtime_new_violations.append(token_id)

        for token_id in qtime_new_violations:
            if token_id not in self._qtime_violated_tokens:
                self._qtime_violated_tokens.add(token_id)
                self.qtime_violation_count += 1

        scan_info: Dict[str, Any] = {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
            "qtime_new_violations": qtime_new_violations,
        }
        self._last_state_scan = scan_info

        total = sum(parts.values())
        if detailed:
            parts["total"] = float(total)
            return parts, scan_info
        return float(total), scan_info

    def _fire(self, t_idx: int, start_time: int, end_time: int) -> Dict[str, Any]:
        t_name = self.id2t_name[t_idx]
        pre_places = self._pre_place_indices[t_idx]
        pst_places = self._pst_place_indices[t_idx]
        if pre_places.size == 0 or pst_places.size == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
        pre_place = self.marks[int(pre_places[0])]
        pst_place = self.marks[int(pst_places[0])]
        if len(pre_place.tokens) == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}

        tok = pre_place.pop_head()
        wafer_id = tok.token_id
        self._track_leave(tok, pre_place.name)
        tok.enter_time = self.time
        tok.stay_time = 0

        if t_name.startswith("u_"):
            src = t_name[2:]
            dst_level_targets = tuple(self._u_targets.get(src, []))
            setattr(tok, "_dst_level_targets", dst_level_targets)
            setattr(tok, "_dst_level_full_on_pick", self._is_dst_level_full(src))
            dst = self._select_target_for_source(src, advance_round_robin=True)
            if dst is not None:
                setattr(tok, "_target_place", dst)
            tok.machine = int(self._next_robot_machine())
            if self.device_mode == "cascade":
                # 级联模式下用 machine 字段承载 TM 语义：
                # 1=TM2（LP/PM7/PM8/LLD/PM9/PM10 侧），2=TM3（LLC/PM1-4 侧）
                # 说明：u_LLD/u_PM9/u_PM10 属于返回外环，应落在 TM2 语义。
                if src in {"LLC", "PM1", "PM2", "PM3", "PM4"}:
                    tok.machine = 2
                else:
                    tok.machine = 1
            if src in self._chamber_active and wafer_id in self._chamber_active[src]:
                idx = self._chamber_active[src].pop(wafer_id)
                e, _, wid = self._chamber_timeline[src][idx]
                self._chamber_timeline[src][idx] = (e, start_time, wid)
            self._on_processing_unload(src)
        elif t_name.startswith("t_"):
            target = t_name[2:]
            if hasattr(tok, "_target_place"):
                delattr(tok, "_target_place")
            if hasattr(tok, "_dst_level_targets"):
                delattr(tok, "_dst_level_targets")
            if hasattr(tok, "_dst_level_full_on_pick"):
                delattr(tok, "_dst_level_full_on_pick")
            tok.step = max(int(getattr(tok, "step", 0)), self._step_map.get(target, 0))
            self._track_enter(tok, target)
            if target == "LP_done":
                self.done_count += 1
                self._per_wafer_reward += float(self.done_event_reward)
            elif target in self._chamber_timeline and wafer_id >= 0:
                idx = len(self._chamber_timeline[target])
                self._chamber_timeline[target].append((end_time, None, wafer_id))
                self._chamber_active[target][wafer_id] = idx

        self._advance_token_route_head(tok)
        pst_place.append(tok)
        pre_place_idx = int(pre_places[0])
        pst_place_idx = int(pst_places[0])
        setattr(tok, "_place_idx", pst_place_idx)
        self.m[pre_place_idx] -= 1
        self.m[pst_place_idx] += 1
        if t_name == "u_LP":
            self._last_u_LP_fire_time = int(start_time)
            self._u_LP_release_count += 1
        return {
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(end_time),
            "token_id": wafer_id,
        }

    @staticmethod
    def _token_route_gate(tok: BasedToken) -> object:
        queue = tuple(getattr(tok, "route_queue", ()))
        if not queue:
            return -1
        idx = int(getattr(tok, "route_head_idx", 0))
        if idx < 0:
            idx = 0
        if idx >= len(queue):
            idx = len(queue) - 1
        return queue[idx]

    @staticmethod
    def _advance_token_route_head(tok: BasedToken) -> None:
        queue = tuple(getattr(tok, "route_queue", ()))
        if not queue:
            return
        idx = int(getattr(tok, "route_head_idx", 0))
        next_idx = idx + 1
        if next_idx >= len(queue):
            next_idx = len(queue) - 1
        tok.route_head_idx = int(max(0, next_idx))

    @staticmethod
    def _route_gate_allows_t(gate: object, t_code: int) -> bool:
        if int(t_code) < 0:
            return True
        if gate == -1:
            return True
        if isinstance(gate, int):
            return int(gate) == int(t_code)
        if isinstance(gate, (list, tuple, set)):
            return int(t_code) in {int(x) for x in gate}
        return True

    def _get_enable_t(self) -> List[int]:
        """返回当前使能的变迁索引列表（仅 transition，供 reset 等使用）。"""
        enabled: set[int] = set()
        struct_enabled_cache: Dict[int, bool] = {}

        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(int(t_idx))
            if cached is not None:
                return bool(cached)
            result = bool(self._transition_structurally_enabled(int(t_idx)))
            struct_enabled_cache[int(t_idx)] = result
            return result

        # 按你的要求保持优先级：u_LP 最先检查（受 _allow_start 节拍门控）。
        u_lp_idx = self._u_transition_by_source.get("LP")
        if u_lp_idx is not None and (self._allow_start()):
            if _is_struct_enabled(u_lp_idx) and self._select_target_for_source("LP") is not None:
                enabled.add(int(u_lp_idx))

        for tok in self._token_pool:
            place_idx = int(getattr(tok, "_place_idx", -1))
            if place_idx < 0 or place_idx >= len(self.id2p_name):
                continue
            place_name = self.id2p_name[place_idx]
            if place_name in {"LP", "LP_done"}:
                continue

            # 剩余加工/停留时间 > 0 则不可放行。
            if self._token_remaining_time(tok, place_idx) > 0:
                continue

            # 运输位 token：尝试 t_*
            if place_name.startswith("d_TM"):
                for t_idx in self._t_transitions_by_transport.get(place_name, []):
                    t_name = self.id2t_name[t_idx]
                    target = t_name[2:]
                    target_hint = getattr(tok, "_target_place", None)
                    if target_hint is not None and target_hint != target:
                        continue
                    if not self._route_gate_allows_t(self._token_route_gate(tok), self._t_route_code_by_idx[t_idx]):
                        continue
                    target_place = self._get_place(target)
                    if bool(getattr(target_place, "is_cleaning", False)):
                        continue
                    if not _is_struct_enabled(t_idx):
                        continue
                    enabled.add(int(t_idx))
                continue

            # 加工腔/缓冲位 token：尝试 u_*
            u_idx = self._u_transition_by_source.get(place_name)
            if u_idx is None:
                continue
            if self._select_target_for_source(place_name) is None:
                continue
            if not _is_struct_enabled(u_idx):
                continue
            enabled.add(int(u_idx))

        return sorted(enabled)

    def _allow_start(self):
        """returns True if u_LP can fire now, based on takt and release count."""
        takt = self._takt_result
        cycle_takts = takt["cycle_takts"]
        cycle_len = takt["cycle_length"]
        if self._u_LP_release_count >= 1:
            required = cycle_takts[self._u_LP_release_count % cycle_len]
            if isinstance(required, float):
                required = int(round(required))
            if (self.time - self._last_u_LP_fire_time) < required:
                return False
            else:
                return True
        else:
            return True

    def _build_transition_index(self) -> None:
        self._u_transition_by_source = {}
        self._t_transitions_by_transport = {}
        for t_idx, t_name in enumerate(self.id2t_name):
            if t_name.startswith("u_"):
                self._u_transition_by_source[t_name[2:]] = int(t_idx)
            elif t_name.startswith("t_"):
                target = t_name[2:]
                transport = self._transport_for_t_target(target)
                self._t_transitions_by_transport.setdefault(transport, []).append(int(t_idx))

    def _rebuild_token_pool(self) -> None:
        pool: List[BasedToken] = []
        for place_idx, place in enumerate(self.marks):
            for tok in place.tokens:
                setattr(tok, "_place_idx", int(place_idx))
                pool.append(tok)
        self._token_pool = pool

    def _token_remaining_time(self, tok: BasedToken, place_idx: int) -> int:
        place = self.marks[place_idx]
        if int(place.type) in (CHAMBER, 5, DELIVERY_ROBOT):
            return int(getattr(place, "processing_time", 0)) - int(getattr(tok, "stay_time", 0))
        return 0

    def _transition_structurally_enabled(self, t_idx: int) -> bool:
        pre_idx = self._pre_place_indices[t_idx]
        if pre_idx.size > 0 and np.any(self.m[pre_idx] < self.pre[pre_idx, t_idx]):
            return False
        pst_idx = self._pst_place_indices[t_idx]
        if pst_idx.size > 0 and np.any(
            self.m[pst_idx] + self.net[pst_idx, t_idx] > self.k[pst_idx]
        ):
            return False
        return True

    def get_enable_actions_with_reasons(
        self,
        wait_action_start: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        返回使能动作列表及不使能动作及原因。
        仅在评估模式下被调用，避免训练时额外开销。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)

        d_tm = self._get_place("d_TM1") if ("d_TM1" in self.id2p_name and self.device_mode != "cascade") else None
        head_tok = d_tm.head() if (d_tm is not None and len(d_tm.tokens) > 0) else None
        locked_sources: Set[str] = set()
        if self.robot_capacity == 2 and self.device_mode != "cascade" and head_tok is not None:
            locked_sources = set(getattr(head_tok, "_dst_level_targets", ()))

        enabled: List[int] = []
        disabled: List[Dict[str, Any]] = []

        # 遍历变迁（使用缓存的 pre/pst 索引）
        for t in range(self.T):
            base_pre_idx = self._pre_place_indices[t]
            t_name = self.id2t_name[t]

            if base_pre_idx.size == 0:
                disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                continue

            if np.any(self.m[base_pre_idx] < self.pre[base_pre_idx, t]):
                disabled.append({"action": t, "name": t_name, "reason": "insufficient_tokens"})
                continue
            if np.any(self.m + self.net[:, t] > self.k):
                disabled.append({"action": t, "name": t_name, "reason": "capacity_exceeded"})
                continue

            if t_name.startswith("u_"):
                src = t_name[2:]
                if self.robot_capacity == 2 and self.device_mode != "cascade":
                    if d_tm is not None and len(d_tm.tokens) > 0 and locked_sources and src not in locked_sources:
                        disabled.append({"action": t, "name": t_name, "reason": "locked_by_arm2_head"})
                        continue
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                        continue
                else:
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                        continue
            elif t_name.startswith("t_"):
                t_code = self._t_route_code_by_idx[t]
                tp_idx = self._transport_pre_place_idx[t]
                if tp_idx >= 0:
                    tp_head = self.marks[tp_idx].head() if len(self.marks[tp_idx].tokens) > 0 else None
                    if tp_head is not None:
                        gate = self._token_route_gate(tp_head)
                        if not self._route_gate_allows_t(gate, t_code):
                            disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                            continue
                target = t_name[2:]
                transport_name = self._transport_for_t_target(target)
                tp = self._get_place(transport_name)
                tp_head = tp.head() if len(tp.tokens) > 0 else None
                tp_head_target = getattr(tp_head, "_target_place", None) if tp_head is not None else None
                if tp_head_target is not None and tp_head_target != target:
                    disabled.append({"action": t, "name": t_name, "reason": "wrong_destination"})
                    continue

            # Stage2
            if t_name.startswith("u_"):
                src = t_name[2:]
                if not self._is_process_ready(src):
                    disabled.append({"action": t, "name": t_name, "reason": "process_not_ready"})
                    continue
                if self._select_target_for_source(src) is None:
                    disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                    continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                target_place = self._get_place(target)
                if bool(getattr(target_place, "is_cleaning", False)):
                    disabled.append({"action": t, "name": t_name, "reason": "target_cleaning"})
                    continue
                d_place = self._get_place(self._transport_for_t_target(target))
                dwell_time = max(0, int(getattr(d_place, "processing_time", self.T_transport)))
                if len(d_place.tokens) > 0 and d_place.head().stay_time < dwell_time:
                    disabled.append({"action": t, "name": t_name, "reason": "dwell_time_not_met"})
                    continue

            # 节拍限制：u_LP 发片间隔不得小于当前周期节拍
            if self._takt_result and t_name == "u_LP":
                takt = self._takt_result
                cycle_takts = takt["cycle_takts"]
                cycle_len = takt["cycle_length"]
                if self._u_LP_release_count >= 1:
                    required = cycle_takts[self._u_LP_release_count % cycle_len]
                    if isinstance(required, float):
                        required = int(round(required))
                    if (self.time - self._last_u_LP_fire_time) < required:
                        disabled.append({"action": t, "name": t_name, "reason": "takt_release_limit"})
                        continue

            enabled.append(t)

        # wait 动作
        restrict_long_wait = self._has_ready_chamber_wafers()
        for offset, duration in enumerate(self.wait_durations):
            action_idx = start + int(offset)
            if restrict_long_wait and int(duration) > 5:
                disabled.append({
                    "action": action_idx,
                    "name": f"WAIT_{int(duration)}s",
                    "reason": "has_ready_wafer_restrict_wait",
                })
            else:
                enabled.append(action_idx)

        return {"enabled": enabled, "disabled": disabled}

    def calc_wafer_statistics(self) -> Dict[str, Any]:
        system_times: List[float] = []
        chamber_stats: Dict[str, List[float]] = {}
        completed = 0
        for _, s in self._token_stats.items():
            enter = s.get("enter_system")
            leave = s.get("exit_system")
            if enter is not None and leave is not None:
                completed += 1
                system_times.append(float(leave - enter))
            for c_name, c_stat in s.get("chambers", {}).items():
                c_enter = c_stat.get("enter")
                c_leave = c_stat.get("exit")
                if c_enter is not None and c_leave is not None:
                    chamber_stats.setdefault(c_name, []).append(float(c_leave - c_enter))

        chamber_summary: Dict[str, Dict[str, float]] = {}
        for name, vals in chamber_stats.items():
            if vals:
                chamber_summary[name] = {"avg": sum(vals) / len(vals), "max": max(vals), "count": len(vals)}

        return {
            "system_avg": (sum(system_times) / len(system_times)) if system_times else 0.0,
            "system_max": max(system_times) if system_times else 0.0,
            "system_diff": 0.0,
            "completed_count": completed,
            "in_progress_count": max(0, self.n_wafer - completed),
            "chambers": chamber_summary,
            "transports": {},
            "transports_detail": {},
            "resident_violation_count": self.resident_violation_count,
            "qtime_violation_count": self.qtime_violation_count,
            "deadlock_count": self.deadlock_count,
            "chamber_processed_counts": {
                p.name: int(getattr(p, "processed_wafer_count", 0))
                for p in self.marks
                if p.name.startswith("PM")
            },
        }

    def _refresh_episode_proc_time(self) -> None:
        """生成本 episode 工序时长（随机+取整到5）并应用到 marks 与 ptime。"""
        if self.proc_rand_enabled:
            sampled: Dict[str, int] = {}
            for chamber in self.chambers:
                raw = self._proc_rand_scale_map.get(chamber, {})
                if not isinstance(raw, dict):
                    raw = {}
                low = float(raw.get("min", 1.0))
                high = float(raw.get("max", 1.0))
                base_time = float(self._base_proc_time_map[chamber])
                sampled_time = base_time * float(np.random.uniform(low, high))
                sampled[chamber] = _round_to_nearest_five(sampled_time)
            self._episode_proc_time_map = sampled
        else:
            self._episode_proc_time_map = dict(self._base_proc_time_map)
        for p in self.marks:
            if p.name in self._episode_proc_time_map:
                p.processing_time = int(self._episode_proc_time_map[p.name])
        for chamber_name, proc_time in self._episode_proc_time_map.items():
            p_idx = self._get_place_index(chamber_name)
            self.ptime[p_idx] = int(proc_time)

    def _preprocess_process_time_map(self, process_time_map: Dict[str, int]) -> Dict[str, int]:
        if self.device_mode == "cascade":
            pm_stage3_default = 600 if self.route_code == 1 else 300
            defaults = {
                "PM7": 70,
                "PM8": 70,
                "PM1": pm_stage3_default,
                "PM2": pm_stage3_default,
                "LLD": 70,
            }
            if self.route_code != 3:
                defaults.update({"PM9": 200, "PM10": 200})
            if self.route_code == 1:
                defaults.update({"PM3": pm_stage3_default, "PM4": pm_stage3_default})
        else:
            defaults = {"PM1": 100, "PM3": 300, "PM4": 300, "PM6": 300}
        return _preprocess_process_time_map(
            process_time_map=process_time_map,
            chambers=self.chambers,
            defaults=defaults,
        )

    # 节拍分析器内部会统一给每道工序 p 加运输时间常量（当前口径为 +20）
    _TRANSPORT_TIME_FOR_TAKT: int = 20

    def _build_takt_stage(self, stage_idx: int, stage_places: List[str]) -> Optional[Dict[str, Any]]:
        """
        将一层 route stage 归一化为 analyzer 输入。
        - 并行 stage 的 p 取该层瓶颈（max）
        - 清洗参数优先取该层中最可能形成慢节拍的腔室（max(p+d)）
        """
        valid_places = [
            place
            for place in stage_places
            if int(self._episode_proc_time_map.get(place, 0) or 0) > 0
        ]
        if not valid_places:
            return None
        base_p = max(int(self._episode_proc_time_map[place]) for place in valid_places)
        q: Optional[int] = None
        d = 0
        if self.cleaning_enabled:
            cleaning_candidates: List[Tuple[int, int, int, str]] = []
            for place in valid_places:
                trigger = int(self._cleaning_trigger_map.get(place, 0))
                if trigger <= 0:
                    continue
                duration = int(self._cleaning_duration_map.get(place, self.cleaning_duration))
                score = int(self._episode_proc_time_map.get(place, 0)) + duration
                cleaning_candidates.append((score, trigger, duration, place))
            if cleaning_candidates:
                _, q, d, _ = max(cleaning_candidates, key=lambda item: (item[0], item[3]))
        return {
            "name": f"s{stage_idx + 1}",
            "p": int(base_p),
            "m": len(valid_places),
            "q": q,
            "d": int(d),
        }

    def _compute_takt_result(self) -> Optional[Dict[str, Any]]:
        """
        根据当前加工配方（路线 + 工序时长 + 清洗参数）调用节拍分析器，
        analyzer 内部会统一把每道工序处理时间按「工序时长 + 运输时间」计入节拍。
        失败或无可分析工序时返回 None。
        """
        analyzer_stages: List[Dict[str, Any]] = []
        for i, stage in enumerate(self._route_stages):
            if not stage:
                continue
            stage_cfg = self._build_takt_stage(stage_idx=i, stage_places=list(stage))
            if stage_cfg is None:
                continue
            analyzer_stages.append(stage_cfg)
        if not analyzer_stages:
            return None
        try:
            return analyze_cycle(analyzer_stages, max_parts=10000)
        except Exception:
            return None

    @staticmethod
    def _clone_marks(marks: List[Place]) -> List[Place]:
        cloned: List[Place] = []
        for p in marks:
            cp = p.clone()
            cloned.append(cp)
        return cloned

    def _get_place(self, name: str) -> Place:
        place = self._place_by_name.get(name)
        if place is None:
            raise KeyError(f"unknown place: {name}")
        return place

    def _get_place_index(self, name: str) -> int:
        return self.id2p_name.index(name)

    def _transport_for_t_target(self, target: str) -> str:
        if self.device_mode != "cascade":
            return "d_TM1"
        if target in {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LLD"}:
            return "d_TM3"
        return "d_TM2"

    def _update_stay_times(self, dt: int) -> None:
        if dt <= 0:
            return
        for p in self.marks:
            if p.type == SOURCE:  # 跳过 LP 中的 wafer
                continue
            for tok in p.tokens:
                tok.stay_time += dt

    def _init_cleaning_state(self) -> None:
        for p in self.marks:
            if not p.name.startswith("PM"):
                continue
            p.processed_wafer_count = int(getattr(p, "processed_wafer_count", 0))
            p.idle_time = int(getattr(p, "idle_time", 0))
            p.last_proc_type = str(getattr(p, "last_proc_type", ""))
            p.is_cleaning = bool(getattr(p, "is_cleaning", False))
            p.cleaning_remaining = int(getattr(p, "cleaning_remaining", 0))
            p.cleaning_reason = str(getattr(p, "cleaning_reason", ""))

    def _advance_cleaning_and_idle(self, dt: int) -> None:
        if dt <= 0:
            return
        for p in self.marks:
            if not p.name.startswith("PM"):
                continue
            if len(p.tokens) == 0:
                p.idle_time += int(dt)
            else:
                p.idle_time = 0

            if not p.is_cleaning:
                continue
            remaining = max(0, int(getattr(p, "cleaning_remaining", 0)) - int(dt))
            was_cleaning = p.is_cleaning
            p.cleaning_remaining = remaining
            if was_cleaning and remaining == 0:
                p.is_cleaning = False
                p.cleaning_reason = ""
                self.fire_log.append(
                    {
                        "event_type": "cleaning_end",
                        "time": int(self.time),
                        "chamber": p.name,
                        "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                    }
                )

    def get_next_event_delta(self) -> Optional[int]:
        """
        计算当前时刻到下一个关键事件的时间差（秒）。
        通过扫描 _token_pool，按 token 所在库所（运输 d_TM* / 加工腔室）用不同规则计算；
        节拍为「下一节拍时刻 - 当前时间」。
        关键事件至少覆盖：
        - 某加工完成（避免跨过关键取片决策点）
        - 运输位达到 T_transport
        - u_LP 到达节拍（下一次允许 u_LP 发片的时刻，用于截断长 wait）
        """
        deltas: List[int] = []
        n_places = len(self.marks)
        for tok in self._token_pool:
            place_idx = int(getattr(tok, "_place_idx", -1))
            if place_idx < 0 or place_idx >= n_places:
                continue
            place = self.marks[place_idx]

            # 运输库所 d_TM*：停留达到 T_transport 即关键事件。
            if place.name.startswith("d_TM"):
                delta_tm = int(self.T_transport) - int(getattr(tok, "stay_time", 0))
                deltas.append(max(0, int(delta_tm)))
                continue

            # 加工库所（腔室）：type in (CHAMBER, 5) 且 processing_time > 0 且非 d_*。
            if int(place.type) in (CHAMBER, 5) and not place.name.startswith("d_"):
                ptime = int(getattr(place, "processing_time", 0))
                if ptime > 0:
                    delta = ptime - int(getattr(tok, "stay_time", 0))
                    deltas.append(max(0, int(delta)))
        # u_LP 节拍使能时刻：下一节拍时刻 - 当前时间（节拍间隔 - 当前未发片时间）
        if self._takt_result and self._u_LP_release_count >= 1:
            takt = self._takt_result
            cycle_takts = takt["cycle_takts"]
            cycle_len = takt["cycle_length"]
            required = cycle_takts[self._u_LP_release_count % cycle_len]
            if isinstance(required, float):
                required = int(round(required))
            next_takt_time = self._last_u_LP_fire_time + required
            delta_takt = next_takt_time - self.time
            if delta_takt > 0:
                deltas.append(int(delta_takt))
        if not deltas:
            return None
        return int(min(deltas))

    def _has_ready_chamber_wafers(self) -> bool:
        """
        判断是否存在“加工完成待取片”晶圆。
        规则：在当前路径定义的任一加工腔室中，存在 token 满足 stay_time >= processing_time。
        """
        for chamber_name in self._ready_chambers:
            place = self._get_place(chamber_name)
            processing_time = int(getattr(place, "processing_time", 0))
            if processing_time <= 0:
                continue
            for tok in place.tokens:
                if int(getattr(tok, "stay_time", 0)) >= processing_time:
                    return True
        return False

    def _scan_runtime_state(self) -> Dict[str, Any]:
        """
        扫描当前 token 状态并返回 scrap 与 qtime 统计结果。
        该扫描在 step 内由 advance_time 调用，避免重复遍历 token。
        """
        scrap_info: Optional[Dict[str, Any]] = None
        is_scrap = False
        qtime_new_violations: List[int] = []
        n_places = len(self.marks)
        qtime_limit = int(self.D_Residual_time)

        for tok in self._token_pool:
            place_idx = int(getattr(tok, "_place_idx", -1))
            if place_idx < 0 or place_idx >= n_places:
                continue
            place = self.marks[place_idx]
            stay_time = int(getattr(tok, "stay_time", 0))

            if (not is_scrap) and place.type == CHAMBER:
                remaining = int(place.processing_time) - stay_time
                if remaining < -int(self.P_Residual_time):
                    overtime = int(-remaining - int(self.P_Residual_time))
                    scrap_info = {
                        "token_id": int(getattr(tok, "token_id", -1)),
                        "place": place.name,
                        "stay_time": stay_time,
                        "proc_time": int(place.processing_time),
                        "overtime": overtime,
                        "type": "resident",
                    }
                    is_scrap = True

            if self._training:
                continue
            if place.type != DELIVERY_ROBOT:
                continue
            token_id = int(getattr(tok, "token_id", -1))
            if token_id < 0 or token_id in self._qtime_violated_tokens:
                continue
            if stay_time > qtime_limit:
                qtime_new_violations.append(token_id)

        return {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
            "qtime_new_violations": qtime_new_violations,
        }

    def advance_time(self, dt: int, event_reason: str = "") -> Dict[str, Any]:
        """推进时间并更新相关状态"""
        _ = event_reason  # 预留参数，便于后续追踪不同推进原因。
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        self._update_stay_times(safe_dt)
        self._advance_cleaning_and_idle(safe_dt)
        scan_info = self._scan_runtime_state()
        for token_id in scan_info.get("qtime_new_violations", []):
            if token_id in self._qtime_violated_tokens:
                continue
            self._qtime_violated_tokens.add(token_id)
            self.qtime_violation_count += 1
        self._last_state_scan = scan_info
        return scan_info

    def _on_processing_unload(self, source_name: str) -> None:
        if not self.cleaning_enabled:
            return
        trigger = self._cleaning_trigger_map.get(source_name, 0)
        if trigger <= 0:
            return
        source_place = self._get_place(source_name)
        source_place.processed_wafer_count = int(getattr(source_place, "processed_wafer_count", 0)) + 1
        source_place.last_proc_type = source_name
        if source_place.is_cleaning:
            return
        if source_place.processed_wafer_count >= trigger:
            count = int(source_place.processed_wafer_count)
            duration = self._cleaning_duration_map.get(source_name, self.cleaning_duration)
            source_place.is_cleaning = True
            source_place.cleaning_remaining = int(duration)
            source_place.cleaning_reason = "processed_wafers"
            source_place.processed_wafer_count = 0
            source_place.idle_time = 0
            self.fire_log.append(
                {
                    "event_type": "cleaning_start",
                    "time": int(self.time),
                    "chamber": source_place.name,
                    "rule": "processed_wafers",
                    "duration": int(duration),
                    "trigger_count": int(count),
                }
            )

    def _is_process_ready(self, place_name: str) -> bool:
        place = self._get_place(place_name)
        if len(place.tokens) == 0:
            return False
        if place.processing_time <= 0:
            return True
        return place.head().stay_time >= place.processing_time

    def _select_target_for_source(
        self,
        source: str,
        preferred_target: Optional[str] = None,
        ignore_cleaning: bool = False,
        advance_round_robin: bool = False,
    ) -> Optional[str]:
        """
        为 u_<source> 选择一个可接收目标（确定性顺序）。
        仅检查目标腔室容量，运输位停留时间约束仍由 t_* 侧控制。
        """
        candidates = self._u_targets.get(source, [])
        if self.device_mode == "cascade" and source in {"PM7", "PM8"}:
            # LLC 满时仍允许从 PM7/PM8 取片到 d_TM1，后续由 t_LLC 容量约束拦截放片时机。
            # 这与用户要求一致：LLC 有片时 u_PM7/u_PM8 仍应可使能。
            if "LLC" in candidates:
                return "LLC"
        if self.device_mode == "cascade" and source in {"PM1", "PM2", "PM3", "PM4"}:
            # LLD 满时仍允许从 PM1-4 先取片到 d_TM1，后续由 t_LLD 实际容量约束放行。
            # 需求：LLD 有晶圆时，u_PM1/u_PM2/u_PM3/u_PM4 仍应保持可使能。
            if "LLD" in candidates:
                return "LLD"
        if self.device_mode == "cascade" and source in self._cascade_round_robin_pairs and preferred_target is None:
            # 级联并行目标采用轮换分配；仅在真实发射时推进轮换指针，避免使能检查污染状态。
            rr_targets = list(self._cascade_round_robin_pairs[source])
            available_targets: List[str] = []
            for target in rr_targets:
                if target not in candidates:
                    continue
                target_place = self._get_place(target)
                if (not ignore_cleaning) and bool(getattr(target_place, "is_cleaning", False)):
                    continue
                if len(target_place.tokens) < target_place.capacity:
                    available_targets.append(target)
            if available_targets:
                next_target = self._cascade_round_robin_next.get(source, rr_targets[0])
                start_idx = rr_targets.index(next_target) if next_target in rr_targets else 0
                chosen_target = available_targets[0]
                for offset in range(len(rr_targets)):
                    candidate = rr_targets[(start_idx + offset) % len(rr_targets)]
                    if candidate in available_targets:
                        chosen_target = candidate
                        break
                if advance_round_robin and len(rr_targets) > 1:
                    chosen_idx = rr_targets.index(chosen_target)
                    self._cascade_round_robin_next[source] = rr_targets[
                        (chosen_idx + 1) % len(rr_targets)
                    ]
                return chosen_target
        if self.device_mode == "single" and source in self._single_round_robin_pairs and preferred_target is None:
            # single 模式并行目标采用轮换分配，避免持续偏置到候选列表中的第一个目标。
            rr_targets = list(self._single_round_robin_pairs[source])
            available_targets: List[str] = []
            for target in rr_targets:
                if target not in candidates:
                    continue
                target_place = self._get_place(target)
                if (not ignore_cleaning) and bool(getattr(target_place, "is_cleaning", False)):
                    continue
                if len(target_place.tokens) < target_place.capacity:
                    available_targets.append(target)
            if len(available_targets) == 2:
                next_target = self._single_round_robin_next.get(source, available_targets[0])
                chosen_target = next_target if next_target in available_targets else available_targets[0]
                if advance_round_robin:
                    self._single_round_robin_next[source] = (
                        available_targets[1] if chosen_target == available_targets[0] else available_targets[0]
                    )
                return chosen_target
            if len(available_targets) == 1:
                return available_targets[0]
        if preferred_target is not None:
            if preferred_target not in candidates:
                return None
            target_place = self._get_place(preferred_target)
            if (not ignore_cleaning) and bool(getattr(target_place, "is_cleaning", False)):
                return None
            if len(target_place.tokens) < target_place.capacity:
                return preferred_target
            return None
        for target in candidates:
            target_place = self._get_place(target)
            if (not ignore_cleaning) and bool(getattr(target_place, "is_cleaning", False)):
                continue
            if len(target_place.tokens) < target_place.capacity:
                return target
        return None

    def _next_robot_machine(self) -> int:
        if self.robot_capacity <= 1:
            return 1
        machine_id = self._next_machine_id
        self._next_machine_id = 2 if machine_id == 1 else 1
        return machine_id

    def _is_dst_level_full(self, source: str) -> bool:
        targets = self._u_targets.get(source, [])
        if not targets:
            return False
        for target in targets:
            p = self._get_place(target)
            if len(p.tokens) < p.capacity:
                return False
        return True

    def _check_scrap(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        for tok in self._token_pool:
            place_idx = int(getattr(tok, "_place_idx", -1))
            if place_idx < 0 or place_idx >= len(self.marks):
                continue
            p = self.marks[place_idx]
            if p.type != CHAMBER:
                continue
            remaining = int(p.processing_time) - int(getattr(tok, "stay_time", 0))
            if remaining < -int(self.P_Residual_time):
                overtime = int(-remaining - int(self.P_Residual_time))
                return True, {
                    "token_id": int(getattr(tok, "token_id", -1)),
                    "place": p.name,
                    "stay_time": int(tok.stay_time),
                    "proc_time": int(p.processing_time),
                    "overtime": overtime,
                    "type": "resident",
                }
        return False, None

    def _check_qtime_violation(self) -> None:
        """检查运输位 Q-Time 超时并按 wafer 去重累计，不施加惩罚。"""
        if self._training:
            return
        qtime_limit = int(self.D_Residual_time)
        for place in self.marks:
            if place.type != DELIVERY_ROBOT or len(place.tokens) == 0:
                continue
            for tok in place.tokens:
                token_id = int(getattr(tok, "token_id", -1))
                if token_id < 0 or token_id in self._qtime_violated_tokens:
                    continue
                if int(getattr(tok, "stay_time", 0)) > qtime_limit:
                    self._qtime_violated_tokens.add(token_id)
                    self.qtime_violation_count += 1

    def blame_release_violations(self) -> Dict[int, float]:
        """
        事后追责：基于当前 fire_log 与 _chamber_timeline，回溯可能导致下游容量冲突的 u_* 动作。
        单设备中会按路径代号聚合站点：s1=PM1，s2=PM3∪PM4；若 code=1 则新增 s3=PM6。
        返回 fire_log_index -> penalty。
        """
        blame: Dict[int, float] = {}
        assert len(self.fire_log) > 0, "fire_log is empty"

        # 1) 预处理：构建占用时间线（enter, leave, wafer_id），leave 可能为 None 表示仍在加工中。
        proc_times = {p.name: p.processing_time for p in self.marks}
        capacities = {p.name: p.capacity for p in self.marks}

        def build_intervals(chamber_name: str) -> List[tuple]:
            intervals: List[tuple] = []
            for (enter, leave, wid) in self._chamber_timeline.get(chamber_name, []):
                l = leave if leave is not None else enter + proc_times.get(chamber_name, 0)
                intervals.append((enter, l, wid))
            intervals.sort(key=lambda x: x[0])
            return intervals

        # 从 fire_log 提取清洁区间（腔室在清洁期间占位，wid=-999 表示清洁占位）
        def build_cleaning_intervals(chamber_name: str) -> List[tuple]:
            if not getattr(self, "cleaning_enabled", False) or int(self._cleaning_trigger_map.get(chamber_name, 0)) <= 0:
                return []
            default_dur = self._cleaning_duration_map.get(chamber_name, self.cleaning_duration)
            out: List[tuple] = []
            for ev in self.fire_log:
                if ev.get("event_type") == "cleaning_start" and ev.get("chamber") == chamber_name:
                    t0 = int(ev.get("time", 0))
                    dur = int(ev.get("duration", default_dur))
                    out.append((t0, t0 + dur, -999))
            return out

        intervals_by_station: Dict[str, List[tuple]] = {}
        capacity_by_station: Dict[str, int] = {}
        proc_time_by_station: Dict[str, int] = {}
        for station, chambers in self._release_station_aliases.items():
            merged: List[tuple] = []
            for chamber_name in chambers:
                merged.extend(build_intervals(chamber_name))
                merged.extend(build_cleaning_intervals(chamber_name))
            merged.sort(key=lambda x: x[0])
            intervals_by_station[station] = merged
            capacity_by_station[station] = int(sum(capacities.get(name, 0) for name in chambers))
            proc_time_by_station[station] = int(max((proc_times.get(name, 0) for name in chambers), default=0))

        edge_transfer = self.T_transport + self.T_load

        def will_exceed_capacity(intervals, at_time, cap, current_wid):
            occupied = sum(1 for (e, l, wid0) in intervals if e <= at_time < l and wid0 < current_wid)
            return occupied + 1 > cap

        # 单设备 downstream chain：按路径代号切换（code0: s1->s2, code1: s1->s2->s3）
        chain_map: Dict[str, List[str]] = dict(self._release_chain_by_u)

        # 2) 回溯 fire_log，针对每个 u_* 动作检查其 downstream chain 是否存在容量冲突。
        # 仅追责释放动作：u_LP（从 LP 释放）、u_LLC（从 LLC 释放）、u_LLD（从 LLD 释放）。
        # u_PM7、u_PM2 等从加工腔卸载的动作不追责。
        RELEASE_BLAME_WHITELIST = frozenset({"u_LP", "u_LLC"})
        penalty_coeff = self.release_event_penalty
        for i, log in enumerate(self.fire_log):
            t_name = log.get("t_name", "")
            wid = int(log.get("token_id", -1))
            if t_name not in RELEASE_BLAME_WHITELIST:
                continue
            # 必须有合法的 wafer_id
            if wid < 0:
                continue
            chain = chain_map.get(t_name, [])
            if not chain:
                continue

            t_leave = int(log.get("t1", 0))
            arrival = t_leave + edge_transfer
            violated = False
            for idx, station in enumerate(chain):
                intervals = intervals_by_station.get(station, [])
                cap = capacity_by_station.get(station, 1)
                exceeds = will_exceed_capacity(intervals, arrival, cap, wid)
                if exceeds:
                    violated = True
                    break
                if idx < len(chain) - 1:
                    arrival = arrival + proc_time_by_station.get(station, 0) + edge_transfer

            if violated:
                blame[i] = penalty_coeff
        return blame

    _MASK_SKIP_PLACES = frozenset({"LP", "LP_done"})
    _MASK_TIMED_TYPES = frozenset((CHAMBER, 5, DELIVERY_ROBOT))

    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """
        返回完整离散动作掩码（transition + wait）。
        遍历 marks（而非 _token_pool），内联 remaining_time / has_ready_chamber 检查。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)
        total_actions = int(
            n_actions if n_actions is not None else (start + len(self.wait_durations))
        )
        mask = np.zeros(total_actions, dtype=bool)
        struct_enabled_cache: Dict[int, bool] = {}

        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(t_idx)
            if cached is not None:
                return cached
            result = bool(self._transition_structurally_enabled(t_idx))
            struct_enabled_cache[t_idx] = result
            return result

        u_lp_idx = self._u_transition_by_source.get("LP")
        if u_lp_idx is not None and self._allow_start():
            if _is_struct_enabled(u_lp_idx) and self._select_target_for_source("LP") is not None:
                t_idx = int(u_lp_idx)
                if 0 <= t_idx < total_actions:
                    mask[t_idx] = True

        has_ready_chamber = False
        ready_chambers = self._ready_chambers
        target_cache: Dict[str, Optional[str]] = {}
        skip = self._MASK_SKIP_PLACES
        timed = self._MASK_TIMED_TYPES
        t_trans_by_transport = self._t_transitions_by_transport
        u_trans_by_source = self._u_transition_by_source
        id2t = self.id2t_name
        route_code_by_idx = self._t_route_code_by_idx
        route_gate_allows = self._route_gate_allows_t
        token_route_gate = self._token_route_gate
        get_place = self._get_place

        for place in self.marks:
            n_tok = len(place.tokens)
            if n_tok == 0:
                continue
            pname = place.name
            if pname in skip:
                continue
            p_type = place.type
            proc_time = place.processing_time

            is_timed = p_type in timed
            is_dtm = pname.startswith("d_TM")

            if is_dtm:
                for tok in place.tokens:
                    if is_timed and proc_time > 0 and tok.stay_time < proc_time:
                        continue
                    for t_idx in t_trans_by_transport.get(pname, ()):
                        target = id2t[t_idx][2:]
                        target_hint = getattr(tok, "_target_place", None)
                        if target_hint is not None and target_hint != target:
                            continue
                        if not route_gate_allows(token_route_gate(tok), route_code_by_idx[t_idx]):
                            continue
                        target_place = get_place(target)
                        if target_place.is_cleaning:
                            continue
                        if not _is_struct_enabled(t_idx):
                            continue
                        if 0 <= t_idx < total_actions:
                            mask[t_idx] = True
            else:
                head = place.tokens[0]
                if is_timed and proc_time > 0 and head.stay_time < proc_time:
                    continue
                u_idx = u_trans_by_source.get(pname)
                if u_idx is not None:
                    cached_target = target_cache.get(pname)
                    if cached_target is None and pname not in target_cache:
                        cached_target = self._select_target_for_source(pname)
                        target_cache[pname] = cached_target
                    if cached_target is not None and _is_struct_enabled(u_idx):
                        if 0 <= u_idx < total_actions:
                            mask[u_idx] = True

            if not has_ready_chamber and p_type == CHAMBER and proc_time > 0 and pname in ready_chambers:
                for tok in place.tokens:
                    if tok.stay_time >= proc_time:
                        has_ready_chamber = True
                        break

        for offset, duration in enumerate(self.wait_durations):
            if has_ready_chamber and int(duration) > 5:
                continue
            idx = start + int(offset)
            if 0 <= idx < total_actions:
                mask[idx] = True

        return mask

    def _track_enter(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            self._token_stats[token.token_id] = {"enter_system": None, "exit_system": None, "chambers": {}}
        if place_name in self._system_entry_places and self._token_stats[token.token_id]["enter_system"] is None:
            self._token_stats[token.token_id]["enter_system"] = self.time
        if place_name == "LP_done":
            self._token_stats[token.token_id]["exit_system"] = self.time
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": self.time, "exit": None})

    def _track_leave(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            return
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": None, "exit": None})["exit"] = self.time

    def render_gantt(self, out_path: str) -> None:
        return None
