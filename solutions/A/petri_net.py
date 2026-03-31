from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from solutions.A.utils import _normalize_wait_durations
import numpy as np
from config.cluster_tool.env_config import PetriEnvConfig
from solutions.A.construct import BasedToken
from solutions.A.model_builder import build_net
from solutions.A.deprecated.pn import Place

CHAMBER = 1
DELIVERY_ROBOT = 2
SOURCE = 3


class ClusterTool:
    def __init__(self, config: PetriEnvConfig = None) -> None:
        assert config is not None, "config must be provided"
        self.config = config

        # ====== 1) 运行边界与奖励超参 ======
        self.MAX_TIME = config.MAX_TIME
        self.n_wafer1 = int(config.n_wafer1)
        self.n_wafer2 = int(config.n_wafer2)
        self.n_wafer = int(self.n_wafer1) + int(self.n_wafer2)
        self.max_wafers1_in_system = int(config.max_wafers1_in_system)
        self.max_wafers2_in_system = int(config.max_wafers2_in_system)
        
        self.done_event_reward = int(config.done_event_reward)
        self.finish_event_reward = self.done_event_reward * 6
        self.scrap_event_penalty = int(config.scrap_event_penalty)
        self.idle_event_penalty = float(config.idle_event_penalty)
        
        self.warn_coef_penalty = float(config.warn_coef_penalty)
        self.processing_coef_reward = float(config.processing_coef_reward)
        self.transport_overtime_coef_penalty = float(config.transport_overtime_coef_penalty)
        self.time_coef_penalty = float(config.time_coef_penalty)
        
        # ====== 2) 驻留/Q-time 与动作时长 ======
        self.P_Residual_time = int(config.P_Residual_time)
        self.D_Residual_time = int(config.D_Residual_time)
        self._dual_arm = bool(getattr(config, 'dual_arm', False))
        self.swap_duration = 10

        # ====== 3) 清洗配置（map 来自配置/路线；开关由 cleaning_enabled） ======
        self._cleaning_enabled = bool(config.cleaning_enabled)
        self._cleaning_default_duration = 0
        self._cleaning_default_trigger = 0
        self._cleaning_duration_map: Dict[str, int] = {
            str(name): max(0, int(value))
            for name, value in dict(getattr(config, "cleaning_duration_map", {}) or {}).items()
        }
        self._cleaning_trigger_map: Dict[str, int] = {
            str(name): max(0, int(value))
            for name, value in dict(getattr(config, "cleaning_trigger_wafers_map", {}) or {}).items()
        }
        self.wait_durations = _normalize_wait_durations(config.wait_durations)
        
        # ====== 6) 构网输入与构网结果 ======
        self.ttime = 5
        self.single_route_config = config.single_route_config
        self.single_route_name = config.single_route_name

        info = build_net(n_wafer1=self.n_wafer1,
                         n_wafer2=self.n_wafer2,
                         ttime=self.ttime,
                         p_residual_time=self.P_Residual_time,
                         d_residual_time=self.D_Residual_time,
                         cleaning_enabled=self._cleaning_enabled,
                         route_config=self.single_route_config, route_name=self.single_route_name)
        self._base_proc_time_map = dict(info.get("process_time_map") or {})
        route_meta = dict(info.get("route_meta") or {})

        # ====== 7) route_meta：阶段/拓扑/类型映射 ======
        route_stages = list(route_meta.get("route_stages") or [])
        self._route_stages = [list(stage) for stage in route_stages]
        self.chambers = tuple(route_meta.get("chambers", ()))
        self._u_targets = dict(route_meta.get("u_targets", {}))
        self._step_map = dict(route_meta.get("step_map", {}))
        self._has_repeat_syntax_reentry = bool(route_meta.get("has_repeat_syntax_reentry", False))
        self._cleaning_duration_map = route_meta.get("cleaning_duration_map")
        self._cleaning_trigger_map = route_meta.get("cleaning_trigger_wafers_map")
        self._multi_subpath = bool(route_meta.get("multi_subpath", False))
        self._subpath_to_type: Dict[str, int] = route_meta.get("subpath_to_type")
        self._wafer_type_to_subpath: Dict[int, str] = route_meta.get("wafer_type_to_subpath")
        self._takt_policy: str = str(route_meta.get("takt_policy", "") or "")
        self._wafer_type_to_load_port: Dict[int, str] = route_meta.get("wafer_type_to_load_port")
        self._load_port_names: Tuple[str, ...] = route_meta.get("load_port_names")
        self._mask_skip_places: frozenset[str] = frozenset(self._load_port_names) | {"LP_done"}
        self._ready_chambers = route_meta.get("chambers")
        self._single_process_chambers = self.chambers

        # ====== 8) Petri 静态结构索引 ======
        self.m0: np.ndarray = info["m0"]
        self.m: np.ndarray = self.m0.copy()
        self.k: np.ndarray = info["capacity"]
        self.id2p_name: List[str] = info["id2p_name"]
        self.id2t_name: List[str] = info["id2t_name"]
        self._t_route_code_map: Dict[str, int] = dict(info.get("t_route_code_map") or {})
        self._token_route_queue_templates_by_type: Dict[int, Tuple[object, ...]] = info.get("token_route_queue_templates_by_type")
        self._token_route_type_sequence: List[int] = info.get("token_route_type_sequence")
        self._t_target_place_map: Dict[str, str] = dict(info.get("t_target_place_map") or {})
        self._route_source_target_transport: Dict[Tuple[str, str], str] = info.get("route_source_target_transport")
        self._t_route_code_by_idx: List[int] = [
            int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name
        ]
        self._t_code_to_place: Dict[int, str] = {}
        for t_name, t_code in self._t_route_code_map.items():
            if not str(t_name).startswith("t_"):
                continue
            if int(t_code) < 0:
                continue
            target = self._t_target_place_map.get(str(t_name))
            if target is None and str(t_name).startswith("t_"):
                parts = str(t_name).split("_")
                target = parts[-1] if len(parts) >= 2 else str(t_name)[2:]
            if target is not None:
                self._t_code_to_place[int(t_code)] = str(target)
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self.P = info["P"]
        self.T = info["T"]
        # 预计算的 pre/pst 库所索引与运输位索引（构网返回或本地计算以兼容旧版）
        self._pre_place_indices: List[np.ndarray] = info["pre_place_indices"]
        self._pst_place_indices: List[np.ndarray] = info["pst_place_indices"]
        self._transport_pre_place_idx: List[int] = info["transport_pre_place_idx"]
        self._fixed_topology: bool = bool(info.get("fixed_topology", False))

        # ====== 9) Episode 状态与统计容器 ======
        self.marks: List[Place] = self._clone_marks(info["marks"])
        self.ori_marks = self._clone_marks(self.marks)

        self.time = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 50
        self.entered_wafer_count = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log: List[Dict[str, Any]] = []
        self._per_wafer_reward = 0.0
        self._qtime_violated_tokens: Set[int] = set()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._place_use_count: Dict[str, int] = {
            str(place.name): 0 for place in self.marks
        }
        self._u_transition_by_source: Dict[str, int] = {}
        self._u_transition_by_source_transport: Dict[Tuple[str, str], int] = {}
        self._t_transitions_by_transport: Dict[str, List[int]] = {}
        self._tm2_transition_indices: List[int] = []
        self._tm3_transition_indices: List[int] = []

        # ====== 10) 清洗状态与节拍缓存 ======
        self._last_deadlock = False
        self._init_cleaning_state()
        takt_payload = dict(info.get("takt_payload") or {})
        self._takt_result_by_type: Dict[int, Optional[Dict[str, Any]]] = {
            int(k): v
            for k, v in dict(takt_payload.get("takt_result_by_type") or {}).items()
        }
        self._takt_result: Optional[Dict[str, Any]] = takt_payload.get("takt_result_default")
        if self._takt_result is None:
            self._takt_result = self._takt_result_by_type.get(1)
        if self._takt_result is None and self._takt_result_by_type:
            self._takt_result = next(iter(self._takt_result_by_type.values()))
        self._last_u_LP_fire_time: int = 0
        self._u_LP_release_count: int = 0
        all_types = set(self._wafer_type_to_subpath.keys()) | set(self._takt_result_by_type.keys())
        if not all_types:
            all_types = {1}
        self._u_LP_release_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        # ====== 11) 训练/性能与奖励开关 ======
        self._training = True
        self._last_state_scan: Dict[str, Any] = {}
        # ====== 12) 观测缓存与索引重建 ======
        self._ready_chambers_set: frozenset = frozenset(self._ready_chambers)
        self._place_by_name: Dict[str, Place] = {}
        self._obs_place_names: List[str] = []
        self._obs_places: List[Place] = []
        self._obs_offsets: List[int] = []
        self._obs_specs: List[Dict[str, Any]] = []
        self.obs_dim: int = 0
        self._obs_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._obs_return_copy: bool = True
        self._place_by_name = {p1.name: p1 for p1 in self.marks}
        self._lp_done = self._place_by_name.get("LP_done")
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._init_obs_cache()
        self._build_transition_index()
        if not self._training:
            print(self._takt_result)

    def train(self):
        """训练模式"""
        self._training = True

    def eval(self):
        """评估模式"""
        self._training = False

    def step(
        self,
        a1=None,
        a2=None,
        detailed_reward: bool = False,
        wait_duration: Optional[int] = None,
        with_reward: bool = True,
    ):
        """
        单设备 / 并发一步推进入口。
        返回：(done, reward_result, scrap, action_mask, obs)
        """
        SCRAPE = False
        self._last_deadlock = False
        _mask_start = int(self.T)
        _mask_n = _mask_start + len(self.wait_durations)
        _lp_done = self._lp_done

        if self.time >= self.MAX_TIME:
            timeout_reward = {"total": -100.0, "timeout": True} if detailed_reward else -100.0
            action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
            obs = self.get_obs()
            return True, timeout_reward, True, action_mask, obs

        transitions: List[int] = []
        if a1 is not None:
            transitions.append(int(a1))
        if a2 is not None:
            transitions.append(int(a2))
        do_wait = (wait_duration is not None) or (len(transitions) == 0)
        scan_info: Dict[str, Any] = {}
        log_entry: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None
        t1 = self.time

        if do_wait:
            requested_wait = (
                int(wait_duration)
                if wait_duration is not None
                else int(self.wait_durations[0] if self.wait_durations else 5)
            )
            episode_finished = len(_lp_done.tokens) >= self.n_wafer
            if episode_finished and requested_wait > 5:
                actual_dt = 5
            elif requested_wait == 5:
                actual_dt = requested_wait
            else:
                next_event_delta = self.get_next_event_delta()
                if next_event_delta is None:
                    actual_dt = requested_wait
                elif next_event_delta <= 0:
                    actual_dt = min(requested_wait, 5)
                else:
                    actual_dt = min(requested_wait, next_event_delta)

            t2 = t1 + actual_dt
            reward_result, scan_info = self._advance_and_compute_reward(
                actual_dt, t1, t2, detailed=detailed_reward)
            self._consecutive_wait_time += (t2 - t1)

            if self._consecutive_wait_time >= self.idle_timeout and not self._idle_penalty_applied:
                self._idle_penalty_applied = True
                if detailed_reward:
                    reward_result["idle_timeout_penalty"] = -float(self.idle_event_penalty)
                    reward_result["total"] -= float(self.idle_event_penalty)
                else:
                    reward_result -= float(self.idle_event_penalty)
        else:
            self._consecutive_wait_time = 0
            swap_indices = {t_idx for t_idx in transitions if self._will_swap(int(t_idx))}
            action_duration = self.swap_duration if swap_indices else self.ttime
            t2 = t1 + action_duration
            reward_result, scan_info = self._advance_and_compute_reward(
                action_duration, t1, t2, detailed=detailed_reward)
            log_entry = self._fire(
                transitions,
                start_time=t1,
                end_time=t2,
                swap_indices=swap_indices,
            )
            if isinstance(log_entry, list):
                self.fire_log.extend(log_entry)
            elif log_entry is not None:
                self.fire_log.append(log_entry)

            if self._per_wafer_reward > 0:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] += self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(_lp_done.tokens) >= self.n_wafer
        scan = scan_info if isinstance(scan_info, dict) else {}
        is_scrap = bool(scan.get("is_scrap", False))
        scrap_info = scan.get("scrap_info")
        if "is_scrap" not in scan:
            is_scrap, scrap_info = self._check_scrap()
        # 非 WAIT 场景下，若本步 u_* 已取走同一 resident wafer，则撤销本步 scrap。
        if self._should_cancel_resident_scrap_after_fire(scan=scan, log_entry=log_entry):
            is_scrap = False
            scrap_info = None
            scan["is_scrap"] = False
            scan["scrap_info"] = None
            self._last_state_scan = scan
        if is_scrap:
            self.scrap_count += 1
            self.resident_violation_count += 1
            if detailed_reward:
                reward_result["scrap_penalty"] -= float(self.scrap_event_penalty)
                reward_result["total"] -= float(self.scrap_event_penalty)
                reward_result["scrap_info"] = scrap_info
            else:
                reward_result -= float(self.scrap_event_penalty)
            action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
            obs = self.get_obs()
            return True, reward_result, True, action_mask, obs

        if finish:
            if detailed_reward:
                reward_result["finish_bonus"] += float(self.finish_event_reward)
                reward_result["total"] += float(self.finish_event_reward)
            else:
                reward_result += float(self.finish_event_reward)
            SCRAPE = False

        action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
        obs = self.get_obs()
        return bool(finish), reward_result, SCRAPE, action_mask, obs

    def get_enable_t(self) -> Tuple[List[int], List[int]]:
        mask = self.get_action_mask(
            wait_action_start=int(self.T),
            n_actions=int(self.T + len(self.wait_durations)),
        )
        tm2_enabled = [t_idx for t_idx in self._tm2_transition_indices if bool(mask[t_idx])]
        tm3_enabled = [t_idx for t_idx in self._tm3_transition_indices if bool(mask[t_idx])]
        return tm2_enabled, tm3_enabled

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self._place_by_name = {p1.name: p1 for p1 in self.marks}
        self._lp_done = self._place_by_name.get("LP_done")
        self._init_obs_cache()
        self.time = 0
        self.entered_wafer_count = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log.clear()
        self._per_wafer_reward = 0.0
        self._qtime_violated_tokens.clear()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._place_use_count = {str(place.name): 0 for place in self.marks}
        self._last_deadlock = False
        self._init_cleaning_state()
        self._last_u_LP_fire_time = 0
        self._u_LP_release_count = 0
        all_types = set(self._wafer_type_to_subpath.keys()) | set(self._takt_result_by_type.keys())
        if not all_types:
            all_types = {1}
        self._u_LP_release_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        for idx, p in enumerate(self.marks):
            if p.name not in self._mask_skip_places:
                p.capacity = 1
                self.k[idx] = 1
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._last_state_scan = {}
        T = int(self.T)
        mask = self.get_action_mask(wait_action_start=T, n_actions=T + len(self.wait_durations))
        enabled_t = sorted(i for i in range(T) if bool(mask[i]))
        return None, enabled_t

    def _get_obs_place_order(self) -> List[str]:
        """返回观测顺序：LP1/LP2 + 运输位 + 腔室。"""
        tm_names = ["TM2", "TM3"]
        tm_names = [n for n in tm_names if n in self.id2p_name]
        candidates = list(self.chambers)
        if "LLC" not in candidates:
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
        lp_names = [n for n in self._load_port_names if n in self.id2p_name]
        return lp_names + tm_names + chambers

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
        self.obs_dim = int(cursor)
        self._obs_buffer = np.zeros(self.obs_dim, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        if self.obs_dim == 0:
            return np.zeros(0, dtype=np.float32)
        self._update_ll_direction_obs_flags()
        buffer = self._obs_buffer
        buffer[:] = 0.0
        for place, offset in zip(self._obs_places, self._obs_offsets):
            place.write_obs_fast(buffer, offset)
        if self._obs_return_copy:
            return buffer.copy()
        return buffer

    def _update_ll_direction_obs_flags(self) -> None:
        """
        方向 one-hot 临时禁用：LLC/LLD 的 in/out 始终置 0。
        """
        for ll_name in ("LLC", "LLD"):
            place = self._place_by_name.get(ll_name)
            if place is None or not hasattr(place, "set_direction_flags"):
                continue
            place.set_direction_flags(False, False)

    def _advance_and_compute_reward(self,dt: int, t1: int, t2: int, detailed: bool = False) -> tuple:
        """
        单次 marks 遍历完成：reward 计算 + stay_time 推进 + 清洗/idle 推进 + scrap/qtime 检测。
        排序：reward 使用推进前 stay_time，scrap/qtime 使用推进后 stay_time。
        返回 (reward_result, scan_info)。
        """
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        _training = self._training

        total_reward = 0.0
        if detailed:
            parts = {
                "time_cost": 0.0, "proc_reward": 0.0, "safe_reward": 0.0,
                "warn_penalty": 0.0, "penalty": 0.0, "wafer_done_bonus": 0.0,
                "finish_bonus": 0.0, "scrap_penalty": 0.0,
            }

        tc = -float(safe_dt * self.time_coef_penalty)
        total_reward += tc
        if detailed:
            parts["time_cost"] = tc

        is_scrap = False
        scrap_info: Optional[Dict[str, Any]] = None
        qtime_new_violations: List[int] = []
        qtime_limit = self.D_Residual_time
        p_residual = self.P_Residual_time
        proc_coef = self.processing_coef_reward
        transport_coef = self.transport_overtime_coef_penalty
        warn_coef = self.warn_coef_penalty
        check_qtime = not _training
        _SOURCE = SOURCE

        for p in self.marks:
            p_type = p.type
            tokens = p.tokens
            has_tok = len(tokens) > 0

            if has_tok and p_type == CHAMBER:
                head = tokens[0]
                head_stay = head.stay_time
                proc_time = p.processing_time
                if proc_time > 0:
                    remain = proc_time - head_stay
                    if remain < 0:
                        remain = 0
                    progress = safe_dt if safe_dt < remain else remain
                    r = proc_coef * float(progress)
                    total_reward += r
                    if detailed:
                        parts["proc_reward"] += r

                left = proc_time + p_residual - head_stay
                if left <= p_residual:
                    r = -(warn_coef * safe_dt)
                    total_reward += r
                    if detailed:
                        parts["warn_penalty"] += r
            elif has_tok and p_type == DELIVERY_ROBOT:
                for tok in tokens:
                    deadline = tok.enter_time + qtime_limit
                    over_start = t1 if t1 > deadline else deadline
                    if t2 > over_start:
                        r = -float(t2 - over_start) * float(transport_coef)
                        total_reward += r
                        if detailed:
                            parts["penalty"] += r

            if p_type != _SOURCE and safe_dt > 0:
                for tok in tokens:
                    tok.stay_time += safe_dt
            elif p_type == _SOURCE and safe_dt > 0 and p.name in self._load_port_names:
                # 装载口上仅推进负 stay_time（节拍倒计时），不累计正驻留时间。
                for tok in tokens:
                    if tok.stay_time < 0:
                        tok.stay_time = min(0, int(tok.stay_time) + safe_dt)

            if safe_dt > 0 and p.is_pm:
                if not has_tok:
                    p.idle_time += safe_dt
                else:
                    p.idle_time = 0
                if p.is_cleaning:
                    remaining_clean = p.cleaning_remaining - safe_dt
                    if remaining_clean < 0:
                        remaining_clean = 0
                    p.cleaning_remaining = remaining_clean
                    if remaining_clean == 0:
                        p.is_cleaning = False
                        p.cleaning_reason = ""
                        if not _training:
                            self.fire_log.append({
                                "event_type": "cleaning_end",
                                "time": int(self.time),
                                "chamber": p.name,
                                "processed_wafer_count": p.processed_wafer_count,
                            })

            if has_tok and (not is_scrap) and p_type in (CHAMBER, 5):
                head = tokens[0]
                post_stay = head.stay_time
                remaining_proc = p.processing_time - post_stay
                resident_limit = (
                    p_residual * 3 if p.name in {"LLC", "LLD"} else p_residual
                )
                if remaining_proc < -resident_limit:
                    overtime = -remaining_proc - resident_limit
                    scrap_info = {
                        "token_id": head.token_id,
                        "place": p.name,
                        "stay_time": post_stay,
                        "proc_time": p.processing_time,
                        "overtime": overtime,
                        "type": "resident",
                    }
                    is_scrap = True
            if check_qtime and has_tok and p_type == DELIVERY_ROBOT:
                for tok in tokens:
                    tid = tok.token_id
                    if tid < 0 or tid in self._qtime_violated_tokens:
                        continue
                    if tok.stay_time > qtime_limit:
                        qtime_new_violations.append(tid)

        for tid in qtime_new_violations:
            if tid not in self._qtime_violated_tokens:
                self._qtime_violated_tokens.add(tid)
                self.qtime_violation_count += 1

        scan_info: Dict[str, Any] = {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
            "qtime_new_violations": qtime_new_violations,
        }
        self._last_state_scan = scan_info

        if detailed:
            parts["total"] = total_reward
            return parts, scan_info
        return total_reward, scan_info

    def _fire(self, t_idx: int | Sequence[int], start_time: int, end_time: int, is_swap: bool = False, swap_indices: Optional[Set[int]] = None) -> Dict[str, Any] | List[Dict[str, Any]]:
        is_multi = not isinstance(t_idx, (int, np.integer))
        transitions = [int(t_idx)] if not is_multi else [int(idx) for idx in t_idx]
        if not transitions:
            return []

        swap_set = {int(idx) for idx in (swap_indices or set())}
        if is_swap and len(transitions) == 1 and not swap_set:
            swap_set.add(int(transitions[0]))

        def _transport_order(idx: int) -> int:
            transport = self._transition_transport_place(int(idx))
            if transport == "TM2":
                return 0
            if transport == "TM3":
                return 1
            return 2

        transitions.sort(key=_transport_order)
        log_entries: List[Dict[str, Any]] = []

        for current_t_idx in transitions:
            t_name = self.id2t_name[current_t_idx]
            pre_places = self._pre_place_indices[current_t_idx]
            pst_places = self._pst_place_indices[current_t_idx]
            if pre_places.size == 0 or pst_places.size == 0:
                log_entries.append(
                    {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
                )
                continue
            pre_place = self.marks[int(pre_places[0])]
            pst_place = self.marks[int(pst_places[0])]
            if len(pre_place.tokens) == 0:
                log_entries.append(
                    {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
                )
                continue

            pre_place_idx = int(pre_places[0])
            pst_place_idx = int(pst_places[0])

            tok = pre_place.pop_head()
            wafer_id = tok.token_id
            tok.enter_time = self.time
            tok.stay_time = 0

            if t_name.startswith("u_"):
                src = pre_place.name
                tok._dst_level_targets = tuple(self._u_targets.get(src, []))
                tok.last_u_source = str(src)
                tok.machine = int(self._next_robot_machine())
                transport = pst_place.name if pst_place.name in {"TM2", "TM3"} else "TM2"
                tok.machine = 2 if transport == "TM3" else 1
                self._on_processing_unload(src)
            elif t_name.startswith("t_"):
                target = pst_place.name

                if current_t_idx in swap_set and self._is_swap_eligible(pst_place):
                    old_tok = pst_place.pop_head()
                    old_wafer_id = old_tok.token_id

                    self._on_processing_unload(target)

                    old_tok.enter_time = self.time
                    old_tok.stay_time = 0
                    old_tok._dst_level_targets = tuple(self._u_targets.get(target, []))
                    old_tok.machine = tok.machine
                    old_tok.route_head_idx += 1

                    tok._dst_level_targets = None
                    tok.step = max(tok.step, self._step_map.get(target, 0))
                    tok.route_head_idx += 1
                    pst_place.append(tok)
                    tok._place_idx = pst_place_idx
                    self._place_use_count[target] = int(self._place_use_count.get(target, 0)) + 1

                    pre_place.append(old_tok)
                    old_tok._place_idx = pre_place_idx

                    log_entries.append(
                        {
                            "t_name": t_name,
                            "t1": int(start_time),
                            "t2": int(end_time),
                            "token_id": wafer_id,
                            "swap": True,
                            "swapped_token_id": old_wafer_id,
                            "swap_source_place": target,
                        }
                    )
                    continue

                tok._dst_level_targets = None
                tok.step = max(tok.step, self._step_map.get(target, 0))
                if target == "LP_done":
                    done_type = int(getattr(tok, "route_type", 1) or 1)
                    self._entered_wafer_count_by_type[done_type] = max(
                        0, int(self._entered_wafer_count_by_type.get(done_type, 0)) - 1
                    )
                    self.entered_wafer_count = max(0, int(self.entered_wafer_count) - 1)
                    self.done_count += 1
                    self._per_wafer_reward += float(self.done_event_reward)

            tok.route_head_idx += 1
            pst_place.append(tok)
            tok._place_idx = pst_place_idx
            if t_name.startswith("t_"):
                self._place_use_count[pst_place.name] = int(self._place_use_count.get(pst_place.name, 0)) + 1
            self.m[pre_place_idx] -= 1
            self.m[pst_place_idx] += 1
            if t_name.startswith("u_") and pre_place.name in self._load_port_names:
                released_type = int(getattr(tok, "route_type", 1) or 1)
                self.entered_wafer_count += 1
                self._entered_wafer_count_by_type[released_type] = (
                    int(self._entered_wafer_count_by_type.get(released_type, 0)) + 1
                )
                self._last_u_LP_fire_time = int(start_time)
                self._u_LP_release_count += 1
                self._u_LP_release_count_by_type[released_type] = (
                    int(self._u_LP_release_count_by_type.get(released_type, 0)) + 1
                )
                self._arm_lp_head_with_takt_delay(released_type)
            log_ret: Dict[str, Any] = {
                "t_name": t_name,
                "t1": int(start_time),
                "t2": int(end_time),
                "token_id": wafer_id,
            }
            if t_name.startswith("u_"):
                log_ret["source_place"] = pre_place.name
            log_entries.append(log_ret)
        if not is_multi:
            return log_entries[0]
        return log_entries

    def _should_cancel_resident_scrap_after_fire(self, scan: Dict[str, Any], log_entry: Optional[Dict[str, Any]]) -> bool:
        if isinstance(log_entry, list):
            for item in log_entry:
                if self._should_cancel_resident_scrap_after_fire(scan=scan, log_entry=item):
                    return True
            return False
        if not isinstance(scan, dict) or not isinstance(log_entry, dict):
            return False
        if not bool(scan.get("is_scrap", False)):
            return False
        scrap_info = scan.get("scrap_info")
        if not isinstance(scrap_info, dict):
            return False
        if scrap_info.get("type") != "resident":
            return False
        t_name = str(log_entry.get("t_name", ""))
        is_u_transition = bool(t_name.startswith("u_"))
        is_swap_transition = bool(log_entry.get("swap", False))
        if not is_u_transition and not is_swap_transition:
            return False
        if is_u_transition:
            _sp = log_entry.get("source_place")
            source_name = str(_sp) if _sp is not None else t_name[2:]
        else:
            source_name = str(log_entry.get("swap_source_place", ""))
        place_match = bool(scrap_info.get("place") == source_name)
        try:
            if is_u_transition:
                fired_token_id = int(log_entry.get("token_id", -1))
            else:
                fired_token_id = int(log_entry.get("swapped_token_id", -1))
            scrap_token_id = int(scrap_info.get("token_id", -2))
        except (TypeError, ValueError):
            return False
        token_match = bool(fired_token_id >= 0 and fired_token_id == scrap_token_id)
        decision = bool(place_match and token_match)
        return decision

    def _token_next_target(self, tok: BasedToken) -> Optional[str]:
        """从 token 的 route_queue 推断下一个目标腔室（用于 u_* 选择）。"""
        queue = tok.route_queue
        if not queue:
            return None
        idx = tok.route_head_idx + 1
        n = len(queue)
        while idx < n:
            gate = queue[idx]
            if gate == -1:
                idx += 1
                continue
            if isinstance(gate, int):
                return self._t_code_to_place.get(int(gate))
            if isinstance(gate, (tuple, list, set, frozenset)) and gate:
                first = next(iter(gate))
                return self._t_code_to_place.get(int(first))
            idx += 1
        return None

    def _gate_targets_from_tok_gate(self, tok_gate: object) -> Tuple[str, ...]:
        if tok_gate == -1:
            return tuple()
        raw_codes: List[int] = []
        if isinstance(tok_gate, int):
            raw_codes.append(int(tok_gate))
        elif isinstance(tok_gate, (tuple, frozenset, list, set)):
            for item in tok_gate:
                if isinstance(item, int):
                    raw_codes.append(int(item))
        targets: List[str] = []
        seen: Set[str] = set()
        for code in raw_codes:
            name = self._t_code_to_place.get(int(code))
            if not isinstance(name, str):
                continue
            if name in seen:
                continue
            seen.add(name)
            targets.append(name)
        return tuple(targets)

    def _stage_targets_for_candidates(self, candidates: Sequence[str], tok_gate: object) -> Tuple[str, ...]:
        candidate_targets = tuple(str(x) for x in candidates)
        if not candidate_targets:
            return tuple()
        gate_targets = self._gate_targets_from_tok_gate(tok_gate)
        if not gate_targets:
            return candidate_targets
        gate_set = set(gate_targets)
        filtered = tuple(t for t in candidate_targets if t in gate_set)
        return filtered if filtered else candidate_targets

    def _select_min_use_count_target(
        self,
        candidates: Sequence[str],
        tok_gate: object,
        cache: Optional[Dict[Tuple[int, Tuple[str, ...]], str]] = None,
        cache_key: Optional[Tuple[int, Tuple[str, ...]]] = None,
    ) -> Optional[str]:
        stage_targets = self._stage_targets_for_candidates(candidates, tok_gate)
        if not stage_targets:
            return None
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return str(cached)
        counts = [int(self._place_use_count.get(str(name), 0)) for name in stage_targets]
        min_count = min(counts)
        min_targets = [str(name) for name, cnt in zip(stage_targets, counts) if cnt == min_count]
        pick_idx = int(np.random.randint(0, len(min_targets)))
        picked = str(min_targets[pick_idx])
        if cache is not None and cache_key is not None:
            cache[cache_key] = picked
        return picked

    @staticmethod
    def _route_gate_allows_t(gate: object, t_code: int) -> bool:
        if t_code < 0:
            return True
        if gate == -1:
            return True
        if isinstance(gate, int):
            return gate == t_code
        if isinstance(gate, (tuple, frozenset)):
            return t_code in gate
        if isinstance(gate, (list, set)):
            return t_code in gate
        return True

    def _allow_t_by_use_count(
        self,
        transport_name: str,
        target_name: str,
        tok_gate: object,
        tok: Optional[BasedToken] = None,
        selected_cache: Optional[Dict[Tuple[int, Tuple[str, ...]], str]] = None,
    ) -> bool:
        if not isinstance(tok_gate, (tuple, frozenset)):
            return True
        _ = transport_name
        dst_level_targets = tuple(getattr(tok, "_dst_level_targets", ()) or ())
        if not dst_level_targets:
            dst_level_targets = self._gate_targets_from_tok_gate(tok_gate)
        cache_key = (int(id(tok)), tuple(str(x) for x in dst_level_targets))
        selected = self._select_min_use_count_target(
            candidates=dst_level_targets,
            tok_gate=tok_gate,
            cache=selected_cache,
            cache_key=cache_key,
        )
        if selected is None:
            return False
        return str(target_name) == str(selected)

    def _lp_type_head_tokens(self) -> Dict[int, BasedToken]:
        heads: Dict[int, BasedToken] = {}
        for lp_name in self._load_port_names:
            lp_place = self._place_by_name.get(lp_name)
            if lp_place is None:
                continue
            for tok in lp_place.tokens:
                t_id = int(getattr(tok, "route_type", 1) or 1)
                if t_id not in heads:
                    heads[t_id] = tok
        return heads

    def _allow_start_for_route_type(self, route_type: int) -> bool:
        """双子路径：route_type 1/2 分别受 max_wafers1_in_system / max_wafers2_in_system 约束。"""
        if not self._multi_subpath:
            return True
        type_id = int(route_type)
        if type_id == 1:
            cap = int(self.max_wafers1_in_system)
        elif type_id == 2:
            cap = int(self.max_wafers2_in_system)
        else:
            raise RuntimeError(f"unsupported route_type for WIP cap: {type_id}")
        current = int(self._entered_wafer_count_by_type.get(type_id, 0))
        return int(current + 1) <= cap

    def _takt_required_interval(self, route_type: Optional[int] = None) -> Optional[int]:
        """
        返回下一次允许 u_LP 发片的最小间隔（秒）。

        口径：
        - 首片（release_count=0）不门控，返回 None
        - 第 2 片（release_count=1）使用 cycle[0]（100 拍的第 1 个拍子）
        - 第 3 片起（release_count>=2）按序推进（idx=1,2,3...），必要时对 100 取模
        """
        type_id = int(route_type if route_type is not None else 1)
        policy = str(self._takt_policy or "").strip().lower()
        if policy == "split_by_subpath":
            takt = self._takt_result_by_type.get(type_id)
            release_count = int(self._u_LP_release_count_by_type.get(type_id, 0))
        else:
            # shared / 默认：所有类型共用同一条 takt_cycle，索引按全局发片计数推进。
            takt = self._takt_result_by_type.get(type_id)
            if takt is None and self._takt_result_by_type:
                takt = next(iter(self._takt_result_by_type.values()))
            release_count = int(self._u_LP_release_count)
        if not takt:
            return None
        if release_count <= 0:
            return None

        cycle_takts = takt.get("cycle_takts") or []
        if not cycle_takts:
            return None
        cycle_len = int(takt.get("cycle_length") or len(cycle_takts))
        if cycle_len <= 0:
            return None

        if release_count == 1:
            idx = 0
        else:
            idx = (release_count - 1) % cycle_len

        required = cycle_takts[idx]
        if isinstance(required, float):
            required = int(round(required))
        else:
            required = int(required)
        if required < 0:
            required = 0
        return required

    def _arm_lp_head_with_takt_delay(self, route_type: Optional[int]) -> None:
        """
        每次 u_LP 发射后，仅给新的 LP 队首 token 写入节拍倒计时（负 stay_time）。
        这样下一片的等待起点是“上一片实际发射时刻”，可严格保证发片间隔。
        """
        type_id = int(route_type if route_type is not None else 1)
        required = self._takt_required_interval(type_id)
        if required is None:
            heads = self._lp_type_head_tokens()
            for head in heads.values():
                if int(head.stay_time) < 0:
                    head.stay_time = 0
            return
        required_int = max(0, int(required))
        policy = str(self._takt_policy or "").strip().lower()
        heads = self._lp_type_head_tokens()
        if policy == "shared":
            for head in heads.values():
                head.stay_time = -required_int
        else:
            head = heads.get(type_id)
            if head is None:
                return
            head.stay_time = -required_int

    def _build_transition_index(self) -> None:
        self._u_transition_by_source = {}
        self._u_transition_by_source_transport = {}
        self._t_transitions_by_transport = {}
        self._tm2_transition_indices = []
        self._tm3_transition_indices = []
        for t_idx, t_name in enumerate(self.id2t_name):
            transport_name: Optional[str] = None
            if t_name.startswith("u_"):
                pre_idx = self._pre_place_indices[t_idx]
                pst_idx = self._pst_place_indices[t_idx]
                if pre_idx.size >= 1 and pst_idx.size >= 1:
                    src = self.id2p_name[int(pre_idx[0])]
                    dst = self.id2p_name[int(pst_idx[0])]
                    if dst in {"TM2", "TM3"}:
                        self._u_transition_by_source_transport[(src, dst)] = int(t_idx)
                        transport_name = str(dst)
                    self._u_transition_by_source[src] = int(t_idx)
            elif t_name.startswith("t_"):
                transport = self._transition_transport_place(int(t_idx))
                if transport is None:
                    continue
                self._t_transitions_by_transport.setdefault(transport, []).append(int(t_idx))
                transport_name = str(transport)
            if transport_name == "TM2":
                self._tm2_transition_indices.append(int(t_idx))
            elif transport_name == "TM3":
                self._tm3_transition_indices.append(int(t_idx))

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

    def _transition_target_place(self, t_idx: int) -> Optional[str]:
        """根据变迁后置库所索引返回目标库所名，避免依赖 t_* 命名格式。"""
        if t_idx < 0 or t_idx >= len(self._pst_place_indices):
            return None
        pst_idx = self._pst_place_indices[int(t_idx)]
        if pst_idx.size == 0:
            return None
        return str(self.id2p_name[int(pst_idx[0])])

    def _transition_transport_place(self, t_idx: int) -> Optional[str]:
        """根据变迁前置库所索引返回运输位库所名。"""
        if t_idx < 0 or t_idx >= len(self._pre_place_indices):
            return None
        pre_idx = self._pre_place_indices[int(t_idx)]
        for p_i in pre_idx:
            p_name = str(self.id2p_name[int(p_i)])
            if p_name in {"TM2", "TM3"}:
                return p_name
        return None

    def _transport_for_t_target(self, source: str, target: str) -> str:
        """按当前 route 的 hop 映射选择 source->target 的 transport。"""
        mapped = self._route_source_target_transport.get((str(source), str(target)))
        if mapped:
            return str(mapped)
        if target in {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6"}:
            return "TM3"
        return "TM2"

    def _init_cleaning_state(self) -> None:
        for p in self.marks:
            if not p.name.startswith("PM"):
                continue
            p.processed_wafer_count = p.processed_wafer_count
            p.idle_time = p.idle_time
            p.last_proc_type = p.last_proc_type
            p.is_cleaning = p.is_cleaning
            p.cleaning_remaining = p.cleaning_remaining
            p.cleaning_reason = p.cleaning_reason

    def get_next_event_delta(self) -> Optional[int]:
        """
        计算当前时刻到下一个关键事件的时间差（秒）。
        遍历 marks 中运输位与加工腔室的 token。
        """
        best = None
        t_transport = self.ttime
        _CHAMBER_TYPES = (CHAMBER, 5)
        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            if place.is_dtm or place.name in {"TM2", "TM3"}:
                for tok in tokens:
                    delta = t_transport - tok.stay_time
                    if delta < 0:
                        delta = 0
                    if best is None or delta < best:
                        best = delta
            elif place.type in _CHAMBER_TYPES:
                ptime = place.processing_time
                if ptime > 0:
                    head = tokens[0]
                    delta = ptime - head.stay_time
                    if delta < 0:
                        delta = 0
                    if best is None or delta < best:
                        best = delta
        lp_heads = self._lp_type_head_tokens()
        if lp_heads:
            deltas: List[int] = []
            for tok in lp_heads.values():
                if int(tok.stay_time) < 0:
                    deltas.append(-int(tok.stay_time))
                else:
                    deltas.append(0)
            if deltas:
                delta_takt = min(deltas)
                if best is None or delta_takt < best:
                    best = delta_takt
        return best

    def _on_processing_unload(self, source_name: str) -> None:
        if not self._cleaning_enabled:
            return
        trigger = self._cleaning_trigger_map.get(source_name, 0)
        if trigger <= 0:
            return
        source_place = self._get_place(source_name)
        source_place.processed_wafer_count = source_place.processed_wafer_count + 1
        source_place.last_proc_type = source_name
        if source_place.is_cleaning:
            return
        if source_place.processed_wafer_count >= trigger:
            count = int(source_place.processed_wafer_count)
            duration = int(self._cleaning_duration_map.get(source_name, 0))
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

    def _is_swap_eligible(self, pst_place: Place) -> bool:
        """目标 PM 是否可执行 swap（仅在 _fire 中调用）。"""
        if not self._dual_arm:
            return False
        if not pst_place.is_pm:
            return False
        if len(pst_place.tokens) < pst_place.capacity:
            return False
        if not pst_place.tokens:
            return False
        if pst_place.tokens[0].stay_time < pst_place.processing_time:
            return False
        return not pst_place.is_cleaning

    def _will_swap(self, t_idx: int) -> bool:
        """判断 t_idx 变迁当前是否会触发 swap（用于 step 计算时长）。"""
        if not self._dual_arm:
            return False
        t_name = self.id2t_name[t_idx]
        if not t_name.startswith("t_"):
            return False
        pst_idx = self._pst_place_indices[t_idx]
        if pst_idx.size == 0:
            return False
        pst_place = self.marks[int(pst_idx[0])]
        return self._is_swap_eligible(pst_place)

    def _is_next_stage_available(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        按 route gate 过滤后的候选集中，优先选择 use_count 最小的目标；
        若并列，随机选择其中一个。单臂满或清洗则不可出片，双臂仅清洗阻塞。
        """
        candidates = tuple(self._u_targets.get(source, ()))
        if not candidates:
            return False, None
        source_place = self._get_place(source)
        head_tok = source_place.tokens[0] if len(source_place.tokens) > 0 else None
        tok_gate = head_tok.route_queue[head_tok.route_head_idx] if head_tok is not None else -1
        target_name = self._select_min_use_count_target(candidates, tok_gate)
        if target_name is None:
            return False, None
        target_place = self._get_place(target_name)
        if target_place.is_cleaning:
            return False, None
        if not self._dual_arm and len(target_place.tokens) >= target_place.capacity:
            return False, None
        return True, target_name

    def _next_robot_machine(self) -> int:
        return 1

    def _check_scrap(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        for p in self.marks:
            if p.type not in (CHAMBER, 5):
                continue
            resident_limit = (
                self.P_Residual_time * 1 if p.name in {"LLC", "LLD"} else self.P_Residual_time
            )
            for tok in p.tokens:
                remaining = p.processing_time - tok.stay_time
                if remaining < -resident_limit:
                    overtime = -remaining - resident_limit
                    return True, {
                        "token_id": tok.token_id,
                        "place": p.name,
                        "stay_time": tok.stay_time,
                        "proc_time": p.processing_time,
                        "overtime": overtime,
                        "type": "resident",
                    }
        return False, None

    _MASK_TIMED_TYPES = frozenset((CHAMBER, 5, DELIVERY_ROBOT))

    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """
        返回完整离散动作掩码（transition + wait）。
        遍历 marks，内联 remaining_time / has_ready_chamber 检查。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)
        total_actions = int(
            n_actions if n_actions is not None else (start + len(self.wait_durations))
        )
        mask = np.zeros(total_actions, dtype=bool)
        struct_enabled_cache: Dict[int, bool] = {}
        selected_parallel_target_cache: Dict[Tuple[int, Tuple[str, ...]], str] = {}

        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(t_idx)
            if cached is not None:
                return cached
            result = not bool((self.m[int(self._pre_place_indices[t_idx][0])] < 1 or
                      self.m[int(self._pst_place_indices[t_idx][0])] + 1 >
                               self.k[int(self._pst_place_indices[t_idx][0])]))
            struct_enabled_cache[t_idx] = result
            return result

        # LP 出片：单路线用全局 WIP（max_wafers1）；双子路径仅按类型上限（_allow_start_for_route_type）+ 队首节拍就绪；可同时允许多条 u_LP*。
        if self._multi_subpath or int(self.entered_wafer_count) < int(self.max_wafers1_in_system):
            for lp_name in self._load_port_names:
                lp_place = self._place_by_name.get(lp_name)
                if lp_place is None or len(lp_place.tokens) == 0:
                    continue
                head = lp_place.tokens[0]
                if int(head.stay_time) < 0:
                    continue
                route_type = int(getattr(head, "route_type", 1) or 1)
                expected_lp = self._wafer_type_to_load_port.get(route_type, "LP1")
                if expected_lp != lp_name:
                    raise RuntimeError(
                        f"wafer_type {route_type} maps to load port {expected_lp} but queue head is on {lp_name}"
                    )
                if not self._allow_start_for_route_type(route_type):
                    continue
                lp_target = self._token_next_target(head)
                if lp_target is None:
                    continue
                lp_transport = self._transport_for_t_target(lp_name, str(lp_target))
                u_lp_idx = self._u_transition_by_source_transport.get((lp_name, lp_transport))
                if u_lp_idx is not None and _is_struct_enabled(int(u_lp_idx)):
                    t_idx = int(u_lp_idx)
                    if 0 <= t_idx < total_actions:
                        mask[t_idx] = True

        has_ready_chamber = False
        ready_chambers = self._ready_chambers_set
        skip = self._mask_skip_places
        timed = self._MASK_TIMED_TYPES
        t_trans_by_transport = self._t_transitions_by_transport
        u_trans_by_source = self._u_transition_by_source
        route_code_by_idx = self._t_route_code_by_idx
        route_gate_allows = self._route_gate_allows_t
        place_by_name = self._place_by_name

        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            pname = place.name
            if pname in skip:
                continue
            p_type = place.type
            proc_time = place.processing_time

            is_timed = p_type in timed

            if place.is_dtm or place.name in {"TM2", "TM3"}:
                for tok in tokens:
                    if is_timed and proc_time > 0 and tok.stay_time < proc_time:
                        continue
                    tok_gate = tok.route_queue[tok.route_head_idx]
                    for t_idx in t_trans_by_transport.get(pname, ()):
                        target = self._transition_target_place(int(t_idx))
                        if target is None:
                            continue
                        allow_rr = self._allow_t_by_use_count(
                            pname,
                            target,
                            tok_gate,
                            tok,
                            selected_parallel_target_cache,
                        )
                        if not allow_rr:
                            continue
                        if not route_gate_allows(tok_gate, route_code_by_idx[t_idx]):
                            continue
                        target_place = place_by_name.get(target)
                        if target_place is None or target_place.is_cleaning:
                            continue
                        if self._dual_arm and target_place.is_pm:
                            if len(target_place.tokens) > 0 and target_place.tokens[0].stay_time < target_place.processing_time:
                                continue
                        else:
                            if not _is_struct_enabled(t_idx):
                                continue
                        if 0 <= t_idx < total_actions:
                            mask[t_idx] = True
            else:
                head = tokens[0]
                if is_timed and proc_time > 0 and head.stay_time < proc_time:
                    continue
                available, target = self._is_next_stage_available(source=pname)
                if available and target is not None:
                    transport = self._transport_for_t_target(pname, str(target))
                    u_idx = self._u_transition_by_source_transport.get((pname, transport))
                    if u_idx is None:
                        u_idx = u_trans_by_source.get(pname)
                    struct_enabled = bool(u_idx is not None and _is_struct_enabled(u_idx))
                    if u_idx is not None and struct_enabled and 0 <= u_idx < total_actions:
                        mask[u_idx] = True

            if not has_ready_chamber and p_type == CHAMBER and proc_time > 0 and pname in ready_chambers:
                for tok in tokens:
                    if tok.stay_time >= proc_time:
                        has_ready_chamber = True
                        break

        wait_durations = self.wait_durations
        for offset in range(len(wait_durations)):
            if has_ready_chamber and wait_durations[offset] > 5:
                continue
            idx = start + offset
            if 0 <= idx < total_actions:
                mask[idx] = True

        return mask

    def render_gantt(self, out_path: str, title_suffix: str | None = None) -> None:
        pass
