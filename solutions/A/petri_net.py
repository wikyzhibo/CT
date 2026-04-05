from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from config.cluster_tool.env_config import PetriEnvConfig
from solutions.A.construct import BasedToken
from solutions.A.construct.build_topology import infer_cascade_transport_by_scope
from solutions.A.model_builder import build_net
from solutions.A.deprecated.pn import Place

CHAMBER = 1
DELIVERY_ROBOT = 2
SOURCE = 3


class ClusterTool:
    # 作用：初始化配置、构网结果、状态容器与索引缓存。
    def __init__(self, config: PetriEnvConfig = None, concurrent: bool = False) -> None:
        assert config is not None, "config must be provided"
        self.config = config
        self._concurrent = bool(concurrent)
        self._cached_auto_tm1_action: Optional[int] = None

        # ====== 1) 运行边界与奖励超参 ======
        self.MAX_TIME = config.MAX_TIME
        self.n_wafer = config.n_wafer
        
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
        self.wait_duration: int = int(config.wait_duration)
        
        # ====== 6) 构网输入与构网结果 ======
        self.ttime = 5
        self.single_route_config = config.single_route_config
        self.single_route_name = config.single_route_name
        route_entry = self.single_route_config["routes"][self.single_route_name]
        route_stage_raw = route_entry.get("route_stage")
        self._gantt_route_stages: List[List[str]] = [
            [str(place_name) for place_name in list(stage or [])]
            for stage in list(route_stage_raw or [])
        ]
        self.max_wafers1_in_system = route_entry.get("max_wafer1_in_system",12)
        self.max_wafers2_in_system = route_entry.get("max_wafer2_in_system",0)
        ratio = route_entry.get("ratio",[1,0])
        self.n_wafer1 = int(self.n_wafer * ratio[0] / sum(ratio))
        self.n_wafer2 = int(self.n_wafer - self.n_wafer1)

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
        self.chambers = tuple(route_meta.get("timeline_chambers") or route_meta.get("chambers", ()))
        self._u_targets = dict(route_meta.get("u_targets", {}))
        self._step_map = dict(route_meta.get("step_map", {}))
        self._cleaning_duration_map = route_meta.get("cleaning_duration_map")
        self._cleaning_trigger_map = route_meta.get("cleaning_trigger_wafers_map")
        self._multi_subpath = bool(route_meta.get("multi_subpath", False))
        self._wafer_type_to_subpath: Dict[int, str] = route_meta.get("wafer_type_to_subpath")
        self._takt_policy: str = str(route_meta.get("takt_policy", "") or "")
        self._load_port_names: Tuple[str, ...] = route_meta.get("load_port_names")
        self._wafer_type_to_release_place: Dict[int, str] = route_meta.get("wafer_type_to_release_place") or {}
        self._release_control_places: Tuple[str, ...] = tuple(route_meta.get("release_control_places") or self._load_port_names)
        self._mask_skip_places: frozenset[str] = frozenset({"LP_done"})
        _raw_cycle = route_entry.get("cycle_type") if str(self._takt_policy or "").strip().lower() == "shared" else None
        _valid_types = set(int(t) for t in self._wafer_type_to_subpath.keys())
        _filtered = tuple(int(t) for t in (_raw_cycle or []) if int(t) in _valid_types)
        self._cycle_type_enabled: bool = bool(_filtered)
        self._cycle_type: Tuple[int, ...] = _filtered
        self._cycle_type_idx: int = 0
        self._lp_pick_cycle_idx: int = 0

        # ====== 8) Petri 静态结构索引 ======
        self.m0: np.ndarray = info["m0"]
        self.m: np.ndarray = self.m0.copy()
        self.k: np.ndarray = np.array(info["capacity"], dtype=int)
        self.id2p_name: List[str] = info["id2p_name"]
        self.id2t_name: List[str] = info["id2t_name"]
        # t_* 变迁名字 -> 变迁的颜色编号
        self._t_route_code_map: Dict[str, int] = dict(info.get("t_route_code_map") or {})
        # t_* 变迁将晶圆装入的库所（腔室）
        self._t_target_place_map: Dict[str, str] = dict(info.get("t_target_place_map") or {})
        # （源腔室，目标腔室）-> 负责的机器手
        self._route_source_target_transport: Dict[Tuple[str, str], str] = info.get("route_source_target_transport")
        # 变迁的编号 -> 颜色编号，u_* 变迁颜色固定为-1
        self._t_route_code_by_idx: List[int] = [int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name]
        self._t_code_to_place: Dict[int, str] = {}
        # t_* 颜色编号 -> 目标腔室
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

        # ====== 9) Episode 状态与统计容器 ======
        self.marks: List[Place] = self._clone_marks(info["marks"])
        self.ori_marks = self._clone_marks(self.marks)

        self.time = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 50
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
        self._place_use_count: Dict[str, int] = {str(place.name): 0 for place in self.marks}
        self._eval_gantt_place_to_sm: Dict[str, Tuple[int, int, int]] = {}
        self._eval_gantt_lane_cursor: Dict[str, int] = {}
        self._eval_gantt_slots: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._eval_gantt_closed_ops: List[Dict[str, Any]] = []
        self._reset_eval_gantt_records()
        self._u_transition_by_source: Dict[str, int] = {}
        self._u_transition_by_source_transport: Dict[Tuple[str, str], int] = {}
        self._t_transitions_by_transport: Dict[str, List[int]] = {}
        self._tm1_transition_indices: List[int] = []
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
        self._last_u_entry_fire_time: int = 0
        self._u_entry_release_count: int = 0
        all_types = set(self._wafer_type_to_subpath.keys()) | set(self._takt_result_by_type.keys())
        if not all_types:
            all_types = {1}
        self._u_entry_release_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        self._entry_release_ready_time_shared: int = 0
        self._entry_release_ready_time_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        # ====== 11) 训练/性能与奖励开关 ======
        self._training = True
        self._last_state_scan: Dict[str, Any] = {}

        # ====== 12) 观测缓存与索引重建 ======
        self._ready_chambers_set: frozenset = frozenset(self.chambers)
        self._place_by_name: Dict[str, Place] = {}
        self._obs_place_names: List[str] = []
        self.obs_dim: int = 0
        self._obs_return_copy: bool = True
        self._place_by_name = {p1.name: p1 for p1 in self.marks}
        self._lp_done = self._place_by_name.get("LP_done")
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        order = list(self._load_port_names) + ["TM1", "TM2", "TM3"] + list(self.chambers)
        _skip_obs_places = frozenset({"AL", "CL", "TM1", "LP_done"}) | frozenset(
            self._load_port_names
        )
        obs_names = [
            name for name in order if name in self._place_by_name and name not in _skip_obs_places
        ]
        self._obs_place_names = obs_names
        self._obs_places = [self._place_by_name[name] for name in obs_names]
        offsets: List[int] = []
        cursor = 0
        for place in self._obs_places:
            dim = int(place.get_obs_dim())
            offsets.append(cursor)
            cursor += dim
        self._obs_offsets = offsets
        self.obs_dim = int(cursor)
        self._obs_buffer = np.zeros(self.obs_dim, dtype=np.float32)
        self._reset_eval_gantt_records()
        self._build_transition_index()
        if not self._training:
            print(self._takt_result)

    # 作用：切换为训练模式。
    def train(self):
        """训练模式"""
        self._training = True

    # 作用：切换为评估模式。
    def eval(self):
        """评估模式"""
        self._training = False

    # 作用：执行一步仿真推进并返回 done/reward/scrap/mask/obs。
    def step(
        self,
        a1=None,
        a2=None,
        a3=None,
        wait_duration: Optional[int] = None,
    ):
        """
        单设备 / 并发一步推进入口。
        返回：(done, reward, scrap, action_mask, obs)。
        reward：标量 float。
        action_mask：非并发为全量 ndarray；并发为 (mask_tm1, mask_tm2, mask_tm3)。
        """
        # 并发模式下，自动消费 get_action_mask 中缓存的 TM1 动作
        if a1 is None and self._concurrent and self._cached_auto_tm1_action is not None:
            a1 = self._cached_auto_tm1_action
            self._cached_auto_tm1_action = None
        SCRAPE = False
        self._last_deadlock = False
        _mask_start = int(self.T)
        _mask_n = _mask_start + 1
        _lp_done = self._lp_done

        if self.time >= self.MAX_TIME:
            action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
            obs = self.get_obs()
            return (
                True,
                -100.0,
                True,
                action_mask,
                obs,
            )

        transitions: List[int] = []
        if a1 is not None:
            transitions.append(int(a1))
        if a2 is not None:
            transitions.append(int(a2))
        if a3 is not None:
            transitions.append(int(a3))
        do_wait = (wait_duration is not None) or (len(transitions) == 0)
        scan_info: Dict[str, Any] = {}
        log_entry: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None
        t1 = self.time

        if do_wait:
            requested_wait = int(wait_duration) if wait_duration is not None else self.wait_duration
            next_event_delta: Optional[int] = None
            episode_finished = len(_lp_done.tokens) >= self.n_wafer
            if episode_finished and requested_wait > 5:
                actual_dt = 5
            elif requested_wait == 5:
                next_event_delta = self.get_next_event_delta()
                if next_event_delta is None or next_event_delta <= 0:
                    actual_dt = requested_wait
                else:
                    actual_dt = int(next_event_delta)
            else:
                next_event_delta = self.get_next_event_delta()
                if next_event_delta is None:
                    actual_dt = requested_wait
                elif next_event_delta <= 0:
                    actual_dt = min(requested_wait, 5)
                else:
                    actual_dt = min(requested_wait, next_event_delta)

            t2 = t1 + actual_dt
            reward, scan_info = self._advance_and_compute_reward(actual_dt, t1, t2)
            self._consecutive_wait_time += (t2 - t1)

            if self._consecutive_wait_time >= self.idle_timeout and not self._idle_penalty_applied:
                self._idle_penalty_applied = True
                reward -= float(self.idle_event_penalty)
        else:
            self._consecutive_wait_time = 0
            swap_indices = {t_idx for t_idx in transitions if self._will_swap(int(t_idx))}
            action_duration = self.swap_duration if swap_indices else self.ttime
            t2 = t1 + action_duration
            reward, scan_info = self._advance_and_compute_reward(action_duration, t1, t2)
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
                reward += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(_lp_done.tokens) >= self.n_wafer
        scan = scan_info
        is_scrap = bool(scan["is_scrap"])
        scrap_info = scan["scrap_info"]
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
            reward -= float(self.scrap_event_penalty)
            action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
            obs = self.get_obs()
            return (
                True,
                float(reward),
                True,
                action_mask,
                obs,
            )

        if finish:
            reward += float(self.finish_event_reward)
            SCRAPE = False

        action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
        obs = self.get_obs()
        return (
            bool(finish),
            float(reward),
            SCRAPE,
            action_mask,
            obs,
        )

    # 作用：重置网状态、统计计数与入口节拍游标。
    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self._place_by_name = {p1.name: p1 for p1 in self.marks}
        self._lp_done = self._place_by_name.get("LP_done")
        self._obs_places = [self._place_by_name[name] for name in self._obs_place_names]
        self.time = 0
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
        self._cached_auto_tm1_action = None
        self._init_cleaning_state()
        self._last_u_entry_fire_time = 0
        self._u_entry_release_count = 0
        all_types = set(self._wafer_type_to_subpath.keys()) | set(self._takt_result_by_type.keys())
        if not all_types:
            all_types = {1}
        self._u_entry_release_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self._entry_release_ready_time_shared = 0
        self._entry_release_ready_time_by_type = {int(t): 0 for t in sorted(all_types)}
        self._cycle_type_idx = 0
        self._lp_pick_cycle_idx = 0
        self._reset_eval_gantt_records()
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        for idx, p in enumerate(self.marks):
            p.capacity = int(self.k[idx])
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._last_state_scan = {}
        T = int(self.T)
        mask = self.get_action_mask(
            wait_action_start=T,
            n_actions=T + 1,
            concurrent=False,
        )
        enabled_t = sorted(i for i in range(T) if bool(mask[i]))
        return None, enabled_t

    # 作用：将当前库所观测写入缓冲并返回观测向量。
    def get_obs(self) -> np.ndarray:
        for place, offset in zip(self._obs_places, self._obs_offsets):
            place.write_obs_fast(self._obs_buffer, offset)
        return self._obs_buffer

    # 作用：推进时间并累计奖励、惩罚与驻留 scrap 扫描结果。
    def _advance_and_compute_reward(self, dt: int, t1: int, t2: int) -> tuple:
        """
        推进仿真时间并累计本步奖励。
        返回：(total_reward, scan_info)。stay 推进后在下方内联驻留 scrap 扫描写入 is_scrap/scrap_info；
        step 只消费该 dict。
        """
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        _training = self._training
        is_scrap = False
        scrap_info = None
        _CUSTOM_LIMITS: Dict[str, int] = {"AL": 200, "CL": 200}

        total_reward = 0.0
        total_reward += -float(safe_dt * self.time_coef_penalty)
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
                        total_reward -= safe_dt * self.warn_coef_penalty
                    progress = safe_dt if safe_dt < remain else remain
                    total_reward += self.processing_coef_reward * float(progress)
            elif has_tok and p_type == DELIVERY_ROBOT:
                tok = tokens[0]
                deadline = tok.enter_time + self.D_Residual_time
                over_start = t1 if t1 > deadline else deadline
                if t2 > over_start:
                    total_reward += -float(t2 - over_start) * float(self.transport_overtime_coef_penalty)

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

            # 驻留违规扫描
            if not has_tok:
                continue
            if p.name in _CUSTOM_LIMITS:
                resident_limit = _CUSTOM_LIMITS[p.name]
            elif p.type == CHAMBER:
                resident_limit = self.P_Residual_time
            elif p.type == 5:
                # LLA/LLB 作为缓冲区，允许晶圆无限停留，不触发 scrap
                if p.name in {"LLA", "LLB"}:
                    continue
                resident_limit = self.P_Residual_time * 3
            else:
                continue
            tok = p.tokens[0]
            remaining = p.processing_time - tok.stay_time
            if remaining < -resident_limit:
                overtime = -remaining - resident_limit
                is_scrap = True
                scrap_info = {
                    "token_id": tok.token_id,
                    "place": p.name,
                    "stay_time": tok.stay_time,
                    "proc_time": p.processing_time,
                    "overtime": overtime,
                    "type": "resident",
                }
                break

        scan_info: Dict[str, Any] = {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
        }
        self._last_state_scan = scan_info
        return total_reward, scan_info

    # 作用：初始化评估态甘特图记录槽位。
    def _reset_eval_gantt_records(self) -> None:
        route_stages = [list(stage) for stage in self._gantt_route_stages]
        place_to_sm: Dict[str, Tuple[int, int, int]] = {}
        lane_cursor: Dict[str, int] = {}
        for si, stage in enumerate(route_stages):
            stage_idx = si + 1
            base_machine = 0
            for pname in stage:
                chamber = str(pname)
                if chamber in place_to_sm:
                    continue
                place = getattr(self, "_place_by_name", {}).get(chamber) if hasattr(self, "_place_by_name") else None
                lane_count = max(1, int(getattr(place, "capacity", 1) or 1))
                place_to_sm[chamber] = (stage_idx, int(base_machine), int(lane_count))
                lane_cursor[chamber] = -1
                base_machine += lane_count
        self._eval_gantt_place_to_sm = place_to_sm
        self._eval_gantt_lane_cursor = lane_cursor
        self._eval_gantt_slots = {name: {} for name in place_to_sm.keys()}
        self._eval_gantt_closed_ops = []

    # 作用：记录晶圆进入腔室时的甘特开始片段。
    def _record_eval_gantt_enter(self, chamber: str, token_id: int, start_time: int, proc_time: int) -> None:
        if self._training:
            return
        if chamber not in self._eval_gantt_place_to_sm:
            return
        stage, base_machine, lane_count = self._eval_gantt_place_to_sm[chamber]
        cursor = int(self._eval_gantt_lane_cursor.get(chamber, -1))
        cursor = (cursor + 1) % max(1, int(lane_count))
        self._eval_gantt_lane_cursor[chamber] = cursor
        machine = int(base_machine + cursor)
        chamber_slots = self._eval_gantt_slots.setdefault(chamber, {})
        chamber_slots[int(token_id)] = {
            "job": int(token_id),
            "stage": int(stage),
            "machine": int(machine),
            "start": float(start_time),
            "proc_end": float(start_time + int(proc_time)),
        }

    # 作用：闭合晶圆离开腔室时的甘特片段。
    def _record_eval_gantt_exit(self, chamber: str, token_id: int, end_time: int) -> None:
        if self._training:
            return
        chamber_slots = self._eval_gantt_slots.get(chamber)
        if chamber_slots is None:
            return
        slot = chamber_slots.pop(int(token_id), None)
        if slot is None:
            return
        self._eval_gantt_closed_ops.append(
            {
                "job": int(slot["job"]),
                "stage": int(slot["stage"]),
                "machine": int(slot["machine"]),
                "start": float(slot["start"]),
                "proc_end": float(slot["proc_end"]),
                "end": float(end_time),
            }
        )

    # 作用：执行一个或多个变迁发射并更新 token、计数与日志。
    def _fire(self,
              t_idx: int | Sequence[int],
              start_time: int,
              end_time: int,
              is_swap: bool = False,
              swap_indices: Optional[Set[int]] = None) -> Dict[str, Any] | List[Dict[str, Any]]:
        # 是否为多变迁发射模式
        is_multi = not isinstance(t_idx, (int, np.integer))
        transitions = [int(t_idx)] if not is_multi else [int(idx) for idx in t_idx]
        if not transitions:
            return []

        swap_set = {int(idx) for idx in (swap_indices or set())}
        if is_swap and len(transitions) == 1 and not swap_set:
            swap_set.add(int(transitions[0]))

        log_entries: List[Dict[str, Any]] = []

        # 遍历变迁列表
        for current_t_idx in transitions:
            t_name = self.id2t_name[current_t_idx]
            pre_places = self._pre_place_indices[current_t_idx]
            pst_places = self._pst_place_indices[current_t_idx]

            pre_place = self.marks[int(pre_places[0])]
            pst_place = self.marks[int(pst_places[0])]

            pre_place_idx = int(pre_places[0])
            pst_place_idx = int(pst_places[0])

            tok = pre_place.pop_head()
            wafer_id = tok.token_id
            tok.enter_time = self.time
            tok.stay_time = 0

            if t_name.startswith("u_"):
                # u_* 卸载变迁：token 设置下游腔室
                src = pre_place.name
                tok._dst_level_targets = tuple(self._u_targets.get(src, []))
                # 处理加工腔卸片后的清洗计数与清洗状态。
                self._on_processing_unload(src)
            elif t_name.startswith("t_"):
                # 设置腔室加工时间（4-14路线中腔室加工时间会变化，需要更新）
                target = pst_place.name
                stage_proc_time = self._token_current_stage_process_time(tok)
                if stage_proc_time is not None:
                    pst_place.processing_time = int(stage_proc_time)

                # 处理双臂模式下的交换变迁
                if current_t_idx in swap_set and self._is_swap_eligible(pst_place):
                    # 取出腔室 token
                    old_tok = pst_place.pop_head()
                    old_wafer_id = old_tok.token_id
                    self._on_processing_unload(target)
                    old_tok.enter_time = self.time
                    old_tok.stay_time = 0
                    old_tok._dst_level_targets = tuple(self._u_targets.get(target, []))
                    old_tok.route_head_idx += 1
                    pre_place.append(old_tok)

                    # 装载腔室 token
                    tok._dst_level_targets = None
                    tok.route_head_idx += 1
                    pst_place.append(tok)
                    self._place_use_count[target] = int(self._place_use_count.get(target, 0)) + 1

                    self._record_eval_gantt_exit(target, old_wafer_id, int(start_time))
                    self._record_eval_gantt_enter(target, wafer_id, int(end_time), int(pst_place.processing_time))
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

                # 处理直接装载的t_*变迁
                tok._dst_level_targets = None
                # 若下游为LP_done，更新在系统晶圆数，计算完工奖励
                if target == "LP_done":
                    done_type = int(getattr(tok, "route_type", 1) or 1)
                    self._entered_wafer_count_by_type[done_type] = max(
                        0, int(self._entered_wafer_count_by_type.get(done_type, 0)) - 1
                    )
                    self.done_count += 1
                    self._per_wafer_reward += float(self.done_event_reward)

            # 更新 token 路由索引
            tok.route_head_idx += 1
            # LLA设置发片节拍控制
            if t_name.startswith("t_") and pst_place.name in self._release_control_places:
                self._apply_entry_release_delay(tok)
            pst_place.append(tok)
            # 记录评估甘特图进入信息
            if t_name.startswith("t_"):
                self._place_use_count[pst_place.name] = int(self._place_use_count.get(pst_place.name, 0)) + 1
                self._record_eval_gantt_enter(pst_place.name, wafer_id, int(end_time), int(pst_place.processing_time))
            elif t_name.startswith("u_"):
                self._record_eval_gantt_exit(pre_place.name, wafer_id, int(start_time))
            # 更新标识
            self.m[pre_place_idx] -= 1
            self.m[pst_place_idx] += 1
            if t_name.startswith("u_") and pre_place.name in self._release_control_places:
                released_type = int(getattr(tok, "route_type", 1) or 1)
                self._entered_wafer_count_by_type[released_type] = (
                    int(self._entered_wafer_count_by_type.get(released_type, 0)) + 1
                )
                self._last_u_entry_fire_time = int(start_time)
                self._u_entry_release_count += 1
                self._u_entry_release_count_by_type[released_type] = (
                    int(self._u_entry_release_count_by_type.get(released_type, 0)) + 1
                )
                self._advance_release_ratio_cycle()
                self._arm_entry_head_with_takt_delay(released_type)
            if t_name.startswith("u_") and pre_place.name in set(self._load_port_names or ()):
                self._advance_lp_pick_cycle()
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

    # 作用：判断发射后是否应撤销同步 resident scrap。
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

    # 作用：从 token 路由队列推断下一目标腔室。
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

    @staticmethod
    def _token_current_stage_process_time(tok: BasedToken) -> Optional[int]:
        """根据 route_proc_time_queue 读取 token 当前阶段工时配置。"""
        proc_queue = tuple(getattr(tok, "route_proc_time_queue", ()) or ())
        if not proc_queue:
            return None
        idx = int(getattr(tok, "route_head_idx", 0))
        if idx < 0 or idx >= len(proc_queue):
            return None
        value = int(proc_queue[idx])
        if value < 0:
            return None
        return value

    # 作用：把路由 gate 归一化为允许目标集合。
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

    # 作用：计算候选目标与 gate 的交集目标序列。
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

    # 作用：按 use_count 最小且稳定顺序选择目标腔室。
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
        if len(min_targets) == 1:
            picked = str(min_targets[0])
        else:
            # 并列时按 stage_targets 固定顺序选择首个最小 use_count 目标。
            # 该策略在同一状态下稳定，不会因重复 mask 查询改变结果。
            picked = str(min_targets[0])
        if cache is not None and cache_key is not None:
            cache[cache_key] = picked
        return picked

    @staticmethod
    # 作用：判定 gate 是否允许指定的 t_route_code。
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

    # 作用：校验 t_* 目标是否满足并行选机一致性约束。
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

    # 作用：收集各 release 入口队首 wafer 的 route_type。
    def _entry_type_head_tokens(self) -> Dict[int, BasedToken]:
        heads: Dict[int, BasedToken] = {}
        for place_name in self._release_control_places:
            place = self._place_by_name.get(place_name)
            if place is None or len(place.tokens) == 0:
                continue
            tok = place.tokens[0]
            t_id = int(getattr(tok, "route_type", 1) or 1)
            if t_id not in heads:
                heads[t_id] = tok
        return heads

    # 作用：按在制上限判断 route_type 是否允许发片。
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

    # 作用：返回当前 release 轮次要求的 route_type。
    def _required_release_type(self) -> Optional[int]:
        if not self._cycle_type_enabled:
            return None
        cycle = self._cycle_type
        if not cycle:
            return None
        idx = int(self._cycle_type_idx) % len(cycle)
        return int(cycle[idx])

    # 作用：结合入口队首状态解析当前可执行 route_type。
    def _resolve_required_release_type_for_entry_heads(self) -> Optional[int]:
        required = self._required_release_type()
        if required is None:
            return None
        cycle = self._cycle_type
        if not cycle:
            return required
        heads = self._entry_type_head_tokens()
        available_types = set(int(t) for t in heads.keys())
        if not available_types or int(required) in available_types:
            return required
        cur_idx = int(self._cycle_type_idx) % len(cycle)
        for offset in range(1, len(cycle) + 1):
            cand_idx = (cur_idx + offset) % len(cycle)
            cand_type = int(cycle[cand_idx])
            if cand_type not in available_types:
                continue
            self._cycle_type_idx = int(cand_idx)
            return cand_type
        return required

    # 作用：推进 release 轮转游标。
    def _advance_release_ratio_cycle(self) -> None:
        if not self._cycle_type_enabled:
            return
        cycle = self._cycle_type
        if not cycle:
            return
        self._cycle_type_idx = (int(self._cycle_type_idx) + 1) % len(cycle)

    # 作用：推进 TM1 从 LP 取料轮转游标。
    def _advance_lp_pick_cycle(self) -> None:
        """TM1 从 LP 取料时推进独立的 LP pick 计数器。"""
        if not self._cycle_type_enabled:
            return
        cycle = self._cycle_type
        if not cycle:
            return
        self._lp_pick_cycle_idx = (int(self._lp_pick_cycle_idx) + 1) % len(cycle)

    # 作用：返回当前 LP 取料轮次要求的 route_type。
    def _required_lp_pick_type(self) -> Optional[int]:
        """返回下一次 TM1 应从哪种 LP 取料（基于独立 LP pick cycle，与 LLA release cycle 解耦）。"""
        if not self._cycle_type_enabled:
            return None
        cycle = self._cycle_type
        if not cycle:
            return None
        idx = int(self._lp_pick_cycle_idx) % len(cycle)
        return int(cycle[idx])

    # 作用：返回指定 route_type 的节拍间隔需求。
    def _takt_required_interval(self, route_type: Optional[int] = None) -> Optional[int]:
        """
        返回下一次允许 release_control_places 发片的最小间隔（秒）。

        口径：
        - 首片（release_count=0）不门控，返回 None
        - 第 2 片（release_count=1）使用 cycle[0]（100 拍的第 1 个拍子）
        - 第 3 片起（release_count>=2）按序推进（idx=1,2,3...），必要时对 100 取模
        """
        type_id = int(route_type if route_type is not None else 1)
        policy = str(self._takt_policy or "").strip().lower()
        if policy == "split_by_subpath":
            takt = self._takt_result_by_type.get(type_id)
            release_count = int(self._u_entry_release_count_by_type.get(type_id, 0))
        else:
            # shared / 默认：所有类型共用同一条 takt_cycle（build_takt 已为各类型赋同一 shared 结果），
            # 索引按全局发片计数 _u_entry_release_count 推进，与 ratio_cycle 交替无关。
            takt = self._takt_result_by_type.get(type_id)
            if takt is None and self._takt_result_by_type:
                takt = next(iter(self._takt_result_by_type.values()))
            release_count = int(self._u_entry_release_count)
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

    # 作用：计算当前 route_type 还需等待的节拍时长。
    def _entry_delay_remaining(self, route_type: Optional[int] = None) -> int:
        type_id = int(route_type if route_type is not None else 1)
        policy = str(self._takt_policy or "").strip().lower()
        if policy == "shared":
            ready_at = int(self._entry_release_ready_time_shared)
        else:
            ready_at = int(self._entry_release_ready_time_by_type.get(type_id, 0))
        return max(0, ready_at - int(self.time))

    # 作用：将节拍延迟写回入口 token 的 stay_time。
    def _apply_entry_release_delay(self, tok: BasedToken) -> None:
        route_type = int(getattr(tok, "route_type", 1) or 1)
        remaining = self._entry_delay_remaining(route_type)
        if remaining > 0:
            tok.stay_time = -int(remaining)
        elif int(tok.stay_time) < 0:
            tok.stay_time = 0

    # 作用：发片后为下一入口队首挂载节拍延迟。
    def _arm_entry_head_with_takt_delay(self, route_type: Optional[int]) -> None:
        """
        每次 release_control_places 发射后，更新下一次允许发射的绝对时刻。
        """
        type_id = int(route_type if route_type is not None else 1)
        required = self._takt_required_interval(type_id)
        now = int(self.time)
        policy = str(self._takt_policy or "").strip().lower()
        if required is None:
            self._entry_release_ready_time_shared = now
            self._entry_release_ready_time_by_type[type_id] = now
            heads = self._entry_type_head_tokens()
            for head in heads.values():
                self._apply_entry_release_delay(head)
            return
        required_int = max(0, int(required))
        if policy == "shared":
            self._entry_release_ready_time_shared = now + required_int
        else:
            self._entry_release_ready_time_by_type[type_id] = now + required_int
        heads = self._entry_type_head_tokens()
        if policy == "shared":
            for head in heads.values():
                self._apply_entry_release_delay(head)
        else:
            head = heads.get(type_id)
            if head is None:
                return
            self._apply_entry_release_delay(head)

    # 作用：构建 u/t 变迁到 source/transport 的索引缓存。
    def _build_transition_index(self) -> None:
        self._u_transition_by_source = {}
        self._u_transition_by_source_transport = {}
        self._t_transitions_by_transport = {}
        self._tm1_transition_indices = []
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
                    if dst in {"TM1", "TM2", "TM3"}:
                        self._u_transition_by_source_transport[(src, dst)] = int(t_idx)
                        transport_name = str(dst)
                    self._u_transition_by_source[src] = int(t_idx)
            elif t_name.startswith("t_"):
                transport = self._transition_transport_place(int(t_idx))
                if transport is None:
                    continue
                self._t_transitions_by_transport.setdefault(transport, []).append(int(t_idx))
                transport_name = str(transport)
            if transport_name == "TM1":
                self._tm1_transition_indices.append(int(t_idx))
            elif transport_name == "TM2":
                self._tm2_transition_indices.append(int(t_idx))
            elif transport_name == "TM3":
                self._tm3_transition_indices.append(int(t_idx))

    @staticmethod
    # 作用：深拷贝 marks 与 token，避免共享引用。
    def _clone_marks(marks: List[Place]) -> List[Place]:
        cloned: List[Place] = []
        for p in marks:
            cp = p.clone()
            cloned.append(cp)
        return cloned

    # 作用：通过后置索引解析 t_idx 对应目标库所名。
    def _transition_target_place(self, t_idx: int) -> Optional[str]:
        """根据变迁后置库所索引返回目标库所名，避免依赖 t_* 命名格式。"""
        if t_idx < 0 or t_idx >= len(self._pst_place_indices):
            return None
        pst_idx = self._pst_place_indices[int(t_idx)]
        if pst_idx.size == 0:
            return None
        return str(self.id2p_name[int(pst_idx[0])])

    # 作用：通过前置索引解析 t_idx 对应运输位库所名。
    def _transition_transport_place(self, t_idx: int) -> Optional[str]:
        """根据变迁前置库所索引返回运输位库所名。"""
        if t_idx < 0 or t_idx >= len(self._pre_place_indices):
            return None
        pre_idx = self._pre_place_indices[int(t_idx)]
        for p_i in pre_idx:
            p_name = str(self.id2p_name[int(p_i)])
            if p_name in {"TM1", "TM2", "TM3"}:
                return p_name
        return None

    # 作用：按 source->target 映射选择运输位名称。
    def _transport_for_t_target(self, source: str, target: str) -> str:
        """按当前 route 的 hop 映射选择 source->target 的 transport。"""
        mapped = self._route_source_target_transport.get((str(source), str(target)))
        if mapped:
            return str(mapped)
        return infer_cascade_transport_by_scope((str(source),), (str(target),))

    # 作用：初始化 PM 清洗阈值与运行状态字段。
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

    # 作用：计算下一关键事件时间差并同步关键标记。
    def get_next_event_delta(self) -> Optional[int]:
        """
        计算当前时刻到下一个关键事件的时间差（秒）。
        仅在一次 marks 扫描中完成关键事件距离计算。
        """
        best = None
        _CHAMBER_TYPES = (CHAMBER, 5)
        has_important_task = False
        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            if place.is_dtm or place.name in {"TM1", "TM2", "TM3"}:
                has_important_task = True
            elif place.type in _CHAMBER_TYPES:
                ptime = place.processing_time
                if ptime > 0:
                    head = tokens[0]
                    delta = ptime - head.stay_time
                    if delta < 0:
                        delta = 0
                    if delta > 0 and (best is None or delta < best):
                        best = delta
                    if head.stay_time >= ptime and place.type == CHAMBER:
                        has_important_task = True
        entry_heads = self._entry_type_head_tokens()
        if entry_heads:
            deltas = [
                int(self._entry_delay_remaining(int(getattr(tok, "route_type", 1) or 1)))
                for tok in entry_heads.values()
            ]
            if deltas:
                delta_takt = min(deltas)
                if delta_takt > 0 and (best is None or delta_takt < best):
                    best = delta_takt
        return 5 if has_important_task else best

    def _on_processing_unload(self, source_name: str) -> None:
        """处理加工腔卸片后的清洗计数与清洗状态"""
        if not self._cleaning_enabled:
            return
        trigger = self._cleaning_trigger_map.get(source_name, 0)
        if trigger <= 0:
            return
        source_place = self._place_by_name.get(source_name)
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

    # 作用：判断目标位是否满足双臂 swap 条件。
    def _is_swap_eligible(self, pst_place: Place) -> bool:
        """目标库所是否可执行 swap（仅在 _fire 中调用）。
        PM 腔室及 TM1 的 AL/CL 目标均可 swap。
        """
        if not self._dual_arm:
            return False
        if not pst_place.is_pm and pst_place.name not in self._TM1_SWAP_PLACES:
            return False
        if len(pst_place.tokens) < pst_place.capacity:
            return False
        if not pst_place.tokens:
            return False
        if pst_place.tokens[0].stay_time < pst_place.processing_time:
            return False
        return not pst_place.is_cleaning

    # 作用：预判当前 t_idx 是否会触发 swap。
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

    # 作用：判断 source 下游是否存在可执行目标并返回目标名。
    def _is_next_stage_available(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        按 route gate 过滤后的候选集中，优先选择 use_count 最小的目标；
        若并列，按当前 use_count 快照做确定性打破。单臂满或清洗则不可出片，双臂仅清洗阻塞。
        """
        candidates = tuple(self._u_targets.get(source, ()))
        if not candidates:
            return False, None
        source_place = self._place_by_name.get(source)
        head_tok = source_place.tokens[0] if len(source_place.tokens) > 0 else None
        tok_gate = head_tok.route_queue[head_tok.route_head_idx] if head_tok is not None else -1
        target_name = self._select_min_use_count_target(candidates, tok_gate)
        if target_name is None:
            return False, None
        target_place = self._place_by_name.get(target_name)
        if target_place.is_cleaning:
            return False, None
        if not self._dual_arm and len(target_place.tokens) >= target_place.capacity:
            return False, None
        return True, target_name

    _MASK_TIMED_TYPES = frozenset((CHAMBER, 5, DELIVERY_ROBOT))
    # TM1 在这些库所支持双臂 swap（AL：送新料同时带走旧料；CL：送入同时取出待送 LP_done 的料）
    _TM1_SWAP_PLACES: frozenset = frozenset({"AL", "CL"})

    # 作用：从全量掩码按优先级自动挑选 TM1 动作。
    def _pick_tm1_from_mask(
        self,
        mask: np.ndarray,
        required_release_type: Optional[int] = None,
    ) -> Optional[int]:
        """从已计算的全量 mask 中按优先级选择 TM1 变迁（不再调用 get_action_mask）。"""
        pbname = self._place_by_name
        tm1_t_indices = self._t_transitions_by_transport.get("TM1", [])
        total_actions = int(mask.shape[0])
        place_by_name = self._place_by_name
        route_code_by_idx = self._t_route_code_by_idx
        route_gate_allows = self._route_gate_allows_t

        # 1. TM1 持有晶圆 → 投递到路由目的地（与 get_action_mask d_TM 分支同一套 gate/并行选机/结构判定）
        tm1_place = pbname.get("TM1")
        if tm1_place is not None and len(tm1_place.tokens) > 0:
            struct_enabled_cache: Dict[int, bool] = {}
            selected_parallel_target_cache: Dict[Tuple[int, Tuple[str, ...]], str] = {}

            # 作用：缓存并判定 TM1 候选 t_idx 的结构性使能。
            def _is_struct_enabled_pick(t_idx: int) -> bool:
                cached = struct_enabled_cache.get(t_idx)
                if cached is not None:
                    return cached
                result = not bool(
                    (
                        self.m[int(self._pre_place_indices[t_idx][0])] < 1
                        or self.m[int(self._pst_place_indices[t_idx][0])] + 1
                        > self.k[int(self._pst_place_indices[t_idx][0])]
                    )
                )
                struct_enabled_cache[t_idx] = result
                return result

            p_type = tm1_place.type
            proc_time = tm1_place.processing_time
            is_timed = p_type in self._MASK_TIMED_TYPES
            for tok in tm1_place.tokens:
                if is_timed and proc_time > 0 and tok.stay_time < proc_time:
                    continue
                tok_gate = tok.route_queue[tok.route_head_idx]
                for t_idx in tm1_t_indices:
                    if not mask[int(t_idx)]:
                        continue
                    target = self._transition_target_place(int(t_idx))
                    if target is None:
                        continue
                    if not self._allow_t_by_use_count(
                        "TM1",
                        target,
                        tok_gate,
                        tok,
                        selected_parallel_target_cache,
                    ):
                        continue
                    if not route_gate_allows(tok_gate, route_code_by_idx[t_idx]):
                        continue
                    target_place = place_by_name.get(target)
                    if target_place is None or target_place.is_cleaning:
                        continue
                    if self._dual_arm and target_place.is_pm:
                        if (
                            len(target_place.tokens) > 0
                            and target_place.tokens[0].stay_time < target_place.processing_time
                        ):
                            continue
                    else:
                        if not _is_struct_enabled_pick(int(t_idx)):
                            continue
                    if 0 <= int(t_idx) < total_actions:
                        return int(t_idx)
            return None

        # 2. LLB 有完成晶圆 → 优先清空（防止 LLB 满阻塞 TM2）
        u_llb = self._u_transition_by_source_transport.get(("LLB", "TM1"))
        if u_llb is not None and mask[u_llb]:
            return u_llb

        # 3. CL 有完成晶圆 → 取走送 LP_done
        u_cl = self._u_transition_by_source_transport.get(("CL", "TM1"))
        if u_cl is not None and mask[u_cl]:
            return u_cl

        # 4. AL 有完成晶圆 且 LLA 未满 → 送料到 LLA
        lla = pbname.get("LLA")
        if lla is not None and len(lla.tokens) < lla.capacity:
            u_al = self._u_transition_by_source_transport.get(("AL", "TM1"))
            if u_al is not None and mask[u_al]:
                return u_al

        # 5. LP 有晶圆 且 AL 空 且 LLA 未满 → 从 LP 取料（shared+ratio 时优先当前发片轮次所需 route_type）
        al = pbname.get("AL")
        if al is not None and len(al.tokens) == 0 and lla is not None and len(lla.tokens) < lla.capacity:
            lp_candidates: List[str] = []
            for lp_name in self._load_port_names:
                u_lp = self._u_transition_by_source_transport.get((lp_name, "TM1"))
                if u_lp is not None and mask[u_lp]:
                    lp_candidates.append(str(lp_name))
            if lp_candidates:
                # 优先使用独立的 LP pick cycle（与 LLA release cycle 解耦），
                # 避免因 LLA 中仍有 type1 wafer 导致 cycle 回滚、TM1 持续选 LP1 的问题。
                lp_pick_type = self._required_lp_pick_type()
                effective_req = lp_pick_type if lp_pick_type is not None else required_release_type
                if effective_req is not None:
                    req = int(effective_req)

                    # 作用：读取指定 LP 队首晶圆的 route_type。
                    def _lp_head_route_type(name: str) -> int:
                        lp = pbname.get(name)
                        if lp is None or len(lp.tokens) == 0:
                            return -1
                        return int(getattr(lp.tokens[0], "route_type", 1) or 1)

                    preferred = [lp for lp in lp_candidates if _lp_head_route_type(lp) == req]
                    others = [lp for lp in lp_candidates if lp not in preferred]
                    pick_lps = preferred + others
                else:
                    pick_lps = lp_candidates
                for lp_name in pick_lps:
                    u_lp = self._u_transition_by_source_transport.get((lp_name, "TM1"))
                    if u_lp is not None and mask[u_lp]:
                        return int(u_lp)

        return None

    # 作用：把全量动作掩码投影成 TM1/TM2/TM3 三段掩码。
    def _tm_masks_from_full(self, full_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tm1_idx = self._tm1_transition_indices
        tm2_idx = self._tm2_transition_indices
        tm3_idx = self._tm3_transition_indices
        mask_tm1 = np.zeros(len(tm1_idx) + 1, dtype=bool)
        for i, t_idx in enumerate(tm1_idx):
            mask_tm1[i] = bool(full_mask[t_idx])
        mask_tm1[-1] = True
        mask_tm2 = np.zeros(len(tm2_idx) + 1, dtype=bool)
        for i, t_idx in enumerate(tm2_idx):
            mask_tm2[i] = bool(full_mask[t_idx])
        mask_tm2[-1] = True
        mask_tm3 = np.zeros(len(tm3_idx) + 1, dtype=bool)
        for i, t_idx in enumerate(tm3_idx):
            mask_tm3[i] = bool(full_mask[t_idx])
        mask_tm3[-1] = True
        return mask_tm1, mask_tm2, mask_tm3

    # 作用：生成全量或并发三头动作掩码并缓存自动 TM1 动作。
    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
        concurrent: Optional[bool] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        返回离散动作掩码。concurrent 为 None 时跟随 self._concurrent。
        concurrent=False：完整 transition + wait 向量。
        concurrent=True：TM1/TM2/TM3 局部动作空间三段掩码（末维为各自 WAIT，恒 True）。
        """
        use_tm = self._concurrent if concurrent is None else bool(concurrent)
        start = int(self.T if wait_action_start is None else wait_action_start)
        total_actions = int(
            n_actions if n_actions is not None else (start + 1)
        )
        mask = np.zeros(total_actions, dtype=bool)
        struct_enabled_cache: Dict[int, bool] = {}
        selected_parallel_target_cache: Dict[Tuple[int, Tuple[str, ...]], str] = {}

        # 作用：缓存并判定通用变迁的结构性使能。
        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(t_idx)
            if cached is not None:
                return cached
            result = not bool((self.m[int(self._pre_place_indices[t_idx][0])] < 1 or
                      self.m[int(self._pst_place_indices[t_idx][0])] + 1 >
                               self.k[int(self._pst_place_indices[t_idx][0])]))
            struct_enabled_cache[t_idx] = result
            return result

        # release_control_places：单路线用全局 WIP；双子路径按类型上限；节拍入口由 route_meta 指定。
        required_release_type = self._resolve_required_release_type_for_entry_heads()
        if self._multi_subpath or sum(self._entered_wafer_count_by_type.values()) < int(self.max_wafers1_in_system):
            for place_name in self._release_control_places:
                control_place = self._place_by_name.get(place_name)
                if control_place is None or len(control_place.tokens) == 0:
                    continue
                head = control_place.tokens[0]
                route_type = int(getattr(head, "route_type", 1) or 1)
                if required_release_type is not None and route_type != required_release_type:
                    continue
                if int(self._entry_delay_remaining(route_type)) > 0:
                    continue
                expected_place = self._wafer_type_to_release_place.get(
                    route_type,
                    str(self._release_control_places[0]) if self._release_control_places else str(place_name),
                )
                if expected_place != place_name:
                    raise RuntimeError(
                        f"wafer_type {route_type} maps to release place {expected_place} but queue head is on {place_name}"
                    )
                if not self._allow_start_for_route_type(route_type):
                    continue
                release_target = self._token_next_target(head)
                if release_target is None:
                    continue
                release_transport = self._transport_for_t_target(place_name, str(release_target))
                u_idx = self._u_transition_by_source_transport.get((place_name, release_transport))
                if u_idx is not None and _is_struct_enabled(int(u_idx)):
                    t_idx = int(u_idx)
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
            if pname in skip or pname in self._release_control_places:
                continue
            p_type = place.type
            proc_time = place.processing_time

            is_timed = p_type in timed

            if place.is_dtm or place.name in {"TM1", "TM2", "TM3"}:
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

        if not (has_ready_chamber and self.wait_duration > 5):
            if 0 <= start < total_actions:
                mask[start] = True

        if self._concurrent:
            self._cached_auto_tm1_action = self._pick_tm1_from_mask(mask, required_release_type)

        if use_tm:
            return self._tm_masks_from_full(mask)
        return mask

    # 作用：基于评估日志渲染腔室甘特图。
    def render_gantt(self, out_path: str, title_suffix: str | None = None) -> None:
        from visualization.plot import Op, plot_gantt_hatched_residence

        route_stages = [list(stage) for stage in self._gantt_route_stages]
        if not route_stages:
            raise ValueError(
                f"render_gantt: route '{self.single_route_name}' missing non-empty route_stage in single_route_config"
            )
        place_to_sm: Dict[str, Tuple[int, int]] = {}
        for si, stage in enumerate(route_stages):
            s = si + 1
            for mi, pname in enumerate(stage):
                pname_s = str(pname)
                if pname_s in place_to_sm:
                    continue
                place_to_sm[pname_s] = (s, int(mi))
        lane_places = frozenset(str(name) for name in place_to_sm.keys())

        S = len(route_stages)
        proc_time: Dict[int, float] = {}
        capacity: Dict[int, int] = {}
        stage_module_names: Dict[int, List[str]] = {}
        for s in range(1, S + 1):
            names = route_stages[s - 1]
            proc_time[s] = float(
                max(int(self._base_proc_time_map.get(str(n), 0)) for n in names),
            )
            capacity[s] = sum(
                max(1, int(getattr(self._place_by_name.get(str(n)), "capacity", 1) or 1))
                for n in names
            )
            stage_module_names[s] = [str(x) for x in names]

        ops: List[Op] = []
        for raw in self._eval_gantt_closed_ops:
            stage = int(raw["stage"])
            machine = int(raw["machine"])
            if stage <= 0 or stage > S:
                continue
            chamber_names = stage_module_names.get(stage, [])
            if machine < 0 or machine >= len(chamber_names):
                continue
            chamber_name = str(chamber_names[machine])
            if chamber_name not in lane_places:
                continue
            ops.append(
                Op(
                    job=int(raw["job"]),
                    stage=stage,
                    machine=machine,
                    start=float(raw["start"]),
                    proc_end=float(raw["proc_end"]),
                    end=float(raw["end"]),
                )
            )
        current_time = float(self.time)
        for chamber_name, slots_dict in self._eval_gantt_slots.items():
            if chamber_name not in lane_places:
                continue
            for raw in slots_dict.values():
                stage = int(raw["stage"])
                machine = int(raw["machine"])
                if stage <= 0 or stage > S:
                    continue
                chamber_names = stage_module_names.get(stage, [])
                if machine < 0 or machine >= len(chamber_names):
                    continue
                if str(chamber_names[machine]) != chamber_name:
                    continue
                ops.append(
                    Op(
                        job=int(raw["job"]),
                        stage=stage,
                        machine=machine,
                        start=float(raw["start"]),
                        proc_end=float(raw["proc_end"]),
                        end=current_time,
                    )
                )

        if not ops:
            raise ValueError("render_gantt: no chamber operations from eval fire records")

        job_ids = {int(op.job) for op in ops if int(op.job) >= 0}
        n_jobs = max(1, len(job_ids))

        base = str(out_path)
        if base.lower().endswith(".png"):
            base = base[:-4]

        arm_info = {"ARM1": [], "ARM2": [], "STAGE2ACT": {}}
        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=proc_time,
            capacity=capacity,
            n_jobs=n_jobs,
            out_path=base,
            arm_info=arm_info,
            with_label=True,
            no_arm=True,
            policy=2,
            stage_module_names=stage_module_names,
            title_suffix=title_suffix,
        )
