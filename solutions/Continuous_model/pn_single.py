"""
单设备 Petri 网（构网驱动、单机械手、单动作）。
执行链：construct_single -> _get_enable_t -> step -> calc_reward
"""

from __future__ import annotations

from collections import deque
import json
import time
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.construct_single import build_single_device_net
from solutions.Continuous_model.pn import Place

MAX_TIME = 4000


class PetriSingleDevice:
    def __init__(self, config: PetriEnvConfig = None) -> None:
        assert config is not None, "config must be provided"
        self.config = config

        self.n_wafer = int(config.n_wafer)
        self.R_done = int(getattr(config, "R_done", 800))
        self.R_finish = int(getattr(config, "R_finish", 800))
        self.R_scrap = int(getattr(config, "R_scrap", 1000))
        self.T_warn = int(getattr(config, "T_warn", 5))
        self.a_warn = float(getattr(config, "a_warn", 1.0))
        self.T_safe = int(getattr(config, "T_safe", 10))
        self.b_safe = float(getattr(config, "b_safe", 0.2))
        self.P_Residual_time = int(getattr(config, "P_Residual_time", 15))
        self.D_Residual_time = int(getattr(config, "D_Residual_time", 10))
        self.T_transport = int(getattr(config, "T_transport", 5))
        self.T_load = int(getattr(config, "T_load", 5))
        self.idle_penalty = float(getattr(config, "idle_penalty", 500))
        self.stop_on_scrap = bool(getattr(config, "stop_on_scrap", True))
        self.release_penalty_coef = float(getattr(config, "release_penalty_coef", 5))
        self.reward_config = dict(getattr(config, "reward_config", {}))
        self.robot_capacity = 2 if int(getattr(config, "single_robot_capacity", 1)) == 2 else 1
        
        # 奖励计算系数
        self.processing_reward_coef = float(getattr(config, "processing_reward_coef", 3.0))
        self.transport_overtime_coef = float(getattr(config, "transport_overtime_coef", 1.0))
        self.time_coef = int(getattr(config, "time_coef", 1))
        self.single_cleaning_enabled = bool(getattr(config, "single_cleaning_enabled", True))
        self.single_cleaning_targets = set(getattr(config, "single_cleaning_targets", ["PM3", "PM4"]))
        self.single_cleaning_trigger_wafers = max(1, int(getattr(config, "single_cleaning_trigger_wafers", 2)))
        self.single_cleaning_duration = max(0, int(getattr(config, "single_cleaning_duration", 150)))
        self.single_u_lp_boundary_enabled = bool(getattr(config, "single_u_lp_boundary_enabled", True))

        info = build_single_device_net(
            n_wafer=self.n_wafer,
            ttime=max(1, self.T_transport),
            robot_capacity=self.robot_capacity,
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
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self._u_targets: Dict[str, List[str]] = {
            "LP": ["PM1"],
            "PM1": ["PM3", "PM4"],
            "PM3": ["LP_done"],
            "PM4": ["LP_done"],
        }
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.ttime = int(np.max(info["ttime"])) if len(info["ttime"]) > 0 else 5

        self.ori_marks: List[Place] = info["marks"]
        self.marks: List[Place] = self._clone_marks(self.ori_marks)

        self.time = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log: List[Dict[str, Any]] = []
        self.enable_statistics = True
        self._per_wafer_reward = 0.0
        self._token_stats: Dict[int, Dict[str, Any]] = {}
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        self._last_deadlock = False
        self.no_release_penalty = False
        self._chamber_timeline: Dict[str, list] = {"PM1": [], "PM3": [], "PM4": []}
        self._chamber_active: Dict[str, Dict[int, int]] = {"PM1": {}, "PM3": {}, "PM4": {}}
        self._init_single_cleaning_state()

    @staticmethod
    def _clone_marks(marks: List[Place]) -> List[Place]:
        cloned: List[Place] = []
        for p in marks:
            cp = Place(
                name=p.name,
                capacity=p.capacity,
                processing_time=p.processing_time,
                type=p.type,
                last_machine=getattr(p, "last_machine", -1),
                processed_wafer_count=int(getattr(p, "processed_wafer_count", 0)),
                idle_time=int(getattr(p, "idle_time", 0)),
                last_proc_type=str(getattr(p, "last_proc_type", "")),
                is_cleaning=bool(getattr(p, "is_cleaning", False)),
                cleaning_remaining=int(getattr(p, "cleaning_remaining", 0)),
                cleaning_reason=str(getattr(p, "cleaning_reason", "")),
            )
            cp.tokens = deque(tok.clone() for tok in p.tokens)
            cloned.append(cp)
        return cloned

    def _debug_log(self, hypothesis_id: str, message: str, data: Dict[str, Any], run_id: str = "train-run") -> None:
        # #region agent log
        try:
            payload = {
                "sessionId": "2838c2",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": "solutions/Continuous_model/pn_single.py",
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            with open("C:/Users/khand/OneDrive/code/dqn/CT/debug-2838c2.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion

    def _get_place(self, name: str) -> Place:
        for p in self.marks:
            if p.name == name:
                return p
        raise KeyError(f"unknown place: {name}")

    def _get_place_index(self, name: str) -> int:
        return self.id2p_name.index(name)

    def _update_marking_vector(self) -> None:
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)

    def _update_stay_times(self, dt: int) -> None:
        if dt <= 0:
            return
        for p in self.marks:
            if p.type == 3:  # 跳过 LP 中的 wafer
                continue
            for tok in p.tokens:
                tok.stay_time += dt

    def _init_single_cleaning_state(self) -> None:
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
                # #region agent log
                self._debug_log(
                    hypothesis_id="H1",
                    message="cleaning_end",
                    data={
                        "time": int(self.time),
                        "chamber": p.name,
                        "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                    },
                )
                # #endregion
                self.fire_log.append(
                    {
                        "event_type": "cleaning_end",
                        "time": int(self.time),
                        "chamber": p.name,
                        "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                    }
                )

    def _start_cleaning(self, place: Place, reason: str, trigger_count: int) -> None:
        if self.single_cleaning_duration <= 0:
            return
        place.is_cleaning = True
        place.cleaning_remaining = int(self.single_cleaning_duration)
        place.cleaning_reason = reason
        place.processed_wafer_count = 0
        place.idle_time = 0
        # #region agent log
        self._debug_log(
            hypothesis_id="H1",
            message="cleaning_start",
            data={
                "time": int(self.time),
                "chamber": place.name,
                "rule": reason,
                "duration": int(self.single_cleaning_duration),
                "trigger_count": int(trigger_count),
            },
        )
        # #endregion
        self.fire_log.append(
            {
                "event_type": "cleaning_start",
                "time": int(self.time),
                "chamber": place.name,
                "rule": reason,
                "duration": int(self.single_cleaning_duration),
                "trigger_count": int(trigger_count),
            }
        )

    def _on_processing_unload(self, source_name: str) -> None:
        if not self.single_cleaning_enabled:
            return
        if source_name not in self.single_cleaning_targets:
            return
        source_place = self._get_place(source_name)
        source_place.processed_wafer_count = int(getattr(source_place, "processed_wafer_count", 0)) + 1
        source_place.last_proc_type = source_name
        if source_place.is_cleaning:
            return
        if source_place.processed_wafer_count >= self.single_cleaning_trigger_wafers:
            self._start_cleaning(
                source_place,
                reason="processed_wafers",
                trigger_count=int(source_place.processed_wafer_count),
            )

    def _is_process_ready(self, place_name: str) -> bool:
        place = self._get_place(place_name)
        if len(place.tokens) == 0:
            return False
        if place.processing_time <= 0:
            return True
        return place.head().stay_time >= place.processing_time

    def _estimate_place_accept_time(self, place_name: str, include_cleaning: bool = True) -> int:
        """
        估计 place 最早可接收新 wafer 的时间点（绝对时间）。
        - 若未满：当前时刻（若清洗中则考虑 cleaning_remaining）
        - 若已满：至少等待一个在制 wafer 可被卸载释放
        """
        place = self._get_place(place_name)
        base_time = int(self.time)
        if include_cleaning and bool(getattr(place, "is_cleaning", False)):
            base_time = max(base_time, int(self.time) + int(getattr(place, "cleaning_remaining", 0)))

        occupancy = len(place.tokens)
        if occupancy < int(place.capacity):
            return base_time

        if place.processing_time <= 0:
            return base_time

        earliest_release = min(
            int(self.time) + max(0, int(place.processing_time) - int(getattr(tok, "stay_time", 0)))
            for tok in place.tokens
        )
        return max(base_time, earliest_release)

    def _allow_u_lp_by_reverse_boundary(self) -> bool:
        """
        基于 PM1 与 PM3/PM4 的最早可接收时间反推 u_LP 是否应放行。
        仅用于 Stage2 动作裁剪，不影响 Stage1 死锁判定口径。
        """
        edge_transfer = int(self.T_transport) + int(self.T_load)

        pm1_accept_time = self._estimate_place_accept_time("PM1", include_cleaning=False)
        pm1_enter_time = max(int(self.time) + edge_transfer, pm1_accept_time)
        pm1_proc_time = max(0, int(self._get_place("PM1").processing_time))
        pm1_ready_to_unload_time = pm1_enter_time + pm1_proc_time

        pm3_accept_time = self._estimate_place_accept_time("PM3", include_cleaning=True)
        pm4_accept_time = self._estimate_place_accept_time("PM4", include_cleaning=True)
        pm2_stage_accept_time = min(pm3_accept_time, pm4_accept_time)

        predicted_pm2_stage_enter_time = pm1_ready_to_unload_time + edge_transfer
        allow = predicted_pm2_stage_enter_time >= pm2_stage_accept_time

        if not allow:
            self._debug_log(
                hypothesis_id="H5",
                message="u_lp_reverse_boundary_blocked",
                data={
                    "time": int(self.time),
                    "predicted_pm2_stage_enter_time": int(predicted_pm2_stage_enter_time),
                    "pm2_stage_accept_time": int(pm2_stage_accept_time),
                    "pm1_accept_time": int(pm1_accept_time),
                    "pm3_accept_time": int(pm3_accept_time),
                    "pm4_accept_time": int(pm4_accept_time),
                    "pm3_cleaning": bool(getattr(self._get_place("PM3"), "is_cleaning", False)),
                    "pm4_cleaning": bool(getattr(self._get_place("PM4"), "is_cleaning", False)),
                },
            )
        return allow

    def _select_target_for_source(
        self,
        source: str,
        preferred_target: Optional[str] = None,
        ignore_cleaning: bool = False,
    ) -> Optional[str]:
        """
        为 u_<source> 选择一个可接收目标（确定性顺序）。
        仅检查目标腔室容量，运输位停留时间约束仍由 t_* 侧控制。
        """
        candidates = self._u_targets.get(source, [])
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
        if source in {"PM1", "LP"} and candidates:
            # #region agent log
            self._debug_log(
                hypothesis_id="H1",
                message="no_target_available",
                data={
                    "time": int(self.time),
                    "source": source,
                    "preferred_target": preferred_target,
                    "candidates": [
                        {
                            "name": tgt,
                            "is_cleaning": bool(getattr(self._get_place(tgt), "is_cleaning", False)),
                            "cleaning_remaining": int(getattr(self._get_place(tgt), "cleaning_remaining", 0)),
                            "tokens": len(self._get_place(tgt).tokens),
                            "capacity": int(self._get_place(tgt).capacity),
                        }
                        for tgt in candidates
                    ],
                },
            )
            # #endregion
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
        for p in self.marks:
            if p.type != 1 or len(p.tokens) == 0:
                continue
            tok = p.head()
            overtime = int(tok.stay_time - (p.processing_time + self.P_Residual_time))
            if overtime > 0:
                return True, {
                    "token_id": int(getattr(tok, "token_id", -1)),
                    "place": p.name,
                    "stay_time": int(tok.stay_time),
                    "proc_time": int(p.processing_time),
                    "overtime": overtime,
                    "type": "resident",
                }
        return False, None

    def blame_release_violations(self) -> Dict[int, float]:
        """
        事后追责：基于当前 fire_log 与 _chamber_timeline，回溯可能导致下游容量冲突的 u_* 动作。
        单设备中将 PM1 统一视为 s1，PM3/PM4 合并视为 s2（并行机台池）。
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

        # 站点别名：PM1 -> s1，PM3/PM4 -> s2（并行机台池）
        intervals_s1 = build_intervals("PM1")
        intervals_s2 = build_intervals("PM3") + build_intervals("PM4")
        intervals_s2.sort(key=lambda x: x[0])
        intervals_by_station: Dict[str, List[tuple]] = {
            "s1": intervals_s1,
            "s2": intervals_s2,
        }
        capacity_by_station: Dict[str, int] = {
            "s1": capacities.get("PM1", 1),
            "s2": capacities.get("PM3", 0) + capacities.get("PM4", 0),
        }
        proc_time_by_station: Dict[str, int] = {
            "s1": proc_times.get("PM1", 0),
            "s2": max(proc_times.get("PM3", 0), proc_times.get("PM4", 0)),
        }

        edge_transfer = self.T_transport + self.T_load

        def will_exceed_capacity(intervals, at_time, cap, current_wid):
            occupied = sum(1 for (e, l, wid0) in intervals if e <= at_time < l and wid0 < current_wid)
            return occupied + 1 > cap

        # 单设备 downstream chain：统一为 s1 -> s2（s2 是 PM3/PM4 合并池）
        chain_map: Dict[str, List[str]] = {"u_LP": ["s1", "s2"]}

        # 2) 回溯 fire_log，针对每个 u_* 动作检查其 downstream chain 是否存在容量冲突。
        penalty_coeff = float(self.release_penalty_coef) * 100.0
        for i, log in enumerate(self.fire_log):
            t_name = log.get("t_name", "")
            wid = int(log.get("token_id", -1))
            # 仅追责 u_* 动作，且必须有合法的 wafer_id
            if wid < 0 or not t_name.startswith("u_"):
                continue
            chain = chain_map.get(t_name, [])
            # 不在追责名单上不进行追责，避免误伤其他动作
            if not chain:
                continue

            t_leave = int(log.get("t1", 0))
            arrival = t_leave + edge_transfer
            violated = False
            for idx, station in enumerate(chain):
                intervals = intervals_by_station.get(station, [])
                cap = capacity_by_station.get(station, 1)
                if will_exceed_capacity(intervals, arrival, cap, wid):
                    violated = True
                    break
                if idx < len(chain) - 1:
                    arrival = arrival + proc_time_by_station.get(station, 0) + edge_transfer

            if violated:
                blame[i] = penalty_coeff
        return blame

    def _get_enable_t_stage1(self) -> List[int]:
        """
        Stage1: 基础使能（pre/pst + 容量 + 防死锁规则），不做“加工完成”就绪检查。
        """

        # =====
        enabled: List[int] = []
        d_tm = self._get_place("d_TM1")
        d_tm_idx = self._get_place_index("d_TM1")
        head_tok = d_tm.head() if len(d_tm.tokens) > 0 else None
        head_target = getattr(head_tok, "_target_place", None) if head_tok is not None else None
        head_where = int(getattr(head_tok, "where", 0)) if head_tok is not None else 0
        color_idx = int(np.clip(head_where, 0, self.pre_color.shape[2] - 1))
        locked_sources: set[str] = set()
        if self.robot_capacity == 2 and head_tok is not None:
            locked_sources = set(getattr(head_tok, "_dst_level_targets", ()))

        # =====self.m >= self.pre[:, t]=======
        # =====self.m + self.pst[:, t] <= self.k=======
        for t in range(self.T):
            base_pre = self.pre[:, t]
            base_pre_idx = np.flatnonzero(base_pre > 0)
            if base_pre_idx.size == 0:
                continue
            color_pre = base_pre
            if base_pre[d_tm_idx] > 0:
                color_pre = self.pre_color[:, t, color_idx]
            if np.any((base_pre > 0) & (color_pre <= 0)):
                continue
            if np.any(self.m[base_pre_idx] < color_pre[base_pre_idx]):
                continue
            if np.any(self.m + self.net[:, t] > self.k):
                continue

            t_name = self.id2t_name[t]
            if t_name.startswith("u_"):
                src = t_name[2:]
                if self.robot_capacity == 2:
                    # 双臂规则2（更新）：只要 d_TM1 队首有晶圆，就锁定后续 u_* 来源到该晶圆的 dst 层。
                    if len(d_tm.tokens) > 0 and locked_sources and src not in locked_sources:
                        continue
                    # 关键约束：无论 d_TM1 是否为空、是否触发锁定，都必须有可解析目标。
                    # 否则会出现 u_* 发射后 _target_place 缺失，进而误放行 t_LP_done 的非法路径。
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        continue
                else:
                    # 单臂保持原规则：下游可接收才允许取片。
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                if head_target is not None and head_target != target:
                    continue
            enabled.append(t)
        if len(enabled) == 0:
            # #region agent log
            self._debug_log(
                hypothesis_id="H2",
                message="stage1_empty",
                data={
                    "time": int(self.time),
                    "head_target": head_target,
                    "head_where": head_where,
                    "d_tm_tokens": len(d_tm.tokens),
                    "locked_sources": sorted(list(locked_sources)),
                    "lp_done_tokens": len(self._get_place("LP_done").tokens),
                    "pm_states": {
                        p.name: {
                            "tokens": len(p.tokens),
                            "is_cleaning": bool(getattr(p, "is_cleaning", False)),
                            "cleaning_remaining": int(getattr(p, "cleaning_remaining", 0)),
                            "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                        }
                        for p in self.marks
                        if p.name.startswith("PM")
                    },
                },
            )
            # #endregion
        return enabled

    def _apply_enable_stage2(self, stage1_enabled: List[int]) -> List[int]:
        """
        Stage2: 在 Stage1 基础上做就绪过滤（加工完成 + 运输位 dwell）。
        """
        enabled: List[int] = []
        d_place = self._get_place("d_TM1")
        dwell_time = max(0, int(getattr(d_place, "processing_time", self.T_transport)))
        blocked_by_cleaning: List[str] = []
        for t in stage1_enabled:
            t_name = self.id2t_name[t]
            if t_name.startswith("u_"):
                src = t_name[2:]
                if not self._is_process_ready(src):
                    continue
                if self._select_target_for_source(src) is None:
                    continue
                if (
                    src == "LP"
                    and self.single_u_lp_boundary_enabled
                    and not self._allow_u_lp_by_reverse_boundary()
                ):
                    continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                target_place = self._get_place(target)
                if bool(getattr(target_place, "is_cleaning", False)):
                    blocked_by_cleaning.append(target)
                    continue
                if len(d_place.tokens) > 0 and d_place.head().stay_time < dwell_time:
                    continue
            enabled.append(t)
        if len(stage1_enabled) > 0 and len(enabled) == 0:
            # #region agent log
            self._debug_log(
                hypothesis_id="H3",
                message="stage2_all_filtered",
                data={
                    "time": int(self.time),
                    "stage1_enabled": [self.id2t_name[idx] for idx in stage1_enabled],
                    "blocked_by_cleaning": blocked_by_cleaning,
                    "d_tm_tokens": len(d_place.tokens),
                    "dwell_time": int(dwell_time),
                    "d_tm_head_stay": int(d_place.head().stay_time) if len(d_place.tokens) > 0 else None,
                },
            )
            # #endregion
        return enabled

    def _get_enable_t(self) -> List[int]:
        stage1_enabled = self._get_enable_t_stage1()
        return self._apply_enable_stage2(stage1_enabled)

    def _is_deadlock_state(self, stage1_enabled: Optional[List[int]] = None) -> bool:
        if len(self._get_place("LP_done").tokens) >= self.n_wafer:
            return False
        if stage1_enabled is None:
            self._update_marking_vector()
            stage1_enabled = self._get_enable_t_stage1()
        return len(stage1_enabled) == 0

    def get_enable_t(self) -> List[int]:
        self._update_marking_vector()
        return self._get_enable_t()

    def _track_enter(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            self._token_stats[token.token_id] = {"enter_system": None, "exit_system": None, "chambers": {}}
        if place_name == "PM1" and self._token_stats[token.token_id]["enter_system"] is None:
            self._token_stats[token.token_id]["enter_system"] = self.time
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": self.time, "exit": None})

    def _track_leave(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            return
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": None, "exit": None})["exit"] = self.time
        if place_name == "LP_done":
            self._token_stats[token.token_id]["exit_system"] = self.time

    def _fire(self, t_idx: int, start_time: int, end_time: int) -> Dict[str, Any]:
        t_name = self.id2t_name[t_idx]
        pre_places = np.flatnonzero(self.pre[:, t_idx] > 0)
        pst_places = np.flatnonzero(self.pst[:, t_idx] > 0)
        if pre_places.size == 0 or pst_places.size == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
        pre_place = self.marks[int(pre_places[0])]
        pst_place = self.marks[int(pst_places[0])]
        if len(pre_place.tokens) == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}

        tok = pre_place.pop_head()
        wafer_id = int(getattr(tok, "token_id", -1))
        self._track_leave(tok, pre_place.name)
        tok.enter_time = self.time
        tok.stay_time = 0

        if t_name.startswith("u_"):
            src = t_name[2:]
            dst_level_targets = tuple(self._u_targets.get(src, []))
            setattr(tok, "_dst_level_targets", dst_level_targets)
            setattr(tok, "_dst_level_full_on_pick", self._is_dst_level_full(src))
            dst = self._select_target_for_source(src)
            if dst is not None:
                setattr(tok, "_target_place", dst)
            tok.machine = int(self._next_robot_machine())
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
            step_map = {"PM1": 1, "PM3": 2, "PM4": 2, "LP_done": 3}
            tok.step = max(int(getattr(tok, "step", 0)), step_map.get(target, 0))
            self._track_enter(tok, target)
            if target == "LP_done":
                self.done_count += 1
                self._per_wafer_reward += float(self.R_done)
            elif target in self._chamber_timeline and wafer_id >= 0:
                idx = len(self._chamber_timeline[target])
                self._chamber_timeline[target].append((end_time, None, wafer_id))
                self._chamber_active[target][wafer_id] = idx

        tok.where = int(getattr(tok, "where", 0)) + 1
        pst_place.append(tok)
        self._update_marking_vector()
        return {
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(end_time),
            "token_id": wafer_id,
        }

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
            parts["time_cost"] = -float(dt * self.time_coef)
        
        # 加工奖励：每个加工位上每个晶圆根据加工进度给予奖励
        if self.reward_config.get("proc_reward", 1):
            for p in self.marks:
                if p.type != 1 or len(p.tokens) == 0 or p.processing_time <= 0:
                    continue
                remain = max(0, p.processing_time - int(p.head().stay_time))
                progress = min(dt, remain)
                parts["proc_reward"] += self.processing_reward_coef * float(progress)

        # 与 pn.py 对齐：运输位(type=2, 单设备主要是 d_TM1)超过 D_Residual_time 后按超时秒数惩罚
        if self.reward_config.get("transport_penalty", 1):
            for p in self.marks:
                if p.type != 2 or len(p.tokens) == 0:
                    continue
                for tok in p.tokens:
                    deadline = int(tok.enter_time) + int(self.D_Residual_time)
                    over_start = max(int(t1), deadline)
                    if int(t2) > over_start:
                        parts["penalty"] -= float(int(t2) - over_start) * float(self.transport_overtime_coef)

        for p in self.marks:
            if p.type != 1 or len(p.tokens) == 0:
                continue
            left = p.processing_time + self.P_Residual_time - p.head().stay_time
            if self.reward_config.get("warn_penalty", 1) and left <= self.T_warn:
                parts["warn_penalty"] -= float(self.a_warn)
            if self.reward_config.get("safe_reward", 1) and left > self.T_safe:
                parts["safe_reward"] += float(self.b_safe)

        total = sum(parts.values())
        if detailed:
            parts["total"] = float(total)
            return parts
        return float(total)

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self.time = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log.clear()
        self._per_wafer_reward = 0.0
        self._token_stats = {}
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        self._last_deadlock = False
        self._chamber_timeline = {"PM1": [], "PM3": [], "PM4": []}
        self._chamber_active = {"PM1": {}, "PM3": {}, "PM4": {}}
        self._init_single_cleaning_state()
        self._update_marking_vector()
        return None, self.get_enable_t()

    def step(
        self,
        a1=None,
        a2=None,
        wait1: bool = False,
        wait2: bool = False,
        with_reward: bool = False,
        detailed_reward: bool = False,
        t: Optional[int] = None,
        wait: Optional[bool] = None,
    ):
        self._last_deadlock = False
        if self.time >= MAX_TIME:
            timeout_reward = {"total": -100.0, "timeout": True} if detailed_reward else -100.0
            return True, timeout_reward, True

        action = t if t is not None else a1
        do_wait = bool(wait) or bool(wait1) or bool(wait2) or action is None

        t1 = self.time
        if do_wait:
            t2 = t1 + 5
            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.time = t2
            self._update_stay_times(5)
            self._advance_cleaning_and_idle(5)
            self._consecutive_wait_time += (t2 - t1)
            if self._consecutive_wait_time >= self.idle_timeout and not self._idle_penalty_applied:
                self._idle_penalty_applied = True
                if detailed_reward:
                    reward_result["idle_timeout_penalty"] = -float(self.idle_penalty)
                    reward_result["total"] -= float(self.idle_penalty)
                else:
                    reward_result -= float(self.idle_penalty)
        else:
            self._consecutive_wait_time = 0
            enabled = set(self.get_enable_t())
            if action not in enabled:
                dt = max(1, self.ttime)
                t2 = t1 + dt
                reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
                if detailed_reward:
                    reward_result["illegal_penalty"] = -5.0
                    reward_result["total"] -= 5.0
                else:
                    reward_result -= 5.0
                self.time = t2
                self._update_stay_times(dt)
                self._advance_cleaning_and_idle(dt)
                return False, reward_result, False

            t2 = t1 + self.ttime
            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.time = t2
            self._update_stay_times(self.ttime)
            self._advance_cleaning_and_idle(self.ttime)
            log_entry = self._fire(int(action), start_time=t1, end_time=t2)
            self.fire_log.append(log_entry)

            # 在线 release 惩罚通道（两阶段训练第一阶段会关闭 no_release_penalty）
            if not self.no_release_penalty and self.reward_config.get("release_violation_penalty", 1):
                latest_idx = len(self.fire_log) - 1
                blame = self.blame_release_violations()
                if latest_idx in blame:
                    pen = float(blame[latest_idx])
                    if detailed_reward:
                        reward_result["release_violation_penalty"] = -pen
                        reward_result["total"] -= pen
                    else:
                        reward_result -= pen

            if self._per_wafer_reward > 0:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] += self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(self._get_place("LP_done").tokens) >= self.n_wafer
        is_scrap, scrap_info = self._check_scrap()
        if is_scrap:
            self.scrap_count += 1
            self.resident_violation_count += 1
            if detailed_reward:
                reward_result["scrap_penalty"] -= float(self.R_scrap)
                reward_result["total"] -= float(self.R_scrap)
                reward_result["scrap_info"] = scrap_info
            else:
                reward_result -= float(self.R_scrap)
            if self.stop_on_scrap:
                return True, reward_result, True

        stage1_enabled = self._get_enable_t_stage1()
        if not finish and self._is_deadlock_state(stage1_enabled):
            # #region agent log
            self._debug_log(
                hypothesis_id="H4",
                message="deadlock_triggered",
                data={
                    "time": int(self.time),
                    "finish": bool(finish),
                    "stage1_enabled": [self.id2t_name[idx] for idx in stage1_enabled],
                    "stage2_enabled": [self.id2t_name[idx] for idx in self._apply_enable_stage2(stage1_enabled)],
                    "lp_tokens": len(self._get_place("LP").tokens),
                    "lp_done_tokens": len(self._get_place("LP_done").tokens),
                    "d_tm_tokens": len(self._get_place("d_TM1").tokens),
                    "pm_states": {
                        p.name: {
                            "tokens": len(p.tokens),
                            "is_cleaning": bool(getattr(p, "is_cleaning", False)),
                            "cleaning_remaining": int(getattr(p, "cleaning_remaining", 0)),
                            "processed_wafer_count": int(getattr(p, "processed_wafer_count", 0)),
                        }
                        for p in self.marks
                        if p.name.startswith("PM")
                    },
                },
            )
            # #endregion
            self.deadlock_count += 1
            self._last_deadlock = True
            deadlock_penalty = float(abs(self.R_scrap))
            if detailed_reward:
                reward_result["deadlock_penalty"] = -deadlock_penalty
                reward_result["total"] -= deadlock_penalty
            else:
                reward_result -= deadlock_penalty
            return True, reward_result, False

        if finish:
            if detailed_reward:
                reward_result["finish_bonus"] += float(self.R_finish)
                reward_result["total"] += float(self.R_finish)
            else:
                reward_result += float(self.R_finish)
        return bool(finish), reward_result, False

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

    def render_gantt(self, out_path: str) -> None:
        return None
