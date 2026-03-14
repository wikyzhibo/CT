"""
单设备 Petri 网（构网驱动、单机械手、单动作）。
执行链：construct_single -> _get_enable_t -> step -> calc_reward
"""

from __future__ import annotations

from collections import deque
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
    "lp_forced_emit_interval_not_met": "u_LP 未达到强制发射间隔",
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
        self.robot_capacity = 2 if int(config.single_robot_capacity) == 2 else 1

        self.cleaning_enabled = bool(config.cleaning_enabled)
        self.cleaning_targets = set(config.cleaning_targets)
        self.cleaning_trigger_wafers = config.cleaning_trigger_wafers
        self.cleaning_duration = max(0, int(config.cleaning_duration))
        self.wait_durations = _normalize_wait_durations(config.wait_durations)
        self.limit_start = bool(getattr(config, "limit_start", True))
        
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
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.ttime = int(np.max(info["ttime"])) if len(info["ttime"]) > 0 else 5

        self.ori_marks: List[Place] = info["marks"]
        self.marks: List[Place] = self._clone_marks(self.ori_marks)
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
        self._cascade_round_robin_pairs: Dict[str, Tuple[str, str]] = {}
        self._cascade_round_robin_next: Dict[str, str] = {}
        self._single_round_robin_pairs: Dict[str, Tuple[str, str]] = {}
        self._single_round_robin_next: Dict[str, str] = {}
        if self.device_mode == "cascade":
            self._cascade_round_robin_pairs["LP"] = ("PM7", "PM8")
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
        self._lp_trigger_interval_history: Deque[float] = deque(maxlen=4)
        self._lp_forced_emit_interval: int = 0
        self._last_u_lp_fire_time: Optional[int] = None
        self._pending_batch_trigger_intervals: List[float] = []
        self._last_batch_trigger_interval_avg: float = 0.0
        self._last_batch_trigger_interval_count: int = 0
        self._init_cleaning_state()
        self._training = True

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

        self._last_deadlock = False
        # ======超出最大运行时间直接判定为超时结束，避免“原地等待”导致的无效步骤======
        if self.time >= self.MAX_TIME:
            timeout_reward = {"total": -100.0, "timeout": True} if detailed_reward else -100.0
            DONE = True
            SCRAPE = True
            action_mask = self.get_action_mask(
                wait_action_start=int(self.T),
                n_actions=int(self.T + len(self.wait_durations)),
            )
            obs = self.get_obs()
            return DONE, timeout_reward, SCRAPE, action_mask, obs

        action = a1
        do_wait = (wait_duration is not None) or action is None

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
                next_event_delta = self.get_next_event_delta()
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
            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.advance_time(actual_dt, event_reason=wait_reason)
            self._check_qtime_violation()
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
            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.advance_time(self.ttime, event_reason="fire_transition")
            self._check_qtime_violation()
            log_entry = self._fire(int(action), start_time=t1, end_time=t2)
            self.fire_log.append(log_entry)

            if self._per_wafer_reward > 0:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] += self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(self._get_place("LP_done").tokens) >= self.n_wafer
        # ====== 违反驻留时间约束 ======
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
                action_mask = self.get_action_mask(
                    wait_action_start=int(self.T),
                    n_actions=int(self.T + len(self.wait_durations)),
                )
                obs = self.get_obs()
                return DONE, reward_result, SCRAPE, action_mask, obs

        # ===== 任务完成奖励 ======
        if finish:
            if detailed_reward:
                reward_result["finish_bonus"] += float(self.finish_event_reward)
                reward_result["total"] += float(self.finish_event_reward)
            else:
                reward_result += float(self.finish_event_reward)
            SCRAPE = False
        action_mask = self.get_action_mask(
            wait_action_start=int(self.T),
            n_actions=int(self.T + len(self.wait_durations)),
        )
        obs = self.get_obs()
        return bool(finish), reward_result, SCRAPE, action_mask, obs

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self._refresh_episode_proc_time()
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
        self._last_u_lp_fire_time = None
        self._init_cleaning_state()
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        return None, self.get_enable_t()

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

    def get_obs(self) -> np.ndarray:
        obs: List[float] = []
        for name in self._get_obs_place_order():
            if name not in self.id2p_name:
                continue
            obs.extend(self._get_place(name).get_obs())
        return np.array(obs, dtype=np.float32)

    def get_obs_dim(self) -> int:
        return sum(
            self._get_place(name).get_obs_dim()
            for name in self._get_obs_place_order()
            if name in self.id2p_name
        )

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
        where_before = int(getattr(tok, "where", 0))
        self._track_leave(tok, pre_place.name)
        tok.enter_time = self.time
        tok.stay_time = 0

        if t_name.startswith("u_"):
            src = t_name[2:]
            if t_name == "u_LP":
                self._last_u_lp_fire_time = int(start_time)
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

        tok.where = int(getattr(tok, "where", 0)) + 1
        pst_place.append(tok)
        pre_place_idx = int(pre_places[0])
        pst_place_idx = int(pst_places[0])
        self.m[pre_place_idx] -= 1
        self.m[pst_place_idx] += 1
        return {
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(end_time),
            "token_id": wafer_id,
        }

    def get_enable_t(self) -> List[int]:
        """
        Stage1: 基础使能（pre/pst + 容量 + 防死锁规则），不做“加工完成”就绪检查。
        """

        # step 1: 若为双臂，预先锁定一些u_*变迁
        stage1_enabled: List[int] = []
        d_tm = self._get_place("d_TM1") if ("d_TM1" in self.id2p_name and self.device_mode != "cascade") else None
        head_tok = d_tm.head() if (d_tm is not None and len(d_tm.tokens) > 0) else None

        # 机器手容量为2时，如果 d_TM1 队首有晶圆，则锁定后续 u_* 来源到该晶圆的 dst 层，直到该晶圆离开 d_TM1。
        locked_sources: set[str] = set()
        if self.robot_capacity == 2 and self.device_mode != "cascade" and head_tok is not None:
            locked_sources = set(getattr(head_tok, "_dst_level_targets", ()))

        # step2: 遍历变迁，结构性使能
        # =====self.m >= self.pre[:, t]=======
        # =====self.m + self.pst[:, t] <= self.k=======
        for t in range(self.T):
            base_pre = self.pre[:, t]
            base_pre_idx = np.flatnonzero(base_pre > 0)
            if base_pre_idx.size == 0:
                continue
            t_name = self.id2t_name[t]
            color_pre = base_pre
            head_where = 0
            color_idx = 0
            transport_pre_idx = np.flatnonzero(
                (base_pre > 0) & np.array([name.startswith("d_") for name in self.id2p_name], dtype=bool)
            )
            if transport_pre_idx.size > 0:
                tp_idx = int(transport_pre_idx[0])
                tp = self.marks[tp_idx]
                tp_head = tp.head() if len(tp.tokens) > 0 else None
                head_where = int(getattr(tp_head, "where", 0)) if tp_head is not None else 0
                color_idx = int(np.clip(head_where, 0, self.pre_color.shape[2] - 1))
                color_pre = self.pre_color[:, t, color_idx]
            if np.any((base_pre > 0) & (color_pre <= 0)):
                continue
            if np.any(self.m[base_pre_idx] < color_pre[base_pre_idx]):
                continue
            if np.any(self.m + self.net[:, t] > self.k):
                continue
            if t_name.startswith("u_"):
                src = t_name[2:]
                if self.robot_capacity == 2 and self.device_mode != "cascade":
                    # 双臂规则2（更新）：只要 d_TM1 队首有晶圆，就锁定后续 u_* 来源到该晶圆的 dst 层。
                    if d_tm is not None and len(d_tm.tokens) > 0 and locked_sources and src not in locked_sources:
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
                # t_*类型变迁，取队首token._target_place进行目标检查
                target = t_name[2:]
                transport_name = self._transport_for_t_target(target)
                tp = self._get_place(transport_name)
                tp_head = tp.head() if len(tp.tokens) > 0 else None
                tp_head_target = getattr(tp_head, "_target_place", None) if tp_head is not None else None
                if tp_head_target is not None and tp_head_target != target:
                    continue
            stage1_enabled.append(t)


        """
        Stage2: 在 Stage1 基础上做就绪过滤（加工完成 + 运输位 dwell）。
        """
        stage2_enabled: List[int] = []
        for t in stage1_enabled:
            t_name = self.id2t_name[t]
            if t_name.startswith("u_"):
                src = t_name[2:]
                if self.limit_start and t_name == "u_LP" and self._lp_forced_emit_interval > 0 and self._last_u_lp_fire_time is not None:
                    elapsed = int(self.time) - int(self._last_u_lp_fire_time)
                    if elapsed < int(self._lp_forced_emit_interval):
                        continue
                if not self._is_process_ready(src):
                    continue
                if self._select_target_for_source(src) is None:
                    continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                target_place = self._get_place(target)
                if bool(getattr(target_place, "is_cleaning", False)):
                    continue
                d_place = self._get_place(self._transport_for_t_target(target))
                dwell_time = max(0, int(getattr(d_place, "processing_time", self.T_transport)))
                if len(d_place.tokens) > 0 and d_place.head().stay_time < dwell_time:
                    continue
            stage2_enabled.append(t)
        return stage2_enabled

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

        # 遍历变迁
        for t in range(self.T):
            base_pre = self.pre[:, t]
            base_pre_idx = np.flatnonzero(base_pre > 0)
            t_name = self.id2t_name[t]

            if base_pre_idx.size == 0:
                disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                continue

            color_pre = base_pre
            transport_pre_idx = np.flatnonzero(
                (base_pre > 0) & np.array([name.startswith("d_") for name in self.id2p_name], dtype=bool)
            )
            if transport_pre_idx.size > 0:
                tp_idx = int(transport_pre_idx[0])
                tp = self.marks[tp_idx]
                tp_head = tp.head() if len(tp.tokens) > 0 else None
                head_where = int(getattr(tp_head, "where", 0)) if tp_head is not None else 0
                color_idx = int(np.clip(head_where, 0, self.pre_color.shape[2] - 1))
                color_pre = self.pre_color[:, t, color_idx]

            if np.any((base_pre > 0) & (color_pre <= 0)):
                disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                continue
            if np.any(self.m[base_pre_idx] < color_pre[base_pre_idx]):
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
                if self.limit_start and t_name == "u_LP" and self._lp_forced_emit_interval > 0 and self._last_u_lp_fire_time is not None:
                    elapsed = int(self.time) - int(self._last_u_lp_fire_time)
                    if elapsed < int(self._lp_forced_emit_interval):
                        disabled.append({"action": t, "name": t_name, "reason": "lp_forced_emit_interval_not_met"})
                        continue
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

    @staticmethod
    def _clone_marks(marks: List[Place]) -> List[Place]:
        cloned: List[Place] = []
        for p in marks:
            cp = p.clone()
            cloned.append(cp)
        return cloned

    def _get_place(self, name: str) -> Place:
        for p in self.marks:
            if p.name == name:
                return p
        raise KeyError(f"unknown place: {name}")

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
        关键事件至少覆盖：
        - 某加工完成（避免跨过关键取片决策点）
        - 某运输位达到 T_transport（运输完成，避免跨过关键放片决策点）
        """
        deltas: List[int] = []
        for place in self.marks:
            if len(place.tokens) == 0:
                continue

            # 运输完成事件：d_TM* 队首停留达到 T_transport。
            if place.name.startswith("d_TM"):
                head = place.head()
                delta_tm = int(self.T_transport) - int(getattr(head, "stay_time", 0))
                deltas.append(max(0, int(delta_tm)))
                continue

            if int(getattr(place, "processing_time", 0)) <= 0:
                continue
            if int(place.type) not in (CHAMBER, 5):
                continue
            head = place.head()
            delta = int(place.processing_time) - int(getattr(head, "stay_time", 0))
            deltas.append(max(0, int(delta)))
        if not deltas:
            return None
        min_delta = int(min(deltas))
        return min_delta

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

    def advance_time(self, dt: int, event_reason: str = "") -> None:
        """推进时间并更新相关状态"""
        _ = event_reason  # 预留参数，便于后续追踪不同推进原因。
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        self._update_stay_times(safe_dt)
        self._advance_cleaning_and_idle(safe_dt)

    def _on_processing_unload(self, source_name: str) -> None:
        if not self.cleaning_enabled:
            return
        if source_name not in self.cleaning_targets:
            return
        source_place = self._get_place(source_name)
        source_place.processed_wafer_count = int(getattr(source_place, "processed_wafer_count", 0)) + 1
        source_place.last_proc_type = source_name
        if source_place.is_cleaning:
            return
        if source_place.processed_wafer_count >= self.cleaning_trigger_wafers:
            count = int(source_place.processed_wafer_count)
            source_place.is_cleaning = True
            source_place.cleaning_remaining = int(self.cleaning_duration)
            source_place.cleaning_reason = "processed_wafers"
            source_place.processed_wafer_count = 0
            source_place.idle_time = 0
            self.fire_log.append(
                {
                    "event_type": "cleaning_start",
                    "time": int(self.time),
                    "chamber": source_place.name,
                    "rule": "processed_wafers",
                    "duration": int(self.cleaning_duration),
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
            # 路线2并行腔体采用轮换分配；仅在真实发射时推进轮换指针，避免使能检查污染状态。
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
            if len(available_targets) == 2:
                next_target = self._cascade_round_robin_next.get(source, available_targets[0])
                chosen_target = next_target if next_target in available_targets else available_targets[0]
                if advance_round_robin:
                    self._cascade_round_robin_next[source] = (
                        available_targets[1] if chosen_target == available_targets[0] else available_targets[0]
                    )
                return chosen_target
            if len(available_targets) == 1:
                return available_targets[0]
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
        for p in self.marks:
            if p.type != CHAMBER or len(p.tokens) == 0:
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

    def _check_qtime_violation(self) -> None:
        """检查运输位 Q-Time 超时并按 wafer 去重累计，不施加惩罚。"""
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

    def blame_release_violations(self, batch_finalize: bool = False) -> Dict[int, float]:
        """
        事后追责：基于当前 fire_log 与 _chamber_timeline，回溯可能导致下游容量冲突的 u_* 动作。
        单设备中会按路径代号聚合站点：s1=PM1，s2=PM3∪PM4；若 code=1 则新增 s3=PM6。
        返回 fire_log_index -> penalty。
        """
        if batch_finalize:
            raw_intervals = [float(x) for x in self._pending_batch_trigger_intervals if float(x) > 0.0]
            batch_avg = 0.0
            if raw_intervals:
                mean_raw = sum(raw_intervals) / len(raw_intervals)
                if mean_raw > 0:
                    filtered = [
                        x for x in raw_intervals
                        if abs(x - mean_raw) / mean_raw <= 0.5
                    ]
                else:
                    filtered = list(raw_intervals)
                if not filtered:
                    filtered = list(raw_intervals)
                capped = [min(float(x), 500.0) for x in filtered]
                batch_avg = float(sum(capped) / len(capped))
            if raw_intervals:
                self._lp_trigger_interval_history.append(float(batch_avg))
            self._last_batch_trigger_interval_avg = float(batch_avg)
            self._last_batch_trigger_interval_count = int(len(raw_intervals))
            if self._lp_trigger_interval_history:
                hist = list(self._lp_trigger_interval_history)
                # 加权平均（新到旧权重 4:3:2:1）。
                # deque 顺序是旧->新，因此映射为旧->新权重 1,2,3,4（按可用长度截断）。
                base_weights = [1.0, 2.0, 3.0, 4.0]
                weights = base_weights[-len(hist):]
                weighted_sum = sum(v * w for v, w in zip(hist, weights))
                weight_total = sum(weights)
                weighted_mean = (weighted_sum / weight_total) if weight_total > 0 else 0.0
                self._lp_forced_emit_interval = int(round(weighted_mean))
            else:
                self._lp_forced_emit_interval = 0
            self._pending_batch_trigger_intervals.clear()
            return {}

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
            if not getattr(self, "cleaning_enabled", False) or chamber_name not in getattr(self, "cleaning_targets", set()):
                return []
            out: List[tuple] = []
            for ev in self.fire_log:
                if ev.get("event_type") == "cleaning_start" and ev.get("chamber") == chamber_name:
                    t0 = int(ev.get("time", 0))
                    dur = int(ev.get("duration", self.cleaning_duration))
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
                if t_name == "u_LP":
                    curr_t = int(log.get("t1", 0))
                    prev_t = None
                    for j in range(i - 1, -1, -1):
                        prev_log = self.fire_log[j]
                        if prev_log.get("t_name", "") == "u_LP":
                            prev_t = int(prev_log.get("t1", 0))
                            break
                    if prev_t is not None:
                        interval = curr_t - prev_t
                        if interval > 0:
                            self._pending_batch_trigger_intervals.append(float(interval))
        return blame

    def get_enable_actions(self, wait_action_start: Optional[int] = None) -> List[int]:
        """
        返回完整离散动作可用索引（transition + wait）。
        wait 规则：
        - 默认启用所有 wait 档位；
        - 若存在加工完成待取片晶圆，仅启用 <=5s 的 wait。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)
        enabled = [int(t) for t in self.get_enable_t()]
        restrict_long_wait = self._has_ready_chamber_wafers()
        for offset, duration in enumerate(self.wait_durations):
            if restrict_long_wait and int(duration) > 5:
                continue
            enabled.append(start + int(offset))
        return enabled

    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """
        返回完整离散动作掩码（transition + wait）。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)
        total_actions = int(
            n_actions if n_actions is not None else (start + len(self.wait_durations))
        )
        mask = np.zeros(total_actions, dtype=bool)
        for action in self.get_enable_actions(wait_action_start=start):
            if 0 <= int(action) < total_actions:
                mask[int(action)] = True
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
