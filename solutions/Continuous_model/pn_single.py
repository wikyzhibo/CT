"""
单设备 Petri 网（构网驱动、单机械手、单动作）。
执行链：construct_single -> _get_enable_t -> step -> calc_reward
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.construct_single import build_single_device_net
from solutions.Continuous_model.pn import Place

MAX_TIME = 4000


class PetriSingleDevice:
    def __init__(self, config: Optional[PetriEnvConfig] = None) -> None:
        if config is None:
            config = PetriEnvConfig()
        self.config = config

        self.n_wafer = int(config.n_wafer)
        self.c_time = int(getattr(config, "c_time", 1))
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
        self.stop_on_scrap = bool(getattr(config, "stop_on_scrap", True))
        self.reward_config = dict(getattr(config, "reward_config", {}))

        info = build_single_device_net(n_wafer=self.n_wafer, ttime=max(1, self.T_transport))
        self.pre: np.ndarray = info["pre"]
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

        self.time = 0
        self.done_count = 0
        self.scrap_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log: List[Dict[str, Any]] = []
        self.enable_statistics = True
        self._per_wafer_reward = 0.0
        self._token_stats: Dict[int, Dict[str, Any]] = {}

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
            )
            cp.tokens = deque(tok.clone() for tok in p.tokens)
            cloned.append(cp)
        return cloned

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
            for tok in p.tokens:
                tok.stay_time += dt

    def _is_process_ready(self, place_name: str) -> bool:
        place = self._get_place(place_name)
        if len(place.tokens) == 0:
            return False
        if place.processing_time <= 0:
            return True
        return place.head().stay_time >= place.processing_time

    def _next_enable_dt(self) -> int:
        remain: List[int] = []
        # 加工完成事件
        for p in self.marks:
            if p.type != 1 or p.processing_time <= 0 or len(p.tokens) == 0:
                continue
            r = p.processing_time - p.head().stay_time
            if r > 0:
                remain.append(r)
        # 运输位 d_TM1 停留完成事件（到达目标腔室前必须停留 T_transport）
        d_tm = self._get_place("d_TM1")
        if len(d_tm.tokens) > 0:
            r_d = self.T_transport - d_tm.head().stay_time
            if r_d > 0:
                remain.append(r_d)
        return min(remain) if remain else max(1, self.ttime)

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

    def _get_enable_t(self) -> List[int]:
        enabled: List[int] = []
        d_tm = self._get_place("d_TM1")
        head_target = getattr(d_tm.head(), "_target_place", None) if len(d_tm.tokens) > 0 else None

        for t in range(self.T):
            pre_idx = np.flatnonzero(self.pre[:, t] > 0)
            if pre_idx.size == 0:
                continue
            if np.any(self.m[pre_idx] < self.pre[pre_idx, t]):
                continue
            if np.any(self.m + self.net[:, t] > self.k):
                continue

            t_name = self.id2t_name[t]
            if t_name.startswith("u_"):
                _, src, dst = t_name.split("_", 2)
                if not self._is_process_ready(src):
                    continue
                # 单机械手 u_* 需要保证目标腔室可接收，否则会把晶圆堵在 d_TM1 导致无使能动作
                dst_place = self._get_place(dst)
                if len(dst_place.tokens) >= dst_place.capacity:
                    continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                if head_target is not None and head_target != target:
                    continue
                # 与级联设备一致：晶圆在 d_TM1 中需停留一个运输周期后，t_* 才允许发射
                d_place = self._get_place("d_TM1")
                if len(d_place.tokens) > 0 and d_place.head().stay_time < self.T_transport:
                    continue
            enabled.append(t)
        return enabled

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

    def _fire(self, t_idx: int) -> None:
        t_name = self.id2t_name[t_idx]
        pre_places = np.flatnonzero(self.pre[:, t_idx] > 0)
        pst_places = np.flatnonzero(self.pst[:, t_idx] > 0)
        if pre_places.size == 0 or pst_places.size == 0:
            return
        pre_place = self.marks[int(pre_places[0])]
        pst_place = self.marks[int(pst_places[0])]
        if len(pre_place.tokens) == 0:
            return

        tok = pre_place.pop_head()
        self._track_leave(tok, pre_place.name)
        tok.enter_time = self.time
        tok.stay_time = 0

        if t_name.startswith("u_"):
            _, _, dst = t_name.split("_", 2)
            setattr(tok, "_target_place", dst)
        elif t_name.startswith("t_"):
            target = t_name[2:]
            if hasattr(tok, "_target_place"):
                delattr(tok, "_target_place")
            step_map = {"PM1": 1, "PM3": 2, "PM4": 2, "PM5": 3, "LP_done": 4}
            tok.step = max(int(getattr(tok, "step", 0)), step_map.get(target, 0))
            self._track_enter(tok, target)
            if target == "LP_done":
                self.done_count += 1
                self._per_wafer_reward += float(self.R_done)

        pst_place.append(tok)
        self._update_marking_vector()

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

        if self.reward_config.get("time_cost", 1):
            parts["time_cost"] = -float(dt * self.c_time)

        if self.reward_config.get("proc_reward", 1):
            for p in self.marks:
                if p.type != 1 or len(p.tokens) == 0 or p.processing_time <= 0:
                    continue
                remain = max(0, p.processing_time - int(p.head().stay_time))
                progress = min(dt, remain)
                parts["proc_reward"] += 0.2 * float(progress)

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
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log.clear()
        self._per_wafer_reward = 0.0
        self._token_stats = {}
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
        else:
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
                return False, reward_result, False

            t2 = t1 + self.ttime
            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.time = t2
            self._update_stay_times(self.ttime)
            self._fire(int(action))
            self.fire_log.append({"time": self.time, "t_name": self.id2t_name[int(action)]})

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
        }

    def render_gantt(self, out_path: str) -> None:
        return None
