"""
并发三动作 Petri 可视化适配器。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from solutions.A.rl_env import Env_PN_Concurrent

from .algorithm_interface import (
    ActionInfo,
    ChamberState,
    RobotState,
    StateInfo,
    WaferState,
)


def _is_transport_place_name(name: str) -> bool:
    return str(name).startswith("d_TM") or str(name) in {"TM1", "TM2", "TM3"}


def _normalize_transport_name(name: str) -> str:
    raw = str(name)
    if raw.endswith("TM1"):
        return "TM1"
    if raw.endswith("TM2"):
        return "TM2"
    if raw.endswith("TM3"):
        return "TM3"
    return raw


class PetriAdapter:
    """Petri 网算法适配器。"""

    def __init__(self, env: Env_PN_Concurrent, step_verbose: bool = True) -> None:
        self.env = env
        self.net = env.net
        self.step_verbose = step_verbose
        self._last_reward_detail: Dict[str, float] = {}
        self._last_action_history: List[Dict[str, Any]] = []
        self._transition_names = list(getattr(self.net, "id2t_name", []))
        self._transport_names = self._collect_transport_names()
        self._transition_transport_map: Dict[int, str] = {}
        self._transition_transport_map = self._build_transition_transport_map()

    def reset(self) -> StateInfo:
        self.env.reset()
        self._last_action_history.clear()
        self._last_reward_detail = {}
        return self._collect_state_info()

    def step(self, action: int | Tuple[int, ...]) -> Tuple[StateInfo, float, bool, Dict]:
        """
        支持单动作与并发动作。
        - int: 全局变迁索引或 WAIT
        - `(a1, a2, a3)`: 并发占位；**仿真仅消费 TM2/TM3**（`-1` 表示 WAIT），TM1 由 `ClusterTool` 规则自动执行。
          History 与 verbose 中 TM1 文案来自 step 前 `_cached_auto_tm1_action`，与 `a1` 无关。
        """
        a1, a2, a3 = self._normalize_action(action)
        tm2_action = None if a2 == -1 else int(a2)
        tm3_action = None if a3 == -1 else int(a3)

        cached_t = getattr(self.net, "_cached_auto_tm1_action", None)
        disp_tm1 = -1 if cached_t is None else int(cached_t)

        if self.step_verbose:
            t = getattr(self.net, "time", 0)
            a1_name = "WAIT" if disp_tm1 == -1 else self.get_action_name(disp_tm1)
            a2_name = "WAIT" if a2 == -1 else self.get_action_name(a2)
            a3_name = "WAIT" if a3 == -1 else self.get_action_name(a3)
            print(f"\n--- Step (t={t}) TM1:{a1_name} | TM2:{a2_name} | TM3:{a3_name} ---")

        result = self._call_net_step_tm2_tm3(tm2_action, tm3_action)
        done, reward_result, scrap, _action_mask, _obs = self._unpack_step_result(result)

        self._last_reward_detail = {}
        reward = float(reward_result)

        if not done:
            done = getattr(self.net, "done_count", 0) >= getattr(self.net, "n_wafer", 0)

        state_info = self._collect_state_info()
        info = {"done": bool(done), "reward": reward, "scrap": bool(scrap), "detail": dict(self._last_reward_detail)}

        self._last_action_history.append(
            {
                "step": len(self._last_action_history) + 1,
                "action": self._format_history_action(disp_tm1, a2, a3),
                "reward": reward,
                "detail": dict(self._last_reward_detail),
            }
        )
        return state_info, reward, bool(done), info

    def get_action_name(self, action: int) -> str:
        if action == self.action_space_size:
            return "WAIT_5s"
        if 0 <= action < len(self._transition_names):
            return self._transition_names[int(action)]
        return f"UNKNOWN_{action}"

    def get_enabled_actions(self) -> List[ActionInfo]:
        enabled_map = self._enabled_transition_indices_by_transport()
        enabled_t = set(enabled_map["TM1"]) | set(enabled_map["TM2"]) | set(enabled_map["TM3"])
        actions: List[ActionInfo] = []
        for t in range(self.action_space_size):
            enabled = t in enabled_t
            actions.append(
                ActionInfo(
                    action_id=t,
                    action_name=self.get_action_name(t),
                    enabled=enabled,
                    description="" if enabled else "当前条件不满足",
                )
            )
        actions.append(
            ActionInfo(
                action_id=self.action_space_size,
                action_name="WAIT_5s",
                enabled=True,
                description="",
            )
        )
        return actions

    def get_enabled_actions_by_robot(self) -> Tuple[List[ActionInfo], List[ActionInfo], List[ActionInfo]]:
        """返回 TM1/TM2/TM3 各自的可用动作列表。"""
        enabled_map = self._enabled_transition_indices_by_transport()
        tm1_enabled = set(enabled_map["TM1"])
        tm2_enabled = set(enabled_map["TM2"])
        tm3_enabled = set(enabled_map["TM3"])

        tm1_actions: List[ActionInfo] = []
        tm2_actions: List[ActionInfo] = []
        tm3_actions: List[ActionInfo] = []

        for t in range(self.action_space_size):
            transport = self._transition_transport_name(t)
            if transport == "TM1":
                enabled = t in tm1_enabled
                tm1_actions.append(
                    ActionInfo(
                        action_id=t,
                        action_name=self.get_action_name(t),
                        enabled=enabled,
                        description="" if enabled else "当前条件不满足",
                    )
                )
            elif transport == "TM2":
                enabled = t in tm2_enabled
                tm2_actions.append(
                    ActionInfo(
                        action_id=t,
                        action_name=self.get_action_name(t),
                        enabled=enabled,
                        description="" if enabled else "当前条件不满足",
                    )
                )
            elif transport == "TM3":
                enabled = t in tm3_enabled
                tm3_actions.append(
                    ActionInfo(
                        action_id=t,
                        action_name=self.get_action_name(t),
                        enabled=enabled,
                        description="" if enabled else "当前条件不满足",
                    )
                )

        tm1_actions.append(ActionInfo(action_id=-1, action_name="WAIT_5s", enabled=True, description=""))
        tm2_actions.append(ActionInfo(action_id=-1, action_name="WAIT_5s", enabled=True, description=""))
        tm3_actions.append(ActionInfo(action_id=-1, action_name="WAIT_5s", enabled=True, description=""))
        return tm1_actions, tm2_actions, tm3_actions

    def get_reward_breakdown(self) -> Dict[str, float]:
        return dict(self._last_reward_detail or {})

    @property
    def action_space_size(self) -> int:
        return int(getattr(self.net, "T", 0))

    def get_current_state(self) -> StateInfo:
        return self._collect_state_info()

    def render_gantt(self, output_path: str) -> bool:
        try:
            self.net.render_gantt(output_path)
            return True
        except Exception:
            return False

    def export_action_sequence(self) -> List[Dict[str, Any]]:
        return list(self._last_action_history)

    def _normalize_action(self, action: int | Tuple[int, ...]) -> Tuple[int, int, int]:
        if isinstance(action, tuple):
            a1 = int(action[0]) if len(action) > 0 else -1
            a2 = int(action[1]) if len(action) > 1 else -1
            a3 = int(action[2]) if len(action) > 2 else -1
            return a1, a2, a3

        raw = int(action)
        if raw == self.action_space_size:
            return -1, -1, -1
        transport = self._transition_transport_name(raw)
        if transport == "TM1":
            return raw, -1, -1
        if transport == "TM2":
            return -1, raw, -1
        if transport == "TM3":
            return -1, -1, raw
        return raw, -1, -1

    def _call_net_step_tm2_tm3(self, a2: Optional[int], a3: Optional[int]):
        return self.net.step(a2=a2, a3=a3)

    @staticmethod
    def _unpack_step_result(result):
        if isinstance(result, tuple) and len(result) >= 5:
            return result[0], result[1], result[2], result[3], result[4]
        if isinstance(result, tuple) and len(result) == 3:
            done, reward_result, scrap = result
            return done, reward_result, scrap, None, None
        raise TypeError(f"Unsupported step return type: {type(result)}")

    def _format_history_action(self, a1: int, a2: int, a3: int) -> str:
        a1_name = "WAIT" if a1 == -1 else self.get_action_name(a1)
        a2_name = "WAIT" if a2 == -1 else self.get_action_name(a2)
        a3_name = "WAIT" if a3 == -1 else self.get_action_name(a3)
        return f"TM1:{a1_name} | TM2:{a2_name} | TM3:{a3_name}"

    def _collect_state_info(self) -> StateInfo:
        place_states: Dict[str, ChamberState] = {}
        transport_states: Dict[str, ChamberState] = {}
        release_schedule: Dict[str, list] = {}

        # 仅由下方循环 _merge_alias_state 填充；勿先 _build_alias_state(全量) 再合并，否则 LP/LP_done 晶圆与容量会翻倍。
        start_alias = self._build_alias_state("LP", [], "start")
        end_alias = self._build_alias_state("LP_done", [], "end")

        for idx, place in enumerate(self.net.marks):
            wafers = self._collect_wafers(idx, place)
            release_schedule[place.name] = list(getattr(place, "release_schedule", []))

            if _is_transport_place_name(place.name):
                transport_name = _normalize_transport_name(place.name)
                transport_states[transport_name] = ChamberState(
                    name=transport_name,
                    place_idx=int(idx),
                    capacity=int(getattr(place, "capacity", 0)),
                    wafers=wafers,
                    proc_time=0.0,
                    status="active" if wafers else "idle",
                    chamber_type="transport",
                )
                continue

            if place.name in self._start_place_names():
                start_alias = self._merge_alias_state(start_alias, place, wafers)
                continue
            if place.name in self._end_place_names():
                end_alias = self._merge_alias_state(end_alias, place, wafers)
                continue

            chamber_type = "processing"
            if self._is_start_buffer_name(place.name):
                chamber_type = "start"
            elif self._is_end_buffer_name(place.name):
                chamber_type = "end"
            place_states[place.name] = ChamberState(
                name=place.name,
                place_idx=int(idx),
                capacity=int(getattr(place, "capacity", 0)),
                wafers=wafers,
                proc_time=float(getattr(place, "processing_time", 0)),
                status=self._calc_chamber_status(place.name, wafers, chamber_type, place),
                chamber_type=chamber_type,
                cleaning_remaining=float(getattr(place, "cleaning_remaining", 0.0)),
                inbound_blocked=bool(getattr(place, "is_cleaning", False)),
                cleaning_wafer_countdown=self._cleaning_countdown(place),
            )

        chambers = [start_alias] + list(place_states.values()) + [end_alias]
        chambers = [c for c in chambers if c is not None]
        transport_buffers = list(transport_states.values())
        robot_states = {
            name: RobotState(
                name=name,
                busy=bool(transport_states.get(name, ChamberState("", -1, 0)).wafers),
                wafers=list(transport_states.get(name, ChamberState("", -1, 0)).wafers),
            )
            for name in ("TM1", "TM2", "TM3")
        }

        done_count = int(getattr(self.net, "done_count", 0))
        total_wafers = int(getattr(self.net, "n_wafer", 0))
        in_progress = max(0, total_wafers - done_count)
        self.display_chambers = [c.name for c in chambers if c.name]

        return StateInfo(
            time=float(getattr(self.net, "time", 0)),
            chambers=chambers,
            transport_buffers=transport_buffers,
            start_buffers=[start_alias],
            end_buffers=[end_alias],
            robot_states=robot_states,
            enabled_actions=self.get_enabled_actions(),
            done_count=done_count,
            total_wafers=total_wafers,
            tpt_wph=(float(done_count) / float(getattr(self.net, "time", 1e-9))) * 3600
            if float(getattr(self.net, "time", 0)) > 0
            else 0.0,
            stats={
                "release_schedule": release_schedule,
                "system_avg": 0.0,
                "system_max": 0,
                "system_diff": 0.0,
                "completed_count": done_count,
                "in_progress_count": in_progress,
                "chambers": {},
                "transports": {},
                "transports_detail": {},
                "resident_violation_count": int(getattr(self.net, "resident_violation_count", 0)),
                "qtime_violation_count": int(getattr(self.net, "qtime_violation_count", 0)),
            },
        )

    def _build_alias_state(self, name: str, source_places: List[str], chamber_type: str) -> ChamberState:
        wafers: List[WaferState] = []
        place_idx = -1
        capacity = 0
        proc_time = 0.0
        for place_name in source_places:
            place = self._get_place_by_name(place_name)
            if place is None:
                continue
            place_idx = int(self.net.id2p_name.index(place_name))
            capacity += int(getattr(place, "capacity", 0))
            proc_time = max(proc_time, float(getattr(place, "processing_time", 0)))
            wafers.extend(self._collect_wafers(place_idx, place))
        status = "active" if wafers else "idle"
        return ChamberState(
            name=name,
            place_idx=place_idx,
            capacity=capacity,
            wafers=wafers,
            proc_time=proc_time,
            status=status,
            chamber_type=chamber_type,
            cleaning_wafer_countdown=-1,
        )

    @staticmethod
    def _merge_alias_state(alias_state: ChamberState, place, wafers: List[WaferState]) -> ChamberState:
        alias_state.place_idx = int(getattr(alias_state, "place_idx", -1)) if alias_state.place_idx >= 0 else int(alias_state.place_idx)
        alias_state.capacity += int(getattr(place, "capacity", 0))
        alias_state.proc_time = max(alias_state.proc_time, float(getattr(place, "processing_time", 0)))
        alias_state.wafers.extend(wafers)
        alias_state.status = "active" if alias_state.wafers else "idle"
        return alias_state

    def _collect_wafers(self, p_idx: int, place) -> List[WaferState]:
        wafers: List[WaferState] = []
        for tok in place.tokens:
            token_id = int(getattr(tok, "token_id", -1))
            if token_id < 0:
                continue
            wafers.append(
                WaferState(
                    token_id=token_id,
                    place_name=str(place.name),
                    place_idx=int(p_idx),
                    place_type=int(getattr(place, "type", 0)),
                    stay_time=float(getattr(tok, "stay_time", 0)),
                    proc_time=float(getattr(place, "processing_time", 0)),
                    time_to_scrap=self._calc_time_to_scrap(place, float(getattr(tok, "stay_time", 0))),
                    route_id=int(getattr(tok, "route_type", 0)),
                    step=int(getattr(tok, "step", 0)),
                )
            )
        return wafers

    def _calc_time_to_scrap(self, place, stay_time: float) -> float:
        proc_time = float(getattr(place, "processing_time", 0))
        name = str(getattr(place, "name", ""))
        if name in {"LLC", "LLD"}:
            return float(proc_time + getattr(self.net, "P_Residual_time", 0) * 3 - stay_time)
        if proc_time > 0:
            return float(proc_time + getattr(self.net, "P_Residual_time", 0) - stay_time)
        if _is_transport_place_name(name):
            return float(getattr(self.net, "D_Residual_time", 0) - stay_time)
        return -1.0

    def _calc_chamber_status(self, name: str, wafers: List[WaferState], chamber_type: str, place) -> str:
        if not wafers:
            return "idle"
        if chamber_type == "transport":
            return "active"
        min_time_to_scrap = min((w.time_to_scrap for w in wafers if w.time_to_scrap >= 0), default=9999)
        if min_time_to_scrap <= 0:
            return "danger"
        if min_time_to_scrap <= 5:
            return "warning"
        return "active"

    def _collect_robot_states(self, transport_states: Dict[str, ChamberState]) -> Dict[str, RobotState]:
        tm2_wafers = list(transport_states.get("TM2", ChamberState("", -1, 0)).wafers)
        tm3_wafers = list(transport_states.get("TM3", ChamberState("", -1, 0)).wafers)
        tm1_wafers = list(transport_states.get("TM1", ChamberState("", -1, 0)).wafers)
        return {
            "TM1": RobotState(name="TM1", busy=bool(tm1_wafers), wafers=tm1_wafers),
            "TM2": RobotState(name="TM2", busy=bool(tm2_wafers), wafers=tm2_wafers),
            "TM3": RobotState(name="TM3", busy=bool(tm3_wafers), wafers=tm3_wafers),
        }

    def _collect_transport_names(self) -> List[str]:
        names: List[str] = []
        for place in getattr(self.net, "marks", []):
            if not _is_transport_place_name(getattr(place, "name", "")):
                continue
            transport = _normalize_transport_name(getattr(place, "name", ""))
            if transport not in names:
                names.append(transport)
        if not names:
            names = ["TM1", "TM2", "TM3"]
        return names

    def _build_transition_transport_map(self) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for t_idx, _name in enumerate(self._transition_names):
            transport = self._transition_transport_name(t_idx)
            if transport is not None:
                mapping[int(t_idx)] = transport
        return mapping

    def _transition_transport_name(self, t_idx: int) -> Optional[str]:
        cache = getattr(self, "_transition_transport_map", None)
        if isinstance(cache, dict):
            cached = cache.get(int(t_idx))
            if cached is not None:
                return cached
        for place_idx in self._transition_place_indices(t_idx, pre=True):
            transport = self._normalize_place_to_transport(int(place_idx))
            if transport is not None:
                return transport
        for place_idx in self._transition_place_indices(t_idx, pre=False):
            transport = self._normalize_place_to_transport(int(place_idx))
            if transport is not None:
                return transport
        return None

    def _transition_place_indices(self, t_idx: int, pre: bool) -> List[int]:
        key = "_pre_place_indices" if pre else "_pst_place_indices"
        indexed = getattr(self.net, key, None)
        if indexed is not None:
            try:
                if 0 <= t_idx < len(indexed):
                    return [int(x) for x in list(indexed[t_idx])]
            except Exception:
                pass

        matrix_name = "pre" if pre else "pst"
        matrix = getattr(self.net, matrix_name, None)
        if matrix is None:
            return []
        try:
            column = matrix[:, int(t_idx)]
            return [int(x) for x in np.flatnonzero(column)]
        except Exception:
            return []

    def _normalize_place_to_transport(self, place_idx: int) -> Optional[str]:
        place = self._get_place_by_index(place_idx)
        if place is None:
            return None
        name = _normalize_transport_name(getattr(place, "name", ""))
        if name in {"TM1", "TM2", "TM3"}:
            return name
        return None

    def _enabled_transition_indices_by_transport(self) -> Dict[str, List[int]]:
        mask = None
        if hasattr(self.net, "get_action_mask"):
            try:
                mask = self.net.get_action_mask(
                    wait_action_start=self.action_space_size,
                    n_actions=self.action_space_size + 1,
                    concurrent=False,
                )
            except TypeError:
                try:
                    mask = self.net.get_action_mask()
                except Exception:
                    mask = None

        enabled: Dict[str, List[int]] = {"TM1": [], "TM2": [], "TM3": []}
        if mask is None:
            return enabled
        for t_idx in range(min(self.action_space_size, len(mask))):
            if not bool(mask[t_idx]):
                continue
            transport = self._transition_transport_name(t_idx)
            if transport in enabled:
                enabled[transport].append(int(t_idx))
        return enabled

    def _get_place_by_name(self, name: str):
        for place in getattr(self.net, "marks", []):
            if getattr(place, "name", None) == name:
                return place
        return None

    def _get_place_by_index(self, idx: int):
        marks = getattr(self.net, "marks", [])
        if 0 <= idx < len(marks):
            return marks[idx]
        return None

    def _start_place_names(self) -> List[str]:
        return [str(place.name) for place in getattr(self.net, "marks", []) if self._is_start_buffer_name(str(place.name))]

    def _end_place_names(self) -> List[str]:
        return [str(place.name) for place in getattr(self.net, "marks", []) if self._is_end_buffer_name(str(place.name))]

    @staticmethod
    def _is_start_buffer_name(name: str) -> bool:
        return str(name).startswith("LP") and str(name) != "LP_done"

    @staticmethod
    def _is_end_buffer_name(name: str) -> bool:
        return str(name) == "LP_done"

    def _cleaning_countdown(self, place) -> int:
        trigger_map = getattr(self.net, "_cleaning_trigger_map", None)
        if isinstance(trigger_map, dict):
            trigger = int(trigger_map.get(getattr(place, "name", ""), 0))
            if trigger > 0:
                processed = int(getattr(place, "processed_wafer_count", 0))
                return max(0, trigger - processed)
        return -1
