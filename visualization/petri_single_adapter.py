"""
单设备 Petri 可视化适配器。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from solutions.Continuous_model.env_single import Env_PN_Single

from .algorithm_interface import (
    ActionInfo,
    AlgorithmAdapter,
    ChamberState,
    RobotState,
    StateInfo,
    WaferState,
)


class PetriSingleAdapter(AlgorithmAdapter):
    def __init__(self, env: Env_PN_Single) -> None:
        self.env = env
        self.net = env.net
        self._last_reward_detail: Dict[str, float] = {}
        self._history: List[Dict[str, Any]] = []
        self.disabled_chambers = {"PM2", "PM5", "PM6"}

    def reset(self) -> StateInfo:
        self.env.reset()
        self._history.clear()
        return self._collect_state_info()

    def step(self, action: int | Tuple[int, int]):
        if isinstance(action, tuple):
            action = action[0] if action[0] != -1 else self.net.T
        if action == self.net.T:
            done, reward_result, scrap = self.net.step(wait=True, with_reward=True, detailed_reward=True)
        else:
            done, reward_result, scrap = self.net.step(t=action, with_reward=True, detailed_reward=True)

        reward = float(reward_result.get("total", 0.0)) if isinstance(reward_result, dict) else float(reward_result)
        self._last_reward_detail = (
            {k: float(v) for k, v in reward_result.items() if isinstance(v, (int, float))}
            if isinstance(reward_result, dict)
            else {"total": reward}
        )
        self._history.append(
            {
                "step": len(self._history) + 1,
                "action": self.get_action_name(action),
                "reward": reward,
                "detail": dict(self._last_reward_detail),
            }
        )
        state = self._collect_state_info()
        info = {"done": bool(done), "reward": reward, "scrap": bool(scrap), "detail": dict(self._last_reward_detail)}
        return state, reward, bool(done), info

    def get_action_name(self, action: int) -> str:
        if action == self.net.T:
            return "WAIT"
        if 0 <= action < len(self.net.id2t_name):
            name = self.net.id2t_name[action]
            if name.startswith("u_"):
                parts = name.split("_")
                # 兼容两种命名：
                # 1) 旧格式 u_src_dst
                # 2) 新格式 u_src（目标由后续 t_* 分流）
                if len(parts) >= 3:
                    _, src, dst = name.split("_", 2)
                    return f"{src}→{dst}"
                if len(parts) == 2:
                    src = parts[1]
                    targets = list(getattr(self.net, "_u_targets", {}).get(src, []))
                    if len(targets) == 1:
                        return f"{src}→{targets[0]}"
                    if len(targets) > 1:
                        return f"{src}→({'|'.join(targets)})"
                    return f"{src}→?"
            if name.startswith("t_"):
                return name[2:]
            return name
        return f"UNKNOWN_{action}"

    def get_enabled_actions(self) -> List[ActionInfo]:
        enabled = set(self.net.get_enable_t())
        actions: List[ActionInfo] = []
        for i, name in enumerate(self.net.id2t_name):
            actions.append(
                ActionInfo(
                    action_id=i,
                    action_name=self.get_action_name(i),
                    enabled=(i in enabled),
                    description="" if i in enabled else "当前条件不满足",
                )
            )
        actions.append(ActionInfo(action_id=self.net.T, action_name="WAIT", enabled=True, description=""))
        return actions

    def get_reward_breakdown(self) -> Dict[str, float]:
        return dict(self._last_reward_detail)

    @property
    def action_space_size(self) -> int:
        return int(self.net.T)

    def get_current_state(self) -> StateInfo:
        return self._collect_state_info()

    def render_gantt(self, output_path: str) -> bool:
        try:
            self.net.render_gantt(output_path)
            return True
        except Exception:
            return False

    def export_action_sequence(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def _collect_state_info(self) -> StateInfo:
        chambers: List[ChamberState] = []
        transports: List[ChamberState] = []
        release_schedule: Dict[str, list] = {}
        trigger = int(getattr(self.net, "single_cleaning_trigger_wafers", 2))
        targets = set(getattr(self.net, "single_cleaning_targets", {"PM3", "PM4"}))
        for idx, place in enumerate(self.net.marks):
            wafers = [
                WaferState(
                    token_id=int(tok.token_id),
                    place_name=place.name,
                    place_idx=idx,
                    place_type=int(place.type),
                    stay_time=float(tok.stay_time),
                    proc_time=float(place.processing_time),
                    time_to_scrap=self._time_to_scrap(place, float(tok.stay_time)),
                    route_id=int(getattr(tok, "route_type", 0)),
                    step=int(getattr(tok, "step", 0)),
                )
                for tok in place.tokens
                if int(getattr(tok, "token_id", -1)) >= 0
            ]
            release_schedule[place.name] = list(getattr(place, "release_schedule", []))
            if place.name == "d_TM1":
                transports.append(
                    ChamberState(
                        name=place.name,
                        place_idx=idx,
                        capacity=int(place.capacity),
                        wafers=wafers,
                        proc_time=float(place.processing_time),
                        status="active" if wafers else "idle",
                        chamber_type="transport",
                    )
                )
                continue

            chamber_type = "disabled" if place.name in self.disabled_chambers else "processing"
            status = self._calc_status(place.name, wafers, chamber_type)
            is_cleaning = bool(getattr(place, "is_cleaning", False))
            cleaning_remaining = float(getattr(place, "cleaning_remaining", 0.0))
            if is_cleaning:
                status = "cleaning"
            processed = int(getattr(place, "processed_wafer_count", 0))
            countdown = max(0, trigger - processed) if place.name in targets else -1
            chambers.append(
                ChamberState(
                    name=place.name,
                    place_idx=idx,
                    capacity=int(place.capacity),
                    wafers=wafers,
                    proc_time=float(place.processing_time),
                    status=status,
                    chamber_type=chamber_type,
                    cleaning_remaining=cleaning_remaining,
                    inbound_blocked=is_cleaning,
                    cleaning_wafer_countdown=countdown,
                )
            )

        robot_wafers_tm2: List[WaferState] = []
        robot_wafers_tm3: List[WaferState] = []
        if transports:
            robot_wafers = transports[0].wafers
            machine_by_token: Dict[int, int] = {}
            d_tm_place = next((p for p in self.net.marks if p.name == "d_TM1"), None)
            if d_tm_place is not None:
                for tok in d_tm_place.tokens:
                    tid = int(getattr(tok, "token_id", -1))
                    if tid >= 0:
                        machine_by_token[tid] = int(getattr(tok, "machine", 1))
            for wafer in robot_wafers:
                machine_id = machine_by_token.get(int(wafer.token_id), 1)
                if machine_id == 2:
                    robot_wafers_tm3.append(wafer)
                else:
                    robot_wafers_tm2.append(wafer)
        stats = self.net.calc_wafer_statistics() if hasattr(self.net, "calc_wafer_statistics") else {}
        return StateInfo(
            time=float(getattr(self.net, "time", 0)),
            chambers=chambers,
            transport_buffers=transports,
            start_buffers=[c for c in chambers if c.name == "LP"],
            end_buffers=[c for c in chambers if c.name == "LP_done"],
            robot_states={
                "TM2": RobotState(name="TM2", busy=bool(robot_wafers_tm2), wafers=robot_wafers_tm2),
                "TM3": RobotState(name="TM3", busy=bool(robot_wafers_tm3), wafers=robot_wafers_tm3),
            },
            enabled_actions=self.get_enabled_actions(),
            done_count=int(getattr(self.net, "done_count", 0)),
            total_wafers=int(getattr(self.net, "n_wafer", 0)),
            tpt_wph=(float(getattr(self.net, "done_count", 0)) / max(1e-9, float(getattr(self.net, "time", 0)))) * 3600
            if float(getattr(self.net, "time", 0)) > 0
            else 0.0,
            stats={
                "release_schedule": release_schedule,
                "system_avg": stats.get("system_avg", 0.0),
                "system_max": stats.get("system_max", 0),
                "system_diff": stats.get("system_diff", 0.0),
                "completed_count": stats.get("completed_count", 0),
                "in_progress_count": stats.get("in_progress_count", 0),
                "chambers": stats.get("chambers", {}),
                "transports": stats.get("transports", {}),
                "transports_detail": stats.get("transports_detail", {}),
                "resident_violation_count": stats.get("resident_violation_count", 0),
                "qtime_violation_count": stats.get("qtime_violation_count", 0),
                "chamber_processed_counts": stats.get("chamber_processed_counts", {}),
            },
        )

    def _time_to_scrap(self, place, stay_time: float) -> float:
        if place.type == 1:
            return float(place.processing_time + getattr(self.net, "P_Residual_time", 0) - stay_time)
        if place.type == 2:
            return float(getattr(self.net, "D_Residual_time", 0) - stay_time)
        return -1.0

    def _calc_status(self, name: str, wafers: List[WaferState], chamber_type: str) -> str:
        if chamber_type == "disabled":
            return "disabled"
        if not wafers:
            return "idle"
        min_t = min((w.time_to_scrap for w in wafers if w.time_to_scrap >= 0), default=9999)
        if min_t <= 0:
            return "danger"
        if min_t <= 5:
            return "warning"
        return "active"
