"""
单设备 Petri 可视化适配器。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from solutions.A.rl_env import Env_PN_Single

REWARD_DESC_VIZ: Dict[str, str] = {
    "total": "本步总奖励",
    "time_cost": "时间惩罚",
    "proc_reward": "加工奖励",
    "warn_penalty": "驻留警告惩罚",
    "penalty": "运输超时惩罚",
    "wafer_done_bonus": "单片完工奖励",
    "finish_bonus": "全部完工奖励",
    "scrap_penalty": "报废惩罚",
    "release_violation_penalty": "释放违规惩罚",
    "idle_timeout_penalty": "闲置超时惩罚",
}

from .algorithm_interface import (
    ActionInfo,
    AlgorithmAdapter,
    ChamberState,
    RobotState,
    StateInfo,
    WaferState,
)

def _is_transport_place_name(name: str) -> bool:
    """级联固定拓扑运输库所为 TM1/TM2/TM3；历史构网兼容 d_TM*。"""
    return name.startswith("d_TM") or name in {"TM1", "TM2", "TM3"}


def _normalize_transport_name(name: str) -> str:
    raw = str(name)
    if raw.startswith("d_TM"):
        return raw[2:]
    return raw


class PetriSingleAdapter(AlgorithmAdapter):
    def __init__(self, env: Env_PN_Single, step_verbose: bool = True) -> None:
        self.env = env
        self.net = env.net
        self.step_verbose = step_verbose
        self.device_mode = str(getattr(self.net, "device_mode", "cascade")).lower()
        self._last_reward_detail: Dict[str, float] = {}
        self._history: List[Dict[str, Any]] = []
        self._step_count = 0
        if self.device_mode == "cascade":
            # Cascade 模式下 PM5/PM6 需要在 UI 中可视化展示（status 由 net 状态决定）。
            # 旧逻辑会把 PM5/PM6 强制标记为 disabled，导致 LED 永远不点亮。
            self.disabled_chambers = set()
        else:
            self.disabled_chambers = {"PM2", "PM5"}
            if "PM6" not in getattr(self.net, "chambers", ()):
                self.disabled_chambers.add("PM6")

    def reset(self) -> StateInfo:
        self.env.reset()
        self._history.clear()
        self._step_count = 0
        return self._collect_state_info()

    def step(self, action: int | Tuple[int, int]):
        if isinstance(action, tuple):
            action = int(action[0]) if action[0] != -1 else int(self.action_space_size)
        self._step_count += 1

        if self.step_verbose:
            action_name = self.get_action_name(action)
            t = getattr(self.net, "time", 0)
            print(f"\n--- Step {self._step_count} (t={t}) 执行: {action_name} ---")

        wait_duration = self.env.parse_wait_action(int(action))
        if wait_duration is not None:
            done, reward_result, scrap, _action_mask, _obs = self.net.step(
                wait_duration=int(wait_duration),
            )
        else:
            _, transition_idx = self.env._decode_action(int(action))
            done, reward_result, scrap, _action_mask, _obs = self.net.step(
                a1=int(transition_idx),
            )

        reward = float(reward_result)
        self._last_reward_detail = {}
        if self.step_verbose and self._last_reward_detail:
            nonzero = {k: v for k, v in self._last_reward_detail.items() if isinstance(v, (int, float)) and v != 0}
            if nonzero:
                print("  详细奖励:")
                for k, v in sorted(nonzero.items()):
                    lbl = REWARD_DESC_VIZ.get(k, k)
                    if isinstance(v, float):
                        print(f"    {lbl}: {v:.4f}")
                    else:
                        print(f"    {lbl}: {v}")
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
        wait_duration = self.env.parse_wait_action(int(action))
        if wait_duration is not None:
            return f"WAIT_{int(wait_duration)}s"
        if 0 <= action < len(self.net.id2t_name):
            # 需求：按钮/历史直接显示原始变迁名（u_src / t_dst）
            return str(self.net.id2t_name[action])
        return f"UNKNOWN_{action}"

    def get_enabled_actions(self) -> List[ActionInfo]:
        mask = self.env._mask()
        actions: List[ActionInfo] = []
        for i in range(int(self.env.n_actions)):
            name = self.get_action_name(i)
            actions.append(
                ActionInfo(
                    action_id=i,
                    action_name=name,
                    enabled=bool(mask[i]),
                    description="" if bool(mask[i]) else "当前条件不满足",
                )
            )
        return actions

    def get_reward_breakdown(self) -> Dict[str, float]:
        return dict(self._last_reward_detail)

    @property
    def action_space_size(self) -> int:
        return int(self.env.wait_action_indices[0]) if self.env.wait_action_indices else int(self.net.T)

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
        trigger_map = getattr(self.net, "_cleaning_trigger_map", None)
        cascade_lp_slots: List[Tuple[int, Any, List[WaferState]]] = []
        for idx, place in enumerate(self.net.marks):
            place_name = str(place.name)
            wafers = [
                WaferState(
                    token_id=int(tok.token_id),
                    place_name=place_name,
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
            release_schedule[place_name] = list(getattr(place, "release_schedule", []))
            if _is_transport_place_name(place_name):
                transports.append(
                    ChamberState(
                        name=_normalize_transport_name(place_name),
                        place_idx=idx,
                        capacity=int(place.capacity),
                        wafers=wafers,
                        proc_time=float(place.processing_time),
                        status="active" if wafers else "idle",
                        chamber_type="transport",
                    )
                )
                continue

            if self.device_mode == "cascade" and place_name in ("LP1", "LP2"):
                cascade_lp_slots.append((idx, place, wafers))
                continue

            chambers.append(
                self._build_chamber_state(
                    display_name=place_name,
                    source_name=place_name,
                    place_idx=idx,
                    capacity=int(place.capacity),
                    wafers=wafers,
                    proc_time=float(place.processing_time),
                    is_cleaning=bool(getattr(place, "is_cleaning", False)),
                    cleaning_remaining=float(getattr(place, "cleaning_remaining", 0.0)),
                    processed=int(getattr(place, "processed_wafer_count", 0)),
                    trigger_map=trigger_map,
                )
            )
        if self.device_mode == "cascade" and cascade_lp_slots:
            merged_w: List[WaferState] = []
            for _, _, ws in cascade_lp_slots:
                merged_w.extend(ws)
            cap_sum = sum(int(p.capacity) for _, p, _ in cascade_lp_slots)
            idx0, place0, _ = cascade_lp_slots[0]
            release_schedule["LP"] = []
            for n in ("LP1", "LP2"):
                release_schedule["LP"].extend(release_schedule.pop(n, []))
            chambers.append(
                self._build_chamber_state(
                    display_name="LP",
                    source_name="LP1",
                    place_idx=idx0,
                    capacity=cap_sum,
                    wafers=merged_w,
                    proc_time=float(place0.processing_time),
                    is_cleaning=bool(getattr(place0, "is_cleaning", False)),
                    cleaning_remaining=float(getattr(place0, "cleaning_remaining", 0.0)),
                    processed=int(getattr(place0, "processed_wafer_count", 0)),
                    trigger_map=trigger_map,
                )
            )
        if self.device_mode == "cascade":
            existing_names = {c.name for c in chambers}
            for name in ("PM5", "PM6"):
                if name in existing_names:
                    continue
                chambers.append(
                    ChamberState(
                        name=name,
                        place_idx=-1,
                        capacity=1,
                        wafers=[],
                        proc_time=0.0,
                        # 缺失时仅作为 idle 占位展示；不再使用 disabled 样式。
                        status="idle",
                        chamber_type="processing",
                        cleaning_wafer_countdown=-1,
                    )
                )

        robot_wafers_tm1: List[WaferState] = []
        robot_wafers_tm2: List[WaferState] = []
        robot_wafers_tm3: List[WaferState] = []
        if transports:
            if self.device_mode == "cascade":
                d_tm1 = next((c for c in transports if c.name == "TM1"), None)
                d_tm2 = next((c for c in transports if c.name == "TM2"), None)
                d_tm3 = next((c for c in transports if c.name == "TM3"), None)
                if d_tm1 is not None:
                    robot_wafers_tm1.extend(d_tm1.wafers)
                if d_tm2 is not None:
                    robot_wafers_tm2.extend(d_tm2.wafers)
                if d_tm3 is not None:
                    robot_wafers_tm3.extend(d_tm3.wafers)
            else:
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

        done_count = int(getattr(self.net, "done_count", 0))
        total_wafers = int(getattr(self.net, "n_wafer", 0))
        in_progress = max(0, total_wafers - done_count)
        return StateInfo(
            time=float(getattr(self.net, "time", 0)),
            chambers=chambers,
            transport_buffers=transports,
            start_buffers=[c for c in chambers if c.name == "LP"],
            end_buffers=[c for c in chambers if c.name == "LP_done"],
            robot_states={
                "TM1": RobotState(name="TM1", busy=bool(robot_wafers_tm1), wafers=robot_wafers_tm1),
                "TM2": RobotState(name="TM2", busy=bool(robot_wafers_tm2), wafers=robot_wafers_tm2),
                "TM3": RobotState(name="TM3", busy=bool(robot_wafers_tm3), wafers=robot_wafers_tm3),
            },
            enabled_actions=self.get_enabled_actions(),
            done_count=done_count,
            total_wafers=total_wafers,
            tpt_wph=(float(done_count) / max(1e-9, float(getattr(self.net, "time", 0)))) * 3600
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
                "chamber_processed_counts": {},
            },
        )

    def _build_chamber_state(
        self,
        display_name: str,
        source_name: str,
        place_idx: int,
        capacity: int,
        wafers: List[WaferState],
        proc_time: float,
        is_cleaning: bool,
        cleaning_remaining: float,
        processed: int,
        trigger_map,
    ) -> ChamberState:
        chamber_type = "disabled" if display_name in self.disabled_chambers else "processing"
        status = self._calc_status(source_name, wafers, chamber_type)
        if is_cleaning:
            status = "cleaning"
        if trigger_map is not None:
            chamber_trigger = int(trigger_map.get(source_name, 0))
            countdown = max(0, chamber_trigger - processed) if chamber_trigger > 0 else -1
        else:
            countdown = -1
        return ChamberState(
            name=display_name,
            place_idx=place_idx,
            capacity=capacity,
            wafers=wafers,
            proc_time=proc_time,
            status=status,
            chamber_type=chamber_type,
            cleaning_remaining=cleaning_remaining,
            inbound_blocked=is_cleaning,
            cleaning_wafer_countdown=countdown,
        )

    def _time_to_scrap(self, place, stay_time: float) -> float:
        if place.type == 1:
            return float(place.processing_time + getattr(self.net, "P_Residual_time", 0) - stay_time)
        if place.type == 5 and place.name in {"LLC", "LLD"}:
            return float(place.processing_time + getattr(self.net, "P_Residual_time", 0) * 3 - stay_time)
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
