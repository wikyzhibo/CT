"""
Petri 网算法适配器
将现有 Env_PN/Petri 环境适配为统一可视化接口
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

from solutions.PPO.enviroment import Env_PN

from .algorithm_interface import (
    AlgorithmAdapter,
    ActionInfo,
    WaferState,
    ChamberState,
    RobotState,
    StateInfo,
)


class PetriAdapter(AlgorithmAdapter):
    """Petri 网算法适配器"""

    def __init__(self, env: Env_PN) -> None:
        self.env = env
        self.net = env.net
        self._last_reward_detail: Dict[str, float] = {}
        self._last_action_history: List[Dict[str, Any]] = []

        # 确保可视化模式下启用统计
        if hasattr(self.net, 'enable_statistics'):
            self.net.enable_statistics = True

        # 腔室映射配置（迁移自 viz.py）
        self.chamber_config = {
            "LLA": {"source": ["LP1", "LP2"], "active": True, "proc_time": 0, "robot": "TM2"},
            "LLB": {"source": "LP_done", "active": True, "proc_time": 0, "robot": "TM2"},
            "PM7": {"source": "s1", "machine": 0, "active": True, "proc_time": 70, "robot": "TM2"},
            "PM8": {"source": "s1", "machine": 1, "active": True, "proc_time": 70, "robot": "TM2"},
            "PM9": {"source": "s5", "machine": 0, "active": True, "proc_time": 200, "robot": "TM2"},
            "PM10": {"source": "s5", "machine": 1, "active": True, "proc_time": 200, "robot": "TM2"},
            "LLC": {"source": "s2", "active": True, "proc_time": 0, "robot": "TM3"},
            "LLD": {"source": "s4", "active": True, "proc_time": 70, "robot": "TM3"},
            "PM1": {"source": "s3", "machine": 0, "active": True, "proc_time": 600, "robot": "TM3"},
            "PM2": {"source": "s3", "machine": 1, "active": True, "proc_time": 600, "robot": "TM3"},
            "PM3": {"source": "s3", "machine": 2, "active": True, "proc_time": 600, "robot": "TM3"},
            "PM4": {"source": "s3", "machine": 3, "active": True, "proc_time": 600, "robot": "TM3"},
            "PM5": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM6": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
        }

        self.display_chambers = list(self.chamber_config.keys())
        # 双运输库所模式：d_TM2 和 d_TM3
        self.transports = ["d_TM2", "d_TM3"]
        self.transports_tm2 = []
        self.transports_tm3 = []

    def reset(self) -> StateInfo:
        self.env.reset()
        return self._collect_state_info()

    def step(self, action: int) -> Tuple[StateInfo, float, bool, Dict]:
        result = self.net.step(
            t=action if action < self.net.T else None,
            wait=(action == self.net.T),
            with_reward=True,
            detailed_reward=True,
        )

        # pn.py: (done, reward_result, scrap)
        done = bool(result[0])
        reward_result = result[1]
        scrap = bool(result[2]) if len(result) > 2 else False

        if isinstance(reward_result, dict):
            self._last_reward_detail = {k: float(v) for k, v in reward_result.items() if isinstance(v, (int, float))}
            reward = float(reward_result.get("total", 0.0))
        else:
            self._last_reward_detail = {}
            reward = float(reward_result)

        if not done:
            done = getattr(self.net, "done_count", 0) >= getattr(self.net, "n_wafer", 0)

        state_info = self._collect_state_info()
        info = {"done": done, "reward": reward, "scrap": scrap, "detail": self._last_reward_detail}
        self._last_action_history.append({
            "step": len(self._last_action_history) + 1,
            "action": self.get_action_name(action),
            "reward": reward,
            "detail": self._last_reward_detail,
        })
        return state_info, reward, done, info

    def get_action_name(self, action: int) -> str:
        if action == self.net.T:
            return "WAIT"
        if 0 <= action < len(self.net.id2t_name):
            return self._format_transition_name(self.net.id2t_name[action])
        return f"UNKNOWN_{action}"

    def get_enabled_actions(self) -> List[ActionInfo]:
        enabled_t = set(self.net.get_enable_t())
        actions: List[ActionInfo] = []
        for t in range(self.net.T):
            enabled = t in enabled_t
            desc = "" if enabled else "当前条件不满足"
            actions.append(ActionInfo(
                action_id=t,
                action_name=self._format_transition_name(self.net.id2t_name[t]),
                enabled=enabled,
                description=desc,
            ))
        actions.append(ActionInfo(
            action_id=self.net.T,
            action_name="WAIT",
            enabled=True,
            description="",
        ))
        return actions

    def get_enabled_actions_by_robot(self) -> Tuple[List[ActionInfo], List[ActionInfo]]:
        """
        返回 TM2 和 TM3 各自的可用动作列表。
        
        Returns:
            (tm2_actions, tm3_actions): 两个 ActionInfo 列表
        """
        from solutions.Continuous_model.pn import TM2_TRANSITIONS, TM3_TRANSITIONS
        
        tm2_enabled, tm3_enabled = self.net.get_enable_t_by_robot()
        tm2_enabled_set = set(tm2_enabled)
        tm3_enabled_set = set(tm3_enabled)
        
        tm2_actions: List[ActionInfo] = []
        tm3_actions: List[ActionInfo] = []
        
        for t in range(self.net.T):
            t_name = self.net.id2t_name[t]
            if t_name in TM2_TRANSITIONS:
                enabled = t in tm2_enabled_set
                tm2_actions.append(ActionInfo(
                    action_id=t,
                    action_name=self._format_transition_name(t_name),
                    enabled=enabled,
                    description="" if enabled else "当前条件不满足",
                ))
            elif t_name in TM3_TRANSITIONS:
                enabled = t in tm3_enabled_set
                tm3_actions.append(ActionInfo(
                    action_id=t,
                    action_name=self._format_transition_name(t_name),
                    enabled=enabled,
                    description="" if enabled else "当前条件不满足",
                ))
        
        # 添加 WAIT 动作
        tm2_actions.append(ActionInfo(
            action_id=-1,  # 特殊 ID 表示 WAIT
            action_name="WAIT",
            enabled=True,
            description="",
        ))
        tm3_actions.append(ActionInfo(
            action_id=-1,
            action_name="WAIT",
            enabled=True,
            description="",
        ))
        
        return tm2_actions, tm3_actions

    def step_concurrent(self, a1: int, a2: int) -> Tuple[StateInfo, float, bool, Dict]:
        """
        执行并发动作。
        
        Args:
            a1: TM2 的变迁索引，-1 表示 WAIT
            a2: TM3 的变迁索引，-1 表示 WAIT
            
        Returns:
            (state_info, reward, done, info)
        """
        # 转换为 pn.py 的参数格式
        tm2_action = None if a1 == -1 else a1
        tm3_action = None if a2 == -1 else a2
        
        result = self.net.step_concurrent(
            a1=tm2_action,
            a2=tm3_action,
            with_reward=True,
            detailed_reward=True,
        )
        
        done = bool(result[0])
        reward_result = result[1]
        scrap = bool(result[2]) if len(result) > 2 else False
        
        if isinstance(reward_result, dict):
            self._last_reward_detail = {k: float(v) for k, v in reward_result.items() if isinstance(v, (int, float))}
            reward = float(reward_result.get("total", 0.0))
        else:
            self._last_reward_detail = {}
            reward = float(reward_result)
        
        if not done:
            done = getattr(self.net, "done_count", 0) >= getattr(self.net, "n_wafer", 0)
        
        state_info = self._collect_state_info()
        info = {"done": done, "reward": reward, "scrap": scrap, "detail": self._last_reward_detail}
        
        # 记录动作历史
        a1_name = "WAIT" if a1 == -1 else self._format_transition_name(self.net.id2t_name[a1])
        a2_name = "WAIT" if a2 == -1 else self._format_transition_name(self.net.id2t_name[a2])
        self._last_action_history.append({
            "step": len(self._last_action_history) + 1,
            "action": f"TM2:{a1_name}, TM3:{a2_name}",
            "reward": reward,
            "detail": self._last_reward_detail,
        })
        
        return state_info, reward, done, info

    def get_reward_breakdown(self) -> Dict[str, float]:
        return dict(self._last_reward_detail or {})

    def _format_transition_name(self, t_name: str) -> str:
        """将技术性变迁名称转换为简短友好的显示格式"""
        place_to_display = {
            "LP1": "LP1",
            "LP2": "LP2",
            "s1": "PM7/PM8",
            "s2": "LLC",
            "s3": "PM1-4",
            "s4": "LLD",
            "s5": "PM9/PM10",
            "LP_done": "LLB",
        }

        if t_name.startswith("u_"):
            parts = t_name.split("_")
            if len(parts) >= 3:
                from_place = parts[1]
                to_place = parts[2]
                from_display = place_to_display.get(from_place, from_place)
                to_display = place_to_display.get(to_place, to_place)
                return f"{from_display}→{to_display}"
        elif t_name.startswith("t_"):
            place = t_name[2:]
            display = place_to_display.get(place, place)
            return display

        return t_name

    @property
    def action_space_size(self) -> int:
        return int(self.net.T)

    def get_current_state(self) -> StateInfo:
        return self._collect_state_info()

    def render_gantt(self, output_path: str) -> bool:
        try:
            self.net.render_gantt(out_path=output_path)
            return True
        except Exception:
            return False

    def export_action_sequence(self) -> List[Dict[str, Any]]:
        return list(self._last_action_history)

    def _collect_state_info(self) -> StateInfo:
        chamber_states: Dict[str, ChamberState] = {}
        transport_states: Dict[str, ChamberState] = {}

        for name, config in self.chamber_config.items():
            chamber_states[name] = ChamberState(
                name=name,
                place_idx=-1,
                capacity=0,
                wafers=[],
                proc_time=float(config.get("proc_time", 0)),
                status="idle",
                chamber_type="processing",
            )

        for t_name in self.transports:
            transport_states[t_name] = ChamberState(
                name=t_name,
                place_idx=-1,
                capacity=0,
                wafers=[],
                proc_time=0.0,
                status="idle",
                chamber_type="transport",
            )

        release_schedule: Dict[str, List[Tuple[int, int]]] = {}

        for p_idx, place in enumerate(self.net.marks):
            p_name = place.name
            if p_name.startswith("r_"):
                continue

            release_schedule[p_name] = list(getattr(place, "release_schedule", []))

            for tok in place.tokens:
                if getattr(tok, "token_id", -1) < 0:
                    continue

                wafer_state = self._build_wafer_state(p_idx, place, tok)

                if p_name in ("LP1", "LP2"):
                    chamber_states["LLA"].wafers.append(wafer_state)
                elif p_name == "LP_done":
                    chamber_states["LLB"].wafers.append(wafer_state)
                elif p_name == "s1":
                    machine = getattr(tok, "machine", 0)
                    chamber_states["PM7" if machine == 0 else "PM8"].wafers.append(wafer_state)
                elif p_name == "s2":
                    chamber_states["LLC"].wafers.append(wafer_state)
                elif p_name == "s3":
                    machine = getattr(tok, "machine", 0)
                    targets = ["PM1", "PM2", "PM3", "PM4"]
                    chamber_states[targets[machine % 4]].wafers.append(wafer_state)
                elif p_name == "s4":
                    chamber_states["LLD"].wafers.append(wafer_state)
                elif p_name == "s5":
                    machine = getattr(tok, "machine", 0)
                    chamber_states["PM9" if machine == 0 else "PM10"].wafers.append(wafer_state)
                elif p_name.startswith("d_") and p_name in transport_states:
                    transport_states[p_name].wafers.append(wafer_state)

        # 更新 place_idx/capacity
        for name, chamber in chamber_states.items():
            source = self.chamber_config[name].get("source")
            place_name = None
            if isinstance(source, list):
                place_name = source[0]
            elif isinstance(source, str):
                place_name = source
            if place_name is not None:
                place = self._get_place_by_name(place_name)
                if place is not None:
                    chamber.place_idx = self.net.id2p_name.index(place_name)
                    chamber.capacity = getattr(place, "capacity", 0)

        for t_name, chamber in transport_states.items():
            place = self._get_place_by_name(t_name)
            if place is not None:
                chamber.place_idx = self.net.id2p_name.index(t_name)
                chamber.capacity = getattr(place, "capacity", 0)

        # 计算状态
        for chamber in list(chamber_states.values()) + list(transport_states.values()):
            chamber.status = self._calc_chamber_status(chamber)

        robot_states = self._collect_robot_states(transport_states)

        # 获取晶圆统计数据（用于左侧面板指标显示）
        wafer_stats = {}
        if hasattr(self.net, "calc_wafer_statistics"):
            wafer_stats = self.net.calc_wafer_statistics()
        
        return StateInfo(
            time=float(getattr(self.net, "time", 0)),
            chambers=list(chamber_states.values()),
            transport_buffers=list(transport_states.values()),
            start_buffers=[chamber_states["LLA"]],
            end_buffers=[chamber_states["LLB"]],
            robot_states=robot_states,
            enabled_actions=self.get_enabled_actions(),
            done_count=int(getattr(self.net, "done_count", 0)),
            total_wafers=int(getattr(self.net, "n_wafer", 0)),
            tpt_wph=(float(getattr(self.net, "done_count", 0)) / float(getattr(self.net, "time", 1e-9))) * 3600 if float(getattr(self.net, "time", 0)) > 0 else 0.0,
            stats={
                "release_schedule": release_schedule,
                # 晶圆统计数据（与 viz.py 中 calc_wafer_statistics 一致）
                "system_avg": wafer_stats.get("system_avg", 0.0),
                "system_max": wafer_stats.get("system_max", 0),
                "system_diff": wafer_stats.get("system_diff", 0.0),
                "completed_count": wafer_stats.get("completed_count", 0),
                "in_progress_count": wafer_stats.get("in_progress_count", 0),
                "chambers": self._remap_chamber_stats(wafer_stats.get("chambers", {})),
                "transports": wafer_stats.get("transports", {}),
                "transports_detail": wafer_stats.get("transports_detail", {}),
                "resident_violation_count": wafer_stats.get("resident_violation_count", 0),
                "qtime_violation_count": wafer_stats.get("qtime_violation_count", 0),
            },
        )

    def _build_wafer_state(self, p_idx, place, tok) -> WaferState:
        stay_time = float(getattr(tok, "stay_time", 0))
        proc_time = float(getattr(place, "processing_time", 0))
        place_type = int(getattr(place, "type", 0))
        time_to_scrap = self._calc_time_to_scrap(place_type, stay_time, proc_time)
        route_id = int(getattr(tok, "route_type", 0))

        return WaferState(
            token_id=int(getattr(tok, "token_id", -1)),
            place_name=place.name,
            place_idx=int(p_idx),
            place_type=place_type,
            stay_time=stay_time,
            proc_time=proc_time,
            time_to_scrap=time_to_scrap,
            route_id=route_id,
            step=int(getattr(tok, "step", 0)),
        )

    def _calc_time_to_scrap(self, place_type: int, stay_time: float, proc_time: float) -> float:
        if place_type == 1:
            return (proc_time + float(getattr(self.net, "P_Residual_time", 0))) - stay_time
        if place_type == 2:
            return float(getattr(self.net, "D_Residual_time", 0)) - stay_time
        return -1.0

    def _calc_chamber_status(self, chamber: ChamberState) -> str:
        if not chamber.wafers:
            return "idle"
        if chamber.chamber_type == "transport":
            return "active"
        min_time_to_scrap = min((w.time_to_scrap for w in chamber.wafers if w.time_to_scrap >= 0), default=9999)
        if min_time_to_scrap <= 0:
            return "danger"
        if min_time_to_scrap <= 5:
            return "warning"
        return "active"

    def _collect_robot_states(self, transport_states: Dict[str, ChamberState]) -> Dict[str, RobotState]:
        tm2_wafers = []
        tm3_wafers = []
        
        # 直接根据运输库所名称分发
        if "d_TM2" in transport_states:
            tm2_wafers.extend(transport_states["d_TM2"].wafers)
        if "d_TM3" in transport_states:
            tm3_wafers.extend(transport_states["d_TM3"].wafers)
                    
        return {
            "TM2": RobotState(name="TM2", busy=bool(tm2_wafers), wafers=tm2_wafers),
            "TM3": RobotState(name="TM3", busy=bool(tm3_wafers), wafers=tm3_wafers),
        }

    def _get_place_by_name(self, name: str):
        for place in self.net.marks:
            if place.name == name:
                return place
        return None

    def _remap_chamber_stats(self, raw_chambers: Dict[str, Any]) -> Dict[str, Any]:
        """将内部腔室名称映射为 UI 显示名称"""
        mapping = {
            "s1": "PM7/8",
            "s3": "PM1/2/3/4",
            "s5": "PM9/10",
        }
        remaped = {}
        for k, v in raw_chambers.items():
            new_key = mapping.get(k, k)
            remaped[new_key] = v
        return remaped
