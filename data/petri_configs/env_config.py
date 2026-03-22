"""
Petri 网环境配置。

支持通过 routes（字典或列表）、end_place_name、no_residence_place_names 等配置
多条路线与无驻留腔室，便于更新路线时仅改配置、不改业务逻辑。
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


def _default_reward_config() -> Dict[str, int]:
    return {
        "proc_reward": 1,
        "safe_reward": 1,
        "penalty": 1,
        "warn_penalty": 1,
        "transport_penalty": 1,
        "congestion_penalty": 1,
        "time_cost": 1,
        "in_system_time_penalty": 1,
        "release_violation_penalty": 1,
    }


def _default_single_process_time_map() -> Dict[str, int]:
    return {
        "PM1": 100,
        "PM3": 300,
        "PM4": 300,
        "PM6": 300,
    }


def _default_wait_durations() -> List[int]:
    return [5, 10, 20, 50, 100]


@dataclass
class PetriEnvConfig:
    """
    Petri 网环境配置。
    
    打印功能：
        - print(config) 或 str(config)：简略模式，显示关键配置
        - print(repr(config)) 或 config.format(detailed=True)：详细模式，显示所有配置项
        
    示例：
        >>> config = PetriEnvConfig(n_wafer=12)
        >>> print(config)  # 简略模式
        >>> print(config.format(detailed=True))  # 详细模式
    """
    MAX_TIME: int = 2000
    n_wafer: int = 12

    # 奖励计算系数
    done_event_reward: int = 10
    finish_event_reward: int = 800
    scrap_event_penalty: int = -500
    warn_coef_penalty: int = 2
    idle_event_penalty: int = 1000

    transport_overtime_coef_penalty: float = 1.0  # 运输超时惩罚系数 (原 Q1_p)
    processing_coef_reward: float = 3.0  # 加工奖励系数 (原 r)
    in_system_time_penalty_coef: float = 0.0  # 系统内停留惩罚系数（温和，避免掩盖加工奖励）
    time_coef_penalty: float = 1.0  # 时间成本系数
    release_event_penalty: float = 0.1  # 释放违规惩罚系数

    # 常量
    D_Residual_time: int = 20
    P_Residual_time: int = 15
    T_transport: int = 5
    T_load: int = 5


    stop_on_scrap: bool = True
    max_wafers_in_system: int = 7
    


    reward_config: Optional[Dict[str, int]] = None

    single_robot_capacity: int = 1 # 单设备机械手容量
    device_mode: str = "single" # 单设备模式：single=原单设备路径，cascade=级联路径模板
    # 单设备清洗配置（训练简化版）
    cleaning_enabled: bool = True
    cleaning_targets: List[str] = field(default_factory=lambda: ["PM3", "PM4"])
    cleaning_trigger_wafers: int = 5
    cleaning_duration: int = 150
    # 单设备工序时间配置（秒）
    process_time_map: Dict[str, int] = field(default_factory=_default_single_process_time_map)
    # 单设备路径代号（整数切换预置路径）
    # single 模式:
    #   0: PM1 -> [PM3/PM4] -> LP_done
    #   1: PM1 -> [PM3/PM4] -> PM6 -> LP_done
    # cascade 模式:
    #   1: PM7/PM8 -> LLC -> PM1/PM2/PM3/PM4 -> LLD -> PM9/PM10
    #   2: PM7/PM8 -> LLC -> PM1/PM2 -> LLD -> PM9/PM10
    #   3: PM7/PM8 -> LLC -> PM1/PM2 -> LLD -> LP_done
    #   4: [PM7 -> PM8 -> LLC -> LLD] * 5 -> LP_done
    #   5: PM7/PM8 -> PM9/PM10 -> LP_done
    route_code: int = 1
    # route_code=4 的手动发片节拍间隔（秒）；<=0 表示不启用节拍门控
    route4_takt_interval: int = 0
    # LLC→d_TM3（u_LLC）发射节拍间隔（秒）；<=0 表示不启用。口径与 u_LP 节拍一致（首次发射不门控）
    llc_tm3_takt_interval: int = 70
    # 单设备配置驱动路径（可选）：提供后优先于 route_code 模板构网
    single_route_config: Optional[Dict[str, Any]] = None
    # 单设备配置驱动路径文件（可选）：在 load(json) 时可自动加载到 single_route_config
    single_route_config_path: Optional[str] = None
    # single_route_config 下的 routes 选择名（为空则按 legacy alias/首条路线）
    single_route_name: Optional[str] = None
    # 单设备 WAIT 动作档位（秒）
    wait_durations: List[int] = field(default_factory=_default_wait_durations)

    # 腔室集成配置（可选）。若提供则覆盖/生成 process_time_map 与清洗映射
    # 每腔室: process_time, cleaning_duration, cleaning_trigger_wafers
    chambers: Optional[Dict[str, Dict[str, Any]]] = None
    # 由 chambers 或扁平字段归一化得到的 per-chamber 映射（__post_init__ 中填充）
    cleaning_trigger_wafers_map: Optional[Dict[str, int]] = None
    cleaning_duration_map: Optional[Dict[str, int]] = None

    # 路线与晶圆分配（可选；无则用默认双路线）
    n_wafer_route1: Optional[int] = None
    n_wafer_route2: Optional[int] = None

    # 路线：Dict[str, List[str]] 或 List[List[str]]，无则用默认两条路线
    routes: Optional[Union[Dict[str, List[str]], List[List[str]]]] = None
    # 起点库所名（用于入口变迁与晶圆数限制），可从 routes 自动推导
    start_place_names: Optional[List[str]] = None
    # 终点库所名，默认 "LP_done"
    end_place_name: str = "LP_done"
    # 无驻留约束的库所名集合（type=5），如 {"s2", "s4"}
    no_residence_place_names: Optional[Set[str]] = None
    # 库所显示名（用于统计/甘特图），可选
    place_display_names: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.reward_config is None:
            self.reward_config = _default_reward_config()
        self._normalize_chamber_config()

    def _normalize_chamber_config(self) -> None:
        """从 chambers 或扁平字段归一化出 cleaning_trigger_wafers_map、cleaning_duration_map 等。"""
        if self.chambers is not None:
            pt_map = dict(self.process_time_map)
            for name, spec in self.chambers.items():
                if not isinstance(spec, dict):
                    continue
                pt = spec.get("process_time")
                if pt is not None:
                    pt_map[name] = int(pt)
            self.process_time_map = pt_map
            self.cleaning_trigger_wafers_map = {
                name: max(0, int(spec.get("cleaning_trigger_wafers", 0)))
                for name, spec in (self.chambers or {}).items()
                if isinstance(spec, dict)
            }
            self.cleaning_duration_map = {
                name: max(0, int(spec.get("cleaning_duration", 0)))
                for name, spec in (self.chambers or {}).items()
                if isinstance(spec, dict)
            }
            # chambers 模式下，清洗目标与 per-chamber 触发映射保持一致
            self.cleaning_targets = sorted(
                [name for name, trig in self.cleaning_trigger_wafers_map.items() if int(trig) > 0]
            )
        else:
            self.cleaning_trigger_wafers_map = {
                c: self.cleaning_trigger_wafers for c in self.cleaning_targets
            }
            self.cleaning_duration_map = {
                c: self.cleaning_duration for c in self.cleaning_targets
            }

    def format(self, detailed: bool = False) -> str:
        """
        格式化配置为字符串。
        
        Args:
            detailed: 是否使用详细模式（显示所有配置项）
            
        Returns:
            格式化的配置字符串
        """
        if detailed:
            return self._format_detailed()
        else:
            return self._format_brief()

    def _format_brief(self) -> str:
        """简略模式：显示关键配置项"""
        lines = ["PetriEnvConfig (简略模式):"]
        lines.append(f"  晶圆数: {self.n_wafer}")
        lines.append(f"  停止条件: stop_on_scrap={self.stop_on_scrap}")
        
        # 路线配置
        if self.routes is not None:
            if isinstance(self.routes, dict):
                route_info = f"{len(self.routes)} 条路线: {', '.join(self.routes.keys())}"
            else:
                route_info = f"{len(self.routes)} 条路线"
            lines.append(f"  路线配置: {route_info}")
        else:
            lines.append("  路线配置: 使用默认双路线")
        
        if self.n_wafer_route1 is not None or self.n_wafer_route2 is not None:
            lines.append(f"  路线分配: route1={self.n_wafer_route1}, route2={self.n_wafer_route2}")
        lines.append(f"  单设备机械手容量: {self.single_robot_capacity}")
        lines.append(f"  单设备模式: {self.device_mode}")
        lines.append(f"  单设备路径代号: {self.route_code}")
        if self.single_route_config is not None:
            sel = self.single_route_name or "<auto>"
            lines.append(f"  单设备配置驱动路径: enabled (route={sel})")
        elif self.single_route_config_path:
            sel = self.single_route_name or "<auto>"
            lines.append(f"  单设备配置驱动路径文件: {self.single_route_config_path} (route={sel})")
        if str(self.device_mode).lower() == "cascade" and int(self.route_code) == 4:
            lines.append(f"  route4手动节拍: {self.route4_takt_interval}s")
        if int(self.llc_tm3_takt_interval) > 0:
            lines.append(f"  LLC→TM3 节拍: {self.llc_tm3_takt_interval}s")
        
        if self.end_place_name != "LP_done":
            lines.append(f"  终点库所: {self.end_place_name}")
        
        if self.no_residence_place_names:
            lines.append(f"  无驻留腔室: {sorted(self.no_residence_place_names)}")

        
        # 奖励配置（仅显示非默认值）
        reward_non_default = {
            k: v for k, v in self.reward_config.items()
            if v != _default_reward_config().get(k, 1)
        }
        if reward_non_default:
            lines.append(f"  奖励开关（非默认）: {reward_non_default}")
        
        return "\n".join(lines)

    def _format_detailed(self) -> str:
        """详细模式：显示所有配置项"""
        lines = ["PetriEnvConfig (详细模式):"]
        lines.append("=" * 60)
        
        # 基础配置
        lines.append("\n【基础配置】")
        lines.append(f"  n_wafer: {self.n_wafer}")
        lines.append(f"  stop_on_scrap: {self.stop_on_scrap}")
        lines.append(f" max_wafers: {self.max_wafers_in_system}")
        
        # 奖励参数
        lines.append("\n【奖励参数】")
        lines.append(f"  done_event_reward: {self.done_event_reward}")
        lines.append(f"  finish_event_reward: {self.finish_event_reward}")
        lines.append(f"  scrap_event_penalty: {self.scrap_event_penalty}")
        lines.append(f"  time_coef: {self.time_coef_penalty}")
        
        # 其他参数
        lines.append("\n【其他参数】")
        lines.append(f"  release_penalty_coef: {self.release_event_penalty}")
        
        # 路线配置
        lines.append("\n【路线配置】")
        if self.routes is not None:
            if isinstance(self.routes, dict):
                lines.append(f"  routes (字典, {len(self.routes)} 条):")
                for name, route in self.routes.items():
                    lines.append(f"    {name}: {route}")
            else:
                lines.append(f"  routes (列表, {len(self.routes)} 条):")
                for i, route in enumerate(self.routes, 1):
                    lines.append(f"    路线{i}: {route}")
        else:
            lines.append("  routes: None (使用默认双路线)")
        
        if self.start_place_names is not None:
            lines.append(f"  start_place_names: {self.start_place_names}")
        else:
            lines.append("  start_place_names: None (从 routes 自动推导)")
        
        lines.append(f"  end_place_name: {self.end_place_name}")
        
        if self.n_wafer_route1 is not None:
            lines.append(f"  n_wafer_route1: {self.n_wafer_route1}")
        if self.n_wafer_route2 is not None:
            lines.append(f"  n_wafer_route2: {self.n_wafer_route2}")
        
        if self.no_residence_place_names:
            lines.append(f"  no_residence_place_names: {sorted(self.no_residence_place_names)}")
        else:
            lines.append("  no_residence_place_names: None")
        
        if self.place_display_names:
            lines.append(f"  place_display_names: {self.place_display_names}")
        else:
            lines.append("  place_display_names: None")

        lines.append(f"  single_robot_capacity: {self.single_robot_capacity}")
        lines.append(f"  route_code: {self.route_code}")
        lines.append(f"  single_route_config: {'set' if self.single_route_config is not None else 'None'}")
        lines.append(f"  single_route_config_path: {self.single_route_config_path}")
        lines.append(f"  single_route_name: {self.single_route_name}")
        lines.append(f"  route4_takt_interval: {self.route4_takt_interval}")
        lines.append(f"  llc_tm3_takt_interval: {self.llc_tm3_takt_interval}")
        
        # 奖励配置
        lines.append("\n【奖励开关配置】")
        for key, value in sorted(self.reward_config.items()):
            lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        """简略模式（默认）"""
        return self.format(detailed=False)

    def __repr__(self) -> str:
        """详细模式"""
        return self.format(detailed=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PetriEnvConfig":
        """从 JSON 文件加载配置。"""
        path = Path(path)
        #print("loading petri config from", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 旧参数名迁移到新参数名（新参数优先）
        if "time_coef" not in data and "c_time" in data:
            data["time_coef"] = data["c_time"]
        if "release_penalty_coef" not in data and "c_release_violation" in data:
            data["release_penalty_coef"] = data["c_release_violation"]
        if "done_event_reward" not in data and "R_done" in data:
            data["done_event_reward"] = data["R_done"]
        if "finish_event_reward" not in data and "R_finish" in data:
            data["finish_event_reward"] = data["R_finish"]
        if "scrap_event_penalty" not in data and "R_scrap" in data:
            data["scrap_event_penalty"] = data["R_scrap"]
        if "idle_event_penalty" not in data and "idle_penalty" in data:
            data["idle_event_penalty"] = data["idle_penalty"]
        # 列表/集合在 JSON 中为列表，需转换
        if "no_residence_place_names" in data and data["no_residence_place_names"] is not None:
            data["no_residence_place_names"] = set(data["no_residence_place_names"])
        if "routes" in data and data["routes"] is not None and isinstance(data["routes"], dict):
            # 保持为 dict
            pass
        # 配置驱动路径文件：若仅给了路径，自动加载成 single_route_config
        route_cfg_path = data.get("single_route_config_path")
        if route_cfg_path and not data.get("single_route_config"):
            route_path = Path(route_cfg_path)
            if not route_path.is_absolute():
                route_path = path.parent / route_path
            with open(route_path, "r", encoding="utf-8") as rf:
                data["single_route_config"] = json.load(rf)

        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})
