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
    }


def _default_single_proc_time_rand_scale_map() -> Dict[str, Dict[str, float]]:
    return {
        "PM1": {"min": 1.0, "max": 1.0},
        "PM3": {"min": 1.0, "max": 1.0},
        "PM4": {"min": 1.0, "max": 1.0},
    }


def _default_single_wait_durations() -> List[int]:
    return [5, 10, 20, 50, 100]


@dataclass
class PetriEnvConfig:
    """
    Petri 网环境配置。
    
    打印功能：
        - print(config) 或 str(config)：简略模式，显示关键配置
        - print(repr(config)) 或 config.format(detailed=True)：详细模式，显示所有配置项
        
    示例：
        >>> config = PetriEnvConfig(n_wafer=12, training_phase=2)
        >>> print(config)  # 简略模式
        >>> print(config.format(detailed=True))  # 详细模式
    """

    n_wafer: int = 12
    R_done: int = 10
    R_finish: int = 800
    R_scrap: int = -500
    T_warn: int = 30
    a_warn: float = 0.1
    T_safe: int = 60
    b_safe: float = 0.05

    c_congest: float = 1.0
    D_Residual_time: int = 10
    P_Residual_time: int = 15

    T_transport: int = 5
    T_load: int = 5

    idle_penalty: int = 10
    stop_on_scrap: bool = True
    training_phase: int = 2
    max_wafers_in_system: int = 7
    
    # 奖励计算系数
    transport_overtime_coef: float = 1.0    # 运输超时惩罚系数 (原 Q1_p)
    chamber_overtime_coef: float = 0.2      # 加工腔室超时惩罚系数 (原 Q2_p)
    processing_reward_coef: float = 3.0     # 加工奖励系数 (原 r)
    in_system_time_penalty_coef: float = 0.0  # 系统内停留惩罚系数（温和，避免掩盖加工奖励）
    time_coef : float = 1.0  # 时间成本系数
    release_penalty_coef: float = 0.1 # 释放违规惩罚系数

    reward_config: Optional[Dict[str, int]] = None

    # 性能优化
    optimize_reward_calc: bool = False
    optimize_enable_check: bool = False
    optimize_state_update: bool = False
    cache_indices: bool = False
    optimize_data_structures: bool = False
    turbo_mode: bool = False
    single_robot_capacity: int = 1 # 单设备机械手容量（默认为1，训练简化版可设置为2或更高以加快训练）
    # 单设备清洗配置（训练简化版）
    single_cleaning_enabled: bool = True
    single_cleaning_targets: List[str] = field(default_factory=lambda: ["PM3", "PM4"])
    single_cleaning_trigger_wafers: int = 2
    single_cleaning_duration: int = 150
    # 单设备工序时间配置（秒）
    single_process_time_map: Dict[str, int] = field(default_factory=_default_single_process_time_map)
    # 单设备工序时间随机扰动（按 episode 固定）
    single_proc_time_rand_enabled: bool = False
    # 单设备工序时间随机扰动区间（按腔室独立配置）
    single_proc_time_rand_scale_map: Dict[str, Dict[str, float]] = field(
        default_factory=_default_single_proc_time_rand_scale_map
    )
    # 单设备 WAIT 动作档位（秒）
    single_wait_durations: List[int] = field(default_factory=_default_single_wait_durations)
    # 兼容旧版统一随机区间（当 single_proc_time_rand_scale_map 缺失时回退使用）
    single_proc_time_rand_min_scale: float = 1.0
    single_proc_time_rand_max_scale: float = 1.0

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
        lines.append(f"  训练阶段: {self.training_phase}")
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
        
        if self.end_place_name != "LP_done":
            lines.append(f"  终点库所: {self.end_place_name}")
        
        if self.no_residence_place_names:
            lines.append(f"  无驻留腔室: {sorted(self.no_residence_place_names)}")
        
        # 性能优化
        optimizations = []
        if self.optimize_reward_calc:
            optimizations.append("reward_calc")
        if self.optimize_enable_check:
            optimizations.append("enable_check")
        if self.optimize_state_update:
            optimizations.append("state_update")
        if self.cache_indices:
            optimizations.append("cache_indices")
        if self.optimize_data_structures:
            optimizations.append("data_structures")
        if self.turbo_mode:
            optimizations.append("turbo_mode")
        
        if optimizations:
            lines.append(f"  性能优化: {', '.join(optimizations)}")
        
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
        lines.append(f"  training_phase: {self.training_phase}")
        lines.append(f"  stop_on_scrap: {self.stop_on_scrap}")
        lines.append(f" max_wafers: {self.max_wafers_in_system}")
        
        # 奖励参数
        lines.append("\n【奖励参数】")
        lines.append(f"  R_done: {self.R_done}")
        lines.append(f"  R_finish: {self.R_finish}")
        lines.append(f"  R_scrap: {self.R_scrap}")
        lines.append(f"  c_time: {self.c_time}")
        
        # 其他参数
        lines.append("\n【其他参数】")
        lines.append(f"  MAX_WAIT_STEP: {self.MAX_WAIT_STEP}")
        lines.append(f"  c_congest: {self.c_congest}")
        lines.append(f"  c_release_violation: {self.c_release_violation}")
        
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
        
        # 性能优化
        lines.append("\n【性能优化】")
        lines.append(f"  optimize_reward_calc: {self.optimize_reward_calc}")
        lines.append(f"  optimize_enable_check: {self.optimize_enable_check}")
        lines.append(f"  optimize_state_update: {self.optimize_state_update}")
        lines.append(f"  cache_indices: {self.cache_indices}")
        lines.append(f"  optimize_data_structures: {self.optimize_data_structures}")
        lines.append(f"  turbo_mode: {self.turbo_mode}")
        lines.append(f"  single_robot_capacity: {self.single_robot_capacity}")
        
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
        print("loading petri config from", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 列表/集合在 JSON 中为列表，需转换
        if "no_residence_place_names" in data and data["no_residence_place_names"] is not None:
            data["no_residence_place_names"] = set(data["no_residence_place_names"])
        if "routes" in data and data["routes"] is not None and isinstance(data["routes"], dict):
            # 保持为 dict
            pass
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})
