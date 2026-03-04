"""
Petri 网环境配置。

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
        "time_cost": 1,
        "release_violation_penalty": 1,
    }


@dataclass
class PetriEnvConfig:
    """
    Petri 网环境配置。
    
    打印功能：
        - print(config) 或 str(config)：简略模式，显示关键配置
        - print(repr(config)) 或 config.format(detailed=True)：详细模式，显示所有配置项

    """
    # =========晶圆数量===========
    n_wafer_route1: Optional[int] = None #路线C晶圆数
    n_wafer_route2: Optional[int] = None #路线D晶圆数
    max_wafers_in_system: int = 7 #系统最大容纳晶圆数
    MAX_TIME: int = 7000  # 最大环境步数/时间

    # =========奖励与惩罚===========
    c_time: int = 1 #时间惩罚系数
    R_done: int = 10 #完成一片晶圆奖励
    R_finish: int = 800 #完工奖励
    R_scrap: int = -500 #违反驻留时间约束惩罚
    a_warn: float = 0.1 #预警系数
    b_safe: float = 0.05 #安全奖励系数
    c_release_violation: float = 0.1 #违反施放时间约束惩罚系数
    idle_penalty: int = 10 #空转惩罚系数
    transport_overtime_coef: float = 3.0  # 运输超时惩罚系数 (原 Q1_p)
    chamber_overtime_coef: float = 0.2  # 加工腔室超时惩罚系数 (原 Q2_p)
    processing_reward_coef: float = 2.0  # 加工奖励系数 (原 r)

    # =========常量===========
    D_Residual_time: int = 10 #最大驻留时间
    P_Residual_time: int = 15 #最大驻留时间
    T_warn: int = 30
    T_safe: int = 60
    T_transport: int = 5 #运输时间
    T_load: int = 5 #装载和卸载时间
    idle_timeout: int = 300 #系统最大停滞时间，防止策略陷入持续空转
    Wait_time = 5 #等待的时长

    # =========向后兼容=================
    enable_release_penalty_detection: bool = False #施放时间惩罚开关

    #==========训练====================
    stop_on_scrap: bool = True #违反驻留时间约束是否截断
    training_phase: int = 2 #训练阶段

    
    reward_config: Optional[Dict[str, int]] = None


    def __post_init__(self) -> None:
        if self.reward_config is None:
            self.reward_config = _default_reward_config()

    def format(self, detailed: bool = True) -> str:
        """
        格式化配置为字符串。
        
        Args:
            detailed: (已弃用，保留兼容性) 是否使用详细模式
            
        Returns:
            格式化的配置字符串
        """
        return self._format_compact()

    def _format_compact(self) -> str:
        """紧凑详细模式：显示所有配置项"""
        # 奖励配置（显示非默认值）
        reward_diff = []
        default_rewards = _default_reward_config()
        for k, v in sorted(self.reward_config.items()):
            if v != default_rewards.get(k, 1):
                reward_diff.append(f"{k}:{v}")
        reward_str = "|".join(reward_diff) if reward_diff else "Default"
        
        # 路线信息
        routes = f"R1:{self.n_wafer_route1}|R2:{self.n_wafer_route2}"
        
        return (
            f"PetriEnvConfig: \n"
            f"[Base] T_phase:{self.training_phase}|stop:{self.stop_on_scrap}|max_wafer:{self.max_wafers_in_system}|MAX_TIME:{self.MAX_TIME} \n"
            f"[Reward] R_done:{self.R_done}|finish:{self.R_finish}|scrap:{self.R_scrap}|c_time:{self.c_time} \n"
            f"[Param] c_release:{self.c_release_violation}|idle_pen:{self.idle_penalty} \n"
            f"[Routes] {routes} \n"
            f"[RConf] {reward_str}\n"
        )

    def __str__(self) -> str:
        """紧凑模式"""
        return self.format()

    def __repr__(self) -> str:
        """紧凑模式"""
        return self.format()

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
