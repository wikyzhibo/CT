"""
可视化状态与动作的数据类型定义。
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ActionInfo:
    """动作信息"""
    action_id: int
    action_name: str
    enabled: bool
    description: str = ""


@dataclass
class WaferState:
    """晶圆状态"""
    token_id: int
    place_name: str
    place_idx: int
    place_type: int  # 1=加工腔, 2=运输位, 3=起点, 4=终点
    stay_time: float
    proc_time: float
    time_to_scrap: float  # -1 表示无超时风险
    route_id: int = 0  # 路线 ID (用于颜色区分)
    step: int = 0      # 当前步骤索引


@dataclass
class ChamberState:
    """腔室状态"""
    name: str
    place_idx: int
    capacity: int
    wafers: List[WaferState] = field(default_factory=list)
    proc_time: float = 0.0
    status: str = "idle"  # "idle", "active", "warning", "danger"
    chamber_type: str = "processing"  # "processing", "transport", "start", "end"
    cleaning_remaining: float = 0.0  # 清洁剩余秒数（仅可视化层）
    inbound_blocked: bool = False    # 清洁期间禁入提示（仅可视化层）
    cleaning_wafer_countdown: int = -1  # 距清洗晶圆倒数，-1 表示不显示（非清洗目标腔室）


@dataclass
class RobotState:
    """机械手状态"""
    name: str
    busy: bool
    wafers: List[WaferState] = field(default_factory=list)


@dataclass
class StateInfo:
    """完整状态信息"""
    time: float
    chambers: List[ChamberState] = field(default_factory=list)
    transport_buffers: List[ChamberState] = field(default_factory=list)
    start_buffers: List[ChamberState] = field(default_factory=list)  # LP1/LP2
    end_buffers: List[ChamberState] = field(default_factory=list)    # LP_done
    robot_states: Dict[str, RobotState] = field(default_factory=dict)
    enabled_actions: List[ActionInfo] = field(default_factory=list)
    done_count: int = 0
    total_wafers: int = 0
    tpt_wph: float = 0.0
    
    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)
