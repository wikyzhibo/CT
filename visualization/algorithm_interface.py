"""
统一算法接口 - 为不同调度算法提供统一的抽象层
支持 PPO/PDR/DFS/遗传算法等多种算法接入可视化系统
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
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


class AlgorithmAdapter(ABC):
    """算法适配器基类 - 统一接口"""
    
    @abstractmethod
    def reset(self) -> StateInfo:
        """重置环境,返回初始状态"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[StateInfo, float, bool, Dict]:
        """
        执行动作,返回(新状态, 奖励, 是否结束, 额外信息)
        
        Args:
            action: 动作 ID
            
        Returns:
            state_info: 新状态信息
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        pass
    
    @abstractmethod
    def get_action_name(self, action: int) -> str:
        """获取动作名称(用于UI显示)"""
        pass
    
    @abstractmethod
    def get_enabled_actions(self) -> List[ActionInfo]:
        """获取当前可用动作列表"""
        pass
    
    @abstractmethod
    def get_reward_breakdown(self) -> Dict[str, float]:
        """
        获取奖励分解(用于详细显示)
        
        Returns:
            字典,包含各个奖励/惩罚项,例如:
            {
                "proc_reward": 10.0,
                "penalty": -5.0,
                "transport_penalty": -2.0,
                ...
            }
        """
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """动作空间大小(不含 WAIT 动作)"""
        pass
    
    @abstractmethod
    def get_current_state(self) -> StateInfo:
        """获取当前状态信息(不执行动作)"""
        pass
    
    def render_gantt(self, output_path: str) -> bool:
        """
        生成甘特图(可选功能)
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否成功生成
        """
        return False
    
    def export_action_sequence(self) -> List[Dict[str, Any]]:
        """
        导出动作序列(可选功能)
        
        Returns:
            动作序列列表,每个元素包含 step, action, reward 等信息
        """
        return []
