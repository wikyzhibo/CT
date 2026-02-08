"""
Timed Petri网的配置模块。这个模块集成所有之前在tdpn.py中硬编码的数据，
让其更容易修改和保持。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
from pathlib import Path


@dataclass
class ModuleSpec:
    """Specification for a Petri net module."""
    tokens: int  # Initial number of tokens
    capacity: int  # Maximum capacity


@dataclass
class PetriConfig:
    """Timed Petri Net 的配置类"""
    
    # Parallel machine groups
    parallel_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "PM1_4": ["PM1", "PM2", "PM3", "PM4"],
        "PM7_8": ["PM7", "PM8"],
        "PM9_10": ["PM9", "PM10"],
    })
    
    # Module specifications
    modules: Dict[str, ModuleSpec] = field(default_factory=lambda: {
        "LP1": ModuleSpec(tokens=50, capacity=100),
        "LP2": ModuleSpec(tokens=25, capacity=100),
        "AL": ModuleSpec(tokens=0, capacity=1),
        "LLA_S2": ModuleSpec(tokens=0, capacity=1),
        "PM7": ModuleSpec(tokens=0, capacity=1),
        "PM8": ModuleSpec(tokens=0, capacity=1),
        "LLC": ModuleSpec(tokens=0, capacity=1),
        "PM1": ModuleSpec(tokens=0, capacity=1),
        "PM2": ModuleSpec(tokens=0, capacity=1),
        "PM3": ModuleSpec(tokens=0, capacity=1),
        "PM4": ModuleSpec(tokens=0, capacity=1),
        "LLD": ModuleSpec(tokens=0, capacity=1),
        "PM9": ModuleSpec(tokens=0, capacity=1),
        "PM10": ModuleSpec(tokens=0, capacity=1),
        "LLB_S1": ModuleSpec(tokens=0, capacity=1),
        "LP_done": ModuleSpec(tokens=0, capacity=100),
    })
    
    # Routes definition
    routes: List[List] = field(default_factory=lambda: [
        ["LP1", "AL", "LLA_S2", ["PM7", "PM8"], "LLC", ["PM1", "PM2", "PM3", "PM4"], 
         "LLD", ["PM9", "PM10"], "LLB_S1", "LP_done"],
        ["LP2", "AL", "LLA_S2", ["PM7", "PM8"], ["PM9", "PM10"], "LLB_S1", "LP_done"],
    ])
    
    # Stage capacity (number of parallel machines per stage)
    stage_capacity: Dict[int, int] = field(default_factory=lambda: {
        1: 2,   # PM7/PM8
        2: 1,   # LLC
        3: 4,   # PM1-PM4
        4: 1,   # LLD
        5: 2,   # PM9/PM10
        6: 1,   # ARM2
        7: 1,   # ARM3
    })
    
    # Processing time per stage (seconds)
    processing_time: Dict[int, int] = field(default_factory=lambda: {
        1: 70,   # PM7/PM8
        2: 0,    # LLC (no processing)
        3: 600,  # PM1-PM4
        4: 70,   # LLD
        5: 200,  # PM9/PM10
        6: 5,    # ARM2 operations
        7: 5,    # ARM3 operations
    })
    
    # Observation history length
    history_length: int = 50
    
    # Reward stage weights
    reward_weights: List[int] = field(default_factory=lambda: [
        0,     # Stage 0: LP1/LP2
        10,    # Stage 1: AL
        30,    # Stage 2: LLA_S2
        100,   # Stage 3: PM7/PM8
        770,   # Stage 4: LLC
        970,   # Stage 5: PM1-4/LLD
        1000,  # Stage 6: PM9/PM10/LLB
    ])

    # @classmthod: 类方法，PetriConfig.default(); 用于实例还没创建
    # 第一个参数写cls，表示调用者的类别
    # 实例方法 a = PetriConfig() , a.default()
    # -> 'PetriConfig' 执行这个方法时，类还没有定义好，使用字符串
    @classmethod
    def default(cls) -> 'PetriConfig':
        """创建一个默认配置"""
        return cls()
    
    @classmethod
    def from_json(cls, filepath: str) -> 'PetriConfig':
        """从 data/ directory 目录下加载JSON配置文件"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert module specs from dict to ModuleSpec objects
        if 'modules' in data:
            data['modules'] = {
                name: ModuleSpec(**spec) if isinstance(spec, dict) else spec
                for name, spec in data['modules'].items()
            }
        
        return cls(**data)
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to a JSON file in data/ directory."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert ModuleSpec objects to dicts for JSON serialization
        data = {
            'parallel_groups': self.parallel_groups,
            'modules': {
                name: {'tokens': spec.tokens, 'capacity': spec.capacity}
                for name, spec in self.modules.items()
            },
            'routes': self.routes,
            'stage_capacity': self.stage_capacity,
            'processing_time': self.processing_time,
            'history_length': self.history_length,
            'reward_weights': self.reward_weights,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
