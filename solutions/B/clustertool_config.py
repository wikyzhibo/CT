from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Union


@dataclass
class ClusterToolCfg:

    MAX_TIME: int = 2000
    n_wafer1: int = 12
    n_wafer2: int = 0
    ttime: int = 5
    search_depth: int = 5
    candidate_k: int = 8
    takt_cycle: list[int] | None = None

    # 奖励
    done_event_reward: int = 10
    finish_event_reward: int = 800
    scrap_event_penalty: int = -500

    transport_overtime_coef_penalty: float = 1.0  # 运输超时惩罚系数
    processing_coef_reward: float = 3.0  # 加工奖励系数 (原 r)
    time_coef_penalty: float = 1.0  # 时间成本系数

    # 常量
    D_Residual_time: int = 20
    P_Residual_time: int = 15
    T_transport: int = 5
    T_load: int = 5

    max_wafers1_in_system: int = 7
    max_wafers2_in_system: int = 7


    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "ClusterToolCfg":
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("clustertool_default.json")
        )
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        valid_keys = {item.name for item in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**filtered)
        if cfg.scrap_event_penalty >= 0:
            raise ValueError(
                "scrap_event_penalty must be negative; "
                f"got {cfg.scrap_event_penalty}"
            )
        return cfg
