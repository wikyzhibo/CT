"""LP 单库所：构网时按 cycle_type 排列 tokens，TM1 自动取料选单一 u_LP_TM1。"""

import json
from pathlib import Path

import yaml

from config.cluster_tool.env_config import PetriEnvConfig
from solutions.A.petri_net import ClusterTool

ROOT = Path(__file__).resolve().parents[1]


def _cfg_4_8_n5() -> PetriEnvConfig:
    data = yaml.safe_load((ROOT / "config" / "cluster_tool" / "cascade.yaml").read_text(encoding="utf-8"))
    data["n_wafer"] = 5
    data["single_route_name"] = "4-8"
    data["single_route_config"] = json.loads(
        (ROOT / "config" / "cluster_tool" / "route_config.json").read_text(encoding="utf-8")
    )
    return PetriEnvConfig.model_validate(data)


def test_lp_token_order_follows_cycle_type():
    """LP tokens 应按 cycle_type=[1,2,2,2,2] 排列：[1,2,2,2,2]。"""
    net = ClusterTool(_cfg_4_8_n5(), concurrent=True)
    net.reset()
    lp = net._place_by_name.get("LP")
    assert lp is not None, "LP place must exist"
    types = [int(getattr(tok, "route_type", 1)) for tok in lp.tokens]
    assert types == [1, 2, 2, 2, 2], f"expected [1,2,2,2,2], got {types}"


def test_tm1_auto_action_picks_lp_when_al_empty():
    """AL 空且 LLA 未满时，TM1 自动动作应为 u_LP_TM1。"""
    net = ClusterTool(_cfg_4_8_n5(), concurrent=True)
    net.reset()
    start = int(net.T)
    net.get_action_mask(wait_action_start=start, n_actions=start + 1)
    act = net._cached_auto_tm1_action
    assert act is not None
    assert net.id2t_name[int(act)] == "u_LP_TM1"
