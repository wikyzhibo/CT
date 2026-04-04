"""TM1 自动选择：shared+ratio 下多 LP 同时可取时优先与比例轮次一致的 route_type。"""

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


def test_tm1_pick_prefers_lp_matching_ratio_when_cycle_wants_type2():
    net = ClusterTool(_cfg_4_8_n5(), concurrent=True)
    net.reset()
    net._shared_ratio_cycle_idx = 1
    req = net._resolve_required_release_type_for_entry_heads()
    assert req == 2
    start = int(net.T)
    net.get_action_mask(
        wait_action_start=start,
        n_actions=start + len(net.wait_durations),
    )
    act = net._cached_auto_tm1_action
    assert act is not None
    assert net.id2t_name[int(act)] == "u_LP2_TM1"


def test_tm1_pick_starts_with_lp1_when_cycle_wants_type1():
    net = ClusterTool(_cfg_4_8_n5(), concurrent=True)
    net.reset()
    net._shared_ratio_cycle_idx = 0
    req = net._resolve_required_release_type_for_entry_heads()
    assert req == 1
    start = int(net.T)
    net.get_action_mask(
        wait_action_start=start,
        n_actions=start + len(net.wait_durations),
    )
    act = net._cached_auto_tm1_action
    assert act is not None
    assert net.id2t_name[int(act)] == "u_LP1_TM1"
