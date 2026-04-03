import json
from pathlib import Path

import torch

from solutions.A.model_builder import build_net
from solutions.A.rl_env import Env_PN_Concurrent


ROOT = Path(__file__).resolve().parents[1]
ROUTE_CONFIG_PATH = ROOT / "config" / "cluster_tool" / "route_config.json"


def _load_route_config() -> dict:
    with ROUTE_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _capacity_map(info: dict) -> dict[str, int]:
    return {
        str(name): int(info["capacity"][idx])
        for idx, name in enumerate(info["id2p_name"])
    }


def test_route_2_1_builds_tm1_outer_chain_and_real_capacities():
    info = build_net(n_wafer1=1, route_config=_load_route_config(), route_name="2-1")

    assert {"TM1", "AL", "LLA", "LLB", "CL"}.issubset(set(info["id2p_name"]))
    assert {
        "u_LP1_TM1",
        "u_AL_TM1",
        "u_LLB_TM1",
        "u_CL_TM1",
        "t_TM1_AL",
        "t_TM1_LLA",
        "t_TM1_CL",
        "t_TM1_LP_done",
        "t_TM2_LLB",
    }.issubset(set(info["id2t_name"]))
    assert info["route_meta"]["route_stages"] == [
        ["LLA"],
        ["PM7", "PM8"],
        ["LLC"],
        ["PM1", "PM2", "PM3", "PM4"],
        ["LLD"],
        ["PM9", "PM10"],
        ["LLB"],
    ]
    assert info["route_meta"]["release_control_places"] == ("LLA",)

    capacity = _capacity_map(info)
    assert capacity["AL"] == 1
    assert capacity["LLA"] == 2
    assert capacity["LLB"] == 2
    assert capacity["CL"] == 6
    assert capacity["TM1"] == 1


def test_route_1_1_uses_tm1_outer_chain_like_2_1():
    info = build_net(n_wafer1=1, route_config=_load_route_config(), route_name="1-1")

    assert "t_TM1_LP_done" in info["id2t_name"]
    assert "t_TM2_LP_done" not in info["id2t_name"]
    assert {
        "u_LP1_TM1",
        "u_AL_TM1",
        "u_LLB_TM1",
        "u_CL_TM1",
        "t_TM1_AL",
        "t_TM1_LLA",
        "t_TM1_CL",
        "t_TM1_LP_done",
    }.issubset(set(info["id2t_name"]))
    assert info["route_meta"]["release_control_places"] == ("LLA",)


def test_concurrent_env_exposes_triple_masks_and_tm1_prefix_step():
    env = Env_PN_Concurrent(single_route_name="2-1", n_wafer=1)
    td = env.reset()

    assert tuple(td["action_mask_tm1"].shape) == (env.n_actions_tm1,)
    assert tuple(td["action_mask_tm2"].shape) == (env.n_actions_tm2,)
    assert tuple(td["action_mask_tm3"].shape) == (env.n_actions_tm3,)
    assert int(td["action_mask_tm1"][:-1].sum().item()) >= 1
    assert int(td["action_mask_tm2"][:-1].sum().item()) == 0
    assert int(td["action_mask_tm3"][:-1].sum().item()) == 0

    first_tm1 = next(i for i, enabled in enumerate(td["action_mask_tm1"][:-1].tolist()) if enabled)
    step_td = td.clone()
    step_td["action_tm1"] = torch.tensor([first_tm1], dtype=torch.int64)
    step_td["action_tm2"] = torch.tensor([env.tm2_wait_action], dtype=torch.int64)
    step_td["action_tm3"] = torch.tensor([env.tm3_wait_action], dtype=torch.int64)
    next_td = env.step(step_td)

    assert tuple(next_td["action_mask_tm1"].shape) == (env.n_actions_tm1,)


def test_concurrent_env_1_1_resets_with_tm1_action_space_aligned_to_2_1():
    env = Env_PN_Concurrent(single_route_name="1-1", n_wafer=1)
    ref = Env_PN_Concurrent(single_route_name="2-1", n_wafer=1)
    td = env.reset()

    assert tuple(td["action_mask_tm1"].shape) == (env.n_actions_tm1,)
    assert tuple(td["action_mask_tm2"].shape) == (env.n_actions_tm2,)
    assert tuple(td["action_mask_tm3"].shape) == (env.n_actions_tm3,)
    assert env.n_actions_tm1 == ref.n_actions_tm1
