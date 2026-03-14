import torch
import numpy as np

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.env_single import Env_PN_Single
from solutions.Continuous_model.pn_single import ClusterTool


def _extract_next_td(td):
    return td["next"] if "next" in td.keys() else td


def test_wait_action_catalog_and_mask_contains_all_wait_bins():
    env = Env_PN_Single()
    td = env.reset()
    mask = td["action_mask"].cpu().numpy()
    net_mask = env.net.get_action_mask(
        wait_action_start=env.wait_action_start,
        n_actions=env.n_actions,
    )

    assert env.wait_durations == sorted(set(env.wait_durations))
    assert 5 in env.wait_durations
    assert env.n_actions == env.net.T + len(env.wait_durations)
    assert np.array_equal(mask, net_mask)

    wait_actions = []
    for idx in range(env.n_actions):
        wait_duration = env.parse_wait_action(idx)
        if wait_duration is not None:
            wait_actions.append(wait_duration)
            assert bool(mask[idx]) is True
    assert wait_actions == env.wait_durations


def test_wait_uses_min_of_requested_and_next_event_delta():
    env = Env_PN_Single(detailed_reward=True)
    td = env.reset()

    for place in env.net.marks:
        place.tokens.clear()
    pm3 = env.net._get_place("PM3")
    tok = env.net.ori_marks[env.net._get_place_index("LP")].tokens[0].clone()
    tok.stay_time = max(0, int(pm3.processing_time) - 10)
    pm3.tokens.append(tok)

    max_wait = int(max(env.wait_durations))
    wait_100_action = next(idx for idx in range(env.n_actions) if env.parse_wait_action(idx) == max_wait)
    step_td = td.clone()
    step_td["action"] = torch.tensor(wait_100_action, dtype=torch.int64)
    td_next = env.step(step_td)
    td_after = _extract_next_td(td_next)

    assert int(td_after["time"].item()) == 10


def test_wait_is_capped_by_transport_complete_event():
    config = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        wait_durations=[5, 20, 100],
    )
    net = ClusterTool(config=config)
    net.reset()

    for place in net.marks:
        place.tokens.clear()
        if place.name.startswith("PM"):
            place.is_cleaning = False
            place.cleaning_remaining = 0

    d_tm = net._get_place("d_TM1")
    tok = net.ori_marks[net._get_place_index("LP")].tokens[0].clone()
    tok.stay_time = int(net.T_transport) - 3
    d_tm.tokens.append(tok)
    net.m[net._get_place_index("d_TM1")] = 1

    assert net.get_next_event_delta() == 3
    _, _, _, _, _ = net.step(detailed_reward=True, wait_duration=20)
    assert net.time == 3


def test_wait_fallback_when_no_future_event():
    config = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        single_wait_durations=[5, 10, 20, 50, 100],
    )
    net = ClusterTool(config=config)
    net.reset()

    for place in net.marks:
        if place.name != "LP":
            place.tokens.clear()
        if place.name.startswith("PM"):
            place.is_cleaning = False
            place.cleaning_remaining = 0

    assert net.get_next_event_delta() is None
    _, reward_result, _, _, _ = net.step(detailed_reward=True, wait_duration=20)

    assert net.time == 20


def test_wait_actions_gt5_disabled_when_ready_chamber_wafer_exists():
    env = Env_PN_Single()
    env.reset()

    lp = env.net._get_place("LP")
    pm1 = env.net._get_place("PM1")
    assert len(lp.tokens) > 0
    ready_tok = lp.tokens.pop()
    ready_tok.stay_time = int(pm1.processing_time)
    pm1.tokens.append(ready_tok)

    mask = env._mask()
    enabled_actions = set(env.get_enable_t())
    enabled_actions_net = set(
        env.net.get_enable_actions(wait_action_start=env.wait_action_start)
    )
    wait_5_idx = next(idx for idx in range(env.n_actions) if env.parse_wait_action(idx) == 5)

    assert bool(mask[wait_5_idx]) is True
    assert wait_5_idx in enabled_actions
    assert wait_5_idx in enabled_actions_net
    for idx in range(env.n_actions):
        wait_duration = env.parse_wait_action(idx)
        if wait_duration is None or int(wait_duration) <= 5:
            continue
        assert bool(mask[idx]) is False
        assert idx not in enabled_actions
        assert idx not in enabled_actions_net


def test_wait_actions_gt5_enabled_when_no_ready_chamber_wafer():
    env = Env_PN_Single()
    env.reset()

    lp = env.net._get_place("LP")
    pm1 = env.net._get_place("PM1")
    assert len(lp.tokens) > 0
    not_ready_tok = lp.tokens.pop()
    not_ready_tok.stay_time = max(0, int(pm1.processing_time) - 1)
    pm1.tokens.append(not_ready_tok)

    mask = env._mask()
    enabled_actions = set(env.get_enable_t())
    wait_5_idx = next(idx for idx in range(env.n_actions) if env.parse_wait_action(idx) == 5)
    assert bool(mask[wait_5_idx]) is True
    assert wait_5_idx in enabled_actions
    for idx in range(env.n_actions):
        wait_duration = env.parse_wait_action(idx)
        if wait_duration is None or int(wait_duration) <= 5:
            continue
        assert bool(mask[idx]) is True
        assert idx in enabled_actions


def test_wait_5_does_not_use_event_capping():
    config = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        single_wait_durations=[5, 20],
    )
    net = ClusterTool(config=config)
    net.reset()

    for place in net.marks:
        if place.name != "LP":
            place.tokens.clear()
    pm3 = net._get_place("PM3")
    lp = net._get_place("LP")
    assert len(lp.tokens) > 0
    pm3.tokens.append(lp.tokens.pop())
    pm3.processing_time = 100
    pm3.tokens[0].stay_time = 97  # next_event_delta = 3

    _, reward_result, _, _, _ = net.step(detailed_reward=True, wait_duration=5)

    assert net.time == 5
