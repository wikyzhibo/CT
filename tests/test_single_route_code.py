import pytest

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import ClusterTool
from solutions.Continuous_model.env_single import Env_PN_Single


def _fire_by_name(net: ClusterTool, transition_name: str) -> None:
    tid = net.id2t_name.index(transition_name)
    net.step(a1=tid, detailed_reward=True)


FULL_PROCESS_TIME_MAP = {
    "PM1": 300,
    "PM2": 300,
    "PM3": 600,
    "PM4": 600,
    "PM6": 300,
    "PM7": 70,
    "PM8": 70,
    "PM9": 200,
    "PM10": 200,
    "LLD": 70,
    "LLC": 0,
}


@pytest.mark.parametrize(
    "device_mode,route_code,expected_keys",
    [
        ("single", 0, {"PM1", "PM3", "PM4"}),
        ("single", 1, {"PM1", "PM3", "PM4", "PM6"}),
        ("cascade", 1, {"PM7", "PM8", "PM1", "PM2", "PM3", "PM4", "LLD", "PM9", "PM10"}),
        ("cascade", 2, {"PM7", "PM8", "PM1", "PM2", "LLD", "PM9", "PM10"}),
        ("cascade", 3, {"PM7", "PM8", "PM1", "PM2", "LLD"}),
        ("cascade", 4, {"PM7", "PM8", "LLD"}),
        ("cascade", 5, {"PM7", "PM8", "PM9", "PM10"}),
    ],
)
def test_episode_proc_time_map_keys_match_route_chambers(
    device_mode: str,
    route_code: int,
    expected_keys: set[str],
):
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode=device_mode,
        route_code=route_code,
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    net = ClusterTool(config=cfg)
    assert set(net._episode_proc_time_map.keys()) == expected_keys


def test_route_code_string_is_normalized_to_int_for_cascade_route5():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code="5",
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    net = ClusterTool(config=cfg)
    assert net.route_code == 5
    assert net.single_route_code == 5
    assert set(net._episode_proc_time_map.keys()) == {"PM7", "PM8", "PM9", "PM10"}


def test_invalid_device_mode_raises_value_error():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="invalid_mode",
        route_code=1,
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    with pytest.raises(ValueError, match="device_mode"):
        ClusterTool(config=cfg)


def test_invalid_route_code_raises_value_error_instead_of_fallback():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=99,
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    with pytest.raises(ValueError, match="route_code"):
        ClusterTool(config=cfg)


def test_single_route_code0_keeps_legacy_topology():
    cfg = PetriEnvConfig(n_wafer=1, stop_on_scrap=False, route_code=0)
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 0
    assert net._u_targets["PM1"] == ["PM3", "PM4"]
    assert net._u_targets["PM3"] == ["LP_done"]
    assert net._u_targets["PM4"] == ["LP_done"]
    assert "t_PM6" not in net.id2t_name
    assert "u_PM6" not in net.id2t_name


def test_single_route_code1_runs_through_pm6_before_done():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        route_code=1,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5, "PM6": 5},
    )
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 1
    assert net._u_targets["PM1"] == ["PM3", "PM4"]
    assert net._u_targets["PM3"] == ["PM6"]
    assert net._u_targets["PM4"] == ["PM6"]
    assert net._u_targets["PM6"] == ["LP_done"]
    assert "t_PM6" in net.id2t_name
    assert "u_PM6" in net.id2t_name

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM1")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM1")
    _fire_by_name(net, "t_PM3")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM3")
    _fire_by_name(net, "t_PM6")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM6")
    _fire_by_name(net, "t_LP_done")

    assert net.done_count == 1


def test_env_single_accepts_route_code_override():
    env = Env_PN_Single(
        detailed_reward=False,
        robot_capacity=1,
        route_code=1,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5, "PM6": 5},
    )
    assert env.net.single_route_code == 1
    assert "t_PM6" in env.net.id2t_name


def test_place_obs_route_code0_uses_three_pm_features():
    env = Env_PN_Single(
        detailed_reward=False,
        robot_capacity=1,
        route_code=0,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5},
    )
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    assert obs_dim == 1 + 8 + 9 * 3
    td = env._reset(None)
    assert int(td["observation"].shape[-1]) == obs_dim


def test_place_obs_route_code1_includes_pm6_features():
    env = Env_PN_Single(
        detailed_reward=False,
        robot_capacity=1,
        route_code=1,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5, "PM6": 5},
    )
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    assert obs_dim == 1 + 8 + 9 * 4
    td = env._reset(None)
    assert int(td["observation"].shape[-1]) == obs_dim


def test_place_obs_cascade_includes_llc_lld_with_core4_features():
    env = Env_PN_Single(
        detailed_reward=False,
        device_mode="cascade",
        robot_capacity=1,
        route_code=2,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    chamber_names = [
        spec["name"]
        for spec in env.net._obs_specs
        if spec["name"].startswith("PM") or spec["name"] in {"LLC", "LLD"}
    ]
    assert "LLC" in chamber_names
    assert "LLD" in chamber_names
    chamber_feature_dim = sum(
        int(spec["dim"])
        for spec in env.net._obs_specs
        if spec["name"] in chamber_names
    )
    tm_feature_dim = sum(
        int(spec["dim"])
        for spec in env.net._obs_specs
        if spec["name"] in {"d_TM2", "d_TM3"}
    )
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    assert tm_feature_dim == 24
    assert obs_dim == 1 + tm_feature_dim + chamber_feature_dim
    td = env._reset(None)
    assert int(td["observation"].shape[-1]) == obs_dim


def test_cascade_tm_target_onehot_uses_fixed_8_destination_slots():
    env = Env_PN_Single(
        detailed_reward=False,
        device_mode="cascade",
        robot_capacity=1,
        route_code=2,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    tm2 = next(p for p in env.net.marks if p.name == "d_TM2")
    tm3 = next(p for p in env.net.marks if p.name == "d_TM3")
    assert tm2.get_obs_dim() == 12
    assert tm3.get_obs_dim() == 12
    assert set(tm2._target_onehot_map.keys()) == {
        "PM7",
        "PM8",
        "PM9",
        "PM10",
        "LLC",
        "LLD",
        "LP_done",
        "LP",
    }
    assert set(tm3._target_onehot_map.keys()) == {
        "PM1",
        "PM2",
        "PM3",
        "PM4",
        "PM5",
        "PM6",
        "LLC",
        "LLD",
    }


def test_ll_obs_adds_in_out_onehot_for_tm3_tm2():
    env = Env_PN_Single(
        detailed_reward=False,
        device_mode="cascade",
        robot_capacity=1,
        route_code=2,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    net = env.net

    ll_specs = {spec["name"]: spec for spec in net._obs_specs if spec["name"] in {"LLC", "LLD"}}
    assert int(ll_specs["LLC"]["dim"]) == 6
    assert int(ll_specs["LLD"]["dim"]) == 6

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM7")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM7")
    _fire_by_name(net, "t_LLC")

    obs = net.get_obs()
    llc_off = int(ll_specs["LLC"]["offset"])
    # LLC 下一跳由 TM3 搬运（in=1, out=0）
    assert obs[llc_off + 4] == pytest.approx(1.0)
    assert obs[llc_off + 5] == pytest.approx(0.0)

    _fire_by_name(net, "u_LLC")
    _fire_by_name(net, "t_PM1")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM1")
    _fire_by_name(net, "t_LLD")

    obs = net.get_obs()
    lld_off = int(ll_specs["LLD"]["offset"])
    # LLD 下一跳由 TM2 搬运（in=0, out=1）
    assert obs[lld_off + 4] == pytest.approx(0.0)
    assert obs[lld_off + 5] == pytest.approx(1.0)


def test_cascade_route_code2_uses_pm1_pm2_only():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=2,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 2
    assert net._u_targets["LLC"] == ["PM1", "PM2"]
    assert "PM3" not in net._single_process_chambers
    assert "PM4" not in net._single_process_chambers
    assert "t_PM3" not in net.id2t_name
    assert "t_PM4" not in net.id2t_name
    assert "u_PM3" not in net.id2t_name
    assert "u_PM4" not in net.id2t_name


def test_cascade_route_code2_defaults_pm1_pm2_to_300s():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=2,
        process_time_map={
            "PM7": 70,
            "PM8": 70,
            "LLD": 70,
            "PM9": 200,
            "PM10": 200,
        },
    )
    net = ClusterTool(config=cfg)
    pm1 = net._get_place("PM1")
    pm2 = net._get_place("PM2")
    assert pm1.processing_time == 300
    assert pm2.processing_time == 300


def test_cascade_route_code2_parallel_pairs_use_round_robin():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=2,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    net = ClusterTool(config=cfg)

    # 使能检查阶段（advance_round_robin=False）不应推进轮换指针
    assert net._select_target_for_source("LP") == "PM7"
    assert net._select_target_for_source("LP") == "PM7"

    # 真实发射阶段（advance_round_robin=True）应轮换
    assert net._select_target_for_source("LP", advance_round_robin=True) == "PM7"
    assert net._select_target_for_source("LP", advance_round_robin=True) == "PM8"
    assert net._select_target_for_source("LLC", advance_round_robin=True) == "PM1"
    assert net._select_target_for_source("LLC", advance_round_robin=True) == "PM2"
    assert net._select_target_for_source("LLD", advance_round_robin=True) == "PM9"
    assert net._select_target_for_source("LLD", advance_round_robin=True) == "PM10"


def test_cascade_route_code3_uses_lld_then_done_without_pm9_pm10():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=3,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
        },
    )
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 3
    assert net._u_targets["LLC"] == ["PM1", "PM2"]
    assert net._u_targets["PM1"] == ["LLD"]
    assert net._u_targets["PM2"] == ["LLD"]
    assert net._u_targets["LLD"] == ["LP_done"]
    assert "PM9" not in net._single_process_chambers
    assert "PM10" not in net._single_process_chambers
    assert "t_PM9" not in net.id2t_name
    assert "t_PM10" not in net.id2t_name
    assert "u_PM9" not in net.id2t_name
    assert "u_PM10" not in net.id2t_name


def test_cascade_route_code3_can_finish_via_lld_then_lp_done():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=3,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM1": 5,
            "PM2": 5,
            "LLD": 5,
        },
    )
    net = ClusterTool(config=cfg)

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM7")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM7")
    _fire_by_name(net, "t_LLC")
    _fire_by_name(net, "u_LLC")
    _fire_by_name(net, "t_PM1")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM1")
    _fire_by_name(net, "t_LLD")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_LLD")
    _fire_by_name(net, "t_LP_done")

    assert net.done_count == 1


def test_cascade_route_code4_topology_is_strict_loop_route():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=4,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "LLD": 5,
        },
    )
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 4
    assert net.id2t_name == [
        "u_LP",
        "t_PM7",
        "u_PM7",
        "t_PM8",
        "u_PM8",
        "t_LLC",
        "u_LLC",
        "t_LLD",
        "u_LLD",
        "t_LP_done",
    ]
    assert net._u_targets["LP"] == ["PM7"]
    assert net._u_targets["PM7"] == ["PM8"]
    assert net._u_targets["PM8"] == ["LLC"]
    assert net._u_targets["LLC"] == ["LLD"]
    assert net._u_targets["LLD"] == ["PM7", "LP_done"]
    assert "PM1" not in net.id2p_name
    assert "PM2" not in net.id2p_name
    assert "PM3" not in net.id2p_name
    assert "PM4" not in net.id2p_name
    assert "PM9" not in net.id2p_name
    assert "PM10" not in net.id2p_name


def test_cascade_route_code4_lld_targets_pm7_first_four_then_lp_done():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=4,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "LLD": 5,
        },
    )
    net = ClusterTool(config=cfg)

    _fire_by_name(net, "u_LP")

    for lap in range(5):
        _fire_by_name(net, "t_PM7")
        _fire_by_name(net, "u_PM7")
        _fire_by_name(net, "t_PM8")
        _fire_by_name(net, "u_PM8")
        _fire_by_name(net, "t_LLC")
        _fire_by_name(net, "u_LLC")
        _fire_by_name(net, "t_LLD")
        _fire_by_name(net, "u_LLD")

        d_tm2 = net._get_place("d_TM2")
        assert len(d_tm2.tokens) == 1
        target = d_tm2.head()._target_place
        net.step(detailed_reward=True, wait_duration=5)
        enabled_names = {net.id2t_name[t] for t in net._get_enable_t()}
        if lap < 4:
            assert target == "PM7"
            assert "t_PM7" in enabled_names
            assert "t_LP_done" not in enabled_names
        else:
            assert target == "LP_done"
            assert "t_LP_done" in enabled_names
            assert "t_PM7" not in enabled_names

    _fire_by_name(net, "t_LP_done")
    assert net.done_count == 1


def test_cascade_route_code4_manual_takt_interval_blocks_u_lp_until_due():
    cfg = PetriEnvConfig(
        n_wafer=2,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=4,
        route4_takt_interval=30,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "LLD": 5,
        },
    )
    net = ClusterTool(config=cfg)
    u_lp_idx = net.id2t_name.index("u_LP")

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM7")
    _fire_by_name(net, "u_PM7")
    _fire_by_name(net, "t_PM8")

    mask = net.get_action_mask(
        wait_action_start=net.T,
        n_actions=net.T + len(net.wait_durations),
    )
    assert not bool(mask[u_lp_idx])

    reasons = net.get_enable_actions_with_reasons(wait_action_start=net.T)
    u_lp_disabled = [item for item in reasons["disabled"] if int(item["action"]) == u_lp_idx]
    assert any(item.get("reason") == "takt_release_limit" for item in u_lp_disabled)

    elapsed = net.time - net._last_u_LP_fire_time
    net.advance_time(max(0, 30 - int(elapsed)), event_reason="test_route4_takt")

    mask2 = net.get_action_mask(
        wait_action_start=net.T,
        n_actions=net.T + len(net.wait_durations),
    )
    assert bool(mask2[u_lp_idx])


def test_cascade_route_code5_uses_pm7pm8_then_pm9pm10():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    net = ClusterTool(config=cfg)

    assert net.single_route_code == 5
    assert net._u_targets["LP"] == ["PM7", "PM8"]
    assert net._u_targets["PM7"] == ["PM9", "PM10"]
    assert net._u_targets["PM8"] == ["PM9", "PM10"]
    assert net._u_targets["PM9"] == ["LP_done"]
    assert net._u_targets["PM10"] == ["LP_done"]

    assert "LLC" not in net.id2p_name
    assert "LLD" not in net.id2p_name
    assert "PM1" not in net.id2p_name
    assert "PM2" not in net.id2p_name
    assert "PM3" not in net.id2p_name
    assert "PM4" not in net.id2p_name

    assert "t_LLC" not in net.id2t_name
    assert "u_LLC" not in net.id2t_name
    assert "t_LLD" not in net.id2t_name
    assert "u_LLD" not in net.id2t_name


def test_cascade_route_code5_defaults_pm7pm8_70_pm9pm10_200():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        process_time_map={},
    )
    net = ClusterTool(config=cfg)
    assert net._get_place("PM7").processing_time == 70
    assert net._get_place("PM8").processing_time == 70
    assert net._get_place("PM9").processing_time == 200
    assert net._get_place("PM10").processing_time == 200


def test_cascade_route_code5_can_finish_via_pm9_then_lp_done():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        process_time_map={
            "PM7": 5,
            "PM8": 5,
            "PM9": 5,
            "PM10": 5,
        },
    )
    net = ClusterTool(config=cfg)

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM7")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM7")
    _fire_by_name(net, "t_PM9")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM9")
    _fire_by_name(net, "t_LP_done")

    assert net.done_count == 1


def test_cascade_route5_filters_full_process_time_map_to_route_chambers():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    net = ClusterTool(config=cfg)
    assert set(net._episode_proc_time_map.keys()) == {"PM7", "PM8", "PM9", "PM10"}


def test_cascade_route5_takt_input_contains_only_two_route_stages(monkeypatch):
    captured: dict = {}

    def _fake_analyze_cycle(stages, max_parts=10000):
        captured["stages"] = [dict(stage) for stage in stages]
        return {
            "fast_takt": 100,
            "peak_slow_takts": [],
            "cycle_length": 100,
            "cycle_takts": [100] * 100,
        }

    monkeypatch.setattr(
        "solutions.Continuous_model.pn_single.analyze_cycle",
        _fake_analyze_cycle,
    )
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        process_time_map=dict(FULL_PROCESS_TIME_MAP),
    )
    net = ClusterTool(config=cfg)
    assert net._takt_result is not None
    assert [stage["name"] for stage in captured["stages"]] == ["s1", "s2"]
    assert [stage["m"] for stage in captured["stages"]] == [2, 2]
    assert [stage["p"] for stage in captured["stages"]] == [70, 200]


def test_single_max_wafers_blocks_u_lp_when_wip_reaches_limit():
    cfg = PetriEnvConfig(
        n_wafer=2,
        stop_on_scrap=False,
        route_code=0,
        max_wafers_in_system=1,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5},
    )
    net = ClusterTool(config=cfg)
    # 聚焦 max_wafers 语义，避免 takt 对 u_LP 产生干扰。
    net._takt_result = None
    u_lp_idx = net.id2t_name.index("u_LP")

    _fire_by_name(net, "u_LP")
    assert net.entered_wafer_count == 1
    _fire_by_name(net, "t_PM1")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM1")
    d_tm1 = net._get_place("d_TM1")
    stage2_target = str(d_tm1.head()._target_place)
    _fire_by_name(net, f"t_{stage2_target}")

    mask = net.get_action_mask(
        wait_action_start=net.T,
        n_actions=net.T + len(net.wait_durations),
    )
    assert not bool(mask[u_lp_idx])

    reasons = net.get_enable_actions_with_reasons(wait_action_start=net.T)
    u_lp_disabled = [item for item in reasons["disabled"] if int(item["action"]) == u_lp_idx]
    assert any(item.get("reason") == "max_wafers_in_system_limit" for item in u_lp_disabled)


def test_single_max_wafers_releases_slot_after_lp_done():
    cfg = PetriEnvConfig(
        n_wafer=2,
        stop_on_scrap=False,
        route_code=0,
        max_wafers_in_system=1,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5},
    )
    net = ClusterTool(config=cfg)
    # 聚焦 max_wafers 语义，避免 takt 对 u_LP 产生干扰。
    net._takt_result = None
    u_lp_idx = net.id2t_name.index("u_LP")

    _fire_by_name(net, "u_LP")
    _fire_by_name(net, "t_PM1")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, "u_PM1")
    d_tm1 = net._get_place("d_TM1")
    stage2_target = str(d_tm1.head()._target_place)
    _fire_by_name(net, f"t_{stage2_target}")
    net.step(detailed_reward=True, wait_duration=5)
    _fire_by_name(net, f"u_{stage2_target}")
    _fire_by_name(net, "t_LP_done")

    assert net.done_count == 1
    assert net.entered_wafer_count == 0

    mask = net.get_action_mask(
        wait_action_start=net.T,
        n_actions=net.T + len(net.wait_durations),
    )
    assert bool(mask[u_lp_idx])
