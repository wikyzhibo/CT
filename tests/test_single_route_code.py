from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import PetriSingleDevice
from solutions.Continuous_model.env_single import Env_PN_Single


def _fire_by_name(net: PetriSingleDevice, transition_name: str) -> None:
    tid = net.id2t_name.index(transition_name)
    net.step(a1=tid, detailed_reward=True)


def test_single_route_code0_keeps_legacy_topology():
    cfg = PetriEnvConfig(n_wafer=1, stop_on_scrap=False, single_route_code=0)
    net = PetriSingleDevice(config=cfg)

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
        single_route_code=1,
        single_process_time_map={"PM1": 5, "PM3": 5, "PM4": 5, "PM6": 5},
    )
    net = PetriSingleDevice(config=cfg)

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
    assert obs_dim == 1 + 4 + 9 * 3
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
    assert obs_dim == 1 + 4 + 9 * 4
    td = env._reset(None)
    assert int(td["observation"].shape[-1]) == obs_dim
