from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import ClusterTool


def _fire_by_name(net: ClusterTool, transition_name: str) -> None:
    tid = net.id2t_name.index(transition_name)
    net.step(a1=tid, detailed_reward=True)


def test_cascade_token_route_queue_advances_on_each_fire():
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

    lp = net._get_place("LP")
    tok = lp.head()
    assert hasattr(tok, "route_queue")
    assert hasattr(tok, "route_head_idx")
    assert tok.route_queue[tok.route_head_idx] == -1

    _fire_by_name(net, "u_LP")
    d_tm2 = net._get_place("d_TM2")
    tm2_tok = d_tm2.head()
    gate = tm2_tok.route_queue[tm2_tok.route_head_idx]
    assert isinstance(gate, tuple)
    assert len(gate) == 2

    _fire_by_name(net, "t_PM7")
    pm7 = net._get_place("PM7")
    pm7_tok = pm7.head()
    assert pm7_tok.route_queue[pm7_tok.route_head_idx] == -1


def test_single_mode_t_routing_uses_route_queue_gate():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        route_code=0,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5},
    )
    net = ClusterTool(config=cfg)

    lp = net._get_place("LP")
    tok = lp.head()
    assert hasattr(tok, "route_queue")
    assert hasattr(tok, "route_head_idx")
    _fire_by_name(net, "u_LP")
    net.step(detailed_reward=True, wait_duration=5)
    enabled_names = {net.id2t_name[t] for t in net.get_enable_t()}
    assert "t_PM1" in enabled_names
    assert "t_PM3" not in enabled_names
    assert "t_PM4" not in enabled_names
