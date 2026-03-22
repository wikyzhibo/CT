from solutions.Continuous_model.construct_single import build_net
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import ClusterTool
from pathlib import Path


def _demo_route_config():
    return {
        "source": {"name": "LP", "capacity": 25},
        "sink": {"name": "LP_done", "capacity": 25},
        "chambers": {
            "PM7": {"kind": "process", "class": "PM", "process_time": 70, "capacity": 1},
            "PM8": {"kind": "process", "class": "PM", "process_time": 70, "capacity": 1},
            "PM9": {"kind": "process", "class": "PM", "process_time": 200, "capacity": 1},
            "PM10": {"kind": "process", "class": "PM", "process_time": 200, "capacity": 1},
            "LLC": {"kind": "buffer", "class": "LL", "process_time": 0, "capacity": 1},
            "LLD": {"kind": "loadlock", "class": "LL", "process_time": 70, "capacity": 1},
        },
        "robots": {
            "TM2": {
                "transport_place": "TM2",
                "managed_chambers": ["LP", "PM7", "PM8", "PM9", "PM10", "LLC", "LLD", "LP_done"],
                "dwell_time": 5,
                "capacity": 1,
                "priority": 10,
            },
            "TM3": {
                "transport_place": "TM3",
                "managed_chambers": ["LLC", "LLD"],
                "dwell_time": 5,
                "capacity": 1,
                "priority": 5,
            },
        },
        "routes": {
            "route_D": {
                "sequence": [
                    {"stage": {"candidates": ["LP"]}},
                    {"stage": {"candidates": ["PM7", "PM8"]}},
                    {"stage": {"candidates": ["PM9", "PM10"]}},
                    {"stage": {"candidates": ["LP_done"]}},
                ]
            },
            "route_A_repeat": {
                "sequence": [
                    {"stage": {"candidates": ["LP"]}},
                    {
                        "repeat": {
                            "count": 5,
                            "sequence": [
                                {"stage": {"candidates": ["PM7"]}},
                                {"stage": {"candidates": ["PM8"]}},
                                {"stage": {"candidates": ["LLC"]}},
                                {"stage": {"candidates": ["LLD"]}},
                            ],
                        }
                    },
                    {"stage": {"candidates": ["LP_done"]}},
                ]
            },
        },
        "legacy": {"route_code_alias": {"cascade": {"4": "route_A_repeat", "5": "route_D"}}},
    }


def test_config_driven_route_d_builds_transport_and_token_queue():
    info = build_net(
        n_wafer=1,
        ttime=5,
        robot_capacity=1,
        process_time_map={"PM7": 5, "PM8": 5, "PM9": 5, "PM10": 5},
        route_code=5,  # 仅用于 legacy alias 选 route
        device_mode="cascade",
        obs_config={"P_Residual_time": 15, "D_Residual_time": 10},
        route_config=_demo_route_config(),
        route_name="route_D",
    )

    assert "TM2" in info["id2p_name"]
    assert "t_TM2_PM7" in info["id2t_name"]
    assert "t_TM2_PM10" in info["id2t_name"]
    assert info["route_meta"]["u_targets"]["LP"] == ["PM7", "PM8"]
    assert info["route_meta"]["u_targets"]["PM7"] == ["PM9", "PM10"]
    assert info["route_meta"]["u_targets"]["PM10"] == ["LP_done"]

    q = info["token_route_queue_template"]
    assert q[0] == -1
    assert len(q) == 6
    assert q[1] == (
        info["t_route_code_map"]["t_TM2_PM7"],
        info["t_route_code_map"]["t_TM2_PM8"],
    )
    assert q[3] == (
        info["t_route_code_map"]["t_TM2_PM9"],
        info["t_route_code_map"]["t_TM2_PM10"],
    )
    assert q[-1] == info["t_route_code_map"]["t_TM2_LP_done"]


def test_build_net_uses_fixed_capacity_and_zero_m0_then_source_inject():
    info = build_net(
        n_wafer=3,
        ttime=5,
        robot_capacity=1,
        process_time_map={"PM7": 5, "PM8": 5, "PM9": 5, "PM10": 5},
        route_code=5,
        device_mode="cascade",
        obs_config={"P_Residual_time": 15, "D_Residual_time": 10},
        route_config=_demo_route_config(),
        route_name="route_D",
    )

    name_to_idx = {name: i for i, name in enumerate(info["id2p_name"])}
    cap = info["capacity"]
    m0 = info["m0"]
    marks = info["marks"]

    assert cap[name_to_idx["LP"]] == 100
    assert cap[name_to_idx["LP_done"]] == 100
    for name, idx in name_to_idx.items():
        if name in {"LP", "LP_done"}:
            continue
        assert cap[idx] == 1

    assert (m0 == 0).all()
    for place in marks:
        if place.name == "LP":
            assert len(place.tokens) == 3
        else:
            assert len(place.tokens) == 0


def test_config_driven_repeat_route_generates_loop_u_targets_and_queue():
    info = build_net(
        n_wafer=1,
        ttime=5,
        robot_capacity=1,
        process_time_map={"PM7": 5, "PM8": 5, "LLD": 5, "LLC": 0},
        route_code=4,
        device_mode="cascade",
        obs_config={"P_Residual_time": 15, "D_Residual_time": 10},
        route_config=_demo_route_config(),
        route_name="route_A_repeat",
    )

    q = info["token_route_queue_template"]
    assert len(q) == 42
    assert q[0] == -1
    assert q[1] == info["t_route_code_map"]["t_TM2_PM7"]
    assert q[3] == info["t_route_code_map"]["t_TM2_PM8"]
    assert q[5] == info["t_route_code_map"]["t_TM2_LLC"]
    assert q[7] == info["t_route_code_map"]["t_TM3_LLD"]
    assert q[-1] == info["t_route_code_map"]["t_TM2_LP_done"]

    # LLD 在重复段既可能回 PM7，也可能在最后一轮去 LP_done
    assert info["route_meta"]["u_targets"]["LLD"] == ["PM7", "LP_done"]


def test_cluster_tool_accepts_single_route_config():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode="cascade",
        route_code=5,
        single_route_config=_demo_route_config(),
        single_route_name="route_D",
        process_time_map={"PM7": 5, "PM8": 5, "PM9": 5, "PM10": 5},
    )
    net = ClusterTool(config=cfg)
    assert net._u_targets["LP"] == ["PM7", "PM8"]
    assert net._u_targets["PM7"] == ["PM9", "PM10"]
    assert net._u_targets["PM9"] == ["LP_done"]
    assert "TM2" in net.id2p_name


def test_petri_env_config_loads_single_route_config_from_path():
    cfg_path = Path("data/petri_configs/cascade.json")
    cfg = PetriEnvConfig.load(cfg_path)
    assert cfg.single_route_config is not None
    assert cfg.single_route_name in (cfg.single_route_config.get("routes") or {})
    cfg.single_route_name = "1-1"
    net = ClusterTool(config=cfg)
    # 1-1: LP -> PM7/PM8 -> LLC -> PM3/PM4 -> LLD -> PM9/PM10 -> LP_done
    assert net._u_targets["LP"] == ["PM7", "PM8"]
    assert net._u_targets["PM7"] == ["LLC"]
    assert net._u_targets["LLC"] == ["PM3", "PM4"]
    # stage 工时覆盖应生效（并按系统口径取整到 5 的倍数）
    pm7 = next(p for p in net.marks if p.name == "PM7")
    pm3 = next(p for p in net.marks if p.name == "PM3")
    lld = next(p for p in net.marks if p.name == "LLD")
    assert pm7.processing_time == 80  # route 1-1: 78s -> round to 80
    assert pm3.processing_time == 110
    assert lld.processing_time == 75
    assert net._base_proc_time_map.get("PM7") == 80
    assert net._base_proc_time_map.get("PM3") == 110
    assert net._base_proc_time_map.get("LLD") == 75
    assert net._cleaning_duration_map.get("PM3") == 88
    assert net._cleaning_trigger_map.get("PM3") == 1


def test_legacy_compatible_route_2_4_matches_route_code_5_topology():
    cfg_path = Path("data/petri_configs/cascade.json")
    cfg = PetriEnvConfig.load(cfg_path)
    cfg.route_code = 5
    cfg.single_route_name = "2-4"
    net = ClusterTool(config=cfg)
    assert net._u_targets["LP"] == ["PM7", "PM8"]
    assert net._u_targets["PM7"] == ["PM9", "PM10"]
    assert net._u_targets["PM9"] == ["LP_done"]
    assert "LLC" in net.chambers
    assert "LLD" in net.chambers
