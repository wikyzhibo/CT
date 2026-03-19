import math

import numpy as np
import pytest

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import PM
from solutions.Continuous_model.pn_single import ClusterTool


def _expected_near_cleaning_norm(trigger_wafers: int, processed_count: int, is_cleaning: bool) -> float:
    trigger = max(1, int(trigger_wafers))
    processed = max(0, int(processed_count))
    remaining_runs = max(trigger - processed, 0)
    near_cleaning_norm = max(0.0, min((2.0 - float(remaining_runs)) / 2.0, 1.0))
    if is_cleaning:
        return 0.0
    return near_cleaning_norm


@pytest.mark.parametrize("trigger_wafers", [1, 2, 3, 13])
def test_pm_obs_near_cleaning_norm_table(trigger_wafers: int):
    pm = PM(
        name="PM_TEST",
        capacity=1,
        processing_time=300,
        cleaning_trigger_wafers=trigger_wafers,
    )
    candidates = {-3, -1, 0, 1, 2, trigger_wafers - 2, trigger_wafers - 1, trigger_wafers, trigger_wafers + 3}

    for processed_count in sorted(candidates):
        pm.processed_wafer_count = int(processed_count)
        pm.is_cleaning = False
        obs = pm.get_obs()
        assert len(obs) == 9
        expected = _expected_near_cleaning_norm(trigger_wafers, processed_count, is_cleaning=False)
        assert math.isclose(float(obs[8]), expected, rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.parametrize("trigger_wafers,processed_count", [(1, 0), (2, 1), (3, 3), (13, 20)])
def test_pm_obs_near_cleaning_norm_forced_zero_while_cleaning(trigger_wafers: int, processed_count: int):
    pm = PM(
        name="PM_TEST",
        capacity=1,
        processing_time=300,
        cleaning_trigger_wafers=trigger_wafers,
    )
    pm.processed_wafer_count = int(processed_count)
    pm.is_cleaning = True
    pm.cleaning_remaining = 100
    obs = pm.get_obs()
    assert len(obs) == 9
    assert math.isclose(float(obs[8]), 0.0, rel_tol=0.0, abs_tol=1e-6)


def test_cluster_obs_near_cleaning_norm_progression_and_cleaning_gate():
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        route_code=0,
        process_time_map={"PM1": 5, "PM3": 5, "PM4": 5},
        cleaning_enabled=True,
        cleaning_targets=["PM3"],
        cleaning_trigger_wafers=3,
    )
    net = ClusterTool(config=cfg)
    net.reset()
    pm3 = net._get_place("PM3")
    pm3_offset = next(spec["offset"] for spec in net._obs_specs if spec["name"] == "PM3")

    def read_pm3_near_cleaning() -> float:
        return float(net.get_obs()[pm3_offset + 8])

    # 卸载 2 次后进入临近窗口：0 -> 0.5
    assert math.isclose(read_pm3_near_cleaning(), 0.0, rel_tol=0.0, abs_tol=1e-6)
    net._on_processing_unload("PM3")
    assert math.isclose(read_pm3_near_cleaning(), 0.0, rel_tol=0.0, abs_tol=1e-6)
    net._on_processing_unload("PM3")
    assert math.isclose(read_pm3_near_cleaning(), 0.5, rel_tol=0.0, abs_tol=1e-6)

    # 手动设置触发边界态，验证 r=0 -> 1.0 的聚合观测语义。
    pm3.processed_wafer_count = 3
    pm3.is_cleaning = False
    assert math.isclose(read_pm3_near_cleaning(), 1.0, rel_tol=0.0, abs_tol=1e-6)

    # 一旦进入清洗，near_cleaning_norm 强制归零。
    net._on_processing_unload("PM3")
    assert pm3.is_cleaning is True
    assert math.isclose(read_pm3_near_cleaning(), 0.0, rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.parametrize(
    "device_mode,route_code,process_time_map,expected_dim",
    [
        ("single", 0, {"PM1": 5, "PM3": 5, "PM4": 5}, 36),
        ("single", 1, {"PM1": 5, "PM3": 5, "PM4": 5, "PM6": 5}, 45),
        (
            "cascade",
            2,
            {"PM7": 5, "PM8": 5, "PM1": 5, "PM2": 5, "LLD": 5, "PM9": 5, "PM10": 5},
            77,
        ),
        (
            "cascade",
            3,
            {"PM7": 5, "PM8": 5, "PM1": 5, "PM2": 5, "LLD": 5},
            59,
        ),
    ],
)
def test_obs_shape_regression_across_device_modes(
    device_mode: str,
    route_code: int,
    process_time_map: dict,
    expected_dim: int,
):
    cfg = PetriEnvConfig(
        n_wafer=1,
        stop_on_scrap=False,
        device_mode=device_mode,
        route_code=route_code,
        process_time_map=process_time_map,
    )
    net = ClusterTool(config=cfg)
    obs = net.get_obs()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert int(obs.shape[-1]) == int(net.get_obs_dim()) == expected_dim
