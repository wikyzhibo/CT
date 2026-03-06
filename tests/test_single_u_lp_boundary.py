import pytest

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn_single import PetriSingleDevice


def _make_env(**kwargs) -> PetriSingleDevice:
    config = PetriEnvConfig(
        n_wafer=2,
        single_robot_capacity=2,
        single_cleaning_enabled=True,
        single_u_lp_boundary_enabled=True,
        **kwargs,
    )
    env = PetriSingleDevice(config=config)
    env.reset()
    return env


def _set_cleaning(env: PetriSingleDevice, chamber: str, remaining: int) -> None:
    place = env._get_place(chamber)
    place.is_cleaning = True
    place.cleaning_remaining = int(remaining)


def test_u_lp_reverse_boundary_allows_when_pm3_pm4_available():
    env = _make_env()
    assert env._allow_u_lp_by_reverse_boundary() is True


def test_u_lp_reverse_boundary_uses_min_of_pm3_pm4():
    env = _make_env()
    _set_cleaning(env, "PM3", 260)
    _set_cleaning(env, "PM4", 50)
    assert env._allow_u_lp_by_reverse_boundary() is True


def test_u_lp_reverse_boundary_blocks_when_both_cleaning_long():
    env = _make_env()
    _set_cleaning(env, "PM3", 240)
    _set_cleaning(env, "PM4", 220)
    assert env._allow_u_lp_by_reverse_boundary() is False


def test_u_lp_boundary_only_applies_stage2_not_stage1():
    env = _make_env()
    _set_cleaning(env, "PM3", 240)
    _set_cleaning(env, "PM4", 220)

    u_lp_idx = env.id2t_name.index("u_LP")
    stage1_enabled = env._get_enable_t_stage1()
    stage2_enabled = env._apply_enable_stage2(stage1_enabled)

    assert u_lp_idx in stage1_enabled
    assert u_lp_idx not in stage2_enabled

