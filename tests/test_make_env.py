import os

import pytest

from solutions.A.rl_env import Env_PN_Concurrent, Env_PN_Single, make_env


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_make_env_returns_single_env_and_reset_works():
    env = make_env(runtime_mode="single", eval_mode=True)

    assert isinstance(env, Env_PN_Single)
    assert env.eval_mode is True

    td = env.reset()

    assert "observation" in td.keys()
    assert "action_mask" in td.keys()


def test_make_env_returns_concurrent_env_and_reset_works():
    env = make_env(runtime_mode="concurrent", device_mode="cascade")

    assert isinstance(env, Env_PN_Concurrent)

    td = env.reset()

    assert "observation" in td.keys()
    assert "action_mask_tm2" in td.keys()
    assert "action_mask_tm3" in td.keys()


def test_make_env_rejects_concurrent_on_single_device_mode():
    with pytest.raises(ValueError, match="并发环境仅支持 cascade 设备模式"):
        make_env(runtime_mode="concurrent", device_mode="single")


def test_make_env_accepts_process_time_map_aliases(monkeypatch):
    captured_direct = {}
    captured_alias = {}

    class DummySingleEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_single_env_direct(**kwargs):
        captured_direct.update(kwargs)
        return DummySingleEnv(**kwargs)

    def fake_single_env_alias(**kwargs):
        captured_alias.update(kwargs)
        return DummySingleEnv(**kwargs)

    monkeypatch.setattr("solutions.A.rl_env.Env_PN_Single", fake_single_env_direct)
    direct_env = make_env(
        runtime_mode="single",
        env_overrides={"process_time_map": {"PM1": 15}},
    )

    monkeypatch.setattr("solutions.A.rl_env.Env_PN_Single", fake_single_env_alias)
    alias_env = make_env(
        runtime_mode="single",
        env_overrides={"single_process_time_map": {"PM1": 15}},
    )

    assert isinstance(direct_env, DummySingleEnv)
    assert isinstance(alias_env, DummySingleEnv)
    assert captured_direct["process_time_map"] == {"PM1": 15}
    assert captured_alias["process_time_map"] == {"PM1": 15}


def test_build_adapter_returns_expected_adapter_types():
    pytest.importorskip("PySide6")

    from visualization.main import build_adapter
    from visualization.petri_adapter import PetriAdapter
    from visualization.petri_single_adapter import PetriSingleAdapter

    single_adapter = build_adapter("petri", step_verbose=False, concurrent=False)
    concurrent_adapter = build_adapter(
        "petri",
        device_mode="cascade",
        env_overrides={"runtime_mode": "concurrent"},
        step_verbose=False,
    )

    assert isinstance(single_adapter, PetriSingleAdapter)
    assert isinstance(concurrent_adapter, PetriAdapter)
