from pathlib import Path

import torch
from tensordict import TensorDict

from solutions.Continuous_model import export_inference_sequence as exporter


class _DummyNet:
    T = 2
    id2t_name = ["u_LP", "u_PM1"]


class _DummyEnv:
    def __init__(self, *args, **kwargs):
        self.net = _DummyNet()
        self.n_actions = 3
        self.observation_spec = {"observation": torch.zeros(2, dtype=torch.float32)}

    def reset(self):
        return TensorDict(
            {
                "observation": torch.tensor([0.1, 0.2], dtype=torch.float32),
                "action_mask": torch.tensor([True, False, True], dtype=torch.bool),
                "time": torch.tensor([0], dtype=torch.int64),
            },
            batch_size=[],
        )

    def step(self, step_td):
        action = int(step_td["action"].item())
        assert action == 1
        next_td = TensorDict(
            {
                "observation": torch.tensor([0.3, 0.4], dtype=torch.float32),
                "action_mask": torch.tensor([True, True, True], dtype=torch.bool),
                "time": torch.tensor([5], dtype=torch.int64),
                "finish": torch.tensor(True, dtype=torch.bool),
                "terminated": torch.tensor(True, dtype=torch.bool),
            },
            batch_size=[],
        )
        return TensorDict({"next": next_td}, batch_size=[])


class _DummyPolicy:
    def __init__(self):
        self.called = False
        self.seen_td = None

    def backbone(self, _obs):
        raise AssertionError("single 导出不应直接调用 backbone 采样")

    def __call__(self, td):
        self.called = True
        self.seen_td = td
        out = td.clone()
        out["action"] = torch.tensor([1], dtype=torch.int64)
        return out


def test_single_rollout_uses_policy_tensordict_path(monkeypatch):
    dummy_policy = _DummyPolicy()

    monkeypatch.setattr(exporter, "Env_PN_Single", _DummyEnv)
    monkeypatch.setattr(exporter, "_build_single_policy", lambda *args, **kwargs: dummy_policy)

    seq, finished, replay_overrides, reward_report = exporter._rollout_single_sequence(
        model_path=Path("dummy.pt"),
        max_steps=3,
        seed=0,
        training_phase=2,
        robot_capacity=1,
    )

    assert dummy_policy.called is True
    assert "scrap_penalty" in reward_report
    assert "release_penalty" in reward_report
    assert "idle_timeout_penalty" in reward_report
    assert dummy_policy.seen_td is not None
    assert set(dummy_policy.seen_td.keys()) == {"observation", "observation_f", "action_mask"}
    assert dummy_policy.seen_td["observation"].dtype == torch.int64
    assert dummy_policy.seen_td["observation_f"].dtype == torch.float32
    assert dummy_policy.seen_td["action_mask"].dtype == torch.bool
    assert finished is True
    assert isinstance(replay_overrides, dict)
    assert len(seq) == 1
    assert seq[0]["action"] == "u_PM1"
    assert seq[0]["actions"] == ["u_PM1"]


def test_single_rollout_retries_until_finish(monkeypatch):
    calls: list[int] = []

    def _fake_rollout_single_sequence(**kwargs):
        calls.append(kwargs["seed"])
        empty_report = {"scrap_penalty": {"count": 0, "steps": []}, "release_penalty": {"count": 0, "steps": []}, "idle_timeout_penalty": {"count": 0, "steps": []}}
        if len(calls) < 3:
            return ([{"step": 1, "time": 5, "action": "WAIT", "actions": ["WAIT"]}], False, {}, empty_report)
        return ([{"step": 1, "time": 5, "action": "u_LP", "actions": ["u_LP"]}], True, {}, empty_report)

    monkeypatch.setattr(exporter, "_rollout_single_sequence", _fake_rollout_single_sequence)

    seq, replay_overrides, reward_report = exporter._rollout_single_sequence_with_retry(
        model_path=Path("dummy.pt"),
        max_steps=10,
        seed=7,
        training_phase=2,
        robot_capacity=1,
        max_retries=10,
    )

    assert calls == [7, 8, 9]
    assert seq[0]["action"] == "u_LP"
    assert isinstance(replay_overrides, dict)
    assert isinstance(reward_report, dict)
