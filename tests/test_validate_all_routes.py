import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config.training.training_config import PPOTrainingConfig
from solutions.A.eval import export_inference_sequence, validate_all_routes
from solutions.A.ppo_trainer import _train_concurrent
from solutions.A.rl_env import Env_PN_Concurrent


class TestTrainingProfiles(unittest.TestCase):
    def test_resolve_training_profile_path(self) -> None:
        path = validate_all_routes._resolve_training_profile_path("simple")
        self.assertEqual(path.name, "simple.yaml")

    def test_resolve_training_profile_rejects_unknown_profile(self) -> None:
        with self.assertRaises(ValueError):
            validate_all_routes._resolve_training_profile_path("unknown")


class TestConcurrentEnvOverrides(unittest.TestCase):
    def test_concurrent_env_accepts_runtime_overrides(self) -> None:
        env = Env_PN_Concurrent(
            device="cpu",
            n_wafer=6,
            single_route_name="2-2",
        )
        self.assertEqual(env.n_wafer, 6)
        self.assertEqual(env.net.n_wafer, 6)
        self.assertEqual(env.net.single_route_name, "2-2")


class TestValidateAllRoutes(unittest.TestCase):
    def test_run_all_routes_uses_profile_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            best_model_path = Path(tmpdir) / "best.pt"
            best_model_path.write_bytes(b"stub")

            with (
                patch.object(
                    validate_all_routes,
                    "_load_route_config",
                    return_value={"routes": {"2-2": {"path": "dummy"}}},
                ),
                patch.object(
                    validate_all_routes,
                    "train_single",
                    return_value=(
                        {},
                        None,
                        {
                            "best_batch_index": 2,
                            "best_batch_makespan": 321.0,
                            "training_time_seconds": 45.5,
                            "best_model_path": str(best_model_path),
                            "run_name": "validate_2-2_trainW6",
                        },
                    ),
                ) as mock_train,
                patch.object(
                    validate_all_routes,
                    "rollout_and_export",
                    return_value={
                        "action_series_path": Path(tmpdir) / "eval.json",
                        "makespan": 999,
                        "n_wafer": 75,
                        "finished": True,
                        "single_route_name": "2-2",
                    },
                ),
            ):
                rows, summary_path = validate_all_routes.run_all_routes(
                    route_wafer_plan={"2-2": {"train": 6, "eval": 75}},
                    route_training_profile={"2-2": "simple"},
                )

            self.assertTrue(summary_path.is_file())
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["route_name"], "2-2")
            self.assertEqual(row["training_profile"], "simple")
            self.assertEqual(row["train_n_wafer"], 6)
            self.assertEqual(row["eval_n_wafer"], 75)
            self.assertEqual(row["train_best_makespan"], 321.0)
            self.assertEqual(row["eval_makespan"], 999)

            train_kwargs = mock_train.call_args.kwargs
            self.assertTrue(train_kwargs["concurrent"])
            self.assertTrue(train_kwargs["batch_progress_only"])
            self.assertEqual(train_kwargs["progress_label"], "2-2 [simple]")
            self.assertEqual(train_kwargs["env_overrides"]["n_wafer"], 6)
            self.assertEqual(train_kwargs["env_overrides"]["single_route_name"], "2-2")
            self.assertIsInstance(train_kwargs["config"], PPOTrainingConfig)
            self.assertEqual(train_kwargs["config"].total_batch, 15)


class TestExportInferenceSequence(unittest.TestCase):
    def test_rollout_and_export_returns_structured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"stub")
            output_path = Path(tmpdir) / "sequence.json"
            dummy_env = SimpleNamespace(
                net=SimpleNamespace(
                    n_wafer=75,
                    time=1234,
                    single_route_name="2-2",
                    render_gantt=lambda *args, **kwargs: None,
                )
            )

            with (
                patch.object(
                    export_inference_sequence,
                    "_rollout_concurrent_sequence_with_retry",
                    return_value=([{"step": 1, "time": 1234, "actions": ["WAIT", "WAIT"]}], {}, {}, True, dummy_env),
                ) as mock_retry,
                patch.object(
                    export_inference_sequence,
                    "_action_sequence_export_path",
                    return_value=output_path,
                ),
            ):
                result = export_inference_sequence.rollout_and_export(
                    model_path=model_path,
                    seed=0,
                    out_name="tmp",
                    concurrent=True,
                    retry=10,
                    env_overrides={"n_wafer": 75, "single_route_name": "2-2"},
                    verbose=False,
                )

            self.assertEqual(result["action_series_path"], output_path)
            self.assertEqual(result["makespan"], 1234)
            self.assertEqual(result["n_wafer"], 75)
            self.assertTrue(result["finished"])
            self.assertEqual(result["single_route_name"], "2-2")
            self.assertEqual(mock_retry.call_args.kwargs["env_overrides"]["n_wafer"], 75)


class TestConcurrentTrainerProgress(unittest.TestCase):
    def test_batch_progress_only_hides_config_and_batch_logs(self) -> None:
        class _DummyEnv:
            def __init__(self):
                self.observation_spec = {"observation": torch.zeros(3)}
                self.n_actions_tm2 = 2
                self.n_actions_tm3 = 2
                self.net = SimpleNamespace(single_route_name="2-2")

        rollout_steps = [
            {
                "obs": torch.zeros((1, 1, 3), dtype=torch.float32),
                "next_obs": torch.zeros((1, 1, 3), dtype=torch.float32),
                "mask_tm2": torch.ones((1, 1, 2), dtype=torch.bool),
                "mask_tm3": torch.ones((1, 1, 2), dtype=torch.bool),
                "actions_tm2": torch.zeros((1, 1), dtype=torch.int64),
                "actions_tm3": torch.zeros((1, 1), dtype=torch.int64),
                "log_probs_tm2": torch.zeros((1, 1), dtype=torch.float32),
                "log_probs_tm3": torch.zeros((1, 1), dtype=torch.float32),
                "rewards": torch.tensor([[10.0]], dtype=torch.float32),
                "dones": torch.tensor([[True]], dtype=torch.bool),
                "finish": torch.tensor([[True]], dtype=torch.bool),
                "scrap": torch.tensor([[False]], dtype=torch.bool),
                "time": torch.tensor([[123]], dtype=torch.int64),
            },
            {
                "obs": torch.zeros((1, 1, 3), dtype=torch.float32),
                "next_obs": torch.zeros((1, 1, 3), dtype=torch.float32),
                "mask_tm2": torch.ones((1, 1, 2), dtype=torch.bool),
                "mask_tm3": torch.ones((1, 1, 2), dtype=torch.bool),
                "actions_tm2": torch.zeros((1, 1), dtype=torch.int64),
                "actions_tm3": torch.zeros((1, 1), dtype=torch.int64),
                "log_probs_tm2": torch.zeros((1, 1), dtype=torch.float32),
                "log_probs_tm3": torch.zeros((1, 1), dtype=torch.float32),
                "rewards": torch.tensor([[5.0]], dtype=torch.float32),
                "dones": torch.tensor([[True]], dtype=torch.bool),
                "finish": torch.tensor([[True]], dtype=torch.bool),
                "scrap": torch.tensor([[False]], dtype=torch.bool),
                "time": torch.tensor([[456]], dtype=torch.int64),
            },
        ]

        def _collect_rollout(*args, **kwargs):
            idx = _collect_rollout.calls
            _collect_rollout.calls += 1
            return rollout_steps[idx], {}

        _collect_rollout.calls = 0

        cfg = PPOTrainingConfig(
            n_hidden=8,
            n_layer=2,
            total_batch=2,
            sub_batch_size=1,
            num_epochs=1,
            device="cpu",
            seed=1,
        )

        stdout = io.StringIO()
        with (
            patch("solutions.A.ppo_trainer._build_concurrent_env", return_value=_DummyEnv()),
            patch("solutions.A.ppo_trainer.collect_rollout_ultra_concurrent", side_effect=_collect_rollout),
            patch("solutions.A.ppo_trainer._compute_gae_and_returns", return_value=(torch.zeros((1, 1)), torch.zeros((1, 1)))),
            patch("solutions.A.ppo_trainer._flatten_rollout_concurrent", return_value={}),
            patch(
                "solutions.A.ppo_trainer._ppo_update_batched_concurrent",
                return_value={"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0},
            ),
            patch("solutions.A.ppo_trainer.plot_metrics"),
            patch("solutions.A.ppo_trainer._postprocess_training_artifacts"),
            redirect_stdout(stdout),
        ):
            _log, _policy, summary = _train_concurrent(
                config=cfg,
                artifact_dir="unit_progress",
                rollout_n_envs=1,
                env_overrides={"single_route_name": "2-2", "n_wafer": 6},
                batch_progress_only=True,
                progress_label="2-2 [simple]",
                return_summary=True,
            )

        output = stdout.getvalue()
        self.assertNotIn("PPO训练配置", output)
        self.assertNotIn("batch 0001 | reward=", output)
        self.assertIn("2-2 [simple]", output)
        self.assertEqual(summary["best_batch_index"], 1)
        self.assertEqual(summary["best_batch_makespan"], 123.0)


if __name__ == "__main__":
    unittest.main()
