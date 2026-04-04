from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from config.cluster_tool.env_config import PetriEnvConfig
from config.training.training_config import PPOTrainingConfig
from results.paths import gantt_output_path, training_log_output_path
from solutions.A.eval.export_inference_sequence import rollout_and_export
from solutions.A.ppo_trainer import train_single


# 每条路线训练/评估使用的晶圆数与训练档位。
# 注1：1-6 路线存在死锁
# 注2：3-* 测试路径只取一部分
# 注3：4-* 4-1未跑通、4-5~4-7 路线仍未建模、4-10~4-12泛化失败、4-15~4-16难度过大未建模
# 注4：4-* 双腔路线（4-9~4-14）未达到最优
# 注5：所有路径clean_enabled=False，降低训练难度
# 注6：训练档位根据路径难度预设，主要影响训练时长和最终效果，low(~4s)，medium(~14s)，high(~26s)，实际效果会有波动
ROUTE_PLAN: dict[str, dict[str, int | str]] = {
    "1-1": {"train": 8, "eval": 75, "profile": "low"},  # 1-* 单腔线路
    "1-2": {"train": 10, "eval": 75, "profile": "medium"},
    "1-3": {"train": 10, "eval": 75, "profile": "medium"},
    #"1-4": {"train": 12, "eval": 75, "profile": "high"},
    "1-5": {"train": 10, "eval": 75, "profile": "medium"},
    "2-1": {"train": 8, "eval": 75, "profile": "low"},  # 2-* 集创赛 ABCD 路径
    "2-2": {"train": 6, "eval": 75, "profile": "low"},
    "2-3": {"train": 6, "eval": 75, "profile": "medium"},
    "2-4": {"train": 6, "eval": 75, "profile": "low"},
    #"3-1": {"train": 9, "eval": 75, "profile": "low"},  # 3-* 单腔测试路径
    #"3-2": {"train": 6, "eval": 75, "profile": "low"},
    #"3-3": {"train": 6, "eval": 15, "profile": "medium"},
    #"4-2": {"train": 8, "eval": 75, "profile": "low"},  # 4-* 双腔路径
    #"4-3": {"train": 8, "eval": 75, "profile": "low"},
    #"4-8": {"train": 10, "eval": 75, "profile": "medium"},
    #"4-9": {"train": 12, "eval": 75, "profile": "medium"},
    #"4-10": {"train": 12, "eval": 75, "profile": "medium"},
    #"4-11": {"train": 12, "eval": 75, "profile": "medium"},
    #"4-12": {"train": 12, "eval": 75, "profile": "high"},
    #"4-13": {"train": 10, "eval": 75, "profile": "medium"},
    #"4-14": {"train": 10, "eval": 75, "profile": "high"},
}







TRAINING_PROFILE_FILES: dict[str, str] = {
    "low": "low.yaml",
    "medium": "medium.yaml",
    "high": "high.yaml",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cascade_config_path() -> Path:
    return _repo_root() / "config" / "cluster_tool" / "cascade.yaml"


def _training_config_dir() -> Path:
    return _repo_root() / "config" / "training"


def _load_route_config() -> dict[str, Any]:
    base_config = PetriEnvConfig.load(_cascade_config_path())
    route_config = dict(base_config.single_route_config or {})
    if not route_config:
        raise ValueError("cascade.yaml 未提供 single_route_config，无法执行批量路线训练")
    return route_config


def _resolve_training_profile_path(profile_name: str) -> Path:
    key = str(profile_name).strip().lower()
    if key not in TRAINING_PROFILE_FILES:
        allowed = ", ".join(sorted(TRAINING_PROFILE_FILES.keys()))
        raise ValueError(f"不支持的训练档位: {profile_name}。只接受: {allowed}")
    return _training_config_dir() / TRAINING_PROFILE_FILES[key]


def _build_env_overrides(route_config: dict[str, Any], route_name: str, n_wafer: int) -> dict[str, Any]:
    return {
        "n_wafer": int(n_wafer),
        "single_route_config": dict(route_config),
        "single_route_name": str(route_name),
    }


def _normalize_route_plan(
    route_plan: dict[str, dict[str, int | str]],
    route_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not route_plan:
        raise ValueError("ROUTE_PLAN 为空，请先填写每条路线的 train / eval / profile")

    route_entries = dict(route_config.get("routes") or {})
    normalized: list[dict[str, Any]] = []
    for route_name in route_plan.keys():
        if route_name not in route_entries:
            raise ValueError(f"single_route_config.routes 中不存在路线: {route_name}")
        route_spec = dict(route_plan[route_name] or {})
        if "train" not in route_spec or "eval" not in route_spec or "profile" not in route_spec:
            raise ValueError(f"路线 {route_name} 必须同时提供 train / eval / profile")
        train_n = int(route_spec["train"])
        eval_n = int(route_spec["eval"])
        if train_n <= 0 or eval_n <= 0:
            raise ValueError(f"路线 {route_name} 的 train/eval 晶圆数必须 > 0")
        profile_name = str(route_spec["profile"]).strip().lower()
        normalized.append(
            {
                "route_name": route_name,
                "train_n_wafer": train_n,
                "eval_n_wafer": eval_n,
                "training_profile": profile_name,
                "training_profile_path": _resolve_training_profile_path(profile_name),
            }
        )
    return normalized


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{float(seconds):.1f}s"


def _format_makespan(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        ("route_name", "Route"),
        ("training_profile", "Profile"),
        ("train_n_wafer", "TrainW"),
        ("train_best_makespan", "TrainM"),
        ("eval_n_wafer", "EvalW"),
        ("eval_makespan", "EvalM"),
        ("training_time_seconds", "TrainTime"),
    ]
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_rows.append(
            [
                str(row["route_name"]),
                str(row["training_profile"]),
                str(row["train_n_wafer"]),
                _format_makespan(row.get("train_best_makespan")),
                str(row["eval_n_wafer"]),
                _format_makespan(row.get("eval_makespan")),
                _format_seconds(row.get("training_time_seconds")),
            ]
        )

    widths: list[int] = []
    for idx, (_, title) in enumerate(headers):
        max_value_len = max([len(title)] + [len(values[idx]) for values in rendered_rows])
        widths.append(max_value_len)

    lines = []
    header_line = " | ".join(title.ljust(widths[idx]) for idx, (_, title) in enumerate(headers))
    sep_line = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for values in rendered_rows:
        lines.append(" | ".join(values[idx].ljust(widths[idx]) for idx in range(len(values))))
    return "\n".join(lines)


def run_all_routes(
    *,
    route_plan: dict[str, dict[str, int | str]] | None = None,
    retry: int = 10,
    compute_device: str | None = None,
    rollout_n_envs: int = 1,
) -> tuple[list[dict[str, Any]], Path]:
    route_config = _load_route_config()
    normalized_routes = _normalize_route_plan(
        route_plan if route_plan is not None else ROUTE_PLAN,
        route_config,
    )

    summary_rows: list[dict[str, Any]] = []
    for route in normalized_routes:
        route_name = str(route["route_name"])
        train_n_wafer = int(route["train_n_wafer"])
        eval_n_wafer = int(route["eval_n_wafer"])
        profile_name = str(route["training_profile"])
        config_path = Path(route["training_profile_path"])

        cfg = PPOTrainingConfig.load(config_path)
        if compute_device is not None:
            cfg.device = str(compute_device).strip()

        train_run_name = f"validate_{route_name}_trainW{train_n_wafer}"
        train_env_overrides = _build_env_overrides(route_config, route_name, train_n_wafer)
        _log, _policy, train_summary = train_single(
            config=cfg,
            checkpoint_path=None,
            rollout_n_envs=rollout_n_envs,
            artifact_dir=train_run_name,
            concurrent=True,
            env_overrides=train_env_overrides,
            batch_progress_only=True,
            progress_label=f"{route_name} [{profile_name}]",
            return_summary=True,
        )

        best_model_path_raw = train_summary.get("best_model_path")
        best_model_path = Path(best_model_path_raw) if best_model_path_raw else None

        eval_summary: dict[str, Any] = {
            "action_series_path": None,
            "makespan": None,
            "n_wafer": eval_n_wafer,
            "finished": False,
            "single_route_name": route_name,
        }
        if best_model_path is not None and best_model_path.is_file():
            eval_env_overrides = _build_env_overrides(route_config, route_name, eval_n_wafer)
            eval_run_name = f"validate_{route_name}_evalW{eval_n_wafer}"
            eval_summary = rollout_and_export(
                model_path=best_model_path,
                seed=int(cfg.seed),
                out_name=eval_run_name,
                concurrent=True,
                retry=int(retry),
                gantt_png_path=gantt_output_path(f"{eval_run_name}_gantt.png"),
                gantt_title_suffix=f"路径 {route_name}",
                env_overrides=eval_env_overrides,
                verbose=False,
            )

        summary_rows.append(
            {
                "route_name": route_name,
                "training_profile": profile_name,
                "train_n_wafer": train_n_wafer,
                "train_best_makespan": train_summary.get("best_batch_makespan"),
                "best_batch_index": train_summary.get("best_batch_index"),
                "best_model_path": str(best_model_path) if best_model_path is not None else None,
                "training_time_seconds": train_summary.get("training_time_seconds"),
                "eval_n_wafer": eval_n_wafer,
                "eval_makespan": eval_summary.get("makespan"),
                "eval_finished": bool(eval_summary.get("finished", False)),
                "eval_action_sequence_path": (
                    str(eval_summary["action_series_path"])
                    if eval_summary.get("action_series_path") is not None
                    else None
                ),
            }
        )

    summary_path = training_log_output_path("validate_all_routes_summary.json")
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_rows, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="按路线批量执行 concurrent 训练与评估")
    parser.add_argument("--retry", type=int, default=10, help="评估 rollout 最大重试次数")
    parser.add_argument("--compute-device", type=str, default=None, help="覆盖 YAML 中的 device，例如 cpu / cuda")
    parser.add_argument("--rollout-n-envs", type=int, default=20, help="训练 rollout 并行环境数")
    args = parser.parse_args()

    summary_rows, summary_path = run_all_routes(
        retry=int(args.retry),
        compute_device=args.compute_device,
        rollout_n_envs=int(args.rollout_n_envs),
    )
    print(_format_summary_table(summary_rows), flush=True)
    print(f"\nsummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
