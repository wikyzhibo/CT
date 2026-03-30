"""
单设备/级联推理序列导出工具。

在 `Env_PN_Single` 上 roll out 策略，将 `sequence`、`replay_env_overrides`、`reward_report`
等写入仓库根目录 `results/action_sequences/<out_name>.json`（默认文件名见 CLI `--out-name`）。

`rollout_and_export(..., gantt_png_path=..., gantt_title_suffix=...)` 可在导出 JSON 后
对本次 rollout 的 `env.net` 调用 `render_gantt` 写出 `gantt.png`（与 `pn_single.render_gantt` 行为一致）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical, ProbabilisticActor

from solutions.model.network import MaskedPolicyHead
from solutions.A.rl_env import Env_PN_Single
from results.paths import action_sequence_path, model_output_path, safe_name


def _to_step_state(td_next: Any) -> Any:
    if "next" in td_next.keys():
        return td_next["next"].clone()
    return td_next.clone()


def _to_terminated(td_next: Any) -> bool:
    if "next" in td_next.keys() and "terminated" in td_next["next"].keys():
        t = td_next["next", "terminated"]
    else:
        t = td_next["terminated"]
    return bool(t.item() if hasattr(t, "item") else t)


def _to_finish(td_next: Any) -> bool:
    if "next" in td_next.keys() and "finish" in td_next["next"].keys():
        f = td_next["next", "finish"]
    else:
        f = td_next.get("finish", False)
    return bool(f.item() if hasattr(f, "item") else f)


def _to_scrap(td_next: Any) -> bool:
    """从 step 返回值中提取 scrap 标志。"""
    for key in [("scrap",), ("next", "scrap")]:
        try:
            v = td_next
            for k in key:
                v = v[k]
            return bool(v.item() if hasattr(v, "item") else v)
        except (KeyError, TypeError):
            continue
    return False


def _to_time(td_state: Any) -> int:
    t = td_state["time"]
    return int(t.item() if hasattr(t, "item") else t)


def _decode_single_action_name(env: Env_PN_Single, action_idx: int) -> str:
    if hasattr(env, "get_action_name"):
        return str(env.get_action_name(int(action_idx)))
    if action_idx == env.net.T:
        return "WAIT"
    return env.net.id2t_name[action_idx]


def _iter_single_candidate_state_dicts(state_dict: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    candidates: list[dict[str, torch.Tensor]] = [state_dict]
    candidates.append(
        {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    )
    candidates.append(
        {k[len("module.0.module."):]: v for k, v in state_dict.items() if k.startswith("module.0.module.")}
    )
    candidates.append(
        {
            k[len("backbone.module.0.module."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.module.0.module.")
        }
    )
    return [c for c in candidates if c]


def _infer_single_model_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    first_linear = state_dict.get("net.0.weight")
    if first_linear is None:
        raise KeyError("权重缺少 net.0.weight，无法推断单设备策略网络结构")
    hidden = int(first_linear.shape[0])
    linear_key_pattern = re.compile(r"^net\.(\d+)\.weight$")
    linear_count = sum(1 for k in state_dict.keys() if linear_key_pattern.match(k))
    if linear_count <= 1:
        raise ValueError("单设备策略网络线性层数量异常，无法推断 n_layers")
    n_layers = linear_count - 1
    return hidden, n_layers


def _build_single_policy(
    env: Env_PN_Single,
    model_path: Path,
    device: torch.device,
) -> ProbabilisticActor:
    n_obs = int(env.observation_spec["observation"].shape[0])
    raw_state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    load_error: Exception | None = None

    candidates = _iter_single_candidate_state_dicts(raw_state_dict)

    for idx, candidate in enumerate(candidates):
        try:
            hidden, n_layers = _infer_single_model_shape(candidate)
            policy_backbone = MaskedPolicyHead(
                hidden=hidden,
                n_obs=n_obs,
                n_actions=env.n_actions,
                n_layers=n_layers,
            ).to(device)
            td_module = TensorDictModule(
                policy_backbone,
                in_keys=["observation_f"],
                out_keys=["logits"],
            )
            policy = ProbabilisticActor(
                module=td_module,
                in_keys={"logits": "logits", "mask": "action_mask"},
                out_keys=["action"],
                distribution_class=MaskedCategorical,
                return_log_prob=True,
            ).to(device)

            # 兼容直接保存 actor.state_dict() 的格式
            try:
                policy.load_state_dict(raw_state_dict)
                policy.eval()
                return policy
            except Exception:
                pass

            # 兼容保存 backbone/policy_module.state_dict() 的格式
            policy_backbone.load_state_dict(candidate)
            policy.eval()
            return policy
        except Exception as e:
            load_error = e
            continue

    raise RuntimeError(f"无法识别的单设备模型权重格式: {model_path}. 原始错误: {load_error}")


def _rollout_single_sequence(
    model_path: Path,
    max_steps: int,
    seed: int,
    robot_capacity: int,
    device_mode: str = "single",
    exploration: str = "mode",
) -> tuple[list[dict[str, Any]], bool, dict[str, Any], dict[str, Any], Env_PN_Single]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    env = Env_PN_Single(
        seed=seed,
        robot_capacity=robot_capacity,
        device_mode=device_mode,
        eval_mode=True,
    )
    policy = _build_single_policy(env, model_path=model_path, device=device)

    td = env.reset()
    sequence: list[dict[str, Any]] = []
    finished = False
    scrap_steps: list[int] = []
    release_steps: list[int] = []
    idle_steps: list[int] = []

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            try:
                obs = td["observation"]
                td_model = TensorDict(
                    {
                        "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0).to(device),
                        "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
                        "action_mask": td["action_mask"].unsqueeze(0).bool().to(device),
                    },
                    batch_size=[1],
                )

                explore_type = ExplorationType.MODE if exploration == "mode" else ExplorationType.RANDOM
                with set_exploration_type(explore_type):
                    td_out = policy(td_model)
                action_idx = int(td_out["action"].squeeze(0).item())

                step_td = td.clone()
                step_td["action"] = torch.tensor(action_idx, dtype=torch.int64)
                td_next = env.step(step_td)
            except Exception:
                raise

            if _to_scrap(td_next):
                scrap_steps.append(step)

            td_after = _to_step_state(td_next)
            current_time = _to_time(td_after)
            action_name = _decode_single_action_name(env, action_idx)
            terminated = _to_terminated(td_next)
            finished = _to_finish(td_next)

            sequence.append(
                {
                    "step": step,
                    "time": current_time,
                    "action": action_name,
                    "actions": [action_name],
                }
            )

            if terminated:
                break
            td = td_after

    reward_report = {
        "scrap_penalty": {"count": len(scrap_steps), "steps": scrap_steps},
        "release_penalty": {"count": len(release_steps), "steps": release_steps},
        "idle_timeout_penalty": {"count": len(idle_steps), "steps": idle_steps},
    }
    replay_env_overrides = {
        # 回放时固定为本次 episode 的实际工序时长，避免可视化重启后与导出时不一致。
        "process_time_map": dict(getattr(env.net, "_base_proc_time_map", {})),
        "robot_capacity": int(robot_capacity),
        "device_mode": str(device_mode),
        # 配置驱动路线信息：用于可视化重建与导出时一致的路径拓扑。
        "single_route_name": getattr(env.net, "single_route_name", None),
    }
    single_route_cfg = getattr(env.net, "single_route_config", None)
    if single_route_cfg is not None:
        replay_env_overrides["single_route_config"] = dict(single_route_cfg)
    return sequence, finished, replay_env_overrides, reward_report, env


def _rollout_single_sequence_with_retry(
    model_path: Path,
    max_steps: int,
    seed: int,
    robot_capacity: int,
    max_retries: int = 10,
    device_mode: str = "single",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool, Env_PN_Single]:
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")

    last_sequence: list[dict[str, Any]] = []
    last_overrides: dict[str, Any] = {}
    last_report: dict[str, Any] = {}
    last_finished = False
    last_env: Env_PN_Single | None = None
    for attempt in range(max_retries):
        attempt_seed = seed + attempt
        sequence, finished, replay_env_overrides, reward_report, env = _rollout_single_sequence(
            model_path=model_path,
            max_steps=max_steps,
            seed=attempt_seed,
            robot_capacity=robot_capacity,
            device_mode=device_mode,
            exploration="mode" if attempt == 0 else "random",
        )
        last_sequence = sequence
        last_overrides = replay_env_overrides
        last_report = reward_report
        last_finished = finished
        last_env = env
        if finished:
            if attempt > 0:
                print(f"[INFO] single 模式第 {attempt + 1} 次推理达到 finish。")
            return sequence, replay_env_overrides, reward_report, finished, env

    print(f"[WARN] single 模式重试 {max_retries} 次后仍未 finish，导出最后一次序列。")
    assert last_env is not None
    return last_sequence, last_overrides, last_report, last_finished, last_env


def rollout_and_export(
    model_path: Path,
    max_steps: int,
    seed: int,
    out_name: str,
    force_overwrite_planb: bool,
    device_mode: str,
    robot_capacity: int,
    single_retries: int,
    gantt_png_path: Path | None = None,
    gantt_title_suffix: str | None = None,
) -> dict[str, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if device_mode == "single":
        sequence, replay_env_overrides, reward_report, _finished, env = _rollout_single_sequence_with_retry(
            model_path=model_path,
            max_steps=max_steps,
            seed=seed,
            robot_capacity=robot_capacity,
            max_retries=single_retries,
            device_mode="single",
        )
        payload: dict[str, Any] | list[dict[str, Any]] = {
            "reward_report": reward_report,
            "schema_version": 2,
            "device_mode": "single",
            "sequence": sequence,
            "replay_env_overrides": replay_env_overrides,
        }
    elif device_mode == "cascade":
        sequence, replay_env_overrides, reward_report, _finished, env = _rollout_single_sequence_with_retry(
            model_path=model_path,
            max_steps=max_steps,
            seed=seed,
            robot_capacity=robot_capacity,
            max_retries=single_retries,
            device_mode="cascade",
        )
        sequence_dual = []
        for item in sequence:
            action_name = item.get("action")
            sequence_dual.append(
                {
                    "step": item.get("step"),
                    "time": item.get("time"),
                    "actions": [action_name, "WAIT"],
                }
            )
        payload = {
            "reward_report": reward_report,
            "schema_version": 2,
            "device_mode": "cascade",
            "sequence": sequence_dual,
            "replay_env_overrides": replay_env_overrides,
        }
    else:
        raise ValueError(f"不支持的 device_mode: {device_mode}")

    safe_out_name = safe_name(str(out_name), "export")
    action_series_path = action_sequence_path(safe_out_name)

    with action_series_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if gantt_png_path is not None:
        env.net.render_gantt(str(gantt_png_path), title_suffix=gantt_title_suffix)

    return {"action_series_path": action_series_path}


def main() -> None:
    default_model = model_output_path("CT_single_best.pt")
    parser = argparse.ArgumentParser(description="导出推理动作序列（级联/单设备）")
    parser.add_argument("--model", type=Path, default=default_model, help="模型权重路径")
    parser.add_argument("--max-steps", type=int, default=500, help="最大推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--out-name",
        type=str,
        default="tmp",
        help="results/action_sequences 下 JSON 文件名（不含 .json）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cascade",
        choices=["cascade", "single"],
        help="设备模式：cascade=级联双机械手，single=单设备单动作",
    )
    parser.add_argument(
        "--robot-capacity",
        type=int,
        default=1,
        choices=[1, 2],
        help="单设备机械手容量（仅 single 模式生效）",
    )
    parser.add_argument(
        "--force-overwrite-planb",
        action="store_true",
        help="允许覆盖 solutions/Td_petri/planB_sequence.json",
    )
    parser.add_argument(
        "--single-retries",
        type=int,
        default=10,
        help="single 模式未 finish 时的最大重试次数",
    )
    args = parser.parse_args()
    out_name = args.out_name
    selected_device = args.device
    if out_name == "concurrent_infer_seq" and selected_device == "single":
        out_name = "single_infer_seq"
    raw_model = args.model
    cand = Path(raw_model)
    if cand.is_file():
        model_path = cand.resolve()
    else:
        model_path = model_output_path(str(raw_model)).resolve()
    out = rollout_and_export(
        model_path=model_path,
        max_steps=args.max_steps,
        seed=args.seed,
        out_name=out_name,
        force_overwrite_planb=args.force_overwrite_planb,
        device_mode=selected_device,
        robot_capacity=args.robot_capacity,
        single_retries=args.single_retries,
    )

    print(f"[INFO] 已导出 action_series: {out['action_series_path']}")


if __name__ == "__main__":
    main()
