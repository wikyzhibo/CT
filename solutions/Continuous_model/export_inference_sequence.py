"""
并发模型推理序列导出工具。

将 Env_PN_Concurrent + 训练好的并发策略模型的推理轨迹导出为：
1) solutions/Continuous_model/action_series/<name>_<timestamp>.json
2) solutions/Td_petri/planB_sequence.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical, ProbabilisticActor

from solutions.PPO.network.models import MaskedPolicyHead
from solutions.Continuous_model.env_single import Env_PN_Single


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

    for candidate in _iter_single_candidate_state_dicts(raw_state_dict):
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
    training_phase: int,
    robot_capacity: int,
    device_mode: str = "single",
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    env = Env_PN_Single(
        seed=seed,
        training_phase=training_phase,
        robot_capacity=robot_capacity,
        device_mode=device_mode,
    )
    policy = _build_single_policy(env, model_path=model_path, device=device)

    td = env.reset()
    sequence: list[dict[str, Any]] = []
    finished = False

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            obs = td["observation"]
            td_model = TensorDict(
                {
                    "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0).to(device),
                    "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
                    "action_mask": td["action_mask"].unsqueeze(0).bool().to(device),
                },
                batch_size=[1],
            )

            with set_exploration_type(ExplorationType.RANDOM):
                td_out = policy(td_model)
            action_idx = int(td_out["action"].squeeze(0).item())

            step_td = td.clone()
            step_td["action"] = torch.tensor(action_idx, dtype=torch.int64)
            td_next = env.step(step_td)

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

    replay_env_overrides = {
        # 回放时固定为本次 episode 的实际工序时长，避免可视化重启后随机采样导致动作序列失配。
        "single_process_time_map": dict(getattr(env.net, "_episode_process_time_map", {})),
        "single_proc_time_rand_enabled": False,
        "single_robot_capacity": int(robot_capacity),
        "single_route_code": int(getattr(env.net, "single_route_code", 0)),
        "single_device_mode": str(device_mode),
        "training_phase": int(training_phase),
    }
    return sequence, finished, replay_env_overrides


def _rollout_single_sequence_with_retry(
    model_path: Path,
    max_steps: int,
    seed: int,
    training_phase: int,
    robot_capacity: int,
    max_retries: int = 10,
    device_mode: str = "single",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")

    last_sequence: list[dict[str, Any]] = []
    last_overrides: dict[str, Any] = {}
    for attempt in range(max_retries):
        attempt_seed = seed + attempt
        sequence, finished, replay_env_overrides = _rollout_single_sequence(
            model_path=model_path,
            max_steps=max_steps,
            seed=attempt_seed,
            training_phase=training_phase,
            robot_capacity=robot_capacity,
            device_mode=device_mode,
        )
        last_sequence = sequence
        last_overrides = replay_env_overrides
        if finished:
            if attempt > 0:
                print(f"[INFO] single 模式第 {attempt + 1} 次推理达到 finish。")
            return sequence, replay_env_overrides

    print(f"[WARN] single 模式重试 {max_retries} 次后仍未 finish，导出最后一次序列。")
    return last_sequence, last_overrides

def rollout_and_export(
    model_path: Path,
    max_steps: int,
    seed: int,
    out_name: str,
    training_phase: int,
    force_overwrite_planb: bool,
    device_mode: str,
    robot_capacity: int,
    single_retries: int,
) -> dict[str, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if device_mode == "single":
        sequence, replay_env_overrides = _rollout_single_sequence_with_retry(
            model_path=model_path,
            max_steps=max_steps,
            seed=seed,
            training_phase=training_phase,
            robot_capacity=robot_capacity,
            max_retries=single_retries,
            device_mode="single",
        )
        payload: dict[str, Any] | list[dict[str, Any]] = {
            "schema_version": 2,
            "device_mode": "single",
            "sequence": sequence,
            "replay_env_overrides": replay_env_overrides,
        }
    elif device_mode == "cascade":
        sequence, replay_env_overrides = _rollout_single_sequence_with_retry(
            model_path=model_path,
            max_steps=max_steps,
            seed=seed,
            training_phase=training_phase,
            robot_capacity=robot_capacity,
            max_retries=single_retries,
            device_mode="cascade",
        )
        # 级联模式继续保留 actions=[tm2, tm3] 结构兼容 Model B 回放，
        # 当前统一 pn_single 后端，第二通道固定填 WAIT。
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
            "schema_version": 2,
            "device_mode": "cascade",
            "sequence": sequence_dual,
            "replay_env_overrides": replay_env_overrides,
        }
    else:
        raise ValueError(f"不支持的 device_mode: {device_mode}")

    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    action_series_dir = Path(__file__).resolve().parent / "action_series"
    action_series_dir.mkdir(parents=True, exist_ok=True)
    action_series_path = action_series_dir / f"{out_name}_{timestamp}.json"

    planb_path = project_root / "solutions" / "Td_petri" / "planB_sequence.json"
    if planb_path.exists() and not force_overwrite_planb:
        raise FileExistsError(
            f"{planb_path} 已存在。若需覆盖，请传 --force-overwrite-planb"
        )
    planb_path.parent.mkdir(parents=True, exist_ok=True)

    with action_series_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with planb_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "action_series_path": action_series_path,
        "planb_path": planb_path,
    }


def main() -> None:
    default_model = Path(__file__).resolve().parent / "saved_models" / "CT_concurrent_phase2_best.pt"
    parser = argparse.ArgumentParser(description="导出推理动作序列（级联/单设备）")
    parser.add_argument("--model", type=Path, default=default_model, help="模型权重路径")
    parser.add_argument("--max-steps", type=int, default=500, help="最大推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out-name", type=str, default="concurrent_infer_seq", help="action_series 输出文件前缀")
    parser.add_argument("--phase", type=int, default=2, help="环境训练阶段(1/2)")
    parser.add_argument(
        "--device",
        type=str,
        default="cascade",
        choices=["cascade", "single"],
        help="设备模式：cascade=级联双机械手，single=单设备单动作",
    )
    parser.add_argument(
        "--device-mode",
        type=str,
        choices=["cascade", "single"],
        help="已弃用，等价于 --device",
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
    selected_device = args.device_mode if args.device_mode else args.device
    if out_name == "concurrent_infer_seq" and selected_device == "single":
        out_name = "single_infer_seq"

    out = rollout_and_export(
        model_path=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
        out_name=out_name,
        training_phase=args.phase,
        force_overwrite_planb=args.force_overwrite_planb,
        device_mode=selected_device,
        robot_capacity=args.robot_capacity,
        single_retries=args.single_retries,
    )

    print(f"[INFO] 已导出 action_series: {out['action_series_path']}")
    print(f"[INFO] 已导出 planB_sequence: {out['planb_path']}")


if __name__ == "__main__":
    main()
