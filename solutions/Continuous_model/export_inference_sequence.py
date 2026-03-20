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
from solutions.Continuous_model.pn_single import REASON_DESC

REWARD_DESC: dict[str, str] = {
    "total": "本步总奖励",
    "time_cost": "时间惩罚",
    "proc_reward": "加工奖励",
    "safe_reward": "安全奖励",
    "warn_penalty": "驻留警告惩罚",
    "penalty": "运输超时惩罚",
    "wafer_done_bonus": "单片完工奖励",
    "finish_bonus": "全部完工奖励",
    "scrap_penalty": "报废惩罚",
    "release_violation_penalty": "释放违规惩罚",
    "idle_timeout_penalty": "闲置超时惩罚",
}

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
) -> tuple[list[dict[str, Any]], bool, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
    action_enable_steps: list[dict[str, Any]] = []
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

            enable_info = getattr(env, "_last_action_enable_info", {})
            reward_detail = getattr(env, "_last_reward_detail", {})
            enabled_idxs = enable_info.get("enabled", [])
            enabled_names = [_decode_single_action_name(env, i) for i in enabled_idxs]
            step_enable = {
                "step": step,
                "time": current_time,
                "action": action_name,
                "enabled": enabled_idxs,
                "enabled_names": enabled_names,
                "disabled": enable_info.get("disabled", []),
                "reward_detail": reward_detail,
            }
            action_enable_steps.append(step_enable)

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
        # 回放时固定为本次 episode 的实际工序时长，避免可视化重启后随机采样导致动作序列失配。
        "process_time_map": dict(getattr(env.net, "_episode_proc_time_map", {})),
        "proc_rand_enabled": False,
        "robot_capacity": int(robot_capacity),
        "route_code": int(getattr(env.net, "single_route_code", 0)),
        "device_mode": str(device_mode),
        # 配置驱动路线信息：用于可视化重建与导出时一致的路径拓扑。
        "single_route_name": getattr(env.net, "single_route_name", None),
    }
    single_route_cfg = getattr(env.net, "single_route_config", None)
    if single_route_cfg is not None:
        replay_env_overrides["single_route_config"] = dict(single_route_cfg)
    return sequence, finished, replay_env_overrides, reward_report, action_enable_steps


def _rollout_single_sequence_with_retry(
    model_path: Path,
    max_steps: int,
    seed: int,
    robot_capacity: int,
    max_retries: int = 10,
    device_mode: str = "single",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")

    last_sequence: list[dict[str, Any]] = []
    last_overrides: dict[str, Any] = {}
    last_report: dict[str, Any] = {}
    last_enable_steps: list[dict[str, Any]] = []
    last_finished = False
    for attempt in range(max_retries):
        attempt_seed = seed + attempt
        sequence, finished, replay_env_overrides, reward_report, action_enable_steps = _rollout_single_sequence(
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
        last_enable_steps = action_enable_steps
        last_finished = finished
        if finished:
            if attempt > 0:
                print(f"[INFO] single 模式第 {attempt + 1} 次推理达到 finish。")
            return sequence, replay_env_overrides, reward_report, action_enable_steps, finished

    print(f"[WARN] single 模式重试 {max_retries} 次后仍未 finish，导出最后一次序列。")
    return last_sequence, last_overrides, last_report, last_enable_steps, last_finished

def _write_action_enable_reports(
    action_enable_steps: list[dict[str, Any]],
    reward_report: dict[str, Any],
    results_dir: Path,
    timestamp: str,
    device_mode: str,
    finished: bool = False,
) -> tuple[Path, Path]:
    """将动作使能信息写入 JSON 与 Markdown 双格式，返回两个文件路径。"""
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"eval_logs"
    json_path = results_dir / f"{base_name}.json"
    md_path = results_dir / f"{base_name}.md"

    payload = {
        "episode_id": base_name,
        "mode": "eval",
        "device_mode": device_mode,
        "finished": finished,
        "reward_report": reward_report,
        "steps": action_enable_steps,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Markdown 可读报告
    lines: list[str] = []
    lines.append(f"# 评估动作使能日志 — {timestamp}")
    lines.append("")
    lines.append("## 摘要")
    lines.append(f"- 总步数：{len(action_enable_steps)}")
    rp = reward_report
    lines.append(f"- 状态：{'finish' if finished else 'terminated'}")
    lines.append(f"- 惩罚：scrap={rp.get('scrap_penalty', {}).get('count', 0)}, release={rp.get('release_penalty', {}).get('count', 0)}, idle={rp.get('idle_timeout_penalty', {}).get('count', 0)}")
    lines.append("")
    lines.append("## 原因说明")
    lines.append("| 代码 | 含义 |")
    lines.append("|------|------|")
    for code, desc in REASON_DESC.items():
        lines.append(f"| {code} | {desc} |")
    lines.append("")
    lines.append("## 奖励说明")
    lines.append("| 代码 | 含义 |")
    lines.append("|------|------|")
    for code, desc in REWARD_DESC.items():
        lines.append(f"| {code} | {desc} |")
    lines.append("")
    lines.append("## 每步详情")
    for s in action_enable_steps:
        step_num = s.get("step", 0)
        t = s.get("time", 0)
        action = s.get("action", "")
        enabled_names = s.get("enabled_names", [])
        disabled = s.get("disabled", [])
        by_reason: dict[str, list[str]] = {}
        for d in disabled:
            r = d.get("reason", "unknown")
            n = d.get("name", str(d.get("action", "")))
            by_reason.setdefault(r, []).append(n)
        lines.append(f"### Step {step_num} (t={t}) — 执行: {action}")
        lines.append(f"**使能({len(enabled_names)})**: {', '.join(enabled_names) or '-'}")
        lines.append("**不使能(按原因)**")
        for r, names in sorted(by_reason.items()):
            desc = REASON_DESC.get(r, r)
            lines.append(f"- {desc}: {', '.join(names)}")
        reward_detail = dict(s.get("reward_detail", {}))
        if reward_detail:
            lines.append("**详细奖励**")
            total = reward_detail.get("total")
            for k in sorted(reward_detail.keys()):
                v = reward_detail[k]
                if k == "scrap_info":
                    lines.append(f"- scrap_info: {v}")
                elif isinstance(v, (int, float)):
                    if v == 0:
                        continue
                    lbl = REWARD_DESC.get(k, k)
                    lines.append(f"- {lbl}: {v:.4f}" if isinstance(v, float) else f"- {lbl}: {v:.1f}")
        lines.append("")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def rollout_and_export(
    model_path: Path,
    max_steps: int,
    seed: int,
    out_name: str,
    force_overwrite_planb: bool,
    device_mode: str,
    robot_capacity: int,
    single_retries: int,
    results_dir: Path | None = None,
) -> dict[str, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if device_mode == "single":
        sequence, replay_env_overrides, reward_report, action_enable_steps, finished = _rollout_single_sequence_with_retry(
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
        sequence, replay_env_overrides, reward_report, action_enable_steps, finished = _rollout_single_sequence_with_retry(
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

    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    action_series_dir = Path(__file__).resolve().parents[2] / "seq"
    action_series_path = action_series_dir / "tmp.json"

    with action_series_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out: dict[str, Path] = {"action_series_path": action_series_path}

    # 写入 results/ 的动作使能日志（JSON + Markdown）
    results_path = results_dir or (project_root / "results")
    json_log, md_log = _write_action_enable_reports(
        action_enable_steps=action_enable_steps,
        reward_report=reward_report,
        results_dir=results_path,
        timestamp=timestamp,
        device_mode=device_mode,
        finished=finished,
    )
    out["action_enable_json"] = json_log
    out["action_enable_md"] = md_log

    return out


def main() -> None:
    default_model = Path(__file__).resolve().parent / "saved_models" / "CT_concurrent_phase2_best.pt"
    parser = argparse.ArgumentParser(description="导出推理动作序列（级联/单设备）")
    parser.add_argument("--model", type=Path, default=default_model, help="模型权重路径")
    parser.add_argument("--max-steps", type=int, default=500, help="最大推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out-name", type=str, default="concurrent_infer_seq", help="action_series 输出文件前缀")
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="动作使能日志输出目录，默认 results/",
    )
    args = parser.parse_args()
    out_name = args.out_name
    selected_device = args.device
    if out_name == "concurrent_infer_seq" and selected_device == "single":
        out_name = "single_infer_seq"
    model_path = Path(__file__).resolve().parents[2] / "models" / args.model
    results_dir = args.results_dir or Path(__file__).resolve().parents[2] / "results"
    out = rollout_and_export(
        model_path=model_path,
        max_steps=args.max_steps,
        seed=args.seed,
        out_name=out_name,
        force_overwrite_planb=args.force_overwrite_planb,
        device_mode=selected_device,
        robot_capacity=args.robot_capacity,
        single_retries=args.single_retries,
        results_dir=results_dir,
    )

    print(f"[INFO] 已导出 action_series: {out['action_series_path']}")
    print(f"[INFO] 已导出动作使能日志: {out['action_enable_json']}, {out['action_enable_md']}")


if __name__ == "__main__":
    main()
