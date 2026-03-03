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

from solutions.PPO.enviroment import Env_PN_Concurrent
from solutions.PPO.network.models import DualHeadPolicyNet
from solutions.Continuous_model.train_concurrent import DualActionPolicyModule


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


def _to_time(td_state: Any) -> int:
    t = td_state["time"]
    return int(t.item() if hasattr(t, "item") else t)


def _decode_tm_action_name(env: Env_PN_Concurrent, tm: str, action_idx: int) -> str | None:
    if tm == "tm2":
        if action_idx == env.tm2_wait_action:
            return None
        t_idx = env.tm2_transition_indices[action_idx]
    else:
        if action_idx == env.tm3_wait_action:
            return None
        t_idx = env.tm3_transition_indices[action_idx]
    return env.net.id2t_name[t_idx]


def _infer_model_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    first_linear = state_dict.get("backbone.backbone.0.weight")
    if first_linear is None:
        raise KeyError("权重缺少 backbone.backbone.0.weight，无法推断网络结构")
    n_hidden = int(first_linear.shape[0])

    linear_key_pattern = re.compile(r"^backbone\.backbone\.(\d+)\.weight$")
    linear_count = sum(1 for k in state_dict.keys() if linear_key_pattern.match(k))
    if linear_count <= 0:
        raise ValueError("未检测到 backbone 线性层权重，无法推断 n_layers")

    # DualHeadPolicyNet 中 backbone 线性层数 = n_layers - 1
    n_layers = linear_count + 1
    return n_hidden, n_layers


def _build_policy(env: Env_PN_Concurrent, model_path: Path, device: torch.device) -> DualActionPolicyModule:
    n_obs = int(env.observation_spec["observation"].shape[0])
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    n_hidden, n_layers = _infer_model_shape(state_dict)

    backbone = DualHeadPolicyNet(
        n_obs=n_obs,
        n_hidden=n_hidden,
        n_actions_tm2=env.n_actions_tm2,
        n_actions_tm3=env.n_actions_tm3,
        n_layers=n_layers,
    ).to(device)
    policy = DualActionPolicyModule(backbone).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def rollout_and_export(
    model_path: Path,
    max_steps: int,
    seed: int,
    out_name: str,
    training_phase: int,
    force_overwrite_planb: bool,
) -> dict[str, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    device = torch.device("cpu")
    torch.manual_seed(seed)

    env = Env_PN_Concurrent(training_phase=training_phase, seed=seed)
    policy = _build_policy(env, model_path=model_path, device=device)

    td = env.reset()
    sequence: list[dict[str, Any]] = []

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            obs_f = td["observation"].unsqueeze(0).float().to(device)
            mask_tm2 = td["action_mask_tm2"].unsqueeze(0).to(device)
            mask_tm3 = td["action_mask_tm3"].unsqueeze(0).to(device)

            a_tm2, a_tm3, _, _, _, _ = policy(obs_f, mask_tm2, mask_tm3)
            a_tm2_idx = int(a_tm2.squeeze(0).item())
            a_tm3_idx = int(a_tm3.squeeze(0).item())

            step_td = td.clone()
            step_td["action_tm2"] = torch.tensor(a_tm2_idx, dtype=torch.int64)
            step_td["action_tm3"] = torch.tensor(a_tm3_idx, dtype=torch.int64)
            td_next = env.step(step_td)

            td_after = _to_step_state(td_next)
            current_time = _to_time(td_after)
            tm2_name = _decode_tm_action_name(env, "tm2", a_tm2_idx)
            tm3_name = _decode_tm_action_name(env, "tm3", a_tm3_idx)

            sequence.append(
                {
                    "step": step,
                    "time": current_time,
                    "actions": [tm2_name, tm3_name],
                }
            )

            if _to_terminated(td_next):
                break

            td = td_after

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

    payload = sequence
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
    parser = argparse.ArgumentParser(description="导出并发模型推理动作序列")
    parser.add_argument("--model", type=Path, default=default_model, help="并发模型权重路径")
    parser.add_argument("--max-steps", type=int, default=500, help="最大推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out-name", type=str, default="concurrent_infer_seq", help="action_series 输出文件前缀")
    parser.add_argument("--phase", type=int, default=2, help="环境训练阶段(1/2)")
    parser.add_argument(
        "--force-overwrite-planb",
        action="store_true",
        help="允许覆盖 solutions/Td_petri/planB_sequence.json",
    )
    args = parser.parse_args()

    out = rollout_and_export(
        model_path=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
        out_name=args.out_name,
        training_phase=args.phase,
        force_overwrite_planb=args.force_overwrite_planb,
    )

    print(f"[INFO] 已导出 action_series: {out['action_series_path']}")
    print(f"[INFO] 已导出 planB_sequence: {out['planb_path']}")


if __name__ == "__main__":
    main()
