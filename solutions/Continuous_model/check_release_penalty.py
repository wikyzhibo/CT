"""
脚本式验证：检查训练收集流程中的二次释放惩罚回填是否正确。

流程对齐 collect_rollout 的两阶段逻辑：
1) 第一阶段关闭在线 release 惩罚执行固定动作序列。
2) 第二阶段调用 blame_release_violations 追责并把惩罚回填到对应 step reward。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from solutions.PPO.enviroment import Env_PN_Concurrent


def _load_sequence(seq) -> list[dict[str, Any]]:
    path = (Path(__file__).resolve().parent/ "action_series"/ seq)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("sequence JSON 顶层必须是 list")
    return data


def _extract_step_result(td_next: Any) -> tuple[float, bool, int]:
    if "next" in td_next.keys():
        reward_t = td_next["next", "reward"]
        terminated_t = td_next["next", "terminated"]
        time_t = td_next["next", "time"]
    else:
        reward_t = td_next["reward"]
        terminated_t = td_next["terminated"]
        time_t = td_next["time"]
    reward = float(reward_t.item() if hasattr(reward_t, "item") else reward_t)
    terminated = bool(terminated_t.item() if hasattr(terminated_t, "item") else terminated_t)
    time_val = int(time_t.item() if hasattr(time_t, "item") else time_t)
    return reward, terminated, time_val


def _to_next_state(td_next: Any) -> Any:
    if "next" in td_next.keys():
        return td_next["next"].clone()
    return td_next.clone()


def _tm2_action_index(env: Env_PN_Concurrent, name: str | None) -> int:
    if name is None or str(name).lower() == "wait":
        return env.tm2_wait_action
    for i, t_idx in enumerate(env.tm2_transition_indices):
        if env.net.id2t_name[t_idx] == name:
            return i
    raise ValueError(f"TM2 动作名无效或不受 TM2 控制: {name}")

def _tm3_action_index(env: Env_PN_Concurrent, name: str | None) -> int:
    if name is None or str(name).lower() == "wait":
        return env.tm3_wait_action
    for i, t_idx in enumerate(env.tm3_transition_indices):
        if env.net.id2t_name[t_idx] == name:
            return i
    raise ValueError(f"TM3 动作名无效或不受 TM3 控制: {name}")

def run_sequence(sequence_path: Path, results_dir: Path) -> Path:
    seq = _load_sequence(sequence_path)
    env = Env_PN_Concurrent(training_phase=2)
    td = env.reset()

    records: list[dict[str, Any]] = []
    fire_log_ranges: list[tuple[int, int]] = []
    target_step_indices: list[int] = []

    env.net.no_release_penalty = True
    try:
        for idx, item in enumerate(seq):
            actions = item.get("actions", [None, None])
            tm2_name = actions[0] if len(actions) > 0 else None
            tm3_name = actions[1] if len(actions) > 0 else None

            a_tm2 = _tm2_action_index(env, tm2_name)
            a_tm3 = _tm3_action_index(env, tm3_name)

            # 检查动作合法性
            mask_tm2 = td["action_mask_tm2"]
            mask_tm3 = td["action_mask_tm3"]
            if not bool(mask_tm2[a_tm2].item()):
                raise RuntimeError(f"第 {idx+1} 步 TM2 动作不可用: {tm2_name}")
            if not bool(mask_tm3[a_tm3].item()):
                raise RuntimeError(f"第 {idx+1} 步 TM3 WAIT 不可用")

            fire_start = len(env.net.fire_log)

            step_td = td.clone()
            step_td["action_tm2"] = torch.tensor(a_tm2, dtype=torch.int64)
            step_td["action_tm3"] = torch.tensor(a_tm3, dtype=torch.int64)
            td_next = env.step(step_td)

            fire_end = len(env.net.fire_log)
            fire_log_ranges.append((fire_start, fire_end))

            reward_before, terminated, sim_time = _extract_step_result(td_next)
            record = {
                "step": idx + 1,
                "time": sim_time,
                "tm2_action": tm2_name if tm2_name is not None else "wait",
                "tm3_action": "wait",
                "reward_before_second_pass": reward_before,
                "reward_after_second_pass": reward_before,
                "second_pass_penalties": [],
                "terminated": terminated,
            }
            records.append(record)

            if tm2_name == "u_LP2_s1":
                target_step_indices.append(idx)

            # 返回下一步状态
            td = _to_next_state(td_next)
    finally:
        env.net.no_release_penalty = False
        for log in env.net.fire_log:
            print(log)


            # 序列结束后做第二阶段追责（不依赖 episode 结束）
    blame = env.net.blame_release_violations()
    mapped_blame: list[dict[str, Any]] = []
    for fire_idx, penalty in blame.items():
        for step_idx, (start, end) in enumerate(fire_log_ranges):
            if start <= fire_idx < end:
                records[step_idx]["reward_after_second_pass"] -= float(penalty)
                records[step_idx]["second_pass_penalties"].append(
                    {"fire_log_index": int(fire_idx), "penalty": float(penalty)}
                )
                mapped_blame.append(
                    {
                        "fire_log_index": int(fire_idx),
                        "penalty": float(penalty),
                        "mapped_step_index": step_idx,
                        "mapped_tm2_action": records[step_idx]["tm2_action"],
                    }
                )
                break

    total_before = float(sum(r["reward_before_second_pass"] for r in records))
    total_after = float(sum(r["reward_after_second_pass"] for r in records))

    # 检查“最后一个 u_LP2_s1”是否被二次惩罚命中
    last_lp2_u_step = target_step_indices[-1] if target_step_indices else None
    last_lp2_u_penalty = 0.0
    if last_lp2_u_step is not None:
        last_lp2_u_penalty = float(
            sum(p["penalty"] for p in records[last_lp2_u_step]["second_pass_penalties"])
        )

    payload = {
        "meta": {
            "script": "solutions/Continuous_model/check_release_penalty.py",
            "sequence_path": str(sequence_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "steps": len(records),
            "tm3_policy": "always_wait",
        },
        "summary": {
            "total_reward_before_second_pass": total_before,
            "total_reward_after_second_pass": total_after,
            "total_second_pass_penalty": total_before - total_after,
            "blame_count": len(blame),
            "last_u_LP2_s1_step_index": last_lp2_u_step,
            "last_u_LP2_s1_second_pass_penalty": last_lp2_u_penalty,
            "last_u_LP2_s1_penalized": last_lp2_u_penalty > 0.0,
        },
        "blame_raw": {str(k): float(v) for k, v in blame.items()},
        "blame_mapped": mapped_blame,
        "step_records": records,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"release_penalty_second_pass.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 结果已输出: {out_path}")
    print(
        "[INFO] 最后一个 u_LP2_s1 二次惩罚 = "
        f"{payload['summary']['last_u_LP2_s1_second_pass_penalty']}"
    )
    return out_path


def main() -> None:
    default_sequence = "wrong_seq.json"
    default_results = Path(__file__).resolve().parents[2] / "results"

    parser = argparse.ArgumentParser(description="验证二次释放惩罚回填的脚本")
    parser.add_argument("--sequence", type=Path, default=default_sequence, help="动作序列 JSON 路径")
    parser.add_argument("--results-dir", type=Path, default=default_results, help="输出目录")
    args = parser.parse_args()

    run_sequence(sequence_path=args.sequence, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
