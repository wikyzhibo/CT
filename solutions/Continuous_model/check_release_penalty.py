"""
脚本式验证：检查单设备(pn_single)流程中的二次释放惩罚回填是否正确。

流程：
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

from solutions.Continuous_model.env_single import Env_PN_Single


def _load_sequence(seq_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    path = Path(seq_path)
    if not path.is_absolute():
        direct = path
        in_action_series = Path(__file__).resolve().parent / "action_series" / path
        path = direct if direct.exists() else in_action_series

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}, path
    if isinstance(data, dict):
        seq = data.get("sequence", [])
        if not isinstance(seq, list):
            raise ValueError("sequence JSON 中 sequence 必须是 list")
        overrides = data.get("replay_env_overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}
        return seq, overrides, path
    raise ValueError("sequence JSON 顶层必须是 list 或 dict(含 sequence)")


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


def _normalize_action_name(raw_name: str | None) -> str:
    if raw_name is None:
        return "WAIT_5s"
    name = str(raw_name).strip()
    if name.upper() == "WAIT":
        return "WAIT_5s"
    return name


def _single_action_index(env: Env_PN_Single, raw_name: str | None) -> int:
    action_name = _normalize_action_name(raw_name)
    for i in range(env.n_actions):
        if env.get_action_name(i) == action_name:
            return i
    raise ValueError(f"单设备动作名无效: {action_name}")


def _build_release_violation_reasons(
    env: Env_PN_Single,
    fire_indices: list[int],
) -> dict[int, dict[str, Any]]:
    net = env.net
    reasons: dict[int, dict[str, Any]] = {}
    if not fire_indices:
        return reasons

    proc_times = {p.name: p.processing_time for p in net.marks}
    capacities = {p.name: p.capacity for p in net.marks}

    def build_intervals(chamber_name: str) -> list[tuple[int, int, int]]:
        intervals: list[tuple[int, int, int]] = []
        for (enter, leave, wid) in net._chamber_timeline.get(chamber_name, []):
            l = leave if leave is not None else enter + proc_times.get(chamber_name, 0)
            intervals.append((int(enter), int(l), int(wid)))
        intervals.sort(key=lambda x: x[0])
        return intervals

    def build_cleaning_intervals(chamber_name: str) -> list[tuple[int, int, int]]:
        if not getattr(net, "cleaning_enabled", False) or chamber_name not in getattr(net, "cleaning_targets", set()):
            return []
        out: list[tuple[int, int, int]] = []
        for ev in net.fire_log:
            if ev.get("event_type") == "cleaning_start" and ev.get("chamber") == chamber_name:
                t0 = int(ev.get("time", 0))
                dur = int(ev.get("duration", getattr(net, "cleaning_duration", 150)))
                out.append((t0, t0 + dur, -999))
        return out

    intervals_by_station: dict[str, list[tuple[int, int, int]]] = {}
    capacity_by_station: dict[str, int] = {}
    proc_time_by_station: dict[str, int] = {}
    for station, chambers in net._release_station_aliases.items():
        merged: list[tuple[int, int, int]] = []
        for chamber_name in chambers:
            merged.extend(build_intervals(chamber_name))
            merged.extend(build_cleaning_intervals(chamber_name))
        merged.sort(key=lambda x: x[0])
        intervals_by_station[station] = merged
        capacity_by_station[station] = int(sum(capacities.get(name, 0) for name in chambers))
        proc_time_by_station[station] = int(max((proc_times.get(name, 0) for name in chambers), default=0))

    edge_transfer = int(net.T_transport + net.T_load)
    chain_map: dict[str, list[str]] = dict(net._release_chain_by_u)

    for fire_idx in fire_indices:
        if not (0 <= int(fire_idx) < len(net.fire_log)):
            continue
        log = net.fire_log[int(fire_idx)]
        t_name = str(log.get("t_name", ""))
        wid = int(log.get("token_id", -1))
        chain = chain_map.get(t_name, [])
        arrival = int(log.get("t1", 0)) + edge_transfer

        reason_item = {
            "reason": "unknown",
            "violated_station": None,
            "arrival": arrival,
            "capacity": None,
            "occupied_prior": None,
        }
        for idx, station in enumerate(chain):
            intervals = intervals_by_station.get(station, [])
            cap = int(capacity_by_station.get(station, 1))
            occupied_prior = int(
                sum(1 for (e, l, wid0) in intervals if e <= arrival < l and wid0 < wid)
            )
            if occupied_prior + 1 > cap:
                reason_item = {
                    "reason": "downstream_capacity_exceeded",
                    "violated_station": station,
                    "arrival": int(arrival),
                    "capacity": cap,
                    "occupied_prior": occupied_prior,
                }
                break
            if idx < len(chain) - 1:
                arrival = arrival + int(proc_time_by_station.get(station, 0)) + edge_transfer
        reasons[int(fire_idx)] = reason_item
    return reasons


def run_sequence(sequence_path: Path, results_dir: Path) -> Path:
    seq, replay_env_overrides, resolved_sequence_path = _load_sequence(sequence_path)
    env = Env_PN_Single(
        seed=0,
        device_mode=str(replay_env_overrides["device_mode"]),
        robot_capacity=int(replay_env_overrides["robot_capacity"]),
        route_code=int(replay_env_overrides["route_code"]),
        process_time_map=replay_env_overrides["process_time_map"],
        detailed_reward=True,
    )
    td = env.reset()

    records: list[dict[str, Any]] = []
    fire_log_ranges: list[tuple[int, int]] = []
    tracked_u_llc_step_indices: list[int] = []

    for idx, item in enumerate(seq):
        actions = item.get("actions", [])
        action_name = actions[0] if len(actions) > 0 else item.get("action")
        action_idx = _single_action_index(env, action_name)

        mask = td["action_mask"]
        if not bool(mask[action_idx].item()):
            raise RuntimeError(f"第 {idx+1} 步动作不可用: {_normalize_action_name(action_name)}")

        fire_start = len(env.net.fire_log)

        step_td = td.clone()
        step_td["action"] = torch.tensor(action_idx, dtype=torch.int64)
        td_next = env.step(step_td)

        fire_end = len(env.net.fire_log)
        fire_log_ranges.append((fire_start, fire_end))

        reward_before, terminated, sim_time = _extract_step_result(td_next)
        normalized_action = _normalize_action_name(action_name)
        records.append(
            {
                "step": idx + 1,
                "time": sim_time,
                "action": normalized_action,
                "reward_before_second_pass": reward_before,
                "reward_after_second_pass": reward_before,
                "second_pass_penalties": [],
                "terminated": terminated,
            }
        )

        if normalized_action == "u_LLC":
            tracked_u_llc_step_indices.append(idx)

        td = _to_next_state(td_next)

    blame = env.net.blame_release_violations()
    fire_indices = sorted(int(k) for k in blame.keys())
    reasons_by_fire_idx = _build_release_violation_reasons(env, fire_indices)
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
                        "mapped_action": records[step_idx]["action"],
                    }
                )
                break

    total_before = float(sum(r["reward_before_second_pass"] for r in records))
    total_after = float(sum(r["reward_after_second_pass"] for r in records))

    tracked_u_llc_penalties: list[dict[str, Any]] = []
    for step_idx in tracked_u_llc_step_indices:
        tracked_u_llc_penalties.append(
            {
                "step_index": step_idx,
                "step": step_idx + 1,
                "penalty": float(sum(p["penalty"] for p in records[step_idx]["second_pass_penalties"])),
            }
        )

    penalized_actions_report: list[dict[str, Any]] = []
    penalized_action_lines: list[dict[str, Any]] = []
    for item in mapped_blame:
        fire_idx = int(item["fire_log_index"])
        step_index = int(item["mapped_step_index"])
        step_num = step_index + 1
        reason = reasons_by_fire_idx.get(fire_idx, {})
        reason_text = str(reason.get("reason", "unknown"))
        penalized_action_lines.append(
            {
                "action_name": str(item["mapped_action"]),
                "step": step_num,
                "reason": reason_text,
            }
        )
        penalized_actions_report.append(
            {
                "action_name": str(item["mapped_action"]),
                "step": step_num,
                "fire_log_index": fire_idx,
                "penalty": float(item["penalty"]),
                "reason": reason_text,
                "violated_station": reason.get("violated_station"),
                "arrival": reason.get("arrival"),
                "where_penalized": {
                    "step_record_index": step_index,
                    "field": "step_records[*].reward_after_second_pass",
                },
            }
        )

    payload = {
        "meta": {
            "script": "solutions/Continuous_model/check_release_penalty.py",
            "sequence_path": str(resolved_sequence_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "steps": len(records),
            "env": "Env_PN_Single",
        },
        "summary": {
            "total_reward_before_second_pass": total_before,
            "total_reward_after_second_pass": total_after,
            "total_second_pass_penalty": total_before - total_after,
            "blame_count": len(blame),
            "tracked_u_llc_count": len(tracked_u_llc_step_indices),
        },
        "blame_raw": {str(k): float(v) for k, v in blame.items()},
        "blame_mapped": mapped_blame,
        "penalized_action_lines": penalized_action_lines,
        "penalized_actions_report": penalized_actions_report,
        "tracked_u_llc_penalties": tracked_u_llc_penalties,
        "step_records": records,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "release_penalty_second_pass.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 结果已输出: {out_path}")
    print("[INFO] 追责逐行摘要(action_name + step + reason):")
    for x in penalized_action_lines:
        print(x)

    return out_path


def main() -> None:
    default_results = Path(__file__).resolve().parents[2] / "results"

    parser = argparse.ArgumentParser(description="验证单设备二次释放惩罚回填的脚本")
    parser.add_argument("--sequence", type=Path, help="动作序列 JSON 路径")
    parser.add_argument("--results-dir", type=Path, default=default_results, help="输出目录")
    args = parser.parse_args()
    sequence_path = Path(__file__).resolve().parents[2] / "seq" / args.sequence

    run_sequence(sequence_path=sequence_path, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
