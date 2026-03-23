from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_single_replay_payload(full_transition_records: list[dict[str, Any]]) -> dict[str, Any]:
    sequence: list[dict[str, Any]] = []
    step = 1
    prev_fire_time: int | None = None

    for item in full_transition_records:
        action_name = str(item["transition"])
        fire_time = int(item["fire_time"])

        if prev_fire_time is None:
            sequence.append(
                {
                    "step": step,
                    "time": fire_time,
                    "actions": [action_name,"WAIT"],
                }
            )
            prev_fire_time = fire_time
            step += 1
            continue

        delta = fire_time - prev_fire_time
        wait_count = (delta - 1) // 5 if delta > 5 else 0
        wait_time = prev_fire_time
        for _ in range(wait_count):
            wait_time += 5
            sequence.append(
                {
                    "step": step,
                    "time": wait_time,
                    "actions": ["WAIT_5s","WAIT"],
                }
            )
            step += 1

        sequence.append(
            {
                "step": step,
                "time": fire_time,
                "actions": [action_name,"WAIT"],
            }
        )
        prev_fire_time = fire_time
        step += 1

    return {
        "schema_version": 2,
        "device_mode": "cascade",
        "sequence": sequence,
    }


def export_single_replay_payload(
    full_transition_records: list[dict[str, Any]],
    out_name: str = "pdr_sequence",
) -> Path:
    payload = build_single_replay_payload(full_transition_records)
    seq_dir = Path(__file__).resolve().parents[2] / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    out_path = seq_dir / f"{out_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _load_records_from_json(records_json_path: Path) -> list[dict[str, Any]]:
    with records_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return list(raw.get("full_transition_records", []))
    return list(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PDR full_transition_records 转为 UI 可回放序列")
    parser.add_argument("--records-json", type=Path, required=True, help="输入 records JSON 文件路径")
    parser.add_argument("--out-name", type=str, default="pdr_sequence", help="输出到 seq/<out_name>.json")
    args = parser.parse_args()

    records = _load_records_from_json(args.records_json)
    out_path = export_single_replay_payload(records, out_name=args.out_name)
    print(f"[INFO] replay sequence exported: {out_path}")


if __name__ == "__main__":
    main()
