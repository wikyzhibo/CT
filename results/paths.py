from __future__ import annotations

import re
from pathlib import Path


_SAFE_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
_REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = _REPO_ROOT / "results"
ACTION_SEQUENCES_DIR = RESULTS_ROOT / "action_sequences"
GANTT_DIR = RESULTS_ROOT / "gantt"
TRAINING_LOGS_DIR = RESULTS_ROOT / "training_logs"
TOPOLOGY_CACHE_DIR = RESULTS_ROOT / "topology_cache"
MODELS_DIR = RESULTS_ROOT / "models"


def ensure_results_dirs() -> None:
    for path in (
        ACTION_SEQUENCES_DIR,
        GANTT_DIR,
        TRAINING_LOGS_DIR,
        TOPOLOGY_CACHE_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def safe_name(raw: str, default: str) -> str:
    cleaned = _SAFE_PATTERN.sub("_", str(raw).strip())
    return cleaned or default


def action_sequence_path(name: str) -> Path:
    ensure_results_dirs()
    return ACTION_SEQUENCES_DIR / f"{safe_name(name, 'sequence')}.json"


def gantt_output_path(filename: str) -> Path:
    ensure_results_dirs()
    path = GANTT_DIR / safe_name(filename, "gantt.png")
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path


def training_log_output_path(filename: str) -> Path:
    ensure_results_dirs()
    return TRAINING_LOGS_DIR / safe_name(filename, "training_log.json")


def model_output_path(filename: str) -> Path:
    ensure_results_dirs()
    return MODELS_DIR / safe_name(filename, "model.pt")


def topology_cache_path(filename: str) -> Path:
    ensure_results_dirs()
    path = TOPOLOGY_CACHE_DIR / safe_name(filename, "cache.npz")
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    return path

