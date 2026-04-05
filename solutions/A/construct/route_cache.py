"""
build_net() 结果持久化缓存。

缓存文件命名：route_{name}__w1{n1}_w2{n2}_tt{tt}_cl{cl}_pr{pr}_dr{dr}_sc{sc}__{config_hash}_v{VER}.pkl
版本失效策略：
  - route_config 内容变更 → SHA256 哈希变化 → 文件名不同 → 自动 miss
  - build_net() 内部逻辑变更 → 手动递增 _ROUTE_CACHE_VERSION
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from results.paths import route_cache_path, safe_name

# 仅当缓存文件 schema 结构改变时手动递增此常量
_ROUTE_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# 内部：哈希 / 文件名
# ---------------------------------------------------------------------------

def _hash_route_config(route_config: Mapping[str, Any]) -> str:
    """对 route_config dict 内容计算 SHA256，返回前 16 位十六进制字符串。"""
    serialized = json.dumps(route_config, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _make_cache_filename(
    route_name: str,
    n_wafer1: int,
    n_wafer2: int,
    ttime: int,
    cleaning_enabled: bool,
    p_residual_time: int,
    d_residual_time: int,
    scrap_clip_threshold: float,
    config_hash: str,
) -> str:
    """生成缓存文件名（不含目录）。"""
    safe_route = safe_name(route_name, "unknown")
    sc_int = int(round(float(scrap_clip_threshold) * 10))
    return (
        f"route_{safe_route}"
        f"__w1{n_wafer1}_w2{n_wafer2}_tt{ttime}"
        f"_cl{int(bool(cleaning_enabled))}"
        f"_pr{p_residual_time}_dr{d_residual_time}"
        f"_sc{sc_int}"
        f"__{config_hash}_v{_ROUTE_CACHE_VERSION}.pkl"
    )


def _get_cache_path(
    route_name: str,
    n_wafer1: int,
    n_wafer2: int,
    ttime: int,
    cleaning_enabled: bool,
    p_residual_time: int,
    d_residual_time: int,
    scrap_clip_threshold: float,
    config_hash: str,
) -> Path:
    filename = _make_cache_filename(
        route_name=route_name,
        n_wafer1=n_wafer1,
        n_wafer2=n_wafer2,
        ttime=ttime,
        cleaning_enabled=cleaning_enabled,
        p_residual_time=p_residual_time,
        d_residual_time=d_residual_time,
        scrap_clip_threshold=scrap_clip_threshold,
        config_hash=config_hash,
    )
    return route_cache_path(filename)


# ---------------------------------------------------------------------------
# 内部：序列化 / 反序列化
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> Optional[Dict[str, Any]]:
    """从 pickle 文件加载，校验版本；失败时静默返回 None。"""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, dict):
            return None
        if blob.get("_cache_version") != _ROUTE_CACHE_VERSION:
            return None
        return blob["data"]
    except Exception:
        return None


def _save_raw(path: Path, data: Dict[str, Any]) -> None:
    """原子写入：先写 .tmp，再 rename，防止并发写入产生损坏文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {"_cache_version": _ROUTE_CACHE_VERSION, "data": data},
                f,
                protocol=4,  # Python 3.8+ 兼容，比 HIGHEST_PROTOCOL 更可移植
            )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_net_cached(
    route_name: str,
    n_wafer1: int,
    n_wafer2: int,
    ttime: int,
    cleaning_enabled: bool,
    p_residual_time: int,
    d_residual_time: int,
    scrap_clip_threshold: float,
    route_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    尝试从缓存加载 build_net() 结果。

    缓存命中时返回深拷贝（marks 通过 .clone() 复制，确保各环境实例独立）。
    缓存未命中或读取失败时返回 None。
    """
    config_hash = _hash_route_config(route_config)
    path = _get_cache_path(
        route_name=route_name,
        n_wafer1=n_wafer1,
        n_wafer2=n_wafer2,
        ttime=ttime,
        cleaning_enabled=cleaning_enabled,
        p_residual_time=p_residual_time,
        d_residual_time=d_residual_time,
        scrap_clip_threshold=scrap_clip_threshold,
        config_hash=config_hash,
    )
    data = _load_raw(path)
    if data is None:
        return None
    # marks 是 List[Place]，Place 含可变状态，每次加载必须 clone
    result = dict(data)
    result["marks"] = [p.clone() for p in data["marks"]]
    return result


def save_net_cached(
    data: Dict[str, Any],
    route_name: str,
    n_wafer1: int,
    n_wafer2: int,
    ttime: int,
    cleaning_enabled: bool,
    p_residual_time: int,
    d_residual_time: int,
    scrap_clip_threshold: float,
    route_config: Mapping[str, Any],
) -> None:
    """将 build_net() 结果持久化到缓存。写入失败时发出警告但不抛异常。"""
    config_hash = _hash_route_config(route_config)
    path = _get_cache_path(
        route_name=route_name,
        n_wafer1=n_wafer1,
        n_wafer2=n_wafer2,
        ttime=ttime,
        cleaning_enabled=cleaning_enabled,
        p_residual_time=p_residual_time,
        d_residual_time=d_residual_time,
        scrap_clip_threshold=scrap_clip_threshold,
        config_hash=config_hash,
    )
    try:
        _save_raw(path, data)
    except Exception as exc:
        warnings.warn(f"route_cache: 写入缓存失败 ({path.name}): {exc}")
