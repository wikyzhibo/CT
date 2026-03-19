"""
单设备 Petri 网（构网驱动、单机械手、单动作）。
执行链：construct_single -> _get_enable_t -> step -> calc_reward
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any, Deque, Dict, List, Mapping, Optional, Set, Tuple
from pathlib import Path
from solutions.Continuous_model.helper_function import (
    _normalize_wait_durations,
    _preprocess_process_time_map,
    _round_to_nearest_five,
)

import numpy as np
try:
    from numba import njit, prange
except Exception:  # pragma: no cover - numba 可选依赖
    njit = None
    prange = range

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.construct import BasedToken
from solutions.Continuous_model.construct_single import (
    BUFFER_NAMES,
    ROUTE_SPECS,
    build_single_device_net,
    parse_route,
)
from solutions.Continuous_model.route_compiler_single import normalize_route_spec
from solutions.Continuous_model.pn import Place
from solutions.Continuous_model.takt_cycle_analyzer import (
    analyze_cycle,
    build_fixed_takt_result,
)
from visualization.plot import Op, plot_gantt_hatched_residence

CHAMBER = 1
DELIVERY_ROBOT = 2
SOURCE = 3

# 动作不使能原因的人性化描述（用于 Markdown 报告）
REASON_DESC: Dict[str, str] = {
    "pre_color_mismatch": "前置颜色/路径约束不满足",
    "insufficient_tokens": "库所 token 不足",
    "capacity_exceeded": "容量超限",
    "locked_by_arm2_head": "双臂模式：来源被队首晶圆目标锁定",
    "no_receiving_target": "下游无可接收腔室",
    "wrong_destination": "与队首晶圆目标不一致",
    "process_not_ready": "腔室加工未完成",
    "target_cleaning": "目标腔室清洗中",
    "dwell_time_not_met": "运输位停留时间未满足",
    "has_ready_wafer_restrict_wait": "有待取晶圆，仅允许短等待",
    "takt_release_limit": "节拍限制：距上次发片间隔未达当前周期节拍",
    "max_wafers_in_system_limit": "在制品已达上限，禁止继续发片",
}


def _step_core_numpy(
    pre: np.ndarray,
    m: np.ndarray,
    k: np.ndarray,
    pst: np.ndarray,
    t_idx: int,
) -> bool:
    """
    纯 numpy 核心判定：变迁结构性可使能（不含业务路由/清洗约束）。
    """
    if int(t_idx) < 0 or int(t_idx) >= pre.shape[1]:
        return False
    p_count = pre.shape[0]
    for p in range(p_count):
        need = pre[p, t_idx]
        if need > 0 and m[p] < need:
            return False
    for p in range(p_count):
        if pst[p, t_idx] > 0 and m[p] >= k[p]:
            return False
    return True


if njit is not None:
    @njit(cache=True, fastmath=True)
    def step_core_numba(
        pre: np.ndarray,
        m: np.ndarray,
        k: np.ndarray,
        pst: np.ndarray,
        t_idx: int,
    ) -> bool:
        if t_idx < 0 or t_idx >= pre.shape[1]:
            return False
        p_count = pre.shape[0]
        for p in range(p_count):
            need = pre[p, t_idx]
            if need > 0 and m[p] < need:
                return False
        for p in range(p_count):
            if pst[p, t_idx] > 0 and m[p] >= k[p]:
                return False
        return True

    @njit(cache=True, fastmath=True, parallel=True)
    def step_core_batch_numba(
        pre: np.ndarray,
        m: np.ndarray,
        k: np.ndarray,
        pst: np.ndarray,
        t_indices: np.ndarray,
        out_mask: np.ndarray,
    ) -> None:
        for i in prange(t_indices.shape[0]):
            t_idx = int(t_indices[i])
            if t_idx < 0 or t_idx >= pre.shape[1]:
                out_mask[i] = False
                continue
            ok = True
            for p in range(pre.shape[0]):
                need = pre[p, t_idx]
                if need > 0 and m[p] < need:
                    ok = False
                    break
            if ok:
                for p in range(pre.shape[0]):
                    if pst[p, t_idx] > 0 and m[p] >= k[p]:
                        ok = False
                        break
            out_mask[i] = ok
else:
    step_core_numba = _step_core_numpy

    def step_core_batch_numba(
        pre: np.ndarray,
        m: np.ndarray,
        k: np.ndarray,
        pst: np.ndarray,
        t_indices: np.ndarray,
        out_mask: np.ndarray,
    ) -> None:
        for i in range(int(t_indices.shape[0])):
            out_mask[i] = _step_core_numpy(pre, m, k, pst, int(t_indices[i]))


class ClusterTool:
    _VALID_ROUTE_CODES: Dict[str, Set[int]] = {
        "single": {0, 1},
        "cascade": {1, 2, 3, 4, 5, 6},
    }

    @classmethod
    def _normalize_device_mode(cls, raw_mode: Any) -> str:
        mode = str(raw_mode).strip().lower()
        if mode not in cls._VALID_ROUTE_CODES:
            valid_modes = sorted(cls._VALID_ROUTE_CODES.keys())
            raise ValueError(
                f"invalid device_mode={raw_mode!r}; expected one of {valid_modes}"
            )
        return mode

    @classmethod
    def _normalize_route_code(cls, raw_route_code: Any, device_mode: str) -> int:
        try:
            route_code = int(raw_route_code)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid route_code={raw_route_code!r}; route_code must be an integer"
            ) from exc
        valid_route_codes = sorted(cls._VALID_ROUTE_CODES[device_mode])
        if route_code not in cls._VALID_ROUTE_CODES[device_mode]:
            raise ValueError(
                f"invalid route_code={route_code!r} for device_mode={device_mode!r}; "
                f"expected one of {valid_route_codes}"
            )
        return route_code

    @staticmethod
    def _resolve_route_name_from_config(
        route_cfg: Mapping[str, Any],
        device_mode: str,
        route_code: int,
        preferred_name: Optional[str],
    ) -> Optional[str]:
        routes = route_cfg.get("routes") or {}
        if not isinstance(routes, Mapping) or not routes:
            return None
        if preferred_name and preferred_name in routes:
            return str(preferred_name)
        legacy_alias = (
            route_cfg.get("legacy", {})
            .get("route_code_alias", {})
            .get(device_mode, {})
        )
        aliased = legacy_alias.get(str(route_code))
        if aliased and aliased in routes:
            return str(aliased)
        return str(next(iter(routes.keys())))

    @staticmethod
    def _extract_route_stage_overrides(
        route_cfg: Mapping[str, Any],
        route_name: str,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, float]]:
        source_name = str((route_cfg.get("source") or {}).get("name", "LP"))
        sink_name = str((route_cfg.get("sink") or {}).get("name", "LP_done"))
        cfg_chambers = dict(route_cfg.get("chambers") or {})
        chamber_kind_map = {
            str(name): str((spec or {}).get("kind", "process"))
            for name, spec in cfg_chambers.items()
        }
        routes = dict(route_cfg.get("routes") or {})
        route_entry = dict(routes.get(route_name) or {})
        normalized = normalize_route_spec(
            route_name=str(route_name),
            route_cfg=route_entry,
            source_name=source_name,
            sink_name=sink_name,
            chamber_kind_map=chamber_kind_map,
        )

        proc_time_map: Dict[str, int] = {}
        cleaning_duration_map: Dict[str, int] = {}
        cleaning_trigger_map: Dict[str, int] = {}
        proc_rand_scale_map: Dict[str, float] = {}

        def _set_consistent_int(
            target: Dict[str, int],
            chamber_name: str,
            value: int,
            field_name: str,
        ) -> None:
            old = target.get(chamber_name)
            if old is not None and int(old) != int(value):
                raise ValueError(
                    f"route {route_name} has conflicting {field_name} for {chamber_name}: "
                    f"{old} vs {value}"
                )
            target[chamber_name] = int(value)

        def _set_consistent_float(
            target: Dict[str, float],
            chamber_name: str,
            value: float,
            field_name: str,
        ) -> None:
            old = target.get(chamber_name)
            if old is not None and float(old) != float(value):
                raise ValueError(
                    f"route {route_name} has conflicting {field_name} for {chamber_name}: "
                    f"{old} vs {value}"
                )
            target[chamber_name] = float(value)

        for stage in normalized:
            is_process_stage = str(stage.stage_type) == "process"
            for chamber_name in stage.candidates:
                if chamber_name in {source_name, sink_name}:
                    continue
                if stage.stage_process_time is not None:
                    p_val = float(stage.stage_process_time)
                    if (not is_process_stage) or p_val > 0:
                        _set_consistent_int(
                            proc_time_map,
                            chamber_name,
                            int(round(p_val)),
                            "process_time",
                        )
                if stage.stage_cleaning_duration is not None:
                    _set_consistent_int(
                        cleaning_duration_map,
                        chamber_name,
                        int(stage.stage_cleaning_duration),
                        "cleaning_duration",
                    )
                if stage.stage_cleaning_trigger_wafers is not None:
                    _set_consistent_int(
                        cleaning_trigger_map,
                        chamber_name,
                        int(stage.stage_cleaning_trigger_wafers),
                        "cleaning_trigger_wafers",
                    )
                if stage.stage_proc_rand_scale is not None:
                    _set_consistent_float(
                        proc_rand_scale_map,
                        chamber_name,
                        float(stage.stage_proc_rand_scale),
                        "proc_rand_scale",
                    )

        return (
            proc_time_map,
            cleaning_duration_map,
            cleaning_trigger_map,
            proc_rand_scale_map,
        )

    def __init__(self, config: PetriEnvConfig = None) -> None:
        assert config is not None, "config must be provided"
        self.config = config

        self.MAX_TIME = config.MAX_TIME
        self.n_wafer = int(config.n_wafer)
        self.max_wafers_in_system = int(config.max_wafers_in_system)
        
        self.done_event_reward = int(config.done_event_reward)
        self.finish_event_reward = self.done_event_reward * 6
        self.scrap_event_penalty = int(config.scrap_event_penalty)
        self.idle_event_penalty = float(config.idle_event_penalty)
        self.release_event_penalty = float(config.release_event_penalty)
        
        self.warn_coef_penalty = int(config.warn_coef_penalty)
        self.processing_coef_reward = float(config.processing_coef_reward)
        self.transport_overtime_coef_penalty = float(config.transport_overtime_coef_penalty)
        self.time_coef_penalty = float(config.time_coef_penalty)
        
        self.P_Residual_time = int(config.P_Residual_time)
        self.D_Residual_time = int(config.D_Residual_time)
        self.T_transport = int(config.T_transport)
        self.T_load = int(config.T_load)
        
        self.stop_on_scrap = bool(config.stop_on_scrap)
        
        self.reward_config = dict(config.reward_config)
        # 临时执行策略：固定单臂，不启用双臂分支。
        self.robot_capacity = 1

        self.cleaning_enabled = bool(config.cleaning_enabled)
        self.cleaning_targets = set(config.cleaning_targets)
        self.cleaning_trigger_wafers = config.cleaning_trigger_wafers
        self.cleaning_duration = max(0, int(config.cleaning_duration))
        self._cleaning_trigger_map: Dict[str, int] = dict(
            config.cleaning_trigger_wafers_map
        ) if getattr(config, "cleaning_trigger_wafers_map", None) else {
            c: config.cleaning_trigger_wafers for c in config.cleaning_targets
        }
        self._cleaning_duration_map: Dict[str, int] = dict(
            config.cleaning_duration_map
        ) if getattr(config, "cleaning_duration_map", None) else {
            c: max(0, int(config.cleaning_duration)) for c in config.cleaning_targets
        }
        self.wait_durations = _normalize_wait_durations(config.wait_durations)
        
        self.device_mode = self._normalize_device_mode(config.device_mode)
        self.route_code = self._normalize_route_code(config.route_code, self.device_mode)
        self.config.device_mode = self.device_mode
        self.config.route_code = self.route_code
        self.route4_takt_interval = int(getattr(config, "route4_takt_interval", 0) or 0)
        self.single_device_mode = self.device_mode
        self.single_route_code = self.route_code
        self.single_route_config = getattr(config, "single_route_config", None)
        self.single_route_name = getattr(config, "single_route_name", None)
        self._selected_single_route_name: Optional[str] = None
        self._route_stage_proc_time_map: Dict[str, int] = {}
        self._route_stage_cleaning_duration_map: Dict[str, int] = {}
        self._route_stage_cleaning_trigger_map: Dict[str, int] = {}
        self._route_stage_proc_rand_scale_map: Dict[str, float] = {}
        if self.single_route_config is not None:
            self._selected_single_route_name = self._resolve_route_name_from_config(
                route_cfg=dict(self.single_route_config or {}),
                device_mode=self.device_mode,
                route_code=self.route_code,
                preferred_name=self.single_route_name,
            )
            if self._selected_single_route_name is not None:
                (
                    self._route_stage_proc_time_map,
                    self._route_stage_cleaning_duration_map,
                    self._route_stage_cleaning_trigger_map,
                    self._route_stage_proc_rand_scale_map,
                ) = self._extract_route_stage_overrides(
                    route_cfg=dict(self.single_route_config or {}),
                    route_name=self._selected_single_route_name,
                )
                self.single_route_name = self._selected_single_route_name
                self.config.single_route_name = self._selected_single_route_name

        route_key = (self.device_mode, self.route_code)
        if self.single_route_config is None:
            stages = ROUTE_SPECS.get(route_key)
            if stages is None:
                valid_route_codes = sorted(self._VALID_ROUTE_CODES[self.device_mode])
                raise ValueError(
                    f"route spec not found for device_mode={self.device_mode!r}, "
                    f"route_code={self.route_code!r}; expected route_code in {valid_route_codes}"
                )
            self._route_stages: List[List[str]] = list(stages)  # 用于节拍分析器
            _bootstrap_route_meta = parse_route(stages, BUFFER_NAMES)
            if self.device_mode == "cascade" and self.route_code == 4:
                _bootstrap_route_meta["u_targets"]["LLD"] = ["PM7", "LP_done"]
            self.chambers = tuple(_bootstrap_route_meta["chambers"])
            self._timeline_chambers = tuple(_bootstrap_route_meta["timeline_chambers"])
            self._u_targets = dict(_bootstrap_route_meta["u_targets"])
            self._step_map = dict(_bootstrap_route_meta["step_map"])
            self._release_station_aliases = dict(_bootstrap_route_meta["release_station_aliases"])
            self._release_chain_by_u = dict(_bootstrap_route_meta["release_chain_by_u"])
            self._system_entry_places = set(_bootstrap_route_meta["system_entry_places"])
            self._ready_chambers = tuple(_bootstrap_route_meta["chambers"])
            self._single_process_chambers = self.chambers
        else:
            self._route_stages = []
            # 配置驱动下，优先从目标 route 归一化结果提取 chambers（不含 source/sink/buffer）
            _cfg = dict(self.single_route_config or {})
            _cfg_chambers = dict(_cfg.get("chambers") or {})
            _routes = dict(_cfg.get("routes") or {})
            _selected_route_name = self._selected_single_route_name

            _route_chambers: List[str] = []
            if _selected_route_name is not None:
                try:
                    _source_name = str((_cfg.get("source") or {}).get("name", "LP"))
                    _sink_name = str((_cfg.get("sink") or {}).get("name", "LP_done"))
                    _kind_map = {
                        name: str((spec or {}).get("kind", "process"))
                        for name, spec in _cfg_chambers.items()
                    }
                    _normalized = normalize_route_spec(
                        route_name=str(_selected_route_name),
                        route_cfg=dict(_routes.get(_selected_route_name) or {}),
                        source_name=_source_name,
                        sink_name=_sink_name,
                        chamber_kind_map=_kind_map,
                    )
                    self._route_stages = [
                        list(stage.candidates) for stage in _normalized[1:-1]
                    ]
                    for stage in _normalized[1:-1]:
                        if str(stage.stage_type) != "buffer":
                            for name in stage.candidates:
                                if name not in _route_chambers:
                                    _route_chambers.append(name)
                except Exception:
                    _route_chambers = []

            if _route_chambers:
                self.chambers = tuple(_route_chambers)
            else:
                self.chambers = tuple(
                    name
                    for name, spec in _cfg_chambers.items()
                    if str((spec or {}).get("kind", "process")) != "buffer"
                ) or tuple(_cfg_chambers.keys())
            self._timeline_chambers = self.chambers
            self._u_targets = {}
            self._step_map = {}
            self._release_station_aliases = {}
            self._release_chain_by_u = {}
            self._system_entry_places = set()
            self._ready_chambers = self.chambers
            self._single_process_chambers = self.chambers

        self.proc_rand_enabled = bool(config.proc_rand_enabled)
        self._proc_rand_scale_map = dict(config.proc_time_rand_scale_map or {})
        raw = dict(config.process_time_map or {})
        if self.single_route_config is not None:
            if self._route_stage_proc_time_map:
                raw.update(self._route_stage_proc_time_map)
            if self._route_stage_cleaning_duration_map:
                self._cleaning_duration_map.update(self._route_stage_cleaning_duration_map)
            if self._route_stage_cleaning_trigger_map:
                self._cleaning_trigger_map.update(self._route_stage_cleaning_trigger_map)
            for chamber_name, scale in self._route_stage_proc_rand_scale_map.items():
                clipped = max(0.0, min(1.0, float(scale)))
                self._proc_rand_scale_map[chamber_name] = {
                    "min": 1.0 - clipped,
                    "max": 1.0 + clipped,
                }
        self._base_proc_time_map = self._preprocess_process_time_map(raw)
        self._episode_proc_time_map: Dict[str, int] = {}

        obs_config = {
            "P_Residual_time": self.P_Residual_time,
            "D_Residual_time": self.D_Residual_time,
            "cleaning_duration": self.cleaning_duration,
            "cleaning_trigger_wafers": self.cleaning_trigger_wafers,
            "cleaning_duration_map": self._cleaning_duration_map,
            "cleaning_trigger_wafers_map": self._cleaning_trigger_map,
            "scrap_clip_threshold": 20.0,
        }
        info = build_single_device_net(
            n_wafer=self.n_wafer,
            ttime=max(1, self.T_transport),
            robot_capacity=self.robot_capacity,
            process_time_map=self._base_proc_time_map,
            route_code=self.route_code,
            device_mode=self.device_mode,
            obs_config=obs_config,
            route_config=self.single_route_config,
            route_name=self._selected_single_route_name or self.single_route_name,
        )
        route_meta = dict(info.get("route_meta") or {})
        if not self._route_stages:
            # 从 route_meta 反推用于节拍分析的内部阶段（不含 LP/LP_done）
            aliases = route_meta.get("release_station_aliases") or {}
            if isinstance(aliases, dict):
                ordered_keys = sorted(
                    [k for k in aliases.keys() if str(k).startswith("s")],
                    key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 0,
                )
                self._route_stages = [list(aliases[k]) for k in ordered_keys]
        self.chambers = tuple(route_meta.get("chambers", ()))
        self._timeline_chambers = tuple(route_meta.get("timeline_chambers", ()))
        self._u_targets = dict(route_meta.get("u_targets", {}))
        self._step_map = dict(route_meta.get("step_map", {}))
        self._release_station_aliases = dict(route_meta.get("release_station_aliases", {}))
        self._release_chain_by_u = dict(route_meta.get("release_chain_by_u", {}))
        self._system_entry_places = set(route_meta.get("system_entry_places", set()))
        self._ready_chambers = tuple(route_meta.get("chambers", ()))
        self._single_process_chambers = self.chambers
        self.pre: np.ndarray = info["pre"]
        self.pre_color: np.ndarray = info.get("pre_color", self.pre[:, :, None])
        self.pst: np.ndarray = info["pst"]
        self.net: np.ndarray = self.pst - self.pre
        self.m0: np.ndarray = info["m0"]
        self.m: np.ndarray = self.m0.copy()
        self.md: np.ndarray = info["md"]
        self.ptime: np.ndarray = info["ptime"]
        self.k: np.ndarray = info["capacity"]
        self.id2p_name: List[str] = info["id2p_name"]
        self.id2t_name: List[str] = info["id2t_name"]
        self._t_route_code_map: Dict[str, int] = dict(info.get("t_route_code_map") or {})
        self._t_route_code_by_idx: List[int] = [
            int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name
        ]
        self._t_code_to_target: Dict[int, str] = {
            v: k for k, v in self._t_route_code_map.items() if k.startswith("t_")
        }
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.ttime = int(np.max(info["ttime"])) if len(info["ttime"]) > 0 else 5

        # 预计算的 pre/pst 库所索引与运输位索引（构网返回或本地计算以兼容旧版）
        self._pre_place_indices: List[np.ndarray] = info.get("pre_place_indices")
        self._pst_place_indices: List[np.ndarray] = info.get("pst_place_indices")
        self._transport_pre_place_idx: List[int] = info.get("transport_pre_place_idx")
        if (
            self._pre_place_indices is None
            or self._pst_place_indices is None
            or self._transport_pre_place_idx is None
        ):
            self._pre_place_indices = [
                np.flatnonzero(self.pre[:, t] > 0) for t in range(self.T)
            ]
            self._pst_place_indices = [
                np.flatnonzero(self.pst[:, t] > 0) for t in range(self.T)
            ]
            self._transport_pre_place_idx = []
            for t in range(self.T):
                indices = self._pre_place_indices[t]
                found = next(
                    (
                        int(idx)
                        for idx in indices
                        if self.id2p_name[int(idx)].startswith("d_")
                    ),
                    -1,
                )
                self._transport_pre_place_idx.append(found)

        self.ori_marks: List[Place] = info["marks"]
        self.marks: List[Place] = self._clone_marks(self.ori_marks)
        self._align_base_proc_time_map_with_route_chambers()
        # 临时执行策略：除 LP/LP_done 外全部按 unit-capacity 运行。
        for idx, p in enumerate(self.marks):
            if p.name not in {"LP", "LP_done"}:
                p.capacity = 1
                self.k[idx] = 1
        self._refresh_episode_proc_time()

        self.time = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 50
        self.entered_wafer_count = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log: List[Dict[str, Any]] = []
        self.enable_statistics = True
        self._per_wafer_reward = 0.0
        self._token_stats: Dict[int, Dict[str, Any]] = {}
        self._qtime_violated_tokens: Set[int] = set()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        self._cascade_round_robin_pairs: Dict[str, Tuple[str, ...]] = {}
        self._cascade_round_robin_next: Dict[str, str] = {}
        self._single_round_robin_pairs: Dict[str, Tuple[str, str]] = {}
        self._single_round_robin_next: Dict[str, str] = {}
        self._u_transition_by_source: Dict[str, int] = {}
        self._u_transition_by_source_transport: Dict[Tuple[str, str], int] = {}
        self._t_transitions_by_transport: Dict[str, List[int]] = {}
        self._token_pool: List[BasedToken] = []
        if self.device_mode == "cascade":
            if self.single_route_config is not None:
                for source, targets in self._u_targets.items():
                    if source in {"LP", "LLC", "LLD"} and len(targets) >= 2:
                        pm_targets = [t for t in targets if t.startswith("PM") or t == "LP_done"]
                        if pm_targets:
                            self._cascade_round_robin_pairs[source] = tuple(pm_targets)
            if not self._cascade_round_robin_pairs and self.route_code != 4:
                self._cascade_round_robin_pairs["LP"] = ("PM7", "PM8")
            if self.route_code == 1 and "LLC" not in self._cascade_round_robin_pairs:
                self._cascade_round_robin_pairs["LLC"] = ("PM1", "PM2", "PM3", "PM4")
            if self.route_code in {2, 3} and "LLC" not in self._cascade_round_robin_pairs:
                self._cascade_round_robin_pairs["LLC"] = ("PM1", "PM2")
            if self.route_code == 2 and "LLD" not in self._cascade_round_robin_pairs:
                self._cascade_round_robin_pairs["LLD"] = ("PM9", "PM10")
            self._cascade_round_robin_next = {
                source: pair[0] for source, pair in self._cascade_round_robin_pairs.items()
            }
        else:
            for source, targets in self._u_targets.items():
                if len(targets) == 2:
                    pair = (str(targets[0]), str(targets[1]))
                    self._single_round_robin_pairs[source] = pair
            self._single_round_robin_next = {
                source: pair[0] for source, pair in self._single_round_robin_pairs.items()
            }
        self._last_deadlock = False
        self._chamber_timeline: Dict[str, list] = {name: [] for name in self._timeline_chambers}
        self._chamber_active: Dict[str, Dict[int, int]] = {name: {} for name in self._timeline_chambers}
        self._init_cleaning_state()
        self._takt_result: Optional[Dict[str, Any]] = self._compute_takt_result()
        print(self._takt_result)
        self._last_u_LP_fire_time: int = 0
        self._u_LP_release_count: int = 0
        self._training = True
        self._profiling_enabled = True
        self._step_profile = {
            "count": 0,
            "total_s": 0.0,
            "get_enable_t_s": 0.0,
            "fire_s": 0.0,
            "build_obs_s": 0.0,
            "advance_and_reward_s": 0.0,
            "next_event_delta_s": 0.0,
            "other_s": 0.0,
        }
        self._last_state_scan: Dict[str, Any] = {}
        self._is_cascade: bool = (self.device_mode == "cascade")
        self._do_time_cost: bool = bool(self.reward_config.get("time_cost", 1))
        self._do_proc_reward: bool = bool(self.reward_config.get("proc_reward", 1))
        self._do_transport_penalty: bool = bool(self.reward_config.get("transport_penalty", 1))
        self._do_warn_penalty: bool = bool(self.reward_config.get("warn_penalty", 1))
        self._ready_chambers_set: frozenset = frozenset(self._ready_chambers)
        self._place_by_name: Dict[str, Place] = {}
        self._obs_place_names: List[str] = []
        self._obs_places: List[Place] = []
        self._obs_offsets: List[int] = []
        self._obs_specs: List[Dict[str, Any]] = []
        self._obs_dim: int = 0
        self._obs_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._obs_return_copy: bool = True
        self._rebuild_place_cache()
        self._init_obs_cache()
        self._build_transition_index()
        self._rebuild_token_pool()
        if not self._training:
            print(self._takt_result)

    def train(self):
        """训练模式"""
        self._training = True

    def eval(self):
        """评估模式"""
        self._training = False

    def step(self, a1=None, detailed_reward: bool = False, wait_duration: Optional[int] = None):
        """
        单设备一步推进入口。
        返回：(done, reward_result, scrap, action_mask, obs)
        """
        _pf = self._profiling_enabled
        SCRAPE = False
        if _pf:
            step_start = perf_counter()
            advance_and_reward_s = fire_s = get_enable_t_s = build_obs_s = next_event_delta_s = 0.0

        self._last_deadlock = False
        _mask_start = int(self.T)
        _mask_n = _mask_start + len(self.wait_durations)
        _lp_done = self._lp_done

        if self.time >= self.MAX_TIME:
            timeout_reward = {"total": -100.0, "timeout": True} if detailed_reward else -100.0
            action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
            obs = self.get_obs()
            if _pf:
                self._record_step_profile(total_s=perf_counter() - step_start,
                    get_enable_t_s=0.0, fire_s=0.0, build_obs_s=0.0,
                    advance_and_reward_s=0.0, next_event_delta_s=0.0)
            return True, timeout_reward, True, action_mask, obs

        action = a1
        do_wait = (wait_duration is not None) or action is None
        scan_info: Dict[str, Any] = {}
        log_entry: Optional[Dict[str, Any]] = None
        t1 = self.time

        if do_wait:
            requested_wait = int(wait_duration)
            episode_finished = len(_lp_done.tokens) >= self.n_wafer
            if episode_finished and requested_wait > 5:
                actual_dt = 5
            elif requested_wait == 5:
                actual_dt = requested_wait
            else:
                if _pf: t_next_event = perf_counter()
                next_event_delta = self.get_next_event_delta()
                if _pf: next_event_delta_s += perf_counter() - t_next_event
                if next_event_delta is None:
                    actual_dt = requested_wait
                elif next_event_delta <= 0:
                    actual_dt = min(requested_wait, 5)
                else:
                    actual_dt = min(requested_wait, next_event_delta)

            t2 = t1 + actual_dt
            if _pf: t_ar = perf_counter()
            reward_result, scan_info = self._advance_and_compute_reward(
                actual_dt, t1, t2, detailed=detailed_reward)
            if _pf: advance_and_reward_s += perf_counter() - t_ar
            self._consecutive_wait_time += (t2 - t1)

            if self._consecutive_wait_time >= self.idle_timeout and not self._idle_penalty_applied:
                self._idle_penalty_applied = True
                if detailed_reward:
                    reward_result["idle_timeout_penalty"] = -float(self.idle_event_penalty)
                    reward_result["total"] -= float(self.idle_event_penalty)
                else:
                    reward_result -= float(self.idle_event_penalty)
        else:
            self._consecutive_wait_time = 0
            t2 = t1 + self.ttime
            if _pf: t_ar = perf_counter()
            reward_result, scan_info = self._advance_and_compute_reward(
                self.ttime, t1, t2, detailed=detailed_reward)
            if _pf: advance_and_reward_s += perf_counter() - t_ar
            if _pf: t_fire = perf_counter()
            log_entry = self._fire(int(action), start_time=t1, end_time=t2)
            if _pf: fire_s += perf_counter() - t_fire
            self.fire_log.append(log_entry)

            if self._per_wafer_reward > 0:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] += self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
                self._per_wafer_reward = 0.0

        finish = len(_lp_done.tokens) >= self.n_wafer
        scan = scan_info if isinstance(scan_info, dict) else {}
        is_scrap = bool(scan.get("is_scrap", False))
        scrap_info = scan.get("scrap_info")
        if "is_scrap" not in scan:
            is_scrap, scrap_info = self._check_scrap()
        # 非 WAIT 场景下，若本步 u_* 已取走同一 resident wafer，则撤销本步 scrap。
        if self._should_cancel_resident_scrap_after_fire(scan=scan, log_entry=log_entry):
            is_scrap = False
            scrap_info = None
            scan["is_scrap"] = False
            scan["scrap_info"] = None
            self._last_state_scan = scan
        if is_scrap:
            self.scrap_count += 1
            self.resident_violation_count += 1
            if detailed_reward:
                reward_result["scrap_penalty"] -= float(self.scrap_event_penalty)
                reward_result["total"] -= float(self.scrap_event_penalty)
                reward_result["scrap_info"] = scrap_info
            else:
                reward_result -= float(self.scrap_event_penalty)
            if self.stop_on_scrap:
                action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
                obs = self.get_obs()
                if _pf:
                    self._record_step_profile(total_s=perf_counter() - step_start,
                        get_enable_t_s=0.0, fire_s=0.0, build_obs_s=0.0,
                        advance_and_reward_s=0.0, next_event_delta_s=0.0)
                return True, reward_result, True, action_mask, obs

        if finish:
            if detailed_reward:
                reward_result["finish_bonus"] += float(self.finish_event_reward)
                reward_result["total"] += float(self.finish_event_reward)
            else:
                reward_result += float(self.finish_event_reward)
            SCRAPE = False

        if _pf: t_mask = perf_counter()
        action_mask = self.get_action_mask(wait_action_start=_mask_start, n_actions=_mask_n)
        if _pf: get_enable_t_s += perf_counter() - t_mask
        if _pf: t_obs = perf_counter()
        obs = self.get_obs()
        if _pf:
            build_obs_s += perf_counter() - t_obs
            self._record_step_profile(total_s=perf_counter() - step_start,
                get_enable_t_s=get_enable_t_s, fire_s=fire_s, build_obs_s=build_obs_s,
                advance_and_reward_s=advance_and_reward_s, next_event_delta_s=next_event_delta_s)
        return bool(finish), reward_result, SCRAPE, action_mask, obs

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
        self._refresh_episode_proc_time()
        self._rebuild_place_cache()
        self._init_obs_cache()
        self.time = 0
        self.entered_wafer_count = 0
        self.done_count = 0
        self.scrap_count = 0
        self.deadlock_count = 0
        self.resident_violation_count = 0
        self.qtime_violation_count = 0
        self.fire_log.clear()
        self._per_wafer_reward = 0.0
        self._token_stats = {}
        self._qtime_violated_tokens.clear()
        self._idle_penalty_applied = False
        self._consecutive_wait_time = 0
        self._next_machine_id = 1
        if self.device_mode == "cascade":
            self._cascade_round_robin_next = {
                source: pair[0] for source, pair in self._cascade_round_robin_pairs.items()
            }
        else:
            self._single_round_robin_next = {
                source: pair[0] for source, pair in self._single_round_robin_pairs.items()
            }
        self._last_deadlock = False
        self._chamber_timeline = {name: [] for name in self._timeline_chambers}
        self._chamber_active = {name: {} for name in self._timeline_chambers}
        self._init_cleaning_state()
        self._takt_result = self._compute_takt_result()
        self._last_u_LP_fire_time = 0
        self._u_LP_release_count = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        for idx, p in enumerate(self.marks):
            if p.name not in {"LP", "LP_done"}:
                p.capacity = 1
                self.k[idx] = 1
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._rebuild_token_pool()
        self._last_state_scan = {}
        return None, self._get_enable_t()

    def _get_obs_place_order(self) -> List[str]:
        """返回观测顺序：LP + 运输位 + 腔室。"""
        tm_names = ["d_TM2", "d_TM3"] if self.device_mode == "cascade" else ["d_TM1"]
        tm_names = [n for n in tm_names if n in self.id2p_name]
        candidates = list(self.chambers)
        if self.device_mode == "cascade" and "LLC" not in candidates:
            candidates.append("LLC")
        chambers: List[str] = []
        seen: Set[str] = set()
        for name in candidates:
            if not (name.startswith("PM") or name in {"LLC", "LLD"}):
                continue
            if name in seen:
                continue
            chambers.append(name)
            seen.add(name)
        if not chambers:
            chambers = ["PM1", "PM3", "PM4"]
        return ["LP"] + tm_names + chambers

    def _rebuild_place_cache(self) -> None:
        self._place_by_name = {p.name: p for p in self.marks}
        self._lp_done = self._place_by_name.get("LP_done")

    def _init_obs_cache(self) -> None:
        order = self._get_obs_place_order()
        obs_names = [name for name in order if name in self._place_by_name]
        obs_places = [self._place_by_name[name] for name in obs_names]
        offsets: List[int] = []
        specs: List[Dict[str, Any]] = []
        cursor = 0
        for place in obs_places:
            dim = int(place.get_obs_dim())
            offsets.append(cursor)
            specs.append({"name": place.name, "offset": cursor, "dim": dim})
            cursor += dim
        self._obs_place_names = obs_names
        self._obs_places = obs_places
        self._obs_offsets = offsets
        self._obs_specs = specs
        self._obs_dim = int(cursor)
        self._obs_buffer = np.zeros(self._obs_dim, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        if self._obs_dim == 0:
            return np.zeros(0, dtype=np.float32)
        self._update_ll_direction_obs_flags()
        buffer = self._obs_buffer
        buffer[:] = 0.0
        for place, offset in zip(self._obs_places, self._obs_offsets):
            place.write_obs_fast(buffer, offset)
        if self._obs_return_copy:
            return buffer.copy()
        return buffer

    def get_obs_dim(self) -> int:
        return int(self._obs_dim)

    def _peek_target_for_source_for_obs(self, source: str) -> Optional[str]:
        """
        仅用于观测构造的无副作用目标窥视：
        - 不推进 round-robin 指针；
        - 尽量给出“下一跳意图”，即便当前容量/清洗导致暂不可发射。
        """
        target = self._select_target_for_source(
            source,
            ignore_cleaning=True,
            advance_round_robin=False,
        )
        if target is not None:
            return target
        candidates = list(self._u_targets.get(source, []))
        if not candidates:
            return None
        if self.device_mode == "cascade" and source in self._cascade_round_robin_pairs:
            rr_targets = list(self._cascade_round_robin_pairs[source])
            next_target = self._cascade_round_robin_next.get(source, rr_targets[0])
            if next_target in candidates:
                return next_target
        return candidates[0]

    def _update_ll_direction_obs_flags(self) -> None:
        """
        LLC/LLD 方向位规则：
        - 下一跳由 TM3 搬运 => in_flag=1
        - 下一跳由 TM2 搬运 => out_flag=1
        """
        for ll_name in ("LLC", "LLD"):
            place = self._place_by_name.get(ll_name)
            if place is None or not hasattr(place, "set_direction_flags"):
                continue
            has_wafer = len(place.tokens) > 0
            if (not has_wafer) or self.device_mode != "cascade":
                place.set_direction_flags(False, False)
                continue
            target = self._peek_target_for_source_for_obs(ll_name)
            if target is None:
                place.set_direction_flags(False, False)
                continue
            transport = self._transport_for_t_target(str(target))
            place.set_direction_flags(transport == "d_TM3", transport == "d_TM2")

    def _record_step_profile(
        self,
        total_s: float,
        get_enable_t_s: float,
        fire_s: float,
        build_obs_s: float,
        advance_and_reward_s: float,
        next_event_delta_s: float = 0.0,
    ) -> None:
        tracked = (
            get_enable_t_s
            + fire_s
            + build_obs_s
            + advance_and_reward_s
            + next_event_delta_s
        )
        other_s = max(0.0, float(total_s) - float(tracked))
        self._step_profile["count"] += 1
        self._step_profile["total_s"] += float(total_s)
        self._step_profile["get_enable_t_s"] += float(get_enable_t_s)
        self._step_profile["fire_s"] += float(fire_s)
        self._step_profile["build_obs_s"] += float(build_obs_s)
        self._step_profile["advance_and_reward_s"] += float(advance_and_reward_s)
        self._step_profile["next_event_delta_s"] += float(next_event_delta_s)
        self._step_profile["other_s"] += float(other_s)

    def get_step_profile_summary(self) -> Dict[str, Any]:
        count = int(self._step_profile.get("count", 0))
        total_s = float(self._step_profile.get("total_s", 0.0))
        summary: Dict[str, Any] = {
            "count": count,
            "total_ms": total_s * 1000.0,
            "avg_ms": (total_s * 1000.0 / count) if count > 0 else 0.0,
            "steps_per_sec": (count / total_s) if total_s > 0 else 0.0,
            "segments": {},
        }
        keys = (
            "get_enable_t",
            "fire",
            "build_obs",
            "advance_and_reward",
            "next_event_delta",
            "other",
        )
        for key in keys:
            segment_total_s = float(self._step_profile.get(f"{key}_s", 0.0))
            ratio_pct = (segment_total_s / total_s * 100.0) if total_s > 0 else 0.0
            summary["segments"][key] = {
                "total_ms": segment_total_s * 1000.0,
                "avg_ms": (segment_total_s * 1000.0 / count) if count > 0 else 0.0,
                "ratio_pct": ratio_pct,
            }
        return summary

    def calc_reward(self, t1: int, t2: int, detailed: bool = False):
        dt = max(0, t2 - t1)
        parts = {
            "time_cost": 0.0,
            "proc_reward": 0.0,
            "safe_reward": 0.0,
            "warn_penalty": 0.0,
            "penalty": 0.0,
            "wafer_done_bonus": 0.0,
            "finish_bonus": 0.0,
            "scrap_penalty": 0.0,
        }

        # 时间惩罚：每步按 dt 线性惩罚，鼓励更快完成
        if self.reward_config.get("time_cost", 1):
            parts["time_cost"] = -float(dt * self.time_coef_penalty)

        # 加工奖励：每个加工位上每个晶圆根据加工进度给予奖励
        if self.reward_config.get("proc_reward", 1):
            for p in self.marks:
                if p.type != CHAMBER or len(p.tokens) == 0 or p.processing_time <= 0:
                    continue
                remain = max(0, p.processing_time - int(p.head().stay_time))
                progress = min(dt, remain)
                parts["proc_reward"] += self.processing_coef_reward * float(progress)

        # 与 pn.py 对齐：运输位(type=2, 单设备主要是 d_TM1)超过 D_Residual_time 后按超时秒数惩罚
        if self.reward_config.get("transport_penalty", 1):
            for p in self.marks:
                if p.type != DELIVERY_ROBOT or len(p.tokens) == 0:
                    continue
                for tok in p.tokens:
                    deadline = int(tok.enter_time) + int(self.D_Residual_time)
                    over_start = max(int(t1), deadline)
                    if int(t2) > over_start:
                        parts["penalty"] -= float(int(t2) - over_start) * float(self.transport_overtime_coef_penalty)

        # 驻留警告惩罚
        for p in self.marks:
            if p.type != CHAMBER or len(p.tokens) == 0:
                continue
            left = p.processing_time + self.P_Residual_time - p.head().stay_time
            if self.reward_config.get("warn_penalty", 1) and left <= self.P_Residual_time:
                parts["warn_penalty"] -= int(self.warn_coef_penalty) * dt

        total = sum(parts.values())
        if detailed:
            parts["total"] = float(total)
            return parts
        return float(total)

    def _advance_and_compute_reward(
        self,
        dt: int,
        t1: int,
        t2: int,
        detailed: bool = False,
    ) -> tuple:
        """
        单次 marks 遍历完成：reward 计算 + stay_time 推进 + 清洗/idle 推进 + scrap/qtime 检测。
        排序：reward 使用推进前 stay_time，scrap/qtime 使用推进后 stay_time。
        返回 (reward_result, scan_info)。
        """
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        _training = self._training

        total_reward = 0.0
        if detailed:
            parts = {
                "time_cost": 0.0, "proc_reward": 0.0, "safe_reward": 0.0,
                "warn_penalty": 0.0, "penalty": 0.0, "wafer_done_bonus": 0.0,
                "finish_bonus": 0.0, "scrap_penalty": 0.0,
            }

        do_time_cost = self._do_time_cost
        do_proc_reward = self._do_proc_reward
        do_transport_penalty = self._do_transport_penalty
        do_warn_penalty = self._do_warn_penalty

        if do_time_cost:
            tc = -float(safe_dt * self.time_coef_penalty)
            total_reward += tc
            if detailed:
                parts["time_cost"] = tc

        is_scrap = False
        scrap_info: Optional[Dict[str, Any]] = None
        qtime_new_violations: List[int] = []
        qtime_limit = self.D_Residual_time
        p_residual = self.P_Residual_time
        proc_coef = self.processing_coef_reward
        transport_coef = self.transport_overtime_coef_penalty
        warn_coef = self.warn_coef_penalty
        check_qtime = not _training
        _SOURCE = SOURCE

        for p in self.marks:
            p_type = p.type
            tokens = p.tokens
            has_tok = len(tokens) > 0

            if has_tok and p_type == CHAMBER:
                head = tokens[0]
                head_stay = head.stay_time
                proc_time = p.processing_time
                if do_proc_reward and proc_time > 0:
                    remain = proc_time - head_stay
                    if remain < 0:
                        remain = 0
                    progress = safe_dt if safe_dt < remain else remain
                    r = proc_coef * float(progress)
                    total_reward += r
                    if detailed:
                        parts["proc_reward"] += r
                if do_warn_penalty:
                    left = proc_time + p_residual - head_stay
                    if left <= p_residual:
                        r = -(warn_coef * safe_dt)
                        total_reward += r
                        if detailed:
                            parts["warn_penalty"] += r
            elif has_tok and p_type == DELIVERY_ROBOT and do_transport_penalty:
                for tok in tokens:
                    deadline = tok.enter_time + qtime_limit
                    over_start = t1 if t1 > deadline else deadline
                    if t2 > over_start:
                        r = -float(t2 - over_start) * float(transport_coef)
                        total_reward += r
                        if detailed:
                            parts["penalty"] += r

            if p_type != _SOURCE and safe_dt > 0:
                for tok in tokens:
                    tok.stay_time += safe_dt

            if safe_dt > 0 and p.is_pm:
                if not has_tok:
                    p.idle_time += safe_dt
                else:
                    p.idle_time = 0
                if p.is_cleaning:
                    remaining_clean = p.cleaning_remaining - safe_dt
                    if remaining_clean < 0:
                        remaining_clean = 0
                    p.cleaning_remaining = remaining_clean
                    if remaining_clean == 0:
                        p.is_cleaning = False
                        p.cleaning_reason = ""
                        if not _training:
                            self.fire_log.append({
                                "event_type": "cleaning_end",
                                "time": int(self.time),
                                "chamber": p.name,
                                "processed_wafer_count": p.processed_wafer_count,
                            })

            if has_tok and (not is_scrap) and p_type == CHAMBER:
                head = tokens[0]
                post_stay = head.stay_time
                remaining_proc = p.processing_time - post_stay
                if remaining_proc < -p_residual:
                    overtime = -remaining_proc - p_residual
                    scrap_info = {
                        "token_id": head.token_id,
                        "place": p.name,
                        "stay_time": post_stay,
                        "proc_time": p.processing_time,
                        "overtime": overtime,
                        "type": "resident",
                    }
                    is_scrap = True
            if check_qtime and has_tok and p_type == DELIVERY_ROBOT:
                for tok in tokens:
                    tid = tok.token_id
                    if tid < 0 or tid in self._qtime_violated_tokens:
                        continue
                    if tok.stay_time > qtime_limit:
                        qtime_new_violations.append(tid)

        for tid in qtime_new_violations:
            if tid not in self._qtime_violated_tokens:
                self._qtime_violated_tokens.add(tid)
                self.qtime_violation_count += 1

        scan_info: Dict[str, Any] = {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
            "qtime_new_violations": qtime_new_violations,
        }
        self._last_state_scan = scan_info

        if detailed:
            parts["total"] = total_reward
            return parts, scan_info
        return total_reward, scan_info

    def _fire(self, t_idx: int, start_time: int, end_time: int) -> Dict[str, Any]:
        t_name = self.id2t_name[t_idx]
        pre_places = self._pre_place_indices[t_idx]
        pst_places = self._pst_place_indices[t_idx]
        if pre_places.size == 0 or pst_places.size == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
        pre_place = self.marks[int(pre_places[0])]
        pst_place = self.marks[int(pst_places[0])]
        if len(pre_place.tokens) == 0:
            return {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}

        tok = pre_place.pop_head()
        wafer_id = tok.token_id
        self._track_leave(tok, pre_place.name)
        tok.enter_time = self.time
        tok.stay_time = 0

        if t_name.startswith("u_"):
            src = pre_place.name
            tok._dst_level_targets = tuple(self._u_targets.get(src, []))
            tok._dst_level_full_on_pick = self._is_dst_level_full(src)
            if self.device_mode == "cascade" and self.route_code == 4 and src == "LLD":
                dst = self._select_route4_lld_target(tok)
            else:
                dst = self._select_target_for_source(src, advance_round_robin=True)
            tok._target_place = dst
            tok.machine = int(self._next_robot_machine())
            if self._is_cascade:
                transport = self._transport_for_t_target(dst) if dst else "d_TM2"
                tok.machine = 2 if transport == "d_TM3" else 1
            if src in self._chamber_active and wafer_id in self._chamber_active[src]:
                idx = self._chamber_active[src].pop(wafer_id)
                e, _, wid = self._chamber_timeline[src][idx]
                self._chamber_timeline[src][idx] = (e, start_time, wid)
            self._on_processing_unload(src)
        elif t_name.startswith("t_"):
            target = t_name[2:]
            tok._target_place = None
            tok._dst_level_targets = None
            tok._dst_level_full_on_pick = False
            tok.step = max(tok.step, self._step_map.get(target, 0))
            self._track_enter(tok, target)
            if target == "LP_done":
                self.entered_wafer_count = max(0, int(self.entered_wafer_count) - 1)
                self.done_count += 1
                self._per_wafer_reward += float(self.done_event_reward)
            elif target in self._chamber_timeline and wafer_id >= 0:
                idx = len(self._chamber_timeline[target])
                self._chamber_timeline[target].append((end_time, None, wafer_id))
                self._chamber_active[target][wafer_id] = idx

        self._advance_token_route_head(tok)
        pst_place.append(tok)
        pre_place_idx = int(pre_places[0])
        pst_place_idx = int(pst_places[0])
        tok._place_idx = pst_place_idx
        self.m[pre_place_idx] -= 1
        self.m[pst_place_idx] += 1
        if t_name == "u_LP":
            self.entered_wafer_count += 1
            self._last_u_LP_fire_time = int(start_time)
            self._u_LP_release_count += 1
        return {
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(end_time),
            "token_id": wafer_id,
        }

    def _should_cancel_resident_scrap_after_fire(
        self,
        scan: Dict[str, Any],
        log_entry: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(scan, dict) or not isinstance(log_entry, dict):
            return False
        if not bool(scan.get("is_scrap", False)):
            return False
        scrap_info = scan.get("scrap_info")
        if not isinstance(scrap_info, dict):
            return False
        if scrap_info.get("type") != "resident":
            return False
        t_name = str(log_entry.get("t_name", ""))
        if not t_name.startswith("u_"):
            return False
        source_name = t_name[2:]
        if scrap_info.get("place") != source_name:
            return False
        try:
            fired_token_id = int(log_entry.get("token_id", -1))
            scrap_token_id = int(scrap_info.get("token_id", -2))
        except (TypeError, ValueError):
            return False
        return fired_token_id >= 0 and fired_token_id == scrap_token_id

    @staticmethod
    def _token_route_gate(tok: BasedToken) -> object:
        queue = tok.route_queue
        if not queue:
            return -1
        idx = tok.route_head_idx
        if idx < 0:
            idx = 0
        n = len(queue)
        if idx >= n:
            idx = n - 1
        return queue[idx]

    @staticmethod
    def _advance_token_route_head(tok: BasedToken) -> None:
        queue = tok.route_queue
        if not queue:
            return
        next_idx = tok.route_head_idx + 1
        n = len(queue)
        if next_idx >= n:
            next_idx = n - 1
        tok.route_head_idx = max(0, next_idx)

    def _token_next_target(self, tok: BasedToken) -> Optional[str]:
        """从 token 的 route_queue 推断下一个目标腔室（用于多 transport 的 u_* 选择）。"""
        queue = tok.route_queue
        if not queue:
            return None
        idx = tok.route_head_idx + 1
        n = len(queue)
        while idx < n:
            gate = queue[idx]
            if gate == -1:
                idx += 1
                continue
            if isinstance(gate, int):
                t_name = self._t_code_to_target.get(gate)
                if t_name:
                    return t_name[2:] if t_name.startswith("t_") else t_name
                return None
            if isinstance(gate, (tuple, list, set, frozenset)) and gate:
                first = next(iter(gate))
                t_name = self._t_code_to_target.get(int(first))
                if t_name:
                    return t_name[2:] if t_name.startswith("t_") else t_name
                return None
            idx += 1
        return None

    @staticmethod
    def _route_gate_allows_t(gate: object, t_code: int) -> bool:
        if t_code < 0:
            return True
        if gate == -1:
            return True
        if isinstance(gate, int):
            return gate == t_code
        if isinstance(gate, (tuple, frozenset)):
            return t_code in gate
        if isinstance(gate, (list, set)):
            return t_code in gate
        return True

    def _route4_next_target_from_token(self, tok: BasedToken) -> Optional[str]:
        queue = tok.route_queue
        if not queue:
            return None
        next_idx = tok.route_head_idx + 1
        if next_idx < 0:
            next_idx = 0
        if next_idx >= len(queue):
            next_idx = len(queue) - 1
        next_gate = queue[next_idx]
        if not isinstance(next_gate, int):
            return None
        if next_gate == int(self._t_route_code_map.get("t_PM7", -999)):
            return "PM7"
        if next_gate == int(self._t_route_code_map.get("t_LP_done", -998)):
            return "LP_done"
        return None

    def _select_route4_lld_target(
        self,
        tok: BasedToken,
        ignore_cleaning: bool = False,
    ) -> Optional[str]:
        target = self._route4_next_target_from_token(tok)
        if target is None:
            return None
        candidates = self._u_targets.get("LLD", [])
        if target not in candidates:
            return None
        target_place = self._get_place(target)
        if (not ignore_cleaning) and target_place.is_cleaning:
            return None
        if len(target_place.tokens) >= target_place.capacity:
            return None
        return target

    def _get_enable_t(self) -> List[int]:
        """返回当前使能的变迁索引列表（仅 transition，供 reset 等使用）。"""
        enabled: set[int] = set()
        struct_enabled_cache: Dict[int, bool] = {}

        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(int(t_idx))
            if cached is not None:
                return bool(cached)
            result = bool(self._transition_structurally_enabled(int(t_idx)))
            struct_enabled_cache[int(t_idx)] = result
            return result

        # 按你的要求保持优先级：u_LP 最先检查（受 _allow_start 节拍门控）。
        u_lp_idx = self._u_transition_by_source.get("LP")
        if u_lp_idx is not None and (self._allow_start()):
            if _is_struct_enabled(u_lp_idx) and self._select_target_for_source("LP") is not None:
                enabled.add(int(u_lp_idx))

        for tok in self._token_pool:
            place_idx = tok._place_idx
            if place_idx < 0 or place_idx >= len(self.id2p_name):
                continue
            place_name = self.id2p_name[place_idx]
            if place_name in {"LP", "LP_done"}:
                continue

            # 剩余加工/停留时间 > 0 则不可放行。
            if self._token_remaining_time(tok, place_idx) > 0:
                continue

            # 运输位 token：尝试 t_*
            if place_name.startswith("d_TM"):
                for t_idx in self._t_transitions_by_transport.get(place_name, []):
                    t_name = self.id2t_name[t_idx]
                    target = t_name[2:]
                    target_hint = tok._target_place
                    if target_hint is not None and target_hint != target:
                        continue
                    if not self._route_gate_allows_t(self._token_route_gate(tok), self._t_route_code_by_idx[t_idx]):
                        continue
                    target_place = self._get_place(target)
                    if target_place.is_cleaning:
                        continue
                    if not _is_struct_enabled(t_idx):
                        continue
                    enabled.add(int(t_idx))
                continue

            # 加工腔/缓冲位 token：尝试 u_*
            target = self._token_next_target(tok)
            if target is not None:
                target_place = self._place_by_name.get(str(target))
                if (
                    target_place is None
                    or target_place.is_cleaning
                    or len(target_place.tokens) >= target_place.capacity
                ):
                    target = None
            if target is None:
                target = self._select_target_for_source(place_name)
            if target is None:
                continue
            transport = self._transport_for_t_target(target)
            u_idx = self._u_transition_by_source_transport.get((place_name, transport))
            if u_idx is None:
                u_idx = self._u_transition_by_source.get(place_name)
            if u_idx is None:
                continue
            struct_enabled = _is_struct_enabled(u_idx)
            if not struct_enabled:
                continue
            enabled.add(int(u_idx))

        return sorted(enabled)

    def _allow_start(self):
        """returns True if u_LP can fire now, based on WIP limit and takt."""
        if not self._allow_start_by_wip_limit():
            return False
        required = self._takt_required_interval()
        if required is None:
            return True
        return (self.time - self._last_u_LP_fire_time) >= int(required)

    def _allow_start_by_wip_limit(self) -> bool:
        return int(self.entered_wafer_count) < int(self.max_wafers_in_system)

    def _takt_required_interval(self) -> Optional[int]:
        """
        返回下一次允许 u_LP 发片的最小间隔（秒）。

        口径：
        - 首片（release_count=0）不门控，返回 None
        - 第 2 片（release_count=1）使用 cycle[0]（100 拍的第 1 个拍子）
        - 第 3 片起（release_count>=2）按序推进（idx=1,2,3...），必要时对 100 取模
        """
        takt = self._takt_result
        if not takt:
            return None
        release_count = int(self._u_LP_release_count)
        if release_count <= 0:
            return None

        cycle_takts = takt.get("cycle_takts") or []
        if not cycle_takts:
            return None
        cycle_len = int(takt.get("cycle_length") or len(cycle_takts))
        if cycle_len <= 0:
            return None

        if release_count == 1:
            idx = 0
        else:
            idx = (release_count - 1) % cycle_len

        required = cycle_takts[idx]
        if isinstance(required, float):
            required = int(round(required))
        else:
            required = int(required)
        if required < 0:
            required = 0
        return required

    def _build_transition_index(self) -> None:
        self._u_transition_by_source = {}
        self._u_transition_by_source_transport = {}
        self._t_transitions_by_transport = {}
        for t_idx, t_name in enumerate(self.id2t_name):
            if t_name.startswith("u_"):
                pre_idx = self._pre_place_indices[t_idx]
                pst_idx = self._pst_place_indices[t_idx]
                if pre_idx.size >= 1 and pst_idx.size >= 1:
                    src = self.id2p_name[int(pre_idx[0])]
                    dst = self.id2p_name[int(pst_idx[0])]
                    if dst.startswith("d_"):
                        self._u_transition_by_source_transport[(src, dst)] = int(t_idx)
                    self._u_transition_by_source[src] = int(t_idx)
            elif t_name.startswith("t_"):
                target = t_name[2:]
                transport = self._transport_for_t_target(target)
                self._t_transitions_by_transport.setdefault(transport, []).append(int(t_idx))

    def _rebuild_token_pool(self) -> None:
        pool: List[BasedToken] = []
        for place_idx, place in enumerate(self.marks):
            for tok in place.tokens:
                tok._place_idx = place_idx
                pool.append(tok)
        self._token_pool = pool

    def _token_remaining_time(self, tok: BasedToken, place_idx: int) -> int:
        place = self.marks[place_idx]
        if place.type in (CHAMBER, 5, DELIVERY_ROBOT):
            return place.processing_time - tok.stay_time
        return 0

    def _transition_structurally_enabled(self, t_idx: int) -> bool:
        m = self.m; pre = self.pre; net = self.net; k = self.k
        # 数值核心优先走 numba（可选），失败时退回原路径。
        try:
            return bool(step_core_numba(pre, m, k, self.pst, int(t_idx)))
        except Exception:
            pass
        pre_idx = self._pre_place_indices[t_idx]
        n = pre_idx.size
        if n == 1:
            i = int(pre_idx[0])
            if m[i] < pre[i, t_idx]:
                return False
        elif n == 2:
            i0 = int(pre_idx[0]); i1 = int(pre_idx[1])
            if m[i0] < pre[i0, t_idx] or m[i1] < pre[i1, t_idx]:
                return False
        elif n > 0:
            if np.any(m[pre_idx] < pre[pre_idx, t_idx]):
                return False
        pst_idx = self._pst_place_indices[t_idx]
        n = pst_idx.size
        if n == 1:
            j = int(pst_idx[0])
            if m[j] + net[j, t_idx] > k[j]:
                return False
        elif n == 2:
            j0 = int(pst_idx[0]); j1 = int(pst_idx[1])
            if m[j0] + net[j0, t_idx] > k[j0] or m[j1] + net[j1, t_idx] > k[j1]:
                return False
        elif n > 0:
            if np.any(m[pst_idx] + net[pst_idx, t_idx] > k[pst_idx]):
                return False
        return True

    def get_enable_actions_with_reasons(
        self,
        wait_action_start: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        返回使能动作列表及不使能动作及原因。
        仅在评估模式下被调用，避免训练时额外开销。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)

        d_tm = self._get_place("d_TM1") if ("d_TM1" in self.id2p_name and self.device_mode != "cascade") else None
        head_tok = d_tm.head() if (d_tm is not None and len(d_tm.tokens) > 0) else None
        locked_sources: Set[str] = set()
        if self.robot_capacity == 2 and self.device_mode != "cascade" and head_tok is not None:
            locked_sources = set(head_tok._dst_level_targets or ())

        enabled: List[int] = []
        disabled: List[Dict[str, Any]] = []

        # 遍历变迁（使用缓存的 pre/pst 索引）
        for t in range(self.T):
            base_pre_idx = self._pre_place_indices[t]
            t_name = self.id2t_name[t]

            if base_pre_idx.size == 0:
                disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                continue

            if np.any(self.m[base_pre_idx] < self.pre[base_pre_idx, t]):
                disabled.append({"action": t, "name": t_name, "reason": "insufficient_tokens"})
                continue
            if np.any(self.m + self.net[:, t] > self.k):
                disabled.append({"action": t, "name": t_name, "reason": "capacity_exceeded"})
                continue

            if t_name.startswith("u_"):
                src = t_name[2:]
                if t_name == "u_LP" and not self._allow_start_by_wip_limit():
                    disabled.append({"action": t, "name": t_name, "reason": "max_wafers_in_system_limit"})
                    continue
                if self.robot_capacity == 2 and self.device_mode != "cascade":
                    if d_tm is not None and len(d_tm.tokens) > 0 and locked_sources and src not in locked_sources:
                        disabled.append({"action": t, "name": t_name, "reason": "locked_by_arm2_head"})
                        continue
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                        continue
                else:
                    if self._select_target_for_source(src, ignore_cleaning=True) is None:
                        disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                        continue
            elif t_name.startswith("t_"):
                t_code = self._t_route_code_by_idx[t]
                tp_idx = self._transport_pre_place_idx[t]
                if tp_idx >= 0:
                    tp_head = self.marks[tp_idx].head() if len(self.marks[tp_idx].tokens) > 0 else None
                    if tp_head is not None:
                        gate = self._token_route_gate(tp_head)
                        if not self._route_gate_allows_t(gate, t_code):
                            disabled.append({"action": t, "name": t_name, "reason": "pre_color_mismatch"})
                            continue
                target = t_name[2:]
                transport_name = self._transport_for_t_target(target)
                tp = self._get_place(transport_name)
                tp_head = tp.head() if len(tp.tokens) > 0 else None
                tp_head_target = tp_head._target_place if tp_head is not None else None
                if tp_head_target is not None and tp_head_target != target:
                    disabled.append({"action": t, "name": t_name, "reason": "wrong_destination"})
                    continue

            # Stage2
            if t_name.startswith("u_"):
                src = t_name[2:]
                if not self._is_process_ready(src):
                    disabled.append({"action": t, "name": t_name, "reason": "process_not_ready"})
                    continue
                if self._select_target_for_source(src) is None:
                    disabled.append({"action": t, "name": t_name, "reason": "no_receiving_target"})
                    continue
            elif t_name.startswith("t_"):
                target = t_name[2:]
                target_place = self._get_place(target)
                if target_place.is_cleaning:
                    disabled.append({"action": t, "name": t_name, "reason": "target_cleaning"})
                    continue
                d_place = self._get_place(self._transport_for_t_target(target))
                dwell_time = max(0, int(getattr(d_place, "processing_time", self.T_transport)))
                if len(d_place.tokens) > 0 and d_place.head().stay_time < dwell_time:
                    disabled.append({"action": t, "name": t_name, "reason": "dwell_time_not_met"})
                    continue

            # 节拍限制：u_LP 发片间隔不得小于当前周期节拍
            if self._takt_result and t_name == "u_LP":
                required = self._takt_required_interval()
                if required is not None and (self.time - self._last_u_LP_fire_time) < int(required):
                    disabled.append({"action": t, "name": t_name, "reason": "takt_release_limit"})
                    continue

            enabled.append(t)

        # wait 动作
        restrict_long_wait = self._has_ready_chamber_wafers()
        for offset, duration in enumerate(self.wait_durations):
            action_idx = start + int(offset)
            if restrict_long_wait and int(duration) > 5:
                disabled.append({
                    "action": action_idx,
                    "name": f"WAIT_{int(duration)}s",
                    "reason": "has_ready_wafer_restrict_wait",
                })
            else:
                enabled.append(action_idx)

        return {"enabled": enabled, "disabled": disabled}

    def calc_wafer_statistics(self) -> Dict[str, Any]:
        system_times: List[float] = []
        chamber_stats: Dict[str, List[float]] = {}
        completed = 0
        for _, s in self._token_stats.items():
            enter = s.get("enter_system")
            leave = s.get("exit_system")
            if enter is not None and leave is not None:
                completed += 1
                system_times.append(float(leave - enter))
            for c_name, c_stat in s.get("chambers", {}).items():
                c_enter = c_stat.get("enter")
                c_leave = c_stat.get("exit")
                if c_enter is not None and c_leave is not None:
                    chamber_stats.setdefault(c_name, []).append(float(c_leave - c_enter))

        chamber_summary: Dict[str, Dict[str, float]] = {}
        for name, vals in chamber_stats.items():
            if vals:
                chamber_summary[name] = {"avg": sum(vals) / len(vals), "max": max(vals), "count": len(vals)}

        return {
            "system_avg": (sum(system_times) / len(system_times)) if system_times else 0.0,
            "system_max": max(system_times) if system_times else 0.0,
            "system_diff": 0.0,
            "completed_count": completed,
            "in_progress_count": max(0, self.n_wafer - completed),
            "chambers": chamber_summary,
            "transports": {},
            "transports_detail": {},
            "resident_violation_count": self.resident_violation_count,
            "qtime_violation_count": self.qtime_violation_count,
            "deadlock_count": self.deadlock_count,
            "chamber_processed_counts": {
                p.name: int(getattr(p, "processed_wafer_count", 0))
                for p in self.marks
                if p.name.startswith("PM")
            },
        }

    def _refresh_episode_proc_time(self) -> None:
        """生成本 episode 工序时长（随机+取整到5）并应用到 marks 与 ptime。"""
        if self.proc_rand_enabled:
            sampled: Dict[str, int] = {}
            for chamber in self.chambers:
                raw = self._proc_rand_scale_map.get(chamber, {})
                if not isinstance(raw, dict):
                    raw = {}
                low = float(raw.get("min", 1.0))
                high = float(raw.get("max", 1.0))
                base_time = float(self._base_proc_time_map[chamber])
                sampled_time = base_time * float(np.random.uniform(low, high))
                sampled[chamber] = _round_to_nearest_five(sampled_time)
            self._episode_proc_time_map = sampled
        else:
            self._episode_proc_time_map = dict(self._base_proc_time_map)
        self._validate_episode_proc_time_map_consistency()
        for p in self.marks:
            if p.name in self._episode_proc_time_map:
                p.processing_time = int(self._episode_proc_time_map[p.name])
        for chamber_name, proc_time in self._episode_proc_time_map.items():
            p_idx = self._get_place_index(chamber_name)
            self.ptime[p_idx] = int(proc_time)

    def _validate_episode_proc_time_map_consistency(self) -> None:
        expected = set(self.chambers)
        actual = set(self._episode_proc_time_map.keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if not missing and not extra:
            return
        raise ValueError(
            "episode process time map is inconsistent with route chambers: "
            f"device_mode={self.device_mode!r}, route_code={self.route_code!r}, "
            f"missing={missing}, extra={extra}, "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )

    def _align_base_proc_time_map_with_route_chambers(self) -> None:
        """
        配置驱动下 route_meta 与预处理口径可能暂时不一致（如 mixed stage: LLC/LLD），
        这里按已构网 place 工时补齐缺失键，保证 episode map 与 route chambers 一致。
        """
        missing = [name for name in self.chambers if name not in self._base_proc_time_map]
        if not missing:
            return
        for chamber_name in missing:
            if chamber_name not in self.id2p_name:
                continue
            p_idx = self._get_place_index(chamber_name)
            self._base_proc_time_map[chamber_name] = int(self.ptime[p_idx])

    def _preprocess_process_time_map(self, process_time_map: Dict[str, int]) -> Dict[str, int]:
        if getattr(self, "single_route_config", None) is not None:
            cfg_chambers = dict((self.single_route_config or {}).get("chambers") or {})
            defaults = {
                name: int((spec or {}).get("process_time", 0))
                for name, spec in cfg_chambers.items()
                if name in self.chambers
            }
            missing_defaults = sorted(
                chamber
                for chamber in self.chambers
                if chamber not in process_time_map and chamber not in defaults
            )
            if missing_defaults:
                raise ValueError(
                    "missing default process times for chambers: "
                    f"{missing_defaults}; device_mode={self.device_mode!r}, route_code={self.route_code!r}"
                )
            return _preprocess_process_time_map(
                process_time_map=process_time_map,
                chambers=self.chambers,
                defaults=defaults,
            )

        if self.device_mode == "cascade":
            if self.route_code == 4:
                defaults = {
                    "PM7": 70,
                    "PM8": 70,
                    "LLD": 70,
                }
            elif self.route_code == 5:
                defaults = {
                    "PM7": 70,
                    "PM8": 70,
                    "PM9": 200,
                    "PM10": 200,
                }
            else:
                pm_stage3_default = 600 if self.route_code == 1 else 300
                defaults = {
                    "PM7": 70,
                    "PM8": 70,
                    "PM1": pm_stage3_default,
                    "PM2": pm_stage3_default,
                    "LLD": 70,
                }
                if self.route_code != 3:
                    defaults.update({"PM9": 200, "PM10": 200})
                if self.route_code == 1:
                    defaults.update({"PM3": pm_stage3_default, "PM4": pm_stage3_default})
        else:
            defaults = {"PM1": 100, "PM3": 300, "PM4": 300, "PM6": 300}
        missing_defaults = sorted(
            chamber
            for chamber in self.chambers
            if chamber not in process_time_map and chamber not in defaults
        )
        if missing_defaults:
            raise ValueError(
                "missing default process times for chambers: "
                f"{missing_defaults}; device_mode={self.device_mode!r}, route_code={self.route_code!r}"
            )
        return _preprocess_process_time_map(
            process_time_map=process_time_map,
            chambers=self.chambers,
            defaults=defaults,
        )

    # 节拍分析器内部会统一给每道工序 p 加运输时间常量（当前口径为 +20）
    _TRANSPORT_TIME_FOR_TAKT: int = 20

    def _build_takt_stage(self, stage_idx: int, stage_places: List[str]) -> Optional[Dict[str, Any]]:
        """
        将一层 route stage 归一化为 analyzer 输入。
        - 并行 stage 的 p 取该层瓶颈（max）
        - 清洗参数优先取该层中最可能形成慢节拍的腔室（max(p+d)）
        """
        valid_places = [
            place
            for place in stage_places
            if int(self._episode_proc_time_map.get(place, 0) or 0) > 0
        ]
        if not valid_places:
            return None
        base_p = max(int(self._episode_proc_time_map[place]) for place in valid_places)
        q: Optional[int] = None
        d = 0
        if self.cleaning_enabled:
            cleaning_candidates: List[Tuple[int, int, int, str]] = []
            for place in valid_places:
                trigger = int(self._cleaning_trigger_map.get(place, 0))
                if trigger <= 0:
                    continue
                duration = int(self._cleaning_duration_map.get(place, self.cleaning_duration))
                score = int(self._episode_proc_time_map.get(place, 0)) + duration
                cleaning_candidates.append((score, trigger, duration, place))
            if cleaning_candidates:
                _, q, d, _ = max(cleaning_candidates, key=lambda item: (item[0], item[3]))
        return {
            "name": f"s{stage_idx + 1}",
            "p": int(base_p),
            "m": len(valid_places),
            "q": q,
            "d": int(d),
        }

    def _validate_takt_stage_inputs(self) -> None:
        stage_details = [
            f"s{idx + 1}={list(stage)}" for idx, stage in enumerate(self._route_stages)
        ]
        stage_place_set: Set[str] = set()
        for stage in self._route_stages:
            for place in stage:
                stage_place_set.add(str(place))

        unknown_stage_places = sorted(
            place for place in stage_place_set if place not in self._step_map
        )
        if unknown_stage_places:
            raise ValueError(
                "route stage contains places not in step map: "
                f"{unknown_stage_places}; device_mode={self.device_mode!r}, "
                f"route_code={self.route_code!r}, stages={stage_details}"
            )

        proc_places = set(self._episode_proc_time_map.keys())
        out_of_stage_proc_places = sorted(proc_places - stage_place_set)
        if out_of_stage_proc_places:
            raise ValueError(
                "takt stage mismatch: process_time map has out-of-route places: "
                f"{out_of_stage_proc_places}; device_mode={self.device_mode!r}, "
                f"route_code={self.route_code!r}, stages={stage_details}, "
                f"process_time_places={sorted(proc_places)}"
            )

        missing_proc_places = sorted(set(self.chambers) - proc_places)
        if missing_proc_places:
            raise ValueError(
                "takt stage mismatch: route chambers missing from process_time map: "
                f"{missing_proc_places}; device_mode={self.device_mode!r}, "
                f"route_code={self.route_code!r}, stages={stage_details}, "
                f"process_time_places={sorted(proc_places)}"
            )

    def _compute_takt_result(self) -> Optional[Dict[str, Any]]:
        """
        根据当前加工配方（路线 + 工序时长 + 清洗参数）调用节拍分析器，
        analyzer 内部会统一把每道工序处理时间按「工序时长 + 运输时间」计入节拍。
        失败或无可分析工序时返回 None。
        """
        self._validate_takt_stage_inputs()
        if self.device_mode == "cascade" and self.route_code == 4:
            if self.route4_takt_interval > 0:
                return build_fixed_takt_result(self.route4_takt_interval)
            return None

        analyzer_stages: List[Dict[str, Any]] = []
        for i, stage in enumerate(self._route_stages):
            if not stage:
                continue
            stage_cfg = self._build_takt_stage(stage_idx=i, stage_places=list(stage))
            if stage_cfg is None:
                continue
            analyzer_stages.append(stage_cfg)
        if not analyzer_stages:
            return None
        try:
            return analyze_cycle(analyzer_stages, max_parts=10000)
        except Exception:
            return None

    @staticmethod
    def _clone_marks(marks: List[Place]) -> List[Place]:
        cloned: List[Place] = []
        for p in marks:
            cp = p.clone()
            cloned.append(cp)
        return cloned

    def _get_place(self, name: str) -> Place:
        place = self._place_by_name.get(name)
        if place is None:
            raise KeyError(f"unknown place: {name}")
        return place

    def _get_place_index(self, name: str) -> int:
        return self.id2p_name.index(name)

    def _transport_for_t_target(self, target: str) -> str:
        if self.device_mode != "cascade":
            return "d_TM1"
        # 优先按实际构网弧推断 transport，避免对 LLD 等目标做硬编码误判。
        t_name = f"t_{target}"
        if t_name in self.id2t_name:
            t_idx = int(self.id2t_name.index(t_name))
            for p_i in self._pre_place_indices[t_idx]:
                p_name = self.id2p_name[int(p_i)]
                if p_name.startswith("d_"):
                    return p_name
        if target in {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6"}:
            return "d_TM3"
        return "d_TM2"

    def _update_stay_times(self, dt: int) -> None:
        if dt <= 0:
            return
        for p in self.marks:
            if p.type == SOURCE:  # 跳过 LP 中的 wafer
                continue
            for tok in p.tokens:
                tok.stay_time += dt

    def _init_cleaning_state(self) -> None:
        for p in self.marks:
            if not p.name.startswith("PM"):
                continue
            p.processed_wafer_count = p.processed_wafer_count
            p.idle_time = p.idle_time
            p.last_proc_type = p.last_proc_type
            p.is_cleaning = p.is_cleaning
            p.cleaning_remaining = p.cleaning_remaining
            p.cleaning_reason = p.cleaning_reason

    def _advance_cleaning_and_idle(self, dt: int) -> None:
        if dt <= 0:
            return
        for p in self.marks:
            if not p.is_pm:
                continue
            if len(p.tokens) == 0:
                p.idle_time += dt
            else:
                p.idle_time = 0

            if not p.is_cleaning:
                continue
            remaining = max(0, p.cleaning_remaining - dt)
            p.cleaning_remaining = remaining
            if remaining == 0:
                p.is_cleaning = False
                p.cleaning_reason = ""
                if not self._training:
                    self.fire_log.append({
                        "event_type": "cleaning_end",
                        "time": int(self.time),
                        "chamber": p.name,
                        "processed_wafer_count": p.processed_wafer_count,
                    })

    def get_next_event_delta(self) -> Optional[int]:
        """
        计算当前时刻到下一个关键事件的时间差（秒）。
        遍历 marks 而非 _token_pool，避免 _place_idx 反查开销。
        """
        best = None
        t_transport = self.T_transport
        _CHAMBER_TYPES = (CHAMBER, 5)
        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            if place.is_dtm:
                for tok in tokens:
                    delta = t_transport - tok.stay_time
                    if delta < 0:
                        delta = 0
                    if best is None or delta < best:
                        best = delta
            elif place.type in _CHAMBER_TYPES:
                ptime = place.processing_time
                if ptime > 0:
                    head = tokens[0]
                    delta = ptime - head.stay_time
                    if delta < 0:
                        delta = 0
                    if best is None or delta < best:
                        best = delta
        required = self._takt_required_interval()
        if required is not None:
            delta_takt = self._last_u_LP_fire_time + int(required) - self.time
            if delta_takt > 0:
                if best is None or delta_takt < best:
                    best = delta_takt
        return best

    def _has_ready_chamber_wafers(self) -> bool:
        """
        判断是否存在“加工完成待取片”晶圆。
        规则：在当前路径定义的任一加工腔室中，存在 token 满足 stay_time >= processing_time。
        """
        for chamber_name in self._ready_chambers:
            place = self._place_by_name.get(chamber_name)
            if place is None:
                continue
            processing_time = place.processing_time
            if processing_time <= 0:
                continue
            for tok in place.tokens:
                if tok.stay_time >= processing_time:
                    return True
        return False

    def _scan_runtime_state(self) -> Dict[str, Any]:
        """
        扫描当前 token 状态并返回 scrap 与 qtime 统计结果。
        该扫描在 step 内由 advance_time 调用，避免重复遍历 token。
        """
        scrap_info: Optional[Dict[str, Any]] = None
        is_scrap = False
        qtime_new_violations: List[int] = []
        n_places = len(self.marks)
        qtime_limit = int(self.D_Residual_time)

        for tok in self._token_pool:
            place_idx = tok._place_idx
            if place_idx < 0 or place_idx >= n_places:
                continue
            place = self.marks[place_idx]
            stay_time = tok.stay_time

            if (not is_scrap) and place.type == CHAMBER:
                remaining = place.processing_time - stay_time
                if remaining < -self.P_Residual_time:
                    overtime = -remaining - self.P_Residual_time
                    scrap_info = {
                        "token_id": tok.token_id,
                        "place": place.name,
                        "stay_time": stay_time,
                        "proc_time": place.processing_time,
                        "overtime": overtime,
                        "type": "resident",
                    }
                    is_scrap = True

            if self._training:
                continue
            if place.type != DELIVERY_ROBOT:
                continue
            token_id = tok.token_id
            if token_id < 0 or token_id in self._qtime_violated_tokens:
                continue
            if stay_time > qtime_limit:
                qtime_new_violations.append(token_id)

        return {
            "is_scrap": is_scrap,
            "scrap_info": scrap_info,
            "qtime_new_violations": qtime_new_violations,
        }

    def advance_time(self, dt: int, event_reason: str = "") -> Dict[str, Any]:
        """推进时间并更新相关状态"""
        _ = event_reason  # 预留参数，便于后续追踪不同推进原因。
        safe_dt = max(0, int(dt))
        self.time += safe_dt
        self._update_stay_times(safe_dt)
        self._advance_cleaning_and_idle(safe_dt)
        scan_info = self._scan_runtime_state()
        for token_id in scan_info.get("qtime_new_violations", []):
            if token_id in self._qtime_violated_tokens:
                continue
            self._qtime_violated_tokens.add(token_id)
            self.qtime_violation_count += 1
        self._last_state_scan = scan_info
        return scan_info

    def _on_processing_unload(self, source_name: str) -> None:
        if not self.cleaning_enabled:
            return
        trigger = self._cleaning_trigger_map.get(source_name, 0)
        if trigger <= 0:
            return
        source_place = self._get_place(source_name)
        source_place.processed_wafer_count = source_place.processed_wafer_count + 1
        source_place.last_proc_type = source_name
        if source_place.is_cleaning:
            return
        if source_place.processed_wafer_count >= trigger:
            count = int(source_place.processed_wafer_count)
            duration = self._cleaning_duration_map.get(source_name, self.cleaning_duration)
            source_place.is_cleaning = True
            source_place.cleaning_remaining = int(duration)
            source_place.cleaning_reason = "processed_wafers"
            source_place.processed_wafer_count = 0
            source_place.idle_time = 0
            self.fire_log.append(
                {
                    "event_type": "cleaning_start",
                    "time": int(self.time),
                    "chamber": source_place.name,
                    "rule": "processed_wafers",
                    "duration": int(duration),
                    "trigger_count": int(count),
                }
            )

    def _is_process_ready(self, place_name: str) -> bool:
        place = self._get_place(place_name)
        if len(place.tokens) == 0:
            return False
        if place.processing_time <= 0:
            return True
        return place.head().stay_time >= place.processing_time

    def _select_target_for_source(
        self,
        source: str,
        preferred_target: Optional[str] = None,
        ignore_cleaning: bool = False,
        advance_round_robin: bool = False,
    ) -> Optional[str]:
        """
        为 u_<source> 选择一个可接收目标（确定性顺序）。
        仅检查目标腔室容量，运输位停留时间约束仍由 t_* 侧控制。
        """
        candidates = self._u_targets.get(source, [])
        if self.device_mode == "cascade" and source in {"PM7", "PM8"}:
            # LLC 满时仍允许从 PM7/PM8 取片到 d_TM1，后续由 t_LLC 容量约束拦截放片时机。
            # 这与用户要求一致：LLC 有片时 u_PM7/u_PM8 仍应可使能。
            if "LLC" in candidates:
                return "LLC"
        if self.device_mode == "cascade" and source in {"PM1", "PM2", "PM3", "PM4"}:
            # LLD 满时仍允许从 PM1-4 先取片到 d_TM1，后续由 t_LLD 实际容量约束放行。
            # 需求：LLD 有晶圆时，u_PM1/u_PM2/u_PM3/u_PM4 仍应保持可使能。
            if "LLD" in candidates:
                return "LLD"
        if self.device_mode == "cascade" and self.route_code == 4 and source == "LLD":
            lld_place = self._get_place("LLD")
            if len(lld_place.tokens) == 0:
                return None
            target = self._select_route4_lld_target(
                lld_place.head(),
                ignore_cleaning=ignore_cleaning,
            )
            if preferred_target is not None and target != preferred_target:
                return None
            return target
        if self.device_mode == "cascade" and source in self._cascade_round_robin_pairs and preferred_target is None:
            # 级联并行目标采用轮换分配；仅在真实发射时推进轮换指针，避免使能检查污染状态。
            rr_targets = list(self._cascade_round_robin_pairs[source])
            available_targets: List[str] = []
            for target in rr_targets:
                if target not in candidates:
                    continue
                target_place = self._get_place(target)
                if (not ignore_cleaning) and target_place.is_cleaning:
                    continue
                if len(target_place.tokens) < target_place.capacity:
                    available_targets.append(target)
            if available_targets:
                next_target = self._cascade_round_robin_next.get(source, rr_targets[0])
                start_idx = rr_targets.index(next_target) if next_target in rr_targets else 0
                chosen_target = available_targets[0]
                for offset in range(len(rr_targets)):
                    candidate = rr_targets[(start_idx + offset) % len(rr_targets)]
                    if candidate in available_targets:
                        chosen_target = candidate
                        break
                if advance_round_robin and len(rr_targets) > 1:
                    chosen_idx = rr_targets.index(chosen_target)
                    self._cascade_round_robin_next[source] = rr_targets[
                        (chosen_idx + 1) % len(rr_targets)
                    ]
                return chosen_target
        if self.device_mode == "single" and source in self._single_round_robin_pairs and preferred_target is None:
            # single 模式并行目标采用轮换分配，避免持续偏置到候选列表中的第一个目标。
            rr_targets = list(self._single_round_robin_pairs[source])
            available_targets: List[str] = []
            for target in rr_targets:
                if target not in candidates:
                    continue
                target_place = self._get_place(target)
                if (not ignore_cleaning) and target_place.is_cleaning:
                    continue
                if len(target_place.tokens) < target_place.capacity:
                    available_targets.append(target)
            if len(available_targets) == 2:
                next_target = self._single_round_robin_next.get(source, available_targets[0])
                chosen_target = next_target if next_target in available_targets else available_targets[0]
                if advance_round_robin:
                    self._single_round_robin_next[source] = (
                        available_targets[1] if chosen_target == available_targets[0] else available_targets[0]
                    )
                return chosen_target
            if len(available_targets) == 1:
                return available_targets[0]
        if preferred_target is not None:
            if preferred_target not in candidates:
                return None
            target_place = self._get_place(preferred_target)
            if (not ignore_cleaning) and target_place.is_cleaning:
                return None
            if len(target_place.tokens) < target_place.capacity:
                return preferred_target
            return None
        for target in candidates:
            target_place = self._get_place(target)
            if (not ignore_cleaning) and target_place.is_cleaning:
                continue
            if len(target_place.tokens) < target_place.capacity:
                return target
        return None

    def _next_robot_machine(self) -> int:
        if self.robot_capacity <= 1:
            return 1
        machine_id = self._next_machine_id
        self._next_machine_id = 2 if machine_id == 1 else 1
        return machine_id

    def _is_dst_level_full(self, source: str) -> bool:
        targets = self._u_targets.get(source, [])
        if not targets:
            return False
        for target in targets:
            p = self._get_place(target)
            if len(p.tokens) < p.capacity:
                return False
        return True

    def _check_scrap(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        for tok in self._token_pool:
            place_idx = tok._place_idx
            if place_idx < 0 or place_idx >= len(self.marks):
                continue
            p = self.marks[place_idx]
            if p.type != CHAMBER:
                continue
            remaining = p.processing_time - tok.stay_time
            if remaining < -self.P_Residual_time:
                overtime = -remaining - self.P_Residual_time
                return True, {
                    "token_id": tok.token_id,
                    "place": p.name,
                    "stay_time": tok.stay_time,
                    "proc_time": p.processing_time,
                    "overtime": overtime,
                    "type": "resident",
                }
        return False, None

    def _check_qtime_violation(self) -> None:
        """检查运输位 Q-Time 超时并按 wafer 去重累计，不施加惩罚。"""
        if self._training:
            return
        qtime_limit = int(self.D_Residual_time)
        for place in self.marks:
            if place.type != DELIVERY_ROBOT or len(place.tokens) == 0:
                continue
            for tok in place.tokens:
                token_id = tok.token_id
                if token_id < 0 or token_id in self._qtime_violated_tokens:
                    continue
                if tok.stay_time > qtime_limit:
                    self._qtime_violated_tokens.add(token_id)
                    self.qtime_violation_count += 1

    def blame_release_violations(self) -> Dict[int, float]:
        """
        事后追责：基于当前 fire_log 与 _chamber_timeline，回溯可能导致下游容量冲突的 u_* 动作。
        单设备中会按路径代号聚合站点：s1=PM1，s2=PM3∪PM4；若 code=1 则新增 s3=PM6。
        返回 fire_log_index -> penalty。
        """
        blame: Dict[int, float] = {}
        assert len(self.fire_log) > 0, "fire_log is empty"

        # 1) 预处理：构建占用时间线（enter, leave, wafer_id），leave 可能为 None 表示仍在加工中。
        proc_times = {p.name: p.processing_time for p in self.marks}
        capacities = {p.name: p.capacity for p in self.marks}

        def build_intervals(chamber_name: str) -> List[tuple]:
            intervals: List[tuple] = []
            for (enter, leave, wid) in self._chamber_timeline.get(chamber_name, []):
                l = leave if leave is not None else enter + proc_times.get(chamber_name, 0)
                intervals.append((enter, l, wid))
            intervals.sort(key=lambda x: x[0])
            return intervals

        # 从 fire_log 提取清洁区间（腔室在清洁期间占位，wid=-999 表示清洁占位）
        def build_cleaning_intervals(chamber_name: str) -> List[tuple]:
            if not getattr(self, "cleaning_enabled", False) or int(self._cleaning_trigger_map.get(chamber_name, 0)) <= 0:
                return []
            default_dur = self._cleaning_duration_map.get(chamber_name, self.cleaning_duration)
            out: List[tuple] = []
            for ev in self.fire_log:
                if ev.get("event_type") == "cleaning_start" and ev.get("chamber") == chamber_name:
                    t0 = int(ev.get("time", 0))
                    dur = int(ev.get("duration", default_dur))
                    out.append((t0, t0 + dur, -999))
            return out

        intervals_by_station: Dict[str, List[tuple]] = {}
        capacity_by_station: Dict[str, int] = {}
        proc_time_by_station: Dict[str, int] = {}
        for station, chambers in self._release_station_aliases.items():
            merged: List[tuple] = []
            for chamber_name in chambers:
                merged.extend(build_intervals(chamber_name))
                merged.extend(build_cleaning_intervals(chamber_name))
            merged.sort(key=lambda x: x[0])
            intervals_by_station[station] = merged
            capacity_by_station[station] = int(sum(capacities.get(name, 0) for name in chambers))
            proc_time_by_station[station] = int(max((proc_times.get(name, 0) for name in chambers), default=0))

        edge_transfer = self.T_transport + self.T_load

        def will_exceed_capacity(intervals, at_time, cap, current_wid):
            occupied = sum(1 for (e, l, wid0) in intervals if e <= at_time < l and wid0 < current_wid)
            return occupied + 1 > cap

        # 单设备 downstream chain：按路径代号切换（code0: s1->s2, code1: s1->s2->s3）
        chain_map: Dict[str, List[str]] = dict(self._release_chain_by_u)

        # 2) 回溯 fire_log，针对每个 u_* 动作检查其 downstream chain 是否存在容量冲突。
        # 仅追责释放动作：u_LP（从 LP 释放）、u_LLC（从 LLC 释放）、u_LLD（从 LLD 释放）。
        # u_PM7、u_PM2 等从加工腔卸载的动作不追责。
        RELEASE_BLAME_WHITELIST = frozenset({"u_LP", "u_LLC"})
        penalty_coeff = self.release_event_penalty
        for i, log in enumerate(self.fire_log):
            t_name = log.get("t_name", "")
            wid = int(log.get("token_id", -1))
            if t_name not in RELEASE_BLAME_WHITELIST:
                continue
            # 必须有合法的 wafer_id
            if wid < 0:
                continue
            chain = chain_map.get(t_name, [])
            if not chain:
                continue

            t_leave = int(log.get("t1", 0))
            arrival = t_leave + edge_transfer
            violated = False
            for idx, station in enumerate(chain):
                intervals = intervals_by_station.get(station, [])
                cap = capacity_by_station.get(station, 1)
                exceeds = will_exceed_capacity(intervals, arrival, cap, wid)
                if exceeds:
                    violated = True
                    break
                if idx < len(chain) - 1:
                    arrival = arrival + proc_time_by_station.get(station, 0) + edge_transfer

            if violated:
                blame[i] = penalty_coeff
        return blame

    _MASK_SKIP_PLACES = frozenset({"LP", "LP_done"})
    _MASK_TIMED_TYPES = frozenset((CHAMBER, 5, DELIVERY_ROBOT))

    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """
        返回完整离散动作掩码（transition + wait）。
        遍历 marks（而非 _token_pool），内联 remaining_time / has_ready_chamber 检查。
        """
        start = int(self.T if wait_action_start is None else wait_action_start)
        total_actions = int(
            n_actions if n_actions is not None else (start + len(self.wait_durations))
        )
        mask = np.zeros(total_actions, dtype=bool)
        struct_enabled_cache: Dict[int, bool] = {}

        def _is_struct_enabled(t_idx: int) -> bool:
            cached = struct_enabled_cache.get(t_idx)
            if cached is not None:
                return cached
            result = bool(self._transition_structurally_enabled(t_idx))
            struct_enabled_cache[t_idx] = result
            return result

        u_lp_idx = self._u_transition_by_source.get("LP")
        if u_lp_idx is not None and self._allow_start():
            if _is_struct_enabled(u_lp_idx) and self._select_target_for_source("LP") is not None:
                t_idx = int(u_lp_idx)
                if 0 <= t_idx < total_actions:
                    mask[t_idx] = True

        has_ready_chamber = False
        ready_chambers = self._ready_chambers_set
        target_cache: Dict[str, Optional[str]] = {}
        skip = self._MASK_SKIP_PLACES
        timed = self._MASK_TIMED_TYPES
        t_trans_by_transport = self._t_transitions_by_transport
        u_trans_by_source = self._u_transition_by_source
        id2t = self.id2t_name
        route_code_by_idx = self._t_route_code_by_idx
        route_gate_allows = self._route_gate_allows_t
        token_route_gate = self._token_route_gate
        place_by_name = self._place_by_name
        _SENTINEL = object()

        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            pname = place.name
            if pname in skip:
                continue
            p_type = place.type
            proc_time = place.processing_time

            is_timed = p_type in timed

            if place.is_dtm:
                for tok in tokens:
                    if is_timed and proc_time > 0 and tok.stay_time < proc_time:
                        continue
                    tok_gate = token_route_gate(tok)
                    target_hint = tok._target_place
                    for t_idx in t_trans_by_transport.get(pname, ()):
                        target = id2t[t_idx][2:]
                        if target_hint is not None and target_hint != target:
                            continue
                        if not route_gate_allows(tok_gate, route_code_by_idx[t_idx]):
                            continue
                        target_place = place_by_name.get(target)
                        if target_place is None or target_place.is_cleaning:
                            continue
                        if not _is_struct_enabled(t_idx):
                            continue
                        if 0 <= t_idx < total_actions:
                            mask[t_idx] = True
            else:
                head = tokens[0]
                if is_timed and proc_time > 0 and head.stay_time < proc_time:
                    continue
                route_target = self._token_next_target(head)
                target = route_target
                if target is not None:
                    target_place = place_by_name.get(str(target))
                    if (
                        target_place is None
                        or target_place.is_cleaning
                        or len(target_place.tokens) >= target_place.capacity
                    ):
                        target = None
                if target is None:
                    cached_target = target_cache.get(pname, _SENTINEL)
                    if cached_target is _SENTINEL:
                        cached_target = self._select_target_for_source(pname)
                        target_cache[pname] = cached_target
                    target = cached_target
                if target is not None:
                    transport = self._transport_for_t_target(target)
                    u_idx = self._u_transition_by_source_transport.get((pname, transport))
                    if u_idx is None:
                        u_idx = u_trans_by_source.get(pname)
                    struct_enabled = bool(u_idx is not None and _is_struct_enabled(u_idx))
                    if u_idx is not None and struct_enabled:
                        if 0 <= u_idx < total_actions:
                            mask[u_idx] = True

            if not has_ready_chamber and p_type == CHAMBER and proc_time > 0 and pname in ready_chambers:
                for tok in tokens:
                    if tok.stay_time >= proc_time:
                        has_ready_chamber = True
                        break

        wait_durations = self.wait_durations
        for offset in range(len(wait_durations)):
            if has_ready_chamber and wait_durations[offset] > 5:
                continue
            idx = start + offset
            if 0 <= idx < total_actions:
                mask[idx] = True

        return mask

    def _track_enter(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            self._token_stats[token.token_id] = {"enter_system": None, "exit_system": None, "chambers": {}}
        if place_name in self._system_entry_places and self._token_stats[token.token_id]["enter_system"] is None:
            self._token_stats[token.token_id]["enter_system"] = self.time
        if place_name == "LP_done":
            self._token_stats[token.token_id]["exit_system"] = self.time
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": self.time, "exit": None})

    def _track_leave(self, token: BasedToken, place_name: str) -> None:
        if token.token_id not in self._token_stats:
            return
        if place_name.startswith("PM"):
            self._token_stats[token.token_id]["chambers"].setdefault(place_name, {"enter": None, "exit": None})["exit"] = self.time

    def render_gantt(self, out_path: str) -> None:
        timelines = getattr(self, "_chamber_timeline", None)
        if not isinstance(timelines, dict) or not timelines:
            print("警告: _chamber_timeline 为空，跳过甘特图绘制")
            return

        # 以 _route_stages 定义 stage 与并行 machine（stage=1..S, machine=0..M-1）
        stage_map: Dict[str, Tuple[int, int]] = {}
        stage_module_names: Dict[int, List[str]] = {}
        proc_time: Dict[int, float] = {}
        capacity: Dict[int, int] = {}

        route_stages: List[List[str]] = list(getattr(self, "_route_stages", []) or [])
        for stage_idx, stage_places in enumerate(route_stages, start=1):
            names = [str(x) for x in (stage_places or [])]
            if not names:
                continue
            stage_module_names[int(stage_idx)] = names
            capacity[int(stage_idx)] = int(len(names))
            max_p = 0
            for machine_idx, name in enumerate(names):
                stage_map[name] = (int(stage_idx), int(machine_idx))
                place = getattr(self, "_place_by_name", {}).get(name)
                ptime = int(getattr(place, "processing_time", 0)) if place is not None else 0
                if ptime > max_p:
                    max_p = ptime
            proc_time[int(stage_idx)] = float(max_p)

        if not stage_map:
            print("警告: _route_stages 为空，无法构建 stage 映射，跳过甘特图绘制")
            return

        now_t = float(getattr(self, "time", 0))
        ops: List[Op] = []
        jobs: Set[int] = set()

        for chamber_name, items in timelines.items():
            if chamber_name not in stage_map:
                continue
            if not isinstance(items, list):
                continue
            stage, machine = stage_map[chamber_name]
            place = getattr(self, "_place_by_name", {}).get(chamber_name)
            ptime = float(getattr(place, "processing_time", 0)) if place is not None else 0.0

            for entry in items:
                try:
                    enter, leave, wid = entry
                except Exception:
                    continue
                try:
                    wid_int = int(wid)
                except Exception:
                    continue
                if wid_int < 0:
                    continue
                try:
                    start = float(enter)
                except Exception:
                    continue
                end = now_t
                if leave is not None:
                    try:
                        end = float(leave)
                    except Exception:
                        end = now_t
                if end < start:
                    continue

                proc_end = start + max(0.0, float(ptime))
                if proc_end > end:
                    proc_end = end

                ops.append(
                    Op(
                        job=wid_int,
                        stage=int(stage),
                        machine=int(machine),
                        start=float(start),
                        proc_end=float(proc_end),
                        end=float(end),
                    )
                )
                jobs.add(wid_int)

        if not ops:
            print("警告: 没有可绘制的腔室 Op 数据（可能尚未进入任何腔室），跳过甘特图绘制")
            return

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        final_path = out.parent / "gantt.png"
        base_path = out.with_suffix("")

        n_jobs = max(1, len(jobs))
        arm_info = {"ARM1": [], "ARM2": [], "STAGE2ACT": {}}
        policy = 2  # plot.py 映射为 RL1

        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=proc_time,
            capacity=capacity,
            n_jobs=int(n_jobs),
            out_path=str(base_path),
            arm_info=arm_info,
            with_label=True,
            no_arm=True,
            policy=int(policy),
            stage_module_names=stage_module_names,
        )

        generated_path = Path(f"{base_path}RL1_job{int(n_jobs)}.png")
        if generated_path.exists():
            try:
                final_path.write_bytes(generated_path.read_bytes())
            except Exception:
                import shutil
                shutil.copyfile(str(generated_path), str(final_path))
        else:
            print(f"警告: 未找到绘图输出文件: {generated_path}")
