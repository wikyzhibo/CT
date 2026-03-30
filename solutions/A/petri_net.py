

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from pathlib import Path
from solutions.A.utils import _normalize_wait_durations

import numpy as np
try:
    from numba import njit, prange
except Exception:  # pragma: no cover - numba 可选依赖
    njit = None
    prange = range

from config.cluster_tool.env_config import PetriEnvConfig
from solutions.A.construct import BasedToken
from solutions.A.model_builder import build_net
from solutions.A.construct.route_compiler_single import normalize_route_spec
from solutions.A.deprecated.pn import Place
from solutions.A.takt_analysis import (
    TAKT_HORIZON,
    analyze_cycle,
    build_fixed_takt_result,
)
from visualization.plot import Op, plot_gantt_hatched_residence

CHAMBER = 1
DELIVERY_ROBOT = 2
SOURCE = 3

# 动作不使能原因的人性化描述（用于 Markdown 报告）
REASON_DESC: Dict[str, str] = {
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
    "action_mask": "动作掩码为 False（与 get_action_mask 一致）",
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
        "cascade": {1, 2, 3, 4, 5, 6},
    }

    @classmethod
    def _normalize_route_code(cls, raw_route_code: Any, device_mode: str) -> int:
        if device_mode not in cls._VALID_ROUTE_CODES:
            valid_modes = sorted(cls._VALID_ROUTE_CODES.keys())
            raise ValueError(
                f"invalid device_mode={device_mode!r}; expected one of {valid_modes}"
            )
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
    def _extract_route_stage_overrides(
        normalized: Sequence[Any],
        route_name: str,
        source_name: str,
        sink_name: str,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        proc_time_map: Dict[str, int] = {}
        cleaning_duration_map: Dict[str, int] = {}
        cleaning_trigger_map: Dict[str, int] = {}

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

        return (
            proc_time_map,
            cleaning_duration_map,
            cleaning_trigger_map,
        )

    def __init__(self, config: PetriEnvConfig = None) -> None:
        assert config is not None, "config must be provided"
        self.config = config

        # ====== 基本配置 =======
        self.MAX_TIME = config.MAX_TIME
        self.n_wafer = int(config.n_wafer)
        self.max_wafers_in_system = int(config.max_wafers_in_system)
        
        self.done_event_reward = int(config.done_event_reward)
        self.finish_event_reward = self.done_event_reward * 6
        self.scrap_event_penalty = int(config.scrap_event_penalty)
        self.idle_event_penalty = float(config.idle_event_penalty)
        self.release_event_penalty = float(config.release_event_penalty)
        
        self.warn_coef_penalty = float(config.warn_coef_penalty)
        self.processing_coef_reward = float(config.processing_coef_reward)
        self.transport_overtime_coef_penalty = float(config.transport_overtime_coef_penalty)
        self.time_coef_penalty = float(config.time_coef_penalty)
        
        self.P_Residual_time = int(config.P_Residual_time)
        self.D_Residual_time = int(config.D_Residual_time)
        self.T_transport = int(config.T_transport)
        self.T_load = int(config.T_load)
        
        self.stop_on_scrap = bool(config.stop_on_scrap)

        self.robot_capacity = 1
        self._dual_arm = bool(getattr(config, 'dual_arm', False))
        self.swap_duration = 10

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
        
        self.device_mode = str(config.device_mode).lower()
        if self.device_mode != "cascade":
            raise ValueError("ClusterTool now supports cascade mode only")
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

        # ====== 路径解析 ======
        self._selected_single_route_name = self.single_route_name
        _cfg = dict(self.single_route_config or {})
        _cfg_chambers = dict(_cfg.get("chambers") or {})
        _routes = dict(_cfg.get("routes") or {})
        self._route_stages = []
        _route_chambers: List[str] = []
        if self._selected_single_route_name is not None:
            _rn = str(self._selected_single_route_name)
            _source_name = str((_cfg.get("source") or {}).get("name", "LP"))
            _sink_name = str((_cfg.get("sink") or {}).get("name", "LP_done"))
            _kind_map = {
                name: str((spec or {}).get("kind", "process"))
                for name, spec in _cfg_chambers.items()
            }
            _route_entry = dict(_routes.get(_rn) or {})
            _subpaths = _route_entry.get("subpaths")
            if isinstance(_subpaths, dict) and _subpaths:
                # 多子路径路由由 build_net 返回的 route_meta 统一提供阶段信息。
                self._route_stages = []
            else:
                _normalized = normalize_route_spec(
                    route_name=_rn,
                    route_cfg=_route_entry,
                    source_name=_source_name,
                    sink_name=_sink_name,
                    chamber_kind_map=_kind_map,
                )
                (
                    self._route_stage_proc_time_map,
                    self._route_stage_cleaning_duration_map,
                    self._route_stage_cleaning_trigger_map,
                ) = self._extract_route_stage_overrides(
                    _normalized,
                    _rn,
                    _source_name,
                    _sink_name,
                )
                for stage in _normalized[1:-1]:
                    self._route_stages.append(list(stage.candidates))
                    if str(stage.stage_type) != "buffer":
                        for name in stage.candidates:
                            if name not in _route_chambers:
                                _route_chambers.append(name)
            self.single_route_name = self._selected_single_route_name
            self.config.single_route_name = self._selected_single_route_name

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
        self._has_repeat_syntax_reentry = False
        self._ready_chambers = self.chambers
        self._single_process_chambers = self.chambers

        raw = dict(config.process_time_map or {})
        if self._route_stage_proc_time_map:
            raw.update(self._route_stage_proc_time_map)
        if self._route_stage_cleaning_duration_map:
            self._cleaning_duration_map.update(self._route_stage_cleaning_duration_map)
        if self._route_stage_cleaning_trigger_map:
            self._cleaning_trigger_map.update(self._route_stage_cleaning_trigger_map)

        obs_config = {
            "P_Residual_time": self.P_Residual_time,
            "D_Residual_time": self.D_Residual_time,
            "cleaning_duration": self.cleaning_duration,
            "cleaning_trigger_wafers": self.cleaning_trigger_wafers,
            "cleaning_duration_map": self._cleaning_duration_map,
            "cleaning_trigger_wafers_map": self._cleaning_trigger_map,
            "scrap_clip_threshold": 20.0,
        }
        info = build_net(
            n_wafer=self.n_wafer,
            ttime=max(1, self.T_transport),
            robot_capacity=self.robot_capacity,
            process_time_map=raw,
            route_code=self.route_code,
            device_mode=self.device_mode,
            obs_config=obs_config,
            route_config=self.single_route_config,
            route_name=self._selected_single_route_name or self.single_route_name,
        )
        self._base_proc_time_map = dict(info.get("process_time_map") or {})
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
        self._has_repeat_syntax_reentry = bool(route_meta.get("has_repeat_syntax_reentry", False))
        self._multi_subpath = bool(route_meta.get("multi_subpath", False))
        self._subpath_to_type: Dict[str, int] = {
            str(k): int(v) for k, v in dict(route_meta.get("subpath_to_type") or {}).items()
        }
        self._wafer_type_to_subpath: Dict[int, str] = {
            int(k): str(v) for k, v in dict(route_meta.get("wafer_type_to_subpath") or {}).items()
        }
        self._wafer_type_alloc_by_type: Dict[int, int] = {
            int(k): int(v) for k, v in dict(route_meta.get("wafer_type_alloc_by_type") or {}).items()
        }
        self._wafer_type_alloc_by_type = {
            int(k): max(0, int(v)) for k, v in self._wafer_type_alloc_by_type.items()
        }
        self._wafer_type_alloc_total_weight: int = int(sum(self._wafer_type_alloc_by_type.values()))
        self._lp_release_pattern_types: Tuple[int, ...] = tuple(
            int(x) for x in list(route_meta.get("lp_release_pattern_types") or [])
        )
        self._takt_policy: str = str(route_meta.get("takt_policy", "") or "")
        self._takt_stages_override: List[Any] = list(route_meta.get("takt_stages_override") or [])
        self._default_subpath: str = str(route_meta.get("default_subpath", "") or "")
        self._subpath_route_stages: Dict[str, List[List[str]]] = {
            str(name): [list(stage) for stage in list(stages or [])]
            for name, stages in dict(route_meta.get("subpath_route_stages") or {}).items()
        }
        self._ready_chambers = tuple(route_meta.get("chambers", ()))
        self._single_process_chambers = self.chambers

        self.m0: np.ndarray = info["m0"]
        self.m: np.ndarray = self.m0.copy()
        self.k: np.ndarray = info["capacity"]
        self.id2p_name: List[str] = info["id2p_name"]
        self.id2t_name: List[str] = info["id2t_name"]
        self._t_route_code_map: Dict[str, int] = dict(info.get("t_route_code_map") or {})
        self._token_route_queue_templates_by_type: Dict[int, Tuple[object, ...]] = {
            int(k): tuple(v)
            for k, v in dict(info.get("token_route_queue_templates_by_type") or {}).items()
        }
        self._token_route_type_sequence: List[int] = [
            int(x) for x in list(info.get("token_route_type_sequence") or [])
        ]
        self._t_target_place_map: Dict[str, str] = dict(info.get("t_target_place_map") or {})
        self._route_source_target_transport: Dict[Tuple[str, str], str] = dict(
            info.get("route_source_target_transport") or {}
        )
        self._t_route_code_by_idx: List[int] = [
            int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name
        ]
        self._t_code_to_place: Dict[int, str] = {}
        for t_name, t_code in self._t_route_code_map.items():
            if not str(t_name).startswith("t_"):
                continue
            if int(t_code) < 0:
                continue
            target = self._t_target_place_map.get(str(t_name))
            if target is None and str(t_name).startswith("t_"):
                parts = str(t_name).split("_")
                target = parts[-1] if len(parts) >= 2 else str(t_name)[2:]
            if target is not None:
                self._t_code_to_place[int(t_code)] = str(target)
        self.idle_idx: Dict[str, int] = info["idle_idx"]
        self.P = info["P"]
        self.T = info["T"]
        self.ttime = 5

        # 预计算的 pre/pst 库所索引与运输位索引（构网返回或本地计算以兼容旧版）
        self._pre_place_indices: List[np.ndarray] = info["pre_place_indices"]
        self._pst_place_indices: List[np.ndarray] = info["pst_place_indices"]
        self._transport_pre_place_idx: List[int] = info["transport_pre_place_idx"]
        self._fixed_topology: bool = bool(info.get("fixed_topology", False))

        self.marks: List[Place] = self._clone_marks(info["marks"])

        self.ori_marks = self._clone_marks(self.marks)

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
        self._cascade_round_robin_owner: Dict[str, str] = {}
        self._single_round_robin_pairs: Dict[str, Tuple[str, str]] = {}
        self._single_round_robin_next: Dict[str, str] = {}
        self._u_transition_by_source: Dict[str, int] = {}
        self._u_transition_by_source_transport: Dict[Tuple[str, str], int] = {}
        self._t_transitions_by_transport: Dict[str, List[int]] = {}
        self._tm2_transition_indices: List[int] = []
        self._tm3_transition_indices: List[int] = []

        # ====== 构造机器轮转对 ========
        if self.single_route_config is not None:
            for source, targets in self._u_targets.items():
                if len(targets) >= 2:
                    # 须保留 LLC/LLD 等非 PM 并行下游（如路线 1-5：PM7/PM8→LLC|LLD），否则
                    # _cascade_round_robin_next.values() 不含 LLC/LLD，TM2 上并行 t_* 被 _allow_t_by_machine_round_robin 永久屏蔽。
                    self._cascade_round_robin_pairs[source] = tuple(targets)
        group_owner: Dict[Tuple[str, ...], str] = {}
        for source, pair in self._cascade_round_robin_pairs.items():
            owner = group_owner.get(pair)
            if owner is None:
                owner = source
                group_owner[pair] = owner
            self._cascade_round_robin_owner[source] = owner
        self._cascade_round_robin_next = {
            source: pair[0] for source, pair in self._cascade_round_robin_pairs.items()
        }

        self._last_deadlock = False
        self._chamber_timeline: Dict[str, list] = {name: [] for name in self._timeline_chambers}
        self._chamber_active: Dict[str, Dict[int, int]] = {name: {} for name in self._timeline_chambers}
        self._init_cleaning_state()
        self._takt_result_by_type: Dict[int, Optional[Dict[str, Any]]] = self._compute_takt_results_by_type()
        self._takt_result: Optional[Dict[str, Any]] = self._takt_result_by_type.get(1)
        if self._takt_result is None and self._takt_result_by_type:
            self._takt_result = next(iter(self._takt_result_by_type.values()))
        self._last_u_LP_fire_time: int = 0
        self._u_LP_release_count: int = 0
        all_types = (
            set(self._wafer_type_to_subpath.keys())
            | set(self._takt_result_by_type.keys())
            | set(self._wafer_type_alloc_by_type.keys())
        )
        if not all_types:
            all_types = {1}
        self._u_LP_release_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type: Dict[int, int] = {int(t): 0 for t in sorted(all_types)}
        self._pending_lp_release_type: Optional[int] = None
        self._lp_release_pattern_idx: int = 0
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
        self._do_time_cost: bool = True
        self._do_proc_reward: bool = True
        self._do_transport_penalty: bool = True
        self._do_warn_penalty: bool = True
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
        if not self._training:
            print(self._takt_result)

    def train(self):
        """训练模式"""
        self._training = True

    def eval(self):
        """评估模式"""
        self._training = False

    def step(
        self,
        a1=None,
        a2=None,
        detailed_reward: bool = False,
        wait_duration: Optional[int] = None,
        with_reward: bool = True,
    ):
        """
        单设备 / 并发一步推进入口。
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

        transitions: List[int] = []
        if a1 is not None:
            transitions.append(int(a1))
        if a2 is not None:
            transitions.append(int(a2))
        do_wait = (wait_duration is not None) or (len(transitions) == 0)
        scan_info: Dict[str, Any] = {}
        log_entry: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None
        t1 = self.time

        if do_wait:
            requested_wait = (
                int(wait_duration)
                if wait_duration is not None
                else int(self.wait_durations[0] if self.wait_durations else 5)
            )
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
            swap_indices = {t_idx for t_idx in transitions if self._will_swap(int(t_idx))}
            action_duration = self.swap_duration if swap_indices else self.ttime
            t2 = t1 + action_duration
            if _pf: t_ar = perf_counter()
            reward_result, scan_info = self._advance_and_compute_reward(
                action_duration, t1, t2, detailed=detailed_reward)
            if _pf: advance_and_reward_s += perf_counter() - t_ar
            if _pf: t_fire = perf_counter()
            log_entry = self._fire(
                transitions,
                start_time=t1,
                end_time=t2,
                swap_indices=swap_indices,
            )
            if _pf: fire_s += perf_counter() - t_fire
            if isinstance(log_entry, list):
                self.fire_log.extend(log_entry)
            elif log_entry is not None:
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

    def get_enable_t(self) -> Tuple[List[int], List[int]]:
        mask = self.get_action_mask(
            wait_action_start=int(self.T),
            n_actions=int(self.T + len(self.wait_durations)),
        )
        tm2_enabled = [t_idx for t_idx in self._tm2_transition_indices if bool(mask[t_idx])]
        tm3_enabled = [t_idx for t_idx in self._tm3_transition_indices if bool(mask[t_idx])]
        return tm2_enabled, tm3_enabled

    def reset(self):
        self.marks = self._clone_marks(self.ori_marks)
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
        self._takt_result_by_type = self._compute_takt_results_by_type()
        self._takt_result = self._takt_result_by_type.get(1)
        if self._takt_result is None and self._takt_result_by_type:
            self._takt_result = next(iter(self._takt_result_by_type.values()))
        self._last_u_LP_fire_time = 0
        self._u_LP_release_count = 0
        all_types = (
            set(self._wafer_type_to_subpath.keys())
            | set(self._takt_result_by_type.keys())
            | set(self._wafer_type_alloc_by_type.keys())
        )
        if not all_types:
            all_types = {1}
        self._u_LP_release_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self._entered_wafer_count_by_type = {int(t): 0 for t in sorted(all_types)}
        self._pending_lp_release_type = None
        self._lp_release_pattern_idx = 0
        self.idle_timeout = max((p.processing_time for p in self.marks), default=0) + 30
        for idx, p in enumerate(self.marks):
            if p.name not in {"LP", "LP_done"}:
                p.capacity = 1
                self.k[idx] = 1
        self.m = np.array([len(p.tokens) for p in self.marks], dtype=int)
        self._last_state_scan = {}
        T = int(self.T)
        mask = self.get_action_mask(wait_action_start=T, n_actions=T + len(self.wait_durations))
        enabled_t = sorted(i for i in range(T) if bool(mask[i]))
        return None, enabled_t

    def _get_obs_place_order(self) -> List[str]:
        """返回观测顺序：LP + 运输位 + 腔室。"""
        tm_names = ["TM2", "TM3"] if self.device_mode == "cascade" else ["TM1"]
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

    def _update_ll_direction_obs_flags(self) -> None:
        """
        方向 one-hot 临时禁用：LLC/LLD 的 in/out 始终置 0。
        """
        for ll_name in ("LLC", "LLD"):
            place = self._place_by_name.get(ll_name)
            if place is None or not hasattr(place, "set_direction_flags"):
                continue
            place.set_direction_flags(False, False)

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
            elif p_type == _SOURCE and safe_dt > 0 and p.name == "LP":
                # LP 上仅推进负 stay_time（节拍倒计时），不累计正驻留时间。
                for tok in tokens:
                    if tok.stay_time < 0:
                        tok.stay_time = min(0, int(tok.stay_time) + safe_dt)

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

            if has_tok and (not is_scrap) and p_type in (CHAMBER, 5):
                head = tokens[0]
                post_stay = head.stay_time
                remaining_proc = p.processing_time - post_stay
                resident_limit = (
                    p_residual * 3 if p.name in {"LLC", "LLD"} else p_residual
                )
                if remaining_proc < -resident_limit:
                    overtime = -remaining_proc - resident_limit
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

    def _fire(
        self,
        t_idx: int | Sequence[int],
        start_time: int,
        end_time: int,
        is_swap: bool = False,
        swap_indices: Optional[Set[int]] = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        is_multi = not isinstance(t_idx, (int, np.integer))
        transitions = [int(t_idx)] if not is_multi else [int(idx) for idx in t_idx]
        if not transitions:
            return []

        swap_set = {int(idx) for idx in (swap_indices or set())}
        if is_swap and len(transitions) == 1 and not swap_set:
            swap_set.add(int(transitions[0]))

        def _transport_order(idx: int) -> int:
            transport = self._transition_transport_place(int(idx))
            if transport == "TM2":
                return 0
            if transport == "TM3":
                return 1
            return 2

        transitions.sort(key=_transport_order)
        log_entries: List[Dict[str, Any]] = []

        for current_t_idx in transitions:
            t_name = self.id2t_name[current_t_idx]
            pre_places = self._pre_place_indices[current_t_idx]
            pst_places = self._pst_place_indices[current_t_idx]
            if pre_places.size == 0 or pst_places.size == 0:
                log_entries.append(
                    {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
                )
                continue
            pre_place = self.marks[int(pre_places[0])]
            pst_place = self.marks[int(pst_places[0])]
            if len(pre_place.tokens) == 0:
                log_entries.append(
                    {"t_name": t_name, "t1": start_time, "t2": end_time, "token_id": -1}
                )
                continue

            pre_place_idx = int(pre_places[0])
            pst_place_idx = int(pst_places[0])

            if t_name.startswith("u_") and self._is_cascade:
                src0 = pre_place.name
                if src0 in self._cascade_round_robin_pairs:
                    head_tok = pre_place.tokens[0] if len(pre_place.tokens) > 0 else None
                    tok_gate = head_tok.route_queue[head_tok.route_head_idx] if head_tok is not None else -1
                    stage_targets = self._current_stage_targets_for_source(src0, tok_gate)
                    if self._dual_arm:
                        sync_tgt = self._first_parallel_target_dual_arm(src0)
                    else:
                        sync_tgt = self._first_receivable_parallel_target(
                            src0,
                            stage_targets=stage_targets,
                        )
                    if sync_tgt is not None:
                        self._rr_set_next(src0, sync_tgt)

            if t_name.startswith("u_") and pre_place.name == "LP":
                tok = self._pop_lp_token_for_release(self._pending_lp_release_type)
            else:
                tok = pre_place.pop_head()
            wafer_id = tok.token_id
            self._track_leave(tok, pre_place.name)
            tok.enter_time = self.time
            tok.stay_time = 0

            if t_name.startswith("u_"):
                src = pre_place.name
                tok._dst_level_targets = tuple(self._u_targets.get(src, []))
                tok.last_u_source = str(src)
                tok.machine = int(self._next_robot_machine())
                if self._is_cascade:
                    transport = pst_place.name if pst_place.name in {"TM2", "TM3"} else "TM2"
                    tok.machine = 2 if transport == "TM3" else 1
                if not (self._is_cascade and src in self._cascade_round_robin_pairs):
                    self._advance_round_robin_after_u_fire(src)
                if src in self._chamber_active and wafer_id in self._chamber_active[src]:
                    idx = self._chamber_active[src].pop(wafer_id)
                    e, _, wid = self._chamber_timeline[src][idx]
                    self._chamber_timeline[src][idx] = (e, start_time, wid)
                self._on_processing_unload(src)
            elif t_name.startswith("t_"):
                target = pst_place.name

                if current_t_idx in swap_set and self._is_swap_eligible(pst_place):
                    old_tok = pst_place.pop_head()
                    old_wafer_id = old_tok.token_id

                    self._track_leave(old_tok, target)
                    if target in self._chamber_active and old_wafer_id in self._chamber_active[target]:
                        tl_idx = self._chamber_active[target].pop(old_wafer_id)
                        e, _, wid = self._chamber_timeline[target][tl_idx]
                        self._chamber_timeline[target][tl_idx] = (e, start_time, wid)
                    self._on_processing_unload(target)

                    old_tok.enter_time = self.time
                    old_tok.stay_time = 0
                    old_tok._dst_level_targets = tuple(self._u_targets.get(target, []))
                    old_tok.machine = tok.machine
                    old_tok.route_head_idx += 1

                    tok._dst_level_targets = None
                    tok.step = max(tok.step, self._step_map.get(target, 0))
                    self._track_enter(tok, target)
                    tok.route_head_idx += 1
                    pst_place.append(tok)
                    tok._place_idx = pst_place_idx
                    if target in self._chamber_timeline and wafer_id >= 0:
                        tl_idx2 = len(self._chamber_timeline[target])
                        self._chamber_timeline[target].append((end_time, None, wafer_id))
                        self._chamber_active[target][wafer_id] = tl_idx2

                    pre_place.append(old_tok)
                    old_tok._place_idx = pre_place_idx

                    if self._is_cascade:
                        rr_source = str(getattr(tok, "last_u_source", "") or "")
                        if rr_source in self._cascade_round_robin_pairs and target in self._cascade_round_robin_pairs[rr_source]:
                            self._advance_round_robin_after_u_fire(rr_source)
                        else:
                            for rr_source, rr_tgts in self._cascade_round_robin_pairs.items():
                                if target in rr_tgts:
                                    self._advance_round_robin_after_u_fire(rr_source)
                                    break

                    log_entries.append(
                        {
                            "t_name": t_name,
                            "t1": int(start_time),
                            "t2": int(end_time),
                            "token_id": wafer_id,
                            "swap": True,
                            "swapped_token_id": old_wafer_id,
                        }
                    )
                    continue

                tok._dst_level_targets = None
                tok.step = max(tok.step, self._step_map.get(target, 0))
                self._track_enter(tok, target)
                if target == "LP_done":
                    done_type = int(getattr(tok, "route_type", 1) or 1)
                    self._entered_wafer_count_by_type[done_type] = max(
                        0, int(self._entered_wafer_count_by_type.get(done_type, 0)) - 1
                    )
                    self.entered_wafer_count = max(0, int(self.entered_wafer_count) - 1)
                    self.done_count += 1
                    self._per_wafer_reward += float(self.done_event_reward)
                elif target in self._chamber_timeline and wafer_id >= 0:
                    idx = len(self._chamber_timeline[target])
                    self._chamber_timeline[target].append((end_time, None, wafer_id))
                    self._chamber_active[target][wafer_id] = idx
                if self._is_cascade:
                    rr_source = str(getattr(tok, "last_u_source", "") or "")
                    if rr_source in self._cascade_round_robin_pairs and target in self._cascade_round_robin_pairs[rr_source]:
                        self._advance_round_robin_after_u_fire(rr_source)
                    else:
                        for rr_source, rr_tgts in self._cascade_round_robin_pairs.items():
                            if target in rr_tgts:
                                self._advance_round_robin_after_u_fire(rr_source)
                                break

            tok.route_head_idx += 1
            pst_place.append(tok)
            tok._place_idx = pst_place_idx
            self.m[pre_place_idx] -= 1
            self.m[pst_place_idx] += 1
            if t_name.startswith("u_LP_"):
                released_type = int(getattr(tok, "route_type", 1) or 1)
                self.entered_wafer_count += 1
                self._entered_wafer_count_by_type[released_type] = (
                    int(self._entered_wafer_count_by_type.get(released_type, 0)) + 1
                )
                self._last_u_LP_fire_time = int(start_time)
                self._u_LP_release_count += 1
                self._u_LP_release_count_by_type[released_type] = (
                    int(self._u_LP_release_count_by_type.get(released_type, 0)) + 1
                )
                if self._lp_release_pattern_types:
                    self._lp_release_pattern_idx += 1
                self._arm_lp_head_with_takt_delay(released_type)
                self._pending_lp_release_type = None
            log_ret: Dict[str, Any] = {
                "t_name": t_name,
                "t1": int(start_time),
                "t2": int(end_time),
                "token_id": wafer_id,
            }
            if t_name.startswith("u_"):
                log_ret["source_place"] = pre_place.name
            log_entries.append(log_ret)
        if not is_multi:
            return log_entries[0]
        return log_entries

    def _should_cancel_resident_scrap_after_fire(
        self,
        scan: Dict[str, Any],
        log_entry: Optional[Dict[str, Any]],
    ) -> bool:
        if isinstance(log_entry, list):
            for item in log_entry:
                if self._should_cancel_resident_scrap_after_fire(scan=scan, log_entry=item):
                    return True
            return False
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
        _sp = log_entry.get("source_place")
        source_name = str(_sp) if _sp is not None else t_name[2:]
        if scrap_info.get("place") != source_name:
            return False
        try:
            fired_token_id = int(log_entry.get("token_id", -1))
            scrap_token_id = int(scrap_info.get("token_id", -2))
        except (TypeError, ValueError):
            return False
        return fired_token_id >= 0 and fired_token_id == scrap_token_id

    def _token_next_target(self, tok: BasedToken) -> Optional[str]:
        """从 token 的 route_queue 推断下一个目标腔室（用于 u_* 选择）。"""
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
                return self._t_code_to_place.get(int(gate))
            if isinstance(gate, (tuple, list, set, frozenset)) and gate:
                first = next(iter(gate))
                return self._t_code_to_place.get(int(first))
            idx += 1
        return None

    def _rr_owner(self, source: str) -> str:
        return str(self._cascade_round_robin_owner.get(str(source), str(source)))

    def _rr_get_next(self, source: str) -> Optional[str]:
        owner = self._rr_owner(source)
        return self._cascade_round_robin_next.get(owner)

    def _rr_set_next(self, source: str, target: str) -> None:
        owner = self._rr_owner(source)
        for src, src_owner in self._cascade_round_robin_owner.items():
            if src_owner == owner:
                self._cascade_round_robin_next[src] = str(target)

    def _gate_targets_from_tok_gate(self, tok_gate: object) -> Tuple[str, ...]:
        if tok_gate == -1:
            return tuple()
        raw_codes: List[int] = []
        if isinstance(tok_gate, int):
            raw_codes.append(int(tok_gate))
        elif isinstance(tok_gate, (tuple, frozenset, list, set)):
            for item in tok_gate:
                if isinstance(item, int):
                    raw_codes.append(int(item))
        targets: List[str] = []
        seen: Set[str] = set()
        for code in raw_codes:
            name = self._t_code_to_place.get(int(code))
            if not isinstance(name, str):
                continue
            if name in seen:
                continue
            seen.add(name)
            targets.append(name)
        return tuple(targets)

    def _current_stage_targets_for_source(self, source: str, tok_gate: object) -> Tuple[str, ...]:
        rr_targets = tuple(self._cascade_round_robin_pairs.get(source, ()))
        if not rr_targets:
            return tuple()
        gate_targets = self._gate_targets_from_tok_gate(tok_gate)
        if not gate_targets:
            return rr_targets
        gate_set = set(gate_targets)
        filtered = tuple(t for t in rr_targets if t in gate_set)
        return filtered if filtered else rr_targets

    def _expected_target_for_source_stage(
        self,
        source: str,
        stage_targets: Tuple[str, ...],
    ) -> Optional[str]:
        if not stage_targets:
            return None
        rr_targets = tuple(self._cascade_round_robin_pairs.get(source, ()))
        if not rr_targets:
            return None
        current = self._rr_get_next(source) or rr_targets[0]
        if current in stage_targets:
            return current
        start_idx = rr_targets.index(current) if current in rr_targets else 0
        allowed = set(stage_targets)
        n = len(rr_targets)
        for i in range(1, n + 1):
            cand = rr_targets[(start_idx + i) % n]
            if cand in allowed:
                return cand
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

    def _allow_t_by_machine_round_robin(
        self,
        transport_name: str,
        target_name: str,
        tok_gate: object,
        tok: Optional[BasedToken] = None,
    ) -> bool:
        if not isinstance(tok_gate, (tuple, frozenset)):
            return True
        rr_source = str(getattr(tok, "last_u_source", "") or "")
        expected_target = self._rr_get_next(rr_source)
        if rr_source in self._cascade_round_robin_pairs:
            stage_targets = self._current_stage_targets_for_source(rr_source, tok_gate)
            stage_expected = self._expected_target_for_source_stage(rr_source, stage_targets)
            if stage_expected is not None:
                expected_target = stage_expected
        if rr_source in self._cascade_round_robin_pairs and expected_target is not None:
            allowed = str(target_name) == str(expected_target)
        else:
            allowed = target_name in self._cascade_round_robin_next.values()
        return allowed

    def _route4_next_target_from_token(self, tok: BasedToken) -> Optional[str]:
        """从 route_queue 下一 gate 推断目标库所名（route4 专用）。"""
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
        return self._t_code_to_place.get(int(next_gate))

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

    def _lp_type_head_tokens(self) -> Dict[int, BasedToken]:
        lp_place = self._place_by_name.get("LP")
        if lp_place is None:
            return {}
        heads: Dict[int, BasedToken] = {}
        for tok in lp_place.tokens:
            t_id = int(getattr(tok, "route_type", 1) or 1)
            if t_id not in heads:
                heads[t_id] = tok
        return heads

    def _peek_lp_token_by_type(self, route_type: int) -> Optional[BasedToken]:
        return self._lp_type_head_tokens().get(int(route_type))

    def _lp_releasable_types(self) -> List[int]:
        heads = self._lp_type_head_tokens()
        if not heads:
            return []
        return sorted([int(tid) for tid, tok in heads.items() if int(tok.stay_time) >= 0])

    def _select_lp_release_type(self, releasable_types: Sequence[int]) -> Optional[int]:
        if not releasable_types:
            return None
        available = [int(t) for t in releasable_types]
        if self._lp_release_pattern_types:
            expected = int(
                self._lp_release_pattern_types[
                    self._lp_release_pattern_idx % len(self._lp_release_pattern_types)
                ]
            )
            return expected if expected in available else None
        if len(available) == 1:
            return int(available[0])
        return int(np.random.choice(np.array(available, dtype=int)))

    def _allow_start_for_route_type(self, route_type: int) -> bool:
        """
        双子路径场景下按 wafer_type_alloc 严格分配 max_wafers_in_system：
        - 不做取整/四舍五入
        - 不允许类型间借额
        判定采用交叉乘法，避免浮点误差。
        """
        if not self._multi_subpath:
            return True
        total_weight = int(self._wafer_type_alloc_total_weight)
        if total_weight <= 0:
            return True
        type_id = int(route_type)
        weight = int(self._wafer_type_alloc_by_type.get(type_id, 0))
        if weight <= 0:
            return False
        current = int(self._entered_wafer_count_by_type.get(type_id, 0))
        lhs = int(current + 1) * total_weight
        rhs = int(self.max_wafers_in_system) * weight
        return lhs <= rhs

    def _allow_start(self):
        """returns True if u_LP can fire now, based on WIP and per-type LP head gate."""
        self._pending_lp_release_type = None
        if not int(self.entered_wafer_count) < int(self.max_wafers_in_system):
            return False
        releasable = [
            int(tid) for tid in self._lp_releasable_types()
            if self._allow_start_for_route_type(int(tid))
        ]
        selected = self._select_lp_release_type(releasable)
        if selected is None:
            return False
        self._pending_lp_release_type = int(selected)
        return True

    def _takt_required_interval(self, route_type: Optional[int] = None) -> Optional[int]:
        """
        返回下一次允许 u_LP 发片的最小间隔（秒）。

        口径：
        - 首片（release_count=0）不门控，返回 None
        - 第 2 片（release_count=1）使用 cycle[0]（100 拍的第 1 个拍子）
        - 第 3 片起（release_count>=2）按序推进（idx=1,2,3...），必要时对 100 取模
        """
        type_id = int(route_type if route_type is not None else 1)
        policy = str(self._takt_policy or "").strip().lower()
        if policy == "split_by_subpath":
            takt = self._takt_result_by_type.get(type_id)
            release_count = int(self._u_LP_release_count_by_type.get(type_id, 0))
        else:
            # shared / 默认：所有类型共用同一条 takt_cycle，索引按全局发片计数推进。
            takt = self._takt_result_by_type.get(type_id)
            if takt is None and self._takt_result_by_type:
                takt = next(iter(self._takt_result_by_type.values()))
            release_count = int(self._u_LP_release_count)
        if not takt:
            return None
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

    def _arm_lp_head_with_takt_delay(self, route_type: Optional[int]) -> None:
        """
        每次 u_LP 发射后，仅给新的 LP 队首 token 写入节拍倒计时（负 stay_time）。
        这样下一片的等待起点是“上一片实际发射时刻”，可严格保证发片间隔。
        """
        type_id = int(route_type if route_type is not None else 1)
        required = self._takt_required_interval(type_id)
        if required is None:
            heads = self._lp_type_head_tokens()
            for head in heads.values():
                if int(head.stay_time) < 0:
                    head.stay_time = 0
            return
        required_int = max(0, int(required))
        policy = str(self._takt_policy or "").strip().lower()
        heads = self._lp_type_head_tokens()
        if policy == "shared":
            for head in heads.values():
                head.stay_time = -required_int
        else:
            head = heads.get(type_id)
            if head is None:
                return
            head.stay_time = -required_int

    def _pop_lp_token_for_release(self, preferred_type: Optional[int]) -> BasedToken:
        lp_place = self._place_by_name.get("LP")
        if lp_place is None or len(lp_place.tokens) == 0:
            raise RuntimeError("LP has no token to release")
        if preferred_type is None:
            return lp_place.pop_head()
        target_type = int(preferred_type)
        for idx, tok in enumerate(lp_place.tokens):
            if int(getattr(tok, "route_type", 1) or 1) != target_type:
                continue
            if idx == 0:
                return lp_place.pop_head()
            token_list = list(lp_place.tokens)
            picked = token_list.pop(idx)
            lp_place.tokens = deque(token_list)
            return picked
        return lp_place.pop_head()

    def _build_transition_index(self) -> None:
        self._u_transition_by_source = {}
        self._u_transition_by_source_transport = {}
        self._t_transitions_by_transport = {}
        self._tm2_transition_indices = []
        self._tm3_transition_indices = []
        for t_idx, t_name in enumerate(self.id2t_name):
            transport_name: Optional[str] = None
            if t_name.startswith("u_"):
                pre_idx = self._pre_place_indices[t_idx]
                pst_idx = self._pst_place_indices[t_idx]
                if pre_idx.size >= 1 and pst_idx.size >= 1:
                    src = self.id2p_name[int(pre_idx[0])]
                    dst = self.id2p_name[int(pst_idx[0])]
                    if dst in {"TM2", "TM3"}:
                        self._u_transition_by_source_transport[(src, dst)] = int(t_idx)
                        transport_name = str(dst)
                    self._u_transition_by_source[src] = int(t_idx)
            elif t_name.startswith("t_"):
                transport = self._transition_transport_place(int(t_idx))
                if transport is None:
                    continue
                self._t_transitions_by_transport.setdefault(transport, []).append(int(t_idx))
                transport_name = str(transport)
            if transport_name == "TM2":
                self._tm2_transition_indices.append(int(t_idx))
            elif transport_name == "TM3":
                self._tm3_transition_indices.append(int(t_idx))

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
            if int(self._base_proc_time_map.get(place, 0) or 0) > 0
        ]
        if not valid_places:
            return None
        base_p = max(int(self._base_proc_time_map[place]) for place in valid_places)
        q: Optional[int] = None
        d = 0
        if self.cleaning_enabled:
            cleaning_candidates: List[Tuple[int, int, int, str]] = []
            for place in valid_places:
                trigger = int(self._cleaning_trigger_map.get(place, 0))
                if trigger <= 0:
                    continue
                duration = int(self._cleaning_duration_map.get(place, self.cleaning_duration))
                score = int(self._base_proc_time_map.get(place, 0)) + duration
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

        proc_places = set(self._base_proc_time_map.keys())
        out_of_stage_proc_places = sorted(proc_places - stage_place_set)
        if out_of_stage_proc_places and not self._fixed_topology:
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
        if not self._route_stages:
            return None
        self._validate_takt_stage_inputs()
        if self._has_repeat_syntax_reentry:
            horizon = int(TAKT_HORIZON)
            return {
                "fast_takt": 0.0,
                "peak_slow_takts": [],
                "cycle_length": horizon,
                "cycle_takts": [0.0 for _ in range(horizon)],
            }
        if self.device_mode == "cascade" and self.route_code == 4:
            if self.route4_takt_interval > 0:
                return build_fixed_takt_result(self.route4_takt_interval)
            return None

        return self._compute_takt_result_from_stage_lists(self._route_stages)

    def _compute_takt_result_from_stage_lists(
        self,
        route_stages: Sequence[Sequence[str]],
    ) -> Optional[Dict[str, Any]]:
        analyzer_stages: List[Dict[str, Any]] = []
        for i, stage in enumerate(route_stages):
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

    def _compute_takt_result_from_override(self) -> Optional[Dict[str, Any]]:
        raw = list(self._takt_stages_override or [])
        if not raw:
            return None

        def _infer_shared_override_m(stage_idx: int) -> int:
            """
            shared 多子路径 + 数字 override 场景：
            以“各子路径的有效工序层（过滤 p<=0）”按顺序对齐，统计该层并行机台总数。
            例如 4-8：
              - 第 1 有效工序层：path1=PM7, path2=PM1/PM2/PM3/PM4 -> m=5
              - 第 2 有效工序层：path1=PM10, path2=PM10 -> m=1（去重）
            """
            policy = str(self._takt_policy or "").strip().lower()
            if not (self._multi_subpath and policy == "shared" and self._subpath_route_stages):
                return 1
            merged_places: Set[str] = set()
            for _, stages in self._subpath_route_stages.items():
                process_layers: List[List[str]] = []
                for stage in list(stages or []):
                    valid_places = [
                        str(place)
                        for place in list(stage or [])
                        if int(self._base_proc_time_map.get(place, 0) or 0) > 0
                    ]
                    if valid_places:
                        process_layers.append(valid_places)
                if stage_idx < len(process_layers):
                    merged_places.update(process_layers[stage_idx])
            return max(1, int(len(merged_places)))

        analyzer_stages: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            if isinstance(item, (int, float)):
                p_val = int(round(float(item)))
                m_val = _infer_shared_override_m(i)
            elif isinstance(item, dict):
                p_val = int(round(float(item.get("p", 0) or 0)))
                m_val = max(1, int(item.get("m", 1) or 1))
            else:
                continue
            if p_val <= 0:
                continue
            analyzer_stages.append(
                {"name": f"s{i + 1}", "p": int(p_val), "m": int(m_val), "q": None, "d": 0}
            )
        if not analyzer_stages:
            return None
        try:
            result = analyze_cycle(analyzer_stages, max_parts=10000)
            return result
        except Exception:
            return None

    def _compute_takt_results_by_type(self) -> Dict[int, Optional[Dict[str, Any]]]:
        if not self._multi_subpath:
            return {1: self._compute_takt_result()}

        all_types = sorted(set(self._wafer_type_to_subpath.keys()) or {1})
        policy = str(self._takt_policy or "").strip().lower()
        if policy == "split_by_subpath":
            out: Dict[int, Optional[Dict[str, Any]]] = {}
            for t_id in all_types:
                subpath = self._wafer_type_to_subpath.get(int(t_id), "")
                stages = self._subpath_route_stages.get(str(subpath), [])
                out[int(t_id)] = self._compute_takt_result_from_stage_lists(stages)
            return out

        # shared / 默认：一套节拍给所有类型
        shared = self._compute_takt_result_from_override()
        if shared is None:
            shared = self._compute_takt_result()
        return {int(t_id): shared for t_id in all_types}

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

    def _transition_target_place(self, t_idx: int) -> Optional[str]:
        """根据变迁后置库所索引返回目标库所名，避免依赖 t_* 命名格式。"""
        if t_idx < 0 or t_idx >= len(self._pst_place_indices):
            return None
        pst_idx = self._pst_place_indices[int(t_idx)]
        if pst_idx.size == 0:
            return None
        return str(self.id2p_name[int(pst_idx[0])])

    def _transition_transport_place(self, t_idx: int) -> Optional[str]:
        """根据变迁前置库所索引返回运输位库所名。"""
        if t_idx < 0 or t_idx >= len(self._pre_place_indices):
            return None
        pre_idx = self._pre_place_indices[int(t_idx)]
        for p_i in pre_idx:
            p_name = str(self.id2p_name[int(p_i)])
            if p_name in {"TM2", "TM3"}:
                return p_name
        return None

    def _transport_for_t_target(self, source: str, target: str) -> str:
        """按当前 route 的 hop 映射选择 source->target 的 transport。"""
        mapped = self._route_source_target_transport.get((str(source), str(target)))
        if mapped:
            return str(mapped)
        if target in {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6"}:
            return "TM3"
        return "TM2"

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


    def get_next_event_delta(self) -> Optional[int]:
        """
        计算当前时刻到下一个关键事件的时间差（秒）。
        遍历 marks 中运输位与加工腔室的 token。
        """
        best = None
        t_transport = self.T_transport
        _CHAMBER_TYPES = (CHAMBER, 5)
        for place in self.marks:
            tokens = place.tokens
            if len(tokens) == 0:
                continue
            if place.is_dtm or place.name in {"TM2", "TM3"}:
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
        lp_heads = self._lp_type_head_tokens()
        if lp_heads:
            deltas: List[int] = []
            if self._lp_release_pattern_types:
                expect_type = int(
                    self._lp_release_pattern_types[
                        self._lp_release_pattern_idx % len(self._lp_release_pattern_types)
                    ]
                )
                tok = lp_heads.get(expect_type)
                if tok is not None and int(tok.stay_time) < 0:
                    deltas.append(-int(tok.stay_time))
                elif tok is not None:
                    deltas.append(0)
            else:
                for tok in lp_heads.values():
                    if int(tok.stay_time) < 0:
                        deltas.append(-int(tok.stay_time))
                    else:
                        deltas.append(0)
            if deltas:
                delta_takt = min(deltas)
                if best is None or delta_takt < best:
                    best = delta_takt
        return best

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

    def _is_swap_eligible(self, pst_place: Place) -> bool:
        """目标 PM 是否可执行 swap（仅在 _fire 中调用）。"""
        if not self._dual_arm:
            return False
        if not pst_place.is_pm:
            return False
        if len(pst_place.tokens) < pst_place.capacity:
            return False
        if not pst_place.tokens:
            return False
        if pst_place.tokens[0].stay_time < pst_place.processing_time:
            return False
        return not pst_place.is_cleaning

    def _will_swap(self, t_idx: int) -> bool:
        """判断 t_idx 变迁当前是否会触发 swap（用于 step 计算时长）。"""
        if not self._dual_arm:
            return False
        t_name = self.id2t_name[t_idx]
        if not t_name.startswith("t_"):
            return False
        pst_idx = self._pst_place_indices[t_idx]
        if pst_idx.size == 0:
            return False
        pst_place = self.marks[int(pst_idx[0])]
        return self._is_swap_eligible(pst_place)

    def _first_parallel_target_dual_arm(self, source: str) -> Optional[str]:
        """双臂模式：从当前指针起循环，找第一个非清洗的腔室（不检查容量）。"""
        if source not in self._cascade_round_robin_pairs:
            return None
        rr_targets = tuple(self._cascade_round_robin_pairs[source])
        if not rr_targets:
            return None
        start = self._rr_get_next(source) or rr_targets[0]
        if start not in rr_targets:
            start = rr_targets[0]
        k = rr_targets.index(start)
        n = len(rr_targets)
        for i in range(n):
            name = rr_targets[(k + i) % n]
            tp = self._get_place(name)
            if tp.is_cleaning:
                continue
            return name
        return None

    def _first_receivable_parallel_target(
        self,
        source: str,
        stage_targets: Optional[Tuple[str, ...]] = None,
    ) -> Optional[str]:
        """级联并行下游：从当前指针起循环，找第一个未满且非清洗的腔室。"""
        if source not in self._cascade_round_robin_pairs:
            return None
        rr_targets = tuple(self._cascade_round_robin_pairs[source])
        if stage_targets:
            allowed = set(stage_targets)
            rr_targets = tuple(t for t in rr_targets if t in allowed)
        if not rr_targets:
            return None
        stage_expected = self._expected_target_for_source_stage(source, rr_targets)
        start = stage_expected or rr_targets[0]
        k = rr_targets.index(start)
        n = len(rr_targets)
        for i in range(n):
            name = rr_targets[(k + i) % n]
            tp = self._get_place(name)
            if tp.is_cleaning:
                continue
            if len(tp.tokens) >= tp.capacity:
                continue
            return name
        return None

    def _is_next_stage_available(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        级联并行源：从 robin 指针起循环选取第一个可接收下游；单候选源：只检查该候选。
        双臂模式跳过容量检查，只检查非清洗。
        """
        candidates = tuple(self._u_targets.get(source, ()))
        if source in self._cascade_round_robin_pairs:
            source_place = self._get_place(source)
            head_tok = source_place.tokens[0] if len(source_place.tokens) > 0 else None
            tok_gate = head_tok.route_queue[head_tok.route_head_idx] if head_tok is not None else -1
            stage_targets = self._current_stage_targets_for_source(source, tok_gate)
            if self._dual_arm:
                tgt = self._first_parallel_target_dual_arm(source)
            else:
                tgt = self._first_receivable_parallel_target(source, stage_targets=stage_targets)
            return (tgt is not None, tgt)
        if not candidates:
            return False, None
        pointer_target = candidates[0]
        target_place = self._get_place(pointer_target)
        if target_place.is_cleaning:
            return False, None
        if not self._dual_arm and len(target_place.tokens) >= target_place.capacity:
            return False, None
        return True, pointer_target

    def _advance_round_robin_after_u_fire(self, source: str) -> None:
        if self.device_mode == "cascade" and source in self._cascade_round_robin_pairs:
            rr_targets = tuple(self._cascade_round_robin_pairs[source])
            if len(rr_targets) <= 1:
                return
            current = self._rr_get_next(source) or rr_targets[0]
            if current not in rr_targets:
                current = rr_targets[0]
            current_idx = rr_targets.index(current)
            self._rr_set_next(source, rr_targets[(current_idx + 1) % len(rr_targets)])
            return
        if self.device_mode == "single" and source in self._single_round_robin_pairs:
            rr_targets = tuple(self._single_round_robin_pairs[source])
            if len(rr_targets) <= 1:
                return
            current = self._single_round_robin_next.get(source, rr_targets[0])
            if current not in rr_targets:
                current = rr_targets[0]
            current_idx = rr_targets.index(current)
            self._single_round_robin_next[source] = rr_targets[
                (current_idx + 1) % len(rr_targets)
            ]

    def _next_robot_machine(self) -> int:
        if self.robot_capacity <= 1:
            return 1
        machine_id = self._next_machine_id
        self._next_machine_id = 2 if machine_id == 1 else 1
        return machine_id

    def _check_scrap(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        for p in self.marks:
            if p.type not in (CHAMBER, 5):
                continue
            resident_limit = (
                self.P_Residual_time * 1 if p.name in {"LLC", "LLD"} else self.P_Residual_time
            )
            for tok in p.tokens:
                remaining = p.processing_time - tok.stay_time
                if remaining < -resident_limit:
                    overtime = -remaining - resident_limit
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

    _MASK_SKIP_PLACES = frozenset({"LP", "LP_done"})
    _MASK_TIMED_TYPES = frozenset((CHAMBER, 5, DELIVERY_ROBOT))

    def get_action_mask(
        self,
        wait_action_start: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """
        返回完整离散动作掩码（transition + wait）。
        遍历 marks，内联 remaining_time / has_ready_chamber 检查。
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
            result = not bool((self.m[int(self._pre_place_indices[t_idx][0])] < 1 or
                      self.m[int(self._pst_place_indices[t_idx][0])] + 1 >
                               self.k[int(self._pst_place_indices[t_idx][0])]))
            struct_enabled_cache[t_idx] = result
            return result

        # 优先检查 LP 出片：按类型队首判定可发，并选择对应 LP->TM2/TM3 的 u 变迁。
        if self._allow_start():
            selected_type = int(self._pending_lp_release_type or 0)
            selected_tok = self._peek_lp_token_by_type(selected_type) if selected_type > 0 else None
            lp_target = self._token_next_target(selected_tok) if selected_tok is not None else None
            if lp_target is not None:
                lp_transport = self._transport_for_t_target("LP", str(lp_target))
                u_lp_idx = self._u_transition_by_source_transport.get(("LP", lp_transport))
                if u_lp_idx is not None and _is_struct_enabled(int(u_lp_idx)):
                    t_idx = int(u_lp_idx)
                    if 0 <= t_idx < total_actions:
                        mask[t_idx] = True

        has_ready_chamber = False
        ready_chambers = self._ready_chambers_set
        skip = self._MASK_SKIP_PLACES
        timed = self._MASK_TIMED_TYPES
        t_trans_by_transport = self._t_transitions_by_transport
        u_trans_by_source = self._u_transition_by_source
        route_code_by_idx = self._t_route_code_by_idx
        route_gate_allows = self._route_gate_allows_t
        place_by_name = self._place_by_name

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

            if place.is_dtm or place.name in {"TM2", "TM3"}:
                for tok in tokens:
                    if is_timed and proc_time > 0 and tok.stay_time < proc_time:
                        continue
                    tok_gate = tok.route_queue[tok.route_head_idx]
                    for t_idx in t_trans_by_transport.get(pname, ()):
                        target = self._transition_target_place(int(t_idx))
                        if target is None:
                            continue
                        allow_rr = self._allow_t_by_machine_round_robin(
                            pname,
                            target,
                            tok_gate,
                            tok,
                        )
                        if not allow_rr:
                            continue
                        if not route_gate_allows(tok_gate, route_code_by_idx[t_idx]):
                            continue
                        target_place = place_by_name.get(target)
                        if target_place is None or target_place.is_cleaning:
                            continue
                        if self._dual_arm and target_place.is_pm:
                            if len(target_place.tokens) > 0 and target_place.tokens[0].stay_time < target_place.processing_time:
                                continue
                        else:
                            if not _is_struct_enabled(t_idx):
                                continue
                        if 0 <= t_idx < total_actions:
                            mask[t_idx] = True
            else:
                head = tokens[0]
                if is_timed and proc_time > 0 and head.stay_time < proc_time:
                    continue
                available, target = self._is_next_stage_available(source=pname)
                if available and target is not None:
                    transport = self._transport_for_t_target(pname, str(target))
                    u_idx = self._u_transition_by_source_transport.get((pname, transport))
                    if u_idx is None:
                        u_idx = u_trans_by_source.get(pname)
                    struct_enabled = bool(u_idx is not None and _is_struct_enabled(u_idx))
                    if u_idx is not None and struct_enabled and 0 <= u_idx < total_actions:
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

    def render_gantt(self, out_path: str, title_suffix: str | None = None) -> None:
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
        flat_names = [str(name) for stage in route_stages for name in (stage or [])]
        seen_chambers: Set[str] = set()
        lane_stage_idx = 1
        for stage_places in route_stages:
            names = [str(x) for x in (stage_places or [])]
            if not names:
                continue
            lane_names = [name for name in names if name not in seen_chambers]
            if not lane_names:
                continue
            stage_module_names[int(lane_stage_idx)] = lane_names
            capacity[int(lane_stage_idx)] = int(len(lane_names))
            max_p = 0
            for machine_idx, name in enumerate(lane_names):
                seen_chambers.add(name)
                stage_map[name] = (int(lane_stage_idx), int(machine_idx))
                place = getattr(self, "_place_by_name", {}).get(name)
                ptime = int(getattr(place, "processing_time", 0)) if place is not None else 0
                if ptime > max_p:
                    max_p = ptime
            proc_time[int(lane_stage_idx)] = float(max_p)
            lane_stage_idx += 1

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
        final_path = out if out.suffix.lower() == ".png" else out.with_suffix(".png")
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
            title_suffix=title_suffix,
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
