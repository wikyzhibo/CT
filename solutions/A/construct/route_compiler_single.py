"""
单设备/级联路径编译器（配置驱动）。

职责：
1. 解析 route path / sequence / repeat；
2. 推断 stage 间 transport robot（级联固定网：`_TM2_SCOPE`/`_TM3_SCOPE` 子集判定，见 `build_topology.infer_cascade_transport_by_scope`）；
3. 生成 token route plan（含兼容 route_queue 视图）；
4. 生成与 pn_single 兼容的 route_meta。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from solutions.A.construct.build_topology import infer_cascade_transport_by_scope


@dataclass(frozen=True, slots=True)
class StageIR:
    stage_idx: int
    candidates: Tuple[str, ...]
    stage_type: str
    label: str
    repeat_origin: Optional[str] = None
    repeat_iter: int = 0
    stage_process_time: Optional[float] = None
    stage_cleaning_duration: Optional[int] = None
    stage_cleaning_trigger_wafers: Optional[int] = None


@dataclass(frozen=True, slots=True)
class TransportIR:
    hop_idx: int
    from_stage_idx: int
    to_stage_idx: int
    robot_name: str
    transport_place: str


@dataclass(frozen=True, slots=True)
class RouteIR:
    route_name: str
    raw_expr: str
    stages: Tuple[StageIR, ...]
    transports: Tuple[TransportIR, ...]


@dataclass(frozen=True, slots=True)
class StagePlan:
    stage_idx: int
    candidates: Tuple[str, ...]
    incoming_transport: Optional[str]
    outgoing_transport: Optional[str]
    entry_transition_names: Tuple[str, ...]
    exit_transition_names: Tuple[str, ...]
    release_chain: Tuple[int, ...]
    is_sink: bool = False


@dataclass(frozen=True, slots=True)
class TokenRoutePlan:
    route_name: str
    stages: Tuple[StagePlan, ...]
    route_queue_template: Tuple[object, ...]
    transition_code_map: Dict[str, int]
    sink_stage_idx: int


@dataclass(frozen=True, slots=True)
class RobotSpec:
    name: str
    managed_chambers: Tuple[str, ...]
    transport_place: str
    priority: int = 0
    allowed_stage_links: Tuple[Tuple[str, str], ...] = ()


def _parse_stage_token(token: str) -> Tuple[str, ...]:
    candidates = tuple(part for part in token.split("/") if part)
    if not candidates:
        raise ValueError("empty stage token")
    return candidates


def parse_route_string(route_expr: str) -> List[Dict[str, Any]]:
    """
    解析字符串形式路径 AST。
    支持：
    - LP->PM7/PM8->PM9/PM10->LP_done
    - LP->[PM7/PM8->PM9/PM10]*5->LP_done
    """
    s = route_expr.replace(" ", "")
    i = 0
    n = len(s)
    ast: List[Dict[str, Any]] = []

    def parse_until(stop_chars: str) -> str:
        nonlocal i
        start = i
        while i < n and s[i] not in stop_chars:
            i += 1
        return s[start:i]

    while i < n:
        if s[i] == "[":
            i += 1
            block_items: List[Dict[str, Any]] = []
            while i < n and s[i] != "]":
                token = parse_until("]-")
                if token:
                    block_items.append({"type": "stage", "candidates": _parse_stage_token(token)})
                if s.startswith("->", i):
                    i += 2
            if i >= n or s[i] != "]":
                raise ValueError("unclosed repeat block")
            i += 1
            if i >= n or s[i] != "*":
                raise ValueError("repeat block missing *N")
            i += 1
            times_raw = parse_until("-")
            times = int(times_raw)
            if times <= 0:
                raise ValueError("repeat count must be > 0")
            ast.append({"type": "repeat", "count": times, "sequence": block_items})
        else:
            token = parse_until("-")
            if token:
                ast.append({"type": "stage", "candidates": _parse_stage_token(token)})
        if s.startswith("->", i):
            i += 2
    return ast


def _expand_sequence_nodes(
    nodes: Sequence[Mapping[str, Any]],
    repeat_origin: Optional[str] = None,
) -> List[Tuple[Tuple[str, ...], Optional[str], int, Mapping[str, Any]]]:
    expanded: List[Tuple[Tuple[str, ...], Optional[str], int, Mapping[str, Any]]] = []
    for node in nodes:
        if "stage" in node:
            stage_obj = node["stage"]
            candidates = tuple(stage_obj.get("candidates", ()))
            if not candidates:
                raise ValueError(f"invalid stage node: {node}")
            expanded.append((candidates, repeat_origin, 0, dict(stage_obj)))
            continue

        node_type = str(node.get("type", "")).lower()
        if node_type == "stage":
            candidates = tuple(node.get("candidates", ()))
            if not candidates:
                raise ValueError(f"invalid stage node: {node}")
            expanded.append((candidates, repeat_origin, 0, dict(node)))
            continue

        if "repeat" in node or node_type == "repeat":
            rep = node["repeat"] if "repeat" in node else node
            count = int(rep.get("count", 0))
            if count <= 0:
                raise ValueError(f"repeat.count must be > 0: {node}")
            seq = rep.get("sequence")
            if not isinstance(seq, list) or not seq:
                raise ValueError(f"repeat.sequence must be non-empty list: {node}")
            origin = f"repeat({count})"
            for k in range(1, count + 1):
                child = _expand_sequence_nodes(seq, repeat_origin=origin)
                for cands, _, _, payload in child:
                    expanded.append((cands, origin, k, payload))
            continue

        raise ValueError(f"unsupported route node: {node}")
    return expanded


def _alias_load_port_tokens(
    cands: Tuple[str, ...],
    source_name: str,
) -> Tuple[str, ...]:
    """JSON 中占位符 LP 在级联双装载口模式下映射为 LP1/LP2。"""
    if source_name not in ("LP1", "LP2"):
        return cands
    return tuple(source_name if x == "LP" else x for x in cands)


def first_load_port_name(route_ir: RouteIR) -> str:
    """首段 source stage 中的装载口名（LP/LP1/LP2）。"""
    for c in route_ir.stages[0].candidates:
        if c in ("LP1", "LP2"):
            return str(c)
        if c == "LP":
            return "LP1"
    raise ValueError(f"route {route_ir.route_name} first stage has no LP/LP1/LP2 candidate")


def _normalize_sequence(
    route_cfg: Mapping[str, Any]
) -> List[Tuple[Tuple[str, ...], Optional[str], int, Mapping[str, Any]]]:
    if "sequence" in route_cfg:
        seq = route_cfg.get("sequence")
        if not isinstance(seq, list) or not seq:
            raise ValueError("route.sequence must be a non-empty list")
        return _expand_sequence_nodes(seq)

    raw_path = str(route_cfg.get("path", "")).strip()
    if not raw_path:
        raise ValueError("route must contain either sequence or path")
    ast = parse_route_string(raw_path)
    return _expand_sequence_nodes(ast)


def normalize_route_spec(
    route_name: str,
    route_cfg: Mapping[str, Any],
    source_name: str,
    sink_name: str,
    chamber_kind_map: Mapping[str, str],
) -> Tuple[StageIR, ...]:
    raw_seq = _normalize_sequence(route_cfg)
    if len(raw_seq) < 2:
        raise ValueError(f"route {route_name} must have at least 2 stages")

    stages: List[StageIR] = []
    for idx, (cands, rep_origin, rep_iter, payload) in enumerate(raw_seq):
        cands = _alias_load_port_tokens(cands, source_name)
        if idx == 0:
            stage_type = "source"
        elif idx == len(raw_seq) - 1:
            stage_type = "sink"
        else:
            first = cands[0]
            if first == source_name:
                stage_type = "source"
            elif first == sink_name:
                stage_type = "sink"
            else:
                ck = chamber_kind_map.get(first, "process")
                if ck in {"buffer", "loadlock"}:
                    stage_type = ck
                else:
                    stage_type = "process"
        stages.append(
            StageIR(
                stage_idx=idx,
                candidates=tuple(cands),
                stage_type=stage_type,
                label="/".join(cands),
                repeat_origin=rep_origin,
                repeat_iter=rep_iter,
                stage_process_time=(
                    float(payload.get("process_time"))
                    if payload.get("process_time", None) is not None
                    else None
                ),
                stage_cleaning_duration=(
                    int(payload.get("cleaning_duration"))
                    if payload.get("cleaning_duration", None) is not None
                    else None
                ),
                stage_cleaning_trigger_wafers=(
                    int(payload.get("cleaning_trigger_wafers"))
                    if payload.get("cleaning_trigger_wafers", None) is not None
                    else None
                ),
            )
        )

    if source_name not in stages[0].candidates:
        raise ValueError(f"route {route_name} first stage must include source {source_name}")
    if sink_name not in stages[-1].candidates:
        raise ValueError(f"route {route_name} last stage must include sink {sink_name}")
    return tuple(stages)


def infer_transport_robot(
    from_candidates: Sequence[str],
    to_candidates: Sequence[str],
    robots: Mapping[str, RobotSpec],
) -> RobotSpec:
    from_set = frozenset(from_candidates)
    to_set = frozenset(to_candidates)
    matched: List[RobotSpec] = []
    for robot in robots.values():
        scope = frozenset(robot.managed_chambers)
        if not from_set.issubset(scope) or not to_set.issubset(scope):
            continue
        if robot.allowed_stage_links:
            allowed = set(robot.allowed_stage_links)
            ok = True
            for src in from_candidates:
                if not any((src, dst) in allowed for dst in to_candidates):
                    ok = False
                    break
            if not ok:
                continue
        matched.append(robot)

    if not matched:
        raise ValueError(
            f"no robot can transport hop {tuple(from_candidates)} -> {tuple(to_candidates)}"
        )

    # 级联重复路由（如 2-3）中，LLC->LLD 期望由内侧机械手执行。
    # 当该 hop 同时被多个机器人覆盖时，优先选择 TM3（或其 transport_place=d_TM3）。
    if (
        len(from_candidates) == 1
        and len(to_candidates) == 1
        and str(from_candidates[0]) == "LLC"
        and str(to_candidates[0]) == "LLD"
    ):
        for rb in matched:
            if rb.name == "TM3" or rb.transport_place == "d_TM3":
                return rb

    matched.sort(key=lambda r: (int(r.priority), len(r.managed_chambers), r.name))
    top = matched[0]
    if len(matched) > 1:
        second = matched[1]
        if int(second.priority) == int(top.priority) and len(second.managed_chambers) == len(top.managed_chambers):
            raise ValueError(
                f"ambiguous robot for hop {tuple(from_candidates)} -> {tuple(to_candidates)}; "
                "set robot.priority or reduce managed_chambers overlap"
            )
    return top


def compile_route_stages(
    route_name: str,
    route_cfg: Mapping[str, Any],
    source_name: str,
    sink_name: str,
    chamber_kind_map: Mapping[str, str],
    robots: Mapping[str, RobotSpec],
) -> RouteIR:
    """robots 保留为调用签名；级联 hop 机械手仅由 scope 推断。"""
    stages = normalize_route_spec(
        route_name=route_name,
        route_cfg=route_cfg,
        source_name=source_name,
        sink_name=sink_name,
        chamber_kind_map=chamber_kind_map,
    )
    transports: List[TransportIR] = []
    for hop_idx in range(len(stages) - 1):
        left = stages[hop_idx]
        right = stages[hop_idx + 1]
        transport_place = infer_cascade_transport_by_scope(left.candidates, right.candidates)
        transports.append(
            TransportIR(
                hop_idx=hop_idx,
                from_stage_idx=left.stage_idx,
                to_stage_idx=right.stage_idx,
                robot_name=transport_place,
                transport_place=transport_place,
            )
        )
    return RouteIR(
        route_name=route_name,
        raw_expr=str(route_cfg.get("path", "")),
        stages=stages,
        transports=tuple(transports),
    )


def build_token_route_plan(
    route_ir: RouteIR,
    transition_names: Optional[Sequence[str]] = None,
) -> TokenRoutePlan:
    allowed_t = set(transition_names or ())

    trans_names: List[str] = []
    for stage in route_ir.stages[1:]:
        for chamber in stage.candidates:
            t_name = f"t_{chamber}"
            if transition_names is not None and t_name not in allowed_t:
                continue
            if t_name not in trans_names:
                trans_names.append(t_name)
    transition_code_map: Dict[str, int] = {name: i + 1 for i, name in enumerate(trans_names)}

    hop_in = {t.to_stage_idx: t.transport_place for t in route_ir.transports}
    hop_out = {t.from_stage_idx: t.transport_place for t in route_ir.transports}
    stage_plans: List[StagePlan] = []
    for stage in route_ir.stages:
        entry_t = tuple(
            f"t_{name}"
            for name in stage.candidates
            if (transition_names is None or f"t_{name}" in allowed_t)
        )
        exit_u = tuple(f"u_{name}" for name in stage.candidates)
        stage_plans.append(
            StagePlan(
                stage_idx=stage.stage_idx,
                candidates=stage.candidates,
                incoming_transport=hop_in.get(stage.stage_idx),
                outgoing_transport=hop_out.get(stage.stage_idx),
                entry_transition_names=entry_t,
                exit_transition_names=exit_u,
                release_chain=tuple(
                    st.stage_idx for st in route_ir.stages[stage.stage_idx + 1 :]
                ),
                is_sink=(stage.stage_idx == len(route_ir.stages) - 1),
            )
        )

    route_queue: List[object] = []
    for idx in range(len(route_ir.stages) - 1):
        route_queue.append(-1)
        next_stage = route_ir.stages[idx + 1]
        gates = tuple(
            transition_code_map[f"t_{name}"]
            for name in next_stage.candidates
            if f"t_{name}" in transition_code_map
        )
        if len(gates) == 1:
            route_queue.append(gates[0])
        else:
            route_queue.append(gates)

    return TokenRoutePlan(
        route_name=route_ir.route_name,
        stages=tuple(stage_plans),
        route_queue_template=tuple(route_queue),
        transition_code_map=transition_code_map,
        sink_stage_idx=len(route_ir.stages) - 1,
    )


def build_route_meta_from_route_ir(
    route_ir: RouteIR,
    buffer_names: Iterable[str],
) -> Dict[str, object]:
    buffer_set = set(buffer_names)
    full_stage_candidates: List[Tuple[str, ...]] = [tuple(s.candidates) for s in route_ir.stages]
    inner_stages: List[Tuple[str, ...]] = full_stage_candidates[1:-1]

    release_station_aliases: Dict[str, List[str]] = {}
    for i, stage in enumerate(inner_stages):
        release_station_aliases[f"s{i + 1}"] = list(stage)

    chamber_list: List[str] = []
    for stage in inner_stages:
        if not (len(stage) == 1 and stage[0] in buffer_set):
            chamber_list.extend(stage)
    chambers = tuple(chamber_list)
    timeline_chambers = chambers + tuple(
        stage[0] for stage in inner_stages if len(stage) == 1 and stage[0] in buffer_set
    )

    step_map: Dict[str, int] = {"LP_done": len(inner_stages) + 1}
    step = 1
    for stage in inner_stages:
        for place in stage:
            step_map[place] = step
        step += 1

    # 聚合生成 u_targets，支持重复段同名 stage（例如循环段的 LLD->PM7/LP_done）
    u_targets: Dict[str, List[str]] = {}
    for i in range(len(full_stage_candidates) - 1):
        src_stage = full_stage_candidates[i]
        dst_stage = full_stage_candidates[i + 1]
        for src in src_stage:
            arr = u_targets.setdefault(src, [])
            for dst in dst_stage:
                if dst not in arr:
                    arr.append(dst)

    system_entry_places = set(inner_stages[0]) if inner_stages else set()

    lp_key = first_load_port_name(route_ir)
    u_lp = f"u_{lp_key}"

    release_chain_by_u: Dict[str, List[str]] = {}
    if inner_stages:
        release_chain_by_u[u_lp] = [f"s{k}" for k in range(1, len(inner_stages) + 1)]
    for i, stage in enumerate(inner_stages):
        if len(stage) == 1 and stage[0] in buffer_set:
            u_name = f"u_{stage[0]}"
            chain = [f"s{k}" for k in range(i + 2, len(inner_stages) + 1)]
            if chain:
                release_chain_by_u[u_name] = chain

    # 兼容既有 release-chain 口径
    if len(inner_stages) >= 2 and len(inner_stages[1]) == 1 and inner_stages[1][0] in buffer_set:
        release_chain_by_u[u_lp] = ["s1", "s2"]
    if len(inner_stages) >= 4 and len(inner_stages[1]) == 1 and inner_stages[1][0] in buffer_set:
        release_chain_by_u["u_LLC"] = ["s3", "s4"]
    if len(inner_stages) >= 5:
        release_chain_by_u["u_LLD"] = ["s5"]

    return {
        "chambers": chambers,
        "timeline_chambers": timeline_chambers,
        "u_targets": u_targets,
        "step_map": step_map,
        "release_station_aliases": release_station_aliases,
        "release_chain_by_u": release_chain_by_u,
        "system_entry_places": system_entry_places,
    }
