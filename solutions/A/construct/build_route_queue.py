"""
级联构网：由 RouteIR 与当前子网 t 变迁集合生成晶圆 route_queue 与 t_route_code_map。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set, Tuple

from solutions.A.construct.route_compiler_single import (
    RouteIR,
    TokenRoutePlan,
    build_token_route_plan,
)


def build_token_route_queue(
    route_ir: RouteIR,
    id2t_name: List[str],
    t_target_place: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, int], Tuple[object, ...], TokenRoutePlan]:
    stage_target_order: List[str] = []
    stage_target_seen: Set[str] = set()
    for stage in route_ir.stages[1:]:
        for dst in stage.candidates:
            d = str(dst)
            if d not in stage_target_seen:
                stage_target_seen.add(d)
                stage_target_order.append(d)

    target_code_map: Dict[str, int] = {
        name: i + 1 for i, name in enumerate(stage_target_order)
    }
    t_route_code_map: Dict[str, int] = {}
    for t_name in id2t_name:
        if not t_name.startswith("t_"):
            continue
        target = t_target_place.get(t_name)
        if target is None:
            continue
        code = int(target_code_map.get(str(target), -1))
        if code > 0:
            t_route_code_map[t_name] = code

    route_queue: List[object] = []
    for idx in range(len(route_ir.stages) - 1):
        route_queue.append(-1)
        next_stage = route_ir.stages[idx + 1]
        gate_codes: List[int] = []
        for dst in next_stage.candidates:
            code = int(target_code_map.get(str(dst), -1))
            if code > 0:
                gate_codes.append(code)
        if len(gate_codes) == 1:
            route_queue.append(gate_codes[0])
        else:
            route_queue.append(tuple(gate_codes))
    token_route_queue = tuple(route_queue)

    token_plan = build_token_route_plan(
        route_ir=route_ir,
        transition_names=[f"t_{name}" for name in stage_target_order],
    )
    return target_code_map, t_route_code_map, token_route_queue, token_plan


def build_token_route_queue_multi(
    route_irs: Mapping[str, RouteIR],
    id2t_name: Sequence[str],
    t_target_place: Mapping[str, str],
) -> Dict[str, object]:
    """
    多子路径版本：
    - 统一构造 target_code_map / t_route_code_map（跨子路径并集）
    - 为每个子路径生成独立 token_route_queue / token_route_plan
    - 输出 wafer_type 与子路径映射
    """
    if not route_irs:
        raise ValueError("route_irs cannot be empty")

    route_items = list(route_irs.items())
    stage_target_order: List[str] = []
    stage_target_seen: Set[str] = set()
    for _, route_ir in route_items:
        for stage in route_ir.stages[1:]:
            for dst in stage.candidates:
                name = str(dst)
                if name in stage_target_seen:
                    continue
                stage_target_seen.add(name)
                stage_target_order.append(name)

    target_code_map: Dict[str, int] = {
        name: i + 1 for i, name in enumerate(stage_target_order)
    }
    t_route_code_map: Dict[str, int] = {}
    for t_name in id2t_name:
        if not str(t_name).startswith("t_"):
            continue
        target = t_target_place.get(str(t_name))
        if target is None:
            continue
        code = int(target_code_map.get(str(target), -1))
        if code > 0:
            t_route_code_map[str(t_name)] = code

    token_route_queue_templates: Dict[str, Tuple[object, ...]] = {}
    token_route_plan_templates: Dict[str, TokenRoutePlan] = {}
    for subpath_name, route_ir in route_items:
        route_queue: List[object] = []
        for idx in range(len(route_ir.stages) - 1):
            route_queue.append(-1)
            next_stage = route_ir.stages[idx + 1]
            gate_codes: List[int] = []
            for dst in next_stage.candidates:
                code = int(target_code_map.get(str(dst), -1))
                if code > 0:
                    gate_codes.append(code)
            if len(gate_codes) == 1:
                route_queue.append(gate_codes[0])
            else:
                route_queue.append(tuple(gate_codes))
        token_route_queue_templates[str(subpath_name)] = tuple(route_queue)
        token_route_plan_templates[str(subpath_name)] = build_token_route_plan(
            route_ir=route_ir,
            transition_names=[f"t_{name}" for name in stage_target_order],
        )

    subpath_names = [name for name, _ in route_items]
    subpath_to_type: Dict[str, int] = {
        name: idx + 1 for idx, name in enumerate(subpath_names)
    }
    wafer_type_to_subpath: Dict[int, str] = {
        t: name for name, t in subpath_to_type.items()
    }

    default_subpath = subpath_names[0]
    return {
        "target_code_map": target_code_map,
        "t_route_code_map": t_route_code_map,
        "token_route_queue_template": token_route_queue_templates[default_subpath],
        "token_route_plan_template": token_route_plan_templates[default_subpath],
        "token_route_queue_templates": token_route_queue_templates,
        "token_route_plan_templates": token_route_plan_templates,
        "subpath_to_type": subpath_to_type,
        "wafer_type_to_subpath": wafer_type_to_subpath,
        "default_subpath": default_subpath,
    }
