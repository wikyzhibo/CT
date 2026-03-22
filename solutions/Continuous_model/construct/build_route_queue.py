"""
级联构网：由 RouteIR 与当前子网 t 变迁集合生成晶圆 route_queue 与 t_route_code_map。
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from solutions.Continuous_model.construct.route_compiler_single import (
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
