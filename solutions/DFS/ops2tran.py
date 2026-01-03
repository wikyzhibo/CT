from typing import List, Tuple

# 你的优先级序列（用于同一时刻的 tie-break）
tmp = ['t8' , 'u8','t7', 'u6', 't6', 'u4', 't5', 'u5', 'u2', 't8', 'u7', 't3', 't2', 't1', 'u1', 'u0', 't4', 'u3']
prio = {a:i for i,a in enumerate(tmp)}

# stage -> (start_action, end_action)
STAGE2ACT = {
    1: ("t3", "u3"),
    2: ("t5", "u5"),
    3: ("t6", "u6"),
    4: ("t7", "u7"),
    5: ("t8", "u8"),
}

def ops_to_actions(ops) -> List[Tuple[str, float]]:
    """
    return: [(action_name, time), ...] 例如 [('t3', 0.0), ('u3', 70.0), ...]
    """
    events = []
    for op in ops:
        a_start, a_end = STAGE2ACT[op.stage]

        # 启动时刻
        events.append((a_start, float(op.start), op.job, op.stage, op.machine))

        # 结束时刻：默认用 proc_end（加工结束）
        events.append((a_end, float(op.proc_end), op.job, op.stage, op.machine))

        # 如果你想用 end（驻留结束/释放）作为“结束动作时间”，改成：
        # events.append((a_end, float(op.end), op.job, op.stage, op.machine))

    # 排序：先时间，再按 tmp 给的动作优先级
    def key(e):
        act, t, *_ = e
        return (t, prio.get(act, 10**9))

    events.sort(key=key)

    # 只返回你要的 (动作, 时间)
    return [(act, t) for act, t, *_ in events]


from typing import List, Tuple, Dict, Any

# 机械手动作集合（按你的描述）
ARM1 = set(["t3", "u3", "u6", "t7", "u7", "t8"])
ARM2 = set(["t5", "u6", "t6"])

def detect_arm_deadlock(
    action_seq: List[Tuple[str, float]],  # [(act, time), ...]
    cap: int = 2,
    prefer_arm1_on_overlap: bool = True
) -> Dict[str, Any]:
    """
    返回:
      {
        "ok": bool,
        "violations": [ {...}, ... ],
        "final_load": {"arm1": x, "arm2": y}
      }
    规则:
      - t* 表示装载: load -= 1
      - u* 表示卸载: load += 1
      - 任意时刻 load > cap 或 load < 0 => 非法
    """

    load = {"arm1": 2, "arm2": 0}
    violations = []

    def which_arm(act: str):
        in1, in2 = act in ARM1, act in ARM2
        if in1 and in2:
            return "arm1" if prefer_arm1_on_overlap else "arm2"
        if in1:
            return "arm1"
        if in2:
            return "arm2"
        return None

    arm1_heap = []
    arm2_heap = []
    for idx, (act, t) in enumerate(action_seq):
        arm = which_arm(act)
        if arm is None:
            continue  # 不属于任何机械手范围的动作，忽略

        if act.startswith("t"):      # 装载
            load[arm] -= 1
            #if arm == "arm1":
            #    arm1_heap.pop(0)
            #else:
            #    arm2_heap.pop(0)

        elif act.startswith("u"):    # 卸载
            load[arm] += 1
            #if arm == "arm1":
            #    arm1_heap.append((act, t))
            #else:
            #    arm2_heap.append((act, t))
        else:
            continue

        if load[arm] > cap:
            violations.append({
                "arm": arm,
                "time": t,
                "action": act,
                "index": idx,
                "cap": cap
            })

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "final_load": load
    }
