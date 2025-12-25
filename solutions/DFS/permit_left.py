from solutions.DFS.plot import plot_gantt_hatched_residence
from solutions.DFS.ops2tran import ops_to_actions
import tomllib
import bisect
from typing import Dict, List, Tuple
from dataclasses import dataclass

Interval = Tuple[float, float]  # [l, r], r>l
INF = 10**18


@dataclass
class Op:
    job: int
    stage: int
    machine: int
    start: float
    proc_end: float
    end: float  # occupied until end (includes residence)


def _num_stages(proc_time: Dict[int, float]) -> int:
    S = max(proc_time.keys())
    for s in range(1, S + 1):
        if s not in proc_time:
            raise ValueError(f"proc_time missing stage {s}")
    return S

def _insert_interval_sorted(busy: List[Interval], itv: Interval) -> None:
    l, _ = itv
    i = bisect.bisect_left(busy, (l, -INF))
    busy.insert(i, itv)


def _earliest_non_overlapping_start(busy: List[Interval], s: float, dur: float) -> float:
    """返回 >=s 的最早 t，使 [t,t+dur) 不与 busy 重叠；busy 按 l 排序"""
    t = s
    if dur <= 0:
        return t

    i = bisect.bisect_right(busy, (t, INF)) - 1
    if i >= 0 and busy[i][1] > t:
        t = busy[i][1]

    j = bisect.bisect_left(busy, (t, -INF))
    while j < len(busy):
        l, r = busy[j]
        if t + dur <= l:
            break
        if r > t:
            t = r
        j += 1
    return t


# ====== 机械手：动作集合（按你定义） ======
ARM1 = set(["t3", "u3", "u6", "t7", "u7", "t8"])
ARM2 = set(["t5", "u6", "t6"])

# stage -> (start_action, end_action)
STAGE2ACT = {
    1: ("t3", "u3"),
    2: ("t5", "u5"),
    3: ("t6", "u6"),
    4: ("t7", "u7"),
    5: ("t8", "u8"),
}


def _which_arm(act: str, prefer_arm1_on_overlap: bool = True) -> str | None:
    in1, in2 = act in ARM1, act in ARM2
    if in1 and in2:
        return "arm1" if prefer_arm1_on_overlap else "arm2"
    if in1:
        return "arm1"
    if in2:
        return "arm2"
    return None


def _count_overlap(intervals: List[Interval], t: float) -> int:
    """统计在时刻 t 覆盖的区间数（O(n)，足够实现效果）"""
    c = 0
    for l, r in intervals:
        if l <= t < r:
            c += 1
    return c


def _next_drop_time(intervals: List[Interval], t: float) -> float:
    """在 t 之后，找到最近一个会使重叠数可能下降的时刻（即某个 r>t 的最小 r）"""
    nxt = INF
    for l, r in intervals:
        if r > t and r < nxt:
            nxt = r
    return nxt


def _arm_delay_needed_for_job(
    arm_intervals: Dict[str, List[Interval]],
    new_intervals: Dict[str, List[Interval]],
    cap: int = 2
) -> float:
    """
    给定现有 arm_intervals，以及当前 job 产生的新持有区间 new_intervals（已包含当前 delta 下的时间），
    返回需要整体延迟的最小量（保守实现：逐个区间看 start 时刻是否已满载）。
    """
    need = 0.0
    for arm, segs in new_intervals.items():
        exist = arm_intervals[arm]
        for a, b in segs:
            if b <= a:
                continue
            # 如果在区间开始时刻 a，已有占用已达到 cap，则必须把 a 推到某个释放点之后
            if _count_overlap(exist, a) >= cap:
                t_free = a
                # 不断跳到“最近的释放时刻”，直到重叠 < cap
                while _count_overlap(exist, t_free) >= cap:
                    nxt = _next_drop_time(exist, t_free)
                    if nxt >= INF:
                        # 理论上不会：除非存在无限长区间
                        return INF
                    t_free = nxt
                need = max(need, t_free - a)
    return need


def schedule_shift_search_window0_with_arrivals(
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    tau: float,
    routes: Dict[int, List[int]],
    arrivals: Dict[int, float],
    t1_min: float = 0.0,
    sort_by_arrival: bool = True,
    arm_cap: int = 2,
    prefer_arm1_on_overlap: bool = True,
) -> List[Op]:
    """
    Window=0:
      start[next] = start[cur] + p[cur] + tau  (若资源冲突则整体右移)
    加入 arrival: stage1 >= max(t1_min, a_j)
    加入机械手容量约束：用“u到下一次t之间的持有区间”表示手上晶圆占用，重叠数 <= arm_cap
      - u: unload -> 手上 +1
      - t: load   -> 手上 -1
    """
    # ====== 机器 busy（按 stage, machine） ======
    machine_busy: Dict[int, List[List[Interval]]] = {
        s: [[] for _ in range(capacity[s])]
        for s in capacity.keys()
    }

    # ====== 机械手持有区间记录器 ======
    # arm_intervals[arm] = sorted list of holding intervals [u_time, t_time)
    arm_intervals: Dict[str, List[Interval]] = {"arm1": [], "arm2": []}

    ops: List[Op] = []

    jobs = list(routes.keys())
    if sort_by_arrival:
        jobs.sort(key=lambda j: (arrivals.get(j, 0.0), j))
    else:
        jobs.sort()

    for job in jobs:
        R = routes[job]
        L = len(R)

        # 基准时间：把 arrival 写进 stage1
        a_j = arrivals.get(job, 0.0)
        base_start: Dict[int, float] = {}
        base_start[R[0]] = max(t1_min, a_j)
        for k in range(1, L):
            u, v = R[k - 1], R[k]
            base_start[v] = base_start[u] + proc_time[u] + tau

        delta = 0.0  # 该 job 的整体右移
        chosen_m: Dict[int, int] = {}
        chosen_s: Dict[int, float] = {}

        # 外层：同时满足“机器层面 + 机械手层面”
        while True:
            # ====== A) 先在机器层面做 shift search，得到一组机器可行的 starts ======
            while True:
                delta_prev = delta
                chosen_m.clear()
                chosen_s.clear()

                for s in R:
                    want = base_start[s] + delta
                    dur = proc_time[s]

                    best_t = INF
                    best_m = -1
                    for m in range(capacity[s]):
                        t = _earliest_non_overlapping_start(machine_busy[s][m], want, dur)
                        if t < best_t:
                            best_t, best_m = t, m

                    chosen_m[s] = best_m
                    chosen_s[s] = best_t

                    need = best_t - want
                    if need > 1e-12:
                        delta = max(delta, delta_prev + need)

                if abs(delta - delta_prev) < 1e-12:
                    break  # 机器层面收敛

            # ====== B) 用当前 starts 生成该 job 的机械手“持有区间”，检查是否超容量 ======
            start = {s: chosen_s[s] for s in R}
            proc_end = {s: start[s] + proc_time[s] for s in R}

            new_hold: Dict[str, List[Interval]] = {"arm1": [], "arm2": []}
            for k in range(L - 1):
                u, v = R[k], R[k + 1]
                u_act = STAGE2ACT[u][1]      # u*
                t_act = STAGE2ACT[v][0]      # t*
                arm = _which_arm(u_act, prefer_arm1_on_overlap) or _which_arm(t_act, prefer_arm1_on_overlap)
                if arm is None:
                    continue
                # u 时刻拿起晶圆(+1)，直到下一次 t 放下(-1)：形成持有区间
                hold = (proc_end[u], start[v])
                if hold[1] > hold[0] + 1e-12:
                    new_hold[arm].append(hold)

            arm_need = _arm_delay_needed_for_job(arm_intervals, new_hold, cap=arm_cap)

            if arm_need <= 1e-12:
                # 机械手也可行：退出外层，提交该 job
                break

            # 机械手不可行：整体右移，并回到机器层面重新检查
            delta += arm_need

        # ====== commit：写入 ops + 更新 machine_busy + 更新 arm_intervals ======
        start = {s: chosen_s[s] for s in R}
        proc_end = {s: start[s] + proc_time[s] for s in R}
        end: Dict[int, float] = {}

        for k in range(L - 1):
            u, v = R[k], R[k + 1]
            end[u] = start[v] - tau
        end[R[-1]] = proc_end[R[-1]]

        # 更新 machine_busy
        for s in R:
            m = chosen_m[s]
            ops.append(Op(job, s, m, start[s], proc_end[s], end[s]))
            _insert_interval_sorted(machine_busy[s][m], (start[s], end[s]))

        # 更新 arm_intervals（持有区间）
        for k in range(L - 1):
            u, v = R[k], R[k + 1]
            u_act = STAGE2ACT[u][1]
            t_act = STAGE2ACT[v][0]
            arm = _which_arm(u_act, prefer_arm1_on_overlap) or _which_arm(t_act, prefer_arm1_on_overlap)
            if arm is None:
                continue
            hold = (proc_end[u], start[v])
            if hold[1] > hold[0] + 1e-12:
                _insert_interval_sorted(arm_intervals[arm], hold)

    return ops



if __name__ == "__main__":


    with open(r"C:\Users\khand\OneDrive\code\dqn\CT\data\DFS_data\taskC.toml", "rb") as f:
        cfg = tomllib.load(f)

    proc_time = {int(k): v for k, v in cfg["proc_time"].items()}
    capacity = {int(k): v for k, v in cfg["capacity"].items()}
    tau = cfg["timing"]["tau"]
    N_JOB1 = cfg["jobs"]['N_JOB1']
    N_JOB2 = cfg["jobs"]['N_JOB2']
    N_JOB = N_JOB1 + N_JOB2
    A = cfg["routes"]["A"]
    B = cfg["routes"]["B"]

    W = {}
    for k, v in cfg["windows"].items():
        u, w = k.split("->")
        W[(int(u), int(w))] = v


    routes = {i: (A if i <= N_JOB1 else B) for i in range(1, N_JOB + 1)}
    arrivals = {i:30*(i-1) for i in range(1,N_JOB+1)}
    ops = schedule_shift_search_window0_with_arrivals(proc_time, capacity,tau=tau,routes=routes,t1_min=0,arrivals=arrivals)
    out_path = "../../res/wxt.png"
    plot_gantt_hatched_residence(ops, proc_time, capacity, N_JOB, out_path,with_label=False)
    print("saved:", out_path)

    tmp = []
    for o in ops:
        if o.job == 2:
            tmp.append(o)

    seq = ops_to_actions(ops)