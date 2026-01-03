from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
import bisect


from solutions.DFS.test import _num_stages,schedule_sequential_with_routes

# =============================
# Parameters
# =============================
proc_time = {1: 70, 2: 600, 3: 70, 4: 200,5:70}
capacity  = {1: 2, 2: 4, 3: 1, 4: 2, 5:1}
tau = 10

# Windows:
# stage1/stage2/stage4 window=15; stage3 no window
# -> interpret as edge windows:
#    1->2:15, 2->3:15, 1->4:15, 3->4:INF(no constraint)
INF = 10**9
W = {(1, 2): 0, (2, 3): 0, (1, 4): 0, (3, 4): 0, (4,5):0}

# A: 1->2->3->4 ; B: 1->4
A = [1,2,3,4,5]
B = [1,4,5]

N_JOB1 = 4
N_JOB2 = 8
N_JOB = N_JOB1 + N_JOB2

@dataclass
class Op:
    job: int
    stage: int
    machine: int
    start: float
    proc_end: float
    end: float  # occupied until end (includes residence)





# --------- interval helpers ---------
def _overlap(a1, b1, a2, b2) -> bool:
    # half-open overlap check: [a1,b1) vs [a2,b2)
    return not (b1 <= a2 or b2 <= a1)

def _fits_interval(busy_list: List[Tuple[float, float]], a: float, b: float) -> bool:
    # busy_list must be sorted by start
    # quick reject by scanning; can be optimized with bisect if needed
    for x, y in busy_list:
        if x >= b:
            break
        if _overlap(a, b, x, y):
            return False
    return True

def build_busy(ops) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """(stage,machine) -> sorted list of busy segments [start,end)"""
    busy: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for op in ops:
        busy.setdefault((op.stage, op.machine), []).append((op.start, op.end))
    for k in busy:
        busy[k].sort()
    return busy

def remove_job_from_busy(busy, ops, job_id: int):
    removed = []
    for op in ops:
        if op.job != job_id:
            continue
        key = (op.stage, op.machine)
        seg = (op.start, op.end)
        if key in busy:
            # remove one matching seg
            try:
                busy[key].remove(seg)
                removed.append((key, seg))
            except ValueError:
                pass
    for key, _ in removed:
        busy[key].sort()
    return removed

def restore_job_to_busy(busy, removed):
    for key, seg in removed:
        busy.setdefault(key, []).append(seg)
        busy[key].sort()

def insert_seg_sorted(busy_list: List[Tuple[float, float]], seg: Tuple[float, float]):
    # keep sorted by start
    i = bisect.bisect_left(busy_list, seg)
    busy_list.insert(i, seg)

# --------- window check for shifted plan ---------
def check_windows(route: List[int],
                  start: Dict[int, float],
                  proc_end: Dict[int, float],
                  W: Dict[Tuple[int,int], float],
                  tau: float) -> bool:
    for k in range(len(route) - 1):
        u, v = route[k], route[k+1]
        arrive = proc_end[u] + tau
        w = W.get((u, v), INF)
        if start[v] < arrive - 1e-9:
            return False
        if start[v] > arrive + w + 1e-9:
            return False
    return True

# --------- choose machines for a fixed shifted timeline ---------
def assign_machines_for_shift(route: List[int],
                              capacity: Dict[int, int],
                              busy: Dict[Tuple[int,int], List[Tuple[float,float]]],
                              start: Dict[int, float],
                              occ_end: Dict[int, float]) -> Optional[Dict[int, int]]:
    """
    For each stage s in route, choose any machine m such that [start[s], occ_end[s]) fits.
    Greedy: pick the first feasible machine.
    """
    mach = {}
    for s in route:
        a, b = start[s], occ_end[s]
        chosen = None
        for m in range(capacity[s]):
            bl = busy.get((s, m), [])
            if _fits_interval(bl, a, b):
                chosen = m
                break
        if chosen is None:
            return None
        mach[s] = chosen
    return mach

# --------- earliest x1 candidate (must be earlier than baseline) ---------
def earliest_x1_before_baseline(stage1: int,
                                p1: float,
                                base_start1: float,
                                capacity: Dict[int, int],
                                busy: Dict[Tuple[int,int], List[Tuple[float,float]]],
                                earliest_lb: float = 0.0) -> float:
    """
    Find a reasonable earliest x1 < base_start1:
    - We don't want x1 to go to 0; use earliest_lb as a floor (you can pass a route-based LB here).
    - We just return earliest_lb if it's < base_start1; actual feasibility will be tested in the loop.
    """
    return min(max(earliest_lb, 0.0), base_start1 - 1e-9)

# --------- main: shift-and-best_res insertion for one job ---------
def try_shift_insert_job(
    job_id: int,
    route: List[int],
    ops,  # List[Op]
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    W: Dict[Tuple[int,int], float],
    tau: float,
    step: float = 30.0,
    earliest_lb_for_x1: float = 0.0,   # IMPORTANT: set this to avoid "0-1" issue
) -> Tuple[bool, List]:
    """
    Implements:
      - Compute baseline relative offsets delta_k from current ops for this job
      - Search x1 from an early value up to baseline start1, increasing by 'step'
      - Keep all later ops at fixed relative offsets (start_k = x1 + delta_k)
      - Allow changing machines: each op can pick any machine that fits its occupied interval
      - If feasible (windows + no overlap), commit and return updated ops

    Returns: (improved?, new_ops)
    """
    # baseline map for this job (by stage)
    job_ops = {op.stage: op for op in ops if op.job == job_id}
    if any(s not in job_ops for s in route):
        raise ValueError(f"job {job_id} missing stages in ops for route {route}")

    base_start1 = job_ops[route[0]].start
    # relative offsets from baseline
    delta = {s: job_ops[s].start - base_start1 for s in route}

    # build busy from current schedule, but remove this job
    busy = build_busy(ops)
    removed = remove_job_from_busy(busy, ops, job_id)

    # search range for x1
    x1 = earliest_x1_before_baseline(route[0], proc_time[route[0]], base_start1,
                                    capacity, busy, earliest_lb=earliest_lb_for_x1)
    x1_max = base_start1  # stop when reaching baseline

    improved = False
    best_plan = None

    # iterate x1 forward by step until baseline
    while x1 < x1_max - 1e-9:
        start = {}
        proc_end = {}
        occ_end = {}

        # build shifted timeline (keep relative offsets)
        for k, s in enumerate(route):
            start[s] = x1 + delta[s]
            proc_end[s] = start[s] + proc_time[s]
            if k < len(route) - 1:
                next_s = route[k + 1]
                occ_end[s] = (x1 + delta[next_s]) - tau  # occupied until start[next]-tau
            else:
                occ_end[s] = proc_end[s]

            # basic sanity: occupied interval must be non-negative length
            if occ_end[s] < start[s] - 1e-9:
                start = None
                break
        if start is None:
            x1 += step
            continue

        # check window constraints
        if not check_windows(route, start, proc_end, W, tau):
            x1 += step
            continue

        # choose machines (allow change)
        mach = assign_machines_for_shift(route, capacity, busy, start, occ_end)
        if mach is None:
            x1 += step
            continue

        # feasible => commit first feasible (closest to left busy by your step search)
        best_plan = (start, proc_end, occ_end, mach)
        improved = True
        break

    # commit (or restore baseline)
    new_ops = [op for op in ops if op.job != job_id]
    if improved and best_plan is not None:
        start, proc_end, occ_end, mach = best_plan
        for s in route:
            op = job_ops[s]
            new_ops.append(op.__class__(job_id, s, mach[s], start[s], proc_end[s], occ_end[s]))
            insert_seg_sorted(busy.setdefault((s, mach[s]), []), (start[s], occ_end[s]))
    else:
        # restore old ops
        restore_job_to_busy(busy, removed)
        for s in route:
            new_ops.append(job_ops[s])

    return improved, new_ops






def plot_gantt_hatched_residence(
    ops: List[Op],
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    n_jobs: int,
    out_path: str,
    with_label = True,
):
    S = _num_stages(proc_time)

    # lanes: S1-M0, S1-M1, ..., S4-Mk
    lanes: List[Tuple[int, int, str]] = []
    for s in range(1, S + 1):
        for m in range(capacity[s]):
            lanes.append((s, m, f"S{s}-M{m}"))
    lane_index = {(s, m): i for i, (s, m, _) in enumerate(lanes)}

    t_max = max(op.end for op in ops) if ops else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    job_color = {j: cmap((j - 1) % 10) for j in range(1, n_jobs + 1)}

    lane_h, lane_gap = 8, 4

    # lanes
    for (s, m, name) in lanes:
        idx = lane_index[(s, m)]
        y0 = idx * (lane_h + lane_gap)
        ax.add_patch(patches.Rectangle((0, y0), t_max, lane_h, fill=False, linewidth=0.5))
        ax.text(-5, y0 + lane_h / 2, name, ha="right", va="center", fontsize=9)

    # ops: dark processing + light hatched residence
    for op in ops:
        idx = lane_index[(op.stage, op.machine)]
        y0 = idx * (lane_h + lane_gap)
        color = job_color[op.job]

        # processing (dark)
        ax.add_patch(Rectangle(
            (op.start, y0),
            max(0.0, op.proc_end - op.start),
            lane_h,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        ))

        # residence (light + hatch)
        if op.end > op.proc_end + 1e-9:
            ax.add_patch(Rectangle(
                (op.proc_end, y0),
                op.end - op.proc_end,
                lane_h,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.35,
                hatch="///",
            ))

        # completion line at proc_end
        #ax.plot([op.proc_end, op.proc_end], [y0, y0 + lane_h], color="black", linewidth=1.5)

        # label
        if with_label:
            ax.text(
                (op.start + op.end) / 2,
                y0 + lane_h / 2,
                f"J{op.job}-S{op.stage}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )



    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
    ax.set_ylim(-lane_gap, len(lanes) * (lane_h + lane_gap))
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(f"{len(proc_time)}-Stage Mixed Routes (A:{A}, B:{B}) | dark=proc, hatched=residence|makespan={t_max}")
    #ax.grid(True, axis="x", linewidth=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)



if __name__ == "__main__":


    routes = {i: (A if i <= N_JOB1 else B) for i in range(1, N_JOB + 1)}
    ops = schedule_sequential_with_routes(proc_time, capacity, W, tau, routes)

    # 给每个 job 尝试插空（earliest_lb_for_x1 你可以传一个“路线下界”，避免跑到 0）
    for j in sorted(routes.keys()):
        improved, ops = try_shift_insert_job(
            job_id=j,
            route=routes[j],
            ops=ops,
            proc_time=proc_time,
            capacity=capacity,
            W=W,
            tau=tau,
            step=30.0,
            earliest_lb_for_x1=0.0,  # 建议你用更强的下界替代 0
        )
    out_path = "../../results/re_chedule.png"
    plot_gantt_hatched_residence(ops, proc_time, capacity, N_JOB, out_path, with_label=False)
