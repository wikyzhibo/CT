import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# =============================
# Parameters
# =============================
proc_time = {1: 70, 2: 600, 3: 70, 4: 200,5:70}
capacity  = {1: 2, 2: 4, 3: 1, 4: 2, 5:1}
tau = 15

# Windows:
# stage1/stage2/stage4 window=15; stage3 no window
# -> interpret as edge windows:
#    1->2:15, 2->3:15, 1->4:15, 3->4:INF(no constraint)
INF = 10**9
W = {(1, 2): 0, (2, 3): 0, (1, 4): 0, (3, 4): 0, (4,5):0}

# Job pattern: A,A,B,A,A,B (6 jobs)
# A: 1->2->3->4 ; B: 1->4
A = [1,2,3,4,5]
B = [1,4,5]


N_JOB1 = 25
N_JOB2 = 50
N_JOB = N_JOB1 + N_JOB2
'''
    26: [1,4],
    27: [1,4],
    28: [1,4],
    29: [1,4],
    30: [1,4],
    31: [1,4],
    32: [1,4],
    33: [1,4],
    34: [1, 4],
    35: [1, 4],
    36: [1, 4],
    37: [1, 4],
    38: [1, 4],
    39: [1, 4],
    40: [1, 4],
    41: [1, 4],
    42: [1, 4],
    43: [1, 4],
    44: [1, 4],
    45: [1, 4],
    46: [1, 4],
    47: [1, 4],
    48: [1, 4],
    49: [1, 4],
    50: [1, 4],
    51: [1, 4],
    52: [1, 4],
    53: [1, 4],
    54: [1, 4],
    55: [1, 4],
    56: [1, 4],
    57: [1, 4],
    58: [1, 4],
    59: [1, 4],
    60: [1, 4],
    61: [1, 4],
    62: [1, 4],
    63: [1, 4],
    64: [1, 4],
    65: [1, 4],
    66: [1, 4],
    67: [1, 4],
    68: [1, 4],
    69: [1, 4],
    70: [1, 4],
    71: [1, 4],
    72: [1, 4],
    73: [1, 4],
    74: [1, 4],
    75: [1, 4],
    '''



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


def schedule_sequential_with_routes(
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    W: Dict[Tuple[int, int], float],
    tau: float,
    routes: Dict[int, List[int]],
) -> List[Op]:
    """
    Sequential: job1 route, then job2 route, ...
    Each stage picks earliest-available machine.
    Window on edge (u->v): start[v] <= finish[u] + tau + W[(u,v)]
    Missing window -> INF.

    Residence occupation:
      if u followed by v in route: end[u] = start[v] - tau
      final stage: end[last] = proc_end[last] (no residence)
    """
    S = _num_stages(proc_time)

    # heap[s] : (avail_time, machine_id)
    heaps = {s: [(0.0, m) for m in range(capacity[s])] for s in range(1, S + 1)}
    for s in range(1, S + 1):
        heapq.heapify(heaps[s])

    ops: List[Op] = []

    for job in sorted(routes.keys()):
        R = routes[job]
        L = len(R)

        slot = {x: heaps[x][0][0] for x in R}

        # backward lower bound on this route
        lb = {R[-1]: 0.0}
        for k in range(L - 2, -1, -1):
            u, v = R[k], R[k + 1]
            w = W.get((u, v), INF)
            lb[u] = max(
                slot[u],
                slot[v] - (proc_time[u] + tau + w),
                lb[v]   - (proc_time[u] + tau + w),
            )

        start, proc_end, end = {}, {}, {}
        mid, avail_fixed = {}, {}

        # forward schedule
        for k in range(L):
            s = R[k]
            avail_s, m = heapq.heappop(heaps[s])
            mid[s] = m
            avail_fixed[s] = avail_s

            if k == 0:
                start[s] = max(avail_s, lb.get(s, 0.0), 0.0)
            else:
                prev = R[k - 1]
                arrive = proc_end[prev] + tau
                start[s] = max(avail_s, arrive, lb.get(s, 0.0))

                # enforce window on edge prev->s
                w_prev = W.get((prev, s), INF)
                if start[s] > arrive + w_prev:
                    # push prev later so arrive >= start[s] - w_prev
                    needed_arrive = start[s] - w_prev
                    needed_finish_prev = needed_arrive - tau
                    needed_start_prev = needed_finish_prev - proc_time[prev]

                    start[prev] = max(start[prev], needed_start_prev, avail_fixed[prev])
                    proc_end[prev] = start[prev] + proc_time[prev]

                    # cascade recompute from current k to end of route (machines fixed)
                    for t in range(k, L):
                        cur = R[t]
                        pre = R[t - 1]
                        arrive_cur = proc_end[pre] + tau
                        start[cur] = max(start[cur], arrive_cur, lb.get(cur, 0.0), avail_fixed[cur])
                        proc_end[cur] = start[cur] + proc_time[cur]

                        w_edge = W.get((pre, cur), INF)
                        if start[cur] > arrive_cur + w_edge:
                            raise RuntimeError(f"Job{job}: infeasible window on edge {pre}->{cur}")

            proc_end[s] = start[s] + proc_time[s]

        # machine occupation ends
        for k in range(L - 1):
            u, v = R[k], R[k + 1]
            end[u] = start[v] - tau
        end[R[-1]] = proc_end[R[-1]]  # last stage ends at proc_end

        # commit
        for s in R:
            ops.append(Op(job, s, mid[s], start[s], proc_end[s], end[s]))
            heapq.heappush(heaps[s], (end[s], mid[s]))

    return ops


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
    out_path = "gantt_mixed_routes_6jobs.png"
    plot_gantt_hatched_residence(ops, proc_time, capacity, N_JOB, out_path,with_label=False)
    print("saved:", out_path)
