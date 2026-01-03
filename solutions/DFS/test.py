import heapq
from typing import Dict, List, Tuple


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


if __name__ == "__main__":


    routes = {i: (A if i <= N_JOB1 else B) for i in range(1, N_JOB + 1)}
    ops = schedule_sequential_with_routes(proc_time, capacity, W, tau, routes)
    out_path = "../../results/v1.png"
    plot_gantt_hatched_residence(ops, proc_time, capacity, N_JOB, out_path,with_label=True)
    print("saved:", out_path)
