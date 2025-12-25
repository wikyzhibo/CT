from typing import Dict, List, Tuple
from solutions.DFS.permit_left import _num_stages,Op
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

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
    # =========================
    # Robot arm occupancy lanes (interval-overlap based)
    # =========================
    ARM1 = ["t3", "u3", "u6", "t7", "u7", "t8"]
    ARM2 = ["t5", "u6", "t6"]

    STAGE2ACT = {1: ("t3", "u3"), 2: ("t5", "u5"), 3: ("t6", "u6"), 4: ("t7", "u7"), 5: ("t8", "u8")}

    def which_arm(act: str):
        if act in ARM1: return 1
        if act in ARM2: return 2
        return None

    # 1) 先按 job 把 ops 排成工艺顺序，生成“持有区间”：[u_time, next_t_time)
    ops_by_job = {}
    for op in ops:
        ops_by_job.setdefault(op.job, []).append(op)
    for j in ops_by_job:
        ops_by_job[j].sort(key=lambda x: x.start)

    arm_intervals = {1: [], 2: []}  # each is list of (l,r)
    for j, oplist in ops_by_job.items():
        # 相邻工序：u 在前一道 proc_end，t 在下一道 start
        for a, b in zip(oplist[:-1], oplist[1:]):
            u_act = STAGE2ACT[a.stage][1]  # u*
            t_act = STAGE2ACT[b.stage][0]  # t*
            arm = which_arm(u_act) or which_arm(t_act)
            if arm is None:
                continue
            l = a.proc_end
            r = b.start
            if r > l + 1e-9:
                arm_intervals[arm].append((l, r))

    # 2) 画两条泳道：按重叠数决定颜色（1/2绿，>=3红）
    arm_lane_h = 6
    arm_gap = 6
    arm_y_start = len(lanes) * (lane_h + lane_gap) + 10

    def overlap_count(intervals, t):
        c = 0
        for l, r in intervals:
            if l <= t < r:
                c += 1
        return c

    for arm in [1, 2]:
        y0 = arm_y_start + (arm - 1) * (arm_lane_h + arm_gap)
        ax.text(-5, y0 + arm_lane_h / 2, f"ARM{arm}", ha="right", va="center", fontsize=9)

        # 用所有端点切分时间轴
        pts = sorted({0.0, t_max} | {x for itv in arm_intervals[arm] for x in itv})
        for t0, t1 in zip(pts[:-1], pts[1:]):
            if t1 <= t0:
                continue
            occ = overlap_count(arm_intervals[arm], t0 + 1e-12)
            if occ <= 0:
                continue

            color = "black"
            if occ >= 3:
                color = "red"
            elif occ == 2:
                color = "blue"
            elif occ == 1:
                color = 'green'

            ax.add_patch(Rectangle(
                (t0, y0), t1 - t0, arm_lane_h,
                facecolor=color, edgecolor="black", alpha=0.8
            ))

    # 3) 关键：把 ylim 拉到能容纳机械手两行
    y_top = arm_y_start + 2 * (arm_lane_h + arm_gap)
    ax.set_ylim(-lane_gap, y_top + 5)

    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    A = [1, 2, 3, 4, 5]
    B = [1, 4, 5]
    ax.set_title(f"{len(proc_time)}-Stage Mixed Routes (A:{A}, B:{B}) | dark=proc, hatched=residence|makespan={t_max}")
    #ax.grid(True, axis="x", linewidth=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)