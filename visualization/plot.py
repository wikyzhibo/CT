from typing import Dict, List, Tuple
#from solutions.DFS.permit_left import _num_stages,Op
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from dataclasses import dataclass

@dataclass
class Op:
    job: int
    stage: int
    machine: int
    start: float
    proc_end: float
    end: float  # occupied until end (includes residence)
    is_arm: bool = False
    kind: int = -1


def _num_stages(proc_time: Dict[int, float]) -> int:
    S = max(proc_time.keys())
    for s in range(1, S + 1):
        if s not in proc_time:
            raise ValueError(f"proc_time missing stage {s}")
    return S

def plot_gantt_hatched_residence(
    ops: List[Op],
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    n_jobs: int,
    out_path: str,
    arm_info: dict,
    with_label = True,
    no_arm = True,
    policy: int = None
):
    S = _num_stages(proc_time)

    # 生成泳道 lanes: S1-M0, S1-M1, ..., S4-Mk
    lanes: List[Tuple[int, int, str]] = []
    for s in range(1, S + 1):
        for m in range(capacity[s]):
            lanes.append((s, m, f"S{s}-M{m}"))
    lane_index = {(s, m): i for i, (s, m, _) in enumerate(lanes)}

    # 完工时间
    t_max = max(op.end for op in ops) if ops else 0.0

    # job 颜色分配
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("turbo")
    job_color = {j: cmap((j - 1) / max(1, n_jobs - 1)) for j in range(1, n_jobs + 1)}

    # 泳道参数
    lane_h, lane_gap = 8, 4

    # 绘制泳道 lanes
    for (s, m, name) in lanes:
        idx = lane_index[(s, m)]
        y0 = idx * (lane_h + lane_gap)
        ax.add_patch(patches.Rectangle((0, y0), t_max, lane_h, fill=False, linewidth=0.5))
        ax.text(-5, y0 + lane_h / 2, name, ha="right", va="center", fontsize=9)

    # ops: dark processing + light hatched residence
    for op in ops:
        idx = lane_index[(op.stage, op.machine)]
        y0 = idx * (lane_h + lane_gap)

        coset = ['green','blue']
        if op.is_arm:
            color = coset[op.kind]
            ax.add_patch(Rectangle(
                (op.start, y0),
                max(0.0, op.proc_end - op.start),
                lane_h,
                facecolor=color,
                edgecolor="black",
                linewidth=0,
                alpha=0.5,
            ))
            continue
        else:
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
        if n_jobs > 50:
            with_label = False

        if with_label:
            ax.text(
                (op.start + op.end) / 2,
                y0 + lane_h / 2,
                f"J{op.job}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
            )

    if no_arm:
        y_top = len(lanes) * (lane_h + lane_gap)
        ax.set_ylim(-lane_gap, y_top + 5)

        ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
        ax.set_yticks([])
        ax.set_xlabel("Time")
        A = [1, 2, 3, 4, 5]
        B = [1, 4, 5]
        ax.set_title(f"{len(proc_time)}-Stage Mixed Routes (A:{A}, B:{B})|makespan={t_max}")
        # ax.grid(True, axis="x", linewidth=0.4)
        plt.tight_layout()

        policy_dict = {0: 'pdr', 1: 'random', 2: 'rl'}
        out_path = out_path + f"{policy_dict[policy]}_job{n_jobs}.png"
        print("save img in:", out_path)
        fig.savefig(out_path, dpi=200)

        return
    # =========================
    # Robot arm occupancy lanes (interval-overlap based)
    # =========================
    ARM1 = arm_info['ARM1']
    ARM2 = arm_info['ARM2']

    STAGE2ACT = arm_info['STAGE2ACT']

    def which_arm(acts: str):
        if isinstance(acts, list):
            for a in acts:
                if a in ARM1: return 1
                if a in ARM2: return 2
        else:
            if acts in ARM1: return 1
            if acts in ARM2: return 2
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
            l = a.end
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
                facecolor=color, edgecolor="black", alpha=0.8,linewidth=0
            ))

    # 3) 关键：把 ylim 拉到能容纳机械手两行
    y_top = arm_y_start + 2 * (arm_lane_h + arm_gap)
    ax.set_ylim(-lane_gap, y_top + 5)

    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    A = [1, 2, 3, 4, 5]
    B = [1, 4, 5]
    ax.set_title(f"{len(proc_time)}-Stage Mixed Routes (A:{A}, B:{B})|makespan={t_max}")
    #ax.grid(True, axis="x", linewidth=0.4)
    plt.tight_layout()

    policy_dict = {0:'pdr',1:'random',2:'rl'}
    out_path = out_path + f"{policy_dict[policy]}_job{n_jobs}.png"
    print("save img in:",out_path)
    fig.savefig(out_path, dpi=200)