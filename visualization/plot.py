from typing import Dict, List, Tuple
#from solutions.DFS.permit_left import _num_stages,Op
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import patheffects
from dataclasses import dataclass
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

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
    from_loc: str = ""   # 取的位置，如 "LLA_S2"
    to_loc: str = ""     # 放的位置，如 "PM7"


def _num_stages(proc_time: Dict[int, float]) -> int:
    S = max(proc_time.keys())
    for s in range(1, S + 1):
        if s not in proc_time:
            raise ValueError(f"proc_time missing stage {s}")
    return S


def _get_luminance(rgb: Tuple[float, float, float]) -> float:
    """
    计算颜色的相对亮度（用于对比度计算）
    RGB值应在0-1范围内
    """
    r, g, b = rgb[:3]  # 只取RGB，忽略alpha
    # 转换为线性空间
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _get_contrast_ratio(color1: Tuple[float, float, float], color2: Tuple[float, float, float]) -> float:
    """
    计算两个颜色之间的对比度比率
    返回对比度比率（WCAG标准：正常文本需要≥4.5:1）
    """
    lum1 = _get_luminance(color1)
    lum2 = _get_luminance(color2)
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def _get_text_color(bg_color: Tuple[float, float, float]) -> str:
    """
    根据背景色返回合适的文字颜色（白色或深色）
    确保对比度≥4.5:1
    """
    # 定义白色和深色文字
    white = (1.0, 1.0, 1.0)
    dark = (0.1, 0.1, 0.1)  # 接近黑色，确保高对比度
    
    # 计算对比度
    contrast_white = _get_contrast_ratio(bg_color, white)
    contrast_dark = _get_contrast_ratio(bg_color, dark)
    
    # 选择对比度更高的颜色
    if contrast_white >= contrast_dark and contrast_white >= 4.5:
        return "white"
    elif contrast_dark >= 4.5:
        return "#1A1A1A"  # 深灰色，确保高对比度
    else:
        # 如果都不够，使用描边增强对比度
        # 返回深色，后续会添加描边
        return "#1A1A1A"

def plot_gantt_hatched_residence(
    ops: List[Op],
    proc_time: Dict[int, float],
    capacity: Dict[int, int],
    n_jobs: int,
    out_path: str,
    arm_info: dict,
    with_label = True,
    no_arm = True,
    policy: int = None,
    stage_module_names: Dict[int, List[str]] = None
):
    S = _num_stages(proc_time)

    # 默认的 stage -> 模块名称映射
    # stage 1: PM7, PM8; stage 2: LLC; stage 3: PM1-PM4; stage 4: LLD; stage 5: PM9, PM10
    if stage_module_names is None:
        stage_module_names = {
            1: ["PM7", "PM8"],
            2: ["LLC"],
            3: ["PM1", "PM2", "PM3", "PM4"],
            4: ["LLD"],
            5: ["PM9", "PM10"],
            6: ["ARM2"],
            7: ["ARM3"],
        }

    # 生成泳道 lanes: S1-M0(PM7), S1-M1(PM8), ...
    lanes: List[Tuple[int, int, str]] = []
    for s in range(1, S + 1):
        for m in range(capacity[s]):
            # 获取具体模块名称
            if s in stage_module_names and m < len(stage_module_names[s]):
                module_name = stage_module_names[s][m]
            else:
                module_name = f"M{m}"
            lanes.append((s, m, f"S{s}-M{m}({module_name})"))
    lane_index = {(s, m): i for i, (s, m, _) in enumerate(lanes)}

    # 完工时间
    t_max = max(op.end for op in ops) if ops else 0.0

    # 根据时间跨度动态调整图像尺寸和分辨率
    # 时间越长，图像越宽，以保证短时间动作（如ARM的5s）可见
    fig_width = 30
    fig_height = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 设置浅色背景，确保所有元素清晰可见
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#FFFFFF')

    cmap = plt.get_cmap("turbo")
    # 修复：job 可能从 0 开始，需要包含所有可能的 job 值
    # 收集所有唯一的 job 值
    all_jobs = set(op.job for op in ops if op.job >= 0)
    if all_jobs:
        min_job = min(all_jobs)
        max_job = max(all_jobs)
        job_range = max_job - min_job if max_job > min_job else 1
        job_color = {j: cmap((j - min_job) / job_range) if job_range > 0 else cmap(0.5) 
                     for j in all_jobs}
    else:
        # 如果没有有效的 job，使用默认颜色
        job_color = {0: cmap(0.5)}

    # ARM 路径颜色分配：收集所有唯一的 (from_loc, to_loc) 组合
    arm_paths = set()
    for op in ops:
        if op.is_arm and op.from_loc and op.to_loc:
            arm_paths.add((op.from_loc, op.to_loc))
    arm_paths = sorted(arm_paths)  # 排序保证颜色一致性
    arm_cmap = plt.get_cmap("tab20")
    arm_path_color = {path: arm_cmap(i / max(1, len(arm_paths))) for i, path in enumerate(arm_paths)}

    # 泳道参数
    lane_h, lane_gap = 4, 1

    # 绘制泳道 lanes
    for (s, m, name) in lanes:
        idx = lane_index[(s, m)]
        y0 = idx * (lane_h + lane_gap)
        ax.add_patch(patches.Rectangle((0, y0), t_max, lane_h, fill=False, 
                                       linewidth=1.2, edgecolor='#1E293B', alpha=0.3))
        ax.text(-5, y0 + lane_h / 2, name, ha="right", va="center", 
               fontsize=16, color='#1E293B', weight='normal')

    # ops: 深色代表加工段 + 浅色斜线代表驻留段
    for op in ops:
        idx = lane_index[(op.stage, op.machine)]
        y0 = idx * (lane_h + lane_gap)

        if op.is_arm:
            # 根据取放路径分配颜色
            path_key = (op.from_loc, op.to_loc)
            if path_key in arm_path_color:
                color = arm_path_color[path_key]
            else:
                # 回退：使用旧的 kind 逻辑
                coset = ['green', 'blue']
                color = coset[op.kind] if 0 <= op.kind < len(coset) else 'gray'
            ax.add_patch(Rectangle(
                (op.start, y0),
                max(0.0, op.proc_end - op.start),
                lane_h,
                facecolor=color,
                edgecolor="#1E293B",
                linewidth=0,
                alpha=1,
            ))
            continue
        else:
            # 获取作业颜色，如果不存在则使用默认颜色
            if op.job in job_color:
                color = job_color[op.job]
            else:
                # 回退：使用灰色作为默认颜色
                color = 'gray'
                print(f"警告: job {op.job} 不在 job_color 中，使用默认颜色")

        # processing (dark)
        # 绘制加工段填充（无边框，边框由统一外边框绘制）
        ax.add_patch(Rectangle(
            (op.start, y0),
            max(0.0, op.proc_end - op.start),
            lane_h,
            facecolor=color,
            edgecolor="none",  # 不绘制内部边框
            alpha=0.9,
        ))

        # residence (light + hatch)
        if op.end > op.proc_end + 1e-9:
            # 绘制驻留段填充（无边框，边框由统一外边框绘制）
            ax.add_patch(Rectangle(
                (op.proc_end, y0),
                op.end - op.proc_end,
                lane_h,
                facecolor=color,
                edgecolor="0.6",
                alpha=0.3,  # 降低透明度，使斜线更明显，与加工段形成更明显对比
                hatch="xxx",  # 使用交叉线图案，比///更明显，提高可见性
            ))
        
        # 添加统一外边框，包裹整个区间（start到end），形成整体感
        # 这样加工段和驻留段看起来像一个统一的整体
        if op.end > op.start:
            ax.add_patch(Rectangle(
                (op.start, y0),
                op.end - op.start,
                lane_h,
                fill=False,
                edgecolor="#1E293B",
                linewidth=0.8,
                zorder=10,  # 确保在外层
            ))

        # completion line at proc_end
        #ax.plot([op.proc_end, op.proc_end], [y0, y0 + lane_h], color="black", linewidth=1.5)

        # label
        if n_jobs > 50:
            with_label = False

        if with_label:
            # 根据背景色动态选择文字颜色，确保对比度≥4.5:1
            # 使用加工段的颜色作为背景色参考
            text_color = _get_text_color(color[:3] if len(color) >= 3 else color)
            
            # 如果对比度不足，添加描边增强可读性
            if text_color == "#1A1A1A":
                # 深色文字，添加白色描边
                ax.text(
                    (op.start + op.end) / 2,
                    y0 + lane_h / 2,
                    f"J{op.job}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color,
                    weight='bold',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='white', alpha=0.8)]
                )
            else:
                # 白色文字，添加深色描边
                ax.text(
                    (op.start + op.end) / 2,
                    y0 + lane_h / 2,
                    f"J{op.job}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color,
                    weight='bold',
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground='#1A1A1A', alpha=0.6)]
                )

    if no_arm:
        y_top = len(lanes) * (lane_h + lane_gap)
        ax.set_ylim(-lane_gap, y_top + 5)

        ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
        ax.set_yticks([])
        
        # 添加垂直网格线提高时间轴可读性
        # 主刻度：500（显示在坐标轴上 + 大网格）
        ax.xaxis.set_major_locator(MultipleLocator(500))
        # 次刻度：50（只用于小网格，不显示在坐标轴）
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        # 大网格（主刻度）
        ax.grid(True, axis='x', which='major',
                linestyle='--', linewidth=0.6, alpha=0.4, color='#94A3B8')
        # 小网格（次刻度）
        ax.grid(True, axis='x', which='minor',
                linestyle=':', linewidth=0.4, alpha=0.4, color='#94A3B8')
        # 只显示主刻度标签（默认就是这样，这句是保险）
        ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
        ax.set_axisbelow(True)
        
        # 优化时间轴标签
        ax.set_xlabel("Time (s)", fontsize=14, color='#1E293B', weight='normal')
        
        # 设置主要时间刻度
        # 主刻度文字格式（替代 set_xticklabels）
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.tick_params(axis='x', which='major', labelsize=11, colors='#475569')
        
        A = [1, 2, 3, 4, 5]
        B = [1, 4, 5]
        ax.set_title(f"{len(proc_time)}-Stage|makespan={t_max:.1f}s",
                    fontsize=17, color='#0F172A', weight='normal', pad=15)

        # 添加 ARM 路径的图例
        if arm_paths:
            legend_handles = []
            for path in arm_paths:
                from_loc, to_loc = path
                color = arm_path_color[path]
                label = f"{from_loc} → {to_loc}"
                handle = patches.Patch(facecolor=color, edgecolor='black', alpha=0.7, label=label)
                legend_handles.append(handle)
            # 图例放在图外右侧，优化样式
            legend = ax.legend(handles=legend_handles, title="ARM path", loc='upper left',
                              bbox_to_anchor=(1.01, 1), fontsize=13, title_fontsize=14,
                              frameon=True, fancybox=True, shadow=False,
                              framealpha=0.95, edgecolor='#E2E8F0', facecolor='#FFFFFF')
            legend.get_title().set_color('#1E293B')
            legend.get_title().set_weight('normal')
            for text in legend.get_texts():
                text.set_color('#475569')

        plt.tight_layout()

        policy_dict = {0: 'pdr', 1: 'random', 2: 'RL1', 3: 'RL2'}
        out_path = out_path + f"{policy_dict[policy]}_job{n_jobs}.png"
        print("save img in:", out_path)
        # 提高分辨率到 300 dpi
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

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
        ax.text(-5, y0 + arm_lane_h / 2, f"ARM{arm}", ha="right", va="center", 
               fontsize=14, color='#1E293B', weight='normal')

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
                facecolor=color, edgecolor="#1E293B", alpha=0.8, linewidth=0.5
            ))

    # 3) 关键：把 ylim 拉到能容纳机械手两行
    y_top = arm_y_start + 2 * (arm_lane_h + arm_gap)
    ax.set_ylim(-lane_gap, y_top + 5)

    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
    ax.set_yticks([])
    
    # 添加垂直网格线提高时间轴可读性
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3, color='#94A3B8')
    
    # 优化时间轴标签
    ax.set_xlabel("Time (s)", fontsize=14, color='#1E293B', weight='normal')
    
    # 设置主要时间刻度
    if t_max > 0:
        major_ticks = np.arange(0, t_max * 1.02, max(50, t_max / 30))
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f"{int(t)}" for t in major_ticks], fontsize=11, color='#475569')
    

    ax.set_title(f"{len(proc_time)}-Stage |makespan={t_max:.1f}s",
                fontsize=17, color='#0F172A', weight='normal', pad=15)

    # 添加 ARM 路径的图例
    if arm_paths:
        legend_handles = []
        for path in arm_paths:
            from_loc, to_loc = path
            color = arm_path_color[path]
            label = f"{from_loc} → {to_loc}"
            handle = patches.Patch(facecolor=color, edgecolor='black', alpha=0.7, label=label)
            legend_handles.append(handle)
        # 图例放在图外右侧，优化样式
        legend = ax.legend(handles=legend_handles, title="ARM 路径", loc='upper left',
                         bbox_to_anchor=(1.01, 1), fontsize=12, title_fontsize=13,
                         frameon=True, fancybox=True, shadow=False,
                         framealpha=0.95, edgecolor='#E2E8F0', facecolor='#FFFFFF')
        legend.get_title().set_color('#1E293B')
        legend.get_title().set_weight('normal')
        for text in legend.get_texts():
            text.set_color('#475569')

    plt.tight_layout()

    policy_dict = {0:'pdr',1:'random',2:'rl'}
    out_path = out_path + f"{policy_dict[policy]}_job{n_jobs}.png"
    print("save img in:",out_path)
    # 提高分辨率到 300 dpi
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)