from typing import Dict, List, Optional, Tuple
#from solutions.DFS.permit_left import _num_stages,Op
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import patheffects
from dataclasses import dataclass
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
    stage_module_names: Dict[int, List[str]] = None,
    title_suffix: Optional[str] = None,
):
    _ = arm_info
    _ = no_arm
    if title_suffix:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ] + list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["axes.unicode_minus"] = False
    S = _num_stages(proc_time)

    proc_ops = [op for op in ops if not op.is_arm]
    if not proc_ops:
        raise ValueError("plot_gantt_hatched_residence: no non-arm operations to plot")

    # 默认的 stage -> 模块名称映射
    # stage 1: PM7, PM8; stage 2: LLC; stage 3: PM1-PM4; stage 4: LLD; stage 5: PM9, PM10
    if stage_module_names is None:
        stage_module_names = {
            1: ["PM7", "PM8"],
            2: ["LLC"],
            3: ["PM1", "PM2", "PM3", "PM4"],
            4: ["LLD"],
            5: ["PM9", "PM10"],
        }

    # 仅按非 ARM 工序生成泳道
    lane_pairs = sorted({(int(op.stage), int(op.machine)) for op in proc_ops})
    lanes: List[Tuple[int, int, str]] = []
    for s, m in lane_pairs:
        if s in stage_module_names and m < len(stage_module_names[s]):
            module_name = stage_module_names[s][m]
        else:
            module_name = f"M{m}"
        lanes.append((s, m, f"S{s}-M{m}({module_name})"))
    lane_index = {(s, m): i for i, (s, m, _) in enumerate(lanes)}

    # 完工时间
    t_max = max(op.end for op in proc_ops)

    # 根据时间跨度动态调整图像尺寸和分辨率
    # 时间越长，图像越宽，以保证短时间动作（如ARM的5s）可见
    fig_width = 30
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 统一为纯白背景
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    cmap = plt.get_cmap("turbo")
    # 修复：job 可能从 0 开始，需要包含所有可能的 job 值
    # 收集所有唯一的 job 值
    all_jobs = set(op.job for op in proc_ops if op.job >= 0)
    if all_jobs:
        min_job = min(all_jobs)
        max_job = max(all_jobs)
        job_range = max_job - min_job if max_job > min_job else 1
        job_color = {j: cmap((j - min_job) / job_range) if job_range > 0 else cmap(0.5) 
                     for j in all_jobs}
    else:
        # 如果没有有效的 job，使用默认颜色
        job_color = {0: cmap(0.5)}

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
    for op in proc_ops:
        idx = lane_index[(op.stage, op.machine)]
        y0 = idx * (lane_h + lane_gap)
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
        if n_jobs > 80:
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

    y_top = len(lanes) * (lane_h + lane_gap)
    ax.set_ylim(-lane_gap, y_top + 5)

    ax.set_xlim(0, t_max * 1.02 if t_max > 0 else 1.0)
    ax.set_yticks([])

    # 添加垂直网格线提高时间轴可读性
    # 主刻度：1000（显示在坐标轴上 + 大网格）
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    # 次刻度：200（只用于小网格，不显示在坐标轴）
    ax.xaxis.set_minor_locator(MultipleLocator(200))
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

    title = f"{S}-Stage|makespan={t_max:.1f}s"
    if title_suffix:
        title = f"{title}|{title_suffix}"
    ax.set_title(title, fontsize=17, color='#0F172A', weight='normal', pad=15)

    plt.tight_layout()

    policy_dict = {0: 'pdr', 1: 'random', 2: 'RL1', 3: 'RL2'}
    policy_key = 2 if policy is None else int(policy)
    out_path = out_path + f"{policy_dict[policy_key]}_job{n_jobs}.png"
    print("save img in:", out_path)
    # 提高分辨率到 300 dpi
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
