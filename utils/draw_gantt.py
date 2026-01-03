import numpy as np
import matplotlib.pyplot as plt

def action_start_end(actions, times):
    actions = np.asarray(actions).tolist()
    times = np.asarray(times, dtype=float).tolist()

    if len(times) == len(actions) + 1:
        # times 是边界时间
        starts = times[:-1]
        ends   = times[1:]
    elif len(times) == len(actions):
        # times 是累计“动作结束时刻”
        ends = times
        starts = [0.0] + times[:-1]
    else:
        raise ValueError(f"times长度应为N或N+1，但现在 len(actions)={len(actions)}, len(times)={len(times)}")

    return starts, ends

def build_pm_intervals(actions, times, pm_action_map):
    """
    pm_action_map: { 'PM7': (stop_ids_array, start_ids_array), ... }
      例如 'PM7': (array([3]), array([19])) 代表：
        动作ID=19 -> PM开始占用
        动作ID=3  -> PM释放
    """
    starts, ends = action_start_end(actions, times)

    pm_intervals = {}
    for pm, (start_ids, stop_ids) in pm_action_map.items():
        stop_set  = set(np.asarray(stop_ids).tolist())
        start_set = set(np.asarray(start_ids).tolist())

        stack = []      # 存放尚未匹配的“开始时刻”
        intervals = []  # (start_time, end_time)

        for i, a in enumerate(actions):
            if a in start_set:
                stack.append(starts[i])

            if a in stop_set:
                if not stack:
                    # 没有匹配到start：可能是数据里stop/start定义反了，或序列不完整
                    continue
                st = stack.pop()   # 一般不重叠，用LIFO即可
                et = ends[i]
                if et > st:
                    intervals.append((st, et))

        pm_intervals[pm] = sorted(intervals, key=lambda x: x[0])

    return pm_intervals

def plot_gantt(pm_intervals, title="PM Working Gantt"):
    pms = list(pm_intervals.keys())
    # 让显示顺序更自然：PM1,PM2,...,PM10
    def pm_key(name):
        import re
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 999
    pms.sort(key=pm_key)

    # 计算整体时间范围
    all_ends = [et for pm in pms for (_, et) in pm_intervals[pm]]
    makespan = max(all_ends) if all_ends else 0.0

    fig_h = max(2.0, 0.6 * len(pms) + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_height = 8
    y_gap = 4
    for idx, pm in enumerate(pms):
        y0 = idx * (y_height + y_gap)
        bars = [(st, et - st) for (st, et) in pm_intervals[pm]]
        if bars:
            ax.broken_barh(bars, (y0, y_height))  # 不指定颜色，走matplotlib默认

    ax.set_yticks([i * (y_height + y_gap) + y_height/2 for i in range(len(pms))])
    ax.set_yticklabels(pms)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.set_xlim(0, makespan * 1.02 if makespan > 0 else 1)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return fig, ax

