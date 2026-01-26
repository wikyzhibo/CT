from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class QueueItem:
    name: str
    enable_time: int

@dataclass
class Interval:
    start: int
    end: int          # 可为 INF_OCC 表示占用中
    tok_key: int
    kind: int = -1
    from_loc: str = ""   # 取的位置，如 "LLA_S2"
    to_loc: str = ""     # 放的位置，如 "PM7"

@dataclass
class Message:
    step: int
    tran_name: str  # 本次选择/发射的变迁名
    cur_time: int                 # 当前系统时间
    job: int
    cand: List[QueueItem]        # 当时的可选队列（已排序）
    def __repr__(self):
        return f'step={self.step}\t name={self.tran_name}\t time={self.cur_time}\t job={self.job}|{self.cand}'

def _is_time_slot_available(start: int, end: int, occupied_intervals: List[Tuple[int, int]]) -> bool:
    """
    检查时间段 [start, end) 是否与已有时间段重叠
    True表示可用（无重叠），False表示不可用
    """
    for occ_start, occ_end in occupied_intervals:
        if start < occ_end and end > occ_start:
            return False
        if end > occ_start and start < occ_end:
            return False
        if start < occ_start and end > occ_end:
            return False
    return True

def _insert_interval_sorted(lst: List[Interval], itv: Interval):
    import bisect
    i = bisect.bisect_right([x.start for x in lst], itv.start)
    lst.insert(i, itv)

def _first_free_time_at(intervals: List[Interval], t: int, t2: int) -> int:
    """给定某资源的占用区间（按 start 升序、互不重叠），返回 >=t 的最早可用时刻（允许插空）"""
    cur = int(t)
    for itv in intervals:
        #if cur < itv.start:
        #    return cur
        if itv.start <= t2 < itv.end:
            cur = itv.end
        if itv.start <= cur < itv.end:
            cur = itv.end
        if t2 < itv.start:
            return cur
    return cur

def _from_interval_find_time(earliest_start: int,
                             duration: int,
                             occupied_intervals: List[Tuple[int, int]]) -> int:
    """
    从 earliest_start 开始，找到最早的可用于执行 duration 时长的时刻
    """
    current_time = earliest_start
    end_time = current_time + duration

    # 如果没有已占用的时间段，直接返回
    if not occupied_intervals:
        return current_time

    # 尝试从 earliest_start 开始，逐步向后查找可用时间段
    max_iterations = 10000  # 防止无限循环
    for _ in range(max_iterations):
        if _is_time_slot_available(current_time, end_time, occupied_intervals):
            return current_time

        # 找到第一个结束时间 > current_time 的区间，将 current_time 设为该区间的结束时间
        next_start = None
        for occ_start, occ_end in occupied_intervals:
            if occ_end > current_time:
                next_start = occ_end
                break

        if next_start is None:
            # 所有占用区间都在 current_time 之前，当前时间可用
            return current_time

        current_time = next_start
        end_time = current_time + duration

    return -1  # 理论上不应到达这里