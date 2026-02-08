"""
Interval utilities for resource occupation management.

This module provides functions for managing time intervals in the resource
occupation timeline, including finding free slots and handling open intervals.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Interval:
    """
    Represents a time interval for resource occupation.
    Uses half-open interval [start, end).
    """
    start: int
    end: int
    tok_key: int = -1  # Token/job ID
    kind: int = -1  # Operation kind (0=PICK, 1=LOAD, 2=MOVE, etc.)
    from_loc: str = ""  # Source location
    to_loc: str = ""  # Destination location
    wafer_type: int = 0  # Wafer type (1=route C, 2=route D)


@dataclass
class ActionInfo:
    """Information about an action/transition."""
    t: int  # Transition ID
    fire_times: List[float]  # Firing times for each transition in chain
    t_name: str  # Transition name
    chain: List[str]  # Chain of transition names


def _insert_interval_sorted(intervals: List[Interval], new_interval: Interval) -> None:
    """
    Insert an interval into a sorted list of intervals.
    Maintains ascending order by start time.
    
    Args:
        intervals: List of intervals sorted by start time
        new_interval: Interval to insert
    """
    # Binary search for insertion point
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid].start < new_interval.start:
            lo = mid + 1
        else:
            hi = mid
    intervals.insert(lo, new_interval)


def _first_free_time_at(intervals: List[Interval], t: int, t2: int) -> int:
    """
    Find the earliest time cur >= t such that [cur, cur+dur) can be inserted
    without overlapping with existing intervals.
    
    This is used for closed intervals (e.g., ARM operations with known duration).
    
    Args:
        intervals: List of intervals sorted by start time, non-overlapping
        t: Earliest desired start time
        t2: End time (t2 - t = duration)
    
    Returns:
        Earliest feasible start time >= t
    """
    cur = int(t)
    dur = int(t2) - int(t)
    
    if dur <= 0 or not intervals:
        return cur
    
    # Binary search: find first interval with start > cur
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid].start <= cur:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    
    # Check if previous interval covers cur
    if i > 0 and intervals[i - 1].end > cur:
        cur = intervals[i - 1].end
    
    # Scan all intervals that conflict with [cur, cur+dur)
    while i < len(intervals):
        itv = intervals[i]
        end = cur + dur
        
        if itv.start >= end:  # No more conflicts
            break
        
        if itv.end <= cur:  # Interval is before cur, skip
            i += 1
            continue
        
        # Overlap detected: push cur to end of this interval
        cur = itv.end
        i += 1
    
    return cur


def _first_free_time_open(intervals: List[Interval], t: int) -> int:
    """
    Find the earliest time cur >= t such that no existing interval covers cur.
    
    This is used for open intervals (e.g., wafer enters a chamber but leave time
    is unknown until the next operation).
    
    Args:
        intervals: List of intervals sorted by start time
        t: Earliest desired start time
    
    Returns:
        Earliest time >= t not covered by any interval
    """
    if len(intervals) == 0:
        return t
    
    # Find the latest end time among all intervals that might cover t
    for itv in intervals:
        if itv.end >= t:
            t = itv.end
    
    return t
