"""
Shared utilities for Petri net implementations.

This module contains common data structures and helper functions used across
different Petri net implementations in the solutions package.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QueueItem:
    """Represents an item in the transition queue."""
    name: str
    enable_time: int


@dataclass
class Interval:
    """
    Represents a time interval for resource occupation.
    
    Attributes:
        start: Start time of the interval
        end: End time of the interval (can be INF_OCC for ongoing occupation)
        tok_key: Token key/ID
        kind: Type/kind of interval (default: -1)
        from_loc: Source location (e.g., "LLA_S2")
        to_loc: Destination location (e.g., "PM7")
        wafer_type: Wafer type (1 or 2, default: 0)
    """
    start: int
    end: int
    tok_key: int
    kind: int = -1
    from_loc: str = ""
    to_loc: str = ""
    wafer_type: int = 0


@dataclass
class Message:
    """
    Represents a log message for transition firing.
    
    Attributes:
        step: Step number
        tran_name: Name of the transition being fired
        cur_time: Current system time
        job: Job ID
        cand: List of candidate transitions (sorted)
    """
    step: int
    tran_name: str
    cur_time: int
    job: int
    cand: List[QueueItem]
    
    def __repr__(self):
        return f'step={self.step}\t name={self.tran_name}\t time={self.cur_time}\t job={self.job}|{self.cand}'


def _is_time_slot_available(start: int, end: int, occupied_intervals: List[Tuple[int, int]]) -> bool:
    """
    Check if time slot [start, end) overlaps with existing intervals.
    
    Args:
        start: Start time of the slot
        end: End time of the slot
        occupied_intervals: List of occupied time intervals
        
    Returns:
        True if available (no overlap), False if unavailable
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
    """
    Insert an interval into a sorted list maintaining order by start time.
    
    Args:
        lst: List of intervals sorted by start time
        itv: Interval to insert
    """
    import bisect
    i = bisect.bisect_right([x.start for x in lst], itv.start)
    lst.insert(i, itv)


def _from_interval_find_time(earliest_start: int,
                             duration: int,
                             occupied_intervals: List[Tuple[int, int]]) -> int:
    """
    Find the earliest available time to execute a task of given duration.
    
    Args:
        earliest_start: Earliest possible start time
        duration: Duration of the task
        occupied_intervals: List of occupied time intervals
        
    Returns:
        Earliest available start time, or -1 if not found
    """
    current_time = earliest_start
    end_time = current_time + duration

    # If no occupied intervals, return immediately
    if not occupied_intervals:
        return current_time

    # Try to find available time slot
    max_iterations = 10000  # Prevent infinite loop
    for _ in range(max_iterations):
        if _is_time_slot_available(current_time, end_time, occupied_intervals):
            return current_time

        # Find first interval ending after current_time
        next_start = None
        for occ_start, occ_end in occupied_intervals:
            if occ_end > current_time:
                next_start = occ_end
                break

        if next_start is None:
            # All intervals are before current_time
            return current_time

        current_time = next_start
        end_time = current_time + duration

    return -1  # Should not reach here
