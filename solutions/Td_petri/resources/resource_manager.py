"""
Resource manager for handling resource occupation and allocation.

This module manages the resource occupation timeline for both robotic arms
and processing modules in the Petri net system.
"""

from typing import List, Dict, Optional
from .interval_utils import Interval, _first_free_time_at, _first_free_time_open, _insert_interval_sorted


INF_OCC = 10**18  # Represents infinity for open intervals


class ResourceManager:
    """
    Manages resource occupation timelines for the Petri net system.
    
    Tracks occupation intervals for:
    - Robotic arms (ARM1, ARM2, ARM3)
    - Processing modules (PM7, PM8, PM1-PM4, PM9, PM10, LLC, LLD, etc.)
    """
    
    def __init__(self, arm_resources: Optional[List[str]] = None,
                 module_resources: Optional[List[str]] = None):
        """
        Initialize the resource manager.
        
        Args:
            arm_resources: List of arm resource names (default: ARM1, ARM2, ARM3)
            module_resources: List of module resource names
        """
        self.res_occ: Dict[str, List[Interval]] = {}
        self.open_mod_occ: Dict[tuple, Interval] = {}  # (module_name, job_id) -> Interval
        
        # Initialize arm resources
        if arm_resources is None:
            arm_resources = ["ARM1", "ARM2", "ARM3"]
        for arm in arm_resources:
            self.res_occ[arm] = []
        
        # Initialize module resources
        if module_resources is None:
            module_resources = ['PM7', 'PM8', 'LLA', 'LLB', 'LLC', 'LLD', 'AL',
                              'PM1', 'PM2', 'PM3', 'PM4', 'PM9', 'PM10']
        for module in module_resources:
            self.res_occ[module] = []
    
    def reset(self) -> None:
        """Reset all resource occupation timelines."""
        for key in self.res_occ:
            self.res_occ[key] = []
        self.open_mod_occ.clear()
    
    def allocate_resource(self, resource_name: str, start: int, end: int,
                         job_id: int = -1, kind: int = -1,
                         from_loc: str = "", to_loc: str = "",
                         wafer_type: int = 0) -> None:
        """
        Allocate a resource for a time interval.
        
        Args:
            resource_name: Name of the resource
            start: Start time
            end: End time (use INF_OCC for open intervals)
            job_id: Job/token ID
            kind: Operation kind (0=PICK, 1=LOAD, 2=MOVE)
            from_loc: Source location
            to_loc: Destination location
            wafer_type: Wafer type (1=route C, 2=route D)
        """
        occ = self.res_occ.setdefault(resource_name, [])
        itv = Interval(
            start=int(start),
            end=int(end),
            tok_key=job_id,
            kind=kind,
            from_loc=from_loc,
            to_loc=to_loc,
            wafer_type=wafer_type
        )
        _insert_interval_sorted(occ, itv)
        
        # Track open intervals for modules
        if end == INF_OCC:
            self.open_mod_occ[(resource_name, job_id)] = itv
    
    def close_open_interval(self, resource_name: str, job_id: int, end_time: int) -> None:
        """
        Close an open interval for a module.
        
        Args:
            resource_name: Name of the module
            job_id: Job/token ID
            end_time: Time to close the interval
        """
        key = (resource_name, job_id)
        if key in self.open_mod_occ:
            self.open_mod_occ[key].end = int(end_time)
            del self.open_mod_occ[key]
    
    def find_earliest_slot(self, resource_names: List[str], start_time: int,
                          duration: int, use_open_check: bool = False) -> int:
        """
        Find the earliest time slot that can accommodate the given duration
        across all specified resources.
        
        Args:
            resource_names: List of resource names to check
            start_time: Earliest desired start time
            duration: Duration of the operation
            use_open_check: If True, use open interval check (for last PROC in chain)
        
        Returns:
            Earliest feasible start time >= start_time
        """
        if not resource_names or duration <= 0:
            return int(start_time)
        
        t = int(start_time)
        
        # Iterative synchronization (usually converges in 1-3 rounds)
        for _ in range(50):
            t_new = t
            for r in resource_names:
                occ = self.res_occ.get(r, [])
                if not occ:
                    continue
                
                if use_open_check:
                    t_r = _first_free_time_open(occ, t)
                else:
                    t_r = _first_free_time_at(occ, t, t + duration)
                
                if t_r > t_new:
                    t_new = t_r
            
            # Converged
            if t_new == t:
                return t
            
            t = t_new
        
        return t
    
    def get_occupation(self, resource_name: str) -> List[Interval]:
        """
        Get the occupation timeline for a resource.
        
        Args:
            resource_name: Name of the resource
        
        Returns:
            List of occupation intervals
        """
        return self.res_occ.get(resource_name, [])
    
    def calculate_utilization(self, current_time: int,
                             window: Optional[int] = None,
                             tool_keys: Optional[List[str]] = None) -> float:
        """
        Calculate resource utilization.
        
        Args:
            current_time: Current system time
            window: Time window for calculation (None = from start)
            tool_keys: List of tools to include (None = all non-ARM resources)
        
        Returns:
            Average utilization across specified tools
        """
        t1 = float(current_time)
        if window is None:
            t0 = 0.0
        else:
            t0 = max(0.0, t1 - float(window))
        span = max(t1 - t0, 1e-9)
        
        if tool_keys is None:
            tool_keys = [k for k in self.res_occ.keys() if not k.startswith("ARM")]
        
        busy_sum = 0.0
        for k in tool_keys:
            occ = self.res_occ.get(k, [])
            for itv in occ:
                if itv.end > 100000:  # Open interval
                    itvend = t1
                else:
                    itvend = itv.end
                
                # Only count intervals that overlap with [t0, t1]
                if itv.start < t1 and itvend > t0:
                    overlap_start = max(itv.start, t0)
                    overlap_end = min(itvend, t1)
                    busy_sum += (overlap_end - overlap_start)
        
        denom = max(len(tool_keys) * span, 1e-9)
        util_sys = busy_sum / denom
        return util_sys
    
    def sync_start(self, res_names: List[str], t0: int, dur: int) -> int:
        """
        从 t0 开始，寻找一个最早时刻 t，使得：
        对所有资源 r ∈ res_names，区间 [t, t+dur) 都是空闲的。
        
        这是 find_earliest_slot 的别名，用于向后兼容。
        
        Args:
            res_names: 资源名列表
            t0: 起始时间
            dur: 持续时间
        
        Returns:
            最早可用时间 >= t0
        """
        return self.find_earliest_slot(res_names, t0, dur, use_open_check=False)
    
    def tool_keys(self) -> List[str]:
        """
        获取所有机台资源的名称（不包含机械手）。
        
        Returns:
            机台资源名列表
        """
        return [k for k in self.res_occ.keys() if not k.startswith("ARM")]

