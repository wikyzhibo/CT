"""
Machine utility functions for Timed Petri Net system.

This module provides helper functions for identifying and grouping machine resources.
"""

from typing import Optional, Dict, List

PARALLEL_GROUPS = {
    "PM1_4": ["PM1", "PM2", "PM3", "PM4"],
    "PM7_8": ["PM7", "PM8"],
    "PM9_10": ["PM9", "PM10"],
}


def parse_to_machine(t_name: str) -> Optional[str]:
    """
    Extract target machine name from transition name.
    
    Example: 
        ARM2_PICK__LLA_S2__TO__PM7 -> "PM7"
        
    Args:
        t_name: Transition name
        
    Returns:
        Machine name if found (starting with PM), else None
    """
    if "__TO__" not in t_name:
        return None
    to_part = t_name.split("__TO__")[-1]
    return to_part if to_part.startswith("PM") else None


def machine_group(machine: str) -> Optional[str]:
    """
    Get the parallel group name for a machine.
    
    Args:
        machine: Machine name (e.g., "PM1")
        
    Returns:
        Group name (e.g., "PM1_4") if found, else None
    """
    for g, ms in PARALLEL_GROUPS.items():
        if machine in ms:
            return g
    return None
