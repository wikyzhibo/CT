"""
Path registry for managing wafer routes through the Petri net.

This module provides the SINGLE SOURCE OF TRUTH for path definitions,
eliminating the duplication between _init_path and _build_rl_action_space.
"""

from typing import List, Tuple, Dict


class PathRegistry:
    """
    Central registry for all path definitions in the Petri net system.
    
    This class is the authoritative source for:
    - Route C (pathC): LP1 → AL → LLA_S2 → PM7/PM8 → LLC → PM1-4 → LLD → PM9/10 → LLB → LP_done
    - Route D (pathD): LP2 → AL → LLA_S2 → PM7/PM8 → PM9/10 → LLB → LP_done
    """
    
    def __init__(self):
        """Initialize the path registry with route definitions."""
        self._pathC = self._define_path_c()
        self._pathD = self._define_path_d()
    
    @property
    def pathC(self) -> List[List[List[str]]]:
        """Get Route C definition (LP1 full route)."""
        return self._pathC
    
    @property
    def pathD(self) -> List[List[List[str]]]:
        """Get Route D definition (LP2 simplified route)."""
        return self._pathD
    
    def _define_path_c(self) -> List[List[List[str]]]:
        """
        Define Route C: Full processing route from LP1.
        
        Structure: List[stage] where each stage is List[branch] and each branch is List[transition_name]
        Parallel stages have multiple branches.
        """
        return [
            # Stage 0: LP1 → AL
            [['ARM1_PICK__LP1__TO__AL', 'ARM1_MOVE__LP1__TO__AL', 'ARM1_LOAD__LP1__TO__AL', 'PROC__AL']],
            
            # Stage 1: AL → LLA_S2
            [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2', 'PROC__LLA_S2']],
            
            # Stage 2: LLA_S2 → PM7/PM8 → LLC (parallel choice)
            [
                ['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7', 'PROC__PM7',
                 'ARM2_PICK__PM7__TO__LLC', 'ARM2_MOVE__PM7__TO__LLC', 'ARM2_LOAD__PM7__TO__LLC', 'PROC__LLC'],
                ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8', 'PROC__PM8',
                 'ARM2_PICK__PM8__TO__LLC', 'ARM2_MOVE__PM8__TO__LLC', 'ARM2_LOAD__PM8__TO__LLC', 'PROC__LLC']
            ],
            
            # Stage 3: LLC → PM1/PM2/PM3/PM4 → LLD (parallel choice)
            [
                ['ARM3_PICK__LLC__TO__PM1', 'ARM3_MOVE__LLC__TO__PM1', 'ARM3_LOAD__LLC__TO__PM1', 'PROC__PM1',
                 'ARM3_PICK__PM1__TO__LLD', 'ARM3_MOVE__PM1__TO__LLD', 'ARM3_LOAD__PM1__TO__LLD', 'PROC__LLD'],
                ['ARM3_PICK__LLC__TO__PM2', 'ARM3_MOVE__LLC__TO__PM2', 'ARM3_LOAD__LLC__TO__PM2', 'PROC__PM2',
                 'ARM3_PICK__PM2__TO__LLD', 'ARM3_MOVE__PM2__TO__LLD', 'ARM3_LOAD__PM2__TO__LLD', 'PROC__LLD'],
                ['ARM3_PICK__LLC__TO__PM3', 'ARM3_MOVE__LLC__TO__PM3', 'ARM3_LOAD__LLC__TO__PM3', 'PROC__PM3',
                 'ARM3_PICK__PM3__TO__LLD', 'ARM3_MOVE__PM3__TO__LLD', 'ARM3_LOAD__PM3__TO__LLD', 'PROC__LLD'],
                ['ARM3_PICK__LLC__TO__PM4', 'ARM3_MOVE__LLC__TO__PM4', 'ARM3_LOAD__LLC__TO__PM4', 'PROC__PM4',
                 'ARM3_PICK__PM4__TO__LLD', 'ARM3_MOVE__PM4__TO__LLD', 'ARM3_LOAD__PM4__TO__LLD', 'PROC__LLD']
            ],
            
            # Stage 4: LLD → PM9/PM10 → LLB_S1 (parallel choice)
            [
                ['ARM2_PICK__LLD__TO__PM10', 'ARM2_MOVE__LLD__TO__PM10', 'ARM2_LOAD__LLD__TO__PM10', 'PROC__PM10',
                 'ARM2_PICK__PM10__TO__LLB_S1', 'ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1', 'PROC__LLB_S1'],
                ['ARM2_PICK__LLD__TO__PM9', 'ARM2_MOVE__LLD__TO__PM9', 'ARM2_LOAD__LLD__TO__PM9', 'PROC__PM9',
                 'ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1', 'PROC__LLB_S1']
            ],
            
            # Stage 5: LLB_S1 → LP_done
            [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done', 'PROC__LP_done']]
        ]
    
    def _define_path_d(self) -> List[List[List[str]]]:
        """
        Define Route D: Simplified route from LP2 (skips LLC and PM1-4).
        
        Structure: Same as pathC
        """
        return [
            # Stage 0: LP2 → AL
            [['ARM1_PICK__LP2__TO__AL', 'ARM1_MOVE__LP2__TO__AL', 'ARM1_LOAD__LP2__TO__AL', 'PROC__AL']],
            
            # Stage 1: AL → LLA_S2
            [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2', 'PROC__LLA_S2']],
            
            # Stage 2: LLA_S2 → PM7/PM8 → PM9/PM10 → LLB_S1 (parallel choice, combined stages)
            [
                ['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7', 'PROC__PM7',
                 'ARM2_PICK__PM7__TO__PM10', 'ARM2_MOVE__PM7__TO__PM10', 'ARM2_LOAD__PM7__TO__PM10', 'PROC__PM10',
                 'ARM2_PICK__PM10__TO__LLB_S1', 'ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1', 'PROC__LLB_S1'],
                ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8', 'PROC__PM8',
                 'ARM2_PICK__PM8__TO__PM9', 'ARM2_MOVE__PM8__TO__PM9', 'ARM2_LOAD__PM8__TO__PM9', 'PROC__PM9',
                 'ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1', 'PROC__LLB_S1']
            ],
            
            # Stage 3: LLB_S1 → LP_done
            [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done', 'PROC__LP_done']]
        ]
    
    def get_path_indices(self, id2t_name: List[str], route_tag: str) -> List[List[List[int]]]:
        """
        Convert path transition names to indices.
        
        Args:
            id2t_name: List mapping transition ID to transition name
            route_tag: 'C' for pathC or 'D' for pathD
        
        Returns:
            Path structure with transition indices instead of names
        """
        path = self._pathC if route_tag == 'C' else self._pathD
        
        path_idx = []
        for stage in path:
            stage_idx = []
            for branch in stage:
                branch_idx = [id2t_name.index(name) for name in branch]
                stage_idx.append(branch_idx)
            path_idx.append(stage_idx)
        
        return path_idx
    
    def get_all_paths(self) -> Dict[str, List[List[List[str]]]]:
        """
        Get all path definitions.
        
        Returns:
            Dictionary with 'C' and 'D' keys mapping to their respective paths
        """
        return {
            'C': self._pathC,
            'D': self._pathD
        }
