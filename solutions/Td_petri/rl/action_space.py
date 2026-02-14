"""
Action space builder for RL environment.

This module constructs the action space from path definitions,
mapping chains to action IDs and tracking parallel stages.
这个模块根据路径构建“链条->动作id”的映射
"""

from typing import List, Dict, Tuple, Set
import numpy as np
from .path_registry import PathRegistry


class ActionSpaceBuilder:
    """
    根据路径构建“链条->动作id”的映射，
    每个动作对应一个必须自动执行的“变迁链条”。
    """
    
    def __init__(self, path_registry: PathRegistry):
        """
        Initialize the action space builder.
        
        Args:
            path_registry: PathRegistry instance containing route definitions
        """
        self.path_registry = path_registry
        self.aid2chain: List[Tuple[str, ...]] = []  # Action ID → chain tuple
        self.chain2aid: Dict[Tuple[str, ...], int] = {}  # Chain tuple → action ID
        self.aid_is_parallel: np.ndarray = None  # Boolean array: is action from parallel stage?
        self.aid_pstage: np.ndarray = None  # Parallel stage ID for each action (-1 if not parallel)
        self.aid2tags: List[Set[str]] = []  # Which routes use this action ('C', 'D', or both)
        self.A: int = 0  # Total number of actions
        
        # Build the action space
        self._build()
    
    def _build(self) -> None:
        """Build the action space from path registry."""
        all_paths = self.path_registry.get_all_paths()
        
        allowed = []  # [(tag, chain_tuple), ...]
        chain_meta = {}  # chain → {"is_parallel": bool, "pstage": int, "tags": set()}
        
        pstage_id = 0  # Parallel stage counter (shared across C/D routes)
        
        def add_path(tag: str, path: List[List[List[str]]]) -> None:
            nonlocal pstage_id
            for stage in path:
                is_parallel = (len(stage) > 1)
                cur_pstage = pstage_id if is_parallel else -1
                if is_parallel:
                    pstage_id += 1
                
                for chain in stage:
                    ch = tuple(chain)
                    allowed.append((tag, ch))
                    
                    if ch not in chain_meta:
                        chain_meta[ch] = {
                            "is_parallel": is_parallel,
                            "pstage": cur_pstage,
                            "tags": set([tag])
                        }
                    else:
                        # Same chain appears in multiple routes: merge tags
                        chain_meta[ch]["tags"].add(tag)
                        # If any route treats it as parallel, mark as parallel
                        if is_parallel and not chain_meta[ch]["is_parallel"]:
                            chain_meta[ch]["is_parallel"] = True
                            chain_meta[ch]["pstage"] = cur_pstage
        
        # Add both routes
        add_path("C", all_paths['C'])
        add_path("D", all_paths['D'])
        
        # Deduplicate chains (same chain used by multiple routes gets single action ID)
        uniq_chains = []
        seen = set()
        for _, ch in allowed:
            if ch in seen:
                continue
            seen.add(ch)
            uniq_chains.append(ch)
        
        self.aid2chain = uniq_chains
        self.chain2aid = {ch: i for i, ch in enumerate(uniq_chains)}
        self.A = len(self.aid2chain)
        
        # Build action metadata
        self.aid_is_parallel = np.zeros(self.A, dtype=bool)
        self.aid_pstage = -np.ones(self.A, dtype=np.int32)
        self.aid2tags = [set() for _ in range(self.A)]
        
        for aid, ch in enumerate(self.aid2chain):
            meta = chain_meta.get(ch)
            if meta is None:
                continue
            self.aid_is_parallel[aid] = bool(meta["is_parallel"])
            self.aid_pstage[aid] = int(meta["pstage"])
            self.aid2tags[aid] = meta["tags"]
    
    def get_action_space_info(self) -> Dict:
        """
        Get complete action space information.
        
        Returns:
            Dictionary containing:
            - aid2chain: List of chain tuples
            - chain2aid: Dict mapping chain to action ID
            - aid_is_parallel: Boolean array
            - aid_pstage: Parallel stage ID array
            - aid2tags: List of tag sets
            - A: Total number of actions
        """
        return {
            'aid2chain': self.aid2chain,
            'chain2aid': self.chain2aid,
            'aid_is_parallel': self.aid_is_parallel,
            'aid_pstage': self.aid_pstage,
            'aid2tags': self.aid2tags,
            'A': self.A
        }
    
    def get_parallel_stage_info(self) -> Dict[int, int]:
        """
        Get information about parallel stages.
        
        Returns:
            Dictionary mapping parallel stage ID to number of chain choices
        """
        stage_chain_count = {}
        for aid in range(self.A):
            if self.aid_is_parallel[aid]:
                pstage = self.aid_pstage[aid]
                if pstage >= 0:
                    if pstage not in stage_chain_count:
                        stage_chain_count[pstage] = set()
                    stage_chain_count[pstage].add(aid)
        
        return {pstage: len(aids) for pstage, aids in stage_chain_count.items()}
