"""
Reward calculation for RL environment.

This module provides reward functions for evaluating Petri net states.
"""

from typing import List
from solutions.model.pn_models import Place, WaferToken


class RewardCalculator:
    """
    Calculates rewards based on Petri net state.
    
    Reward is based on wafer progress through the manufacturing process,
    normalized by time to encourage efficiency.
    """
    
    def __init__(self, obs_place_idx: List[int], idle_idx_start: List[int],
                 weights_map: dict = None):
        """
        Initialize the reward calculator.
        
        Args:
            obs_place_idx: Indices of observable places
            idle_idx_start: Indices of starting places (LP1, LP2)
            weights_map: Map of wafer type to its stage weights.
                        Example: {1: [0, 10...], 2: [0, 10...]}
        """
        self.obs_place_idx = obs_place_idx
        self.idle_idx_start = idle_idx_start
        
        if weights_map is None:
            # Default stage weights
            self.weights_map = {
                1: [0, 10, 30, 100, 770, 970, 1000],
                2: [0, 10, 30, 300, 330]
            }
        else:
            self.weights_map = weights_map
    
    def calculate_reward(self, marks: List[Place], current_time: int) -> float:
        """
        Calculate reward based on wafer progress.
        
        Args:
            marks: Current marking
            current_time: Current system time
        
        Returns:
            Reward value (progress / time)
        """
        work_finish = 0
        
        for p in self.obs_place_idx:
            if p not in self.idle_idx_start:
                place = marks[p]
                for tok in place.tokens:
                    if isinstance(tok, WaferToken):
                        ww = tok.where  # Current stage index
                        w_type = tok.type
                        
                        # Get weights for this specific type
                        weights = self.weights_map.get(w_type, [])
                        
                        # Reward based on progress for this wafer type
                        if ww < len(weights):
                            work_finish += weights[ww]
        
        # Normalize by time to encourage efficiency
        if current_time > 0:
            return work_finish / current_time
        else:
            return 0.0
