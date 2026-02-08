"""
Observation builder for RL environment.

This module handles the construction of observations from Petri net state,
including token positions and action history.
"""

from typing import List
import numpy as np
from solutions.model.pn_models import Place, WaferToken


class ObservationBuilder:
    """
    Builds observations for the RL agent from Petri net state.
    
    Observation structure:
    - obs1[16]: Count of type=1 tokens (route C wafers) in each P_READY place
    - obs2[16]: Count of type=2 tokens (route D wafers) in each P_READY place
    - his_a[history_length]: Recent action history
    
    Total dimension: 32 + history_length
    """
    
    def __init__(self, obs_place_idx: List[int], history_length: int = 50):
        """
        Initialize the observation builder.
        
        Args:
            obs_place_idx: Indices of places to observe (P_READY places)
            history_length: Number of recent actions to track
        """
        self.obs_place_idx = obs_place_idx
        self.history_length = history_length
        self.his_a = [0] * history_length  # Action history
        
        # Calculate observation dimension
        self.obs_dim = 2 * len(obs_place_idx) + history_length
    
    def build_observation(self, marks: List[Place]) -> np.ndarray:
        """
        Build observation vector from current Petri net state.
        
        Args:
            marks: Current marking (list of Place objects)
        
        Returns:
            Observation vector of shape (obs_dim,)
        """
        obs1 = np.zeros(len(self.obs_place_idx))  # Type 1 tokens
        obs2 = np.zeros(len(self.obs_place_idx))  # Type 2 tokens
        
        for k, idx in enumerate(self.obs_place_idx):
            p = marks[idx]
            for tok in p.tokens:
                if tok.type == 1:
                    obs1[k] += 1
                elif tok.type == 2:
                    obs2[k] += 1
        
        # Combine with action history
        his_a = np.array(self.his_a, dtype=np.float32)
        obs = np.concatenate([obs1, obs2, his_a])
        return obs
    
    def update_history(self, action: int) -> None:
        """
        Update action history with a new action.
        
        Args:
            action: Action ID that was just taken
        """
        self.his_a.pop(0)  # Remove oldest
        self.his_a.append(int(action))  # Add newest
    
    def reset_history(self) -> None:
        """Reset action history to initial state."""
        self.his_a = [0] * self.history_length
    
    def get_observation_dim(self) -> int:
        """Get the dimension of the observation vector."""
        return self.obs_dim
