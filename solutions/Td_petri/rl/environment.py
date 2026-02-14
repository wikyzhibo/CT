"""
RL Environment for Timed Petri Net.

This module now re-exports TimedPetri directly as the environment.
The wrapper layer has been removed - TimedPetri class now inherits 
from TorchRL's EnvBase directly.

For backward compatibility, TimedPetriEnv and CT_v2_Refactored are 
aliased to TimedPetri.
"""

from solutions.Td_petri.tdpn import TimedPetri

# Backward compatibility aliases
TimedPetriEnv = TimedPetri
TimedPetriEnvSimple = TimedPetri
CT_v2_Refactored = TimedPetri

__all__ = [
    'TimedPetri',
    'TimedPetriEnv',
    'TimedPetriEnvSimple', 
    'CT_v2_Refactored',
]
