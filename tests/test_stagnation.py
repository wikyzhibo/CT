import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def test_stagnation_penalty():
    print("Initializing Petri environment for stagnation test...")
    config = PetriEnvConfig()
    config.idle_timeout = 100
    config.idle_penalty = 50
    config.Wait_time = 10
    
    petri = Petri(config)
    petri.reset()
    
    # Run wait steps
    # 100 / 10 = 10 steps to reach timeout
    # Step 1: time 0->10, accumulated wait 10
    # ...
    # Step 10: time 90->100, accumulated wait 100 -> trigger
    
    print("Running wait steps...")
    for i in range(9):
        _, reward, _ = petri.step(wait=True)
        # Should be just time cost
        # c_time * Wait_time = 1 * 10 = 10
        # reward = -10
        print(f"Step {i+1}: Reward {reward} (expected around -10)")
        
    # 10th step (total time 100) -> should trigger penalty
    _, reward, _ = petri.step(wait=True)
    print(f"Step 10: Reward {reward} (expected around -60 = -10 - 50)")
    
    if petri._idle_penalty_applied:
        print("PASS: Idle penalty was applied.")
    else:
        print("FAIL: Idle penalty was NOT applied.")
        exit(1)

    expected_reward = -config.c_time * config.Wait_time - config.idle_penalty
    if abs(reward - expected_reward) < 1e-5:
         print(f"PASS: Reward value is correct ({reward}).")
    else:
         print(f"FAIL: Reward value incorrect. Expected {expected_reward}, Got {reward}")
         exit(1)

if __name__ == "__main__":
    test_stagnation_penalty()
