import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from solutions.Continuous_model.pn import Petri
import numpy as np

def test_concurrent_step():
    print("Initializing Petri net...")
    p = Petri()
    p.reset()
    
    print(f"Initial Time: {p.time}")
    
    # Try to find a state where both robots can move
    # We might need to step a bit to get to a state where both machines have something to do
    # Initial state: tokens in LP1/LP2 and Robot tokens in r_TM2/r_TM3.
    # TM2 can do u_LP1_s1, u_LP2_s1
    # TM3 can do nothing initially until something reaches s2/s3.
    
    # Let's force a state or run a few steps until we find concurrent actions
    found = False
    
    for i in range(100):
        tm2_enabled, tm3_enabled = p.get_enable_t_by_robot()
        
        # If both have enabled transitions
        if tm2_enabled and tm3_enabled:
            # Check for non-conflicting pair
            for a1 in tm2_enabled:
                for a2 in tm3_enabled:
                    if p._check_robot_conflict(a1, a2):
                        print(f"Found concurrent actions at step {i}:")
                        print(f"  TM2 action: {a1} ({p.id2t_name[a1]})")
                        print(f"  TM3 action: {a2} ({p.id2t_name[a2]})")
                        
                        t_before = p.time
                        print(f"  Time before: {t_before}")
                        
                        # Execute concurrent step
                        p.step_concurrent(a1, a2)
                        
                        t_after = p.time
                        print(f"  Time after: {t_after}")
                        print(f"  Time delta: {t_after - t_before}")
                        
                        if t_after - t_before > 5:
                            print("FAIL: Time incremented by more than 5s!")
                        else:
                            print("PASS: Time incremented by 5s or less.")
                        return

        # If not found, just step random one to advance state
        enabled = p.get_enable_t()
        if not enabled:
            p.step(wait=True)
        else:
            t = enabled[0]
            p.step(t=t)
            
    print("Could not find concurrent actions in 100 steps.")

if __name__ == "__main__":
    test_concurrent_step()
