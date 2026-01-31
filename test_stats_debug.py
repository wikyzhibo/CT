
import sys
import os

# fix path to allow imports
sys.path.append(os.getcwd())

from solutions.PPO.enviroment import Env_PN
from visualization.petri_adapter import PetriAdapter

def test_stats():
    env = Env_PN()
    print("Transitions:", env.net.id2t_name)
    
    # Ensure stats enabled
    env.net.enable_statistics = True
    
    # Run some steps
    print("\nRunning steps...")
    for i in range(50):
        # random action
        enabled = env.net.get_enable_t()
        if not enabled:
            env.net.step(wait=True)
            print(f"Step {i}: WAIT (Time {env.net.time})")
        else:
            action = enabled[0] 
            t_name = env.net.id2t_name[action]
            env.net.step(t=action)
            print(f"Step {i}: {t_name} (Time {env.net.time})")
            
    # Check stats
    stats = env.net.calc_wafer_statistics()
    print("\nCalculated Stats:")
    import json
    print(json.dumps(stats, indent=2))
    
    if stats.get('system_avg', 0) == 0 and stats.get('in_progress_count', 0) > 0:
        print("\n[WARNING] system_avg is 0 but wafers are in progress??")
        # Check raw storage
        print("Raw wafer_stats keys:", list(env.net.wafer_stats.keys()))
        if env.net.wafer_stats:
            first_key = list(env.net.wafer_stats.keys())[0]
            print(f"Sample wafer {first_key}:", env.net.wafer_stats[first_key])
            
test_stats()
