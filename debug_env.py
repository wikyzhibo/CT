
import torch
import os
import sys
sys.path.append(os.getcwd())
from solutions.PPO.enviroment import CT_v2

def main():
    print("Initializing CT_v2...")
    env = CT_v2()
    print(f"Env initialized. n_actions: {env.n_actions}")
    print(f"Action spec n: {env.action_spec.space.n}")
    
    # Check id2t_name
    print(f"Transitions: {env.net.id2t_name}")
    
    td = env.reset()
    print("Reset done.")
    
    action = torch.tensor([7])
    print(f"Stepping with action {action}...")
    
    td["action"] = action
    try:
        td = env.step(td)
        print("Step success.")
    except Exception as e:
        print(f"Step failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
