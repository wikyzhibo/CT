
import torch
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from visualization.plot import plot_gantt_hatched_residence,Op
from solutions.PPO.enviroment import CT_v2
from torchrl.envs import TransformedEnv, ActionMask
from solutions.PPO.network.models import MaskedPolicyHead
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tdpn_parser import TDPNParser
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from typing import List, Optional, Tuple

def res_occ_to_event(res_occ: dict):
    events = []
    for itv in res_occ['ARM2']:
        arm = 2
        time = itv.start
        kind = itv.kind #0 PICK ,1 LOAD
        from_loc = getattr(itv, 'from_loc', '')
        to_loc = getattr(itv, 'to_loc', '')
        wafer_type = getattr(itv, 'wafer_type', 0)
        events.append((time, arm, kind, from_loc, to_loc, wafer_type))
        
    for itv in res_occ['ARM3']:
        arm = 3
        time = itv.start
        kind = itv.kind
        from_loc = getattr(itv, 'from_loc', '')
        to_loc = getattr(itv, 'to_loc', '')
        wafer_type = getattr(itv, 'wafer_type', 0)
        events.append((time, arm, kind, from_loc, to_loc, wafer_type))

    return events


def load_policy(model_path, env, device="cpu"):
    state_dict = torch.load(model_path)

    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]

    policy_backbone = MaskedPolicyHead(hidden=128, n_obs=n_m, n_actions=n_actions,n_layers=4)
    td_module = TensorDictModule(
        policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)

    policy.load_state_dict(state_dict)
    policy.eval()
    return policy




def main():
    device = "cpu"
    # Path to the best model in the planB_route_c directory


    ROOT = Path(__file__).resolve().parents[1]

    model_path = os.path.join(ROOT, "PPO", "syc_model", "taskD", "CT_phase2_best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    base_env = CT_v2()
    transform = Compose([
        ActionMask(),
        DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.float32,
            in_keys="observation",
            out_keys="observation_f"
        ),
    ])
    env = TransformedEnv(base_env, transform)
    
    policy = load_policy(model_path, env, device)


    policy.eval()
    max_steps = 2000
    with torch.no_grad():
        with set_exploration_type(ExplorationType.MODE):
            _ = env.rollout(max_steps, policy)

    events = res_occ_to_event(env.net.res_occ)

    parser = TDPNParser()
    sequence = parser.parse(events)
    
    # Save to the same directory as this script
    output_dir = Path(__file__).parent
    output_file = output_dir / "planB_sequence.json"

    
    with open(output_file, "w") as f:
        json.dump(sequence, f, indent=2)
        
    print(f"Generated {output_file} with {len(sequence)} steps.")

    # save gantt chart
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    out_file = str(results_dir / "planB_")
    env.net.render_gantt(out_path=out_file,policy=3)

if __name__ == "__main__":
    main()
