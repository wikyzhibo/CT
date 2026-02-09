
import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from solutions.PPO.network.models import MaskedPolicyHead

def load_policy(model_path, env, device="cpu"):
    """
    加载 PPO 策略模型。
    """
    state_dict = torch.load(model_path, map_location=device)

    # 尝试从环境规范中获取维度，或者使用默认值
    try:
        n_actions = env.action_spec.space.n
        n_m = env.observation_spec["observation"].shape[0]
    except (AttributeError, KeyError):
        # 如果环境规范不可用，可以尝试从环境本身获取
        n_actions = getattr(env, 'A', 0)
        n_m = getattr(env, 'obs_dim', 0)

    policy_backbone = MaskedPolicyHead(hidden=128, n_obs=n_m, n_actions=n_actions, n_layers=4)
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
