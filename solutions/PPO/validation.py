import os
import torch
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask

from solutions.PPO.enviroment import CT2
from solutions.PPO.network.models import MaskedPolicyHead


def load_policy(model_path, env, device="cpu", n_hidden=128):
    """
    加载训练好的策略模型
    
    Args:
        model_path: 模型文件路径
        env: 环境对象（用于获取动作和观察空间维度）
        device: 设备（cpu/cuda）
        n_hidden: 隐藏层大小（需与训练时一致）
    
    Returns:
        policy: 加载的策略模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 尝试直接加载整个模型对象
    try:
        policy = torch.load(model_path, map_location=device)
        if isinstance(policy, ProbabilisticActor):
            policy = policy.to(device)
            policy.eval()
            print(f"成功加载模型（完整对象）: {model_path}")
            return policy
    except Exception as e:
        print(f"尝试直接加载失败: {e}，尝试加载 state_dict...")
    
    # 如果直接加载失败，尝试加载 state_dict
    # 获取环境参数
    n_actions = env.action_spec.space.n
    in_dim = env.net.low_dim
    
    # 构建策略网络结构（需与训练时一致）
    policy_backbone = MaskedPolicyHead(hidden=n_hidden, n_obs=in_dim, n_actions=n_actions)
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
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict):
        policy.load_state_dict(state_dict)
        print(f"成功加载模型（state_dict）: {model_path}")
    else:
        raise ValueError(f"无法识别的模型格式: {type(state_dict)}")
    
    return policy


def ev(env, policy=None, max_steps=1500):
    """
    评估策略在环境中的表现
    
    Args:
        env: 环境对象
        policy: 策略模型
        max_steps: 最大步数
    
    Returns:
        makespan: 完成时间（如果任务完成），否则返回 None
    """
    if policy is None:
        raise ValueError("policy 不能为 None")
    
    policy.eval()
    env.reset()
    
    with torch.no_grad():
        out = env.rollout(max_steps, policy)
    
    # 检查是否完成任务
    if out["next", "finish"].sum().item() <= 0:
        print("警告: 任务未完成")
        env.reset()
        return None
    
    # 计算 makespan
    time = out["time"].squeeze().tolist()
    time.append(out["next", "time"][-1].item())
    
    makespan = time[-1]
    n_wafer = out["next", "reward"].sum().item() / 200 if "reward" in out["next"] else None
    
    env.reset()
    
    print(f"评估结果 - Makespan: {makespan:.2f}", end="")
    if n_wafer is not None:
        print(f" | 完成晶圆数: {n_wafer:.0f}", end="")
    print(f" | 步数: {len(out)}")
    
    return makespan


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证训练好的PPO模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/CT_latest.pt",
        help="模型文件路径（默认: saved_models/CT_latest.pt）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="N8",
        choices=["N2", "N5", "N6", "N7", "N8"],
        help="使用的配置文件（默认: N8）"
    )
    parser.add_argument(
        "--n_wafer",
        type=int,
        default=75,
        help="晶圆数量（默认: 75）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备（默认: cpu）"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1500,
        help="最大评估步数（默认: 1500）"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="运行次数（默认: 1）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 导入配置
    config_module = f"config.params_{args.config}"
    if args.config == "N8":
        from config.params_N8 import params_N8 as params
    elif args.config == "N7":
        from config.params_N7 import params_N7 as params
    elif args.config == "N6":
        from config.params_N6 import params_N6 as params
    elif args.config == "N5":
        from config.params_N5 import params_N5 as params
    elif args.config == "N2":
        from config.params_N2 import params_N2 as params
    else:
        raise ValueError(f"不支持的配置: {args.config}")
    
    # 设置晶圆数量
    params['n_wafer'] = args.n_wafer
    
    # 创建环境
    print(f"创建环境 - 配置: {args.config}, 晶圆数: {args.n_wafer}")
    base_env = CT2(device=device, allow_idle=False, **params)
    
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
    
    # 加载模型
    model_path = args.model_path
    if not os.path.isabs(model_path):
        # 相对路径，从当前文件所在目录计算
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
    
    print(f"加载模型: {model_path}")
    policy = load_policy(model_path, env, device=device)
    
    # 运行评估
    print(f"\n开始评估 (运行 {args.n_runs} 次)...")
    makespans = []
    
    for run in range(args.n_runs):
        print(f"\n--- 运行 {run + 1}/{args.n_runs} ---")
        makespan = ev(env, policy, max_steps=args.max_steps)
        if makespan is not None:
            makespans.append(makespan)
    
    # 统计结果
    if makespans:
        print(f"\n{'='*50}")
        print(f"评估完成!")
        print(f"成功运行次数: {len(makespans)}/{args.n_runs}")
        print(f"平均 Makespan: {sum(makespans)/len(makespans):.2f}")
        if len(makespans) > 1:
            print(f"最佳 Makespan: {min(makespans):.2f}")
            print(f"最差 Makespan: {max(makespans):.2f}")
        print(f"{'='*50}")
    else:
        print("\n警告: 所有运行都未完成任务")