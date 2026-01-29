import os
import torch
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from torchrl.envs.utils import set_exploration_type, ExplorationType
from solutions.PPO.enviroment import CT2, CT_v2, Env_PN
from solutions.PPO.network.models import MaskedPolicyHead
from visualization.plot import plot_gantt_hatched_residence,Op

def load_policy(model_path, env, device="cpu"):
    state_dict = torch.load(model_path)

    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]

    policy_backbone = MaskedPolicyHead(hidden=256, n_obs=n_m, n_actions=n_actions,n_layers=4)
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
    with torch.no_grad():
        with set_exploration_type(ExplorationType.RANDOM):
            out = env.rollout(max_steps, policy)
    
    # 检查是否完成任务
    if out["next", "finish"].sum().item() <= 0:
        print("警告: 任务未完成")
        env.reset()
        return None
    
    # 计算 makespan
    makespan = torch.where(
        out["next", "finish"],
        out["next", "time"],
        100000
    ).min().item()

    print(f"评估结果 - Makespan: {makespan:.2f}", end="")
    print(f" | 步数: {len(out)}")
    
    return makespan


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="验证训练好的PPO模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 验证 Env_PN 模型（默认）
  python validation.py --env_type Env_PN --model_path saved_models/CT_phase2_latest.pt

  # 验证 CT_v2 (Td_petri) 模型
  python validation.py --env_type CT_v2 --model_path saved_models/CT_tdpn_latest.pt

  # 使用 GPU 验证
  python validation.py --env_type CT_v2 --device cuda
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型文件路径（如果不指定，根据 env_type 使用默认路径）"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="Env_PN",
        choices=["Env_PN", "CT_v2"],
        help="环境类型: Env_PN（连续模型）或 CT_v2（Td_petri）。默认: Env_PN"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="N7",
        choices=["N2", "N5", "N6", "N7", "N8"],
        help="使用的配置文件（仅 Env_PN 需要，默认: N7）"
    )
    parser.add_argument(
        "--n_wafer",
        type=int,
        default=75,
        help="晶圆数量（仅 Env_PN 需要，默认: 75）"
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
        default=8,
        help="运行次数（默认: 8）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 设置默认模型路径
    if args.model_path is None:
        if args.env_type == "CT_v2":
            args.model_path = "saved_models/CT_tdpn_latest.pt"
        else:
            args.model_path = "saved_models/CT_phase2_latest.pt"
    
    # 创建环境
    print(f"创建环境 - 类型: {args.env_type}")
    
    if args.env_type == "CT_v2":
        # Td_petri 环境
        base_env = CT_v2(device=device, seed=args.seed)
        print(f"  动作空间大小: {base_env.action_spec.space.n}")
        print(f"  观测维度: {base_env.observation_spec['observation'].shape[0]}")
    else:
        # 连续模型环境
        print(f"  配置: {args.config}, 晶圆数: {args.n_wafer}")
        base_env = Env_PN(device=device)
    
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

    min_makespan = 10**5
    ops2 = []
    for run in range(args.n_runs):
        print(f"\n--- 运行 {run + 1}/{args.n_runs} ---")
        env.reset()
        makespan = ev(env, policy, max_steps=args.max_steps)

        if makespan is not None:
            if makespan < min_makespan:
                min_makespan = makespan


        if makespan is not None:
            makespans.append(makespan)

    # 绘制甘特图（如果环境支持）
    if hasattr(env, 'net') and hasattr(env.net, 'render_gantt'):
        net = env.net
        # 使用相对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        out_path = "./results/"
        net.render_gantt(out_path=out_path)



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