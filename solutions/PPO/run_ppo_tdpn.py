import argparse
import torch
from solutions.Td_petri.tdpn import TimedPetri
from solutions.PPO.train import train
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from torchrl.envs.utils import check_env_specs
import time
from data.ppo_configs.training_config import PPOTrainingConfig
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="size_average and reduce args will be deprecated*",
    category=UserWarning,
)


def create_env(device, seed=None):
    """
    创建训练和评估环境（Td_petri 版本）

    Args:
        device: 计算设备
        seed: 随机种子（可选）

    Returns:
        (train_env, eval_env): 训练和评估环境
    """
    base_env1 = TimedPetri(device=device, seed=seed, reward_mode='time')
    base_env2 = TimedPetri(device=device, seed=seed, reward_mode='time')

    transform = Compose([
        ActionMask(),
        DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.float32,
            in_keys="observation",
            out_keys="observation_f"
        ),
    ])


    train_env = TransformedEnv(base_env1, transform)
    eval_env = TransformedEnv(base_env2, transform)

    return train_env,eval_env


def get_config_path(custom_config: str = None):
    """
    获取配置文件路径

    Args:
        custom_config: 自定义配置文件路径

    Returns:
        配置文件的绝对路径
    """
    if custom_config:
        if os.path.isabs(custom_config):
            return custom_config
        else:
            # 相对于当前工作目录
            ROOT_DIR = Path(__file__).resolve().parents[2]
            path = os.path.join(ROOT_DIR, "data", "ppo_configs", custom_config)
            return os.path.abspath(path)
    
    # 默认配置：使用 tdpn_config.json（如果不存在则使用 phase2_config.json）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    
    # 优先尝试 tdpn_config.json
    tdpn_config = os.path.join(project_root, "data", "ppo_configs", "tdpn_config.json")
    if os.path.exists(tdpn_config):
        return os.path.abspath(tdpn_config)
    
    # 回退到 phase2_config.json
    phase2_config = os.path.join(project_root, "data", "ppo_configs", "phase2_config.json")
    return os.path.abspath(phase2_config)


def train_tdpn(device, config_path: str = None, checkpoint_path: str = None, seed: int = None):
    """
    训练 Td_petri 环境

    Args:
        device: 计算设备
        config_path: 配置文件路径（可选）
        checkpoint_path: checkpoint文件路径（可选）
        seed: 随机种子（可选）

    Returns:
        (log, policy): 训练日志和策略网络
    """
    print("\n" + "=" * 60)
    print("[Td_petri] 开始训练")
    print("  环境: TimedPetri (Timed Discrete Petri Net)")
    print("  动作空间: Chain-based")
    print("=" * 60)

    # 创建环境
    train_env, eval_env = create_env(device, seed=seed)
    
    # 打印环境信息
    obs_dim = train_env.observation_spec["observation"].shape[0]
    act_dim = train_env.action_spec.space.n
    print(f"观测维度: {obs_dim}")
    print(f"动作空间大小: {act_dim}")

    # 加载配置
    config_file = get_config_path(config_path)
    if not os.path.exists(config_file):
        print(f"警告: 配置文件不存在: {config_file}，使用默认配置")
        config = PPOTrainingConfig()
    else:
        print(f"加载配置: {config_file}")
        config = PPOTrainingConfig.load(config_file)
    
    config.device = str(device)  # 更新设备
    if seed is not None:
        config.seed = seed  # 更新种子

    # 加载checkpoint（如果提供）
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint文件不存在: {checkpoint_path}")
            checkpoint_path = None
        else:
            print(f"加载checkpoint: {checkpoint_path}")

    # 开始训练
    start_time = time.time()
    log, policy = train(
        train_env, 
        eval_env, 
        config=config, 
        checkpoint_path=checkpoint_path
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n训练完成! 用时: {elapsed_time:.2f}s")
    
    return log, policy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PPO训练工具 - Td_petri (Timed Discrete Petri Net)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置训练
  python run_ppo_tdpn.py

  # 使用自定义配置
  python run_ppo_tdpn.py --config data/ppo_configs/tdpn_config.json

  # 从checkpoint继续训练
  python run_ppo_tdpn.py --checkpoint saved_models/CT_tdpn_best.pt

  # 指定随机种子
  python run_ppo_tdpn.py --seed 42

  # 使用GPU训练
  python run_ppo_tdpn.py --device cuda
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义配置文件路径（可选）。如果不指定，优先使用 data/ppo_configs/tdpn_config.json，否则使用 phase2_config.json"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint文件路径（可选），用于继续训练"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="计算设备: cpu 或 cuda。默认: 自动检测"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 确定计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("PPO训练工具 - Td_petri")
    print(f"计算设备: {device}")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 执行训练
    train_tdpn(
        device=device,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
