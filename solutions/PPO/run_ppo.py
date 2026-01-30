import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from solutions.PPO.enviroment import Env_PN
from solutions.PPO.train import train
from torchrl.envs import (Compose, DTypeCastTransform, TransformedEnv, ActionMask)
import time
from data.ppo_configs.training_config import PPOTrainingConfig
import warnings
import os
import sys

warnings.filterwarnings(
    "ignore",
    message="size_average and reduce args will be deprecated*",
    category=UserWarning,
)


def create_env(device, training_phase: int = 1):
    """
    创建训练和评估环境

    Args:
        device: 计算设备
        training_phase: 训练阶段（1 或 2）

    Returns:
        (train_env, eval_env): 训练和评估环境
    """
    base_env1 = Env_PN(device=device, training_phase=training_phase)
    base_env2 = Env_PN(device=device, training_phase=training_phase)

    transform = Compose([
        ActionMask(),
        DTypeCastTransform(dtype_in=torch.int64, dtype_out=torch.float32,
                           in_keys="observation", out_keys="observation_f"),
    ])

    train_env = TransformedEnv(base_env1, transform)
    eval_env = TransformedEnv(base_env2, transform)

    return train_env, eval_env


def get_config_path(phase: int, custom_config: str = None):
    """
    获取配置文件路径

    Args:
        phase: 训练阶段（1 或 2）
        custom_config: 自定义配置文件路径

    Returns:
        配置文件的绝对路径
    """
    if custom_config:
        if os.path.isabs(custom_config):
            return custom_config
        else:
            # 相对于当前工作目录
            return os.path.abspath(custom_config)
    
    # 默认配置：相对于项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    config_path = os.path.join(project_root, "data", "ppo_configs", f"phase{phase}_config.json")
    return os.path.abspath(config_path)


def train_single_phase(phase: int, device, config_path: str = None, checkpoint_path: str = None):
    """
    训练单个阶段

    Args:
        phase: 训练阶段（1 或 2）
        device: 计算设备
        config_path: 配置文件路径（可选）
        checkpoint_path: checkpoint文件路径（可选）

    Returns:
        (log, policy): 训练日志和策略网络
    """
    print("\n" + "=" * 60)
    print(f"[Phase {phase}] 开始训练")
    if phase == 1:
        print("  训练目标: 仅考虑报废惩罚（加工腔室超时）")
    else:
        print("  训练目标: 完整奖励（加工腔室超时 + 运输位超时）")
    print("=" * 60)

    # 创建环境
    train_env, eval_env = create_env(device, training_phase=phase)

    # 加载配置
    config_file = get_config_path(phase, config_path)
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)
    
    print(f"加载配置: {config_file}")
    config = PPOTrainingConfig.load(config_file)
    config.device = str(device)  # 更新设备

    # 加载checkpoint（如果提供）
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"警告: checkpoint文件不存在: {checkpoint_path}")
            checkpoint_path = None
        else:
            print(f"加载checkpoint: {checkpoint_path}")

    # 开始训练
    start_time = time.time()
    log, policy = train(train_env, eval_env, config=config, checkpoint_path=checkpoint_path)
    elapsed_time = time.time() - start_time
    
    print(f"\nPhase {phase} 训练完成! 用时: {elapsed_time:.2f}s")
    
    return log, policy


def train_multi_phase(phases: list, device, config_path: str = None, 
                     checkpoint_path: str = None, auto_load: bool = True):
    """
    多阶段课程学习训练

    Args:
        phases: 训练阶段列表，例如 [1, 2]
        device: 计算设备
        config_path: 配置文件路径（可选，仅用于第一阶段）
        checkpoint_path: 初始checkpoint路径（可选，仅用于第一阶段）
        auto_load: 是否自动加载前一阶段的最佳模型

    Returns:
        训练结果字典
    """
    print("\n" + "=" * 60)
    print(f"多阶段课程学习训练: {' → '.join([f'Phase {p}' for p in phases])}")
    print("=" * 60)

    results = {}
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    
    for i, phase in enumerate(phases):
        # 第一阶段使用提供的checkpoint，后续阶段使用前一阶段的最佳模型
        if i == 0:
            current_checkpoint = checkpoint_path
        else:
            if auto_load:
                prev_phase = phases[i - 1]
                current_checkpoint = os.path.join(saved_models_dir, f"CT_phase{prev_phase}_best.pt")
                if not os.path.exists(current_checkpoint):
                    print(f"警告: 未找到 Phase {prev_phase} 的最佳模型，将从头训练")
                    current_checkpoint = None
            else:
                current_checkpoint = None

        # 训练当前阶段
        log, policy = train_single_phase(
            phase=phase,
            device=device,
            config_path=config_path if i == 0 else None,  # 仅第一阶段使用自定义配置
            checkpoint_path=current_checkpoint
        )
        
        results[f"phase{phase}"] = {"log": log, "policy": policy}

    print("\n" + "=" * 60)
    print("所有阶段训练完成!")
    print("=" * 60)
    
    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PPO训练工具 - 支持单阶段/多阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 只训练 Phase 1
  python run_ppo.py --phase 1

  # 只训练 Phase 2，从 Phase 1 checkpoint 开始
  python run_ppo.py --phase 2 --checkpoint saved_models/30_1241.pt

  # 两阶段课程学习（默认，自动加载）
  python run_ppo.py --phase 1,2

  # 使用自定义配置
  python run_ppo.py --phase 1 --config data/ppo_configs/custom/my_config.json

  # 从中断处继续训练
  python run_ppo.py --phase 1 --checkpoint solutions/PPO/saved_models/30_1241.pt

  # Phase 2 从头训练（不加载 Phase 1）
  python run_ppo.py --phase 2 --no-auto-load
        """
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        default="1,2",
        help="训练阶段: 1（仅报废惩罚）, 2（完整奖励）, 或 1,2（两阶段课程学习）。默认: 1,2"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义配置文件路径（可选）。如果不指定，使用 data/ppo_configs/phase{N}_config.json"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint文件路径（可选），用于继续训练或迁移学习"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="计算设备: cpu 或 cuda。默认: 自动检测"
    )
    
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="多阶段训练时，禁用自动加载前一阶段的最佳模型"
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
    print("PPO训练工具")
    print(f"计算设备: {device}")
    print("=" * 60)
    
    # 解析训练阶段
    phases = [int(p.strip()) for p in args.phase.split(",")]
    
    # 验证阶段参数
    for phase in phases:
        if phase not in [1, 2]:
            print(f"错误: 无效的训练阶段: {phase}。仅支持 1 或 2")
            sys.exit(1)
    
    # 执行训练
    if len(phases) == 1:
        # 单阶段训练
        train_single_phase(
            phase=phases[0],
            device=device,
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
    else:
        # 多阶段训练
        train_multi_phase(
            phases=phases,
            device=device,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            auto_load=not args.no_auto_load
        )


if __name__ == "__main__":
    main()
