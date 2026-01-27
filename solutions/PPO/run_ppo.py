import matplotlib.pyplot as plt
import numpy as np
import torch
from solutions.PPO.enviroment import Env_PN
from train import train
from torchrl.envs import (Compose, DTypeCastTransform, TransformedEnv, ActionMask)
import time
from data.ppo_configs.training_config import PPOTrainingConfig
import warnings

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


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("自动两阶段课程学习训练")
    print("=" * 60)

    # Phase 1: 仅报废惩罚
    print("\n[Phase 1] 仅考虑报废惩罚...")
    # 获取 Phase 1 最佳模型路径
    #saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    #phase1_checkpoint = os.path.join(saved_models_dir, "CT_phase1_best.pt")
    train_env, eval_env = create_env(device, training_phase=1)

    # 加载 Phase 1 配置
    config1 = PPOTrainingConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\ppo_configs\phase1_config.json")

    start_time = time.time()
    log1, policy1 = train(train_env, eval_env, config=config1)
    print(f"Phase 1 training time: {time.time() - start_time:.2f}s")

    # 获取 Phase 1 最佳模型路径
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    phase1_checkpoint = os.path.join(saved_models_dir, "CT_phase1_best.pt")

    # Phase 2: 完整奖励（加载 Phase 1 checkpoint）
    print("\n[Phase 2] 完整奖励（加载 Phase 1 模型）...")
    train_env, eval_env = create_env(device, training_phase=2)

    # 加载 Phase 2 配置
    config2 = PPOTrainingConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\ppo_configs\phase2_config.json")

    start_time = time.time()
    log2, policy2 = train(
        train_env, eval_env,
        config=config2,
        checkpoint_path=phase1_checkpoint
    )
    print(f"Phase 2 training time: {time.time() - start_time:.2f}s")
