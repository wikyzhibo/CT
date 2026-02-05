"""
PPO训练配置使用示例

展示如何使用PPOTrainingConfig来配置和启动训练
"""

from data.ppo_configs.training_config import PPOTrainingConfig
# from solutions.PPO.train import train
# 注意：实际使用时需要导入环境等其他必要的模块


def example_1_use_default_config():
    """示例1: 使用默认配置"""
    print("=" * 60)
    print("示例1: 使用默认配置")
    print("=" * 60)
    
    config = PPOTrainingConfig()
    print(config)
    
    # 训练（需要先创建环境）
    # log, policy = train(env, eval_env, config=config)


def example_2_use_custom_config():
    """示例2: 创建自定义配置"""
    print("\n" + "=" * 60)
    print("示例2: 创建自定义配置")
    print("=" * 60)
    
    config = PPOTrainingConfig(
        n_hidden=256,           # 增大隐藏层
        n_layer=5,              # 增加层数
        total_batch=200,        # 增加训练批次
        lr=5e-4,                # 调整学习率
        training_phase=1,       # 阶段1训练
        device="cuda",          # 使用GPU
        with_pretrain=True,     # 启用预训练
    )
    print(config)
    
    # 保存配置供后续使用
    config.save("data/ppo_configs/my_custom_config.json")


def example_3_load_from_file():
    """示例3: 从文件加载配置"""
    print("\n" + "=" * 60)
    print("示例3: 从文件加载配置")
    print("=" * 60)
    
    # 加载阶段1配置
    config = PPOTrainingConfig.load("data/ppo_configs/phase1_config.json")
    print(config)
    
    # 可以在加载后修改部分参数
    config.total_batch = 300
    config.lr = 1e-3
    print("\n修改后的配置:")
    print(config)


def example_4_train_with_config_path():
    """示例4: 直接使用配置文件路径训练"""
    print("\n" + "=" * 60)
    print("示例4: 直接使用配置文件路径训练")
    print("=" * 60)
    
    # 这种方式train函数会自动加载配置
    # log, policy = train(
    #     env, 
    #     eval_env, 
    #     config_path="data/ppo_configs/phase2_config.json"
    # )
    print("使用配置文件: data/ppo_configs/phase2_config.json")


def example_5_two_phase_training():
    """示例5: 两阶段训练示例"""
    print("\n" + "=" * 60)
    print("示例5: 两阶段训练")
    print("=" * 60)
    
    # 阶段1：仅报废惩罚
    config_phase1 = PPOTrainingConfig.load("data/ppo_configs/phase1_config.json")
    print("阶段1配置:")
    print(config_phase1)
    
    # log1, policy1 = train(env, eval_env, config=config_phase1)
    # checkpoint1 = "solutions/PPO/saved_models/CT_phase1_latest.pt"
    
    # 阶段2：完整奖励，从阶段1的checkpoint继续
    config_phase2 = PPOTrainingConfig.load("data/ppo_configs/phase2_config.json")
    print("\n阶段2配置:")
    print(config_phase2)
    
    # log2, policy2 = train(
    #     env, 
    #     eval_env, 
    #     config=config_phase2,
    #     checkpoint_path=checkpoint1
    # )


def example_6_hyperparameter_tuning():
    """示例6: 超参数调优"""
    print("\n" + "=" * 60)
    print("示例6: 超参数调优")
    print("=" * 60)
    
    # 定义要尝试的超参数组合
    configs = []
    
    # 不同的学习率
    for lr in [1e-3, 5e-4, 1e-4]:
        config = PPOTrainingConfig(lr=lr)
        configs.append(("lr_" + str(lr), config))
    
    # 不同的网络大小
    for n_hidden in [64, 128, 256]:
        config = PPOTrainingConfig(n_hidden=n_hidden)
        configs.append((f"hidden_{n_hidden}", config))
    
    # 不同的熵系数
    for entropy_start in [0.01, 0.02, 0.05]:
        config = PPOTrainingConfig(entropy_start=entropy_start)
        configs.append((f"entropy_{entropy_start}", config))
    
    print(f"创建了 {len(configs)} 个配置用于超参数调优")
    
    # 为每个配置运行训练
    # for name, config in configs:
    #     print(f"\n训练配置: {name}")
    #     config.save(f"data/ppo_configs/tuning_{name}.json")
    #     log, policy = train(env, eval_env, config=config)


if __name__ == "__main__":
    # 运行所有示例
    example_1_use_default_config()
    example_2_use_custom_config()
    example_3_load_from_file()
    example_4_train_with_config_path()
    example_5_two_phase_training()
    example_6_hyperparameter_tuning()
    
    print("\n" + "=" * 60)
    print("所有示例运行完毕")
    print("=" * 60)
