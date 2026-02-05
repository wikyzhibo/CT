"""
Petri网环境配置使用示例

展示如何使用PetriEnvConfig来配置和创建Petri网环境
"""

from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri


def example_1_default_config():
    """示例1: 使用默认配置"""
    print("=" * 60)
    print("示例1: 使用默认配置")
    print("=" * 60)
    
    # 方法1: 直接创建（使用内部默认配置）
    net = Petri()
    print(f"Training Phase: {net.training_phase}")
    print(f"N Wafer: {net.n_wafer}")
    print(f"R_done: {net.R_done}")
    print(f"Transport Penalty: {net.reward_config['transport_penalty']}")


def example_2_custom_config():
    """示例2: 创建自定义配置"""
    print("\n" + "=" * 60)
    print("示例2: 创建自定义配置")
    print("=" * 60)
    
    # 创建自定义配置
    config = PetriEnvConfig(
        n_wafer=6,
        R_done=200,
        R_scrap=150,
        c_time=1.0,
        training_phase=1,
        stop_on_scrap=False  # 不在报废时停止
    )
    
    print(config)
    
    # 使用配置创建环境
    net = Petri(config=config)
    print(f"\nN Wafer: {net.n_wafer}")
    print(f"Stop on Scrap: {net.stop_on_scrap}")


def example_3_load_from_file():
    """示例3: 从配置文件加载"""
    print("\n" + "=" * 60)
    print("示例3: 从配置文件加载")
    print("=" * 60)
    
    # 加载阶段1配置
    config = PetriEnvConfig.load("data/petri_configs/phase1_config.json")
    print(config)
    
    # 创建环境
    net = Petri(config=config)
    print(f"\nTraining Phase: {net.training_phase}")
    print(f"Transport Penalty Enabled: {net.reward_config['transport_penalty']}")


def example_4_save_config():
    """示例4: 保存配置到文件"""
    print("\n" + "=" * 60)
    print("示例4: 保存配置到文件")
    print("=" * 60)
    
    # 创建配置
    config = PetriEnvConfig(
        n_wafer=8,
        R_done=150,
        c_time=0.8,
        training_phase=2,
        idle_timeout=150,
        idle_penalty=2000
    )
    
    # 保存配置
    config.save("data/petri_configs/custom_config.json")
    print("配置已保存到: data/petri_configs/custom_config.json")
    
    # 加载验证
    loaded = PetriEnvConfig.load("data/petri_configs/custom_config.json")
    print(f"\n验证 - N Wafer: {loaded.n_wafer}")
    print(f"验证 - Idle Timeout: {loaded.idle_timeout}")


def example_5_phase_comparison():
    """示例5: 对比两阶段配置"""
    print("\n" + "=" * 60)
    print("示例5: 对比两阶段配置")
    print("=" * 60)
    
    # 阶段1
    config1 = PetriEnvConfig.load("data/petri_configs/phase1_config.json")
    net1 = Petri(config=config1)
    
    # 阶段2
    config2 = PetriEnvConfig.load("data/petri_configs/phase2_config.json")
    net2 = Petri(config=config2)
    
    print("阶段1配置:")
    print(f"  Training Phase: {net1.training_phase}")
    print(f"  Transport Penalty: {net1.reward_config['transport_penalty']}")
    
    print("\n阶段2配置:")
    print(f"  Training Phase: {net2.training_phase}")
    print(f"  Transport Penalty: {net2.reward_config['transport_penalty']}")


def example_6_custom_reward_config():
    """示例6: 自定义奖励配置"""
    print("\n" + "=" * 60)
    print("示例6: 自定义奖励配置")
    print("=" * 60)
    
    # 创建自定义奖励配置
    custom_reward = {
        'proc_reward': 1,
        'safe_reward': 0,          # 禁用安全裕量奖励
        'penalty': 1,
        'warn_penalty': 0,         # 禁用预警惩罚
        'transport_penalty': 1,
        'congestion_penalty': 1,   # 启用堵塞预测惩罚
        'time_cost': 1,
        'release_violation_penalty': 1,
    }
    
    config = PetriEnvConfig(
        training_phase=2,
        reward_config=custom_reward,
        c_congest=100  # 增大堵塞惩罚系数
    )
    
    net = Petri(config=config)
    
    print("自定义奖励配置:")
    for key, value in net.reward_config.items():
        status = "启用" if value else "禁用"
        print(f"  {key}: {status}")
    print(f"\n堵塞惩罚系数: {net.c_congest}")


def example_7_backward_compatibility():
    """示例7: 向后兼容性（旧代码仍可运行）"""
    print("\n" + "=" * 60)
    print("示例7: 向后兼容性")
    print("=" * 60)
    
    # 旧方式（仍然有效）
    net = Petri(stop_on_scrap=True, training_phase=2)
    
    print(f"Training Phase: {net.training_phase}")
    print(f"Stop on Scrap: {net.stop_on_scrap}")
    print("旧代码仍然可以正常运行！")


def example_8_modify_existing_config():
    """示例8: 修改现有配置"""
    print("\n" + "=" * 60)
    print("示例8: 修改现有配置")
    print("=" * 60)
    
    # 加载现有配置
    config = PetriEnvConfig.load("data/petri_configs/phase1_config.json")
    
    # 修改参数
    config.n_wafer = 8
    config.R_done = 150
    config.idle_timeout = 200
    
    # 保存为新配置
    config.save("data/petri_configs/phase1_extended.json")
    print("修改后的配置已保存为: phase1_extended.json")
    
    # 创建环境
    net = Petri(config=config)
    print(f"\nN Wafer: {net.n_wafer}")
    print(f"R_done: {net.R_done}")


def example_9_parameter_sweep():
    """示例9: 参数扫描（超参数调优）"""
    print("\n" + "=" * 60)
    print("示例9: 参数扫描")
    print("=" * 60)
    
    # 扫描不同的时间成本参数
    c_time_values = [0.1, 0.5, 1.0, 2.0]
    
    configs = []
    for c_time in c_time_values:
        config = PetriEnvConfig(
            c_time=c_time,
            training_phase=2
        )
        configs.append((f"c_time_{c_time}", config))
        print(f"创建配置: c_time={c_time}")
    
    print(f"\n共创建 {len(configs)} 个配置用于参数扫描")
    
    # 可以为每个配置运行实验
    # for name, config in configs:
    #     net = Petri(config=config)
    #     # 运行实验...


if __name__ == "__main__":
    # 运行所有示例
    example_1_default_config()
    example_2_custom_config()
    example_3_load_from_file()
    example_4_save_config()
    example_5_phase_comparison()
    example_6_custom_reward_config()
    example_7_backward_compatibility()
    example_8_modify_existing_config()
    example_9_parameter_sweep()
    
    print("\n" + "=" * 60)
    print("所有示例运行完毕")
    print("=" * 60)
