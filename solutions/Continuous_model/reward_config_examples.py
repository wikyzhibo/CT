"""
奖励配置示例

展示如何使用 reward_config 参数来灵活控制奖励函数的各个组件。
"""

from solutions.PPO.enviroment import Env_PN

# ============ 示例 1: 默认配置（所有奖励启用）============
print("=" * 60)
print("示例 1: 默认配置")
print("=" * 60)

env_default = Env_PN(training_phase=2)
print("使用默认配置，所有奖励组件启用")
print(f"奖励配置: {env_default.net.reward_config}")
print()

# ============ 示例 2: Phase 1 配置（仅报废惩罚）============
print("=" * 60)
print("示例 2: Phase 1 配置")
print("=" * 60)

# training_phase=1 会自动禁用 transport_penalty
env_phase1 = Env_PN(training_phase=1)
print("Phase 1: 仅考虑加工腔室约束，运输位超时惩罚被禁用")
print(f"奖励配置: {env_phase1.net.reward_config}")
print()

# ============ 示例 3: 自定义配置（只启用基础奖励）============
print("=" * 60)
print("示例 3: 只启用基础奖励和惩罚")
print("=" * 60)

reward_config_basic = {
    'proc_reward': 1,        # 启用：加工奖励
    'safe_reward': 0,        # 禁用：安全裕量奖励
    'penalty': 1,            # 启用：加工腔室超时惩罚
    'warn_penalty': 0,       # 禁用：预警惩罚
    'transport_penalty': 1,  # 启用：运输位超时惩罚
    'congestion_penalty': 0, # 禁用：堵塞预测惩罚
    'time_cost': 1,          # 启用：时间成本
}

env_basic = Env_PN(training_phase=2, reward_config=reward_config_basic)
print("只启用基础的加工奖励、超时惩罚和时间成本")
print(f"奖励配置: {env_basic.net.reward_config}")
print()

# ============ 示例 4: 消融实验配置（逐个移除奖励）============
print("=" * 60)
print("示例 4: 消融实验 - 禁用堵塞预测惩罚")
print("=" * 60)

reward_config_ablation = {
    'proc_reward': 1,
    'safe_reward': 1,
    'penalty': 1,
    'warn_penalty': 1,
    'transport_penalty': 1,
    'congestion_penalty': 0,  # 禁用堵塞预测，观察影响
    'time_cost': 1,
}

env_ablation = Env_PN(training_phase=2, reward_config=reward_config_ablation)
print("用于消融实验：禁用堵塞预测惩罚，观察对训练的影响")
print(f"奖励配置: {env_ablation.net.reward_config}")
print()

# ============ 示例 5: 极简配置（最小奖励集）============
print("=" * 60)
print("示例 5: 极简配置")
print("=" * 60)

reward_config_minimal = {
    'proc_reward': 0,        # 禁用所有正向奖励
    'safe_reward': 0,
    'penalty': 1,            # 只保留必要的惩罚
    'warn_penalty': 0,
    'transport_penalty': 1,
    'congestion_penalty': 0,
    'time_cost': 1,          # 保留时间成本
}

env_minimal = Env_PN(training_phase=2, reward_config=reward_config_minimal)
print("极简配置：只保留必要的惩罚和时间成本")
print(f"奖励配置: {env_minimal.net.reward_config}")
print()

# ============ 示例 6: 三阶段训练配置 ============
print("=" * 60)
print("示例 6: 三阶段课程学习")
print("=" * 60)

# Stage 1: 只学习加工奖励和基础惩罚
reward_config_stage1 = {
    'proc_reward': 1,
    'safe_reward': 0,
    'penalty': 1,            # 加工腔室超时
    'warn_penalty': 0,
    'transport_penalty': 0,  # 暂不考虑运输
    'congestion_penalty': 0,
    'time_cost': 1,
}

# Stage 2: 添加运输位约束
reward_config_stage2 = {
    'proc_reward': 1,
    'safe_reward': 0,
    'penalty': 1,
    'warn_penalty': 0,
    'transport_penalty': 1,  # 添加运输约束
    'congestion_penalty': 0,
    'time_cost': 1,
}

# Stage 3: 添加所有优化奖励
reward_config_stage3 = {
    'proc_reward': 1,
    'safe_reward': 1,
    'penalty': 1,
    'warn_penalty': 1,
    'transport_penalty': 1,
    'congestion_penalty': 1,  # 添加堵塞预测
    'time_cost': 1,
}

print("Stage 1: 基础约束")
print(f"  配置: {reward_config_stage1}")
print()
print("Stage 2: 添加运输约束")
print(f"  配置: {reward_config_stage2}")
print()
print("Stage 3: 完整奖励")
print(f"  配置: {reward_config_stage3}")
print()

# ============ 使用方法总结 ============
print("=" * 60)
print("使用方法总结")
print("=" * 60)
print("""
1. 默认使用（所有奖励启用）:
   env = Env_PN()

2. 两阶段训练（通过 training_phase 控制）:
   env_phase1 = Env_PN(training_phase=1)  # 禁用运输惩罚
   env_phase2 = Env_PN(training_phase=2)  # 启用所有奖励

3. 自定义奖励配置:
   custom_config = {
       'proc_reward': 1,
       'transport_penalty': 0,
       ...
   }
   env = Env_PN(reward_config=custom_config)

4. 消融实验:
   逐个禁用奖励组件，观察对训练效果的影响

5. 课程学习:
   逐步增加奖励组件的复杂度，从简单到复杂

奖励配置字典的键:
- proc_reward: 加工奖励
- safe_reward: 安全裕量奖励
- penalty: 加工腔室超时惩罚
- warn_penalty: 预警惩罚
- transport_penalty: 运输位超时惩罚
- congestion_penalty: 堵塞预测惩罚
- time_cost: 时间成本

值: 1=启用, 0=禁用
""")
