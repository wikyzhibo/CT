# Quick Start Guide: 连续 Petri 网晶圆加工仿真系统

**Date**: 2026-01-25  
**Feature**: 001-continuous-petri

## Overview

本指南帮助您快速开始使用连续 Petri 网晶圆加工仿真系统。系统用于模拟半导体制造中的晶圆加工流程，支持强化学习训练。

## Prerequisites

- Python 3.x
- NumPy
- 项目依赖已安装

## Basic Usage

### 1. 创建环境

```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

# 创建配置
config = PetriEnvConfig(
    n_wafer=4,           # 4 个晶圆
    training_phase=2,     # 阶段2（完整奖励）
    stop_on_scrap=True   # 报废时停止
)

# 创建环境
env = Petri(config=config)
```

### 2. 重置环境

```python
env.reset()
print(f"系统时间: {env.time}")
print(f"LP 中的晶圆数: {env.m[env.id2p_name.index('LP')]}")
```

### 3. 执行动作

```python
# 获取可使能的变迁
enabled = env.get_enable_t()
print(f"可使能的变迁: {[env.id2t_name[t] for t in enabled]}")

# 执行第一个可使能的变迁
if enabled:
    t = enabled[0]
    done, reward, scrap = env.step(t=t, with_reward=True)
    print(f"奖励: {reward}, 完成: {done}, 报废: {scrap}")
```

### 4. 执行 WAIT 动作

```python
# 等待 5 秒
done, reward, scrap = env.step(wait=True, with_reward=True)
print(f"等待后奖励: {reward}")
```

### 5. 获取详细奖励信息

```python
done, reward_dict, scrap = env.step(
    t=t, 
    with_reward=True, 
    detailed_reward=True
)

print(f"总奖励: {reward_dict['total']}")
print(f"加工奖励: {reward_dict['proc_reward']}")
print(f"超时惩罚: {reward_dict['penalty']}")
print(f"时间成本: {reward_dict['time_cost']}")
```

## Complete Example

### 运行一个完整的 Episode

```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig
import numpy as np

# 创建环境
config = PetriEnvConfig(n_wafer=4, training_phase=2)
env = Petri(config=config)
env.reset()

# 运行直到完成或报废
total_reward = 0
step_count = 0
max_steps = 1000

while step_count < max_steps:
    # 获取可使能的变迁
    enabled = env.get_enable_t()
    
    if not enabled:
        # 没有可使能的变迁，执行 WAIT
        done, reward, scrap = env.step(wait=True, with_reward=True)
    else:
        # 随机选择一个可使能的变迁
        t = np.random.choice(enabled)
        done, reward, scrap = env.step(t=t, with_reward=True)
    
    total_reward += reward
    step_count += 1
    
    if done:
        print(f"Episode 结束: 完成={not scrap}, 报废={scrap}")
        print(f"总步数: {step_count}, 总奖励: {total_reward:.2f}")
        print(f"最终时间: {env.time}")
        break

# 获取统计信息
stats = env.calc_wafer_statistics()
print(f"\n统计信息:")
print(f"  已完成晶圆数: {stats['completed_count']}")
print(f"  平均系统滞留时间: {stats['system_avg']:.2f} 秒")
print(f"  最大系统滞留时间: {stats['system_max']} 秒")
```

### 生成甘特图

```python
# 运行一个 episode 后
env.render_gantt("results/my_gantt.png")
print("甘特图已保存到 results/my_gantt.png")
```

## Configuration Examples

### 阶段1配置（仅报废惩罚）

```python
config = PetriEnvConfig(
    n_wafer=4,
    training_phase=1,  # 阶段1
    stop_on_scrap=True
)
env = Petri(config=config)
```

### 阶段2配置（完整奖励）

```python
config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,  # 阶段2
    R_done=20,
    R_finish=500,
    R_scrap=100,
    stop_on_scrap=True
)
env = Petri(config=config)
```

### 自定义奖励配置

```python
config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,
    reward_config={
        'proc_reward': 1,        # 启用加工奖励
        'safe_reward': 1,        # 启用安全裕量奖励
        'penalty': 1,            # 启用超时惩罚
        'warn_penalty': 0,       # 禁用预警惩罚
        'transport_penalty': 1,  # 启用运输惩罚
        'congestion_penalty': 0, # 禁用堵塞惩罚
        'time_cost': 1,          # 启用时间成本
        'release_violation_penalty': 1  # 启用释放违规惩罚
    }
)
env = Petri(config=config)
```

### 从文件加载配置

```python
# 保存配置
config = PetriEnvConfig(n_wafer=4, training_phase=2)
config.save("data/petri_configs/my_config.json")

# 加载配置
config = PetriEnvConfig.load("data/petri_configs/my_config.json")
env = Petri(config=config)
```

## Common Patterns

### 检查系统状态

```python
# 检查是否完成
lp_done_idx = env.id2p_name.index("LP_done")
is_finished = (env.m[lp_done_idx] == env.n_wafer)

# 检查是否有报废
is_scrap, scrap_info = env._check_scrap(return_info=True)
if is_scrap:
    print(f"检测到报废: {scrap_info}")

# 检查系统时间
if env.time >= 20000:  # MAX_TIME
    print("系统超时")
```

### 遍历所有库所

```python
for i, place in enumerate(env.marks):
    print(f"{place.name}: {len(place.tokens)}/{place.capacity} tokens")
    if place.tokens:
        for token in place.tokens:
            print(f"  Token {token.token_id}: 进入时间={token.enter_time}, 滞留={token.stay_time}")
```

### 查询特定变迁

```python
# 查找变迁索引
if "u_LP_s1" in env.id2t_name:
    t_idx = env.id2t_name.index("u_LP_s1")
    print(f"变迁 u_LP_s1 的索引: {t_idx}")
    
    # 检查是否可使能
    enabled = env.get_enable_t()
    if t_idx in enabled:
        print("该变迁当前可使能")
```

## Troubleshooting

### 问题：没有可使能的变迁

**原因**: 可能所有前置条件不满足，或容量已满。

**解决**:
```python
# 执行 WAIT 动作，等待时间推进
env.step(wait=True)
```

### 问题：奖励始终为负

**原因**: 可能是时间成本过高，或惩罚过多。

**解决**:
```python
# 调整配置
config = PetriEnvConfig(
    c_time=0.1,  # 降低时间成本
    reward_config={'time_cost': 0}  # 或禁用时间成本
)
```

### 问题：系统很快报废

**原因**: 可能是动作选择不当，导致晶圆滞留时间过长。

**解决**:
- 检查动作选择策略
- 增加 `P_Residual_time` 或 `D_Residual_time`
- 优化调度策略

## Next Steps

- 阅读 [data-model.md](./data-model.md) 了解数据模型
- 阅读 [contracts/api.md](./contracts/api.md) 了解完整 API
- 查看 `solutions/Continuous_model/test_env.py` 了解测试示例
- 集成到强化学习框架进行训练
