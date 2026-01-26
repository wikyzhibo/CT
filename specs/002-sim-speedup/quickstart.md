# Quick Start Guide: 性能优化使用指南

**Date**: 2026-01-25  
**Feature**: 002-sim-speedup

## Overview

本指南帮助您快速使用性能优化功能，包括极速模式配置和性能测试方法。

## Prerequisites

- Python 3.11+
- NumPy
- 项目依赖已安装

## Basic Usage

### 1. 启用极速模式（推荐用于训练）

```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

# 创建极速模式配置
config = PetriEnvConfig(
    n_wafer=4,              # 4 个晶圆
    training_phase=2,        # 阶段2（完整奖励）
    turbo_mode=True,         # 启用极速模式
    stop_on_scrap=True       # 报废时停止
)

# 创建环境
env = Petri(config=config)
env.reset()
```

### 2. 性能测试

```python
import time

# 性能测试：执行 100,000 步
env.reset()
start = time.time()

for i in range(100000):
    enabled = env.get_enable_t()
    if enabled:
        done, reward, scrap = env.step(
            t=enabled[0], 
            with_reward=True, 
            detailed_reward=False  # 极速模式建议不使用详细奖励
        )
        if done:
            env.reset()
    else:
        env.step(wait=True, with_reward=True, detailed_reward=False)

elapsed = time.time() - start
steps_per_sec = 100000 / elapsed

print(f"执行时间: {elapsed:.2f}秒")
print(f"步数/秒: {steps_per_sec:.0f}")
print(f"目标: >100,000 步/秒")
print(f"是否达标: {'✓' if steps_per_sec > 100000 else '✗'}")
```

### 3. 功能一致性验证

```python
import numpy as np

# 使用固定随机种子确保可重复
np.random.seed(42)

# 创建两个环境：优化前和优化后
config_old = PetriEnvConfig(
    n_wafer=4,
    turbo_mode=False,
    optimize_reward_calc=False,
    optimize_enable_check=False,
    optimize_state_update=False
)
env_old = Petri(config=config_old)
env_old.reset()

config_new = PetriEnvConfig(
    n_wafer=4,
    turbo_mode=True,  # 启用所有优化
    optimize_reward_calc=True,
    optimize_enable_check=True,
    optimize_state_update=True
)
env_new = Petri(config=config_new)
env_new.reset()

# 执行相同动作序列并比较结果
actions = [0, 1, 2, 3, 4, 5]  # 示例动作序列
for action in actions:
    enabled_old = env_old.get_enable_t()
    enabled_new = env_new.get_enable_t()
    
    assert enabled_old == enabled_new, f"步骤 {action}: 使能变迁不一致"
    
    if enabled_old:
        _, reward_old, scrap_old = env_old.step(
            t=enabled_old[0], 
            with_reward=True
        )
        _, reward_new, scrap_new = env_new.step(
            t=enabled_new[0], 
            with_reward=True
        )
        
        assert abs(reward_old - reward_new) < 1e-6, f"步骤 {action}: 奖励不一致"
        assert scrap_old == scrap_new, f"步骤 {action}: 报废状态不一致"

print("✓ 功能一致性验证通过")
```

## Advanced Usage

### 1. 选择性启用优化

```python
# 只启用奖励计算优化，其他保持原样
config = PetriEnvConfig(
    n_wafer=4,
    optimize_reward_calc=True,      # 启用
    optimize_enable_check=False,     # 禁用
    optimize_state_update=False,     # 禁用
    cache_indices=False              # 禁用
)

env = Petri(config=config)
```

### 2. 性能分析

```python
import cProfile
import pstats

# 使用 cProfile 进行性能分析
profiler = cProfile.Profile()
profiler.enable()

# 执行代码
env.reset()
for _ in range(10000):
    enabled = env.get_enable_t()
    if enabled:
        env.step(t=enabled[0], with_reward=True, detailed_reward=False)
    else:
        env.step(wait=True, with_reward=True, detailed_reward=False)

profiler.disable()

# 分析结果
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 显示前20个最耗时的函数
```

### 3. 批量训练性能测试

```python
import time

config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,
    turbo_mode=True
)
env = Petri(config=config)

# 运行 100 个 episode
start = time.time()
episode_count = 0

while episode_count < 100:
    env.reset()
    done = False
    
    while not done:
        enabled = env.get_enable_t()
        if enabled:
            done, reward, scrap = env.step(
                t=enabled[0], 
                with_reward=True, 
                detailed_reward=False
            )
        else:
            done, reward, scrap = env.step(
                wait=True, 
                with_reward=True, 
                detailed_reward=False
            )
    
    episode_count += 1

elapsed = time.time() - start
avg_time_per_episode = elapsed / 100

print(f"100 个 episode 总时间: {elapsed:.2f}秒")
print(f"平均每个 episode: {avg_time_per_episode:.3f}秒")
print(f"目标: 比优化前减少至少 20%")
```

## Performance Benchmarks

### 基准测试脚本

创建 `tests/test_performance.py`:

```python
import time
import pytest
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def test_turbo_mode_performance():
    """测试极速模式性能：100k 步/秒"""
    config = PetriEnvConfig(
        n_wafer=4,
        turbo_mode=True
    )
    env = Petri(config=config)
    env.reset()
    
    start = time.time()
    for _ in range(100000):
        enabled = env.get_enable_t()
        if enabled:
            env.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"性能测试失败: {elapsed}秒 > 1秒"

def test_optimization_improvement():
    """测试优化效果：1000 步执行时间减少至少 20%"""
    # 优化前
    config_old = PetriEnvConfig(
        n_wafer=4,
        optimize_reward_calc=False,
        optimize_enable_check=False,
        optimize_state_update=False
    )
    env_old = Petri(config=config_old)
    env_old.reset()
    
    start = time.time()
    for _ in range(1000):
        enabled = env_old.get_enable_t()
        if enabled:
            env_old.step(t=enabled[0], with_reward=True)
        else:
            env_old.step(wait=True, with_reward=True)
    time_old = time.time() - start
    
    # 优化后
    config_new = PetriEnvConfig(
        n_wafer=4,
        optimize_reward_calc=True,
        optimize_enable_check=True,
        optimize_state_update=True
    )
    env_new = Petri(config=config_new)
    env_new.reset()
    
    start = time.time()
    for _ in range(1000):
        enabled = env_new.get_enable_t()
        if enabled:
            env_new.step(t=enabled[0], with_reward=True)
        else:
            env_new.step(wait=True, with_reward=True)
    time_new = time.time() - start
    
    improvement = (time_old - time_new) / time_old * 100
    assert improvement >= 20, f"优化效果不足: {improvement:.1f}% < 20%"
```

## Troubleshooting

### 性能未达标

1. **检查配置**: 确保 `turbo_mode=True` 且 `detailed_reward=False`
2. **检查硬件**: 确保在标准开发机器（Intel i7/AMD Ryzen 7）上测试
3. **检查优化开关**: 确保所有优化开关都已启用

### 功能不一致

1. **检查随机种子**: 使用固定随机种子确保可重复
2. **检查配置**: 确保两个环境的配置参数完全一致
3. **检查状态**: 确保在相同初始状态下开始测试

### 内存使用增加

1. **检查缓存**: 如果内存使用超过 10%，考虑禁用 `cache_indices`
2. **检查数据规模**: 确保测试场景与生产场景一致

## Next Steps

1. 运行性能测试验证优化效果
2. 运行功能一致性测试确保正确性
3. 在强化学习训练中启用极速模式
4. 监控性能指标，持续优化
