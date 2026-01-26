# Quick Start: 在 002-sim-speedup 基础上优化数据结构

**Date**: 2026-01-26  
**Feature**: 003-optimize-data-structures

## Overview

本指南介绍如何启用和使用数据结构优化功能，在 002-sim-speedup 已实现的性能优化基础上，进一步提升 Petri 网模拟器的性能。

## Prerequisites

- 已实现 002-sim-speedup 的优化措施
- Python 3.11+
- NumPy 已安装

## Basic Usage

### 1. 启用数据结构优化

数据结构优化默认启用（如果配置支持）。可以通过 `PetriEnvConfig` 控制：

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 启用所有优化（包括 002-sim-speedup 和数据结构优化）
config = PetriEnvConfig(
    n_wafer=24,
    training_phase=2,
    turbo_mode=True,  # 启用 002-sim-speedup 的极速模式
    optimize_data_structures=True,  # 启用数据结构优化（如果支持）
)

env = Petri(config=config)
env.reset()
```

### 2. 运行模拟

使用方式与之前完全相同：

```python
# 获取使能变迁
enabled = env.get_enable_t()

# 执行动作
if enabled:
    env.step(t=enabled[0], with_reward=True, detailed_reward=False)
else:
    env.step(wait=True, with_reward=True, detailed_reward=False)
```

### 3. 验证功能一致性

运行功能一致性测试，确保优化不破坏现有功能：

```bash
python -m pytest tests/test_functionality.py -v
```

### 4. 性能基准测试

运行性能基准测试，验证优化效果：

```bash
python -m pytest tests/test_performance.py::TestPerformance::test_data_structure_optimization -v -s
```

## Performance Comparison

### 预期性能提升

在标准开发机器上，启用数据结构优化后：

- **执行时间**: 比仅启用 002-sim-speedup 时减少至少 5%
- **频繁访问操作**: 减少至少 8%
- **内存使用**: 增加不超过 10%

### 性能测试示例

```python
import time
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 仅启用 002-sim-speedup
config_old = PetriEnvConfig(
    n_wafer=24,
    training_phase=2,
    turbo_mode=True,
    optimize_data_structures=False,  # 禁用数据结构优化
)
env_old = Petri(config=config_old)
env_old.reset()

# 启用 002-sim-speedup + 数据结构优化
config_new = PetriEnvConfig(
    n_wafer=24,
    training_phase=2,
    turbo_mode=True,
    optimize_data_structures=True,  # 启用数据结构优化
)
env_new = Petri(config=config_new)
env_new.reset()

# 性能测试
target_steps = 100000

# 测试旧版本
start = time.time()
for _ in range(target_steps):
    enabled = env_old.get_enable_t()
    if enabled:
        env_old.step(t=enabled[0], with_reward=True, detailed_reward=False)
    else:
        env_old.step(wait=True, with_reward=True, detailed_reward=False)
    if env_old.m[env_old.id2p_name.index("LP_done")] == env_old.n_wafer:
        env_old.reset()
elapsed_old = time.time() - start

# 测试新版本
start = time.time()
for _ in range(target_steps):
    enabled = env_new.get_enable_t()
    if enabled:
        env_new.step(t=enabled[0], with_reward=True, detailed_reward=False)
    else:
        env_new.step(wait=True, with_reward=True, detailed_reward=False)
    if env_new.m[env_new.id2p_name.index("LP_done")] == env_new.n_wafer:
        env_new.reset()
elapsed_new = time.time() - start

# 计算改进
improvement = (elapsed_old - elapsed_new) / elapsed_old * 100
print(f"执行时间改进: {improvement:.1f}%")
print(f"旧版本: {elapsed_old:.3f}秒")
print(f"新版本: {elapsed_new:.3f}秒")
```

## Optimization Details

### 1. `__slots__` 优化

`Place` 和 `BasedToken` 类使用 `__slots__` 减少内存占用和属性访问开销：

```python
@dataclass
class Place:
    __slots__ = ('name', 'capacity', 'processing_time', 'type', 
                 'tokens', 'release_schedule', 'last_machine')
    # ... 字段定义
```

### 2. 按类型分组访问

使用 `_marks_by_type` 缓存按类型分组的库所列表：

```python
# 优化前：遍历所有库所
for place in self.marks:
    if place.type == 1:
        # 处理加工腔室

# 优化后：直接访问特定类型的库所
for place in self._marks_by_type[1]:
    # 处理加工腔室
```

### 3. 字典访问优化

使用局部变量缓存字典引用：

```python
# 优化前
t_name = self.id2t_name[t]

# 优化后
id2t_name = self.id2t_name
t_name = id2t_name[t]
```

### 4. NumPy 数组优化

使用 `np.flatnonzero` 而非 `np.nonzero`：

```python
# 优化前
pre_places = np.nonzero(self.pre[:, t] > 0)[0]

# 优化后
pre_places = np.flatnonzero(self.pre[:, t] > 0)
```

## Troubleshooting

### 问题：功能不一致

如果发现功能不一致，检查：

1. 是否同时启用了所有优化措施
2. 缓存是否正确更新（在 `reset` 中）
3. `__slots__` 是否影响了动态属性访问

### 问题：性能提升不明显

如果性能提升不明显，检查：

1. 是否在正确的配置下测试（`turbo_mode=True`）
2. 测试数据量是否足够大（至少 10,000 步）
3. 系统负载是否影响测试结果

### 问题：内存使用增加

如果内存使用增加超过 10%，检查：

1. 缓存大小是否合理
2. 是否有内存泄漏
3. 是否可以使用更紧凑的数据结构

## Next Steps

1. 阅读 [research.md](./research.md) 了解技术决策
2. 阅读 [data-model.md](./data-model.md) 了解数据结构详情
3. 运行测试验证功能和性能
4. 根据实际使用情况调整优化策略
