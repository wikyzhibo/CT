# Data Model: 性能优化配置

**Date**: 2026-01-25  
**Feature**: 002-sim-speedup

## Overview

本文档描述了性能优化相关的数据模型和配置结构。主要扩展了现有的 `PetriEnvConfig` 配置类，添加极速模式和相关优化开关。

## Entities

### PerformanceConfig

性能优化配置对象，扩展自 `PetriEnvConfig`。

**Location**: `data/petri_configs/env_config.py`

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `turbo_mode` | `bool` | `False` | 极速模式开关。启用时禁用详细统计追踪（wafer_stats）以提升性能 |
| `optimize_reward_calc` | `bool` | `True` | 是否优化奖励计算（使用 NumPy 向量化） |
| `optimize_enable_check` | `bool` | `True` | 是否优化使能条件检查 |
| `optimize_state_update` | `bool` | `True` | 是否优化状态更新逻辑 |
| `cache_indices` | `bool` | `True` | 是否缓存库所和变迁索引 |

**Relationships**:
- 继承自 `PetriEnvConfig`
- 被 `Petri` 类使用

**Validation Rules**:
- `turbo_mode=True` 时，自动禁用 `wafer_stats` 追踪
- 所有优化开关默认启用，但可以通过配置禁用

**State Transitions**:
- 配置在 `Petri.__init__()` 时加载，运行时不可修改
- `reset()` 方法不改变配置，只重置状态

### CachedIndices

缓存的索引映射，用于快速查找库所和变迁。

**Location**: `solutions/Continuous_model/pn.py` (Petri 类内部)

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `_place_indices` | `Dict[str, int]` | 库所名称到索引的映射 |
| `_transition_indices` | `Dict[str, int]` | 变迁名称到索引的映射 |

**Relationships**:
- 属于 `Petri` 类实例
- 在 `__init__()` 或首次访问时构建

**Validation Rules**:
- 必须与 `id2p_name` 和 `id2t_name` 保持一致
- 在 `reset()` 时不需要重建（因为网络结构不变）

### OptimizedRewardCache

优化的奖励计算缓存，存储中间计算结果。

**Location**: `solutions/Continuous_model/pn.py` (Petri 类内部，可选)

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `_reward_cache` | `Optional[Dict]` | 奖励计算中间结果缓存（可选，用于进一步优化） |

**Relationships**:
- 属于 `Petri` 类实例
- 在 `calc_reward()` 中使用

**Validation Rules**:
- 缓存可能失效，需要谨慎使用
- 当前实现可能不使用缓存（先实现向量化优化）

## Data Flow

### 配置加载流程

```
PetriEnvConfig (with turbo_mode)
    ↓
Petri.__init__(config)
    ↓
self.config = config
self.turbo_mode = config.turbo_mode
    ↓
构建缓存索引（如果 cache_indices=True）
    ↓
初始化优化标志
```

### 性能优化执行流程

```
step() 调用
    ↓
检查优化标志
    ↓
[如果 optimize_enable_check]
    _resource_enable() (优化版本)
    ↓
[如果 optimize_state_update]
    _fire() (优化版本)
    ↓
[如果 optimize_reward_calc]
    calc_reward() (向量化版本)
    ↓
[如果 not turbo_mode]
    _track_wafer_statistics()
    ↓
返回结果
```

## Constraints

### 功能一致性约束

1. **核心功能必须保留**:
   - 状态转换逻辑
   - 奖励计算（数值必须一致）
   - 报废检测
   - 释放时间链式更新

2. **可选功能可禁用**:
   - 详细统计追踪（wafer_stats）在 `turbo_mode=True` 时可禁用

### 性能约束

1. **内存使用**: 优化不应增加内存使用超过 10%
2. **API 兼容性**: 不能改变外部接口
3. **向后兼容**: 默认配置应与优化前行为一致

## Migration Notes

### 现有代码兼容性

- `PetriEnvConfig` 需要添加新字段，但提供默认值确保向后兼容
- 现有调用代码无需修改（除非要启用极速模式）
- 性能优化默认启用，但可以通过配置禁用

### 配置迁移

```python
# 旧代码（仍然有效）
config = PetriEnvConfig(n_wafer=4, training_phase=2)
env = Petri(config=config)

# 新代码（启用极速模式）
config = PetriEnvConfig(
    n_wafer=4, 
    training_phase=2,
    turbo_mode=True  # 新增
)
env = Petri(config=config)
```
