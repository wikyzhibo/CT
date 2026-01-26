# API Contracts: 在 002-sim-speedup 基础上优化数据结构

**Date**: 2026-01-26  
**Feature**: 003-optimize-data-structures

## Overview

本文档描述了数据结构优化功能的 API 合约。所有优化都是内部实现细节，不改变外部 API 接口，确保向后兼容性。

## Public API (Unchanged)

### Petri Class

所有公共 API 保持不变，与 002-sim-speedup 和原始实现兼容。

#### Methods

- `__init__(config, stop_on_scrap, training_phase, reward_config)` - 初始化 Petri 网环境
- `reset()` - 重置环境到初始状态
- `step(t, wait, with_reward, detailed_reward)` - 执行一步动作
- `get_enable_t()` - 获取当前可使能的变迁列表
- `calc_reward(t1, t2, moving_pre_places, detailed)` - 计算奖励
- `calc_wafer_statistics()` - 计算晶圆统计信息

#### Properties

- `time` - 当前系统时间
- `marks` - 库所列表（只读）
- `m` - 当前标记向量
- `n_wafer` - 晶圆数量

### Place Class

所有公共 API 保持不变。

#### Methods

- `clone()` - 克隆库所
- `head()` - 获取队头 token
- `pop_head()` - 弹出队头 token
- `append(token)` - 添加 token
- `res_time(current_time, P_Residual_time, D_Residual_time)` - 计算剩余超时时间
- `add_release(token_id, release_time)` - 添加释放时间
- `update_release(token_id, new_release_time)` - 更新释放时间
- `pop_release(token_id)` - 移除释放时间记录
- `earliest_release()` - 获取最早释放时间

#### Properties

- `name` - 库所名称
- `capacity` - 库所容量
- `processing_time` - 加工时间
- `type` - 库所类型
- `tokens` - Token 队列
- `release_schedule` - 释放时间调度队列
- `last_machine` - 上次分配的机器编号

### BasedToken Class

所有公共 API 保持不变。

#### Properties

- `enter_time` - 进入时间
- `stay_time` - 停留时间
- `token_id` - Token 唯一标识
- `machine` - 分配的机器编号

## Internal API (Optimized)

以下 API 是内部实现细节，可能已优化，但不影响外部使用。

### Internal Caches

- `_marks_by_type: Dict[int, List[Place]]` - 按类型分组的库所列表缓存
- `_pre_places_cache: Dict[int, np.ndarray]` - 前置库所索引缓存
- `_pst_places_cache: Dict[int, np.ndarray]` - 后置库所索引缓存

### Internal Methods

- `_build_marks_by_type_cache()` - 构建按类型分组的缓存
- `_update_marks_by_type_cache()` - 更新按类型分组的缓存
- `_get_marks_by_type(type: int) -> List[Place]` - 获取特定类型的库所列表

## Configuration API

### PetriEnvConfig

可以通过配置控制优化措施（如果支持）：

```python
@dataclass
class PetriEnvConfig:
    # ... 现有配置项 ...
    
    # 数据结构优化配置（可选）
    optimize_data_structures: bool = True  # 是否启用数据结构优化
```

## Compatibility Guarantees

### 向后兼容性

- ✅ 所有公共 API 保持不变
- ✅ 所有方法签名保持不变
- ✅ 所有返回值类型和结构保持不变
- ✅ 所有行为语义保持不变

### 与 002-sim-speedup 的兼容性

- ✅ 可以同时启用两种优化措施
- ✅ 优化措施不会冲突
- ✅ 性能提升可以叠加

### 序列化兼容性

- ✅ 如果使用序列化，数据结构优化不影响序列化格式
- ✅ 序列化/反序列化保持兼容

## Performance Contracts

### 性能保证

在标准开发机器上，启用数据结构优化后：

- **执行时间**: 比仅启用 002-sim-speedup 时减少至少 5%
- **频繁访问操作**: 减少至少 8%
- **内存使用**: 增加不超过 10%

### 功能保证

- **功能一致性**: 100%（与优化前完全一致）
- **兼容性**: 与 002-sim-speedup 完全兼容

## Error Handling

所有错误处理保持不变：

- `ValueError` - 无效参数
- `IndexError` - 索引越界
- `AttributeError` - 属性不存在（如果 `__slots__` 影响动态属性访问，需要处理）

## Testing Contracts

### 功能测试

所有功能测试必须通过，确保功能一致性：

```python
def test_functionality():
    # 测试功能一致性
    assert old_result == new_result
```

### 性能测试

性能测试必须达到预期目标：

```python
def test_performance():
    # 测试性能提升
    assert improvement >= 0.05  # 至少 5% 改进
```

## Migration Guide

### 从原始实现迁移

无需迁移，API 完全兼容。

### 从 002-sim-speedup 迁移

无需迁移，可以同时启用两种优化措施。

### 禁用优化

如果遇到问题，可以禁用优化：

```python
config = PetriEnvConfig(
    turbo_mode=True,  # 保持 002-sim-speedup 优化
    optimize_data_structures=False,  # 禁用数据结构优化
)
```
