# Research: 在 002-sim-speedup 基础上优化数据结构以进一步加速

**Date**: 2026-01-26  
**Feature**: 003-optimize-data-structures  
**Status**: Complete

## Overview

本文档记录了在 002-sim-speedup 已实现的算法和计算逻辑优化基础上，通过优化数据结构进一步提升性能的技术决策和研究结果。目标是减少内存访问开销、属性查找开销和对象创建开销，在相同硬件配置下，执行时间比仅启用 002-sim-speedup 时减少至少 5%。

## Technology Decisions

### Decision 1: 使用 `__slots__` 优化对象内存布局

**Decision**: 为 `Place` 和 `BasedToken` 类添加 `__slots__` 属性，减少对象内存占用和属性访问开销

**Rationale**:
- Python 默认使用字典存储对象属性，每个对象都有 `__dict__`，占用额外内存
- `__slots__` 将属性存储在固定大小的数组中，减少内存占用约 30-50%
- 属性访问更快，因为不需要字典查找
- 对于频繁创建和访问的对象（如 `BasedToken`），性能提升显著

**Alternatives considered**:
- 使用 `dataclass` 的 `frozen=True`：只提供不可变性，不减少内存占用
- 使用 `namedtuple`：功能受限，不支持可变属性
- 保持现状：内存占用和访问开销较大

**Implementation**: 
- 为 `Place` 类添加 `__slots__ = ('name', 'capacity', 'processing_time', 'type', 'tokens', 'release_schedule', 'last_machine')`
- 为 `BasedToken` 类添加 `__slots__ = ('enter_time', 'stay_time', 'token_id', 'machine')`
- 注意：`tokens` 和 `release_schedule` 是 `deque`，不能放入 `__slots__`，但 `Place` 对象本身的内存占用会减少

### Decision 2: 按类型分组访问 marks 列表

**Decision**: 预先按类型分组 `marks` 列表，避免频繁遍历所有库所

**Rationale**:
- 当前代码中频繁遍历 `marks` 列表查找特定类型的库所（如 `type=1` 的加工腔室）
- 每次遍历都需要检查所有库所的类型，开销较大
- 预先按类型分组，可以快速访问特定类型的库所
- 对于奖励计算、状态更新等频繁操作，性能提升明显

**Alternatives considered**:
- 保持遍历方式：简单但性能开销大
- 使用字典索引：需要维护索引，但访问更快
- 使用 NumPy 布尔掩码：适合向量化操作，但需要额外转换

**Implementation**:
- 在 `__init__` 中构建 `_marks_by_type: Dict[int, List[Place]]` 缓存
- 在 `reset` 中更新缓存（如果 `marks` 列表发生变化）
- 在奖励计算、状态更新等操作中使用缓存，避免遍历所有库所

### Decision 3: 优化字典访问模式

**Decision**: 减少字典查找次数，使用局部变量缓存频繁访问的字典

**Rationale**:
- 当前代码中频繁访问 `id2p_name`、`id2t_name` 等字典
- 字典查找虽然平均 O(1)，但在热点路径中仍有开销
- 使用局部变量缓存字典引用，减少属性查找开销
- 对于频繁调用的函数（如 `_fire`、`_resource_enable`），性能提升明显

**Alternatives considered**:
- 保持字典访问：简单但性能开销较大
- 使用列表替代字典：如果索引是连续的，列表访问更快，但当前索引可能不连续
- 使用 `__slots__` 优化字典对象：不适用，字典是内置类型

**Implementation**:
- 在热点函数中使用局部变量缓存字典引用：`id2p_name = self.id2p_name`
- 在循环外缓存字典，避免重复属性查找
- 对于频繁访问的映射（如 `release_chain`），考虑使用列表或 NumPy 数组替代

### Decision 4: 优化 NumPy 数组访问模式

**Decision**: 减少不必要的数组复制和转换，使用视图而非副本

**Rationale**:
- NumPy 数组操作可能产生副本，增加内存分配和复制开销
- 使用视图（view）可以避免复制，提高性能
- 对于大型数组（如 `pre`、`pst`），避免复制可以显著提升性能
- 在向量化操作中，使用 `np.flatnonzero` 而非 `np.nonzero` 可以减少维度

**Alternatives considered**:
- 保持数组复制：安全但性能开销大
- 使用原地操作：可能改变原始数据，需要谨慎
- 使用视图：性能好，但需要确保不修改原始数据

**Implementation**:
- 在 `_resource_enable` 等函数中使用 `np.flatnonzero` 而非 `np.nonzero`
- 使用数组视图而非副本：`arr_view = arr[:]` 而非 `arr_copy = arr.copy()`
- 在向量化操作中，避免不必要的类型转换

### Decision 5: 优化 release_schedule 的存储和访问

**Decision**: 考虑使用更高效的数据结构存储 `release_schedule`，或优化查找算法

**Rationale**:
- 当前 `release_schedule` 使用 `deque` 存储 `(token_id, release_time)` 元组
- `earliest_release()` 需要遍历所有元素找最小值，时间复杂度 O(n)
- 如果使用堆（heap）或排序列表，可以更快找到最小值
- 但考虑到 `release_schedule` 通常元素较少（<10），优化收益可能有限

**Alternatives considered**:
- 使用 `heapq`：插入和查找最小值都是 O(log n)，但需要维护堆
- 使用排序列表：查找最小值是 O(1)，但插入是 O(n)
- 保持 `deque`：简单但查找最小值是 O(n)
- 使用 NumPy 数组：如果元素较多，可以向量化查找

**Implementation**:
- 如果 `release_schedule` 元素较少（<10），保持 `deque`，但优化 `earliest_release()` 使用 `min()` 的内置优化
- 如果元素较多，考虑使用 `heapq` 或 NumPy 数组
- 优化 `update_release()` 和 `pop_release()` 的查找算法

### Decision 6: 优化缓存数据结构的访问

**Decision**: 优化 `_pre_places_cache` 和 `_pst_places_cache` 的访问模式，减少字典查找

**Rationale**:
- 当前缓存使用字典存储，每次访问都需要字典查找
- 如果缓存命中率高，可以考虑使用列表或 NumPy 数组替代
- 对于频繁访问的缓存，使用局部变量缓存可以减少属性查找开销

**Alternatives considered**:
- 保持字典缓存：灵活但查找有开销
- 使用列表缓存：如果索引连续，访问更快，但需要处理缺失值
- 使用 NumPy 数组缓存：适合向量化操作，但需要处理动态大小

**Implementation**:
- 在热点函数中使用局部变量缓存字典引用
- 对于频繁访问的缓存项，考虑预计算并存储在实例变量中
- 优化缓存的构建和更新逻辑，减少不必要的计算

## Performance Analysis

### 预期性能提升

1. **`__slots__` 优化**: 
   - 内存占用减少 30-50%
   - 属性访问速度提升 10-20%
   - 对象创建速度提升 5-10%

2. **按类型分组访问**:
   - 奖励计算中遍历库所的时间减少 50-70%（如果只访问特定类型）
   - 状态更新中遍历库所的时间减少 50-70%

3. **字典访问优化**:
   - 热点函数中的字典查找开销减少 5-10%
   - 总体性能提升 2-5%

4. **NumPy 数组优化**:
   - 数组操作时间减少 10-20%
   - 内存使用减少（避免不必要的副本）

5. **综合效果**:
   - 执行时间减少 5-10%（目标：至少 5%）
   - 频繁访问操作减少 8-15%（目标：至少 8%）

### 风险评估

1. **兼容性风险**: 
   - `__slots__` 可能影响某些动态属性访问
   - 需要确保所有代码路径都兼容

2. **内存风险**:
   - 按类型分组需要额外内存存储索引
   - 需要确保内存使用增加不超过 10%

3. **维护性风险**:
   - 优化后的代码可能更复杂
   - 需要确保代码可读性和可维护性

## Implementation Strategy

1. **渐进式优化**: 先实现 `__slots__` 优化，验证功能一致性后再实现其他优化
2. **性能测试**: 每个优化措施都要进行性能基准测试，确保达到预期效果
3. **功能测试**: 每个优化措施都要进行功能一致性测试，确保不破坏现有功能
4. **兼容性测试**: 确保与 002-sim-speedup 的优化措施兼容，可以同时启用

## References

- Python `__slots__` 文档: https://docs.python.org/3/reference/datamodel.html#slots
- NumPy 性能优化指南: https://numpy.org/doc/stable/user/basics.performance.html
- Python 性能优化最佳实践: https://docs.python.org/3/howto/optimization.html
