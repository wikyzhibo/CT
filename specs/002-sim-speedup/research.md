# Research: 加速 Petri 网模拟器性能

**Date**: 2026-01-25  
**Feature**: 002-sim-speedup  
**Status**: Complete

## Overview

本文档记录了 Petri 网模拟器性能优化的技术决策和研究结果。目标是实现在标准开发机器上 1 秒内执行超过 100,000 个模拟步数，同时保持核心功能的完全一致性。

## Technology Decisions

### Decision 1: 代码分析和性能分析工具

**Decision**: 使用 Python 内置的 `cProfile` 和 `line_profiler` 进行性能分析

**Rationale**:
- `cProfile` 是 Python 标准库，无需额外依赖，可以快速识别热点函数
- `line_profiler` 提供逐行性能分析，帮助精确定位性能瓶颈
- 结合使用可以全面了解性能瓶颈分布

**Alternatives considered**:
- `py-spy`: 功能强大但需要额外安装，且对 Windows 支持有限
- `memory_profiler`: 主要用于内存分析，当前需求主要关注 CPU 性能
- 手动计时：不够精确，难以定位具体瓶颈

**Implementation**: 在优化前先运行性能分析，识别 `calc_reward`, `_fire`, `_resource_enable` 等函数的性能瓶颈

### Decision 2: NumPy 向量化优化

**Decision**: 尽可能使用 NumPy 向量化操作替代 Python 循环

**Rationale**:
- NumPy 的向量化操作在 C 层面执行，比 Python 循环快 10-100 倍
- 当前代码中 `calc_reward` 函数包含大量嵌套循环遍历所有库所和 token
- 可以通过 NumPy 的布尔索引和向量化计算大幅提升性能

**Alternatives considered**:
- Numba JIT 编译：需要额外依赖，且对动态类型支持有限
- Cython：需要编译步骤，增加构建复杂度
- 保持 Python 循环：性能提升有限，无法达到 100k 步/秒目标

**Implementation**: 
- 将 `calc_reward` 中的循环改为 NumPy 向量化操作
- 使用 NumPy 数组存储 token 时间信息，避免逐个访问
- 使用布尔掩码筛选需要计算的 token

### Decision 3: 数据结构优化

**Decision**: 优化数据结构访问模式，减少不必要的遍历和查找

**Rationale**:
- 当前代码使用 `deque` 存储 token，每次访问需要遍历
- 字典查找比列表索引慢，应尽量减少字典操作
- 缓存频繁访问的数据（如库所索引、变迁名称映射）

**Alternatives considered**:
- 使用 NumPy 数组替代 deque：需要重构较多代码，且 deque 的 FIFO 特性很重要
- 使用 C 扩展：过度工程化，Python 优化已足够
- 保持现有结构：无法达到性能目标

**Implementation**:
- 缓存 `id2p_name.index()` 和 `id2t_name.index()` 的结果
- 使用列表而非字典存储频繁访问的映射关系
- 优化 `_update_stay_times` 函数，只在必要时更新

### Decision 4: 条件检查优化

**Decision**: 减少重复的条件检查和提前退出

**Rationale**:
- `_resource_enable` 和 `get_enable_t` 函数在每次 step 时都被调用
- 当前实现检查所有变迁的使能条件，即使大部分不可使能
- 可以通过早期退出和条件短路优化减少计算

**Alternatives considered**:
- 缓存使能状态：状态变化频繁，缓存失效成本高
- 增量更新：实现复杂，收益有限
- 保持现有逻辑：无法达到性能目标

**Implementation**:
- 使用 NumPy 的向量化操作检查使能条件
- 提前退出不可使能的变迁检查
- 优化容量检查逻辑

### Decision 5: 极速模式实现策略

**Decision**: 通过配置标志和条件编译（运行时）实现极速模式

**Rationale**:
- 极速模式需要禁用详细统计追踪（wafer_stats），但保留核心功能
- 使用配置标志可以在运行时切换，无需重新编译
- 通过条件判断避免不必要的计算，而不是完全移除代码

**Alternatives considered**:
- 完全分离代码路径：代码重复，维护成本高
- 使用装饰器：增加函数调用开销，不适合高频调用
- 编译时优化：Python 不支持，需要 Cython 等工具

**Implementation**:
- 在 `PetriEnvConfig` 中添加 `turbo_mode` 标志
- 在 `_track_wafer_statistics` 函数中添加条件检查
- 在 `calc_reward` 中根据配置跳过不必要的计算分支

### Decision 6: 奖励计算优化

**Decision**: 优化 `calc_reward` 函数的计算逻辑，减少重复计算和循环

**Rationale**:
- `calc_reward` 是性能瓶颈之一，包含大量嵌套循环
- 当前实现为每个 token 计算时间重叠，存在重复计算
- 可以通过批量计算和缓存中间结果优化

**Alternatives considered**:
- 延迟计算奖励：不符合实时性要求
- 简化奖励公式：改变业务逻辑，不符合需求
- 保持现有实现：无法达到性能目标

**Implementation**:
- 使用 NumPy 向量化计算时间重叠
- 缓存处理时间和截止时间，避免重复计算
- 根据 `reward_config` 提前退出不需要的计算分支

### Decision 7: 释放时间追踪优化

**Decision**: 优化释放时间链式更新逻辑，但必须保留功能

**Rationale**:
- 释放时间链式更新是核心功能，不能禁用
- 当前实现使用循环遍历和字典查找，可以优化
- 可以通过预计算链路和批量更新提升性能

**Alternatives considered**:
- 禁用链式更新：违反需求（链式冲突检测很重要）
- 简化更新逻辑：可能影响准确性
- 保持现有实现：性能瓶颈之一

**Implementation**:
- 预计算释放时间链路映射（已在 `_build_release_chain` 中实现）
- 使用列表而非字典存储释放时间队列
- 优化 `_chain_record_release` 和 `_chain_update_release` 函数

### Decision 8: 性能测试策略

**Decision**: 创建专门的性能基准测试套件

**Rationale**:
- 需要验证优化效果和功能一致性
- 性能测试需要可重复和可比较的结果
- 功能一致性测试确保优化不改变核心行为

**Alternatives considered**:
- 手动测试：不够系统化，难以复现
- 集成到现有测试：可能影响现有测试性能
- 不进行测试：无法验证优化效果

**Implementation**:
- 创建 `tests/test_performance.py` 进行性能基准测试
- 创建 `tests/test_functionality.py` 验证功能一致性
- 使用固定随机种子确保结果可重复

## Performance Optimization Techniques

### 1. NumPy 向量化

**技术**: 使用 NumPy 数组和向量化操作替代 Python 循环

**示例**:
```python
# 优化前：Python 循环
for tok in place.tokens:
    if tok.enter_time < t2:
        # 计算奖励

# 优化后：NumPy 向量化
enter_times = np.array([tok.enter_time for tok in place.tokens])
mask = enter_times < t2
rewards = np.sum(enter_times[mask] * r)
```

**预期收益**: 10-50 倍性能提升（取决于数据规模）

### 2. 缓存频繁访问的数据

**技术**: 缓存库所索引、变迁名称映射等频繁访问的数据

**示例**:
```python
# 优化前：每次查找
s1_idx = self.id2p_name.index("s1")

# 优化后：缓存结果
if not hasattr(self, '_place_indices'):
    self._place_indices = {name: idx for idx, name in enumerate(self.id2p_name)}
s1_idx = self._place_indices["s1"]
```

**预期收益**: 减少字典查找开销，提升 2-5 倍

### 3. 条件短路和提前退出

**技术**: 在循环中使用条件短路，提前退出不必要的计算

**示例**:
```python
# 优化前：总是计算所有分支
if self.reward_config.get('proc_reward', 1):
    # 计算加工奖励
if self.reward_config.get('penalty', 1):
    # 计算惩罚

# 优化后：提前退出
if not self.reward_config.get('proc_reward', 1):
    return 0.0  # 如果不需要奖励计算，直接返回
```

**预期收益**: 减少不必要的计算，提升 20-50%

### 4. 批量更新状态

**技术**: 批量更新 token 状态，减少函数调用开销

**示例**:
```python
# 优化前：逐个更新
for tok in place.tokens:
    tok.stay_time = int(self.time - tok.enter_time)

# 优化后：批量更新
if place.tokens:
    enter_times = np.array([tok.enter_time for tok in place.tokens])
    stay_times = (self.time - enter_times).astype(int)
    for tok, stay_time in zip(place.tokens, stay_times):
        tok.stay_time = stay_time
```

**预期收益**: 减少函数调用开销，提升 10-20%

### 5. 极速模式条件检查

**技术**: 在极速模式下跳过非关键功能的计算

**示例**:
```python
# 优化后：条件检查
if not self.config.turbo_mode:
    self._track_wafer_statistics(t_name, wafer_id, start_time, enter_new)
```

**预期收益**: 减少追踪开销，提升 5-15%

## Benchmarking Strategy

### 性能测试场景

1. **基础性能测试**: 执行 100,000 个 step，测量总时间
2. **Episode 性能测试**: 运行完整 episode，测量平均时间
3. **批量训练测试**: 连续运行 100 个 episode，测量总时间

### 功能一致性测试

1. **状态一致性**: 使用相同随机种子和动作序列，比较最终状态
2. **奖励一致性**: 比较奖励序列和总奖励
3. **事件一致性**: 比较事件日志和关键事件

### 测试环境

- 硬件：标准开发机器（Intel i7/AMD Ryzen 7 或同等性能）
- 配置：训练模式（with_reward=True, detailed_reward=False, 极速模式启用）
- 随机种子：固定种子确保结果可重复

## Risk Assessment

### 高风险项

1. **功能一致性**: 优化可能意外改变核心行为
   - **缓解措施**: 全面的功能一致性测试，使用固定随机种子

2. **性能目标**: 100k 步/秒是激进目标，可能无法完全达到
   - **缓解措施**: 设定阶段性目标，逐步优化

3. **代码可维护性**: 过度优化可能降低代码可读性
   - **缓解措施**: 添加详细注释，保持代码结构清晰

### 中风险项

1. **内存使用**: 优化可能增加内存使用
   - **缓解措施**: 监控内存使用，确保不超过 10% 增长限制

2. **兼容性**: 优化可能影响与 PPO 训练系统的兼容性
   - **缓解措施**: 保持 API 接口不变，进行集成测试

## Next Steps

1. 运行性能分析，识别具体瓶颈
2. 实施 NumPy 向量化优化
3. 实施数据结构优化
4. 添加极速模式支持
5. 运行性能测试验证效果
6. 运行功能一致性测试确保正确性
