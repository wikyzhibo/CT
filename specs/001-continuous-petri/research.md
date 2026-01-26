# Research: 连续 Petri 网晶圆加工仿真系统

**Date**: 2026-01-25  
**Feature**: 001-continuous-petri  
**Status**: Complete

## Overview

本文档记录了连续 Petri 网晶圆加工仿真系统的技术决策和研究结果。该系统是一个基于 Petri 网理论的离散事件仿真系统，用于模拟半导体制造中的晶圆加工流程。

## Technology Decisions

### Decision 1: Python 作为实现语言

**Decision**: 使用 Python 3.x 实现系统

**Rationale**:
- Python 在科学计算和仿真领域有丰富的生态系统（numpy, scipy）
- 代码可读性强，便于维护和扩展
- 与强化学习框架（如 PPO）集成方便
- 已有代码库使用 Python，保持技术栈一致性

**Alternatives considered**:
- C++: 性能更好，但开发效率低，与 RL 框架集成复杂
- Java: 企业级应用常用，但科学计算库不如 Python 丰富
- MATLAB: 仿真领域常用，但商业许可和部署复杂

### Decision 2: NumPy 用于矩阵运算

**Decision**: 使用 NumPy 进行 Petri 网的矩阵表示和运算

**Rationale**:
- Petri 网的前置矩阵（pre）和后置矩阵（pst）天然适合矩阵表示
- NumPy 提供高效的矩阵运算，支持大规模 Petri 网
- 与 Python 科学计算生态完美集成

**Alternatives considered**:
- 纯 Python 列表：性能差，不适合大规模网络
- SciPy: 功能更全面但更重，当前需求不需要

### Decision 3: Dataclasses 用于数据结构

**Decision**: 使用 Python dataclasses 定义 Place, Token, ModuleSpec 等数据结构

**Rationale**:
- 代码简洁，自动生成 `__init__`, `__repr__` 等方法
- 类型提示支持良好，提高代码可维护性
- 性能开销小，适合频繁创建的对象

**Alternatives considered**:
- 普通类：需要手动编写更多样板代码
- NamedTuple: 不可变，不适合需要修改状态的对象
- Pydantic: 功能强大但更重，当前需求不需要验证功能

### Decision 4: Deque 用于队列管理

**Decision**: 使用 collections.deque 管理库所中的 token 队列

**Rationale**:
- FIFO 队列操作（popleft, append）时间复杂度 O(1)
- 适合 Petri 网中 token 的先进先出特性
- 标准库实现，无需额外依赖

**Alternatives considered**:
- list: pop(0) 操作是 O(n)，性能差
- queue.Queue: 线程安全但不需要，增加开销

### Decision 5: 配置驱动的架构

**Decision**: 使用 PetriEnvConfig 类管理所有配置参数

**Rationale**:
- 集中管理配置，便于调整和实验
- 支持 JSON 序列化，便于保存和加载配置
- 根据训练阶段自动调整奖励配置，提高灵活性

**Alternatives considered**:
- 硬编码参数：难以调整，不利于实验
- 环境变量：适合部署，但不适合复杂的配置结构

## Design Patterns

### Pattern 1: Builder Pattern (SuperPetriBuilder)

**Decision**: 使用 Builder 模式构建 Petri 网

**Rationale**:
- 复杂的 Petri 网构建过程需要多步骤
- 分离构建逻辑和最终对象，提高可维护性
- 支持灵活的模块、机器人、路线配置

### Pattern 2: State Machine (Petri Net Transitions)

**Decision**: 使用 Petri 网变迁表示状态转换

**Rationale**:
- Petri 网理论天然支持状态机建模
- 变迁的使能条件和执行逻辑清晰
- 支持并发和资源约束建模

### Pattern 3: Observer Pattern (Statistics Tracking)

**Decision**: 在变迁执行时追踪晶圆统计信息

**Rationale**:
- 不侵入核心逻辑，保持关注点分离
- 便于扩展新的统计指标
- 性能开销小

## Performance Considerations

### Time Complexity

- **变迁使能检查**: O(P × T)，其中 P 是库所数，T 是变迁数
- **奖励计算**: O(P × W)，其中 W 是每个库所的平均 token 数
- **释放时间追踪**: O(C)，其中 C 是链长度（通常很小）

### Space Complexity

- **Petri 网矩阵**: O(P × T)
- **Token 存储**: O(P × W)
- **统计信息**: O(N)，其中 N 是晶圆数量

### Optimization Opportunities

1. **矩阵运算**: 使用 NumPy 的向量化操作
2. **队列管理**: 使用 deque 的 O(1) 操作
3. **缓存**: 释放时间计算可以缓存结果

## Integration Points

### 1. 强化学习框架集成

系统设计为强化学习环境，需要实现标准的 gym/stable-baselines3 接口：
- `reset()`: 重置环境
- `step(action)`: 执行动作并返回奖励
- `observation_space`: 定义观察空间
- `action_space`: 定义动作空间

### 2. 配置系统集成

与 `data/petri_configs/env_config.py` 集成：
- 支持从 JSON 文件加载配置
- 支持训练阶段自动调整
- 支持奖励开关配置

### 3. 可视化集成

与 `visualization/plot.py` 集成：
- 生成甘特图展示加工时间线
- 支持多机器、多晶圆的可视化

## Best Practices Applied

1. **类型提示**: 所有函数和类都使用类型提示，提高代码可读性
2. **文档字符串**: 关键函数都有详细的 docstring
3. **模块化设计**: 功能分离到不同模块（pn.py, construct.py, env_config.py）
4. **配置驱动**: 通过配置类管理参数，避免硬编码
5. **错误处理**: 关键操作都有边界检查（如容量检查、时间检查）

## Unresolved Questions

无。所有技术决策都已基于现有实现确定。

## References

- Petri 网理论: 离散事件系统的标准建模方法
- NumPy 文档: https://numpy.org/doc/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- 强化学习环境接口: OpenAI Gym / Stable Baselines3
