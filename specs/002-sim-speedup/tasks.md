# Tasks: 加速 Petri 网模拟器性能

**Input**: Design documents from `/specs/002-sim-speedup/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 性能测试和功能一致性测试是必需的，用于验证优化效果

**Organization**: 任务按用户故事组织，每个故事可独立实现和测试

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可以并行执行（不同文件，无依赖）
- **[Story]**: 任务所属的用户故事（US1, US2, US3）
- 描述中包含确切文件路径

## Path Conventions

- **单项目结构**: `solutions/Continuous_model/`, `data/petri_configs/`, `tests/`
- 路径基于现有项目结构

---

## Phase 1: Setup (共享基础设施)

**Purpose**: 项目初始化和性能分析

- [x] T001 运行性能分析工具（cProfile）识别当前性能瓶颈，分析 `solutions/Continuous_model/pn.py` 中的热点函数
- [x] T002 [P] 创建性能测试目录结构 `tests/` 用于性能基准测试
- [x] T003 [P] 创建功能一致性测试框架 `tests/test_functionality.py` 用于验证优化后功能一致性

---

## Phase 2: Foundational (阻塞性先决条件)

**Purpose**: 核心基础设施，必须在任何用户故事实现前完成

**⚠️ CRITICAL**: 所有用户故事工作必须在此阶段完成后才能开始

- [x] T004 在 `data/petri_configs/env_config.py` 中扩展 `PetriEnvConfig` 类，添加性能优化配置字段（turbo_mode, optimize_reward_calc, optimize_enable_check, optimize_state_update, cache_indices）
- [x] T005 在 `solutions/Continuous_model/pn.py` 的 `Petri.__init__()` 方法中加载性能优化配置，初始化优化标志
- [x] T006 在 `solutions/Continuous_model/pn.py` 中实现 `_build_cached_indices()` 方法，构建缓存的库所和变迁索引映射（如果 cache_indices=True）

**Checkpoint**: 基础设施就绪 - 用户故事实现现在可以开始并行进行

---

## Phase 3: User Story 1 - 提高模拟吞吐量 (Priority: P1) 🎯 MVP

**Goal**: 实现极速模式，在标准开发机器上达到 1 秒内执行超过 100,000 个模拟步数的目标

**Independent Test**: 运行固定数量的模拟步数（100,000 步）并测量执行时间。在极速模式下，系统应在 1 秒内完成至少 100,000 个模拟步数。

### Tests for User Story 1

> **NOTE: 先编写这些测试，确保它们在实现前失败**

- [x] T007 [P] [US1] 创建性能基准测试 `tests/test_performance.py`，测试极速模式性能（100k步/秒）
- [x] T008 [P] [US1] 在 `tests/test_performance.py` 中添加常规优化性能测试（1000步减少20%）

### Implementation for User Story 1

- [ ] T009 [US1] 在 `solutions/Continuous_model/pn.py` 中实现极速模式条件检查，在 `_track_wafer_statistics()` 方法中添加 `turbo_mode` 判断，跳过详细统计追踪
- [ ] T010 [US1] 在 `solutions/Continuous_model/pn.py` 中优化 `calc_reward()` 函数，使用 NumPy 向量化操作替代 Python 循环，减少不必要的计算
- [ ] T011 [US1] 在 `solutions/Continuous_model/pn.py` 中优化 `_resource_enable()` 方法，使用 NumPy 向量化操作检查使能条件，减少计算开销
- [ ] T012 [US1] 在 `solutions/Continuous_model/pn.py` 中优化 `get_enable_t()` 方法，利用缓存的索引和优化的使能检查
- [ ] T013 [US1] 在 `solutions/Continuous_model/pn.py` 中优化 `_fire()` 方法的状态更新逻辑，减少每次变迁执行时的开销
- [ ] T014 [US1] 在 `solutions/Continuous_model/pn.py` 中优化 `_update_stay_times()` 方法，使用批量更新减少函数调用开销
- [ ] T015 [US1] 在 `solutions/Continuous_model/pn.py` 的 `calc_reward()` 中根据 `reward_config` 提前退出不需要的计算分支，实现条件短路优化

**Checkpoint**: 此时，User Story 1 应该完全功能化并可独立测试

---

## Phase 4: User Story 2 - 保持模拟准确性 (Priority: P1)

**Goal**: 确保性能优化后核心功能（状态转换、奖励计算、报废检测）保持完全一致

**Independent Test**: 使用相同随机种子和动作序列，比较优化前后的最终状态、奖励序列和核心事件日志。核心功能结果应完全一致。

### Tests for User Story 2

- [ ] T016 [P] [US2] 在 `tests/test_functionality.py` 中创建状态一致性测试，验证优化前后状态转换一致
- [ ] T017 [P] [US2] 在 `tests/test_functionality.py` 中创建奖励一致性测试，验证优化前后奖励计算一致
- [ ] T018 [P] [US2] 在 `tests/test_functionality.py` 中创建事件一致性测试，验证优化前后核心事件日志一致

### Implementation for User Story 2

- [ ] T019 [US2] 在 `solutions/Continuous_model/pn.py` 中验证 `calc_reward()` 的向量化实现与原始实现产生相同的数值结果
- [ ] T020 [US2] 在 `solutions/Continuous_model/pn.py` 中确保 `_fire()` 方法的状态更新逻辑与优化前完全一致
- [ ] T021 [US2] 在 `solutions/Continuous_model/pn.py` 中确保释放时间链式更新逻辑（`_chain_record_release`, `_chain_update_release`）保持不变，功能完全保留
- [ ] T022 [US2] 在 `solutions/Continuous_model/pn.py` 中确保报废检测逻辑（`_check_scrap()`）不受优化影响
- [ ] T023 [US2] 运行功能一致性测试套件，验证所有核心功能在优化后保持一致

**Checkpoint**: 此时，User Stories 1 和 2 都应该独立工作

---

## Phase 5: User Story 3 - 支持可配置的性能优化和极速模式 (Priority: P1)

**Goal**: 实现可配置的性能优化开关，允许用户选择性地启用/禁用优化措施

**Independent Test**: 通过配置开关启用/禁用优化和极速模式，验证核心功能一致性。禁用优化时应恢复到原始行为。

### Tests for User Story 3

- [ ] T024 [P] [US3] 在 `tests/test_functionality.py` 中创建配置测试，验证禁用所有优化时行为与未优化版本一致
- [ ] T025 [P] [US3] 在 `tests/test_performance.py` 中创建配置性能测试，验证不同配置组合的性能效果

### Implementation for User Story 3

- [ ] T026 [US3] 在 `solutions/Continuous_model/pn.py` 中实现配置驱动的优化开关，根据 `optimize_reward_calc` 标志选择使用向量化或原始奖励计算
- [ ] T027 [US3] 在 `solutions/Continuous_model/pn.py` 中实现配置驱动的使能检查优化，根据 `optimize_enable_check` 标志选择优化或原始实现
- [ ] T028 [US3] 在 `solutions/Continuous_model/pn.py` 中实现配置驱动的状态更新优化，根据 `optimize_state_update` 标志选择优化或原始实现
- [ ] T029 [US3] 在 `solutions/Continuous_model/pn.py` 中实现配置驱动的索引缓存，根据 `cache_indices` 标志决定是否构建和使用缓存索引
- [ ] T030 [US3] 在 `solutions/Continuous_model/pn.py` 的 `reset()` 方法中确保配置驱动的优化标志正确重置
- [ ] T031 [US3] 更新 `data/petri_configs/env_config.py` 中的配置文档，说明各优化开关的用途和默认值

**Checkpoint**: 所有用户故事现在应该独立功能化

---

## Phase 6: 优化释放时间追踪 (支持所有用户故事)

**Purpose**: 优化释放时间链式更新逻辑，但必须保留功能（链式冲突检测很重要）

- [ ] T032 在 `solutions/Continuous_model/pn.py` 中优化 `_chain_record_release()` 方法，减少链式更新的计算开销，但保持功能完全一致
- [ ] T033 在 `solutions/Continuous_model/pn.py` 中优化 `_chain_update_release()` 方法，使用预计算的链路映射提升性能
- [ ] T034 在 `solutions/Continuous_model/pn.py` 中优化释放时间队列的数据结构访问，减少查找开销

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 影响多个用户故事的改进

- [ ] T035 [P] 更新 `solutions/Continuous_model/test_env.py` 以支持极速模式配置（如果需要）
- [ ] T036 [P] 验证与强化学习训练系统（`solutions/PPO`）的兼容性，确保优化不影响训练流程
- [ ] T037 [P] 运行 `specs/002-sim-speedup/quickstart.md` 中的示例代码，验证快速开始指南的准确性
- [ ] T038 添加代码注释，说明性能优化的实现细节和配置选项
- [ ] T039 运行完整的功能一致性测试套件，确保所有核心功能在优化后保持一致
- [ ] T040 运行完整的性能测试套件，验证所有性能目标（100k步/秒、20%改进等）都已达到
- [ ] T041 监控内存使用，确保优化后内存使用增加不超过 10%
- [ ] T042 运行稳定性测试，验证在启用极速模式的情况下，模拟器能稳定运行至少 10000 个 episode

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖 - 可立即开始
- **Foundational (Phase 2)**: 依赖 Setup 完成 - 阻塞所有用户故事
- **User Stories (Phase 3-5)**: 都依赖 Foundational 阶段完成
  - 用户故事可以并行进行（如果有足够人员）
  - 或按优先级顺序进行（US1 → US2 → US3）
- **优化释放时间追踪 (Phase 6)**: 可以在用户故事完成后进行，支持所有故事
- **Polish (Phase 7)**: 依赖所有期望的用户故事完成

### User Story Dependencies

- **User Story 1 (P1)**: 可在 Foundational (Phase 2) 后开始 - 不依赖其他故事
- **User Story 2 (P1)**: 可在 Foundational (Phase 2) 后开始 - 验证 US1 的优化，但应独立可测试
- **User Story 3 (P1)**: 可在 Foundational (Phase 2) 后开始 - 为 US1 和 US2 提供配置支持，但应独立可测试

### Within Each User Story

- 测试必须在实现前编写并确保失败
- 核心实现优先于集成
- 故事完成后再进行下一个优先级

### Parallel Opportunities

- 所有 Setup 任务标记 [P] 的可以并行运行
- 所有 Foundational 任务标记 [P] 的可以并行运行（在 Phase 2 内）
- Foundational 阶段完成后，所有用户故事可以并行开始（如果团队容量允许）
- 用户故事的所有测试标记 [P] 的可以并行运行
- 不同用户故事可以由不同团队成员并行工作

---

## Parallel Example: User Story 1

```bash
# 并行启动 User Story 1 的所有测试：
Task: "创建性能基准测试 tests/test_performance.py"
Task: "在 tests/test_performance.py 中添加常规优化性能测试"

# 优化实现可以按顺序进行（因为都在同一个文件中）：
Task: "实现极速模式条件检查"
Task: "优化 calc_reward() 函数"
Task: "优化 _resource_enable() 方法"
```

---

## Parallel Example: User Story 2

```bash
# 并行启动 User Story 2 的所有测试：
Task: "创建状态一致性测试"
Task: "创建奖励一致性测试"
Task: "创建事件一致性测试"

# 验证实现可以并行进行（不同函数）：
Task: "验证 calc_reward() 的向量化实现"
Task: "确保 _fire() 方法的状态更新逻辑一致"
Task: "确保释放时间链式更新逻辑保持不变"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational (CRITICAL - 阻塞所有故事)
3. 完成 Phase 3: User Story 1
4. **停止并验证**: 独立测试 User Story 1
5. 如果就绪，部署/演示

### Incremental Delivery

1. 完成 Setup + Foundational → 基础设施就绪
2. 添加 User Story 1 → 独立测试 → 部署/演示 (MVP!)
3. 添加 User Story 2 → 独立测试 → 部署/演示
4. 添加 User Story 3 → 独立测试 → 部署/演示
5. 每个故事在不破坏先前故事的情况下增加价值

### Parallel Team Strategy

多个开发人员时：

1. 团队一起完成 Setup + Foundational
2. Foundational 完成后：
   - 开发者 A: User Story 1（性能优化实现）
   - 开发者 B: User Story 2（功能一致性验证）
   - 开发者 C: User Story 3（配置系统实现）
3. 故事独立完成和集成

---

## Notes

- [P] 任务 = 不同文件，无依赖
- [Story] 标签将任务映射到特定用户故事以便追溯
- 每个用户故事应该独立完成和可测试
- 实现前验证测试失败
- 每个任务或逻辑组后提交
- 在任何检查点停止以独立验证故事
- 避免：模糊任务、同一文件冲突、破坏独立性的跨故事依赖

---

## Task Summary

- **Total Tasks**: 42
- **Phase 1 (Setup)**: 3 tasks
- **Phase 2 (Foundational)**: 3 tasks
- **Phase 3 (US1)**: 9 tasks (2 tests + 7 implementation)
- **Phase 4 (US2)**: 8 tasks (3 tests + 5 implementation)
- **Phase 5 (US3)**: 8 tasks (2 tests + 6 implementation)
- **Phase 6 (Release Chain Optimization)**: 3 tasks
- **Phase 7 (Polish)**: 8 tasks

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) = 15 tasks

**Parallel Opportunities**: 
- Setup tasks can run in parallel
- Test tasks within each story can run in parallel
- User stories can be worked on in parallel after Foundational phase
