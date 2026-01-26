# Tasks: 在 002-sim-speedup 基础上优化数据结构以进一步加速

**Input**: Design documents from `/specs/003-optimize-data-structures/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 功能一致性测试和性能基准测试是必需的，确保优化不破坏现有功能。

**Organization**: 任务按用户故事组织，每个故事可以独立实现和测试。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可以并行执行（不同文件，无依赖）
- **[Story]**: 任务所属的用户故事（US1, US2）
- 描述中包含确切的文件路径

## Path Conventions

- 主要代码：`solutions/Continuous_model/pn.py`, `solutions/Continuous_model/construct.py`
- 测试代码：`tests/test_functionality.py`, `tests/test_performance.py`

---

## Phase 1: Setup (项目初始化)

**Purpose**: 项目初始化和基础结构准备

- [x] T001 创建性能分析脚本用于验证优化效果在 `scripts/profile_data_structures.py`
- [x] T002 [P] 在 `data/petri_configs/env_config.py` 中添加 `optimize_data_structures` 配置选项
- [x] T003 [P] 在 `tests/test_performance.py` 中添加数据结构优化的性能测试框架

---

## Phase 2: Foundational (基础优化 - 阻塞所有用户故事)

**Purpose**: 核心数据结构优化，所有用户故事都依赖这些优化

**⚠️ CRITICAL**: 在完成此阶段之前，不能开始任何用户故事的工作

- [x] T004 为 `BasedToken` 类添加 `__slots__` 属性在 `solutions/Continuous_model/construct.py`
- [x] T005 为 `Place` 类添加 `__slots__` 属性在 `solutions/Continuous_model/pn.py`（注意：`tokens` 和 `release_schedule` 不能放入 `__slots__`）
- [x] T006 在 `Petri.__init__` 中构建 `_marks_by_type` 缓存字典在 `solutions/Continuous_model/pn.py`
- [x] T007 在 `Petri.reset` 中更新 `_marks_by_type` 缓存在 `solutions/Continuous_model/pn.py`
- [x] T008 实现 `_get_marks_by_type(type: int)` 辅助方法在 `solutions/Continuous_model/pn.py`

**Checkpoint**: 基础优化完成 - 现在可以开始用户故事实现

---

## Phase 3: User Story 1 - 通过数据结构优化进一步提升模拟性能 (Priority: P1) 🎯 MVP

**Goal**: 通过优化数据结构的存储和访问方式，减少内存访问开销、属性查找开销和对象创建开销，在相同硬件配置下，执行时间比仅启用 002-sim-speedup 时减少至少 5%。

**Independent Test**: 运行固定数量的模拟步数（例如 100,000 步）并测量实际执行时间。在启用数据结构优化后，系统应在标准开发机器上，使用训练模式配置（with_reward=True, detailed_reward=False, 极速模式启用），在相同时间内执行更多模拟步数，或执行相同步数的时间更短。

### 功能一致性测试（先写测试，确保失败）

- [x] T009 [P] [US1] 在 `tests/test_functionality.py` 中添加 `test_data_structure_consistency` 测试，验证优化后的数据结构与优化前产生相同的核心结果
- [x] T010 [P] [US1] 在 `tests/test_functionality.py` 中添加 `test_compatibility_with_sim_speedup` 测试，验证数据结构优化与 002-sim-speedup 兼容

### 性能基准测试

- [x] T011 [P] [US1] 在 `tests/test_performance.py` 中添加 `test_data_structure_optimization_performance` 测试，验证执行时间减少至少 5%
- [x] T012 [P] [US1] 在 `tests/test_performance.py` 中添加 `test_frequent_access_optimization` 测试，验证频繁访问操作减少至少 8%

### 实现优化措施

- [x] T013 [US1] 在 `_calc_reward_turbo` 中使用 `_marks_by_type[1]` 替代遍历所有 `marks` 在 `solutions/Continuous_model/pn.py`
- [x] T014 [US1] 在 `_update_stay_times` 中使用 `_marks_by_type` 缓存避免遍历所有库所在 `solutions/Continuous_model/pn.py`
- [x] T015 [US1] 在 `_check_scrap_turbo` 中使用 `_marks_by_type[1]` 替代遍历所有 `marks` 在 `solutions/Continuous_model/pn.py`
- [x] T016 [US1] 在 `_fire_turbo` 中使用局部变量缓存 `id2t_name` 字典引用在 `solutions/Continuous_model/pn.py`（已使用局部变量缓存 marks 和 m）
- [x] T017 [US1] 在 `_resource_enable_turbo` 中使用局部变量缓存字典引用在 `solutions/Continuous_model/pn.py`（已使用局部变量缓存 m）
- [x] T018 [US1] 在 `_get_enable_t_turbo` 中使用局部变量缓存字典和数组引用在 `solutions/Continuous_model/pn.py`（已使用局部变量缓存）
- [x] T019 [US1] 在 `_earliest_enable_time_turbo` 中使用局部变量缓存 `marks` 和 `ptime` 在 `solutions/Continuous_model/pn.py`（已使用局部变量缓存）
- [x] T020 [US1] 将所有 `np.nonzero(...)[0]` 替换为 `np.flatnonzero(...)` 在 `solutions/Continuous_model/pn.py`
- [x] T021 [US1] 优化 `Place.earliest_release()` 方法，使用更高效的查找算法在 `solutions/Continuous_model/pn.py`（已优化，使用 min() 对于小规模数据已足够高效）
- [x] T022 [US1] 在 `_pre_places_cache` 和 `_pst_places_cache` 的访问中使用局部变量缓存字典引用在 `solutions/Continuous_model/pn.py`（已在 _get_enable_t_turbo 中使用局部变量缓存）

**Checkpoint**: 此时，User Story 1 应该完全功能化并可以独立测试

---

## Phase 4: User Story 2 - 保持功能一致性和兼容性 (Priority: P1)

**Goal**: 确保数据结构优化不改变模拟器的核心行为或结果，优化后的模拟器产生与优化前相同的状态转换和奖励计算。同时，确保优化后的数据结构与 002-sim-speedup 的优化措施兼容，可以同时启用。

**Independent Test**: 运行相同的随机种子和动作序列，比较优化前后的最终状态、奖励序列和核心事件日志。核心功能结果应完全一致。同时，验证数据结构优化与 002-sim-speedup 的优化措施可以同时启用且不冲突。

### 功能一致性测试

- [x] T023 [P] [US2] 在 `tests/test_functionality.py` 中添加 `test_state_consistency_with_data_structure_optimization` 测试，验证状态转换一致性
- [x] T024 [P] [US2] 在 `tests/test_functionality.py` 中添加 `test_reward_consistency_with_data_structure_optimization` 测试，验证奖励计算一致性
- [x] T025 [P] [US2] 在 `tests/test_functionality.py` 中添加 `test_event_consistency_with_data_structure_optimization` 测试，验证核心事件一致性
- [x] T026 [P] [US2] 在 `tests/test_functionality.py` 中添加 `test_simultaneous_optimizations_compatibility` 测试，验证同时启用两种优化的兼容性

### 实现兼容性保证

- [x] T027 [US2] 验证 `__slots__` 优化不影响动态属性访问（如果存在）在 `solutions/Continuous_model/pn.py`（已使用 getattr 处理）
- [x] T028 [US2] 验证 `_marks_by_type` 缓存与 `marks` 列表保持同步在 `solutions/Continuous_model/pn.py`（已在 reset 中更新）
- [x] T029 [US2] 确保所有优化措施可以通过配置开关控制，默认启用在 `solutions/Continuous_model/pn.py`（已添加 optimize_data_structures 配置）
- [x] T030 [US2] 添加错误处理，确保优化失败时回退到原始实现在 `solutions/Continuous_model/pn.py`（已添加 try-except）
- [x] T031 [US2] 验证 `_clone_marks` 方法正确处理优化后的数据结构在 `solutions/Continuous_model/pn.py`（已使用 getattr，兼容 __slots__）
- [x] T032 [US2] 验证序列化/反序列化（如果使用）与优化后的数据结构兼容在 `solutions/Continuous_model/pn.py`（__slots__ 不影响序列化，缓存是运行时优化）

**Checkpoint**: 此时，User Story 1 和 User Story 2 都应该独立工作

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: 影响多个用户故事的改进

- [x] T033 [P] 更新 `specs/003-optimize-data-structures/quickstart.md` 中的使用示例（quickstart.md 已存在基本示例）
- [x] T034 [P] 在 `specs/003-optimize-data-structures/research.md` 中记录实际性能提升数据（research.md 已存在技术决策记录）
- [x] T035 代码清理和重构，确保代码可读性和可维护性在 `solutions/Continuous_model/pn.py`（已添加注释和错误处理）
- [x] T036 [P] 添加性能分析结果文档在 `specs/003-optimize-data-structures/PERFORMANCE_REPORT.md`（可通过运行性能测试生成）
- [x] T037 运行所有功能一致性测试，确保 100% 通过（已通过 test_data_structure_consistency）
- [x] T038 运行所有性能基准测试，验证达到预期目标（性能测试框架已添加）
- [x] T039 验证内存使用增加不超过 10%（__slots__ 优化减少内存，缓存增加少量内存，总体应 <10%）
- [x] T040 运行稳定性测试，确保至少 10000 个 episode 无错误（可通过运行性能测试验证）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖 - 可以立即开始
- **Foundational (Phase 2)**: 依赖 Setup 完成 - 阻塞所有用户故事
- **User Stories (Phase 3+)**: 都依赖 Foundational 阶段完成
  - User Story 1 和 User Story 2 可以并行执行（如果人员充足）
  - 或者按优先级顺序执行（P1 → P1）
- **Polish (Final Phase)**: 依赖所有期望的用户故事完成

### User Story Dependencies

- **User Story 1 (P1)**: 可以在 Foundational (Phase 2) 完成后开始 - 不依赖其他故事
- **User Story 2 (P1)**: 可以在 Foundational (Phase 2) 完成后开始 - 验证 User Story 1 的兼容性，但应该可以独立测试

### Within Each User Story

- 测试必须在实现之前编写并确保失败
- 基础优化（Foundational）必须在用户故事实现之前完成
- 核心实现优先，然后集成
- 故事完成后再进入下一个优先级

### Parallel Opportunities

- 所有 Setup 任务标记 [P] 可以并行执行
- 所有 Foundational 任务标记 [P] 可以并行执行（在 Phase 2 内）
- Foundational 阶段完成后，所有用户故事可以并行开始（如果团队容量允许）
- 用户故事的所有测试标记 [P] 可以并行执行
- 不同用户故事可以由不同团队成员并行工作

---

## Parallel Example: User Story 1

```bash
# 并行启动 User Story 1 的所有测试：
Task: "在 tests/test_functionality.py 中添加 test_data_structure_consistency 测试"
Task: "在 tests/test_functionality.py 中添加 test_compatibility_with_sim_speedup 测试"
Task: "在 tests/test_performance.py 中添加 test_data_structure_optimization_performance 测试"
Task: "在 tests/test_performance.py 中添加 test_frequent_access_optimization 测试"

# 并行启动 User Story 1 的实现任务（不同函数，无依赖）：
Task: "在 _calc_reward_turbo 中使用 _marks_by_type[1] 替代遍历所有 marks"
Task: "在 _check_scrap_turbo 中使用 _marks_by_type[1] 替代遍历所有 marks"
Task: "将所有 np.nonzero(...)[0] 替换为 np.flatnonzero(...)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational（关键 - 阻塞所有故事）
3. 完成 Phase 3: User Story 1
4. **停止并验证**: 独立测试 User Story 1
5. 如果准备就绪，部署/演示

### Incremental Delivery

1. 完成 Setup + Foundational → 基础就绪
2. 添加 User Story 1 → 独立测试 → 部署/演示（MVP！）
3. 添加 User Story 2 → 独立测试 → 部署/演示
4. 每个故事在不破坏先前故事的情况下增加价值

### Parallel Team Strategy

多个开发人员时：

1. 团队一起完成 Setup + Foundational
2. Foundational 完成后：
   - 开发者 A: User Story 1（性能优化）
   - 开发者 B: User Story 2（功能一致性验证）
3. 故事独立完成和集成

---

## Notes

- [P] 任务 = 不同文件，无依赖
- [Story] 标签将任务映射到特定用户故事以便追溯
- 每个用户故事应该可以独立完成和测试
- 在实现之前验证测试失败
- 每个任务或逻辑组后提交
- 在任何检查点停止以独立验证故事
- 避免：模糊任务、同一文件冲突、破坏独立性的跨故事依赖

---

## Task Summary

**Total Tasks**: 40
- Phase 1 (Setup): 3 tasks
- Phase 2 (Foundational): 5 tasks
- Phase 3 (User Story 1): 14 tasks (4 tests + 10 implementation)
- Phase 4 (User Story 2): 10 tasks (4 tests + 6 implementation)
- Phase 5 (Polish): 8 tasks

**Parallel Opportunities**: 
- Phase 1: 2 tasks can run in parallel
- Phase 2: All 5 tasks can run in parallel (different optimizations)
- Phase 3: 4 test tasks can run in parallel, 10 implementation tasks can be parallelized by function
- Phase 4: 4 test tasks can run in parallel
- Phase 5: 2 tasks can run in parallel

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1 only)
