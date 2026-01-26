# Implementation Plan: 连续 Petri 网晶圆加工仿真系统

**Branch**: `001-continuous-petri` | **Date**: 2026-01-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-continuous-petri/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

本系统实现了一个基于 Petri 网理论的连续时间晶圆加工仿真环境，用于强化学习训练。系统支持完整的晶圆加工流程（LP → s1 → s2 → s3 → s4 → s5 → LP_done），包括双机械手协作、多机器腔室、释放时间追踪、奖励/惩罚机制等核心功能。技术栈采用 Python + NumPy，使用 dataclasses 和 deque 进行高效的数据管理。

## Technical Context

**Language/Version**: Python 3.x  
**Primary Dependencies**: numpy, dataclasses (标准库), collections (标准库), typing (标准库)  
**Storage**: N/A (内存中运行，无持久化存储)  
**Testing**: pytest (推荐，当前代码库未强制要求)  
**Target Platform**: Linux/Windows/macOS (Python 跨平台)  
**Project Type**: single (单项目结构)  
**Performance Goals**: 
- 支持至少 10 个晶圆的完整加工流程仿真
- 单步执行（step）延迟 < 10ms
- 奖励计算时间 < 5ms
- 内存占用 < 100MB（10 个晶圆场景）

**Constraints**: 
- 实时计算奖励，不能有显著延迟
- 支持强化学习训练的高频调用（每秒数千次 step）
- 内存占用需合理，支持批量训练
- 必须支持重置到初始状态

**Scale/Scope**: 
- 单个仿真环境实例
- 支持 4-20 个晶圆同时加工
- 5 个加工腔室（s1, s2, s3, s4, s5）
- 2 个机械手（TM2, TM3）
- 支持强化学习训练循环

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: PASSED

由于项目 constitution 文件是模板且未定义具体约束，本系统遵循以下通用原则：

1. **代码质量**: 使用类型提示和文档字符串，提高代码可维护性
2. **模块化设计**: 功能分离到不同模块（pn.py, construct.py, env_config.py）
3. **配置驱动**: 通过配置类管理参数，避免硬编码
4. **性能优化**: 使用 NumPy 向量化操作和 deque 高效队列

无违反项需要记录。

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
solutions/Continuous_model/
├── pn.py                    # 核心 Petri 网类（Place, Petri）
├── construct.py             # Petri 网构建器（SuperPetriBuilder）
├── run_supernet.py          # 运行脚本
├── test_env.py              # 测试环境
└── reward_config_examples.py # 奖励配置示例

data/petri_configs/
├── env_config.py            # 环境配置类（PetriEnvConfig）
├── default_config.json      # 默认配置
├── phase1_config.json       # 阶段1配置
└── phase2_config.json       # 阶段2配置

visualization/
└── plot.py                  # 甘特图可视化

utils/
└── draw_gantt.py            # 甘特图绘制工具
```

**Structure Decision**: 采用单项目结构，代码组织在 `solutions/Continuous_model/` 目录下。配置管理在 `data/petri_configs/`，可视化功能在 `visualization/` 和 `utils/`。这种结构清晰分离了核心逻辑、配置和工具代码。

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
