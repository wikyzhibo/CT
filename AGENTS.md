
# Agent Rules (Project Assistant)

## Language
- Always respond in Simplified Chinese (简体中文).

## Core Workflow (Definition of Done)
当用户请求新增/修改/删除任何功能时，必须同时完成：
1) 功能实现（代码/配置/逻辑变更）
2) 文档更新（反映本次变更）
只有两者都完成，才算任务完成；不得只交付代码不更新文档。

### What counts as "functionality change"
包含但不限于：
- 新功能、功能行为变化、参数/默认值变化
- API/接口输入输出变化
- 训练/推理流程变化、reward/环境逻辑变化
- 配置项新增/修改/删除
- 性能优化导致的行为差异、边界条件变化

## Documentation Policy (Must Update)
### Where to update docs
按影响范围选择更新位置（可多处）：
- **README.md**：对外使用方式、快速开始、核心行为变化
- **docs/**：较完整的功能说明、设计说明、参数解释、FAQ
- **CHANGELOG.md**（如存在）：版本/变更记录（面向发布）
- **代码注释/docstring**：仅限局部逻辑、接口细节、关键约束

### Minimum doc update requirements
每次功能变更后，文档至少包含：
- **What changed**：变更点一句话总结
- **Why**：变更原因（修复/需求/性能/一致性）
- **How to use / Impact**：使用方式或影响面（是否破坏兼容、参数是否变化）
- **Examples**（如适用）：新增/变化的示例命令、配置片段、输入输出样例

### If docs are missing or unclear
- 若找不到对应文档位置，必须先询问或给出你认为最合理的放置位置建议（例如新增 `docs/<feature>.md`）。
- 若仓库没有 docs 目录，可建议创建 `docs/` 并给出初始结构。

## Output Format (Always)
在完成任务时，输出必须包含以下小节（除非用户明确不需要）：

1. **变更摘要**
2. **代码改动**（关键点/文件/接口变化）
3. **文档改动**（具体更新了哪些文件、更新内容要点）
4. **验证方式**（如何运行/如何测试/如何回归）
5. **风险与回滚**（可选，但建议提供）

## Git Discipline (Commit / Branch)
### Branch naming
- 功能：`feat/<short-desc>`
- 修复：`fix/<short-desc>`
- 重构：`refactor/<short-desc>`
- 实验：`exp/<short-desc>`
- 保护 main 快照：`backup/main-before-merge` 或 `backup/main-YYYY-MM-DD`

### Commit message (Conventional Commits)
使用：
- `feat(scope): ...`
- `fix(scope): ...`
- `refactor(scope): ...`
- `docs(scope): ...`
- `chore(scope): ...`
- `perf(scope): ...`

**要求：** 若功能变更影响文档，commit 中应体现 docs 更新：
- 同一 commit 同时包含代码与文档，或
- 代码 commit 后紧跟一个 `docs:` commit（不推荐拖太久）

## Safety / Collaboration
- 任何可能需要 `git push --force` 的操作，必须先提示风险与影响范围，并给出更安全替代方案（如新建分支、revert）。
- 不确定需求/边界时，先列出假设并请用户确认关键点（但不要阻塞交付；能做的先做）。

## Quick Checklist (Before Final Answer)
- [ ] 功能是否完成并解释清楚
- [ ] 文档是否已更新（或给出要更新的具体内容与位置）
- [ ] 是否提供了验证方式
- [ ] 是否说明了影响/兼容性/风险

