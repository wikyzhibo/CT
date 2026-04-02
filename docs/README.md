# Docs Index

## Abstract
- What: 本文档是 CT 项目的唯一文档导航入口。
- When: 查找任何架构、实现、训练、可视化、Td_petri 说明时先读本页。
- Not: 本页不承载实现细节；细节以主题主文档为准。
- Key rules:
  - 权威规范仅维护在 5 个主题主文档中。
  - 旧文件名只作为兼容跳转入口。
  - 新增规范文档必须同步更新本索引。

## Canonical Docs
- [`overview/project-context.md`](./overview/project-context.md)：项目描述、系统边界、模块地图与统一入口命令。
- [`continuous-model/pn-single.md`](./continuous-model/pn-single.md)：pn_single 架构、接口、执行链与关键约束。
- [`visualization/ui-guide.md`](./visualization/ui-guide.md)：可视化界面参数、模型加载与回放数据契约。
- [`training/training-guide.md`](./training/training-guide.md)：single/cascade/concurrent 训练入口、配置优先级与产物位置。
- [`td-petri/td-petri-guide.md`](./td-petri/td-petri-guide.md)：Td_petri 边界、入口脚本与序列契约。

## Appendices
- [`pn_api.md`](./pn_api.md)：`pn.py` API 说明（补充参考）。
- [`pn_design.md`](./pn_design.md)：`pn.py` 设计说明（补充参考）。
- [`pdr-search.md`](./pdr-search.md)：PDR 构网命名与 DFS 搜索约束（补充参考）。
- [`routes.md`](./routes.md)：调度技术路线与约束说明（历史与背景）。
- [`rl-design.md`](./rl-design.md)：强化学习奖励设计与状态示例（补充参考）。
- [`modeling.md`](./modeling.md)：Petri 网建模基础说明。
- [`gantt.md`](./gantt.md)：甘特图绘制工具规范。
- [`problem_description.md`](./problem_description.md)：问题定义规范入口。
- [`cloud-gpu-training-review.md`](./cloud-gpu-training-review.md)：云端 GPU 训练审查记录。
- [`可能增加的obs_features.md`](./可能增加的obs_features.md)：候选观测特征扩展方案。

## Deprecated / Compatibility
- [`deprecated/README.md`](./deprecated/README.md)：旧文档迁移映射与兼容策略。
- [`project.md`](./project.md)：兼容跳转页。
- [`架构.md`](./架构.md)：兼容跳转页。
- [`continuous_solution_design.md`](./continuous_solution_design.md)：兼容跳转页。
- [`viz.md`](./viz.md)：兼容跳转页。
- [`td_petri.md`](./td_petri.md)：兼容跳转页。
- [`td_petri_modeling.md`](./td_petri_modeling.md)：兼容跳转页。
- [`env_place_obs.md`](./env_place_obs.md)：历史接口说明（已移除功能）。
- [`problem_discription.md`](./problem_discription.md)：旧拼写文件兼容入口。
- [`Petri animate tool.md`](./Petri%20animate%20tool.md)：旧工具说明兼容入口。

## Writing Rules
1. 所有主文档必须包含固定章节：`Abstract / Scope / Architecture or Data Flow / Interfaces / Behavior Rules / Examples / Edge Cases / Related Docs / Change Notes`。
2. 新增或修改 CLI 参数后，必须同步更新对应主题主文档和本索引。
3. 若移除接口，必须在兼容页写明“新路径 + 迁移日期 + 差异说明”。
4. 输出产物路径必须遵守仓库统一规范：`results/action_sequences`、`results/gantt`、`results/training_logs`、`results/topology_cache`、`results/models`。

## Change Notes
- 2026-04-02: `CHANGELOG.md` / `training/training-guide.md` / `continuous-model/pn-single.md` / `overview/project-context.md` / `gantt.md`：`export_inference_sequence` CLI 收敛（`--concurrent`、`--retry` 等）；输出文件名为 `<out_name>(W<n_wafer>-M<time>).json`；见 `CHANGELOG.md`。
- 2026-03-31: `training/training-guide.md` / `CHANGELOG.md`：`ppo_trainer` 并发训练结束导出动作序列与甘特（`device_mode=concurrent`）；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md` / `gantt.md` / `training/training-guide.md` / `CHANGELOG.md`：`ClusterTool.render_gantt` 基于 `fire_log` 输出腔室甘特 PNG；`visualization/plot.py` 修正 `title_suffix`/`policy` 空值；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md` / `continuous-model/pn-single.md` / `CHANGELOG.md`：删除 `ClusterTool._check_scrap`，驻留 scrap 判定内联至 `_advance_and_compute_reward`；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md` / `continuous-model/pn-single.md` / `CHANGELOG.md`：`ClusterTool.step` 仅消费 `_advance_and_compute_reward` 的 `scan_info`，不再对 scrap 做外层 `_check_scrap` 兜底；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md` / `CHANGELOG.md`：A 方案 `ClusterTool.step` 第二返回值仅为标量 `float`；移除 `detailed_reward` 与奖励分项 `dict`；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md` / `training/training-guide.md`：`ClusterTool(config, concurrent=...)`；`get_action_mask` 并发时返回 TM2/TM3 掩码；移除 `get_enable_t`；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md`：`solutions/A/petri_net.py`（`ClusterTool`）移除 step 分段 profiling 与 `get_step_profile_summary`；`ppo_trainer` 不再打印 `[Step Time Profile]`；见 `CHANGELOG.md`。
- 2026-03-31: `pn_api.md`：`solutions/A/petri_net.py`（`ClusterTool`）移除 `_token_stats` / `calc_wafer_statistics`；可视化 `StateInfo.stats` 时长类为占位；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：移除 `lp_release_pattern` 构网链与 `route_meta.lp_release_pattern_types`；行为规则 32–33；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：`n_wafer` 替换为 `n_wafer1`/`n_wafer2`，移除构网 `wafer_type_alloc_by_type`；行为规则 33；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：`max_wafers_in_system` 替换为 `max_wafers1_in_system`/`max_wafers2_in_system`；行为规则 32；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：`build_takt_payload` 清洗口径收敛到 `chamber_blocks`；行为规则 34；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：`preprocess_chamber_runtime_blocks` 顺序与清洗严格缺省、移除默认清洗形参；`cascade_routes_1_star.json` chambers 补全；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：`preprocess_chamber_runtime_blocks` 移除 `process_time_map` 形参；行为规则与 Change Notes 同步；见 `CHANGELOG.md`。
- 2026-03-31: `continuous-model/pn-single.md`：并行并列目标 tie-break 从随机改为确定性顺序打破（并列按候选顺序选首个最小项）；见 `CHANGELOG.md`。
- 2026-03-30: `continuous-model/pn-single.md`：A 方案 `ClusterTool/PetriEnvConfig` 初始化字段收敛（移除 `stop_on_scrap`、`T_transport`、`T_load`、`single_robot_capacity`、`device_mode`、`cleaning_trigger_wafers`、`cleaning_duration` 等配置/运行时冗余口径；保留 `cleaning_enabled`；`single_route_config` 必填；清洗参数运行时统一 map 入口）。
- 2026-03-30: 移除配置/运行时 `route_code`、`route4_takt_interval` 与 `build_fixed_takt_result`；路线统一 `single_route_name` + `single_route_config`；见 `continuous-model/pn-single.md`、`visualization/ui-guide.md`、`CHANGELOG.md`。
- 2026-03-30: `continuous-model/pn-single.md`：级联并行选机由 robin 改为 `use_count` 最小优先（并列随机），并移除 robin 指针相关实现；详见该文档 Change Notes 与 `CHANGELOG.md`。
- 2026-03-30: `continuous-model/pn-single.md`：`ClusterTool` 装载口 `u_LP*` 掩码改为按 `LP1`/`LP2` 独立使能；移除 `_allow_start` / `_pending_lp_release_type` 链；详见该文档 Change Notes。
- 2026-03-30: `continuous-model/pn-single.md`：级联固定拓扑由单 `LP` 改为 `LP1`/`LP2`；`build_net`/`ClusterTool` 与 TM2 观测维度（9 维）同步说明见该文档 Change Notes。
- 2026-03-29: 可视化主文档新增并发双动作入口与回放约束：`visualization/ui-guide.md` 现覆盖 `--concurrent`、并发权重自动识别、并发序列自动切换 runtime。
- 2026-03-19: 重构为“1 个入口 + 5 个主文档 + 附录 + deprecated 兼容层”。
