# PN Single Guide

## Abstract
- What: 本文档定义级联设备连续时间 Petri 网（pn_single）在当前仓库中的架构、接口与行为约束（cascade-only）。
- When: 修改 `pn_single.py`、`env_single.py`、`train_single.py`、导出/追责脚本前必须先读。
- Not: 不覆盖并发双机械手 `pn.py` 的完整实现细节。
- Key rules:
  - 统一入口是 `Env_PN_Single`，且仅支持 `device_mode=cascade`。
  - 构网采用固定全拓扑 + 动态 route 装配。
  - 关键执行链固定为 `构网 -> mask -> step -> reward`。

## Scope
- In:
  - `ClusterTool`（`solutions/Continuous_model/pn_single.py`）职责与执行链。
  - `Env_PN_Single`（`solutions/Continuous_model/env_single.py`）接口。
  - 单设备相关脚本：训练、导出、二次释放惩罚验证。
- Out:
  - 并发模型 `pn.py` 的细节。
  - 可视化 UI 实现细节。

## Architecture or Data Flow
1. `construct_single.py` 的 `build_net` 仅基于 `route_config(+route_name)` 构建网络结构与元数据；晶圆 `route_queue` / `t_route_code_map` / `token_route_plan` 由 `build_route_queue.build_token_route_queue` 生成。
2. `ClusterTool` 维护标识、使能判定、时间推进、reward 计算、违规统计。
3. `Env_PN_Single` 封装 TorchRL 风格 `reset/step`，并暴露固定动作空间下的 `action_mask`。
4. 训练脚本 `train_single.py` 调用 `collect_rollout_ultra` 执行 CPU rollout + batched PPO update。
5. 导出脚本 `export_inference_sequence.py` 生成 `results/action_sequences/<out_name>.json`（默认 `out_name=tmp` 即 `results/action_sequences/tmp.json`；含 `sequence`、`replay_env_overrides`、`reward_report` 等）。

## Interfaces
- 环境接口:
  - 类: `solutions.Continuous_model.env_single.Env_PN_Single`
  - 关键参数: `process_time_map`（可选，设备固定级联）
  - 配置驱动参数（必需构网输入）: `single_route_config`, `single_route_config_path`, `single_route_name`
- 训练入口:
  - `python -m solutions.Continuous_model.train_single --device cascade --rollout-n-envs 1`
  - 关键参数: `--device`, `--compute-device`, `--checkpoint`, `--rollout-n-envs`, `--artifact-dir`（`--artifact-dir` 作为运行名称前缀；产物写入 `results/models`、`results/training_logs`、`results/gantt`、`results/action_sequences`；图标题带 `路径 <路线名>` 后缀）
- 推理导出入口:
  - `python -m solutions.Continuous_model.export_inference_sequence --device cascade --model <model_path>`
  - action sequence 输出为 `results/action_sequences/<--out-name>.json`（默认 `tmp` → `results/action_sequences/tmp.json`）
  - 导出的 `replay_env_overrides` 会携带 `single_route_name`，并在可用时携带 `single_route_config`，用于可视化回放时保持与导出一致的构网路线
- 二次释放惩罚验证入口:
  - `python -m solutions.Continuous_model.check_release_penalty --sequence <json_name> --results-dir results`
  - `--sequence` 必填，脚本按仓库根目录 `seq/<json_name>` 解析。

## Behavior Rules
1. 构网输入严格校验：`build_net` 必须提供 `route_config` 且 `device_mode` 必须是 `cascade`，否则直接报错。
2. 构网固定走“固定拓扑 + 配置驱动 route 编译链”（`load/build static topology -> parse/normalize -> route IR -> token route -> dynamic marks/token`）；路线由 `single_route_name` + `single_route_config` 选择，不再使用数字 `route_code`。
3. 固定拓扑包含 `LP1, LP2, PM1-PM10, LLC, LLD, LP_done, TM2, TM3`（双装载口；单路线仅使用 `LP1` 发片时 `LP2` 可无 token）。
4. 变迁命名统一为 `u_src_dst` 与 `t_src_dst`；其中 `u_src_dst` 的 `dst` 是 `TM2/TM3`，`t_src_dst` 的 `src` 是 `TM2/TM3`。
5. 固定拓扑除 `results/topology_cache/topology_vN.npz` 外，另存 `results/topology_cache/transition_id_vN.npz`：`transition_id` 映射 **键** `(TM2 或 TM3, 目标腔室)` → **值** 对应 `t_*` 全名。`get_topology()` 返回的 dict 必含 `transition_id`（与缓存同版本）。
6. `compile_route_stages` 相邻 stage 的机械手由 `build_topology.infer_cascade_transport_by_scope` 判定：左、右 stage 的 candidates 须同时落在 `_TM2_SCOPE` 或 `_TM3_SCOPE` 之一；若两套 scope 同时满足则取 **TM3**。
7. 不再兼容旧命名（`d_TM*`、`t_PM*` 等）；运行时仅消费新命名。
8. 固定动作空间下，未被当前 route 使用的变迁必须在 `get_action_mask` 中恒为 0。
9. 当通过 `PetriEnvConfig.load(json/yaml)` 加载且配置中提供 `single_route_config_path` 时，会自动读取该文件并填充 `single_route_config`；`single_route_name` 必须显式提供且必须命中 `single_route_config.routes`。
10. WAIT 掩码规则：存在加工完成待取片晶圆时，仅允许短 WAIT（5s）。
11. 导出脚本的 `--out-name` 参与文件命名：`results/action_sequences/<out_name>.json`（非法字符会替换为 `_`）。
12. `check_release_penalty.py` 未设置 `--sequence` 时不能执行。
13. 旧观测分支（place-obs）不再作为当前实现接口。
14. 配置驱动路径启用时，`model_builder.build_net` 通过 `preprocess_config.py` 先构建“每腔室一块”的预处理真源（包含 stage 覆盖后的 `process_time/cleaning_*`）；`build_marks.py` 仅消费该真源构造 place，返回的 `process_time_map` 只来自该预处理真源并与 `marks` 一致，`ClusterTool._base_proc_time_map` 直接取自该字段。
15. `preprocess_chamber_runtime_blocks` 固定顺序为：先扫描 `RouteIR.stages` 收集 route stage 级 `process_time` 与 `cleaning_*` 映射；再对路线中出现的非 buffer 腔室合并工序工时并取整到 5 秒；最后按 `route_config.chambers` 与固定拓扑并集写入 `ChamberRuntimeBlock`。清洗时长与触发片数**不得**由 `obs_config` 或 `build_net` 传入默认清洗参数兜底；每个腔室必须在 `single_route_config.chambers` 中显式给出 `cleaning_duration` 与 `cleaning_trigger_wafers`，或由 route stage 覆盖二者，否则构网直接 `ValueError`。
16. 构网容量口径固定：`source/sink` 容量恒为 `100`，其余库所容量恒为 `1`；`m0` 恒为全 0，token 仅在构网末尾注入 source place。
17. 级联模式 `TM2` 的目标 one-hot 映射在 `build_marks.py` 中固定硬编码为 **9** 维（`PM7–PM10, LLC, LLD, LP_done, LP1, LP2`）；`TM3` 为 **8** 维（`PM1–PM6, LLC, LLD`），不再按 route 动态构造。
18. 并行候选选机改为 `use_count` 最小优先：`u_*` 使能在 `source` 的候选下游中，先用 route gate 过滤，再从 `use_count` 最小的目标中选择一个（同值随机）；单臂若该目标满或清洗中则不可出片，双臂若该目标清洗中则不可出片。非并行源（单一 `u_targets`）仍只校验该单一候选。
19. `get_action_mask` 的 TM 分支在并行 gate（`tuple/frozenset`）下必须仅放行 1 个 `t_*` 目标：按“最小 `use_count` + 同值随机”决策，禁止同一步并行目标多放行。
20. `_fire` 不再维护任何 round-robin 指针状态；并行目标选择完全由 `get_action_mask` / `_is_next_stage_available` 的 `use_count` 规则决定。
21. 级联观测中 `TM2` 的目标 one-hot 采用固定 **9** 维逐目标编码；`TM3` 为 **8** 维（不再按目标组压缩）；`LLC/LLD` 观测由 4 维扩展为 6 维，新增 `in/out` 两维方向 one-hot。当前版本将 `LLC/LLD` 的 `in/out` 方向位临时固定为全 0。
22. 驻留 scrap 口径：普通腔室阈值为 `process_time + P_Residual_time`；`LLC/LLD` 阈值为 `process_time + 3 * P_Residual_time`。超过阈值会按 `resident` 类型计入 scrap 判定。
23. 同一步内先推进时间再发射变迁：若 `_advance_and_compute_reward` 已对本步标 `resident` scrap，且本步随后发射的 `u_*` 从**同一腔室**取走**同一 `token_id`** 的晶圆，则 `step` 会撤销该 scrap（不计入 `scrap_count`/惩罚）。库所匹配以 `_fire` 写入 `fire_log` 的 `source_place` 为准，**禁止**用 `t_name` 去掉前缀 `u_` 后的整段作为腔室名（级联命名为 `u_PM7_TM2` 等时该段含运输后缀，与 `p.name` 不一致）。
24. `get_action_mask` 在 d_TM 分支中，当 `tok_gate` 为并行集合时，使用 token 当前并行候选（`_dst_level_targets`）与 route gate 交集做筛选，仅放行本步被选中的单一最小 `use_count` 目标对应 `t_*`。
25. 并行候选出现 `use_count` 并列最小时，必须随机选择其中一个目标；该随机仅用于并列打破，同一步 mask 内对同一 token 的筛选结果保持一致。
26. `use_count` 更新时机固定：仅当 `t_*` 变迁将晶圆放入目标库所时对该目标 `use_count += 1`；`u_*` 发射、round-robin 指针推进等历史语义不再存在。
27. 节拍门控**仅**作用于自装载口出发的 `u_*`（物理变迁名为 `u_LP1_TM2`/`u_LP1_TM3`/`u_LP2_TM2`/`u_LP2_TM3` 等；逻辑上对应原 `u_LP` 发片节拍，节拍序列来自 `analyze_cycle` 或路线级 `takt_policy` / `takt_stages_override` 等现行路径）；**不对** LLC→TM3（`u_LLC*`）施加时间间隔门控。`get_action_mask` 与 `get_next_event_delta` **不会**因 LLC 出片间隔而屏蔽或推迟；`PetriEnvConfig` **不包含** `llc_tm3_takt_interval`。
28. 双臂模式（`PetriEnvConfig.dual_arm=True`）启用 swap 操作。机械手容量固定为 1，不因双臂改变。`swap_duration` 固定为 10s。
29. 双臂模式 `get_action_mask` 对 `t_*` 变迁的 PM 目标：**仅检查**（a）腔室内晶圆是否加工完成（空腔室直接通过）、（b）机械手晶圆路由匹配（`route_gate_allows`）、（c）运输完成（TM 上 token 的 `stay_time >= proc_time`）。**不检查** `_is_struct_enabled` 容量约束。并行目标筛选同样使用 `use_count` 最小优先。对非 PM 目标（LLC/LLD/LP_done 等）沿用单臂逻辑（含 `_is_struct_enabled`）。
30. 双臂模式 `_is_next_stage_available` 在并行源上沿用规则 18 的 `use_count` 选机，仅要求目标腔室非清洗；源位不做并行环扫，`u_*` 也不维护额外选机状态。
31. 双臂模式 `_fire` 对 `t_*` 变迁：在执行时调用 `_is_swap_eligible(pst_place)` 判定是否 swap。条件：`_dual_arm=True`、目标 `is_pm`、满载、head wafer 加工完成、非清洗中。swap 时原子交换 TM 与 PM 的 token（`m` 不变），触发 `_on_processing_unload`（清洗计数）；`step` 在 `_advance_and_compute_reward` 之前计算 swap 决策，确保时长（10s）与 `_fire` 行为一致。
32. `get_action_mask` 对 `LP1`/`LP2` 的 `u_LP*`：**对每个**装载口独立判定；若该口队首 `route_type` 与 `wafer_type_to_load_port` 一致、在制上限满足（单路线：`entered_wafer_count < max_wafers1_in_system`；双子路径：不再使用单一全局在制数，仅 `_allow_start_for_route_type(route_type)` 按类型比较 `_entered_wafer_count_by_type` 与 `max_wafers1_in_system`/`max_wafers2_in_system`）、队首 `stay_time>=0`（节拍倒计时结束）、且对应 `u_LP*→TM2/TM3` 结构性可使能，则置位该变迁。**不再**使用全局「下一次发片类型」状态在掩码中排他地只放行一条 LP；`_fire` 从装载口取 token 一律为 `pre_place.pop_head()`。构网**不**输出 `lp_release_pattern` / `lp_release_pattern_types`。
33. `ClusterTool.__init__` 不再执行 `normalize_route_spec` 或 stage 覆盖提取；运行时透传 `route_config` 与 `route_name`，晶圆数由 `PetriEnvConfig` 的 `n_wafer1`/`n_wafer2` 显式给定并传入 `build_net`（单条子路径须 `n_wafer2==0`；总片数 `n_wafer1+n_wafer2`）。`build_net` **不接受**独立 `process_time_map` 形参；腔室工时仅在构网内由 `preprocess_config.preprocess_chamber_runtime_blocks` 从 `route_config.chambers` 与 route stage 覆盖推导。`ClusterTool` 只消费 `build_net` 返回的 `route_meta.route_stages`、`route_meta.cleaning_*_map`、`process_time_map` 等作为唯一预处理来源。`route_meta` **不**包含 `wafer_type_alloc_by_type`；双子路径下 `token_route_type_sequence` 仅由 `n_wafer1`/`n_wafer2` 决定（先全部类型 1 再全部类型 2）。
34. 构网期 `build_takt.build_takt_payload` 与 `build_marks` **同源**消费 `preprocess_chamber_runtime_blocks` 返回的 `chamber_blocks`：节拍分析中每腔室清洗时长与触发片数只读对应 `ChamberRuntimeBlock`；`obs_config` **仅**提供 `cleaning_enabled` 总开关（关闭时不参与节拍 q/d 计算），**不得**通过 `obs_config` 传入构网期节拍清洗数值。

## Examples
- 正例:
  - 单设备训练（CPU rollout，多环境采样）
  - 单设备推理序列导出后，用可视化回放 JSON
- 反例:
  - 继续使用旧版观测切换参数
  - 假设导出路径为历史 `action_series/<name>_<timestamp>.json`

## Edge Cases
- `train_single.py` 最佳模型当前写入 `results/models/CT_single_best.pt`，并在 `results/models/single_<timestamp>/` 保留备份。
- `check_release_penalty.py` 的 `--sequence` 参数若传完整路径，会被额外拼接到 `seq/`，建议只传文件名。

## Related Docs
- `../overview/project-context.md`
- `../training/training-guide.md`
- `../visualization/ui-guide.md`
- `../deprecated/continuous-solution-design.md`

## Change Notes
- 2026-03-31: **`ClusterTool` 并发掩码与移除 `get_enable_t`**：`ClusterTool(config, concurrent=False|True)` 由 `Env_PN_Single` / `Env_PN_Concurrent` 传入；`get_action_mask` 在并发实例上返回 `(mask_tm2, mask_tm3)`，`step` 第四项一致；删除 `get_enable_t()`；`reset()` 使能列表仍用 `get_action_mask(..., concurrent=False)` 前 `T` 维；见 `CHANGELOG.md` 与 `docs/pn_api.md`。
- 2026-03-31: **`n_wafer` → `n_wafer1`/`n_wafer2`；移除 `wafer_type_alloc_by_type`**：`PetriEnvConfig` 与 `build_net` 用显式片数；删除 `n_wafer_route1`/`n_wafer_route2`；`model_builder` 不再按 `wafer_type_alloc` 权重计算类型序列；`route_meta` 不再输出 `wafer_type_alloc_by_type`；行为规则 33 已同步；见 `CHANGELOG.md`。
- 2026-03-31: **`max_wafers_in_system` → `max_wafers1_in_system`/`max_wafers2_in_system`**：`PetriEnvConfig` 与 `ClusterTool` 改用两固定上限；删除 `wafer_type_alloc` 比例交叉乘法门控与 `_wafer_type_alloc_total_weight`；行为规则 32 已同步；见 `CHANGELOG.md`。
- 2026-03-31: **`build_takt_payload` 清洗收敛到 `chamber_blocks`**：`build_takt.py` 删除 `cleaning_duration` / `cleaning_duration_map` / `cleaning_trigger_map` 形参；`model_builder.build_net` 向 `build_takt_payload` 传入 `chamber_blocks`，节拍 q/d 与 `build_marks` 同源；`obs_config` 仅 `cleaning_enabled`。行为规则 34 已同步；见 `CHANGELOG.md`。
- 2026-03-31: **`preprocess_chamber_runtime_blocks` 顺序与清洗严格缺省**：先收集 route stage 映射，再合并工序工时并取整，最后写入 `ChamberRuntimeBlock`；删除 `default_cleaning_duration` / `default_cleaning_trigger_wafers` 形参；每个腔室的 `cleaning_duration` 与 `cleaning_trigger_wafers` 须由 `single_route_config.chambers` 或 route stage 显式给出，否则 `ValueError`。`config/cluster_tool/cascade_routes_1_star.json` 的 `chambers` 已补全上述字段。行为规则 14–15、33 已同步。
- 2026-03-31: **`preprocess_config` 移除 `process_time_map` 形参**：`solutions/A/construct/preprocess_config.py` 的 `preprocess_chamber_runtime_blocks` 不再接收独立 `process_time_map`；构网侧仅以 `route_config.chambers` 与 route stage 覆盖合并后再取整，避免调用方传 `None` 时 `dict(None)` 崩溃。行为规则 33 已同步：`build_net` 不接受该形参；`ClusterTool` 仍只消费 `build_net` 返回的 `process_time_map`。
- 2026-03-30: **A 方案初始化预处理继续下沉到构网层**：`solutions/A/petri_net.py` 删除 `normalize_route_spec` 与 `_extract_route_stage_overrides` 初始化链；`ClusterTool` 不再在构网前合并 stage 覆盖 map。`solutions/A/model_builder.py` 新增 `route_meta.route_stages` 与构网期 `cleaning_duration_map/cleaning_trigger_wafers_map` 输出；运行时仅消费构网返回元数据。`config/cluster_tool/env_config.py` 增加校验：`single_route_name` 必填且必须命中 `single_route_config.routes`。
- 2026-03-30: **初始化字段收敛（A 方案）**：`ClusterTool` 删除运行时字段 `stop_on_scrap`、`T_transport`、`T_load`、`robot_capacity`、`single_device_mode` 与 `_selected_single_route_name`，统一为固定级联口径：动作时长 `ttime=5`、scrap 必停、`single_route_name` 单一来源。`PetriEnvConfig` 同步下线 `stop_on_scrap`、`T_transport`、`T_load`、`single_robot_capacity`、`device_mode`、`cleaning_trigger_wafers`、`cleaning_duration`（全局默认阈值）；`cleaning_enabled` 保留为总开关。`single_route_config` 变为必填（可通过 `single_route_config_path` 自动装载）。清洗参数运行时统一消费 `cleaning_trigger_wafers_map`/`cleaning_duration_map` 与路线 stage 覆盖。
- 2026-03-30: **A 方案节拍计算职责迁移到构网层**：`solutions/A/construct/build_takt.py` 新增构网期节拍计算入口并在 `solutions/A/model_builder.py` 内调用；`ClusterTool` 初始化仅消费 `build_net` 返回的 `takt_payload`，`reset()` 不再重算节拍。`_compute_takt_result_from_override` 已删除，当前运行时不再消费 `takt_stages_override` 计算节拍（后续如需新口径需在构网层重建）。
- 2026-03-30: **移除 `route_code` / `route4_takt_interval`**：`PetriEnvConfig` 与 `ClusterTool` 不再接受配置项 `route_code`、`route4_takt_interval`；`model_builder.build_net` 不再接收/回传 `route_code`；删除 `takt_analysis.build_fixed_takt_result` 及 `_compute_takt_result` 中依赖旧 route4 固定节拍的特例；回放与可视化仅以 `single_route_name` / `single_route_config` 重建路线。变迁级 `t_route_code_map`（子路径编码）保留，与已删除的配置字段无关。
- 2026-03-30: **并行腔室轮转（级联）**：删除 `_first_receivable_parallel_target` 与 `_first_parallel_target_dual_arm`。`_is_next_stage_available` 对并行源仅校验 `_expected_target_for_source_stage` 所指单一腔室：单臂满或清洗则屏蔽 `u_*`，双臂清洗则屏蔽；**不再**环扫其它并行腔室。`_fire` 的 u_* 路径移除对 `_current_stage_targets_for_source` / `_rr_set_next` 的调用；指针仅在 `t_*` 落入并行候选腔室时推进。行为规则 17/18/24/29 已同步重写。
- 2026-03-31: **移除 `lp_release_pattern` 构网链**：`build_token_route_queue_multi` / `model_builder` 不再读取路线 JSON 的 `lp_release_pattern`，`route_meta` 不再含 `lp_release_pattern_types`；`cascade_routes_1_star.json` 的 `routes.4-8` 已删该字段；类型序列仅由 `n_wafer1`/`n_wafer2` 生成。行为规则 32–33 已同步；见 `CHANGELOG.md`。
- 2026-03-30: `ClusterTool` 装载口 `u_LP*`：`get_action_mask` 改为按 `LP1`/`LP2` **各自**队首独立使能（在制上限 + 队首 `stay_time` + 结构使能；2026-03-31 起上限为 `max_wafers1_in_system`/`max_wafers2_in_system`，此前曾为全局 WIP + `wafer_type_alloc` 比例门控）；移除 `_allow_start` 与 `_pending_lp_release_type`、`_pop_lp_token_for_release`；`_fire` 从装载口取 token 仅为 `pop_head()`。`get_next_event_delta` 对 LP 节拍取各类型队首倒计时的最小值。
- 2026-03-30: 级联固定拓扑版本 bump：`LP` 拆为 `LP1`/`LP2` 双装载口；`build_net` 将 JSON 中 `source.name: LP` 归一为 `LP1`；双子路径（`subpaths` 恰为 2 条）默认第一条绑定 `LP1`、第二条绑定 `LP2`，亦可在子路径上显式写 `source_name`。`route_meta` 新增 `wafer_type_to_load_port`、`load_port_names`。`n_wafer_route1`/`n_wafer_route2` 若同时给出则须与按类型分配后的 LP1/LP2 初始晶圆数一致。旧 checkpoint（TM2 观测 8 维）与当前 TM2 9 维**不兼容**。
- 2026-03-30: A 方案级联路线配置新增 `4-8/4-9` 双子路径模式：路线条目可携带 `subpaths`、`wafer_type_alloc`、`takt_policy`，其中 `4-8` 额外支持 `takt_stages_override=[3000,180]`。A 方案 `ClusterTool` 的 `u_LP` 门控为 LP token 负 `stay_time` 倒计时；**同日后续变更**已改为各装载口掩码独立使能（见本文件 Change Notes 首条与行为规则 31）。**2026-03-31** 起构网已移除 `lp_release_pattern`（见上条 Change Note）。
- 2026-03-27: 修复共享目标集的“全局严格轮转”口径：当多个 source（如 `PM7/PM8`）拥有相同并行目标集合（如 `PM9/PM10`）时，共享同一个 round-robin 指针（owner 同步），避免出现 `PM9,PM9,PM10,PM10` 的成对输出。
- 2026-03-27: 修复严格轮转口径：`get_action_mask` 对并行 gate 的 `t_*` 放行改为优先按 token 的 `last_u_source` 对应 round-robin 指针单点判定，不再仅按 `_cascade_round_robin_next.values()` 集合判定，避免同一步同时放行 `PM9/PM10` 导致非严格 robin。
- 2026-03-27: 修复 `PM7/PM8 -> PM9/PM10` 共享目标集场景的 round-robin 推进来源判定：`u_*` 会把 source 写入 `token.last_u_source`，`t_*` 发射后优先按该 source 推进对应指针，避免按“首个匹配 source”推进导致 `PM10` 长期被 gate 屏蔽。
- 2026-03-27: 新增双臂 swap 模式：`PetriEnvConfig.dual_arm=True` 启用。`get_action_mask` 双臂 `t_*` 对 PM 目标仅检查加工完成 + 路由匹配 + 运输完成（跳过容量约束）；`_fire` 在 `t_*` 目标 PM 满载且加工完成时执行 swap（原子交换 TM/PM token，`m` 不变，时长 10s）；`_is_next_stage_available` / `_fire` u_* 同步改用 `_first_parallel_target_dual_arm`（跳过容量检查）。仅 PM 可 swap，LLC/LLD/LP_done 沿用单臂逻辑。
- 2026-03-22: 修复 `_cascade_round_robin_pairs` 仅用 `PM*`/`LP_done` 过滤并行目标导致路线 `1-5`（PM7/PM8→LLC|LLD）等场景下 LLC/LLD 从未进入 `values()`，TM2 上 `t_TM2_LLC`/`t_TM2_LLD` 被 `_allow_t_by_machine_round_robin` 永久屏蔽；现对 `len(u_targets)>=2` 的源使用**完整**候选元组。
- 2026-03-22: 并行下游 `u_*` 使能从「仅指针单点」改为「从指针起循环首个可接收腔室」：`get_action_mask` 与 `_fire`（`pop_head` 前）共用 `_first_receivable_parallel_target`，避免指针落在已满并行腔（如 `PM2`）时忽略其它空腔（如 `PM3`/`PM5`）导致 `u_PM6_TM3` 等永久屏蔽；同步更新行为规则 17/18。
- 2026-03-22: 修复同一步「驻留超时 + `u_*` 取片」应撤销 scrap 却不生效：`_fire` 的 `u_*` 日志增加 `source_place`，`_should_cancel_resident_scrap_after_fire` 与 `scrap_info["place"]` 对齐；原 `t_name[2:]` 在 `u_*_TM2` 命名下与腔室名不一致导致永远不匹配。
- 2026-03-22: 移除 `llc_tm3_takt_interval` 全链路：`PetriEnvConfig` 与 `pn_single` 不再提供 LLC→TM3 节拍门控；`get_action_mask/get_next_event_delta` 仅保留 `u_LP` 节拍限制。同时将 `LLD.kind` 与固定缺省口径统一为 `buffer`，并将 `LLC/LLD` 驻留 scrap 阈值调整为 `process_time + 3 * P_Residual_time`。（后续提交已从代码中删除此前残留的节拍实现，与行为规则 24 一致。）
- 2026-03-22: 修复并行 stage 下 `u_*` 与 TM2/TM3 上 `t_*` 的 round-robin 错位：对 `_cascade_round_robin_pairs` 覆盖的源，`u_*` 发射不再推进指针，改在 `t_*` 落入该源并行 PM 腔室时推进；避免 `u_LLC_TM3` 后指针已指向下一个 PM 导致仅错误方向 `t_TM3_PM*` 使能（配置路线 `2-1` 等）。
- 2026-03-22: `construct_single` 的配置预处理拆分到 `preprocess_config.py`，以“预处理后的腔室块”作为 `process_time_map` 唯一真源；`build_marks.py` 新增 marks 构造入口并接管 `m0/md/ptime/capacity/marks` 产出。容量口径固定为 `LP/LP_done=100`、其余 place=1，`m0` 固定全 0，token 仅注入 source place。`TM2/TM3` one-hot 改为固定硬编码集合。
- 2026-03-22: `construct_single.build_single_device_net` 与内部 `_build_single_device_net_from_route_config` 合并为单一入口 `build_net`（签名与旧 `build_single_device_net` 一致）；外部调用须改用 `build_net`。
- 2026-03-22: 拓扑缓存版本升为 v3；新增 `data/cache/transition_id_v3.npz`（`transition_id`：`t_*` 的 `(TMx, 目标腔室)` 索引）。`compile_route_stages` 的 hop 机械手改为 `infer_cascade_transport_by_scope`（`_TM2_SCOPE`/`_TM3_SCOPE`）；双覆盖时取 TM3。`construct_single` 装配 `active_t_names` 时查 `transition_id`。
- 2026-03-22: 晶圆路由队列构造从 `construct_single` 迁至 `build_route_queue.py`（`build_cascade_token_route_queue`），`route_compiler_single` 仍只负责解析/归一化与 `build_token_route_plan` 等编译逻辑。
- 2026-03-21: 运输位命名从 `d_TM2/d_TM3` 统一为 `TM2/TM3`；变迁命名统一为 `u_src_dst/t_src_dst`，不再兼容旧命名。
- 2026-03-21: 下线 single 路径：`Env_PN_Single`/`ClusterTool`/构网入口均收敛到 cascade-only；传入 `device_mode!=cascade` 会直接报错。
- 2026-03-21: 构网切换为“固定全拓扑 + 动态 route 变迁集合 + 跨进程缓存”；未使用变迁在 mask 中恒为 0。
- 2026-03-21: `get_action_mask` 的 d_TM 分支新增并行 gate 机器轮转筛选：当 `tok_gate` 为 `tuple/frozenset` 时，仅允许后置目标命中 `_cascade_round_robin_next.values()` 的 `t_*` 通过 mask。
- 2026-03-21: 删除 `_peek_target_for_source_for_obs` 与 `_select_target_for_source`；`u_*` 使能改为 `_is_next_stage_available` 的“round-robin 指针单点判定”（指针目标满/清洗即直接屏蔽，不再回退其它候选）；`_fire` 不再执行目标重选与写入 `tok._target_place`；`LLC/LLD` 方向 one-hot 临时固定为全 0。
- 2026-03-21: `construct_single` 构网收敛为 `route_config`-only；移除函数内部 legacy 手写拓扑分支。影响是：未提供 `single_route_config` 的调用链会直接报错，需要迁移到配置驱动路线输入。
- 2026-03-21: `ClusterTool` 在 `single_route_name` 已设置时对同一路由只调用一次 `normalize_route_spec`，由该结果同时生成 stage 工时/清洗覆盖、`_route_stages` 与 chambers 顺序；移除后续宽泛 `try/except`，归一化或派生失败时不再静默回退到仅 `chambers` 配置的 chambers 列表。
- 2026-03-21: `_normalize_route_code` 在非法 `device_mode` 时抛出 `ValueError`（含 `device_mode` 字样），不再因 `_VALID_ROUTE_CODES` 下标访问产生 `KeyError`。
- 2026-03-20: 移除单设备「按 episode 随机缩放工序时长」能力：删除 `PetriEnvConfig.proc_rand_enabled`、`proc_time_rand_scale_map`、`chambers[].proc_rand_scale` 与 route stage `proc_rand_scale`；`train_single` 去掉 `--proc-time-rand-enabled`。
- 2026-03-20: 删除 `_episode_proc_time_map` 及 `_refresh_episode_proc_time`、`_validate_episode_proc_time_map_consistency`、`_align_base_proc_time_map_with_route_chambers`；节拍与导出统一读 `_base_proc_time_map`（来自构网返回的 `process_time_map`）；工序预处理迁入 `construct_single.py`，`pn_single` 不再含 `_preprocess_process_time_map` / `_apply_base_proc_times_to_marks_and_ptime`。
- 2026-03-20: 移除 `ClusterTool._rebuild_token_pool` / `_token_pool`；驻留 scrap 扫描改为遍历 `marks` 中腔室 token；删除无引用的 `_token_remaining_time`。
- 2026-03-20: 移除环境使能侧车与使能导出：`Env_PN_Single` 不再维护 `_last_action_enable_info`；`export_inference_sequence.py` 不再写入 `results/eval_logs.*`，CLI 去掉 `--results-dir`；可视化 verbose 仅打印步摘要。
- 2026-03-20: 移除 `ClusterTool.get_enable_actions_with_reasons` 及仅其使用的 `_has_ready_chamber_wafers`、`_is_process_ready`；使能以 `get_action_mask` 为准。
- 2026-03-20: 移除 `ClusterTool._get_enable_t`：`reset()` 第二返回值改为由 `get_action_mask` 前 `T` 维派生，使能变迁以 mask 为单一实现来源。
- 2026-03-20: 修复配置驱动路线在 `n_wafer` 大于 route 配置 `source/sink.capacity` 时的终点堵塞：`construct_single.py` 现将 source/sink 有效容量下限提升到 `n_wafer`，避免 `LP_done` 满仓后 `LLD` 长期无可选目标（`u_LLD` 不使能）。
- 2026-03-20: 配置驱动级联路线下，`pn_single` 的 round-robin 覆盖范围从 `{LP, LLC, LLD}` 扩展到“所有多候选 `u_targets` 源位”；修复 `1-2` 等路径中 `PM7/PM8 -> PM9/PM10` 长期偏置到首候选（`PM9`）的问题。
- 2026-03-20: 导出与可视化回放链路补齐配置驱动路线透传：`export_inference_sequence.py` 在 `replay_env_overrides` 中输出 `single_route_name/single_route_config`，`visualization/main.py` 构建环境时透传到 `Env_PN_Single`，修复配置路线（如 `1-2`）在 UI 回放中退化为 `route_code` 默认拓扑的问题。
- 2026-03-20: 新增可选 `llc_tm3_takt_interval`：`ClusterTool` 对 LLC→TM3 出片施加与 `u_LP` 同口径的节拍（首次不门控），用于配置驱动路线（如 `1-2`）实验。
- 2026-03-19: 修复 TM/LL 观测扩维：cascade 下 `d_TM2/d_TM3` 目标 one-hot 固定为 8 维逐目标编码（`TM2: PM7/PM8/PM9/PM10/LLC/LLD/LP_done/LP`，`TM3: PM1/PM2/PM3/PM4/PM5/PM6/LLC/LLD`）；`LLC/LLD` 从 4 维扩展到 6 维，新增 `in/out` 方向 one-hot，解决路线扩展后观测信息不足导致策略混淆。
- 2026-03-19: 配置驱动编译器新增 `LLC -> LLD` hop 选机规则：当该 hop 可被多机器人覆盖时，优先选择 `TM3 (d_TM3)`，用于满足 `2-3` 路径的内侧搬运约束。
- 2026-03-19: 修复配置驱动 route chamber 与 episode process_time map 键集合不一致问题（典型为 `1-5` 中 mixed stage `LLC/LLD`），`pn_single` 在构网后会按实际构网结果补齐缺失 chamber 工时键，避免设备切换时报 `episode process time map is inconsistent`。
- 2026-03-19: 修复配置驱动重复路径的同名变迁弧权重累加问题（同一 place-transition 弧保持单位权重），避免 `2-3` 等 repeat 路径出现 `t_PM7/t_LLD` 结构性误禁用；同时将 cascade 下目标 transport 选择改为按实际构网弧推断（不再对 `LLD` 硬编码），修复 `1-5`/`1-6` 的机械手-目标不一致导致的放片失败。
- 2026-03-19: 级联模式 LLD/LLC 多目标路径修复：同一源（如 LLD）到不同 transport（TM2/TM3）时，构网创建分离的 u_* 变迁（如 u_LLD_d_TM3、u_LLD_d_TM2）；machine 分配按目标腔室选择 TM2/TM3；cascade round_robin 从 route 的 u_targets 动态推断；route_code 6 已加入 cascade 有效集合。
- 2026-03-19: 修复配置驱动路径 `single_route_name` 下 stage 工时被全局 `process_time_map` 回写的问题；现已在 `pn_single` 初始化阶段先注入 route stage 覆盖（含 cleaning map），并统一 route 选择名与构网阶段一致。
- 2026-03-19: `PetriEnvConfig.load` 支持 `single_route_config_path` 自动装载外部路线配置文件；`cascade.yaml` 可直接通过 `single_route_name` 切换目标路线而不改代码。
- 2026-03-19: `construct_single.py` 新增配置驱动构网通路，支持 `routes.sequence/repeat`、机器人自动推断 transport place、自动编译 `token.route_queue` 与 `t_route_code_map`；`pn_single.py` 优先消费构网返回的 `route_meta`，避免与运行时路由元数据不一致。
- 2026-03-19: 建立 pn_single 主文档，统一单设备入口、脚本接口与行为规则说明。
- 2026-03-19: 修复 `1-6` 路径中 `PM3` 满载时 `u_PM1` 仍被使能的问题：`get_action_mask`（及当时并行的使能列表实现）在使用 `route_target` 时新增目标容量与清洗校验，禁止绕过 `_select_target_for_source` 的接收约束。
