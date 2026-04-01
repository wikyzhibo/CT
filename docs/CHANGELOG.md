# CHANGELOG

## 2026-04-01

### A 方案：shared+ratio 严格比例发片门控（2026-04-01）

- **What changed**：`solutions/A/petri_net.py` 新增 shared 比例循环状态（`_shared_ratio_cycle_*`）；当路线 `takt_policy=shared` 且配置存在 `ratio` 时，`get_action_mask` 在 LP 发片段仅放行“当前轮次 route_type”的 `u_LP*`。`_fire` 在 `u_LP*` 成功发片后推进比例轮次，`reset` 重置轮次到起点。新增 `tests/test_use_count_tie_breaker.py` 用例覆盖比例循环构建、轮次读取与推进回绕。
- **Why**：shared 节拍原实现只控制发片时间间隔，不控制多子路径类型发片顺序；当需要固定比例顺序时无法保证 release 序列。
- **Impact**：仅 `shared+ratio` 路线启用硬门控；其他路线（含无 `ratio` 的 shared、`split_by_subpath`）行为不变。该改动不影响现有节拍倒计时机制（`stay_time<0`）与 `step`/`action_mask` 接口形态。

## 2026-03-31

### A 方案：`render_gantt` 适配多子路线 stage 合并（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 的 `ClusterTool.__init__` 新增缓存 `route_meta.subpath_route_stages`；`render_gantt` 在 `multi_subpath` 下不再只用 `_route_stages`，改为按 stage 序号合并所有 subpath 的 stage 列表，再构建 `place->(stage,machine)`、`capacity` 与 `stage_module_names`。同一 chamber 若跨 subpath 重复出现，保留最早 stage 映射。
- **Why**：多子路线时 `route_meta.route_stages` 仅对应默认子路径，直接用于甘特会导致泳道不全且 stage 难解析。
- **Impact**：双子路径/多子路径配方的甘特泳道覆盖完整子路径并集；单路径行为保持不变。见 `docs/pn_api.md`、`docs/gantt.md`。

### A 方案：并行并列目标从随机改为确定性顺序打破（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 的 `_select_min_use_count_target` 在并列最小 `use_count` 时不再 `np.random.randint`，也不再依赖可变游标，改为在最小目标集合中按候选顺序取首个目标；`get_action_mask` 与 `_is_next_stage_available` 共享该规则。新增 `tests/test_use_count_tie_breaker.py` 覆盖“同一状态稳定 + 双目标交替 + 三目标按顺序循环”。
- **Why**：修复同一状态下多次 `get_action_mask` 查询会推进 tie-break 状态、导致并列目标抖动的问题。
- **Impact**：并列目标在状态不变时保持稳定；并列打破受候选顺序约束（例如 `PM2/PM3/PM5` 会按 `PM2 -> PM3 -> PM5` 循环）。非并列场景与 `use_count` 更新时机不变（仍在 `t_*` 入目标库所时递增）。

### A 方案：buffer 阶段腔室非零工时进入 `process_time_map`（2026-03-31）

- **What changed**：`solutions/A/construct/preprocess_config.py` 在 `_preprocess_process_time_map` 之后，对 `route_stage_proc_time` 中**有正工时**、但因 `_route_ir_preprocess_chambers` 跳过 buffer 阶段而未写入 `processed_pt` 的腔室（典型为 **LLD**）再合并一次预处理结果，使 `build_marks`/`ClusterTool._base_proc_time_map` 与库所 `processing_time` 一致。
- **Why**：路线 stage 已写 `LLD(70s)` 时，原逻辑仅对非 buffer 阶段展开 `ch_pre`，LLD 的工时从未进入 `processed_pt`，甘特 `proc_end=start`、整段显示为驻留斜线。
- **Impact**：仅影响「buffer/loadlock 阶段且 stage 工时为正」的腔室；**不**把 `LLC(0)` 等零工时送入取整（避免 `0→5`）。见 `docs/continuous-model/pn-single.md` Change Notes。

### `ppo_trainer` 并发训练结束导出动作序列与甘特（2026-03-31）

- **What changed**：`solutions/A/eval/export_inference_sequence.py` 新增 `device_mode=concurrent`：`Env_PN_Concurrent` + `DualHeadPolicyNet` 推理 rollout，序列 JSON 含每步 `actions: [TM2名, TM3名]`。`solutions/A/ppo_trainer.py` 中 `_train_concurrent` 训练结束后若存在 `CT_concurrent_best.pt`，调用 `_postprocess_training_artifacts(..., concurrent=True)`，与单动作路径一致写出 `results/action_sequences/` 与 `results/gantt/`。`_artifact_route_label` 支持 `Env_PN_Concurrent`。
- **Why**：并发训练此前仅写指标图与权重，无与单动作对齐的序列/甘特产物。
- **Impact**：依赖 `ClusterTool.render_gantt` 与 rollout 含腔室事件；无 best 时仍跳过导出。

### A 方案：`ClusterTool.render_gantt` 基于 `fire_log` 输出腔室甘特 PNG（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 实现 `render_gantt`：遍历 `fire_log`（跳过 `event_type` 清洗事件），按变迁名与 `_pst_place_indices`/`source_place` 配对腔室进出与双臂 `swap`，调用 `visualization.plot.plot_gantt_hatched_residence`。`visualization/plot.py`：`title_suffix` 为空时不显示字面 `None`；`policy is None` 时按 `2`（RL1/rl）拼输出文件名。
- **Why**：恢复训练导出与 UI 的无头甘特产物；不复用已删除的网内 `_chamber_timeline` 状态。
- **Impact**：`export_inference_sequence.rollout_and_export` 与可视化适配器在存在腔室事件时将写出 PNG；`fire_log` 无腔室占用时 `render_gantt` 报错。见 `docs/pn_api.md`、`docs/gantt.md`、`docs/training/training-guide.md`。

### A 方案：路线级 `takt_cycle` 直通覆盖（2026-03-31）

- **What changed**：
  - `solutions/A/model_builder.py` 新增 `route_entry.takt_cycle` 透传，并写入 `build_takt_payload`。
  - `solutions/A/construct/build_takt.py` 新增 `takt_cycle` 覆盖分支：命中配置后直接构建 `takt_result_by_type/takt_result_default`，不再调用 `analyze_cycle`。
  - `takt_cycle` 支持 shared 数组；双子路径支持 `takt_cycle.by_subpath` 按 `wafer_type_to_subpath` 映射到各 type。
  - `config/cluster_tool/cascade_routes_1_star.json` 的 `routes.4-13` 新增 shared 循环 `[0,150,150,150,150,150,150,150,150,150,150,150,150,150]`。
- **Why**：已知固定节拍循环时无需重复计算，配置直通可消除分析链路开销并确保实验口径可复现。
- **Impact**：配置存在 `takt_cycle` 的路线会覆盖默认节拍计算路径；未配置 `takt_cycle` 的路线保持原行为不变。

### A 方案：token 级阶段工时队列 + 4-13 变工时路线（2026-03-31）

- **What changed**：
  - `solutions/A/construct/build_route_queue.py` / `build_route_queue_multi` 新增 `token_proc_time_queue`（与 `route_queue` 对齐；`u_*=-1`，`t_*` 写 stage `process_time`）。
  - `solutions/A/construct/build_marks.py` 的 `BasedToken` 新增 `route_proc_time_queue`；构网注入 token 时按子路径模板赋值。
  - `solutions/A/petri_net.py` 的 `_fire` 在 `t_*` 入目标库所时按 token 当前 `route_head_idx` 读取 `route_proc_time_queue`，并写入目标库所 `processing_time`。
  - `solutions/A/construct/preprocess_config.py` 与 `solutions/A/model_builder.py` 放开“同 chamber 多 stage 不同工时”冲突，`chamber_blocks.process_time` 仅保留基础值。
  - `solutions/A/construct/build_takt.py` 与 `solutions/A/model_builder.py` 新增 stage 工时口径透传，节拍计算优先使用 stage `process_time`。
  - `config/cluster_tool/cascade_routes_1_star.json` 新增双子路径 `routes.4-13`。
- **Why**：同一腔室重复访问且工时变化（如 `425 -> 800`）需要按“每片 wafer 的访问序号”生效，不能由全局 place 单值工时表达。
- **Impact**：`use_count` 仍只用于并行选机；重复访问工时由 token 指针驱动。旧路线行为保持不变，新路线可表达变工时重入。

### A 方案：删除 `ClusterTool._check_scrap`，驻留 scrap 内联至 `_advance_and_compute_reward`（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 删除 `ClusterTool._check_scrap`；原循环与 `LLC`/`LLD` 阈值逻辑并入 `_advance_and_compute_reward` stay 推进循环之后。`docs/pn_api.md`、`docs/continuous-model/pn-single.md` 规则 22–23 同步（规则 22 明确 `LLC`/`LLD` 为 `3 * P_Residual_time`）。
- **Why**：单一实现位置；避免独立方法与内联双轨。
- **Impact**：外部不再存在 `ClusterTool._check_scrap` 方法名；行为与删除前一致。

### A 方案：`ClusterTool.step` 不再对 scrap 做兜底 `_check_scrap`（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 中 `step` 删除对 `scan_info` 的类型分支与「缺 `is_scrap` 键则再调 `_check_scrap`」分支；直接读取 `_advance_and_compute_reward` 返回的 `scan_info`。`_advance_and_compute_reward` 增补 docstring，明确第二项为含 `is_scrap`/`scrap_info` 的 dict。
- **Why**：驻留 scrap 单一来源已在时间推进后由 `_advance_and_compute_reward` 写入；外层防御与主路径重复。
- **Impact**：`step` 假定每步均经 `_advance_and_compute_reward` 得到完整 `scan_info`（与当前两条分支一致）。

### A 方案：`ClusterTool` 奖励仅标量（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 中 `_advance_and_compute_reward` 返回 `(float, scan_info)`；`step` 第二返回值**始终**为标量 `float`，删除 `detailed_reward` 形参及内部分项 `dict`。`Env_PN_Single` / `Env_PN_Concurrent` / `FastEnvWrapper` / `visualization/petri_adapter.py` / `petri_single_adapter.py` 同步去掉 `detailed_reward` / `with_reward` 传参与 dict 解析。
- **Why**：训练与推理只消费总奖励；避免维护双形态出口。
- **Impact**：可视化 `get_reward_breakdown()` / `info["detail"]` 恒为空（无分项）；数值与旧版 `detailed_reward=False` 时 `total` 一致。

### A 方案：`ClusterTool` 驻留 scrap 与奖励出口（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 中 `_advance_and_compute_reward` 在 stay 推进后写入驻留 `scan_info`。（后续同日条目「奖励仅标量」已改为 `_advance_and_compute_reward` 返回 `float`、不再返回分项 `dict`；再后来同日条目删除 `_check_scrap` 方法、逻辑内联至本方法。）
- **Why**：单一 scrap 语义来源。
- **Impact**：见 `docs/continuous-model/pn-single.md` 规则 22–23。


### A 方案：`ClusterTool` 并发掩码与 `get_enable_t` 移除（2026-03-31）

- **What changed**：`ClusterTool(config, concurrent=False)`；`get_action_mask(..., concurrent=None)` 在并发实例上返回 `(mask_tm2, mask_tm3)`，否则返回全量 `ndarray`；`step` 第四项与之一致。删除 `get_enable_t()`。`Env_PN_Concurrent` 使用 `ClusterTool(..., concurrent=True)` 并删除 `_build_action_masks`；`visualization/petri_adapter.py` 对使能枚举显式使用 `get_action_mask(..., concurrent=False)`；`visualization/main.py` 并发推理改用 `env.net.get_action_mask(...)`。
- **Why**：掩码形态由网构造参数决定，不经 `PetriEnvConfig`；TM 局部掩码在 `petri_net` 单点实现。
- **Impact**：直接调用 `get_enable_t` 的外部代码需改为 `get_action_mask` 或消费 `step`/环境返回的掩码；依赖 `Env_PN_Concurrent._build_action_masks` 的调用改为 `ClusterTool.get_action_mask`。

### A 方案：`ClusterTool` 观测缓存生命周期（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 删除 `_init_obs_cache`、`_obs_specs`；观测顺序、各库所 `offset`、`obs_dim` 与 `_obs_buffer` 仅在 `__init__` 中按 `load_port_names + TM2/TM3 + chambers` 构建一次；`reset()` 仅按 `_obs_place_names` 重绑 `_obs_places` 到克隆后的 `Place` 实例。
- **Why**：`reset` 替换 `marks` 后需指向新库所对象，但布局与维度不变，避免重复全量重建。
- **Impact**：无对外 API 变化；若未来在运行中改变网结构或各库所 `get_obs_dim()`，须在构造时保证与训练假设一致（当前不支持在 `reset` 中变更观测布局）。

### A 方案：`ClusterTool.render_gantt` 占位；移除 `_chamber_timeline` / `_chamber_active`（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 删除 `_timeline_chambers`、`_chamber_timeline`、`_chamber_active` 及 `_fire` 中仅服务于时间线的分支；`render_gantt(out_path, title_suffix=None)` **保留方法签名**，方法体为空（不生成 PNG、不写文件）。删除 `Op` / `plot_gantt_hatched_residence` 与仅甘特使用的 `pathlib.Path` 导入。
- **Why**：甘特绘制待重写；网内不再维护腔室占用时间线。
- **Impact**：`export_inference_sequence`、`visualization` 适配器仍可调用 `render_gantt`，**当前不会**产出 `gantt.png`。`route_meta.timeline_chambers` 仍可由构网生成，供后续实现复用。

### A 方案：`petri_net.ClusterTool` 移除 step 分段 Profiling（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 删除 `_profiling_enabled`、`_step_profile`、`_record_step_profile`、`get_step_profile_summary()`；`step()` 内不再使用 `perf_counter` 分段计时。`solutions/A/ppo_trainer.py` 训练结束不再打印 `[Step Time Profile]` 及各分段耗时。
- **Why**：移除网内热路径性能采样与训练结束汇总输出。
- **Impact**：外部若曾依赖 `get_step_profile_summary()`，需改用 `cProfile` 等工具；训练日志仅保留既有 batch/汇总时间指标。

### A 方案：`petri_net.ClusterTool` 移除晶圆时间统计链（2026-03-31）

- **What changed**：`solutions/A/petri_net.py` 删除 `_token_stats`、`_track_enter`、`_track_leave`、`calc_wafer_statistics`、仅用于统计的 `route_meta.system_entry_places` 缓存与 `enable_statistics`。`_fire` 不再调用追踪方法；保留 `get_next_event_delta`（后续 `render_gantt` / `_chamber_timeline` 变更见同日期条目）。
- **Why**：不再在网内维护 per-token 驻留/系统时间聚合；降低状态与 `u_*`/`t_*` 路径耦合。
- **Impact**：可视化 `StateInfo.stats` 中系统平均/最大、腔室与 TM 聚合时长显示为 0 或空；`completed_count`/`in_progress_count` 与违规计数仍来自 `done_count`、`n_wafer` 与网计数器。见 `docs/pn_api.md`「统计」与 `visualization/petri_adapter.py` / `petri_single_adapter.py`。

### A 方案：移除 `lp_release_pattern` / `lp_release_pattern_types` 构网链（2026-03-31）

- **What changed**：`build_token_route_queue_multi` 删除参数与返回值中的 `lp_release_pattern` / `lp_release_pattern_types`；`model_builder._build_token_type_sequence` 仅保留 `n_wafer1`/`n_wafer2` → 连续类型 1 再类型 2；`route_meta` 不再含 `lp_release_pattern_types`；`config/cluster_tool/cascade_routes_1_star.json` 的 `routes.4-8` 删除 `lp_release_pattern` 字段。
- **Why**：发片顺序不再由路线 JSON 的 pattern 表达，与已删除的 `wafer_type_alloc` 构网口径一致。
- **Impact**：若仓库外路线 JSON 仍含 `lp_release_pattern`，**会被忽略**（`extra` 或未读键）；类型序列只由 `PetriEnvConfig.n_wafer1`/`n_wafer2` 决定。

### A 方案：`n_wafer` → `n_wafer1`/`n_wafer2`；移除 `wafer_type_alloc_by_type` 与权重配额序列（2026-03-31）

- **What changed**：`PetriEnvConfig` 删除 `n_wafer`、`n_wafer_route1`、`n_wafer_route2`；新增 `n_wafer1`、`n_wafer2`。`build_net(n_wafer1, n_wafer2, ...)` 内部总片数 `n_wafer=n_wafer1+n_wafer2`；单条子路径路线要求 `n_wafer2==0`。`model_builder._build_token_type_sequence` 由 `n_wafer1`/`n_wafer2` 直接生成类型序列（连续 `1…1` 再 `2…2`）。`build_token_route_queue_multi` 不再接收 `wafer_type_alloc`；`route_meta` 不再包含 `wafer_type_alloc_by_type`。`ClusterTool` 删除对 `route_meta.wafer_type_alloc_by_type` 的依赖。`config/cluster_tool/cascade.yaml`、可视化 `config_editor`、`solutions/B/clustertool_config.py`/`clustertool_default.json`/`Env.py`/`core.py`、`PetriEnvConfig.load` 与旧键迁移同步。
- **Why**：片数与类型分配由配置显式给定，不再从 JSON `wafer_type_alloc` 权重推导或按比例取整。
- **Impact**：`cascade_routes_1_star.json` 中 `wafer_type_alloc` 字段仍可为文档/参考保留，**构网不再读取**；双子路径实验须设置与目标一致的 `n_wafer1`/`n_wafer2`。

### `preprocess_chamber_runtime_blocks` 顺序调整、清洗严格缺省、移除默认清洗形参（2026-03-31）

- **What changed**：`solutions/A/construct/preprocess_config.py` 先扫描 `RouteIR.stages` 收集 `route_stage_*`，再对路线腔室做工时合并与 5s 取整，最后按 `route_config.chambers` 与固定拓扑并集构造 `ChamberRuntimeBlock`。删除 `default_cleaning_duration` 与 `default_cleaning_trigger_wafers` 参数；清洗时长与触发片数仅来自 route stage 或 `route_config.chambers` 显式字段，缺一即 `ValueError`。`solutions/A/model_builder.py` 不再向预处理传入上述默认。`config/cluster_tool/cascade_routes_1_star.json` 为全部 `chambers` 条目补充 `cleaning_duration` / `cleaning_trigger_wafers`。
- **Why**：避免 `obs_config` 隐式兜底与构网真源不一致；配置与路线显式声明清洗参数。
- **Impact**：自定义 `single_route_config` 必须为每个参与固定拓扑并集的腔室提供清洗字段（或由 stage 覆盖）；否则构网失败。

### `build_takt_payload` 清洗口径收敛到 `chamber_blocks`（2026-03-31）

- **What changed**：`solutions/A/construct/build_takt.py` 的 `build_takt_payload` 及内部 `_build_takt_stage` / `_compute_takt_result*` 删除形参 `cleaning_duration`、`cleaning_duration_map`、`cleaning_trigger_map`，改为接收 `chamber_blocks`（`preprocess_chamber_runtime_blocks` 输出）。`solutions/A/model_builder.py` 调用处传入 `chamber_blocks`；`cleaning_enabled` 仍来自 `obs_config`，构网期节拍不再从 `obs_config` 读取清洗数值。
- **Why**：节拍与 `build_marks`、`route_meta.cleaning_*_map` 同源，避免 obs 覆盖与预处理真源分叉。
- **Impact**：不再支持仅用 `obs_config.cleaning_duration_map` 等改变构网期节拍而保持其余构网输出不变；须改 `single_route_config` 或 route stage 使 `ChamberRuntimeBlock` 一致。

### `preprocess_chamber_runtime_blocks` 移除 `process_time_map` 形参（2026-03-31）

- **What changed**：`solutions/A/construct/preprocess_config.py` 中 `preprocess_chamber_runtime_blocks` 删除参数 `process_time_map`；合并工时初值仅来自 route stage 覆盖字典，再经 `_preprocess_process_time_map` 与 `route_config.chambers` 缺省对齐并取整到 5 秒。`solutions/A/model_builder.py` 去掉 `process_time_map=None` 调用实参；固定拓扑占位腔室未声明时 `process_time` 初值为 `0`。
- **Why**：构网路径从未向 `build_net` 传入独立 `process_time_map`，原 `None` 会在 `dict(process_time_map)` 处失败；删除形参使契约与实现一致。
- **Impact**：若存在仓库外调用并依赖向预处理注入独立 `process_time_map`，需改为写入 `single_route_config` 的 `chambers` 或 route stage；返回的 `build_net`["process_time_map"] 仍为腔室工时真源。

### A 方案：`max_wafers_in_system` 替换为 `max_wafers1_in_system` / `max_wafers2_in_system`（2026-03-31）

- **What changed**：`PetriEnvConfig` 删除 `max_wafers_in_system`；新增 `max_wafers1_in_system`、`max_wafers2_in_system`（固定整数，不由构网计算或返回）。`solutions/A/petri_net.py`：单路线用 `entered_wafer_count < max_wafers1_in_system`；双子路径不再用单一全局在制门控与 `wafer_type_alloc` 比例交叉乘法门控，`_allow_start_for_route_type` 对 `route_type` 1/2 分别比较 `_entered_wafer_count_by_type` 与 `max_wafers1_in_system`/`max_wafers2_in_system`；删除 `_wafer_type_alloc_total_weight`。`config/cluster_tool/cascade.yaml`、`visualization/config_editor.py`、`solutions/B/clustertool_config.py`、`solutions/A/deprecated/pn.py` 同步。
- **Why**：在制上限由配置显式给定；`wafer_type_alloc` 仍用于构网/token 序列等，不再参与 WIP 上限推导。
- **Impact**：仅含旧键 `max_wafers_in_system` 的 YAML/JSON 会被忽略（`extra=ignore`），须改为两键；两值可相同以复现原「单一全局上限」语义。

### A 方案：`max_wafers_in_system` 按类型比例硬约束发片（2026-03-30）**已废止**

- **Note**：2026-03-31 起由 `max_wafers1_in_system`/`max_wafers2_in_system` 替代；下述为历史记录。
- **What changed（历史）**：`solutions/A/petri_net.py` 的 `u_LP` 放行曾从“仅全局 WIP `< max_wafers_in_system`”改为“双重门控”：在保持全局上限校验的同时，对双子路径场景按 `wafer_type_alloc` 施加类型内在制品上限；交叉乘法判定 `(type_wip+1)*sum_alloc <= max_wafers_in_system*type_alloc`。
- **Impact（历史）**：当某类型达到其比例上限后，即使全局 WIP 仍有余量，也不会继续放该类型晶圆。

## 2026-03-30

### A 方案：初始化预处理全量下沉到构网层（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 删除初始化阶段的 `normalize_route_spec` 与 `_extract_route_stage_overrides`，不再在运行时构网前提取/合并 stage 覆盖。`solutions/A/model_builder.py` 的 `build_net` 新增输出 `route_meta.route_stages`、`route_meta.cleaning_duration_map`、`route_meta.cleaning_trigger_wafers_map`，由 `ClusterTool` 直接消费。`config/cluster_tool/env_config.py` 新增校验：`single_route_name` 必填且必须存在于 `single_route_config.routes`。
- **Why**：预处理链路在运行时和构网层重复实现会导致口径漂移；把预处理收敛到构网层可保证路线阶段、清洗参数和工时来源唯一。
- **Impact**：`ClusterTool.__init__` 现在只负责透传配置并接收 `build_net` 输出；若 `single_route_name` 缺失或拼写错误会在配置加载阶段直接报错，不再延迟到运行时 fallback。

### A 方案：`ClusterTool/PetriEnvConfig` 初始化字段收敛（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 删除运行时字段 `stop_on_scrap`、`T_transport`、`T_load`、`robot_capacity`、`single_device_mode`、`_selected_single_route_name`；固定 `ttime=5`，发生 scrap 即终止；路由名统一只用 `single_route_name`。`config/cluster_tool/env_config.py` 下线配置字段 `stop_on_scrap`、`T_transport`、`T_load`、`single_robot_capacity`、`device_mode`、`cleaning_trigger_wafers`、`cleaning_duration`（全局默认阈值）；`cleaning_enabled` 保留为总开关。`single_route_config` 设为必填（支持通过 `single_route_config_path` 自动装载）。`visualization/petri_single_adapter.py` 不再依赖 `cleaning_targets`。
- **Why**：这些参数在当前 A 方案级联路径已固定常量或重复表达，保留会增加配置歧义与维护成本。
- **Impact**：`cascade.yaml` 需移除上述下线键；运行时清洗参数统一从 `cleaning_trigger_wafers_map/cleaning_duration_map` 读取；`cleaning_enabled` 仍为总开关；旧脚本若仍设置已下线字段将被忽略或需迁移。

### A 方案：并行选机从 robin 改为 `use_count` 最小优先（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 删除 robin 指针相关状态与函数（`_cascade_round_robin_*`、`_single_round_robin_*`、`_rr_*`、`_allow_t_by_machine_round_robin`、`_advance_round_robin_after_u_fire` 等），新增库所级 `use_count` 映射。并行 gate 下的 `get_action_mask` 改为按候选目标 `use_count` 最小优先，仅放行 1 个目标；最小值并列时随机选 1 个。`_is_next_stage_available` 同步改用相同规则。`_fire` 在 `t_*` 把晶圆放入目标库所时对该目标 `use_count += 1`。
- **Why**：去掉轮转指针维护链路，直接用“最少使用优先”提升并行机台均衡度，并让并行选机规则在掩码与执行路径保持一致。
- **Impact**：并行目标不再按 source 严格轮转，改为动态计数均衡；同计数目标存在随机性，会影响逐步动作可复现性（训练统计层面通常更均衡）。

### A 方案：节拍计算迁移到构网层并移除 override 分支（2026-03-30）

- **What changed**：新增 `solutions/A/construct/build_takt.py` 作为节拍计算入口，`solutions/A/model_builder.py` 在 `build_net` 内完成节拍计算并返回 `takt_payload`；`solutions/A/petri_net.py` 删除 `_build_takt_stage`、`_validate_takt_stage_inputs`、`_compute_takt_result*` 系列本地计算函数（含 `_compute_takt_result_from_override`），改为仅消费构网返回的节拍结果。
- **Why**：节拍口径应与构网元数据同源，避免运行时重复计算与多处逻辑分叉；同时为后续重写 override 逻辑预留单一实现入口。
- **Impact**：`ClusterTool.reset()` 不再重算节拍；当前版本不会使用 `takt_stages_override` 参与节拍计算，shared/split 仅走主路径/子路径 stage 计算结果。

### A 方案：移除 `route_code`、`route4_takt_interval` 与 `build_fixed_takt_result`（2026-03-30）

- **What changed**：删除 `PetriEnvConfig.route_code`、`route4_takt_interval`；`ClusterTool` 不再设置 `route_code` / `single_route_code`；`model_builder.build_net` 去掉 `route_code` 参数及返回中的 `single_route_code`；删除 `takt_analysis.build_fixed_takt_result` 与 `_compute_takt_result` 内旧 route4 固定节拍分支；`Env_PN_Single` / `Env_PN_Concurrent` 去掉 `route_code` 形参；可视化 `build_adapter` 去掉 `--single-route-code`；`export_inference_sequence` 的 `replay_env_overrides` 不再输出 `route_code`。
- **Why**：路线已统一由 `single_route_name` + `single_route_config` 驱动；数字别名与 route4 专用节拍入口已弃用。
- **Impact**：旧回放 JSON 若仅含 `route_code` 需改为提供 `single_route_name`（及必要时内嵌 `single_route_config`）。变迁级 `t_route_code_map` 命名保留，与已删除的配置字段无关。

### A 方案：级联并行腔室轮转改为严格指针（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 删除 `_first_receivable_parallel_target`、`_first_parallel_target_dual_arm`。`_is_next_stage_available` 对 `_cascade_round_robin_pairs` 中的源仅根据 `_expected_target_for_source_stage(source, _current_stage_targets_for_source(...))` 检查**单一**期望腔室：单臂满或清洗则 `u_*` 不使能，双臂仅清洗不使能。`_fire` 的 `u_*` 分支不再在 `pop_head` 前调用 `_current_stage_targets_for_source` 或 `_rr_set_next`。
- **Why**：使能/掩码层统一决定期望下游；`_fire` 只执行已选变迁；轮转指针仅在 `t_*` 装入并行腔室后推进，不再通过环扫改投其它 PM。
- **Impact**：指针落在满/清洗腔室时可能增加等待，不再自动跳过到其它空腔；详见 `continuous-model/pn-single.md` 规则 17/18。

### A 方案：`ClusterTool` 装载口发片掩码改为按 LP 独立使能（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 删除 `_allow_start`、`_pending_lp_release_type`、`_pop_lp_token_for_release` 及 `_peek_lp_token_by_type` / `_lp_releasable_types` / `_select_lp_release_type`；`get_action_mask` 对 `LP1`/`LP2` 各自队首独立应用在制上限（见 2026-03-31 `max_wafers1_in_system`/`max_wafers2_in_system` 条目；此前为全局 WIP + `wafer_type_alloc` 比例门控）、`_allow_start_for_route_type`、队首 `stay_time` 与结构使能；`_fire` 从装载口取 token 仅为 `pre_place.pop_head()`。`get_next_event_delta` 对 LP 节拍取各类型队首倒计时的最小值。队首 `route_type` 与 `wafer_type_to_load_port` 不一致时抛出 `RuntimeError`。
- **Why**：约定每装载口仅一种晶圆类型、无需「全局下一次发哪一种」与 deque 内按类型重排；双 LP 可同时出现在掩码中，由策略择一动作。
- **Impact**：依赖「掩码同刻至多一条 LP」排他发片的旧实验需改用新语义。构网 `lp_release_pattern` 已于 2026-03-31 移除（见同日条目）。

### A 方案：双臂 swap 场景驻留豁免改为匹配被换出晶圆（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 的驻留豁免函数 `_should_cancel_resident_scrap_after_fire` 新增对 swap 日志的支持：当本步 `t_*` 触发 `swap=True` 时，豁免匹配从原先 `u_*` 的 `source_place + token_id` 扩展为 `swap_source_place + swapped_token_id`。`_fire` 的 swap 日志同步写入 `swap_source_place`（PM 目标腔室名）。
- **Why**：双臂交换中，驻留违规发生在 PM 内“被换出”的晶圆，而原逻辑仅允许 `u_*` 且只读取 `token_id`（进入 PM 的晶圆），会导致同一步已完成 swap 取片仍误判驻留违规。
- **Impact**：仅影响双臂 swap 的同步豁免判定；非 swap 与普通 `u_*` 豁免口径保持不变。

### A 方案：`u_LP` 节拍门控改为 LP 队首 token 负 `stay_time` 倒计时（2026-03-30）

- **What changed**：`solutions/A/petri_net.py` 的 `u_LP` 放行判定不再使用 `time - _last_u_LP_fire_time`，改为读取 `LP` 队首 token：`stay_time < 0` 表示仍在节拍倒计时、`stay_time >= 0` 才允许发片。每次 `u_LP` 实际发射后，仅给“新的 LP 队首 token”写入 `-required_interval`；`_advance_and_compute_reward` 对 `LP` 上负 `stay_time` 做向 0 推进；`get_next_event_delta` 同步读取该倒计时作为下一事件。
- **Why**：旧方案若提前给后续多片 token 预写累计负值（如第二片 `-180`、第三片 `-360`），在前片延后发射时会破坏“相邻发片最小间隔”约束。
- **Impact**：首片保持可立即发射（初始 `stay_time=0`）；后续每片的等待起点变为“上一片实际发射时刻”，可严格保证节拍间隔。resident/Q-time 判定口径保持不变（仍作用于腔室与运输位）。

### A 方案：4-8/4-9 支持双子路径、双类型晶圆与分策略节拍（2026-03-30）

- **What changed**：`config/cluster_tool/cascade_routes_1_star.json` 新增 `routes.4-8` 与 `routes.4-9`，并引入增量字段：`subpaths`、`wafer_type_alloc`、`takt_policy`、`takt_stages_override`（4-8）。`solutions/A/construct/build_route_queue.py` 新增 `build_token_route_queue_multi`，对多子路径统一构造 `target_code_map/t_route_code_map`，并输出按子路径与按 wafer_type 的 queue/template。`solutions/A/model_builder.py` 在多子路径下按并集激活变迁，返回 `token_route_queue_templates_by_type`、`token_route_type_sequence` 及 route-level 元数据（`subpath_to_type`、`wafer_type_to_subpath`、`takt_policy` 等）。`solutions/A/construct/build_marks.py` 支持按类型初始化 LP token（`route_type + route_queue`）。（**2026-03-31**：`lp_release_pattern` 已从构网移除，见同日 Change Notes。）
- **Why**：4-8/4-9 同时存在两条子路径且需两种晶圆并行；原单模板 queue 与单路由节拍口径无法表达“按类型静态绑定子路径”与“按策略共享/拆分节拍”。
- **Impact**：A 方案在 `single_route_name=4-8/4-9` 下可直接运行双子路径；4-8 使用共享节拍（含抽象 `3000/180` override）且 LP 发片按 pattern 循环；4-9 按子路径拆分节拍组，LP 同时可发时随机选类型（仍单步只发一片）。

### A 方案：新增 `4-10/4-11/4-12` 双子路径路线（2026-03-30）

- **What changed**：`config/cluster_tool/cascade_routes_1_star.json` 新增 `routes.4-10`、`routes.4-11`、`routes.4-12`。三条路线均采用 `subpaths(path1/path2)` + `takt_policy=split_by_subpath` + `wafer_type_alloc={path1:1,path2:2}`。其中 `4-11` 的两条子路径均包含 `repeat` 语法段（`count=2`）以表达中段循环。
- **Why**：补充 4 系列新的双子路径工艺模板，并按“各自计算节拍”口径让 path1/path2 独立分析节拍周期，避免共享节拍对不同子路径节奏的干扰。
- **Impact**：可直接使用 `single_route_name=4-10|4-11|4-12` 运行配置驱动级联构网；LP 类型配比默认为 1:2，且节拍门控按子路径拆分生效（非 shared）。

## 2026-03-29

### A 方案并发环境：从 deprecated `Petri` 切到当前 `ClusterTool`（2026-03-29）

- **What changed**：`solutions/A/rl_env.py` 的 `Env_PN_Concurrent` 不再导入 `solutions.A.deprecated.pn.Petri`，而是和单动作级联环境一样读取 `config/cluster_tool/cascade.yaml` 并直接实例化 `solutions.A.petri_net.ClusterTool`。`ClusterTool.step()` 和 `_fire()` 扩展为同时支持单动作与双动作并发执行，并继续返回统一的 5 元组结果；TM2/TM3 动作分组改为从当前 `id2t_name` 与 transport 映射动态生成。
- **Why**：deprecated `Petri` 已经不是当前 A 方案的真实运行时；继续绑定旧网会让并发 rollout、UI、模型动作空间与当前级联逻辑脱节。
- **Impact**：A 方案并发训练、并发 UI、并发回放现在统一复用当前 `ClusterTool` 运行时；并发 WAIT 固定为 `5s`，不再存在 legacy 并发动作名兼容层。

### 可视化：并发适配器与回放判定改为当前运行时口径（2026-03-29）

- **What changed**：`visualization/main.py` 的并发分支会同步透传 `route_code`、`single_route_name`、`single_route_config`、`process_time_map` 给 `Env_PN_Concurrent`。`visualization/petri_adapter.py` 改为直接消费当前 net 的真实库所与真实变迁，不再依赖 `LP1/LP2/s1-s5/d_TM2/d_TM3` 硬编码。`visualization/main_window.py` 的 Model B 判定改为“显式 `runtime_mode` / `device_mode` 优先，双非 WAIT 动作兜底”，并恢复并发级联 runtime 的路线横幅与路径切换。
- **Why**：旧 UI 口径同时依赖过期导入、旧阶段名映射和错误的并发判定规则，会把当前 `ClusterTool` 导出的序列误判，或者把并发动作压回单动作执行。
- **Impact**：需要并发双动作可视化时，继续使用 `python -m visualization.main --device cascade --concurrent`；同时当前单动作导出的 `actions=[action, \"WAIT\"]` 序列不会再被误识别为并发。`visualization/smoke_test.py` 的环境引用也已修正到 `solutions.A.rl_env.Env_PN_Concurrent`。

## 2026-03-27

### 级联路线配置新增 `3-1/3-2/3-3`（2026-03-27）

- **What changed**：在 `config/cluster_tool/cascade_routes_1_star.json` 新增 `routes.3-1`、`routes.3-2`、`routes.3-3`。`3-1` 为 `LP->PM7/PM8/PM9(267s)[320s/6片]->LP_done`，`3-2` 为 `LP->PM7(267s)[320s/6片]->LP_done`，`3-3` 为 `LP->[PM8(267s)->PM10(267s)]*5->LP_done`（无清洗字段）。
- **Why**：补充 3 组新工艺路径模板，便于在配置驱动路由下直接按 `single_route_name` 切换实验路线。
- **Impact**：可直接使用 `single_route_name=3-1|3-2|3-3`；`legacy.route_code_alias.cascade` 保持不变，不影响既有 `1-*` 与 `2-*` 路线映射。

### A 方案：repeat 重入路径节拍自动归零（2026-03-27）

- **What changed**：`solutions/A/model_builder.py` 在构网时基于 `route_ir.stages[*].repeat_origin` 写入 `route_meta.has_repeat_syntax_reentry`。`solutions/A/petri_net.py` 初始化时读取该标记；`_compute_takt_result` 在常规 `analyze_cycle` 前短路，若命中重入则直接返回全 0 节拍结果结构（`cycle_length=100`、`cycle_takts` 全 0）。
- **Why**：repeat 语法代表路径存在重入，需要统一按“自动零节拍”口径处理，避免重入路径继续走常规节拍分析。
- **Impact**：仅当路线通过 `repeat` 语法展开时触发该行为；非 repeat 路线保持原有节拍分析流程不变。

### requirements：补充可视化依赖 PySide6（2026-03-27）

- **What changed**：根目录 `requirements.txt` 新增 `PySide6>=6.0`。`README.md` 的 Quickstart 同步增加 `pip install -r requirements.txt` 安装提示。
- **Why**：当前可视化入口 `python -m visualization.main` 直接依赖 `PySide6`，未声明时按 README 主路径执行会在导入阶段失败。
- **Impact**：新环境按 `requirements.txt` 安装后即可满足可视化界面的基础 GUI 依赖。

### `solutions` 顶层导入入口兼容 A/B 模块（2026-03-27）

- **What changed**：`solutions/__init__.py` 新增懒加载导出入口，可直接 `from solutions import petri_net, rl_env, core, train`，也可继续 `from solutions import A, B`。`construct` 在 `A` 与 `B` 中同名，保持为命名空间导入，不做顶层扁平化。`README.md` 与 `docs/overview/project-context.md` 已同步加入导入规则与示例。
- **Why**：减少 A/B 方案脚本中的重复长路径导入，同时避免同名模块扁平化后的歧义。
- **Impact**：需要顶层导入时，只能直接导入无重名模块；`construct` 仍必须写成 `solutions.A.construct` 或 `solutions.B.construct`。

### PetriEnvConfig：Pydantic + YAML；移除 reward_config（2026-03-27）

- **What changed**：`data/petri_configs/env_config.py` 改为 Pydantic 模型；默认级联配置为 `data/petri_configs/cascade.yaml`；删除 `cascade.json`。`PetriEnvConfig` 不再包含 `reward_config` 字段；`load` 会丢弃文件中的 `reward_config` 键。`ClusterTool` 中奖励分项布尔开关固定为 `True`；`solutions/A/deprecated/pn.py` 内 `Petri.reward_config` 仍为全开字典。`Env_PN_Single` 默认加载 `cascade.yaml` 并去掉 `reward_config` 形参。`warn_coef_penalty` 类型改为 `float`（与 `cascade.yaml` 一致）。可视化 `config_editor` 去掉 `reward_config` 表单项并支持 YAML 读写。
- **Why**：配置只保留系数与拓扑；奖励分项不再在 cascade 中切换。
- **Impact**：外部若传入 `Env_PN_Single(..., reward_config=...)` 需删除该参数；旧 JSON 中含 `reward_config` 仍可加载但被忽略。

### PPO 训练配置：YAML + Pydantic（2026-03-27）

- **What changed**：`data/ppo_configs/training_config.py` 中 `PPOTrainingConfig` 改为 Pydantic v2 模型；默认磁盘配置为 `data/ppo_configs/s_train.yaml`；`load` 按后缀支持 `.yaml`/`.yml`/`.json`；`save` 按后缀写 YAML 或 JSON；删除 `s_train.json`。根目录新增 `requirements.txt`（`pydantic>=2`、`PyYAML>=6`）。`solutions/A/ppo_trainer.py` 默认加载 `s_train.yaml`；`solutions/C/train.py` 将运行中备份配置扩展名改为 `.yaml`。
- **Why**：用结构化模型校验训练超参，YAML 便于手写与 diff。
- **Impact**：须在环境中安装上述依赖；若外部脚本仍指向 `s_train.json`，请改为 `s_train.yaml` 或继续使用 `load` 读任意路径的 `.json` 文件。

## 2026-03-26

### construct 迁移兼容：deprecated 承接 legacy 构网类 (2026-03-26)
- **What changed**：新增 `solutions/Continuous_model/deprecated/construct.py`，承接 `BasedToken`、`SuperPetriBuilder`、`ModuleSpec`、`RobotSpec`、`SharedGroup`、`Stage`、`INF`；`solutions/Continuous_model/construct/__init__.py` 从动态加载 `construct.py` 改为显式从 `deprecated.construct` 转发同名导出。
- **Why**：`solutions/Continuous_model/construct.py` 已迁移/移除后，原 `importlib` 按路径加载会直接失败，导致 `from solutions.Continuous_model.construct import ...` 断裂。
- **Impact**：保留历史导入路径兼容；主路径与 deprecated 并发路径可继续通过 `construct` 包入口获取 `BasedToken` 与 legacy 构网类。

## 2026-03-23

### PDR：记录发射时刻并导出 5s 粒度回放序列 (2026-03-23)
- **What changed**：`solutions/PDR/net.py` 在 DFS 路径上新增 `full_transition_records`（`transition + fire_time`），叶子容器新增 `LEAF_PATH_RECORDS`。新增 `solutions/PDR/parse_sequences.py`：将 `full_transition_records` 转换为 `schema_version=2` 的 single 回放序列；当相邻真实动作时间差超过 5 秒时自动插入 `WAIT_5s`。`solutions/PDR/run_pdr.py` 在 `search()` 后自动导出 `seq/pdr_sequence.json`，并保留解析脚本 CLI 独立导出能力。
- **Why**：仅有变迁名无法还原时间轴；可视化回放需要带时间的动作序列，并需要 5 秒粒度的等待动作补齐间隔。
- **Impact**：PDR 搜索结果可直接转换为 UI Model B 可加载序列；`full_transition_path` 兼容保留，新增消费者可切到 `full_transition_records` 获取时序信息。

## 2026-03-22

### train_single：`--artifact-dir` 下恢复甘特与图标题路径后缀 (2026-03-22)
- **What changed**：存在 `best.pt` 时 `rollout_and_export` 支持 `gantt_png_path` / `gantt_title_suffix`，训练后写入 `artifact_dir/gantt.png`；`plot_gantt_hatched_residence` 与 `plot_metrics` 支持标题后缀。`train_single` 用 `env.net.single_route_name` 生成 `路径 <name>` 后缀。
- **Why**：指标图与甘特需标注路线；甘特依赖 rollout 后的网内时间线。
- **Impact**：仅传 `gantt_png_path` 的调用方会多一次无头绘图；CLI `plot_train_metrics` 增加 `--route-label`。

### train_single：训练曲线改由 `eval.plot_train_metrics.plot_metrics` (2026-03-22)
- **What changed**：删除 `train_single.py` 内联 `_plot_training_dashboard` 及对 `matplotlib` 的依赖；`--artifact-dir` 在写入 `training_metrics.json` 后调用 `solutions.Continuous_model.eval.plot_train_metrics.plot_metrics` 生成 `training_metrics_plot.png`。`eval/plot_train_metrics.py` 提供 `plot_metrics` 与 CLI（`--input` / `--output` / `--smooth-window` / `--show`）。
- **Why**：绘图逻辑集中到 eval，便于单独复现与调样式。
- **Impact**：不再生成 `training_dashboard.png`；依赖该文件名的外部流程需改为 `training_metrics_plot.png` 或自行调用 CLI。文档已同步 `training-guide.md`、`pn-single.md`、`project-context.md`。

### train_single：`training_metrics.json` 四指标导出 (2026-03-22)
- **What changed**：`--artifact-dir` 下除 `training_log.json` 外，同目录写入 `training_metrics.json`，仅含键 `reward`、`makespan`、`finish`、`scrap`（与 batch 顺序对齐的列表）。
- **Why**：下游脚本只需曲线数据时不必解析完整训练 log。
- **Impact**：无；未传 `--artifact-dir` 时不生成该文件。文档已同步 `training-guide.md`、`pn-single.md`。

### train_single：`--artifact-dir` 与训练后导出/甘特/dashboard；export 按 `out-name` 写 seq (2026-03-22)
- **What changed**：`train_single.py` 增加 `--artifact-dir`：在该目录写入 `best.pt`/`final.pt`/`training_log.json`，训练结束后用 best 导出 `seq/<目录名>.json`、无头回放生成 `gantt.png`、并保存 `training_dashboard.png`（reward+makespan / finish+scrap / 嵌入甘特）。`export_inference_sequence.rollout_and_export` 改为写入 `seq/<out_name>.json`（`out_name` 非安全字符替换为 `_`）；CLI `--model` 若为已存在文件路径则直接使用，否则 `models/<相对路径>`。默认 `--out-name tmp` 保持 `seq/tmp.json`。
- **Why**：单机实验需要可复现产物目录；旧版 `out_name` 未参与命名导致并发覆盖 `tmp.json`。
- **Impact**：依赖「永远写 tmp.json」的外部脚本需显式 `--out-name tmp` 或更新路径。全程未写出 `best.pt` 时跳过序列导出与 dashboard（仍写 `final.pt` 与 `training_log.json`）。文档已同步 `training-guide.md`、`pn-single.md`、`ui-guide.md`、`project-context.md`。

### 文档：根 README 架构图、可视化模块与 Quickstart (2026-03-22)
- **What changed**：根目录 `README.md` 在 `## Docs` 前新增 **Quickstart**（级联/并发训练、推理导出、PySide6 可视化 CLI）与 **Continuous_model 架构（概览）**（Mermaid：构网 / 设备模拟 / 训练 / 可视化；文件映射与数据流摘要）。
- **Why**：为新人提供与 `docs/training/training-guide.md`、`docs/visualization/ui-guide.md`、`docs/overview/project-context.md` 对齐的一页入口，并与 `solutions/Continuous_model/`、`visualization/main.py` 模块边界一致。
- **Impact**：命令与参数以 README 表格为索引，行为与边界仍以 `docs/` 与源码为准；注明 `train_single` 默认 `--device` 与当前 `Env_PN_Single` cascade-only 的差异，推荐使用 `--device cascade`。

### single(cascade)：`1-5` 等 LLC|LLD 并行段 TM2 放片被误屏蔽 (2026-03-22)
- **What changed**：`pn_single.ClusterTool` 构造 `_cascade_round_robin_pairs` 时，对 `len(u_targets)>=2` 的源改为使用**全部**下游候选（不再仅保留 `PM*` 与 `LP_done`）。
- **Why**：路线 `1-5` 中 `PM7`/`PM8` 的 `u_targets` 为 `LLC` 与 `LLD`；旧过滤使 `PM7`/`PM8` 不进入轮转对，`get_action_mask` 里并行 `t_*` 要求目标 ∈ `_cascade_round_robin_next.values()`，而 `values()` 中永远没有 `LLC`/`LLD`，导致晶圆出 PM7 后无法在 TM2 上使能进入 LLC/LLD 的 `t_*`。
- **Impact**：凡并行下游含 LLC/LLD（或其它非 PM 腔室）的级联路线，TM 上对应 `t_*` 与 `u_*` 的轮转/掩码与构网 `u_targets` 一致。

### single(cascade)：并行下游 `u_*` 使能循环扫描可接收腔室 (2026-03-22)
- **What changed**：`pn_single.ClusterTool` 新增 `_first_receivable_parallel_target`；`_is_next_stage_available` 对 `_cascade_round_robin_pairs` 中的源从当前 `_cascade_round_robin_next[source]` 起在并行候选上循环，取第一个未满且非清洗的下游。`_fire` 在 `u_*` 的 `pop_head` 之前将 `_cascade_round_robin_next[source]` 同步为该次扫描结果，与掩码一致。
- **Why**：旧实现只检查指针所指单一并行腔；若该腔已满而其它并行腔为空（如路线 `1-4` 上 `PM6→PM2/PM3/PM5` 且指针停在已满的 `PM2`），会导致 `u_PM6_TM3` 等误屏蔽，且与「存在可放片空腔」的直觉不符。
- **Impact**：多候选并行 stage 下，只要环上至少有一个可接收腔，对应源的 `u_*` 即可使能；`t_*` 并行 gate 与同步后的指针一致。详见 `docs/continuous-model/pn-single.md` 行为规则 17/18。

### single(cascade)：驻留 scrap 与同步 `u_*` 取片撤销逻辑修复 (2026-03-22)
- **What changed**：`ClusterTool._fire` 对 `u_*` 发射在 `fire_log` 条目中增加 `source_place`（取片前库所名，与 `marks` 中腔室名一致）。`_should_cancel_resident_scrap_after_fire` 用 `source_place` 与 `scrap_info["place"]` 比较，不再使用 `t_name[2:]`（级联下为 `PM7_TM2` 等，与 `PM7` 永不相等）。**实现约束**：必须先构建 `log_ret`、再按需写入 `source_place`、最后 **一次** `return log_ret`；不得在 `return {…}` 之后再写 `source_place`（该段为不可达代码，`log_entry` 将缺字段，撤销失效）。
- **Why**：本步先 `_advance_and_compute_reward` 再 `_fire`：恰超驻留阈值时 `scan` 会标 `resident` scrap，但若本步立即 `u_*` 取走同一 token，应撤销该 scrap；旧实现因库所名与变迁名后缀不一致导致永远不撤销。
- **Impact**：同一步内「驻留超时 + 从该腔室 unload」时不再误计 scrap/惩罚；导出或回放若重放 `fire_log`，新字段可选；缺 `source_place` 时仍回退 `t_name[2:]`（与修复前行为相同）。

### 可视化：LLC/LLD 显示加工进度、完成与 scrap 提醒 (2026-03-22)
- **What changed**：`visualization/petri_single_adapter.py` 的 `_time_to_scrap` 新增 `place_type=5`（`LLC/LLD`）分支，按 `process_time + 3*P_Residual_time - stay_time` 计算；`visualization/graphics/wafer_item.py` 与 `visualization/widgets/chamber_widget.py` 将 `place_type=5` 且 `proc_time>0` 按加工腔渲染，支持外圈进度、完成橙色、scrap 红色提示。
- **Why**：`pn_single` 已将 `LLC/LLD` 纳入驻留超时判定，UI 需要展示同口径状态，避免 LLC/LLD 一直显示为次级颜色且无进度反馈。
- **Impact**：级联回放中 `LLD` 等带工时的 LL 节点会出现外圈进度与颜色状态；`LLC` 若 `proc_time=0` 仍不显示进度环。

### single(cascade)：LLC/LLD 驻留阈值调整并移除 LLC 节拍门控 (2026-03-22)
- **What changed**：`cascade_routes_1_star.json` 中 `LLD.kind` 改为 `buffer`；构网预处理口径统一 `LLC/LLD` 均为 `buffer`，并保留其 stage `process_time` 覆盖。`pn_single.ClusterTool` 驻留 scrap 判定改为：普通腔室使用 `process_time + P_Residual_time`，`LLC/LLD` 使用 `process_time + 3*P_Residual_time`。同时移除 `llc_tm3_takt_interval` 及对应的 `get_action_mask/get_next_event_delta` LLC→TM3 节拍门控链路。
- **Why**：`LLC/LLD` 需要与当前工艺口径一致地参与驻留超时判定，并取消对 LLC 出片的额外节拍限制，避免与新驻留约束叠加形成不必要的放行动作屏蔽。
- **Impact**：`LLC/LLD` 超时将按 `resident` 类型计入 scrap；配置项 `llc_tm3_takt_interval` 不再生效；`route4_takt_interval` 与 `u_LP` 节拍逻辑保持不变。
- **Follow-up（实现对齐）**：此前同日文档已写移除 LLC→TM3 节拍，但 `solutions/Continuous_model/pn_single.py` 与 `data/petri_configs/env_config.py` 仍保留 `llc_tm3_takt_interval` 及门控；本次从代码删除该字段与 `ClusterTool` 中 LLC→TM3 节拍状态、`get_action_mask`/`get_next_event_delta`/`_fire` 相关分支。JSON 若仍含 `llc_tm3_takt_interval`，`PetriEnvConfig.load` 仅加载 dataclass 已知字段，该键被忽略。

### 可视化：`--debug` 变迁按钮按顺序两列排布 (2026-03-22)
- **What changed**：`visualization/transition_labels.py` 用 `build_transition_rows_two_columns` 替代按 `u_LP`/`u_PM*` 等规则配对；`control_panel.update_actions` 中级联与单设备均按变迁列表顺序每行两个按钮。
- **Why**：拓扑变迁名为 `u_*_TM2`、`t_TM2_*` 等时旧规则几乎无法配对，未命中项独占一行，界面退化为单列。
- **Impact**：仅影响调试面板布局；`action_id` 与使能逻辑不变。级联下 `u_LLD*` 仍附加 LLD 去向 tooltip。

### 可视化：级联运输库所名 `TM2`/`TM3` 与 `PetriSingleAdapter` 对齐 (2026-03-22)
- **What changed**：`visualization/petri_single_adapter.py` 将运输库所识别为 `d_TM*` **或** 库所名 `TM2`/`TM3`；级联下机械手晶圆列表从 `TM2`/`TM3`（及兼容 `d_TM2`/`d_TM3`）读取。`solutions/Continuous_model/construct/__init__.py` 从同级 `construct.py` 转发导出 `BasedToken`/`SuperPetriBuilder` 等，避免子包目录遮蔽原模块导致 `import ...construct` 失败。
- **Why**：固定拓扑 v3 中运输库所名为 `TM2`/`TM3`（见 `construct/build_topology.py`），旧逻辑仅匹配 `d_TM*`，`transport_buffers` 为空导致 TM2/TM3 不显示晶圆。
- **Impact**：级联可视化与统计面板的运输位/机械手状态恢复；依赖 `from solutions.Continuous_model.construct import BasedToken` 的代码在存在 `construct/` 子包时仍可导入。

### 构网预处理拆分到 `preprocess_config` + marks 构造迁移 (2026-03-22)
- **What changed**：新增 `solutions/Continuous_model/preprocess_config.py`，将 route/chamber 的 stage 覆盖、工时取整、清洗参数归并收敛为“每腔室一块”的预处理真源；新增 `solutions/Continuous_model/build_marks.py`，集中产出 `m0/md/ptime/capacity/marks/process_time_map`。`construct_single.py` 删除 `SingleModuleSpec/modules` 与内联 marks 构造路径，改为调用新模块。
- **Why**：去除构网阶段散乱 map 与重复合并，统一 `process_time_map` 口径并缩短 `build_net` 主路径。
- **Impact**：构网容量固定为 `LP/LP_done=100`、其他库所=1；`m0` 固定全 0，token 仅在 source 注入；`TM2/TM3` one-hot 固定硬编码。`ClusterTool._base_proc_time_map` 继续直接消费构网返回 `process_time_map`。

### `construct_single` 构网入口合并为 `build_net` (2026-03-22)
- **What changed**：删除 `build_single_device_net` 与 `_build_single_device_net_from_route_config`，合并为 `build_net`（形参、返回值与旧 `build_single_device_net` 相同）。`pn_single.ClusterTool` 与相关测试改为调用 `build_net`。
- **Why**：单一入口，去掉薄包装与私有实现分裂。
- **Impact**：直接 `from construct_single import build_single_device_net` 的代码须改为 `build_net`；校验失败时 `ValueError` 文案中的函数名为 `build_net`。

### `transition_id` 缓存与按 scope 选 hop 机械手 (2026-03-22)
- **What changed**：`build_topology` 拓扑版本升为 v3；`get_topology()` 返回 `transition_id`（键 `(TM2|TM3, 目标腔室)` → `t_*` 全名），并写入 `data/cache/transition_id_v3.npz`。`route_compiler_single.compile_route_stages` 用 `infer_cascade_transport_by_scope` 替代原 `infer_transport_robot` 的 hop 判定；`construct_single` 装配 `active_t_names` 时查 `transition_id`。
- **Why**：单一真源索引 `t_*`；hop 与 `_TM2_SCOPE`/`_TM3_SCOPE` 对齐。
- **Impact**：旧 `topology_v2.npz` 会被忽略并重建；双 scope 同时覆盖同一 hop 时取 **TM3**。

### 晶圆 route_queue 迁至 `build_route_queue.py` (2026-03-22)
- **What changed**：新增 `solutions/Continuous_model/build_route_queue.py` 中 `build_cascade_token_route_queue`；`construct_single` 仅调用该函数生成 `t_route_code_map`、`token_route_queue_template`、`token_route_plan_template`（内部仍调用 `route_compiler_single.build_token_route_plan`）。
- **Why**：路由队列构造与构网主流程分离，不塞进 `route_compiler_single` 的解析/归一化模块。
- **Impact**：行为与迁出前一致；扩展队列逻辑时改 `build_route_queue.py`。

## 2026-03-20

### 工序时长预处理迁入构网 (`construct_single`) (2026-03-20)
- **What changed**：默认填充与取整到 5 秒（原 `helper_function._preprocess_process_time_map` 口径）在 `construct_single.preprocess_process_time_map_for_single_net` 中执行；`build_single_device_net` 返回值新增/保证 `process_time_map`（与 `route_meta["chambers"]` 及库所 `processing_time` 一致）。`pn_single.ClusterTool` 传入未预处理的合并 `process_time_map`，构网后设 `_base_proc_time_map = info["process_time_map"]`，删除 `_preprocess_process_time_map` 与 `_apply_base_proc_times_to_marks_and_ptime`。
- **Why**：避免构网与运行时各算一遍预处理，消除 `marks`/`ptime` 与 `_base_proc_time_map` 漂移。
- **Impact**：直接调用 `build_single_device_net` 的代码可依赖返回的 `process_time_map` 作为权威腔室工时表。

### 移除单设备随机工序时长及配置项 (2026-03-20)
- **What changed**：删除按 episode 对腔室 `process_time` 做均匀随机缩放的能力：`PetriEnvConfig` 移除 `proc_rand_enabled`、`proc_time_rand_scale_map`；`chambers` 与 route `sequence` 不再读取 `proc_rand_scale`；`pn_single.ClusterTool` 移除 `_episode_proc_time_map` 及 `_refresh_episode_proc_time`、`_validate_episode_proc_time_map_consistency`、`_align_base_proc_time_map_with_route_chambers`，节拍/导出仅用 `_base_proc_time_map`（构网后取自 `info["process_time_map"]`）；`Env_PN_Single` / `visualization/main.py` 去掉对应构造参数；`train_single.py` 移除 `--proc-time-rand-enabled` 及 `train_single(...)` 相关形参；`export_inference_sequence` 的 `replay_env_overrides` 不再写入 `proc_rand_enabled`。
- **Why**：该功能不再使用，保留会造成配置与文档双轨维护。
- **Impact**：旧 JSON 中上述键名会被 `PetriEnvConfig.load` 忽略；需不确定性时请改用显式 `process_time_map` 或外部实验设计，而非运行时随机缩放。

### 移除 `ClusterTool._rebuild_token_pool` 与 `_token_pool` (2026-03-20)
- **What changed**：删除 `_rebuild_token_pool` 及 `_token_pool`；`_check_scrap` 改为直接扫描 `marks` 中 `CHAMBER` 内 token（与 `get_next_event_delta` 一致，均不依赖全局 token 列表）；删除已无引用的 `_token_remaining_time`。
- **Why**：`_fire` 已维护变迁语义与 `marks`；池结构仅服务 `_check_scrap`，可去掉冗余同步点。
- **Impact**：手动篡改 `marks` 后不再需要调用 `_rebuild_token_pool`；若外部曾依赖 `_token_pool` 引用需改为遍历 `marks`。

### 移除使能侧车与 `export_inference_sequence` 使能日志导出 (2026-03-20)
- **What changed**：删除 `Env_PN_Single._last_action_enable_info`、`_action_enable_info_from_mask` 及 `eval_mode` 下逐步填充；`export_inference_sequence.py` 不再生成 `results/eval_logs.json` / `eval_logs.md`，移除 `--results-dir` 与 `rollout_and_export` 的 `results_dir`；`petri_single_adapter` 的 `step_verbose` 仅打印步号、时间与执行动作名。
- **Why**：不在环境或独立文件中回传使能明细；策略与导出仍可从 TensorDict / `step` 的 `action_mask` 获知合法性。
- **Impact**：依赖 `action_enable_json` / `action_enable_md` 或 `_last_action_enable_info` 的外部流程需改为消费 mask。

### 移除 `get_enable_actions_with_reasons`（ClusterTool）(2026-03-20)
- **What changed**：删除 `pn_single.ClusterTool.get_enable_actions_with_reasons`；删除仅被其使用的 `_has_ready_chamber_wafers`、`_is_process_ready`。
- **Why**：消除与 `get_action_mask` 并行的第三套使能逻辑，避免口径漂移。
- **Impact**：使能以 `get_action_mask` 为准；`REASON_DESC` 仍保留于 `pn_single` 供解读历史日志（若需要）。

### 移除 `ClusterTool._get_enable_t`，由 `get_action_mask` 派生 reset 第二返回值 (2026-03-20)
- **What changed**：删除 `pn_single.ClusterTool._get_enable_t`；`reset()` 仍返回 `(None, enabled_transition_indices)`，第二项改为对 `get_action_mask` 前 `T` 维为 `True` 的下标排序列表；单测改为通过 `get_action_mask` 取使能变迁集合。
- **Why**：消除与 `get_action_mask` 并行的第二套 token 扫描使能逻辑，降低漂移风险；对外契约中「变迁使能」以 mask 为单一事实来源。
- **Impact**：直接调用 `_get_enable_t` 的外部代码需改为 `get_action_mask` 或消费 `reset()` 第二项；行为与删前以 mask 为准的语义对齐。

### 可选 LLC→TM3 出片节拍（与 u_LP 同口径）(2026-03-20)
- **What changed**：`PetriEnvConfig` 新增 `llc_tm3_takt_interval`（秒，`<=0` 关闭）。`pn_single.ClusterTool` 在间隔 `>0` 时对构网得到的 LLC→`d_TM3` 释放变迁施加节拍：使用 `takt_cycle_analyzer.build_fixed_takt_result` 生成周期序列；**首次 LLC→TM3 发射不因节拍被禁**，第二次起按序列取最小间隔；并写入 `get_action_mask` 与 `get_next_event_delta`。
- **Why**：路线 `1-2` 等实验需单独限制 TM3 侧 LLC 出片节奏，与 LP 发片节拍解耦。
- **Impact**：默认 `0`，行为与旧版一致。实验时在 JSON 中设置例如 `llc_tm3_takt_interval: 150` 即可。
- **How to use**：与 `single_route_name`（如 `1-2`）同配置文件增加 `llc_tm3_takt_interval`。

## 2026-03-19

### 修复 TM/LL 观测扩维以匹配新路线目标空间 (2026-03-19)
- **What changed**：`construct_single.py` 将 cascade 下 `TM2/TM3` 目标 one-hot 从“按目标组压缩”改为固定 8 维逐目标编码。`TM2` 目标集合为 `PM7/PM8/PM9/PM10/LLC/LLD/LP_done/LP`，`TM3` 目标集合为 `PM1/PM2/PM3/PM4/PM5/PM6/LLC/LLD`。同时 `pn.py` 中 `LLC/LLD` 观测从 4 维扩展为 6 维，新增 `in/out` 两维方向 one-hot（`TM3=进`，`TM2=出`），并由 `pn_single.py` 在构造观测时无副作用推断方向位。
- **Why**：路线更新后 TM 与 LL 的潜在去向显著增加，旧观测维度无法完整表达目标语义，容易造成策略混淆。
- **Impact**：cascade 模式观测维度变化（TM 总维度由 14 变为 24，LL 每个库所由 4 变为 6），旧模型输入层与新观测不兼容，需要重新训练或重建策略输入头。新增/更新 `tests/test_single_route_code.py` 覆盖 TM 固定 8 维编码与 LL in/out 语义。

### 新增 legacy 兼容路线组 2-*（映射旧 route_code 1/3/4/5）(2026-03-19)
- **What changed**：在 `data/petri_configs/cascade_routes_1_star.json` 中新增 `2-1/2-2/2-3/2-4` 四条路线，分别对应旧版 `construct_single.py` 的 cascade `route_code=1/3/4/5` 拓扑，并按当前 `routes.sequence` 格式补齐 stage 级 `process_time/cleaning_*` 字段。
- **Why**：解决“`route_code` 与 `single_route_name` 拓扑不一致”时的防御性校验冲突，便于在保留新工艺模板 `1-*` 的同时继续使用 legacy 拓扑口径。
- **Impact**：当需要与旧 `route_code=5` 行为对齐时，可直接使用 `single_route_name=2-4`（路径 `LP->PM7/PM8->PM9/PM10->LP_done`，不含 `LLC/LLD` stage）。同步新增测试覆盖该组合。

### 修复配置驱动路径 stage 工时被全局配置覆盖 (2026-03-19)
- **What changed**：`solutions/Continuous_model/pn_single.py` 在 `single_route_config` 模式下，初始化时先解析所选 route 的 stage 覆盖（`process_time/cleaning_duration/cleaning_trigger_wafers`；当时曾含 `proc_rand_scale`，已于 2026-03-20 移除）并合并进传入构网的 `process_time_map` 与 cleaning 映射；构网侧完成预处理与取整（2026-03-20 起为 `construct_single.preprocess_process_time_map_for_single_net`）；同时统一 `single_route_name` 解析结果，保证与构网阶段使用同一路线。
- **Why**：此前即使路径拓扑已切到 `1-1`，`_refresh_episode_proc_time()` 仍可能被全局 `process_time_map` 回写，导致腔室工时显示为 `cascade.json` 默认值。
- **Impact**：配置驱动下工时优先级变为 `route stage > process_time_map/chambers default`（并保持取整到 5 的倍数）；新增回归断言 `tests/test_single_route_config_driven.py::test_petri_env_config_loads_single_route_config_from_path`，覆盖 `1-1` 的 PM/LLD 工时和清洗参数。

### 新增级联路线配置文件（1-* 命名）(2026-03-19)
- **What changed**：新增 `data/petri_configs/cascade_routes_1_star.json`，将 6 条级联路线整理为统一配置，路线键名按 `1-1` 到 `1-6` 命名；起点统一 `LP`，终点统一 `LP_done`，并在各 stage 记录工序时间与清洗触发信息（如 `[88s/1片]`）。
- **Why**：便于按路线模板集中管理路径、阶段候选与工艺参数，降低后续手工抄写和维护成本。
- **Impact**：可直接作为 `single_route_config` 输入使用；`legacy.route_code_alias.cascade` 已映射到 `1-*` 路线键名。并且 `cascade.json` 新增 `single_route_config_path/single_route_name`，可仅改配置切换路线。

### 单设备构网新增配置驱动编译链（2026-03-19）
- **What changed**：`construct_single.py` 新增配置驱动路径编译模块（`route_compiler_single.py`），支持 `routes.sequence/repeat`、按 `robots.managed_chambers` 自动推断 stage 间 transport place、自动生成 `token_route_queue_template` 与 `t_route_code_map`。`build_single_device_net` 新增可选参数 `route_config/route_name`；`pn_single.py` 改为优先消费构网返回的 `route_meta`，避免运行时与构网元数据分叉。
- **Why**：旧实现对 `route_code` 与手写队列耦合过深，路径、并行候选与机械手归属变更成本高，且构网阶段存在重复分支与重复字符串处理。
- **Impact**：保持 `Place/PM/TM/pre/post/token.route_queue` 等旧概念兼容；当提供 `single_route_config` 时优先走配置驱动编译链，`route_code` 退化为兼容 alias 入口。新增 `tests/test_single_route_config_driven.py` 覆盖配置驱动路径与 repeat 队列编译。

### 文档体系重构：5 个主题主文档 + 兼容层 (2026-03-19)
- **What changed**：新增 5 个主题主文档：`overview/project-context.md`、`continuous-model/pn-single.md`、`visualization/ui-guide.md`、`training/training-guide.md`、`td-petri/td-petri-guide.md`；新增 `docs/deprecated/README.md` 迁移索引；将 `project.md`、`架构.md`、`continuous_solution_design.md`、`viz.md`、`td_petri.md`、`td_petri_modeling.md`、`Petri animate tool.md` 改为兼容跳转页；`docs/README.md` 重构为唯一导航入口。
- **Why**：原文档分散且重复，AI 与新成员难以在短时间定位“项目描述 / pn_single / 可视化 / 训练 / td_petri”权威说明。
- **Impact**：文档入口收敛为“1 个索引 + 5 个主文档 + deprecated 兼容层”；旧链接仍可用但不再承载规范内容。新增 `scripts/docs/validate_docs.py` 用于校验索引覆盖、章节完整性、链接可达与过时关键词。

### 文档与实现一致性修订 (2026-03-19)
- **What changed**：同步修订 `env_place_obs.md`、`continuous_solution_design.md`，移除对 `Env_PN_Single_PlaceObs` 和 `--place-obs` 的旧描述；同时修正 `check_release_penalty.py` 的必填 `--sequence`、`export_inference_sequence.py` 的固定输出路径 `seq/tmp.json`、以及 `train_single.py` 的正式入口说明。
- **Why**：避免文档继续指向已移除接口或错误输出路径，降低新脚本接入和回放验证时的误用风险。
- **Impact**：当前文档会明确区分历史接口与现行实现；单设备统一入口、验证脚本参数和导出产物位置都与代码保持一致。

### 单设备路线参数与节拍输入一致性严格校验 (2026-03-19)
- **What changed**：`solutions/Continuous_model/pn_single.py` 对 `device_mode/single_route_code` 增加严格规范化与合法性校验：`device_mode` 仅允许 `single/cascade`，`route_code` 强制转 `int` 并按模式校验（single: `0/1`，cascade: `1/2/3/4/5`），非法值直接抛 `ValueError`，不再静默回退。新增 `_episode_proc_time_map` 一致性守卫（必须与当前路线 `chambers` 完全一致）以及节拍分析前的 stage-工时一致性检查（越界/缺失会带 `device_mode/route_code/stage` 明细报错）。
- **Why**：防止路线配置与工时映射不一致时继续运行，导致节拍计算错误且难以定位（例如 route5 混入非本路线腔室）。
- **Impact**：配置错误将前移到初始化阶段暴露；合法配置行为不变。`route_code=\"5\"` 这类字符串输入会被规范化为整数后按 route5 执行。同步更新 `takt_cycle_analyzer.py` 的 stage 级错误上下文与 `tests/test_single_route_code.py` 覆盖用例。

## 2026-03-18

### 单设备 PM 第 9 维改为临近清洗分数（clip window=2）(2026-03-18)
- **What changed**：`solutions/Continuous_model/pn.py` 中 `PM.get_obs()` 第 9 维由 `remaining_runs_before_clean_norm` 改为 `near_cleaning_norm`，公式为 `near_cleaning_norm = (1-is_cleaning) * clip((2-r)/2, 0, 1)`，其中 `r=max(N-c,0)`、`N=max(1,cleaning_trigger_wafers)`、`c=max(0,processed_wafer_count)`。
- **Why**：将清洗相位特征压缩到“触发前 2 片窗口”，减少对绝对触发阈值的绑定，提升跨 `cleaning_trigger_wafers` 的泛化稳定性。
- **Impact**：PM 观测维度与拼接顺序保持不变（仍为 9 维且位置不变）；第 9 维语义更新为分段值 `r>=2 -> 0`、`r=1 -> 0.5`、`r=0 -> 1`，且 `is_cleaning=True` 时强制为 `0`。同步更新 `env_place_obs.md`、`continuous_solution_design.md`、`pn_api.md` 与测试用例。

### Cascade 新增 route_code=5（路线 D）(2026-03-18)
- **What changed**：`construct_single/pn_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=5`，路径为 `LP -> PM7/PM8(70) -> PM9/PM10(200) -> LP_done`；该路线不生成 `LLC/LLD/PM1/PM2/PM3/PM4` 相关变迁。
- **Why**：需要支持一条不经过中段缓冲与前段加工腔室的直连级联工艺模板（路线 D）。
- **Impact**：级联模式可通过 `single_route_code=1/2/3/4/5` 切换模板；`route_code=4` 的循环路线与 `route4_takt_interval` 行为保持不变，`route_code=5` 不启用该手动节拍门控。

### Cascade 新增 route_code=4 循环路线与手动节拍 (2026-03-18)
- **What changed**：`construct_single/pn_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=4`，路径为 `LP -> [PM7 -> PM8 -> LLC -> LLD] * 5 -> LP_done`；不再生成 `PM1/PM2/PM3/PM4/PM9/PM10` 相关变迁。新增 `route4_takt_interval`，并在 `takt_cycle_analyzer.py` 增加 `build_fixed_takt_result()` 供 route4 使用固定节拍门控。
- **Why**：需要一条严格串行、固定循环次数的 cascade 工艺模板，并允许 route4 单独配置发片节拍而不依赖自动节拍分析。
- **Impact**：级联模式可通过 `single_route_code=1/2/3/4` 切换模板；`route_code=4` 下 `u_LLD` 会按 token 路由队列在前 4 轮回 `PM7`、第 5 轮去 `LP_done`。`route4_takt_interval > 0` 时 `u_LP` 按固定间隔门控，`<=0` 不门控。

## 2026-03-17

### Ultra 并行采样子环境终止即自重置修复 (2026-03-17)
- **What changed**：`env_single.py` 的 `FastEnvWrapper` 终止判定改为 `scrap/finish/terminated` 联合触发；并新增 `net._last_state_scan.is_scrap` 兜底，避免 `stop_on_scrap=False` 时 scrap 标志在并行采样链路中丢失。`VectorEnv` 下仅对触发终止的子环境执行自动 `reset`。
- **Why**：修复并行 rollout 中“某个子环境发生 scrap 但未被视为终止，导致不重置并持续污染轨迹”的问题。
- **Impact**：`collect_rollout_ultra()` 的 `dones` 现在会正确覆盖 scrap/finish 终止；GAE 与回报截断按子环境独立生效，不会影响其他并行子环境继续采样。
- **Examples**：`--rollout-n-envs 8` 训练时，任一子环境触发 scrap 或 finish 后会立刻进入下一回合起点，其余 7 个子环境保持当前轨迹继续运行。

### 单设备同步取片 resident scrap 撤销修复 (2026-03-17)
- **What changed**：`pn_single.step()` 在非 WAIT 分支保留“先 `_advance_and_compute_reward` 再 `_fire`”顺序；新增同步撤销判定：若本步 `u_*` 已取走与 `scan_info.scrap_info` 同 `token_id` 且同源腔室的 resident wafer，则撤销本步 scrap（不终止、不追加 `scrap_penalty`）。
- **Why**：修复“晶圆在驻留边界/超界时，本步已同步取片却仍被先判 scrap”导致的误终止与误扣分。
- **Impact**：行为仅影响单设备非 WAIT 同步取片场景；WAIT 分支、qtime 统计、mask/obs 构造保持不变。

### 可视化模型推理切换到 `env.net.get_obs()` 直连 (2026-03-17)
- **What changed**：`visualization/main.py` 的 Model 推理入口（单动作/并发）统一改为直接读取 `env.net.get_obs()`，移除对 `Env_PN_Single._build_obs()` 的依赖。
- **Why**：单设备观测已统一由 `pn_single` 构网层输出，继续依赖环境私有方法会导致接口漂移与运行时错误。
- **Impact**：可视化界面的 `Model Step/Auto` 在 single/cascade 模式下不再触发 `_build_obs` 缺失报错；外部脚本若仍调用 `_build_obs()` 需迁移到 `env.net.get_obs()`。

### train_single 收敛为 Ultra-only 并修复 CPU 卡顿路径 (2026-03-17)
- **What changed**：`train_single.py` 移除 `single collector`、`blame`、`benchmark` 相关训练入口与代码分支，训练主循环固定为 ultra rollout；GAE 在 CPU 上强制走 eager（仅 CUDA 尝试 compile）。
- **Why**：简化分支可降低维护成本；同时避免本地 CPU 训练触发 compile 首次编译开销，出现“卡住/长时间无响应”。
- **Impact**：CLI 参数收敛为 `--device/--compute-device/--checkpoint/--rollout-n-envs`（后续已移除 `--proc-time-rand-enabled`）；训练行为更可预测，CPU 本地调试路径更稳定。

### train_single 更新阶段极限优化（Batched PPO + 长轨迹连续采样）(2026-03-17)
- **What changed**：`train_single.py` 更新路径改为纯 `dict[tensor]`，移除 rollout->TensorDict->tensor 往返；PPO 更新由 `epoch + minibatch` 双层循环改为“每个 epoch 单次大 batch forward”；策略更新的 `log_prob/entropy` 改为 fused `masked log_softmax` 计算（不再构造 `MaskedCategorical`）；GAE 改为 `[T,N]` 计算并采用 `torch.compile` 优先、失败自动回退 eager。
- **Why**：训练瓶颈位于 update 阶段，原实现小 batch 多次前向与 Python 循环开销过高，GPU 利用率偏低。
- **Impact**：形成“CPU rollout + 单次 CPU->GPU 搬运 + GPU 全并行 update”数据流；ultra/single collector 均支持跨 batch 连续轨迹（不强制 reset），长 episode（1000+ steps）训练稳定性更好。

### 单设备 Ultra Rollout 高性能采样链路 (2026-03-17)
- **What changed**：`env_single.py` 新增 `FastEnvWrapper` 与 `VectorEnv`，统一 `reset()/step()` 为纯数组接口；`train_single.py` 新增 `collect_rollout_ultra()`（纯 tensor 预分配、CPU rollout、手写 masked 采样），并将训练主循环默认 collector 切到 `ultra`（支持 `--collector ultra|single`、`--rollout-n-envs`、`--rollout-device`）；`pn_single.py` 新增 `step_core_numba/step_core_batch_numba`（纯 numpy 结构可使能核心）。
- **Why**：原 rollout 路径依赖 TensorDict clone / dispatch 与频繁对象构造，采样吞吐显著低于 `env.step` 的 CPU 峰值。
- **Impact**：可用 `python -m solutions.Continuous_model.train_single --benchmark-ultra --benchmark-envs 1,8` 直接对比 baseline/ultra 的 steps/sec；训练默认走 ultra 采样，若需要旧逻辑可显式传 `--collector single`，`--blame` 打开时也会自动回退 single collector。

### 单设备使能判定切到 token 扫描快路径 (2026-03-17)
- **What changed**：`pn_single.get_enable_t` 从“逐变迁两阶段扫描”改为“先检查 `u_LP`，再扫描 token 生成 `u_*/t_*` 候选”；新增运行时 token 池与变迁索引缓存；`_check_scrap` 改为按 token 剩余时间判定（`remaining < -P_Residual_time`）。
- **Why**：`get_enable_t` 是 `step` 热路径，逐变迁做重复结构性判断与目标解析开销较高；token 扫描在单臂场景下可减少无效判定。
- **Impact**：单设备当前按“单臂 + 非 FIFO + unit-capacity（除 LP/LP_done 外）”临时策略运行；双臂锁定规则不再作为默认路径。

## 2026-03-16

### 单设备 t_* 路由改为 token 队列门控 (2026-03-16)
- **What changed**：`pn_single/construct_single` 新增 token 路由队列模板（`route_queue + route_head_idx`）与 `t_*` 路由码映射；`pn_single` 使能/mask 路径的路由判定改为读取运输位队首 token 的当前队头门控。
- **Why**：`get_enable_t` 是热路径，原 `where + pre_color` 的颜色切片判定在每步都有额外矩阵开销；改为队头码匹配可减少分支与切片成本，并保持路径语义。
- **Impact**：仅 `t_*` 受路由门控（支持 `-1` 通配、单码、多码集合）；`u_*` 不再做路由门控，但 token 每次 fire 仍推进一次队头（`u_*` 步通常对应 `-1` 占位）。

## 2026-03-14

### 单设备 wait 截断新增运输完成事件 (2026-03-14)
- **What changed**：`pn_single.get_next_event_delta()` 在保留“加工完成”事件的同时，新增 `d_TM*` 运输位“达到 `T_transport`”作为关键事件；`step(wait)` 的长 WAIT 现在会被运输完成时刻截断。
- **Why**：当晶圆已在运输位停留接近运输完成时，继续整段长 WAIT 会跨过关键放片决策点，不利于调度策略及时响应。
- **Impact**：除 `WAIT_5s` 仍固定 5 秒外，其他 WAIT 的 `actual_wait=min(requested_wait,next_event_delta)` 中 `next_event_delta` 候选扩展为“加工完成 + 运输完成 + 清洗完成”，长 WAIT 截断会更及时。

## 2026-03-10

### train_single 新增 --blame 参数 (2026-03-16)
- **What changed**：`train_single.py` 新增命令行参数 `--blame`。传入时在 episode 结束后执行二次追责（`blame_release_violations` 回填惩罚）；不传则不进行二次追责。
- **Why**：支持按需开启或关闭二次追责，便于对比实验与调试。
- **Impact**：默认不传 `--blame` 时不执行追责；需与旧版“始终追责”行为一致时请显式加上 `--blame`。

### 单设备 Q-Time 违规统计接入 (2026-03-10)
- **What changed**：`pn_single` 新增 `_check_qtime_violation`，在 `step()` 的 wait/fire 两条时间推进路径后检查运输位（type=2）超时，并更新 `qtime_violation_count`。
- **Why**：可视化面板已提供 `Q-TIME` 指标卡片，需要单设备后端提供稳定一致的违规计数来源。
- **Impact**：Q-Time 违规按 `stay_time > D_Residual_time` 判定，同一 wafer 仅首次违规计数 1 次；该检测不引入奖励惩罚，仅影响统计展示。

### Cascade 观测纳入 LLC/LLD (2026-03-10)
- **What changed**：`Env_PN_Single` 在 `device_mode=cascade` 下将 `LLC/LLD` 纳入腔室观测列表；`LLC/LLD` 使用 4 维核心特征（`occupied/processing/done_waiting_pick/remaining_process_time_norm`），其余 PM 继续使用 9 维特征。
- **Why**：需要让策略在级联流程中直接感知 `LLC/LLD` 状态，避免关键缓冲/交接位信息缺失。
- **Impact**：仅 cascade 模式 observation 维度发生变化；single 模式维度保持不变（`route_code=0 -> 32`，`route_code=1 -> 41`）。

## 2026-03-09

### Cascade 新增 route_code=3 (2026-03-09)
- **What changed**：`pn_single/construct_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=3`，路径为 `LP -> PM7/PM8(70) -> LLC(0) -> PM1/PM2(300) -> LLD(70) -> LP_done`。
- **Why**：支持“保留 LLD 清洗/缓冲段，但去掉 PM9/PM10 末段”的级联工艺模板，满足新工艺路径验证需求。
- **Impact**：级联模式可通过 `single_route_code=1/2/3` 切换三套模板；`route_code=3` 下不再生成 `PM9/PM10` 相关变迁动作。

### Cascade 新增 route_code=2 (2026-03-09)
- **What changed**：`pn_single/construct_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=2`，路径为 `LP -> PM7/PM8(70) -> LLC(0) -> PM1/PM2(300) -> LLD(70) -> PM9/PM10(200) -> LP_done`。
- **Why**：支持新增级联工艺路线，并保留现有 `route_code=1`（`PM1/2/3/4`）兼容既有流程。
- **Impact**：级联模式可通过 `single_route_code=1/2` 切换两套模板；`route_code=2` 下不再生成 `PM3/PM4` 对应变迁动作。

### 追责简要报告增强 (2026-03-09)
- **What changed**：`check_release_penalty.py` 输出新增 `penalized_actions_report`，覆盖所有被惩罚动作，包含 `action_name + step + reason + where_penalized`。
- **Why**：便于快速定位“哪一步、哪个动作、因哪个站点超容量而被 second-pass 惩罚”。
- **Impact**：可直接在结果文件里查看完整追责清单，无需手工对照 `blame_raw` 与 `step_records`。

### 释放违规追责限定 (2026-03-09)
- **What changed**：`blame_release_violations` 仅追责 `u_LP`、`u_LLC`、`u_LLD` 释放动作，不再追责 `u_PM7`、`u_PM2` 等从加工腔卸载的动作。
- **Why**：释放违规应归因于「从 LP/LLC/LLD 释放晶圆」的决策，不应惩罚 PM 腔室的取片动作。
- **Impact**：导出的 `release_penalty.steps` 将只包含 u_LP/u_LLC/u_LLD 动作所在步，u_PM7/u_PM2 所在步不再计入。

### 推理序列奖励报告 (2026-03-09)
- `export_inference_sequence.py` 在导出的 JSON 最前面增加 `reward_report` 区块。
- 包含 `scrap_penalty`、`release_penalty`、`idle_timeout_penalty` 的次数及触发步数（`count`, `steps`）。
- 回放逻辑仅读取 `sequence`、`replay_env_overrides`，兼容现有 `planB_sequence.json` 消费者。

### What changed
- 单设备环境观测已统一：`Env_PN_Single` 合并 `Env_PN_Single_PlaceObs`，默认采用库所中心 obs 构造（`LP -> TM(d_TM1) -> PM*`）。
- 删除环境类 `Env_PN_Single_PlaceObs`，单设备仅保留 `Env_PN_Single` 一个入口。
- `train_single.py` 与 `export_inference_sequence.py` 移除 `--place-obs` 参数与对应分支逻辑。
- `pn_single` 新增 `single_device_mode`，支持 `single/cascade` 双模板路径；`cascade` 路径迁移为 `LP->PM7/PM8->LLC->PM1/PM2/PM3/PM4->LLD->PM9/PM10->LP_done`。
- 可视化 `visualization/main.py` 新增 `--device`（兼容 `--device-mode`），并在 `cascade` 模式下改为调用 `pn_single/env_single`，不再走 `pn.py`。

### Why
- 旧版 wafer-list obs 已不再使用，继续保留双环境实现会增加维护成本与行为漂移风险。
- 需要统一逻辑层到 `pn_single`，同时保留原单设备路径并新增级联模板，减少双后端维护复杂度。

### Impact / How to use
- 训练与推理导出在 single 模式下统一使用 `Env_PN_Single`，无需再切换 obs 形态。
- 若外部脚本仍 import `Env_PN_Single_PlaceObs` 或传入 `--place-obs`，需要迁移为统一入口。
- route 感知维度保持一致：`route_code=0` 为 `32` 维，`route_code=1` 为 `41` 维。
- 新增设备参数统一入口：`--device single|cascade`（`export_inference_sequence.py` 保留 `--device-mode` 兼容别名）。
- 若需要旧单设备路径，使用 `single_device_mode=single`；级联路径使用 `single_device_mode=cascade`。

### Debug fix (2026-03-09)
- 修复级联模式下 `u_LLD/u_PM9/u_PM10` 的机械手显示归属：动作后晶圆应显示在 `TM2` 而非 `TM3`。
- 修复级联模式 `PM2` 的腔室类型标记：`PM2` 现在按加工腔处理，恢复外圈加工进度条显示。
- 修复级联模式中 `LLD` 有晶圆时 `u_PM1/u_PM2/u_PM3/u_PM4` 被提前禁用的问题：允许先取片到 `d_TM1`，实际落片仍由 `t_LLD` 容量约束控制。
- 修复 wait 崩溃：当 `next_event_delta=0`（关键事件与当前时刻重合）时，`step(wait)` 不再计算 `actual_dt=0`，改为回退到最小 5s 推进，避免触发断言导致程序崩溃。

### Wait 50s 截断修复 (2026-03-09)
- 修复：当有晶圆进入 LP_done 时，WAIT 50s 被错误截断为 5s 的问题。原逻辑用 `_has_completed_wafers()`（任意晶圆在 LP_done）截断 wait；现改为仅在 episode 全部完成（`len(LP_done) >= n_wafer`）时才截断，允许在部分完工时执行完整 WAIT 50s。

### Cascade 双运输位重构 (2026-03-09)
- 级联模式构网由单一 `d_TM1` 改为双运输位 `d_TM2` + `d_TM3`：TM2 负责 LP/PM7/PM8/LLD/PM9/PM10 取放与 LLC 放片，TM3 负责 LLC/PM1–4 取放与 LLD 放片。
- `pn_single` 新增 `_transport_for_u_source()`、`_transport_for_t_target()`，使能与 dwell 检查按对应运输位执行。
- `get_next_event_delta` 排除所有 `d_*` 库所，不再仅排除 `d_TM1`。
- `env_single._extract_tm_features()` 遍历所有 `d_*` 运输位聚合观测。
- 可视化 `petri_single_adapter` 支持级联：运输位识别 `d_TM*`，TM2/TM3 晶圆来自 `d_TM2`/`d_TM3`。
