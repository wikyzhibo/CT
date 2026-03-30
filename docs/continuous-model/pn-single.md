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
  - 关键参数: `device_mode=cascade`, `robot_capacity`, `route_code`, `process_time_map`（可选）
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
2. 构网固定走“固定拓扑 + 配置驱动 route 编译链”（`load/build static topology -> parse/normalize -> route IR -> token route -> dynamic marks/token`）；`route_code` 仅用于兼容 alias 选择，不再触发 legacy 手写拓扑分支。
3. 固定拓扑包含 `LP, PM1-PM10, LLC, LLD, LP_done, TM2, TM3`。
4. 变迁命名统一为 `u_src_dst` 与 `t_src_dst`；其中 `u_src_dst` 的 `dst` 是 `TM2/TM3`，`t_src_dst` 的 `src` 是 `TM2/TM3`。
5. 固定拓扑除 `results/topology_cache/topology_vN.npz` 外，另存 `results/topology_cache/transition_id_vN.npz`：`transition_id` 映射 **键** `(TM2 或 TM3, 目标腔室)` → **值** 对应 `t_*` 全名。`get_topology()` 返回的 dict 必含 `transition_id`（与缓存同版本）。
6. `compile_route_stages` 相邻 stage 的机械手由 `build_topology.infer_cascade_transport_by_scope` 判定：左、右 stage 的 candidates 须同时落在 `_TM2_SCOPE` 或 `_TM3_SCOPE` 之一；若两套 scope 同时满足则取 **TM3**。
7. 不再兼容旧命名（`d_TM*`、`t_PM*` 等）；运行时仅消费新命名。
8. 固定动作空间下，未被当前 route 使用的变迁必须在 `get_action_mask` 中恒为 0。
9. 当通过 `PetriEnvConfig.load(json)` 加载且 json 中提供 `single_route_config_path` 时，会自动读取该文件并填充 `single_route_config`。
10. WAIT 掩码规则：存在加工完成待取片晶圆时，仅允许短 WAIT（5s）。
11. 导出脚本的 `--out-name` 参与文件命名：`results/action_sequences/<out_name>.json`（非法字符会替换为 `_`）。
12. `check_release_penalty.py` 未设置 `--sequence` 时不能执行。
13. 旧观测分支（place-obs）不再作为当前实现接口。
14. 配置驱动路径启用时，`construct_single.build_net` 通过 `preprocess_config.py` 先构建“每腔室一块”的预处理真源（包含 stage 覆盖后的 `process_time/cleaning_*`）；`build_marks.py` 仅消费该真源构造 place，返回的 `process_time_map` 只来自该预处理真源并与 `marks` 一致，`ClusterTool._base_proc_time_map` 直接取自该字段。
15. 构网容量口径固定：`source/sink` 容量恒为 `100`，其余库所容量恒为 `1`；`m0` 恒为全 0，token 仅在构网末尾注入 source place。
16. 级联模式 `TM2/TM3` 的目标 one-hot 映射在 `build_marks.py` 中固定硬编码为 8 维目标集合，不再按 route 动态构造。
17. 对 `_cascade_round_robin_pairs` 中的源，`u_*` 使能采用“从当前指针起的循环扫描”：在 `ClusterTool._first_receivable_parallel_target` 中按 `self._cascade_round_robin_next[source]` 在并行候选腔室序列上循环，返回**第一个**可接收目标（未满且非清洗）；若一圈内均不可接收，则该源的 `u_*` 不可使能。非并行源（单一 `u_targets` 候选）仍只校验该单一候选。
18. `_fire` 对 `_cascade_round_robin_pairs` 中的源：在 `pop_head` 之前将 `_cascade_round_robin_next[source]` 设为当前 `_first_receivable_parallel_target(source)`，与 `get_action_mask` 所用判定一致，使 TM2/TM3 上并行 `t_*` 的 `_cascade_round_robin_next.values()` 筛选与本次实际去向一致。该类源发射 `u_*` **不**调用 `_advance_round_robin_after_u_fire`；**round-robin 指针仍在 `t_*` 将晶圆放入其并行候选中的某一腔室后推进**（与既有规则一致）。
19. 对共享同一并行目标集的多个上游源（如 `PM7` 与 `PM8` 同时指向 `PM9/PM10`），`t_*` 发射后的 round-robin 推进必须基于该 token 最近一次 `u_*` 的真实 source（`token.last_u_source`）；禁止按“第一个匹配目标集的 source”推进，否则会导致某些 source 指针长期不动并引发目标腔室偏置。
20. 级联观测中 `TM2/TM3` 的目标 one-hot 采用固定 8 维逐目标编码（不再按目标组压缩）；`LLC/LLD` 观测由 4 维扩展为 6 维，新增 `in/out` 两维方向 one-hot。当前版本将 `LLC/LLD` 的 `in/out` 方向位临时固定为全 0。
21. 驻留 scrap 口径：普通腔室阈值为 `process_time + P_Residual_time`；`LLC/LLD` 阈值为 `process_time + 3 * P_Residual_time`。超过阈值会按 `resident` 类型计入 scrap 判定。
22. 同一步内先推进时间再发射变迁：若 `_advance_and_compute_reward` 已对本步标 `resident` scrap，且本步随后发射的 `u_*` 从**同一腔室**取走**同一 `token_id`** 的晶圆，则 `step` 会撤销该 scrap（不计入 `scrap_count`/惩罚）。库所匹配以 `_fire` 写入 `fire_log` 的 `source_place` 为准，**禁止**用 `t_name` 去掉前缀 `u_` 后的整段作为腔室名（级联命名为 `u_PM7_TM2` 等时该段含运输后缀，与 `p.name` 不一致）。
23. `get_action_mask` 在 d_TM 分支中，当 `tok_gate` 为并行集合（`tuple/frozenset`）时，仅放行后置目标位于 `_cascade_round_robin_next.values()` 的 `t_*`；不在该集合中的目标直接屏蔽。
24. 当 `tok_gate` 为并行集合时，若 token 上可解析到 `last_u_source` 且该 source 受 round-robin 管理，则 `get_action_mask` 必须仅放行 `target == _cascade_round_robin_next[last_u_source]` 的单一 `t_*`；只有无法解析 source 时才允许退回 values 级别筛选。该规则用于保证“按 source 严格轮转”，避免 `PM7/PM8` 共享 `PM9/PM10` 目标集时同一步同时放行两个目标。
25. `ClusterTool` 初始化时，`_cascade_round_robin_pairs` 对 `route_meta.u_targets` 中 **`len(targets) >= 2` 的源** 使用**完整** `tuple(targets)` 作为并行候选序列；**禁止**再丢弃 `LLC`/`LLD` 等非 `PM*`、`LP_done` 的下游名，否则 `values()` 可能永不包含这些库所，规则 22 会误屏蔽对应 `t_*`（典型：路线 `1-5` 上 PM7→LLC|LLD）。
26. 节拍门控**仅**作用于 `u_LP`（含 `route_code==4` 时的 `route4_takt_interval` 与 `analyze_cycle` 产出的周期节拍）；**不对** LLC→TM3（`u_LLC*`）施加时间间隔门控。`get_action_mask` 与 `get_next_event_delta` **不会**因 LLC 出片间隔而屏蔽或推迟；`PetriEnvConfig` **不包含** `llc_tm3_takt_interval`。
27. 双臂模式（`PetriEnvConfig.dual_arm=True`）启用 swap 操作。`robot_capacity` 始终为 1，不因双臂改变。`swap_duration` 固定为 10s。
28. 双臂模式 `get_action_mask` 对 `t_*` 变迁的 PM 目标：**仅检查**（a）腔室内晶圆是否加工完成（空腔室直接通过）、（b）机械手晶圆路由匹配（`route_gate_allows`）、（c）运输完成（TM 上 token 的 `stay_time >= proc_time`）。**不检查** `_is_struct_enabled` 容量约束。仍检查清洗和 round-robin。对非 PM 目标（LLC/LLD/LP_done 等）沿用单臂逻辑（含 `_is_struct_enabled`）。
29. 双臂模式 `_is_next_stage_available` 和 `_fire` u_* round-robin 同步：**跳过** `_first_receivable_parallel_target`（含容量检查），改用 `_first_parallel_target_dual_arm`（仅检查非清洗）。单候选源也跳过容量检查。
30. 双臂模式 `_fire` 对 `t_*` 变迁：在执行时调用 `_is_swap_eligible(pst_place)` 判定是否 swap。条件：`_dual_arm=True`、目标 `is_pm`、满载、head wafer 加工完成、非清洗中。swap 时原子交换 TM 与 PM 的 token（`m` 不变），触发 `_on_processing_unload`（清洗计数），推进 round-robin。`step` 在 `_advance_and_compute_reward` 之前计算 swap 决策，确保时长（10s）与 `_fire` 行为一致。

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
- 2026-03-30: A 方案级联路线配置新增 `4-8/4-9` 双子路径模式：路线条目可携带 `subpaths`、`wafer_type_alloc`、`takt_policy`，其中 `4-8` 额外支持 `takt_stages_override=[3000,180]` 与 `lp_release_pattern=["path1","path2","path2","path2","path2"]`。A 方案 `ClusterTool` 的 `u_LP` 门控改为 LP token 负 `stay_time` 倒计时，并在双类型场景按类型队首独立判定可发；若无 pattern 且两类型同刻可发，则随机选择一种发片。
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
