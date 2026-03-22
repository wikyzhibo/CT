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
5. 导出脚本 `export_inference_sequence.py` 生成 `seq/tmp.json`（含 `sequence`、`replay_env_overrides`、`reward_report` 等）。

## Interfaces
- 环境接口:
  - 类: `solutions.Continuous_model.env_single.Env_PN_Single`
  - 关键参数: `device_mode=cascade`, `robot_capacity`, `route_code`, `process_time_map`（可选）
  - 配置驱动参数（必需构网输入）: `single_route_config`, `single_route_config_path`, `single_route_name`
- 训练入口:
  - `python -m solutions.Continuous_model.train_single --device cascade --rollout-n-envs 1`
  - 关键参数: `--device`, `--compute-device`, `--checkpoint`, `--rollout-n-envs`
- 推理导出入口:
  - `python -m solutions.Continuous_model.export_inference_sequence --device cascade --model <model_path>`
  - 当前 action sequence 输出固定为 `seq/tmp.json`
  - 导出的 `replay_env_overrides` 会携带 `single_route_name`，并在可用时携带 `single_route_config`，用于可视化回放时保持与导出一致的构网路线
- 二次释放惩罚验证入口:
  - `python -m solutions.Continuous_model.check_release_penalty --sequence <json_name> --results-dir results`
  - `--sequence` 必填，脚本按仓库根目录 `seq/<json_name>` 解析。

## Behavior Rules
1. 构网输入严格校验：`build_net` 必须提供 `route_config` 且 `device_mode` 必须是 `cascade`，否则直接报错。
2. 构网固定走“固定拓扑 + 配置驱动 route 编译链”（`load/build static topology -> parse/normalize -> route IR -> token route -> dynamic marks/token`）；`route_code` 仅用于兼容 alias 选择，不再触发 legacy 手写拓扑分支。
3. 固定拓扑包含 `LP, PM1-PM10, LLC, LLD, LP_done, TM2, TM3`。
4. 变迁命名统一为 `u_src_dst` 与 `t_src_dst`；其中 `u_src_dst` 的 `dst` 是 `TM2/TM3`，`t_src_dst` 的 `src` 是 `TM2/TM3`。
5. 固定拓扑除 `data/cache/topology_vN.npz` 外，另存 `data/cache/transition_id_vN.npz`：`transition_id` 映射 **键** `(TM2 或 TM3, 目标腔室)` → **值** 对应 `t_*` 全名。`get_topology()` 返回的 dict 必含 `transition_id`（与缓存同版本）。
6. `compile_route_stages` 相邻 stage 的机械手由 `build_topology.infer_cascade_transport_by_scope` 判定：左、右 stage 的 candidates 须同时落在 `_TM2_SCOPE` 或 `_TM3_SCOPE` 之一；若两套 scope 同时满足则取 **TM3**。
7. 不再兼容旧命名（`d_TM*`、`t_PM*` 等）；运行时仅消费新命名。
8. 固定动作空间下，未被当前 route 使用的变迁必须在 `get_action_mask` 中恒为 0。
9. 当通过 `PetriEnvConfig.load(json)` 加载且 json 中提供 `single_route_config_path` 时，会自动读取该文件并填充 `single_route_config`。
10. WAIT 掩码规则：存在加工完成待取片晶圆时，仅允许短 WAIT（5s）。
11. 导出脚本的 `--out-name` 当前不参与文件命名，仅保留兼容。
12. `check_release_penalty.py` 未设置 `--sequence` 时不能执行。
13. 旧观测分支（place-obs）不再作为当前实现接口。
14. 配置驱动路径启用时，`construct_single.build_net` 通过 `preprocess_config.py` 先构建“每腔室一块”的预处理真源（包含 stage 覆盖后的 `process_time/cleaning_*`）；`build_marks.py` 仅消费该真源构造 place，返回的 `process_time_map` 只来自该预处理真源并与 `marks` 一致，`ClusterTool._base_proc_time_map` 直接取自该字段。
15. 构网容量口径固定：`source/sink` 容量恒为 `100`，其余库所容量恒为 `1`；`m0` 恒为全 0，token 仅在构网末尾注入 source place。
16. 级联模式 `TM2/TM3` 的目标 one-hot 映射在 `build_marks.py` 中固定硬编码为 8 维目标集合，不再按 route 动态构造。
17. `u_*` 使能采用“指针目标单点判定”：先按 round-robin 指针选定下一目标，再仅校验该目标可接收（未清洗且未满）；若该指针目标不可接收，则该 `u_*` 直接不可使能，不再回退扫描其它候选目标。
18. `_fire` 阶段不再执行目标重选；并行目标的放行完全由掩码阶段决定，`_fire` 仅推进指针与执行状态更新。
19. 级联观测中 `TM2/TM3` 的目标 one-hot 采用固定 8 维逐目标编码（不再按目标组压缩）；`LLC/LLD` 观测由 4 维扩展为 6 维，新增 `in/out` 两维方向 one-hot。当前版本将 `LLC/LLD` 的 `in/out` 方向位临时固定为全 0。
20. `PetriEnvConfig.llc_tm3_takt_interval > 0` 时，对构网得到的 **LLC→`d_TM3`** 释放变迁（`("LLC","d_TM3")` 对应的 `u_*`）施加节拍门控：口径与 `u_LP` 的 `_takt_required_interval` 一致（**首次发射不因节拍被禁**，第二次起按 `build_fixed_takt_result(interval)` 的 `cycle_takts` 取最小间隔）；`<=0` 为默认关闭。门控同步作用于 `get_action_mask` 与 `get_next_event_delta`。
21. `get_action_mask` 在 d_TM 分支中，当 `tok_gate` 为并行集合（`tuple/frozenset`）时，仅放行后置目标位于 `_cascade_round_robin_next.values()` 的 `t_*`；不在该集合中的目标直接屏蔽。

## Examples
- 正例:
  - 单设备训练（CPU rollout，多环境采样）
  - 单设备推理序列导出后，用可视化回放 JSON
- 反例:
  - 继续使用旧版观测切换参数
  - 假设导出路径为历史 `action_series/<name>_<timestamp>.json`

## Edge Cases
- `train_single.py` 最佳模型当前写入 `models/tmp.pt`，并会在 `saved_models/single_<timestamp>/` 保留备份。
- `check_release_penalty.py` 的 `--sequence` 参数若传完整路径，会被额外拼接到 `seq/`，建议只传文件名。

## Related Docs
- `../overview/project-context.md`
- `../training/training-guide.md`
- `../visualization/ui-guide.md`
- `../deprecated/continuous-solution-design.md`

## Change Notes
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
- 2026-03-19: `PetriEnvConfig.load` 支持 `single_route_config_path` 自动装载外部路线配置文件；`cascade.json` 可直接通过 `single_route_name` 切换目标路线而不改代码。
- 2026-03-19: `construct_single.py` 新增配置驱动构网通路，支持 `routes.sequence/repeat`、机器人自动推断 transport place、自动编译 `token.route_queue` 与 `t_route_code_map`；`pn_single.py` 优先消费构网返回的 `route_meta`，避免与运行时路由元数据不一致。
- 2026-03-19: 建立 pn_single 主文档，统一单设备入口、脚本接口与行为规则说明。
- 2026-03-19: 修复 `1-6` 路径中 `PM3` 满载时 `u_PM1` 仍被使能的问题：`get_action_mask`（及当时并行的使能列表实现）在使用 `route_target` 时新增目标容量与清洗校验，禁止绕过 `_select_target_for_source` 的接收约束。
