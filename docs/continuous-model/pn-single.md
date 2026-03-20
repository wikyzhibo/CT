# PN Single Guide

## Abstract
- What: 本文档定义单设备连续时间 Petri 网（pn_single）在当前仓库中的架构、接口与行为约束。
- When: 修改 `pn_single.py`、`env_single.py`、`train_single.py`、导出/追责脚本前必须先读。
- Not: 不覆盖并发双机械手 `pn.py` 的完整实现细节。
- Key rules:
  - 单设备统一入口是 `Env_PN_Single`。
- 旧版 place-obs 环境入口与观测切换参数已移除。
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
1. `construct_single.py` 根据 `device_mode + route_code` 构建网络结构与元数据。
2. `ClusterTool` 维护标识、使能判定、时间推进、reward 计算、违规统计。
3. `Env_PN_Single` 封装 TorchRL 风格 `reset/step`，并暴露 `action_mask`。
4. 训练脚本 `train_single.py` 调用 `collect_rollout_ultra` 执行 CPU rollout + batched PPO update。
5. 导出脚本 `export_inference_sequence.py` 生成 `seq/tmp.json` 与动作使能日志。

## Interfaces
- 环境接口:
  - 类: `solutions.Continuous_model.env_single.Env_PN_Single`
  - 关键参数: `device_mode`, `robot_capacity`, `route_code`, `proc_time_rand_enabled`
  - 配置驱动参数（可选）: `single_route_config`, `single_route_config_path`, `single_route_name`
- 训练入口:
  - `python -m solutions.Continuous_model.train_single --device single --rollout-n-envs 1`
  - 关键参数: `--device`, `--compute-device`, `--checkpoint`, `--proc-time-rand-enabled`, `--rollout-n-envs`
- 推理导出入口:
  - `python -m solutions.Continuous_model.export_inference_sequence --device single --model <model_path>`
  - 当前 action sequence 输出固定为 `seq/tmp.json`
  - 导出的 `replay_env_overrides` 会携带 `single_route_name`，并在可用时携带 `single_route_config`，用于可视化回放时保持与导出一致的构网路线
- 二次释放惩罚验证入口:
  - `python -m solutions.Continuous_model.check_release_penalty --sequence <json_name> --results-dir results`
  - `--sequence` 必填，脚本按仓库根目录 `seq/<json_name>` 解析。

## Behavior Rules
1. 路径参数严格校验：`device_mode` 与 `route_code` 非法时直接报错。
2. 当提供 `single_route_config` 时，构网优先走“配置驱动编译链”（`parse/normalize -> route IR -> token route -> net build`）；`route_code` 仅用于兼容 alias 选择。
3. 当通过 `PetriEnvConfig.load(json)` 加载且 json 中提供 `single_route_config_path` 时，会自动读取该文件并填充 `single_route_config`。
4. WAIT 掩码规则：存在加工完成待取片晶圆时，仅允许短 WAIT（5s）。
5. 导出脚本的 `--out-name` 当前不参与文件命名，仅保留兼容。
6. `check_release_penalty.py` 未设置 `--sequence` 时不能执行。
7. 旧观测分支（place-obs）不再作为当前实现接口。
8. 配置驱动路径启用时，所选 `route.sequence` 中 stage 级 `process_time/cleaning_*` 会在 `pn_single` 侧先覆盖 `process_time_map/cleaning_*_map`，再进入 `_preprocess_process_time_map`；因此最终工时以 route stage 为准（仍按系统口径取整到 5 的倍数）。
9. `u_*` 使能在按 `route_queue` 推断到 `route_target` 时，仍必须满足目标库所可接收（未清洗且未满）；不满足时该分支必须退化为“无目标/不可使能”。
10. 级联观测中 `TM2/TM3` 的目标 one-hot 采用固定 8 维逐目标编码（不再按目标组压缩）；`LLC/LLD` 观测由 4 维扩展为 6 维，新增 `in/out` 两维方向 one-hot（下一跳由 `TM3` 搬运记为 `in`，由 `TM2` 搬运记为 `out`）。
11. `PetriEnvConfig.llc_tm3_takt_interval > 0` 时，对构网得到的 **LLC→`d_TM3`** 释放变迁（`("LLC","d_TM3")` 对应的 `u_*`）施加节拍门控：口径与 `u_LP` 的 `_takt_required_interval` 一致（**首次发射不因节拍被禁**，第二次起按 `build_fixed_takt_result(interval)` 的 `cycle_takts` 取最小间隔）；`<=0` 为默认关闭。门控同步作用于 `_get_enable_t`、`get_action_mask`、`get_enable_actions_with_reasons`（原因码 `llc_tm3_takt_release_limit`）与 `get_next_event_delta`。

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
- 2026-03-19: 修复 `1-6` 路径中 `PM3` 满载时 `u_PM1` 仍被使能的问题：`_get_enable_t/get_action_mask` 在使用 `route_target` 时新增目标容量与清洗校验，禁止绕过 `_select_target_for_source` 的接收约束。
