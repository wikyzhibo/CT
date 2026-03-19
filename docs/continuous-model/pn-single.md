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
- 2026-03-19: 修复配置驱动路径 `single_route_name` 下 stage 工时被全局 `process_time_map` 回写的问题；现已在 `pn_single` 初始化阶段先注入 route stage 覆盖（含 cleaning map），并统一 route 选择名与构网阶段一致。
- 2026-03-19: `PetriEnvConfig.load` 支持 `single_route_config_path` 自动装载外部路线配置文件；`cascade.json` 可直接通过 `single_route_name` 切换目标路线而不改代码。
- 2026-03-19: `construct_single.py` 新增配置驱动构网通路，支持 `routes.sequence/repeat`、机器人自动推断 transport place、自动编译 `token.route_queue` 与 `t_route_code_map`；`pn_single.py` 优先消费构网返回的 `route_meta`，避免与运行时路由元数据不一致。
- 2026-03-19: 建立 pn_single 主文档，统一单设备入口、脚本接口与行为规则说明。
