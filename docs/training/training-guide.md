# Training Guide

## Abstract
- What: 本文档定义 Continuous_model 的训练入口、配置优先级、产物输出和复现实验最小流程。
- When: 启动训练、更新训练参数、排查训练产物时使用。
- Not: 不覆盖 Td_petri 的链式搜索策略实现。
- Key rules:
  - 级联训练入口是 `train_single.py --device cascade`。
  - 并发训练入口是 `train_concurrent.py`。
  - 命令示例必须与脚本实际参数一致。

## Scope
- In:
  - cascade/concurrent 训练入口。
  - 配置来源与覆盖关系。
  - 输出模型路径与日志观察点。
- Out:
  - 具体网络结构推导。
  - UI 回放细节。

## Architecture or Data Flow
1. 读取配置（`data/ppo_configs/*.yaml` 或 `.json`；级联默认 `s_train.yaml`）。
2. 构建环境 (`Env_PN_Single` 或 `Env_PN_Concurrent`)。
3. rollout 采样与 PPO 更新。
4. 保存 best/final 权重。
5. 可选导出推理序列用于可视化回放。

## Interfaces
- 单设备训练:
  - `python -m solutions.Continuous_model.train_single --device cascade --rollout-n-envs 1`
  - `python -m solutions.Continuous_model.train_single --device cascade --artifact-dir exp_001`（`--artifact-dir` 作为运行名称前缀；权重写入 `results/models/`，日志与指标写入 `results/training_logs/`，序列写入 `results/action_sequences/`；存在 `best.pt` 时导出链调用 `ClusterTool.render_gantt` 将甘特写入 `results/gantt/`（具体文件名见 `docs/gantt.md`）；标题可带 `路径 <single_route_name>` 后缀）
  - 参数: `--device`, `--compute-device`, `--checkpoint`, `--rollout-n-envs`, `--artifact-dir`
- 训练指标图（独立运行）:
  - `python -m solutions.Continuous_model.eval.plot_train_metrics --input <training_metrics.json> --output <out.png>`（输出会统一落到 `results/training_logs/`；可选 `--smooth-window`、`--show`、`--route-label`）
- 并发训练:
  - `python -m solutions.Continuous_model.train_concurrent --config data/ppo_configs/concurrent_phase2_config.json`
  - 参数: `--config`, `--checkpoint`
- A 方案 PPO 训练（默认并发）:
  - `python -m solutions.A.ppo_trainer`
  - 并发关闭（回退单动作）: `python -m solutions.A.ppo_trainer --no-concurrent --device cascade`
  - 参数: `--concurrent`(默认开启), `--no-concurrent`, `--checkpoint`, `--compute-device`, `--rollout-n-envs`, `--artifact-dir`
  - `--rollout-n-envs` 对并发模式同样生效（使用 `VectorEnv_Concurrent` 并行采样）
  - 并发环境 `Env_PN_Concurrent` 直接消费 `config/cluster_tool/cascade.yaml + ClusterTool` 当前级联运行时；观测维度与 TM2/TM3 动作维度由真实 net 探针动态读取，不再依赖 legacy 动作名表
  - 训练结束若存在 `results/models/CT_concurrent_best.pt`，与单动作路径相同会调用 `export_inference_sequence.rollout_and_export(..., device_mode=concurrent)` 写出 `results/action_sequences/` 下 JSON 与 `results/gantt/` 下甘特 PNG（`run_name` 为 `train_concurrent` 或 `--artifact-dir` 安全名前缀；需 `render_gantt` 成功则见 `docs/gantt.md`）
- 导出推理序列:
  - `python -m solutions.Continuous_model.export_inference_sequence --device cascade --model <model_path>`（输出 `results/action_sequences/<out_name>.json`，默认 `--out-name tmp` 即 `results/action_sequences/tmp.json`）
  - `--model` 为已存在的 `.pt` 文件路径时直接使用；否则按 `results/models/<相对路径>` 解析。
- 关键配置优先级:
  - cascade: `data/ppo_configs/s_train.yaml` 作为基础，CLI 参数覆盖。
  - concurrent: `--config` 文件优先，不存在时退回默认配置对象。

## Behavior Rules
1. 训练文档必须同时列出 cascade 与 concurrent 入口，不混用参数。
2. 产物说明必须区分“公共 best 路径”和“时间戳备份目录”。
3. 未传 `--artifact-dir` 时输出前缀固定为 `train_single`；传入 `--artifact-dir` 时将其作为输出文件名前缀，产物仍统一写入 `results/` 规范目录。
4. 禁止继续在主文档中引用已移除的旧观测切换参数。
5. `solutions.A.ppo_trainer` 默认走双动作并发训练；仅在显式传 `--no-concurrent` 时回退单动作路径。
6. 并发模式下 WAIT 只保留单档 `5s`（TM2/TM3 各一个 WAIT 动作）。
7. 并发模式由训练入口选用 `Env_PN_Concurrent`，对应 `ClusterTool(..., concurrent=True)`；TM2/TM3 掩码以 `ClusterTool.get_action_mask`（并发实例返回两段局部掩码）与当前 `id2t_name` 为唯一真源；禁止继续绑定 `deprecated.pn.Petri` 的旧动作名集合。

## Examples
- 正例:
  - 级联 CPU 训练: `python -m solutions.Continuous_model.train_single --device cascade --compute-device cpu`
  - 级联 GPU 更新 + 多环境 rollout: `python -m solutions.Continuous_model.train_single --device cascade --compute-device cuda --rollout-n-envs 8`
  - 带产物目录: `python -m solutions.Continuous_model.train_single --device cascade --artifact-dir models/exp_001`
  - 并发训练: `python -m solutions.Continuous_model.train_concurrent --config data/ppo_configs/concurrent_phase2_config.json`
  - A 方案默认并发训练: `python -m solutions.A.ppo_trainer`
  - A 方案并发 + 4 并行环境: `python -m solutions.A.ppo_trainer --rollout-n-envs 4`
  - A 方案单动作回退: `python -m solutions.A.ppo_trainer --no-concurrent --device cascade`
- 反例:
  - 将 concurrent 参数传给 train_single。
  - 在 A 方案并发模式中假设存在 `WAIT_10s/20s` 等多档 WAIT（不存在，仅 `WAIT_5s`）。
  - 误以为未传 `--artifact-dir` 时也会自动生成 `training_metrics_plot.png`（不会；仅 `artifact-dir` 流程会生成）。

## Edge Cases
- `train_concurrent.py` 的默认 `--config` 是本机绝对路径，跨机器时应显式传相对路径。
- cascade 训练 best 权重会覆盖 `results/models/CT_single_best.pt`，并行实验需使用不同运行前缀区分产物。
- 导出脚本按 `--out-name` 写入 `results/action_sequences/<out_name>.json`；并发运行须使用不同 `out-name`。
- `training_metrics_plot.png` 由 `eval/plot_train_metrics.py` 绘制：左图 reward（滑动平均）+ makespan 双 y 轴（`makespan==0` 不绘制），右图 finish/scrap 并列柱图；环境若带 `single_route_name`，两子图标题后缀为 `路径 <name>`。标题含中文时依赖系统已安装的无衬线中文字体（Windows 通常已有微软雅黑/黑体；若仍为方框，请安装 Noto Sans CJK 或在环境中配置 Matplotlib 字体）。
- `solutions.A.ppo_trainer` 在单动作与并发训练结束时，只要 `CT_single_best.pt` / `CT_concurrent_best.pt` 存在即调用 `rollout_and_export`（单动作为 `device_mode=single|cascade`，并发为 `device_mode=concurrent`），写出动作序列 JSON 并尝试甘特；与是否传入 `--artifact-dir` 无关。甘特文件名由 `plot_gantt_hatched_residence` 在基路径上追加策略后缀（见 `docs/gantt.md`）。`ClusterTool.render_gantt` 从 `fire_log` 写 PNG；若 rollout 无腔室进出事件则调用会失败（见 `docs/gantt.md`）。标题后缀为 `title_suffix`（`路径 <single_route_name>`）。
- **train_all 多路线批量训练**：暂缓，当前仓库不提供该入口；验收 `train_single --artifact-dir` 后再扩展。

## Related Docs
- `../overview/project-context.md`
- `../continuous-model/pn-single.md`
- `../visualization/ui-guide.md`
- `../deprecated/continuous-solution-design.md`

## Change Notes
- 2026-03-31: `solutions.A.ppo_trainer` 并发训练结束与单动作一致：存在 `CT_concurrent_best.pt` 时 `rollout_and_export(..., device_mode=concurrent)` 导出动作序列与甘特；见 `solutions/A/eval/export_inference_sequence.py`。
- 2026-03-31: A 方案 `solutions/A/petri_net.py` 中 `ClusterTool.render_gantt` 已实现：由 `fire_log` 写腔室甘特 PNG（见 `docs/gantt.md`）。
- 2026-03-29: `solutions.A.ppo_trainer` 的并发探针环境继续是 `Env_PN_Concurrent`，但该环境现已直接复用 `ClusterTool` 当前级联运行时；TM2/TM3 动作维度与观测维度由真实 net 探针动态读取，和 UI / rollout wrapper 保持同一口径，不再依赖 legacy `u_LP1_s1/t_s1` 命名集合。
- 2026-03-29: 并发训练路径切换为 `collect_rollout_ultra_concurrent` + `VectorEnv_Concurrent`，`--rollout-n-envs` 现对并发模式同样生效；输出新增 rollout/update 分段计时与 steps/sec 统计。
- 2026-03-29: `solutions.A.ppo_trainer` 新增并发默认训练路径；支持 `--concurrent/--no-concurrent` 切换；并发模式 WAIT 约束为单档 5s。
- 2026-03-27: 统一输出规范：训练与导出产物全面迁移到 `results/` 目录族（`action_sequences/gantt/training_logs/topology_cache/models`）；`--artifact-dir` 改为运行名称前缀，不再控制目录位置。
- 2026-03-22: `--artifact-dir` 且存在 `best.pt` 时恢复写出 `gantt.png`；`training_metrics_plot.png` 与甘特标题均支持 `路径 <single_route_name>` 后缀（无路线名则不加）。
- 2026-03-22: 移除 `train_single` 内联 matplotlib dashboard；`training_metrics_plot.png` 改由 `eval.plot_train_metrics.plot_metrics` 在写入 `training_metrics.json` 后生成。
- 2026-03-22: `--artifact-dir` 下增加 `training_metrics.json`，仅含每 batch 的 `reward`、`makespan`、`finish`、`scrap`。
- 2026-03-22: `train_single` 增加 `--artifact-dir`（训练 log、metrics、指标图、导出序列）；`export_inference_sequence` 按 `--out-name` 写入 `seq/<name>.json`，默认 `tmp`；`train_all` 批量路线暂缓。
- 2026-03-21: single 训练路径下线，文档入口收敛为 cascade/concurrent；`train_single` 示例统一为 `--device cascade`。
- 2026-03-19: 建立训练主文档，统一 single/cascade/concurrent 入口与产物说明。
