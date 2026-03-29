# UI Guide

## Abstract
- What: 本文档定义 PySide6 可视化界面的入口参数、数据契约、模型加载与回放流程。
- When: 修改 `visualization/main.py`、`main_window.py` 或回放数据格式前必须先读。
- Not: 不描述训练算法实现细节。
- Key rules:
  - 可视化入口统一为 `python -m visualization.main`。
  - 回放数据优先使用 `schema_version=2` 的序列结构。
  - 设备模式必须与模型/序列来源一致（single 或 cascade）。
  - `--concurrent` 只会在 `device=cascade` 下启用，并切换到 `Env_PN_Concurrent + PetriAdapter`。

## Scope
- In:
  - 可视化启动参数与设备切换行为。
  - Model A（在线模型）与 Model B（JSON 回放）数据输入契约。
  - 常见失败场景和定位方式。
- Out:
  - 具体奖励计算。
  - 训练过程细节。

## Architecture or Data Flow
1. `visualization/main.py` 解析 CLI，创建 adapter/viewmodel/window。
2. 启动时按 `--device` 构建 `single/cascade` 对应后端；若显式传 `--concurrent`，则在 `cascade` 下改为并发双动作后端。
3. 选择含 `head_tm2/head_tm3` 的权重时，UI 会自动重建为并发 runtime。
4. 选择旧并发变迁名动作序列时，UI 会在回放前自动重建为并发 runtime。
5. Model A 模式读取模型权重并在线推理。
6. Model B 模式读取动作序列 JSON（默认由导出脚本生成）并逐步回放。
7. 状态与动作展示由 `main_window.py + center_canvas.py + stats_panel.py` 负责；级联模式下画布上方路径框与 `--debug` 变迁区由 `main_window.py + control_panel.py + transition_labels.py` 负责。

## Interfaces
- 启动命令:
  - `python -m visualization.main --device single`
  - `python -m visualization.main --device cascade --model <model_path>`
  - `python -m visualization.main --device cascade --concurrent --model results/models/CT_concurrent_best.pt`
- 关键参数:
  - `--device {single,cascade}`
  - `--device-mode`（兼容参数，等价于 `--device`）
  - `--concurrent`：启用并发双动作可视化；仅支持 `--device cascade`
  - `--single-route-code {0,1}`（single）；cascade 下 legacy 代号见训练文档；配置驱动路线以 `single_route_name` 为准
  - `--model/-m`, `--no-model`, `--debug`, `--quiet`
- 配置菜单（级联）:
  - **路径（级联）**：可选 `1-1`…`1-6`、`2-1`…`2-4`（与 `data/petri_configs/cascade_routes_1_star.json` 中 `routes` 键一致）。确认后重建环境与适配器；若已选模型文件会尝试按新拓扑重载，不兼容则卸载 Model A。
  - 依赖启动时加载的 `data/petri_configs/cascade.yaml`（含 `single_route_config_path`）；仅 `device=cascade` 时菜单项可用。
- 画布顶栏:
  - 级联模式下在中间画布**上方**显示带边框区域：仅展示配置中 `routes[name].path`（**不**显示路线键名如 `1-6`，无额外标题行）；约 15px **粗体**，腔室名 / `(加工秒)` / `[清洁秒/片]` / `->` 分色。单设备模式下该区域隐藏。
- 回放 JSON 契约（建议）:
  - 顶层字段: `schema_version`, `device_mode`, `sequence`, `reward_report`, `replay_env_overrides`
  - `sequence` 单步字段: `step`, `time`, `actions`（可兼容 `action`）
  - `--concurrent` 下 `sequence[*].actions` 必须提供 `[tm2, tm3]` 两个动作名；WAIT 只支持 `WAIT` / `WAIT_5s`
  - 若 `sequence` 前几步中的非 WAIT 动作名全部属于 `Env_PN_Concurrent.TM2_TRANSITION_NAMES + Env_PN_Concurrent.TM3_TRANSITION_NAMES`，UI 会自动切到并发 runtime
  - `replay_env_overrides` 建议包含: `route_code`, `single_route_name`, `single_route_config`（可选）以保证回放构网拓扑与导出时一致
  - 导出脚本默认 `--out-name tmp`，输出 `results/action_sequences/tmp.json`；其它名字为 `results/action_sequences/<out_name>.json`

## Behavior Rules
1. UI 设备模式与模型设备模式不一致时必须报错并阻止加载。
2. `--no-model` 启用时，仍可使用 Model B 进行离线回放。
3. 回放时序列来源若为 single，应保证 `actions` 与 `action` 字段兼容。
4. 可视化文档中的命令示例必须与 `visualization/main.py` 参数一致。
5. 级联 / 单设备 + `--debug`：右侧 **TRANSITIONS** 区按当前网 `id2t_name` 顺序**每两个变迁一行**（左先右后，奇数个时末行右侧留白）；全量列出，用 `enabled` 区分可点/禁用。展示名映射（仅影响按钮文案，不改变 `action_id`）：`t_LLC`→`t_TM2_LLC`，`u_LLC`→`u_LLC_TM3`，`t_LLD`→`t_TM3_LLD`，`u_LLD`→`u_LLD_TM2`；级联下物理名为 `u_LLD*` 的卸载变迁 tooltip 可附 `LLD` 去向列表。
6. `adapter_factory` 在级联模式下会把窗口当前选择的 `single_route_name` 以 `setdefault` 写入 `env_overrides`，以便与回放 JSON 中的 `replay_env_overrides` 合并时后者仍可覆盖。
7. 晶圆可视化口径：`LLC/LLD`（`place_type=5`）在 `proc_time>0` 时按加工腔渲染，显示外圈进度与加工完成橙色；其 scrap 判定阈值与 `pn_single` 一致为 `process_time + 3 * P_Residual_time`，超时显示红色。
8. 手动导出（甘特图/统计/动作）必须强制写入 `results/` 规范目录：`results/gantt`、`results/training_logs`、`results/action_sequences`。
9. `--concurrent` 下 Model A 默认权重路径是 `results/models/CT_concurrent_best.pt`；未显式传 `--model` 时会优先尝试该文件。
10. `--concurrent` 下 WAIT 只会显示并执行 `5s`；调试区单击单个变迁时，仅对应机械手执行该变迁，另一机械手固定执行 `WAIT_5s`。
11. `--concurrent` 下不显示级联路线横幅，因为并发环境不消费 `single_route_name/single_route_config`。
12. 只要权重 `state_dict` 含 `head_tm2/head_tm3`，Model A 加载必须自动切到并发 runtime；禁止要求用户手动改代码路径才能加载当前 A 方案并发权重。
13. 只要回放序列被识别为并发动作命名，Model B 回放必须自动切到并发 runtime；禁止继续把 `(a1, a2)` 压缩成单动作执行。

## Examples
- 正例:
  - `python -m visualization.main --device single --model results/models/CT_single_best.pt`
  - `python -m visualization.main --device cascade --concurrent --model results/models/CT_concurrent_best.pt`
  - 导出序列后，在 UI 中选择 JSON 并用 Model B 回放。
- 反例:
  - single 序列在 cascade 模式下直接回放。
  - 在 `device=single` 下传 `--concurrent`。
  - 文档仍引用已废弃的旧可视化入口。

## Edge Cases
- 未指定 `--model` 时，程序会尝试默认权重；不存在则进入手动模式。
- `--concurrent` 依赖运行环境已安装 `PySide6`；缺失时入口会在导入 GUI 模块阶段失败。
- 序列文件路径正确但字段不完整时，Model B 回放可能失败。

## Related Docs
- `../overview/project-context.md`
- `../continuous-model/pn-single.md`
- `../training/training-guide.md`
- `../td-petri/td-petri-guide.md`
- `../deprecated/viz.md`

## Change Notes
- 2026-03-29: 可视化并发模式接入当前 A 方案：`main.py` 新增 `--concurrent`，在 `device=cascade` 下切换到 `Env_PN_Concurrent + PetriAdapter`；默认并发模型路径改为 `results/models/CT_concurrent_best.pt`。同时，若加载的权重 `state_dict` 含 `head_tm2/head_tm3`，或回放序列前几步动作名被识别为 `Env_PN_Concurrent.TM2_TRANSITION_NAMES + Env_PN_Concurrent.TM3_TRANSITION_NAMES`，UI 会自动重建为并发 runtime。`main_window.py` 在并发适配器下改为消费 adapter 侧 enabled actions、WAIT 固定 `5s`、并复用 `Env_PN_Concurrent` 的动作名表判断并发序列；`petri_adapter.py` 单击单个调试变迁时会把另一机械手固定为 `WAIT_5s`。
- 2026-03-22: LLC/LLD 可视化同步：`petri_single_adapter._time_to_scrap` 新增 `place_type=5`（`LLC/LLD`）分支，阈值使用 `process_time + 3*P_Residual_time - stay_time`；`wafer_item/chamber_widget` 在 `proc_time>0` 时将 `place_type=5` 按加工腔渲染（外圈进度、完成橙色、scrap 红色）。
- 2026-03-22: `--debug` 变迁区：`TRANSITIONS` 按 `id2t_name` 顺序固定两列（顺次两两一排），不再按 `u_LP`/`t_PM*` 等语义规则配对；级联与单设备一致。
- 2026-03-22: `PetriSingleAdapter` 级联运输位与构网一致：库所名 `TM2`/`TM3`（及历史 `d_TM2`/`d_TM3`）进入 `transport_buffers` 并驱动 TM2/TM3 晶圆绘制；避免仅识别 `d_TM*` 时运输区为空。
- 2026-03-20: 级联 UI：配置菜单「路径（级联）」、`1-1`…`2-4` 切换重建环境；画布顶栏仅 `routes[*].path`（无「当前路径」标题，粗体 15px，富文本分色）；`--debug` 变迁区双列与 LLC/LLD 展示名；`main.py` 工厂合并 `single_route_name`。
- 2026-03-20: 回放环境重建新增透传 `single_route_name/single_route_config`；当序列来自配置驱动路线（如 `1-2`）时，UI 回放将按原路线构网，不再退化为仅 `route_code` 的默认拓扑。
- 2026-03-19: 建立可视化主文档，统一入口参数、模型加载与回放数据契约。
- 2026-03-19: 修复级联（`--device cascade`）模式下 `PM5/PM6` 被强制标记为 `disabled` 的问题；现在会按适配器状态展示（缺失时作为 `idle` 占位）。
