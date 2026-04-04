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
  - 当前 `Env_PN_Concurrent` 直接消费 `solutions.A.petri_net.ClusterTool` 的真实级联运行时，不再绑定 `deprecated.pn.Petri`。

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
2. 启动时按 `--device` 构建 `single/cascade` 对应后端；若显式传 `--concurrent`，则在 `cascade` 下改为并发三动作后端，且底层仍复用当前 `ClusterTool` 的路线配置。
3. 选择含 `head_tm2` 与 `head_tm3` 的权重（`DualHeadPolicyNet`，当前 A 方案并发训练产物）时，UI 会自动重建为并发 runtime；可选兼容仍含 `head_tm1` 的旧键以识别并发类权重。
4. Model B 回放优先读取 `replay_env_overrides.runtime_mode` 或顶层 `device_mode`；仅当两者都缺失且序列中出现“同一步两个非 WAIT 动作”时，UI 才会自动推断为并发 runtime。
5. Model A 模式读取模型权重并在线推理。
6. Model B 模式读取动作序列 JSON（默认由导出脚本生成）并逐步回放。
7. 状态与动作展示由 `main_window.py + center_canvas.py + stats_panel.py` 负责；级联模式下画布上方路径框与 `--debug` 变迁区由 `main_window.py + widgets/route_path_display.py + widgets/control_panel.py + widgets/transition_labels.py` 负责。

## Interfaces
- 启动命令:
  - `python -m visualization.main --device single`
  - `python -m visualization.main --device cascade --model <model_path>`
  - `python -m visualization.main --device cascade --concurrent --model results/models/CT_concurrent_best.pt`
- 关键参数:
  - `--device {single,cascade}`
  - `--device-mode`（兼容参数，等价于 `--device`）
  - `--concurrent`：启用并发三动作可视化；仅支持 `--device cascade`
  - `--single-route-code {0,1}`（single）；cascade 下 legacy 代号见训练文档；配置驱动路线以 `single_route_name` 为准
  - `--model/-m`, `--no-model`, `--debug`, `--quiet`
- 配置菜单（级联）:
  - **路径（级联）**：可选 `1-1`…`1-6`、`2-1`…`2-4`（与 `data/petri_configs/cascade_routes_1_star.json` 中 `routes` 键一致）。确认后重建环境与适配器；单动作与并发 runtime 都会消费所选 `single_route_name`。若已选模型文件会尝试按新拓扑重载，不兼容则卸载 Model A。
  - 依赖启动时加载的 `data/petri_configs/cascade.yaml`（含 `single_route_config_path`）；仅 `device=cascade` 时菜单项可用。
- 画布顶栏:
  - 级联模式下在中间画布**上方**显示带边框区域：仅展示配置中 `routes[name].path`（**不**显示路线键名如 `1-6`，无额外标题行）；约 15px **粗体**，腔室名 / `(加工秒)` / `[清洁秒/片]` / `->` 分色。当前规则对单动作与并发级联 runtime 一致；单设备模式下该区域隐藏。
- 回放 JSON 契约（建议）:
  - 顶层字段: `schema_version`, `device_mode`, `sequence`, `reward_report`, `replay_env_overrides`
  - `sequence` 单步字段: `step`, `time`, `actions`（可兼容 `action`）
  - `--concurrent` 下 `sequence[*].actions` 优先提供 `[tm1, tm2, tm3]`；旧格式 `[tm2, tm3]` 仅做兼容读取并按 `TM1=WAIT` 解释；WAIT 只支持 `WAIT` / `WAIT_5s`
  - `sequence[*]` 建议同时写入 `action_tm1` / `action_tm2` / `action_tm3`；UI 可据此直接识别并发回放。**运行时 TM1 由 `ClusterTool` 规则执行**，Model B 推进仿真时仍只向 `step` 传 TM2/TM3；`actions[0]` 与 `action_tm1` 为记录/对齐用（与导出脚本规则解码一致）
  - `replay_env_overrides` 建议显式包含 `runtime_mode`；若缺失，UI 只会在序列显式带三头动作字段或同一步出现至少两个非 WAIT 动作时自动切到并发 runtime
  - `replay_env_overrides` 建议包含: `runtime_mode`, `single_route_name`, `single_route_config`（可选）以保证回放构网拓扑与导出时一致
  - 导出脚本默认 `--out-name tmp`，输出 `results/action_sequences/tmp(W<n_wafer>-M<time>).json`；其它名字同样追加 `(W-M)` 后缀

## Behavior Rules
1. UI 设备模式与模型设备模式不一致时必须报错并阻止加载。
2. `--no-model` 启用时，仍可使用 Model B 进行离线回放。
3. 回放时序列来源若为 single，应保证 `actions` 与 `action` 字段兼容。
4. 可视化文档中的命令示例必须与 `visualization/main.py` 参数一致。
5. 级联 / 单设备 + `--debug`：右侧 **TRANSITIONS** 区按当前网 `id2t_name` 顺序**每两个变迁一行**（左先右后，奇数个时末行右侧留白）；全量列出，用 `enabled` 区分可点/禁用。展示名映射（仅影响按钮文案，不改变 `action_id`）：`t_LLC`→`t_TM2_LLC`，`u_LLC`→`u_LLC_TM3`，`t_LLD`→`t_TM3_LLD`，`u_LLD`→`u_LLD_TM2`；级联下物理名为 `u_LLD*` 的卸载变迁 tooltip 可附 `LLD` 去向列表。
6. `adapter_factory` 在级联模式下会把窗口当前选择的 `single_route_name` 以 `setdefault` 写入 `env_overrides`，以便与回放 JSON 中的 `replay_env_overrides` 合并时后者仍可覆盖。
7. 晶圆可视化口径：`LLC/LLD`（`place_type=5`）在 `proc_time>0` 时按加工腔渲染，显示外圈进度与加工完成橙色；其 scrap 判定阈值与 `pn_single` 一致为 `process_time + 3 * P_Residual_time`，超时显示红色。`time_to_scrap < 0` 表示“无 scrap 风险”，禁止把该值解释为已 scrap；`AL/CL/LLA/LLB` 这类正工时 buffer 只显示计时/完成态，不显示 `SCRAP`。
8. 手动导出（甘特图/统计/动作）必须强制写入 `results/` 规范目录：`results/gantt`、`results/training_logs`、`results/action_sequences`。
9. `--concurrent` 下 Model A 默认权重路径是 `results/models/CT_concurrent_best.pt`；未显式传 `--model` 时会优先尝试该文件。
10. `--concurrent` 下 WAIT 只会显示并执行 `5s`；调试区单击单个变迁时，仅对应机械手执行该变迁，其余机械手固定执行 `WAIT_5s`。
11. 并发级联 runtime 与单动作级联 runtime 一样消费 `single_route_name/single_route_config`；路线横幅与路径切换必须可用。
12. 只要权重 `state_dict` 含 `head_tm2` 与 `head_tm3`（当前 A 方案 `DualHeadPolicyNet`），Model A 加载必须自动切到并发 runtime；含 `head_tm1` 的旧三头键仍可识别为并发类权重。在线推理只输出 TM2/TM3；TM1 由网内规则执行；History 仍显示三列。
13. Model B 回放必须优先按显式 `runtime_mode` 判断；只有缺少显式标记时，才允许用“显式三头动作字段 / 至少两个非 WAIT 动作”兜底识别并发。禁止继续把 `(a1, a2, a3)` 压缩成单动作执行。
14. `device=cascade` 且单臂模式下，中心画布必须在 `LLA/LLB` 下方固定新增 `AL/CL/LP/LP_done`，并在 `LP/AL/CL/LP_done/LLA/LLB` 几何中心展示 `TM1 ARM`。这些腔室与 `TM1 ARM` 在当前实现中直接消费真实运行时状态，不再是纯 UI 占位。单动作级联 runtime 下，`LP1/LP2` 必须聚合显示为 `LP`，`LP_done` 必须保持 `LP_done` 名称，`TM1` 只通过 `TM1 ARM` 展示，不单独作为 chamber 渲染。
15. `device=cascade` 且单臂模式下，`center_canvas` 固定使用 `cell_w=96`、`cell_h=84`、`chamber_scale=0.9`、`robot_scale=0.8` 作为基准布局参数；实际步距必须按“缩放后腔室尺寸 + 10px”自动抬高，避免腔室重叠。single 与 cascade 双臂布局禁止复用该缩放与自动扩距规则。
16. 并发 runtime（`PetriAdapter`）下，`LP`/`LP_done` 聚合状态**只会**在主循环遍历库所时通过 `_merge_alias_state` 各合并一次；禁止先对 `_start_place_names()`/`_end_place_names()` 全量 `_build_alias_state` 再对同名库所合并，否则界面上的 `LP`/`LP_done` 晶圆条数与容量会变为真实值的 **2 倍**。
17. 级联构网使用的 `config/cluster_tool/route_config.json` 中 **`source.capacity` 与 `sink.capacity` 必须 ≥ 当前 `PetriEnvConfig.n_wafer`（或 `n_wafer1+n_wafer2`）**；否则终点 `LP_done` 满仓后 `t_TM1_LP_done` 无法投片，晶圆会停在 TM1（用户易误判为「从 CL 出来后进不了 LP_done」）。

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
- 由单动作导出脚本生成的 `actions=[<single_action>, "WAIT", "WAIT"]` 序列不会再被自动识别为并发 runtime。
- 级联单臂若手动调大 `chamber_scale` / `robot_scale`，画布会优先自动扩距避免腔室重叠，而不是强制压回旧尺寸；超出视口时由 `QGraphicsView` 自身滚动显示。

## Related Docs
- `../overview/project-context.md`
- `../continuous-model/pn-single.md`
- `../training/training-guide.md`
- `../td-petri/td-petri-guide.md`
- `../viz.md`

## Change Notes
- 2026-04-04: `visualization/` 目录清理与收敛：删除不可达适配器与脚本模块 `dfs_adapter.py`、`ga_adapter.py`、`pdr_adapter.py`、`scripted_adapter.py`，删除无调用工具模块 `config_editor.py`、`debug_tools.py`、`export_tools.py`、`smoke_test.py`；`transition_labels.py` 与 `route_path_display.py` 迁移到 `visualization/widgets/`；`algorithm_interface.py` 移除 `AlgorithmAdapter` 抽象基类，仅保留 `ActionInfo/StateInfo` 等 UI 数据契约类型，运行时保持 `petri` 单适配器路径不变。
- 2026-04-04: `visualization/main.py` 的环境构造收敛到 `solutions.A.rl_env.make_env(...)`，与 `solutions.A.eval.export_inference_sequence.py` 共享同一套 `runtime_mode/device_mode + n_wafer/single_route_name/single_route_config/process_time_map` 过滤与校验；保留 `single_process_time_map` 兼容别名；CLI 与运行时行为不变。
- 2026-04-03: `config/cluster_tool/route_config.json`：`source`/`sink` 容量由 `25` 提至 `100`，避免 `n_wafer`（如 `30`）大于旧 `sink.capacity` 时终点堵塞与 TM1 持片无法卸入 `LP_done`。
- 2026-04-03: `petri_adapter.py`：修复并发级联下 `LP`/`LP_done` 聚合重复合并（先 `_build_alias_state(全量)` 再 `_merge_alias_state`）导致腔室晶圆数与容量显示翻倍；现以空壳 `_build_alias_state(..., [], ...)` 初始化，仅由主循环合并一次。
- 2026-04-03: `center_canvas.py` 扩展级联单臂布局：在 `LLA/LLB` 下方新增 `AL/CL/LP/LP_done`，并增加 `TM1 ARM`；`petri_single_adapter.py` 现将 `LP1/LP2` 聚合为 `LP`、`LP_done` 直显为 `LP_done`，且 `TM1` 只通过 `TM1 ARM` 展示真实持片；该模式当前使用 `cell_w=96`、`cell_h=84`、`chamber_scale=0.9`、`robot_scale=0.8`，实际步距会按缩放后卡片尺寸自动扩至最少留 `10px` 间隔，single 与级联双臂布局保持不变。
- 2026-04-03: `wafer_item.py` / `chamber_widget.py` / `wafer_widget.py` 收敛 `time_to_scrap` 口径：`time_to_scrap < 0` 表示无 scrap 风险，不再把 `AL/CL/LLA/LLB` 这类正工时 buffer 渲染为红色或 `SCRAP`。
- 2026-03-30: 移除 `route_code` / `--single-route-code`：回放与 CLI 仅以 `single_route_name` / `single_route_config` 对齐构网；`replay_env_overrides` 不再包含 `route_code`。
- 2026-03-29: 并发可视化运行时切换到当前 `ClusterTool`：`Env_PN_Concurrent` 不再绑定 `deprecated.pn.Petri`，而是直接加载 `config/cluster_tool/cascade.yaml` 构建级联并发环境；`main.py` 在并发分支会同步透传 `single_route_name/single_route_config/process_time_map`。`petri_adapter.py` 改为遍历当前 net 的真实库所和变迁，不再依赖 `LP1/LP2/s1-s5/d_TM2/d_TM3` 硬编码。
- 2026-03-29: Model B 并发识别收口为“显式 runtime 优先 + 双非 WAIT 兜底”：`main_window.py` 优先读取 `replay_env_overrides.runtime_mode` 或顶层 `device_mode`；仅当缺失显式标识且同一步出现两个非 WAIT 动作时才切并发 runtime，避免把单动作导出序列 `actions=[action, "WAIT"]` 误判为并发。同时恢复并发级联 runtime 的路线横幅与路径切换。
- 2026-03-22: LLC/LLD 可视化同步：`petri_single_adapter._time_to_scrap` 新增 `place_type=5`（`LLC/LLD`）分支，阈值使用 `process_time + 3*P_Residual_time - stay_time`；`wafer_item/chamber_widget` 在 `proc_time>0` 时将 `place_type=5` 按加工腔渲染（外圈进度、完成橙色、scrap 红色）。
- 2026-03-22: `--debug` 变迁区：`TRANSITIONS` 按 `id2t_name` 顺序固定两列（顺次两两一排），不再按 `u_LP`/`t_PM*` 等语义规则配对；级联与单设备一致。
- 2026-03-22: `PetriSingleAdapter` 级联运输位与构网一致：库所名 `TM2`/`TM3`（及历史 `d_TM2`/`d_TM3`）进入 `transport_buffers` 并驱动 TM2/TM3 晶圆绘制；避免仅识别 `d_TM*` 时运输区为空。
- 2026-03-20: 级联 UI：配置菜单「路径（级联）」、`1-1`…`2-4` 切换重建环境；画布顶栏仅 `routes[*].path`（无「当前路径」标题，粗体 15px，富文本分色）；`--debug` 变迁区双列与 LLC/LLD 展示名；`main.py` 工厂合并 `single_route_name`。
- 2026-03-20: 回放环境重建新增透传 `single_route_name/single_route_config`；当序列来自配置驱动路线（如 `1-2`）时，UI 回放将按原路线构网，不再退化为仅 `route_code` 的默认拓扑。
- 2026-03-19: 建立可视化主文档，统一入口参数、模型加载与回放数据契约。
- 2026-03-19: 修复级联（`--device cascade`）模式下 `PM5/PM6` 被强制标记为 `disabled` 的问题；现在会按适配器状态展示（缺失时作为 `idle` 占位）。
- 2026-04-03: 并发可视化升级为 `TM1/TM2/TM3` 三动作：`main.py` 并发模型为 `DualHeadPolicyNet`（`head_tm2`/`head_tm3`）；`main_window.py` 回放读取 `actions`/`action_tm*`；`petri_adapter.py` 仅 `step(a2,a3)` 推进、`TM1` 文案来自规则缓存；单臂级联下 `AL/CL/LP/LP_done/TM1 ARM` 已接入真实状态；见 `docs/CHANGELOG.md`。
