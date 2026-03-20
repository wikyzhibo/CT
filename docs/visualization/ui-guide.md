# UI Guide

## Abstract
- What: 本文档定义 PySide6 可视化界面的入口参数、数据契约、模型加载与回放流程。
- When: 修改 `visualization/main.py`、`main_window.py` 或回放数据格式前必须先读。
- Not: 不描述训练算法实现细节。
- Key rules:
  - 可视化入口统一为 `python -m visualization.main`。
  - 回放数据优先使用 `schema_version=2` 的序列结构。
  - 设备模式必须与模型/序列来源一致（single 或 cascade）。

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
2. 启动时按 `--device` 构建 `single/cascade` 对应后端。
3. Model A 模式读取模型权重并在线推理。
4. Model B 模式读取动作序列 JSON（默认由导出脚本生成）并逐步回放。
5. 状态与动作展示由 `main_window.py + center_canvas.py + stats_panel.py` 负责。

## Interfaces
- 启动命令:
  - `python -m visualization.main --device single`
  - `python -m visualization.main --device cascade --model <model_path>`
- 关键参数:
  - `--device {single,cascade}`
  - `--device-mode`（兼容参数，等价于 `--device`）
  - `--single-route-code {0,1}`
  - `--model/-m`, `--no-model`, `--debug`, `--quiet`
- 回放 JSON 契约（建议）:
  - 顶层字段: `schema_version`, `device_mode`, `sequence`, `reward_report`, `replay_env_overrides`
  - `sequence` 单步字段: `step`, `time`, `actions`（可兼容 `action`）
  - `replay_env_overrides` 建议包含: `route_code`, `single_route_name`, `single_route_config`（可选）以保证回放构网拓扑与导出时一致
  - 当前导出脚本默认输出: `seq/tmp.json`

## Behavior Rules
1. UI 设备模式与模型设备模式不一致时必须报错并阻止加载。
2. `--no-model` 启用时，仍可使用 Model B 进行离线回放。
3. 回放时序列来源若为 single，应保证 `actions` 与 `action` 字段兼容。
4. 可视化文档中的命令示例必须与 `visualization/main.py` 参数一致。

## Examples
- 正例:
  - `python -m visualization.main --device single --model solutions/Continuous_model/saved_models/CT_single_phase2_best.pt`
  - 导出序列后，在 UI 中选择 JSON 并用 Model B 回放。
- 反例:
  - single 序列在 cascade 模式下直接回放。
  - 文档仍引用已废弃的旧可视化入口。

## Edge Cases
- 未指定 `--model` 时，程序会尝试默认权重；不存在则进入手动模式。
- 序列文件路径正确但字段不完整时，Model B 回放可能失败。

## Related Docs
- `../overview/project-context.md`
- `../continuous-model/pn-single.md`
- `../training/training-guide.md`
- `../td-petri/td-petri-guide.md`
- `../deprecated/viz.md`

## Change Notes
- 2026-03-20: 回放环境重建新增透传 `single_route_name/single_route_config`；当序列来自配置驱动路线（如 `1-2`）时，UI 回放将按原路线构网，不再退化为仅 `route_code` 的默认拓扑。
- 2026-03-19: 建立可视化主文档，统一入口参数、模型加载与回放数据契约。
- 2026-03-19: 修复级联（`--device cascade`）模式下 `PM5/PM6` 被强制标记为 `disabled` 的问题；现在会按适配器状态展示（缺失时作为 `idle` 占位）。
