# Env_PN_Single_PlaceObs（已移除）

## Abstract
- What: 记录旧版单设备 place-centered observation 方案及其迁移信息。
- When: 仅在迁移旧脚本、对照历史实验、解释旧日志时使用。
- Not: 不代表当前仓库仍提供 `Env_PN_Single_PlaceObs` 类或 `--place-obs` CLI。
- Key rules:
  - 当前单设备统一入口是 `Env_PN_Single`。
  - `--place-obs` 已删除，训练与导出脚本不再切换观测类型。
  - 旧文档中的维度与参数仅用于历史对照，不能直接当作当前接口说明。

## When to use
- 迁移旧训练脚本或推理脚本。
- 对照历史模型输出或旧版文档。

## When NOT to use
- 编写新的训练、验证或推理流程。
- 查找当前单设备观测接口、CLI 参数或观测维度。

## Behavior / Rules
- 当前实现不再提供 `Env_PN_Single_PlaceObs`。
- 当前单设备统一由 `solutions/Continuous_model/env_single.py` 中的 `Env_PN_Single` 构建观测。
- `train_single.py` 与 `export_inference_sequence.py` 不接受 `--place-obs`。
- 若外部脚本仍引用旧类或旧参数，必须先迁移到 `Env_PN_Single`，再执行当前流程。

## Configuration / API
- 当前入口：
  - `solutions/Continuous_model/env_single.py`
  - `solutions/Continuous_model/train_single.py`
  - `solutions/A/eval/export_inference_sequence.py`
- 历史说明：
  - 旧版 place-centered 观测将库所状态直接拼接进 observation。
  - 该接口已被统一入口替代，不再作为当前实现的一部分。

## Examples
- 正例：将旧脚本中的 `Env_PN_Single_PlaceObs` 替换为 `Env_PN_Single` 后，再按当前 CLI 运行。
- 反例：继续传入 `--place-obs` 或 import 已移除的旧类。

## Edge Cases / Gotchas
- 旧实验日志可能仍使用 `place obs` 或 `place-centered obs` 的描述，需按历史语境理解。
- 旧维度说明不应与当前 `Env_PN_Single` 的 observation 维度混用。

## Related Docs
- `docs/continuous_solution_design.md`
- `docs/CHANGELOG.md`
- `docs/README.md`
