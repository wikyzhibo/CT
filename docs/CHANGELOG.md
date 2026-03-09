# CHANGELOG

## 2026-03-09

### What changed
- 单设备环境观测已统一：`Env_PN_Single` 合并 `Env_PN_Single_PlaceObs`，默认采用库所中心 obs 构造（`LP -> TM(d_TM1) -> PM*`）。
- 删除环境类 `Env_PN_Single_PlaceObs`，单设备仅保留 `Env_PN_Single` 一个入口。
- `train_single.py` 与 `export_inference_sequence.py` 移除 `--place-obs` 参数与对应分支逻辑。

### Why
- 旧版 wafer-list obs 已不再使用，继续保留双环境实现会增加维护成本与行为漂移风险。

### Impact / How to use
- 训练与推理导出在 single 模式下统一使用 `Env_PN_Single`，无需再切换 obs 形态。
- 若外部脚本仍 import `Env_PN_Single_PlaceObs` 或传入 `--place-obs`，需要迁移为统一入口。
- route 感知维度保持一致：`route_code=0` 为 `32` 维，`route_code=1` 为 `41` 维。
