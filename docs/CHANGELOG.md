# CHANGELOG

## 2026-03-09

### What changed
- 单设备环境观测已统一：`Env_PN_Single` 合并 `Env_PN_Single_PlaceObs`，默认采用库所中心 obs 构造（`LP -> TM(d_TM1) -> PM*`）。
- 删除环境类 `Env_PN_Single_PlaceObs`，单设备仅保留 `Env_PN_Single` 一个入口。
- `train_single.py` 与 `export_inference_sequence.py` 移除 `--place-obs` 参数与对应分支逻辑。
- `pn_single` 新增 `single_device_mode`，支持 `single/cascade` 双模板路径；`cascade` 路径迁移为 `LP->PM7/PM8->LLC->PM1/PM2/PM3/PM4->LLD->PM9/PM10->LP_done`。
- 可视化 `visualization/main.py` 新增 `--device`（兼容 `--device-mode`），并在 `cascade` 模式下改为调用 `pn_single/env_single`，不再走 `pn.py`。

### Why
- 旧版 wafer-list obs 已不再使用，继续保留双环境实现会增加维护成本与行为漂移风险。
- 需要统一逻辑层到 `pn_single`，同时保留原单设备路径并新增级联模板，减少双后端维护复杂度。

### Impact / How to use
- 训练与推理导出在 single 模式下统一使用 `Env_PN_Single`，无需再切换 obs 形态。
- 若外部脚本仍 import `Env_PN_Single_PlaceObs` 或传入 `--place-obs`，需要迁移为统一入口。
- route 感知维度保持一致：`route_code=0` 为 `32` 维，`route_code=1` 为 `41` 维。
- 新增设备参数统一入口：`--device single|cascade`（`export_inference_sequence.py` 保留 `--device-mode` 兼容别名）。
- 若需要旧单设备路径，使用 `single_device_mode=single`；级联路径使用 `single_device_mode=cascade`。

### Debug fix (2026-03-09)
- 修复级联模式下 `u_LLD/u_PM9/u_PM10` 的机械手显示归属：动作后晶圆应显示在 `TM2` 而非 `TM3`。
- 修复级联模式 `PM2` 的腔室类型标记：`PM2` 现在按加工腔处理，恢复外圈加工进度条显示。
- 修复级联模式中 `LLD` 有晶圆时 `u_PM1/u_PM2/u_PM3/u_PM4` 被提前禁用的问题：允许先取片到 `d_TM1`，实际落片仍由 `t_LLD` 容量约束控制。
- 修复 wait 崩溃：当 `next_event_delta=0`（关键事件与当前时刻重合）时，`step(wait)` 不再计算 `actual_dt=0`，改为回退到最小 5s 推进，避免触发断言导致程序崩溃。

### Wait 50s 截断修复 (2026-03-09)
- 修复：当有晶圆进入 LP_done 时，WAIT 50s 被错误截断为 5s 的问题。原逻辑用 `_has_completed_wafers()`（任意晶圆在 LP_done）截断 wait；现改为仅在 episode 全部完成（`len(LP_done) >= n_wafer`）时才截断，允许在部分完工时执行完整 WAIT 50s。

### Cascade 双运输位重构 (2026-03-09)
- 级联模式构网由单一 `d_TM1` 改为双运输位 `d_TM2` + `d_TM3`：TM2 负责 LP/PM7/PM8/LLD/PM9/PM10 取放与 LLC 放片，TM3 负责 LLC/PM1–4 取放与 LLD 放片。
- `pn_single` 新增 `_transport_for_u_source()`、`_transport_for_t_target()`，使能与 dwell 检查按对应运输位执行。
- `get_next_event_delta` 排除所有 `d_*` 库所，不再仅排除 `d_TM1`。
- `env_single._extract_tm_features()` 遍历所有 `d_*` 运输位聚合观测。
- 可视化 `petri_single_adapter` 支持级联：运输位识别 `d_TM*`，TM2/TM3 晶圆来自 `d_TM2`/`d_TM3`。
