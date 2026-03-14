# CHANGELOG

## 2026-03-14

### 单设备 wait 截断新增运输完成事件 (2026-03-14)
- **What changed**：`pn_single.get_next_event_delta()` 在保留“加工完成”事件的同时，新增 `d_TM*` 运输位“达到 `T_transport`”作为关键事件；`step(wait)` 的长 WAIT 现在会被运输完成时刻截断。
- **Why**：当晶圆已在运输位停留接近运输完成时，继续整段长 WAIT 会跨过关键放片决策点，不利于调度策略及时响应。
- **Impact**：除 `WAIT_5s` 仍固定 5 秒外，其他 WAIT 的 `actual_wait=min(requested_wait,next_event_delta)` 中 `next_event_delta` 候选扩展为“加工完成 + 运输完成 + 清洗完成”，长 WAIT 截断会更及时。

## 2026-03-10

### 单设备 Q-Time 违规统计接入 (2026-03-10)
- **What changed**：`pn_single` 新增 `_check_qtime_violation`，在 `step()` 的 wait/fire 两条时间推进路径后检查运输位（type=2）超时，并更新 `qtime_violation_count`。
- **Why**：可视化面板已提供 `Q-TIME` 指标卡片，需要单设备后端提供稳定一致的违规计数来源。
- **Impact**：Q-Time 违规按 `stay_time > D_Residual_time` 判定，同一 wafer 仅首次违规计数 1 次；该检测不引入奖励惩罚，仅影响统计展示。

### Cascade 观测纳入 LLC/LLD (2026-03-10)
- **What changed**：`Env_PN_Single` 在 `device_mode=cascade` 下将 `LLC/LLD` 纳入腔室观测列表；`LLC/LLD` 使用 4 维核心特征（`occupied/processing/done_waiting_pick/remaining_process_time_norm`），其余 PM 继续使用 9 维特征。
- **Why**：需要让策略在级联流程中直接感知 `LLC/LLD` 状态，避免关键缓冲/交接位信息缺失。
- **Impact**：仅 cascade 模式 observation 维度发生变化；single 模式维度保持不变（`route_code=0 -> 32`，`route_code=1 -> 41`）。

## 2026-03-09

### Cascade 新增 route_code=3 (2026-03-09)
- **What changed**：`pn_single/construct_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=3`，路径为 `LP -> PM7/PM8(70) -> LLC(0) -> PM1/PM2(300) -> LLD(70) -> LP_done`。
- **Why**：支持“保留 LLD 清洗/缓冲段，但去掉 PM9/PM10 末段”的级联工艺模板，满足新工艺路径验证需求。
- **Impact**：级联模式可通过 `single_route_code=1/2/3` 切换三套模板；`route_code=3` 下不再生成 `PM9/PM10` 相关变迁动作。

### Cascade 新增 route_code=2 (2026-03-09)
- **What changed**：`pn_single/construct_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=2`，路径为 `LP -> PM7/PM8(70) -> LLC(0) -> PM1/PM2(300) -> LLD(70) -> PM9/PM10(200) -> LP_done`。
- **Why**：支持新增级联工艺路线，并保留现有 `route_code=1`（`PM1/2/3/4`）兼容既有流程。
- **Impact**：级联模式可通过 `single_route_code=1/2` 切换两套模板；`route_code=2` 下不再生成 `PM3/PM4` 对应变迁动作。

### 追责简要报告增强 (2026-03-09)
- **What changed**：`check_release_penalty.py` 输出新增 `penalized_actions_report`，覆盖所有被惩罚动作，包含 `action_name + step + reason + where_penalized`。
- **Why**：便于快速定位“哪一步、哪个动作、因哪个站点超容量而被 second-pass 惩罚”。
- **Impact**：可直接在结果文件里查看完整追责清单，无需手工对照 `blame_raw` 与 `step_records`。

### 释放违规追责限定 (2026-03-09)
- **What changed**：`blame_release_violations` 仅追责 `u_LP`、`u_LLC`、`u_LLD` 释放动作，不再追责 `u_PM7`、`u_PM2` 等从加工腔卸载的动作。
- **Why**：释放违规应归因于「从 LP/LLC/LLD 释放晶圆」的决策，不应惩罚 PM 腔室的取片动作。
- **Impact**：导出的 `release_penalty.steps` 将只包含 u_LP/u_LLC/u_LLD 动作所在步，u_PM7/u_PM2 所在步不再计入。

### 推理序列奖励报告 (2026-03-09)
- `export_inference_sequence.py` 在导出的 JSON 最前面增加 `reward_report` 区块。
- 包含 `scrap_penalty`、`release_penalty`、`idle_timeout_penalty` 的次数及触发步数（`count`, `steps`）。
- 回放逻辑仅读取 `sequence`、`replay_env_overrides`，兼容现有 `planB_sequence.json` 消费者。

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
