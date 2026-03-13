# Env_PN_Single_PlaceObs

## Abstract
- What: 定义单设备单动作 TorchRL 环境 `Env_PN_Single_PlaceObs` 的 observation 结构。
- When: 当训练策略需要按库所/机器状态建模，而不是按晶圆列表拼接时使用。
- Not: 不改 observation 语义，不改 `reset/step/reward` 接口形态；动作空间已扩展为多档 WAIT。
- Key rules:
 - observation 按 `LP -> TM -> chamber*` 顺序拼接，chamber 列表由 `single_route_code` 与 `device_mode` 决定。
 - `route_code=0` 时 `PM* = PM1, PM3, PM4`；`route_code=1` 时 `PM* = PM1, PM3, PM4, PM6`；
 - `device_mode=cascade` 时会额外纳入 `LLC/LLD` 两个腔室；其中 `LLC/LLD` 使用 4 维核心特征，其余 PM 仍使用 9 维特征。
 - `LP_done` 不进入主体 observation。
 - 时间相关特征统一归一化并裁剪到 `[0, 1]`。

## When to use
- 需要显式观察 PM3 与 PM4 分离状态（含清洗）时。
- 需要低维、语义稳定的 place-centered 状态向量时。

## When NOT to use
- 需要逐晶圆 one-hot 路由信息时（请使用 `Env_PN_Single`）。
- 需要保留按 `token_id` 排序的观测语义时。

## Behavior / Rules
- 动作空间沿用 `Env_PN_Single` 的统一动作目录：`transition + multi-wait`（默认 `5/10/20/50/100s`）。
- WAIT 推进规则：
 - 当 `wait_duration == 5` 时，固定推进 5 秒，不做事件截断。
 - 当 `LP_done` 已有完工晶圆时，大于 5 秒的 WAIT 会被动作掩码屏蔽。
 - 其他 WAIT 仍按 `min(wait_duration, next_event_delta)` 推进，避免一次跳过多个关键决策点。
- 观测维度随路线与设备模式变化：
 - LP: 1 维
 - TM: single 模式 8 维（4 维时间 + 4 维去向 one-hot）；cascade 模式 14 维（TM2 块 8 维 + TM3 块 6 维，含晶圆去向 one-hot）
 - `route_code=0`（single）：`PM1/PM3/PM4`，每个 9 维，总维度 `1 + 8 + 9*3 = 36`
 - `route_code=1`（single）：`PM1/PM3/PM4/PM6`，每个 9 维，总维度 `1 + 8 + 9*4 = 45`
 - cascade：TM 14 维；PM 按 9 维、`LLC/LLD` 按 4 维，总维度 `1 + 14 + 9*len(PM*) + 4*len({LLC,LLD}∩chamber*)`
- LP 特征：
 - `remaining_wafer_norm = clip(len(LP.tokens) / n_wafer, 0, 1)`
- TM 特征（single 模式，8 维 = 4 维时间 + 4 维去向 one-hot）：
 - 时间：`transport_complete`、`wafer_stay_over_long`、`wafer_stay_time_norm`、`distance_to_penalty_norm`
 - 去向 one-hot（4 类）：往 PM1、往 PM3 或 PM4、往 PM6、往 LP_done
- TM 特征（cascade 模式，14 维 = TM2 块 8 维 + TM3 块 6 维）：
 - TM2 块：上述 4 维时间特征 + 晶圆去向 one-hot（4 类：往 PM7/8、往 LLC、往 PM9/10、往 LP_done）
 - TM3 块：上述 4 维时间特征 + 晶圆去向 one-hot（2 类：往 PM1/2/3/4、往 LLD）
- PM 特征（每腔室一致）：
 - `occupied`
 - `processing`
 - `done_waiting_pick`
 - `remaining_process_time_norm`
 - `wafer_stay_time_norm`
 - `wafer_time_to_scrap_norm`
 - `is_cleaning`
 - `clean_remaining_time_norm`
 - `remaining_runs_before_clean_norm`
- LLC/LLD 特征（4 维核心）：
 - `occupied`
 - `processing`
 - `done_waiting_pick`
 - `remaining_process_time_norm`

## Configuration / API
- 类名：`Env_PN_Single_PlaceObs`
- 基于：`solutions/Continuous_model/env_single.py`
- 路线配置：`single_route_code`（影响 observation 的 PM 列表与维度）
- 关键归一化参数：
 - `P_Residual_time`
 - `D_Residual_time`
 - `cleaning_duration`（从 self.net 读取）
 - `cleaning_trigger_wafers`（从 self.net 读取）
 - `SCRAP_CLIP_THRESHOLD`
- 单设备工序时间参数：
 - `single_process_time_map.PM1/PM3/PM4`（`route_code=0`）
 - `single_process_time_map.PM1/PM3/PM4/PM6`（`route_code=1`）
 - 工序时间会在环境内部预处理为最接近的 5 的倍数（最小 5）
- 单设备工序时间随机参数（episode 固定采样）：
 - `proc_rand_enabled`
 - `single_proc_time_rand_scale_map.<PM>.{min,max}`（`<PM>` 由路线下的加工腔室集合决定，缺失时视为不扰动）
- 训练脚本参数：
 - `solutions/Continuous_model/train_single.py` 支持 `--place-obs`
 - 支持 `--proc-time-rand-enabled`
 - 不再支持 CLI 最小/最大随机区间覆盖（统一由配置文件控制）
 - 传入后使用 `Env_PN_Single_PlaceObs`；不传时默认 `Env_PN_Single`
- 推理导出参数：
 - `solutions/Continuous_model/export_inference_sequence.py` 支持 `--place-obs`
 - 在 `--device-mode single` 下传入后使用 `Env_PN_Single_PlaceObs`

## Examples
- 正例：`route_code=0` 场景下，使用 36 维 place obs（`LP+TM(8 维)+PM1/3/4`）。
- 正例：`route_code=1` 场景下，使用 45 维 place obs（额外包含 `PM6` 9 维特征）。
- 反例：需要把每片晶圆位置编码成 one-hot 序列时不应使用本环境。

## Edge Cases / Gotchas
- TM 无晶圆时：single 模式 TM 特征为 `[0, 0, 0, 1, 0, 0, 0, 0]`（时间回退 + 去向 one-hot 全 0）；cascade 模式对应 TM 块的去向 one-hot 全为 0，时间特征同 single。
- PM 无晶圆时，工艺与超时相关特征置 0，仅保留清洗相关状态。
- 归一化分母均设置下限 `>=1`，避免除零。
- 若开启工序时间随机扰动，同一 episode 内工序时长固定；不同 episode 才会重新采样。

## Related Docs
- `docs/README.md`
