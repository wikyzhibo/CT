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
 - PM 第 9 维为 `near_cleaning_norm`（固定 `clip window=2`），仅在距离触发清洗 2 片以内提供非零信号。

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
 - `near_cleaning_norm = (1-is_cleaning) * clip((2-r)/2, 0, 1)`，其中 `r=max(N-c,0)`，`N=max(1,cleaning_trigger_wafers)`，`c=max(0,processed_wafer_count)`
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
 - `cleaning_duration` / `cleaning_trigger_wafers`：按腔室从 self.net 的 PM 实例或 `_cleaning_duration_map` / `_cleaning_trigger_map` 读取（支持 per-chamber 配置）
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
- `near_cleaning_norm` 分段语义：`r>=2 -> 0`，`r=1 -> 0.5`，`r=0 -> 1`；`is_cleaning=True` 时强制为 `0`。

## Place 子类与 obs 构造

特征由 Place 子类（SR、TM、PM、LL）的 `get_obs()` 提供。`Place.get_obs_dim()` 返回单库所观测维度（SR 1/0、TM 4+onehot_dim、PM 9、LL 4）；基类默认 `len(self.get_obs())`，子类可覆写。构网时 `build_single_device_net(obs_config=...)` 根据 `obs_config` 创建对应子类。`ClusterTool.get_obs()` 仅负责：1）`_get_obs_place_order()` 确定观测顺序（LP + 运输位 + 腔室）；2）依次调用 `place.get_obs()` 并聚合；`ClusterTool.get_obs_dim()` 对顺序内库所求和 `place.get_obs_dim()`。`step()` 返回 `(done, reward_result, scrap, action_mask, obs)`。Env 使用 `net.get_obs()` 与 `step` 返回值，不再自行构造 obs。详见 `docs/架构.md` 3.2 节。

## Recent Update: Obs Build Path Performance

- What changed:
 - `ClusterTool` 初始化阶段新增观测缓存：`place_by_name`、`obs_place_names`、`obs_places`、`obs_offsets`、`obs_dim`、`obs_specs`、`obs_buffer`。
 - `ClusterTool.get_obs()` 改为预分配 `float32` buffer 的原地写入路径，不再走 `list.extend(...) + np.array(...)`。
 - `Place` 新增 `write_obs(buffer, offset)` 原地写接口；SR/TM/PM/LL 子类已实现固定维度写入，`get_obs()` 仅保留兼容层。
- Why:
 - 降低每步观测构建中的 Python 小对象分配、列表扩容和二次拷贝成本。
 - 避免按名称线性查找 place，改为 O(1) 哈希映射。
- Impact:
 - 观测语义、顺序、维度定义保持不变（仍为 `LP -> TM -> chamber*`，且 `LP_done` 不进入主体观测）。
 - 训练/推理调用方式不变，外部仍通过 `get_obs()` 获取 `np.ndarray`。
- Example:
 - 正例：单步调用 `get_obs()` 时直接复用内部 `obs_buffer` 并按 offset 写入，每个 place 只写自身片段。
 - 反例：每步重建观测顺序并用 `extend + np.array` 做拼装与转换。

## Related Docs
- `docs/README.md`
- `docs/架构.md` Place 类继承结构
