# CHANGELOG

## 2026-03-19

### 单设备路线参数与节拍输入一致性严格校验 (2026-03-19)
- **What changed**：`solutions/Continuous_model/pn_single.py` 对 `device_mode/single_route_code` 增加严格规范化与合法性校验：`device_mode` 仅允许 `single/cascade`，`route_code` 强制转 `int` 并按模式校验（single: `0/1`，cascade: `1/2/3/4/5`），非法值直接抛 `ValueError`，不再静默回退。新增 `_episode_proc_time_map` 一致性守卫（必须与当前路线 `chambers` 完全一致）以及节拍分析前的 stage-工时一致性检查（越界/缺失会带 `device_mode/route_code/stage` 明细报错）。
- **Why**：防止路线配置与工时映射不一致时继续运行，导致节拍计算错误且难以定位（例如 route5 混入非本路线腔室）。
- **Impact**：配置错误将前移到初始化阶段暴露；合法配置行为不变。`route_code=\"5\"` 这类字符串输入会被规范化为整数后按 route5 执行。同步更新 `takt_cycle_analyzer.py` 的 stage 级错误上下文与 `tests/test_single_route_code.py` 覆盖用例。

## 2026-03-18

### 单设备 PM 第 9 维改为临近清洗分数（clip window=2）(2026-03-18)
- **What changed**：`solutions/Continuous_model/pn.py` 中 `PM.get_obs()` 第 9 维由 `remaining_runs_before_clean_norm` 改为 `near_cleaning_norm`，公式为 `near_cleaning_norm = (1-is_cleaning) * clip((2-r)/2, 0, 1)`，其中 `r=max(N-c,0)`、`N=max(1,cleaning_trigger_wafers)`、`c=max(0,processed_wafer_count)`。
- **Why**：将清洗相位特征压缩到“触发前 2 片窗口”，减少对绝对触发阈值的绑定，提升跨 `cleaning_trigger_wafers` 的泛化稳定性。
- **Impact**：PM 观测维度与拼接顺序保持不变（仍为 9 维且位置不变）；第 9 维语义更新为分段值 `r>=2 -> 0`、`r=1 -> 0.5`、`r=0 -> 1`，且 `is_cleaning=True` 时强制为 `0`。同步更新 `env_place_obs.md`、`continuous_solution_design.md`、`pn_api.md` 与测试用例。

### Cascade 新增 route_code=5（路线 D）(2026-03-18)
- **What changed**：`construct_single/pn_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=5`，路径为 `LP -> PM7/PM8(70) -> PM9/PM10(200) -> LP_done`；该路线不生成 `LLC/LLD/PM1/PM2/PM3/PM4` 相关变迁。
- **Why**：需要支持一条不经过中段缓冲与前段加工腔室的直连级联工艺模板（路线 D）。
- **Impact**：级联模式可通过 `single_route_code=1/2/3/4/5` 切换模板；`route_code=4` 的循环路线与 `route4_takt_interval` 行为保持不变，`route_code=5` 不启用该手动节拍门控。

### Cascade 新增 route_code=4 循环路线与手动节拍 (2026-03-18)
- **What changed**：`construct_single/pn_single` 在 `single_device_mode=cascade` 下新增 `single_route_code=4`，路径为 `LP -> [PM7 -> PM8 -> LLC -> LLD] * 5 -> LP_done`；不再生成 `PM1/PM2/PM3/PM4/PM9/PM10` 相关变迁。新增 `route4_takt_interval`，并在 `takt_cycle_analyzer.py` 增加 `build_fixed_takt_result()` 供 route4 使用固定节拍门控。
- **Why**：需要一条严格串行、固定循环次数的 cascade 工艺模板，并允许 route4 单独配置发片节拍而不依赖自动节拍分析。
- **Impact**：级联模式可通过 `single_route_code=1/2/3/4` 切换模板；`route_code=4` 下 `u_LLD` 会按 token 路由队列在前 4 轮回 `PM7`、第 5 轮去 `LP_done`。`route4_takt_interval > 0` 时 `u_LP` 按固定间隔门控，`<=0` 不门控。

## 2026-03-17

### Ultra 并行采样子环境终止即自重置修复 (2026-03-17)
- **What changed**：`env_single.py` 的 `FastEnvWrapper` 终止判定改为 `scrap/finish/terminated` 联合触发；并新增 `net._last_state_scan.is_scrap` 兜底，避免 `stop_on_scrap=False` 时 scrap 标志在并行采样链路中丢失。`VectorEnv` 下仅对触发终止的子环境执行自动 `reset`。
- **Why**：修复并行 rollout 中“某个子环境发生 scrap 但未被视为终止，导致不重置并持续污染轨迹”的问题。
- **Impact**：`collect_rollout_ultra()` 的 `dones` 现在会正确覆盖 scrap/finish 终止；GAE 与回报截断按子环境独立生效，不会影响其他并行子环境继续采样。
- **Examples**：`--rollout-n-envs 8` 训练时，任一子环境触发 scrap 或 finish 后会立刻进入下一回合起点，其余 7 个子环境保持当前轨迹继续运行。

### 单设备同步取片 resident scrap 撤销修复 (2026-03-17)
- **What changed**：`pn_single.step()` 在非 WAIT 分支保留“先 `_advance_and_compute_reward` 再 `_fire`”顺序；新增同步撤销判定：若本步 `u_*` 已取走与 `scan_info.scrap_info` 同 `token_id` 且同源腔室的 resident wafer，则撤销本步 scrap（不终止、不追加 `scrap_penalty`）。
- **Why**：修复“晶圆在驻留边界/超界时，本步已同步取片却仍被先判 scrap”导致的误终止与误扣分。
- **Impact**：行为仅影响单设备非 WAIT 同步取片场景；WAIT 分支、qtime 统计、mask/obs 构造保持不变。

### 可视化模型推理切换到 `env.net.get_obs()` 直连 (2026-03-17)
- **What changed**：`visualization/main.py` 的 Model 推理入口（单动作/并发）统一改为直接读取 `env.net.get_obs()`，移除对 `Env_PN_Single._build_obs()` 的依赖。
- **Why**：单设备观测已统一由 `pn_single` 构网层输出，继续依赖环境私有方法会导致接口漂移与运行时错误。
- **Impact**：可视化界面的 `Model Step/Auto` 在 single/cascade 模式下不再触发 `_build_obs` 缺失报错；外部脚本若仍调用 `_build_obs()` 需迁移到 `env.net.get_obs()`。

### train_single 收敛为 Ultra-only 并修复 CPU 卡顿路径 (2026-03-17)
- **What changed**：`train_single.py` 移除 `single collector`、`blame`、`benchmark` 相关训练入口与代码分支，训练主循环固定为 ultra rollout；GAE 在 CPU 上强制走 eager（仅 CUDA 尝试 compile）。
- **Why**：简化分支可降低维护成本；同时避免本地 CPU 训练触发 compile 首次编译开销，出现“卡住/长时间无响应”。
- **Impact**：CLI 参数收敛为 `--device/--compute-device/--checkpoint/--proc-time-rand-enabled/--rollout-n-envs`；训练行为更可预测，CPU 本地调试路径更稳定。

### train_single 更新阶段极限优化（Batched PPO + 长轨迹连续采样）(2026-03-17)
- **What changed**：`train_single.py` 更新路径改为纯 `dict[tensor]`，移除 rollout->TensorDict->tensor 往返；PPO 更新由 `epoch + minibatch` 双层循环改为“每个 epoch 单次大 batch forward”；策略更新的 `log_prob/entropy` 改为 fused `masked log_softmax` 计算（不再构造 `MaskedCategorical`）；GAE 改为 `[T,N]` 计算并采用 `torch.compile` 优先、失败自动回退 eager。
- **Why**：训练瓶颈位于 update 阶段，原实现小 batch 多次前向与 Python 循环开销过高，GPU 利用率偏低。
- **Impact**：形成“CPU rollout + 单次 CPU->GPU 搬运 + GPU 全并行 update”数据流；ultra/single collector 均支持跨 batch 连续轨迹（不强制 reset），长 episode（1000+ steps）训练稳定性更好。

### 单设备 Ultra Rollout 高性能采样链路 (2026-03-17)
- **What changed**：`env_single.py` 新增 `FastEnvWrapper` 与 `VectorEnv`，统一 `reset()/step()` 为纯数组接口；`train_single.py` 新增 `collect_rollout_ultra()`（纯 tensor 预分配、CPU rollout、手写 masked 采样），并将训练主循环默认 collector 切到 `ultra`（支持 `--collector ultra|single`、`--rollout-n-envs`、`--rollout-device`）；`pn_single.py` 新增 `step_core_numba/step_core_batch_numba`（纯 numpy 结构可使能核心）。
- **Why**：原 rollout 路径依赖 TensorDict clone / dispatch 与频繁对象构造，采样吞吐显著低于 `env.step` 的 CPU 峰值。
- **Impact**：可用 `python -m solutions.Continuous_model.train_single --benchmark-ultra --benchmark-envs 1,8` 直接对比 baseline/ultra 的 steps/sec；训练默认走 ultra 采样，若需要旧逻辑可显式传 `--collector single`，`--blame` 打开时也会自动回退 single collector。

### 单设备使能判定切到 token 扫描快路径 (2026-03-17)
- **What changed**：`pn_single.get_enable_t` 从“逐变迁两阶段扫描”改为“先检查 `u_LP`，再扫描 token 生成 `u_*/t_*` 候选”；新增运行时 token 池与变迁索引缓存；`_check_scrap` 改为按 token 剩余时间判定（`remaining < -P_Residual_time`）。
- **Why**：`get_enable_t` 是 `step` 热路径，逐变迁做重复结构性判断与目标解析开销较高；token 扫描在单臂场景下可减少无效判定。
- **Impact**：单设备当前按“单臂 + 非 FIFO + unit-capacity（除 LP/LP_done 外）”临时策略运行；双臂锁定规则不再作为默认路径。

## 2026-03-16

### 单设备 t_* 路由改为 token 队列门控 (2026-03-16)
- **What changed**：`pn_single/construct_single` 新增 token 路由队列模板（`route_queue + route_head_idx`）与 `t_*` 路由码映射；`pn_single.get_enable_t` 与 `get_enable_actions_with_reasons` 的路由判定改为读取运输位队首 token 的当前队头门控。
- **Why**：`get_enable_t` 是热路径，原 `where + pre_color` 的颜色切片判定在每步都有额外矩阵开销；改为队头码匹配可减少分支与切片成本，并保持路径语义。
- **Impact**：仅 `t_*` 受路由门控（支持 `-1` 通配、单码、多码集合）；`u_*` 不再做路由门控，但 token 每次 fire 仍推进一次队头（`u_*` 步通常对应 `-1` 占位）。

## 2026-03-14

### 单设备 wait 截断新增运输完成事件 (2026-03-14)
- **What changed**：`pn_single.get_next_event_delta()` 在保留“加工完成”事件的同时，新增 `d_TM*` 运输位“达到 `T_transport`”作为关键事件；`step(wait)` 的长 WAIT 现在会被运输完成时刻截断。
- **Why**：当晶圆已在运输位停留接近运输完成时，继续整段长 WAIT 会跨过关键放片决策点，不利于调度策略及时响应。
- **Impact**：除 `WAIT_5s` 仍固定 5 秒外，其他 WAIT 的 `actual_wait=min(requested_wait,next_event_delta)` 中 `next_event_delta` 候选扩展为“加工完成 + 运输完成 + 清洗完成”，长 WAIT 截断会更及时。

## 2026-03-10

### train_single 新增 --blame 参数 (2026-03-16)
- **What changed**：`train_single.py` 新增命令行参数 `--blame`。传入时在 episode 结束后执行二次追责（`blame_release_violations` 回填惩罚）；不传则不进行二次追责。
- **Why**：支持按需开启或关闭二次追责，便于对比实验与调试。
- **Impact**：默认不传 `--blame` 时不执行追责；需与旧版“始终追责”行为一致时请显式加上 `--blame`。

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
