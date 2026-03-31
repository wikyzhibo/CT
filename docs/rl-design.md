### 强化学习方案





#### 观测空间（Observation Space）

**观测维度**：15 维（3 个 wafer × 5 元组）

**设计目标**：只展示机械手能操作到的 wafer 信息，减少观测维度，提高学习效率。

**选择逻辑**：

1. **优先级 1**：收集所有在加工腔室（type=1）和运输位（type=2）中的 wafer
2. **优先级 2**：如果不满 3 个，从 LP（type=3）中取队首的 **1 个** wafer（下一个被操作的）
3. **补零**：如果还不够 3 个，用全零的 5 元组 `[0, 0, 0, 0, 0]` 补齐

**五元组内容**：`(token_id, place_idx, place_type, stay_time, time_to_scrap)`

| 字段            | 含义                       | 示例                                          |
| --------------- | -------------------------- | --------------------------------------------- |
| `token_id`      | wafer 编号                 | 0 ~ n_wafer-1                                 |
| `place_idx`     | 当前所在库所索引           | 2（PM1）                                      |
| `place_type`    | 库所类型                   | 1=加工腔, 2=运输位, 3=起点, 4=终点            |
| `stay_time`     | 在当前库所的停留时间（秒） | 25                                            |
| `time_to_scrap` | 距离报废的剩余时间（秒）   | 5（加工腔室），10（运输位），-1（LP/LP_done） |

**排序规则**：选中的 wafer 按 `token_id` 升序排列后输出（从左到右）。

**计算公式**：

- 加工腔室（type=1）：`time_to_scrap = 20 - (stay_time - proc_time)`
- 运输位（type=2）：`time_to_scrap = 10 - stay_time`
- LP/LP_done：`time_to_scrap = -1`（无报废风险）

**示例观测**：

```python
# 假设系统状态：
# - wafer 0 在 PM1（已加工 25s，proc_time=30s）
# - wafer 1 在 d_PM2（停留 3s）
# - wafer 2 在 LP（队首）
obs = [
    0, 2, 1, 25, 25,   # wafer 0: 在 PM1，停留 25s，距报废 25s
    1, 6, 2, 3, 7,     # wafer 1: 在 d_PM2，停留 3s，距报废 7s
    2, 0, 3, 0, -1,    # wafer 2: 在 LP，无时间约束
]
```

#### 动作空间（Action Space）

**动作数量**：11（6 个变迁 + 5 个 WAIT 档位）

**变迁动作**（0-5）：

| 索引 | 变迁名称        | 含义                         |
| ---- | --------------- | ---------------------------- |
| 0    | `u_LP_PM1`      | 从 LP 卸载晶圆到 PM1 运输位  |
| 1    | `t_PM1`         | 将晶圆从运输位装载到 PM1     |
| 2    | `u_PM1_PM2`     | 从 PM1 卸载晶圆到 PM2 运输位 |
| 3    | `t_PM2`         | 将晶圆从运输位装载到 PM2     |
| 4    | `u_PM2_LP_done` | 从 PM2 卸载晶圆到完成位      |
| 5    | `t_LP_done`     | 将晶圆装载到 LP_done（完成） |

**WAIT 动作**（索引 6-10）：`WAIT_5s / WAIT_10s / WAIT_20s / WAIT_50s / WAIT_100s`。
执行 WAIT 时采用分层规则：

- 当 `wait_duration == 5` 时固定推进 5 秒（不计算 `next_event_delta`）；
- 当 `PM1/PM3/PM4` 任一加工腔室存在“加工完成待取片”晶圆（`stay_time >= processing_time`）时，大于 5 秒的 WAIT 在动作掩码中禁用；
- 其余 WAIT 才执行 `actual_wait = min(wait_duration, next_event_delta)`。

其中 `next_event_delta` 至少考虑：
- 某加工腔室达到“加工完成”时刻；
- 某运输位（`d_TM*`）达到“运输完成（`stay_time >= T_transport`）”时刻；
- 某清洗中的腔室达到“清洗完成”时刻。

这样可避免一次跨越多个关键决策点。

**动作掩码**：通过 `get_action_mask()` 计算当前可用变迁（满足前置条件且容量未满）并直接写 mask；单设备下 WAIT 档位默认进入掩码，但当存在“加工完成待取片”晶圆（`PM1/PM3/PM4` 中有 token 满足 `stay_time >= processing_time`）时，仅保留 `WAIT_5s`。该规则由 `pn_single` 统一生成，`env_single` 与可视化仅消费掩码结果。

**使能条件**：

- 前置库所有足够的 token（`pre <= m`）
- 后置库所容量未满（`m + pst <= capacity`）
- 前置库所的 token 已完成处理（`current_time >= enter_time + proc_time`）

#### 奖励函数（Reward Function）

奖励函数采用**稠密奖励**设计，引导智能体学习高效调度策略。

**正向奖励：**

| 组件           | 含义         | 计算公式                       | 系数       |
| -------------- | ------------ | ------------------------------ | ---------- |
| `proc_reward`  | 加工奖励     | 在加工时间内每秒 +r            | r=2        |
| `safe_reward`  | 安全裕量奖励 | 距离报废还有充足时间时给予奖励 | b_safe=0.5 |
| `finish_bonus` | 完工奖励     | 每片完工 +100 × n_wafer        | done_event_reward=100 |

**惩罚：**

| 组件                 | 含义             | 计算公式                                  | 系数         | 阶段    |
| -------------------- | ---------------- | ----------------------------------------- | ------------ | ------- |
| `scrap_penalty`      | 报废惩罚         | 晶圆超过 `proc_time + 20s` 报废           | scrap_event_penalty=-100 | 全部    |
| `penalty`            | 加工腔室超时惩罚 | 超过 `proc_time + 15s` 后逐秒惩罚         | Q2_p=0.2     | 全部    |
| `transport_penalty`  | 运输位超时惩罚   | 运输位停留超过 `10s` 后逐秒惩罚           | Q1_p=0.2     | Phase 2 |
| `congestion_penalty` | 堵塞预测惩罚     | 上游多个 wafer 即将同时完成但下游容量不足 | c_congest=50 | 全部    |
| `time_cost`          | 时间成本         | 每秒 -0.5，防止躺平                       | time_coef=0.5   | 全部    |

**奖励计算代码**：

```python
total = (proc_reward + safe_reward
         - penalty - transport_penalty - congestion_penalty 
         - time_cost)
if finish:
    total += finish_bonus
if scrap:
    total -= scrap_penalty
```

**关键设计思想**：

- **稠密奖励**：每个时间步都有奖励/惩罚信号，避免稀疏奖励导致的探索困难
- **时间成本**：每秒都在亏钱，鼓励智能体尽快完成任务
- **堵塞预测**：提前检测下游容量不足，避免晶圆在上游等待过久





### 训练：PPO 完整奖励训练

统一使用完整奖励配置（加工腔室超时 + 运输位超时）进行训练。

**奖励组件**：
- `proc_reward`（加工奖励）
- `penalty`（加工腔室超时惩罚）
- `scrap_penalty`（报废惩罚）
- `transport_penalty`（运输位超时惩罚）
- `time_cost`（时间成本）
- `finish_bonus`（完工奖励）

**训练命令**

```bash
# 使用默认配置训练
python -m solutions.PPO.run_ppo

# 使用自定义配置
python -m solutions.PPO.run_ppo --config data/ppo_configs/custom/my_config.json

# 从 checkpoint 继续训练
python -m solutions.PPO.run_ppo --checkpoint solutions/PPO/saved_models/CT_ppo_best.pt
```

### 训练时间评估指标

`train_single.py` 在每个 batch 和训练结束时输出以下时间指标：

**Batch 级别（每轮打印）**：
- `rollout`: rollout 采样耗时（秒）
- `update`: PPO 参数更新耗时（秒）
- `steps/s`: 环境交互速度（env steps / rollout 时间）
- `ETA`: 基于最近 10 个 batch 的滑动平均估算剩余时间

**训练结束汇总**：
- 总训练 wall-clock 时间
- 平均 batch 耗时（rollout vs update 占比百分比）
- 平均 steps/sec

### 核心文件

| 文件                                      | 功能              | 关键类/函数                                      |
| ----------------------------------------- | ----------------- | ------------------------------------------------ |
| `solutions/Continuous_model/pn.py`        | 连续 Petri 网实现 | `Petri` 类，`calc_reward()`, `step()`, `_fire()` |
| `solutions/Continuous_model/construct.py` | 自动构建子网      | `SuperPetriBuilder`, `ModuleSpec`, `RobotSpec`   |
| `solutions/PPO/enviroment.py`             | RL 环境封装       | `Env_PN` 类，`_build_obs()`, `_step()`           |
| `solutions/Continuous_model/test_env.py`  | Pygame 可视化     | `PetriVisualizer` 类                             |
| `solutions/PPO/run_ppo.py`                | 训练脚本          | PPO 训练循环，checkpoint 管理                    |
| `solutions/PPO/validation.py`             | 模型评估          | 策略评估，性能统计                               |

### 配置参数

在 `solutions/Continuous_model/pn.py` 中定义的关键参数：

| 参数              | 值   | 含义                                 |
| ----------------- | ---- | ------------------------------------ |
| `n_wafer`         | 12   | 系统中晶圆数量                       |
| `P_Residual_time` | 15   | 加工腔室报废裕量（秒），超过后报废   |
| `D_Residual_time` | 10   | 运输位超时裕量（秒），超过后逐秒惩罚 |
| `scrap_event_penalty`         | 100  | 报废惩罚值                           |
| `done_event_reward`          | 100  | 每片完工奖励                         |
| `time_coef`       | 0.5  | 时间成本系数（每秒）                 |
| `c_congest`       | 50   | 堵塞预测惩罚系数                     |

