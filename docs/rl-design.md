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

**动作数量**：7（6 个变迁 + 1 个 WAIT）

**变迁动作**（0-5）：

| 索引 | 变迁名称        | 含义                         |
| ---- | --------------- | ---------------------------- |
| 0    | `u_LP_PM1`      | 从 LP 卸载晶圆到 PM1 运输位  |
| 1    | `t_PM1`         | 将晶圆从运输位装载到 PM1     |
| 2    | `u_PM1_PM2`     | 从 PM1 卸载晶圆到 PM2 运输位 |
| 3    | `t_PM2`         | 将晶圆从运输位装载到 PM2     |
| 4    | `u_PM2_LP_done` | 从 PM2 卸载晶圆到完成位      |
| 5    | `t_LP_done`     | 将晶圆装载到 LP_done（完成） |

**WAIT 动作**（索引 6 = `net.T`）：时间推进 5 秒，不执行任何变迁。

**动作掩码**：通过 `get_enable_t()` 计算当前可用的变迁（满足前置条件且容量未满），WAIT 动作始终可用。

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
| `finish_bonus` | 完工奖励     | 每片完工 +100 × n_wafer        | R_done=100 |

**惩罚：**

| 组件                 | 含义             | 计算公式                                  | 系数         | 阶段    |
| -------------------- | ---------------- | ----------------------------------------- | ------------ | ------- |
| `scrap_penalty`      | 报废惩罚         | 晶圆超过 `proc_time + 20s` 报废           | R_scrap=-100 | 全部    |
| `penalty`            | 加工腔室超时惩罚 | 超过 `proc_time + 15s` 后逐秒惩罚         | Q2_p=0.2     | 全部    |
| `transport_penalty`  | 运输位超时惩罚   | 运输位停留超过 `10s` 后逐秒惩罚           | Q1_p=0.2     | Phase 2 |
| `congestion_penalty` | 堵塞预测惩罚     | 上游多个 wafer 即将同时完成但下游容量不足 | c_congest=50 | 全部    |
| `time_cost`          | 时间成本         | 每秒 -0.5，防止躺平                       | c_time=0.5   | 全部    |

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





### 训练：两阶段课程学习训练

#### 训练阶段划分

为降低强化学习的难度，采用**两阶段课程学习**（Curriculum Learning）策略：

**Phase 1：仅考虑加工腔室报废惩罚**

- **训练目标**：学习避免晶圆在 PM1/PM2 中超时报废
- **奖励组件**：
  - `proc_reward`（加工奖励）
  - `penalty`（加工腔室超时惩罚）
  - `scrap_penalty`（报废惩罚）
  - `time_cost`（时间成本）
  - `finish_bonus`（完工奖励）
- **不包含**：运输位超时惩罚（`transport_penalty` = 0）
- **Checkpoint**：`solutions/PPO/saved_models/CT_phase1_latest.pt`
- **训练参数**：`training_phase=1`

**Phase 2：完整奖励（加工腔室 + 运输位超时）**

- **训练目标**：在避免报废的基础上，优化运输位停留时间
- **奖励组件**：Phase 1 的所有组件 + `transport_penalty`（运输位超时惩罚）
- **初始化**：加载 Phase 1 的 checkpoint 继续训练
- **Checkpoint**：`solutions/PPO/saved_models/CT_phase2_latest.pt`
- **训练参数**：`training_phase=2`

#### 训练命令

```bash
# Phase 1: 仅报废惩罚
python -m solutions.PPO.run_ppo --phase 1

# Phase 2: 加载 Phase 1 checkpoint 继续训练
python -m solutions.PPO.run_ppo --phase 2

# 自动两阶段训练（先 Phase 1，再 Phase 2）
python -m solutions.PPO.run_ppo --auto-phase2

# 使用可视化工具测试模型
python -m solutions.Continuous_model.test_env --model solutions/PPO/saved_models/CT_phase2_latest.pt
```

#### 设计动机

**为什么需要两阶段训练？**

1. **降低学习难度**：
   - 同时学习报废约束和运输超时约束会导致奖励信号复杂，难以收敛
   - Phase 1 专注于主要约束（报废），简化学习目标
   - Phase 2 在已有基础上微调次要约束（运输超时）

2. **加速收敛**：
   - Phase 1 提供良好的策略初始化（已学会避免报废）
   - Phase 2 从非随机策略开始，收敛速度更快

3. **避免局部最优**：
   - 分阶段学习防止智能体陷入"只优化某个约束而忽略其他约束"的局部最优
   - Phase 1 确保基础约束满足，Phase 2 在此基础上优化性能

**实验结果**：

- Phase 1 训练后，智能体基本可以避免晶圆报废（报废率 < 5%）
- Phase 2 训练后，运输位停留时间显著减少，makespan 进一步优化

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
| `R_scrap`         | 100  | 报废惩罚值                           |
| `R_done`          | 100  | 每片完工奖励                         |
| `c_time`          | 0.5  | 时间成本系数（每秒）                 |
| `MAX_WAIT_STEP`   | 20   | WAIT 动作最大跳跃时间（秒）          |
| `c_congest`       | 50   | 堵塞预测惩罚系数                     |

