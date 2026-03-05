## Petri 网建模



#### 建模思路

本项目采用**模块化子网建模**方法，使用 `SuperPetriBuilder` 自动构建 Petri 网结构：

1. **模块定义**：每个加工腔室（PM1, PM2）、装载位（LP）、卸载位（LP_done）建模为独立的库所（Place）
2. **机械手建模**：机械手通过 `RobotSpec.reach` 做可达性约束，并通过按机械手分组的运输库所（如 `d_TM1`）承载搬运过程
3. **路径展开**：每条工艺路径（如 `LP → PM1 → PM2 → LP_done`）的每个边 `A → B` 自动展开为 **u-d-t 子结构**：
   - `u_A_B`（卸载变迁）：从源库所 A 卸载晶圆
   - `d_TMx`（运输库所）：晶圆在对应机械手通道中的缓冲位置
   - `t_B`（装载变迁）：将晶圆装载到目标库所 B

**单设备简化规则（Continuous single）**：
- 当且仅当为单机械手、单运输通道（如 `d_TM1`）且需要压缩动作空间时，可将 `u_A_B` 简化为 `u_A`。
- 简化后结构为：`u_A -> d_TM1 -> t_B`，同一源库所可由多个 `t_B` 共享一个上游卸载动作。
- 对于并行目标机台（例如 `PM1 -> [PM3|PM4]`），仅保留一个 `u_PM1`，由后续 `t_PM3`/`t_PM4` 完成分流。
- 该简化不会改变 `d_TM1` 停留时间约束；运输位 dwell 仍在 `t_*` 发射前检查。
- 单设备支持通过 `PetriEnvConfig.single_robot_capacity` 在网络层切换机械手模式：
  - `single_robot_capacity=1`：`d_TM1.capacity=1`（Single Arm）
  - `single_robot_capacity=2`：`d_TM1.capacity=2`（Dual Arm）
- `d_TM1` 使用 FIFO 队列；token 进入 `d_TM1` 时按机器轮换分配 `machine` 标识（单臂固定 1，双臂在 1/2 间交替）。

**示例**：路径 `LP → PM1` 展开为：

```
LP --[消耗]--> u_LP_PM1 --[生产]--> d_TM1 --[消耗]--> t_PM1 --[生产]--> PM1
```

#### 库所类型（Place Types）

系统中的库所根据功能分为 4 种类型（通过 `place.type` 标识）：

| 类型       | 名称      | 示例                    | 特点                            | 时间约束                                |
| ---------- | --------- | ----------------------- | ------------------------------- | --------------------------------------- |
| **type=1** | 加工腔室  | PM1, PM2                | 有处理时间（`processing_time`） | 超过 `proc_time + P_Residual_time` 报废 |
| **type=2** | 运输位    | d_PM1, d_PM2, d_LP_done | 晶圆运输中的缓冲位置            | 超过 `D_Residual_time` 后有超时惩罚     |
| **type=3** | 起点      | LP                      | 晶圆初始位置                    | 无时间约束                              |
| **type=4** | 其他/兼容保留 | （可选）           | 兼容旧模型中的非工艺库所分类    | 无时间约束                              |

#### Token 机制

每个 token 代表一个晶圆，携带以下信息：

- **`token_id`**：晶圆唯一标识（0 ~ n_wafer-1），用于追踪晶圆状态
- **`enter_time`**：进入当前库所的时间戳
- **`stay_time`**：在当前库所的停留时间（`current_time - enter_time`）

Token 在变迁发射时传递 `token_id`，实现晶圆在 Petri 网中的追踪：

```python
# 变迁发射时
consumed_token_ids = [place.head().token_id for place in pre_places]
for p in pst_places:
    place.append(BasedToken(enter_time=new_time, token_id=consumed_token_ids[0]))
```

**注意**：LP 库所中的 wafer 不更新 `stay_time`（因为起点无时间约束）。

#### 构建示例

以下代码展示如何使用 `SuperPetriBuilder` 构建一个包含 LP、PM1、PM2 和 LP_done 的 Petri 网：

```python
from solutions.Continuous_model.construct import SuperPetriBuilder, ModuleSpec, RobotSpec

modules = {
    "LP": ModuleSpec(tokens=12, ptime=0, capacity=12),      # 初始 12 个晶圆
    "LP_done": ModuleSpec(tokens=0, ptime=0, capacity=12),  # 完成位置
    "PM1": ModuleSpec(tokens=0, ptime=30, capacity=2),      # PM1 容量=2，处理时间=30s
    "PM2": ModuleSpec(tokens=0, ptime=80, capacity=1),      # PM2 容量=1，处理时间=80s
}

robots = {
    "TM1": RobotSpec(tokens=1, reach={"LP", "PM1", "PM2", "LP_done"}),  # 1 个机械手
}

routes = [
    ["LP", "PM1", "PM2", "LP_done"],  # 工艺路径
]

builder = SuperPetriBuilder(d_ptime=5, default_ttime=5)
info = builder.build(modules=modules, robots=robots, routes=routes)
```

生成的 Petri 网包含（单机械手示例）：

- **库所**：`LP`, `PM1`, `PM2`, `LP_done`, `d_TM1`
- **变迁**：`u_LP_PM1`, `t_PM1`, `u_PM1_PM2`, `t_PM2`, `u_PM2_LP_done`, `t_LP_done`
