## Petri 网建模



#### 建模思路

本项目采用**模块化子网建模**方法，使用 `SuperPetriBuilder` 自动构建 Petri 网结构：

1. **模块定义**：每个加工腔室（PM1, PM2）、装载位（LP）、卸载位（LP_done）建模为独立的库所（Place）
2. **资源建模**：机械手（TM1）建模为资源库所（`r_TM1`），容量等于机械手数量
3. **路径展开**：每条工艺路径（如 `LP → PM1 → PM2 → LP_done`）的每个边 `A → B` 自动展开为 **u-d-t 子结构**：
   - `u_A_B`（卸载变迁）：从源库所 A 卸载晶圆，消耗机械手资源
   - `d_B`（运输库所）：晶圆在运输过程中的缓冲位置
   - `t_B`（装载变迁）：将晶圆装载到目标库所 B，归还机械手资源

**示例**：路径 `LP → PM1` 展开为：

```
LP --[消耗]--> u_LP_PM1 --[生产]--> d_PM1 --[消耗]--> t_PM1 --[生产]--> PM1
r_TM1 --[消耗]--> u_LP_PM1                    t_PM1 --[生产]--> r_TM1
```

#### 库所类型（Place Types）

系统中的库所根据功能分为 4 种类型（通过 `place.type` 标识）：

| 类型       | 名称      | 示例                    | 特点                            | 时间约束                                |
| ---------- | --------- | ----------------------- | ------------------------------- | --------------------------------------- |
| **type=1** | 加工腔室  | PM1, PM2                | 有处理时间（`processing_time`） | 超过 `proc_time + P_Residual_time` 报废 |
| **type=2** | 运输位    | d_PM1, d_PM2, d_LP_done | 晶圆运输中的缓冲位置            | 超过 `D_Residual_time` 后有超时惩罚     |
| **type=3** | 起点      | LP                      | 晶圆初始位置                    | 无时间约束                              |
| **type=4** | 资源/终点 | r_TM1, LP_done          | 机械手资源或完成位置            | 无时间约束                              |

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

生成的 Petri 网包含：

- **库所**：`LP`, `PM1`, `PM2`, `LP_done`, `r_TM1`, `d_PM1`, `d_PM2`, `d_LP_done`
- **变迁**：`u_LP_PM1`, `t_PM1`, `u_PM1_PM2`, `t_PM2`, `u_PM2_LP_done`, `t_LP_done`