# petri-net-construction Specification

## Purpose
定义 Petri 网构建器 `SuperPetriBuilderV3` 的规范，用于为组合设备（Cluster Tool）生成时间 Petri 网模型。构建器支持机器手旋转动作建模、资源竞争建模和同步模式，能够生成完整的四步搬运动作序列（ROTATE、PICK、MOVE、LOAD）。
## Requirements
### Requirement: 机器手资源库所
Petri 网构建器 SHALL 为每个机器手创建资源库所，表示机器手的可用性。

#### Scenario: 创建机器手资源库所
- **WHEN** 构建 Petri 网
- **THEN** 为 ARM1/ARM2/ARM3 分别创建资源库所 `r__ARM1`, `r__ARM2`, `r__ARM3`

#### Scenario: 设置初始 token 数
- **WHEN** 初始化机器手资源库所
- **THEN** 根据机器手类型设置初始 token 数：ARM1=1, ARM2=2, ARM3=2

### Requirement: 机器手旋转动作建模
Petri 网构建器 SHALL 为每个晶圆搬运边（A→B）生成 ROTATE 变迁，表示机器手旋转到源模块的动作。

#### Scenario: 生成旋转变迁
- **WHEN** 构建从模块 A 到模块 B 的搬运边
- **THEN** 生成名为 `{arm}_ROTATE__{a}__TO__{b}` 的变迁，时长为 3 秒

#### Scenario: 旋转变迁的前置库所
- **WHEN** 生成 ROTATE 变迁
- **THEN** 前置库所为机器手资源库所 `r__{arm}`

#### Scenario: 旋转变迁的后置库所
- **WHEN** 生成 ROTATE 变迁
- **THEN** 后置库所为 `P_ARM_AT__{arm}__{a}`，表示机器手已在模块 A 处就绪

### Requirement: 旋转时间常量
构建器 SHALL 定义旋转时间常量 `T_ROTATE = 3` 秒。

#### Scenario: 使用旋转时间
- **WHEN** 创建 ROTATE 变迁
- **THEN** 该变迁的时长为 `T_ROTATE` 秒（3 秒）

### Requirement: 机器手就位状态库所
构建器 SHALL 为每个 ROTATE 变迁创建对应的就位状态库所，表示机器手已旋转到源模块。

#### Scenario: 生成机器手就位库所
- **WHEN** 为边 A→B 生成 ROTATE 变迁
- **THEN** 创建名为 `P_ARM_AT__{arm}__{a}` 的库所，容量为无限（INF）

#### Scenario: ROTATE 弧连接
- **WHEN** 构建 ROTATE 变迁的弧
- **THEN** 连接关系为：`r__{arm} → ROTATE → P_ARM_AT__{arm}__{a}`

### Requirement: PICK 变迁同步模式
构建器 SHALL 修改 PICK 变迁，使其同时消耗晶圆就绪库所和机器手就位库所的 token。

#### Scenario: PICK 的两个前置库所
- **WHEN** 生成 PICK 变迁（从 A 到 B）
- **THEN** PICK 有两个前置库所：
  - `A_READY`（晶圆在 A 处就绪）
  - `P_ARM_AT__{arm}__{a}`（机器手已在 A 处）

#### Scenario: PICK 的弧连接
- **WHEN** 构建 PICK 变迁的输入弧
- **THEN** 添加两条弧：
  - `A_READY → tr_pick`
  - `P_ARM_AT__{arm}__{a} → tr_pick`

#### Scenario: PICK 时释放源模块容量
- **WHEN** 生成 PICK 变迁且 A 不是起点模块（LP1/LP2）
- **THEN** 添加弧 `tr_pick → c_cap(a)`，释放源模块 A 的容量

### Requirement: 机器手资源生命周期
构建器 SHALL 在 LOAD 变迁完成后归还机器手资源，完成机器手的使用周期。

#### Scenario: LOAD 后归还机器手
- **WHEN** 生成 LOAD 变迁
- **THEN** 添加弧 `tr_load → r__{arm}`，归还机器手资源

#### Scenario: 机器手完整周期
- **WHEN** 追踪机器手资源的流动
- **THEN** 流程为：`r__{arm} → ROTATE → P_ARM_AT → PICK → HAND → MOVE → AT_B → LOAD → r__{arm}`

### Requirement: 容量控制调整
构建器 SHALL 在 ROTATE 变迁时消耗目标模块容量（若适用）。

#### Scenario: ROTATE 时消耗目标容量
- **WHEN** 生成 ROTATE 变迁（从 A 到 B）且 B 不是终点模块（LP_done）
- **THEN** 添加弧 `c_cap(b) → tr_rotate`，消耗目标模块 B 的容量

### Requirement: 四步搬运动作序列
对于每个晶圆搬运边 A→B，构建器 SHALL 生成完整的四步动作序列：ROTATE、PICK、MOVE、LOAD。

#### Scenario: 完整动作序列
- **WHEN** 为边 A→B 生成搬运动作
- **THEN** 生成 4 个变迁：
  - `{arm}_ROTATE__{a}__TO__{b}` (3s)
  - `{arm}_PICK__{a}__TO__{b}` (5s)
  - `{arm}_MOVE__{a}__TO__{b}` (3s)
  - `{arm}_LOAD__{a}__TO__{b}` (5s)

#### Scenario: Petri 网结构（同步模式）
- **WHEN** 查看完整的 Petri 网结构
- **THEN** 结构为：
```
r__{arm} → ROTATE → P_ARM_AT__{arm}__{a} ↘
                                           PICK → HAND → MOVE → AT_B → LOAD → B_IN
A_READY ─────────────────────────────────↗                             ↓
                                                                    r__{arm}
```

#### Scenario: 并行性
- **WHEN** 晶圆在 A 处加工时
- **THEN** 机器手可以并行旋转到 A，两者在 PICK 处同步

### Requirement: 机器手资源库所配置
构建器 SHALL 提供机器手资源配置接口或使用默认配置。

#### Scenario: 默认机器手配置
- **WHEN** 未提供机器手配置时
- **THEN** 使用默认配置：
  - ARM1: 容量 1
  - ARM2: 容量 2
  - ARM3: 容量 2

#### Scenario: 机器手资源库所初始化
- **WHEN** 生成初始标识 m0
- **THEN** 机器手资源库所的初始 token 数等于其容量

### Requirement: 向后兼容性
构建器的 `build()` 方法签名 SHALL 保持不变，确保调用代码无需修改。

#### Scenario: 接口不变
- **WHEN** 调用 `builder.build(modules, routes)`
- **THEN** 方法签名与原来相同，返回值结构不变（包含 m0, pre, pst, t_time 等字段）

#### Scenario: 自动发现新库所和变迁
- **WHEN** 下游代码通过 `id2p_name` 和 `id2t_name` 遍历
- **THEN** 可以自动发现新增的机器手资源库所、P_ARM_AT 库所和 ROTATE 变迁

