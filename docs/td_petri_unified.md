# Td_petri 全面技术文档 (Unified Documentation)

> **版本**: 2.0 (重构版)  
> **更新日期**: 2026-02-09
> **涵盖范围**: 使用指南、架构设计、实现细节、建模规范

本文档整合了 Td_petri 调度系统的所有相关文档，分为四个部分：
1. **[用户指南](#part-1-用户指南)** - 面向普通使用者，包含快速上手、配置说明。
2. **[架构设计](#part-2-架构设计)** - 核心调度思想、时间模型、动作空间设计。
3. **[实现细节](#part-3-实现细节)** - 代码结构、模块划分、API 说明。
4. **[建模规范](#part-4-建模规范)** - Petri 网构建规则、数学定义。

---

# Part 1: 用户指南

本部分面向普通使用者，旨在帮助你快速上手 Td_petri 调度系统。

## 1.1 系统简介

**Td_petri** (Timed Discrete Petri Net) 是一个用于半导体晶圆生产线（Cluster Tool）的高性能**调度仿真器**。

### 核心功能
- **模拟生产**：模拟晶圆（Wafer）在不同机台（PM）和机械手（Arm）之间的流转过程。
- **验证策略**：支持强化学习算法验证，评估调度效率。
- **发现问题**：严格检查物理限制（如机械手冲突、机台容量），帮助发现死锁或瓶颈。

### 关键概念
1. **晶圆 (Wafer) 与 路线 (Route)**
   - **路线 C**：完整流程（经过所有关键工序：LP1 -> PM7/8 -> LLC -> PM1-4 -> LLD -> PM9/10 -> LP_done）
   - **路线 D**：简化流程（跳过部分工序：LP2 -> PM7/8 -> PM9/10 -> LP_done）

2. **动作链 (Chain)**
   - 系统将多个细微动作（移动、抓取、放置）打包为一个完整的**动作链**。
   - 例如：“把晶圆从 PM1 搬到 PM2” 是一个原子动作，系统会自动处理中间的细节。

3. **资源 (Resources)**
   - **机械手 (Arm)** 和 **机台 (PM)** 都是资源，同一时间只能被一个任务占用。

## 1.2 快速开始

### 运行仿真

```bash
# 运行 PPO 训练（使用默认配置）
python solutions/PPO/run_ppo_tdpn.py

# 验证已有模型
python solutions/PPO/validation.py --env_type CT_v2
```

### Python API 调用

```python
from solutions.Td_petri.tdpn import TimedPetri

# 初始化环境
net = TimedPetri()
obs, mask = net.reset()

# 执行动作
action = 0  # 假设动作 0 可行
mask, obs, time, done, reward = net.step(action)
```

## 1.3 配置指南

你需要通过 **配置文件** 来调整系统参数，无需修改代码。

### 典型配置修改

1. **导出默认配置**：
   ```python
   from solutions.Td_petri.core.config import PetriConfig
   PetriConfig.default().to_json('my_config.json')
   ```

2. **修改 `my_config.json`**：
   ```json
   {
     "history_length": 100,
     "reward_weights": [0, 10, 30, 100, 800, 980, 1000],
     "modules": {
       "LP1": {"capacity": 25},
       "PM7": {"capacity": 1}
     }
   }
   ```

3. **加载配置运行**：
   ```bash
   python solutions/PPO/run_ppo_tdpn.py --config my_config.json
   ```

---

# Part 2: 架构设计

本部分详细介绍系统的内部运作机制和设计思想。

## 2.1 核心思想

与传统的连续时间模型不同，Td_petri 采用 **离散事件仿真** 结合 **Chain-based 动作空间**。

### 时间离散化
- **时间单位**：整数秒。
- **变迁具有时间属性**：只有变迁（Transition）消耗时间，库所（Place）仅表示状态。
- **状态推进**：当变迁触发时，系统时间直接跳跃到变迁结束时刻。

### 资源占用时间轴 (Resource Timeline)
系统使用时间区间列表管理资源占用，支持**预约**和**冲突检测**。

```python
self.res_occ = {
    "ARM1": [Interval(start=10, end=15), Interval(start=20, end=25)],
    "PM7":  [Interval(start=10, end=INF)]  # 开放区间，表示正在加工
}
```
- **开放区间**：加工模块使用 `[start, INF)` 表示占用，直到离开时才关闭区间。

## 2.2 动作空间设计 (Chain-based)

### 动作定义
动作不再是单一的变迁，而是一条完整的执行链（Chain）。每个 Chain 对应路线中的一个阶段（Stage）。

- **Route C (完整路线)** 分为多个阶段：
  - Stage 0: LP1 -> AL
  - Stage 1: AL -> LLA
  - Stage 2: LLA -> PM7/PM8 (并行选择)
  - ...

- **Route D (简化路线)** 类似。

### 并行选择
对于并行阶段（如 LLA -> PM7 或 LLA -> PM8），系统会生成多个可选的 Chain，智能体需要从中选择一个。

### 动作掩码 (Action Masking)
在每一步，系统会计算哪些 Chain 是可执行的（Enable）：
1. **结构检查**：Petri 网结构是否允许（Token 是否到位）。
2. **颜色检查**：Token 的路线颜色是否匹配。
3. **Dry-run (预演)**：尝试在该时间点插入资源占用，检查是否有资源冲突。只有通过预演的 Chain 才是真正可执行的。

## 2.3 观测空间 (Observation Space)

观测维度：`82` 维（以默认配置为例）。

1. **P_READY 状态 (32维)**：
   - 统计每个 P_READY 库所中，Route C 和 Route D 的晶圆数量。
   - `obs1[16]` (Route C) + `obs2[16]` (Route D)。

2. **动作历史 (50维)**：
   - 记录最近执的 50 个动作 ID。

## 2.4 奖励机制 (Reward)

采用稠密奖励设计，引导智能体高效调度。

- **进度奖励**：晶圆每完成一个阶段，获得对应阶段的奖励分。
- **时间惩罚**：每过一秒，给予微小负奖励，鼓励快速完成。
- **完成奖励**：所有晶圆加工完成时给予大额奖励。

---

# Part 3: 实现细节

本部分介绍代码结构和模块划分，适合开发者阅读。

## 3.1 模块结构

系统已重构为模块化架构，位于 `solutions/Td_petri/`：

```
solutions/Td_petri/
├── tdpn.py                    # [主入口] TimedPetri 类
├── core/
│   └── config.py             # 配置管理 (PetriConfig)
├── resources/
│   ├── interval_utils.py     # 时间区间算法
│   └── resource_manager.py   # 资源占用管理
└── rl/
    ├── path_registry.py      # 路径定义 (PathRegistry)
    ├── action_space.py       # 动作空间构建
    ├── observation.py        # 观测向量生成
    └── reward.py             # 奖励计算
```

## 3.2 关键类说明

### `TimedPetri` (tdpn.py)
系统的核心类，继承自 `Env`。
- `reset()`: 初始化状态。
- `step(action)`: 执行一步仿真。
- `get_enable_t()`: 获取当前可执行动作列表。

### `PetriConfig` (core/config.py)
数据类，管理所有静态参数。支持从 JSON 加载和保存。

### `ResourceManager` (resources/resource_manager.py)
负责管理所有资源的 `res_occ` 表。
- `find_earliest_slot()`: 寻找最早可用的时间槽。
- `allocate_resource()`: 占用资源。

### `PathRegistry` (rl/path_registry.py)
定义了 Route C 和 Route D 的详细路径结构，是路径信息的唯一权威来源。

## 3.3 扩展指南

- **添加新路径**：修改 `rl/path_registry.py`，并在 `PetriConfig` 中注册。
- **修改奖励函数**：修改 `rl/reward.py` 中的 `RewardCalculator` 类。
- **调整观测**：修改 `rl/observation.py` 中的 `ObservationBuilder` 类。

---

# Part 4: 建模规范

本部分定义了 Petri 网构建器 (`SuperPetriBuilderV3`) 的详细规范。

## 4.1 基本要素

- **库所 (Place)**:
  - `P_READY__{module}`: 模块加工完成，等待取出。
  - `P_IN__{module}`: 模块正在加工。
  - `c_{module}`: 模块容量控制。
  - `r__{arm}`: 机械手资源。

- **变迁 (Transition)**:
  - `PROC__{module}`: 加工过程。
  - `{arm}_PICK/.../MOVE/.../LOAD`: 搬运过程。

## 4.2 搬运动作序列规范

对于每个晶圆搬运边 A→B，构建器必须生成完整的四步动作序列：

1. **ROTATE** (3s): 机械手旋转到源模块 A。
   - 消耗: `r__{arm}`
   - 生成: `P_ARM_AT__{arm}__{a}`

2. **PICK** (5s): 从 A 取出晶圆。
   - 消耗: `P_ARM_AT__{arm}__{a}` + `A_READY`
   - 释放: `c_cap(a)` (释放 A 的容量)

3. **MOVE** (3s): 移动到目标模块 B。

4. **LOAD** (5s): 放入 B。
   - 生成: `B_IN`
   - 归还: `r__{arm}` (释放机械手)

## 4.3 资源竞争建模

- **机械手资源库所** (`r__ARM1` 等) 初始 Token 数等于机械手容量。
- 任何需要机械手的变迁（ROTATE, PICK, MOVE, LOAD）都必须确保持有或消耗相应的资源 Token。
- **容量库所** (`c_{module}`) 初始 Token 数等于模块容量，用于限制同时在内的晶圆数量。

---

**文档结束**
