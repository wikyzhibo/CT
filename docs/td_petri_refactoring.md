# Td_petri 重构说明

> **重构日期**: 2026-02-08  
> **版本**: 2.0 (重构版)

## 概述

Td_petri 系统已完成模块化重构，将原本1484行的单一文件拆分为多个职责清晰的模块，提高了代码的可维护性、可测试性和可扩展性。

## 重构亮点

### 🎯 核心改进

1. **解决路径定义重复** - `PathRegistry` 作为路径定义的唯一权威来源
2. **模块化架构** - 按职责拆分为7个独立模块
3. **配置灵活性** - 支持JSON配置文件
4. **向后兼容** - 保持所有公共接口不变

### 📦 新模块结构

```
solutions/Td_petri/
├── tdpn.py                    # 主文件（已重构）
├── core/
│   └── config.py             # 配置管理
├── resources/
│   ├── interval_utils.py     # 区间工具
│   └── resource_manager.py   # 资源管理
└── rl/
    ├── path_registry.py      # 路径注册表
    ├── action_space.py       # 动作空间构建
    ├── observation.py        # 观测构建
    └── reward.py             # 奖励计算
```

## 使用方法

### 基本使用（向后兼容）

```python
from solutions.Td_petri.tdpn import TimedPetri

# 方式1：使用默认配置
net = TimedPetri()
obs, mask = net.reset()

# 训练循环
done = False
while not done:
    valid_actions = np.where(mask)[0]
    action = select_action(obs, valid_actions)
    mask, obs, time, done, reward = net.step(action)
```

### 使用自定义配置

```python
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig

# 方式2：从JSON加载配置
config = PetriConfig.from_json('data/my_config.json')
net = TimedPetri(config)

# 方式3：修改默认配置
config = PetriConfig.default()
config.history_length = 100
config.reward_weights = [0, 10, 30, 100, 800, 980, 1000]
net = TimedPetri(config)
```

### 保存和加载配置

```python
from solutions.Td_petri.core.config import PetriConfig

# 保存配置
config = PetriConfig.default()
config.to_json('data/petri_configs/my_config.json')

# 加载配置
config = PetriConfig.from_json('data/petri_configs/my_config.json')
```

## 模块详解

### 1. core/config.py - 配置管理

**PetriConfig** 类集中管理所有配置参数：

- `modules`: 模块规格（初始token数、容量）
- `routes`: 路由定义
- `parallel_groups`: 并行机器组
- `stage_capacity`: 各阶段容量
- `processing_time`: 处理时间
- `history_length`: 观测历史长度
- `reward_weights`: 奖励权重

### 2. resources/ - 资源管理

#### interval_utils.py
- `Interval`: 时间区间数据结构
- `_first_free_time_at()`: 查找可用时间槽
- `_first_free_time_open()`: 开放区间处理
- `_insert_interval_sorted()`: 有序插入区间

#### resource_manager.py
- `ResourceManager`: 资源占用管理器
  - `allocate_resource()`: 分配资源
  - `close_open_interval()`: 关闭开放区间
  - `find_earliest_slot()`: 查找最早可用时间
  - `calculate_utilization()`: 计算利用率

### 3. rl/ - 强化学习组件

#### path_registry.py
- `PathRegistry`: **路径定义的唯一来源**
  - `pathC`: Route C (LP1 完整路径)
  - `pathD`: Route D (LP2 简化路径)
  - `get_path_indices()`: 转换为索引
  - `get_all_paths()`: 获取所有路径

#### action_space.py
- `ActionSpaceBuilder`: 动作空间构建器
  - 自动去重共享的chain
  - 跟踪并行阶段
  - 生成动作元数据

#### observation.py
- `ObservationBuilder`: 观测构建器
  - 构建观测向量
  - 管理动作历史
  - 计算观测维度

#### reward.py
- `RewardCalculator`: 奖励计算器
  - 基于晶圆进度计算奖励
  - 时间归一化

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/td_petri/ -v

# 运行性能测试
pytest tests/td_petri/test_performance.py -v -s

# 运行集成测试
pytest tests/td_petri/test_integration.py -v
```

### 测试覆盖

- **单元测试**: config, interval_utils, path_registry, action_space
- **集成测试**: TimedPetri 完整功能
- **性能测试**: 初始化、reset、step、吞吐量

## 性能基准

重构后的性能指标：

| 操作 | 预期时间 |
|------|---------|
| 初始化 | < 5秒 |
| Reset | < 1秒 |
| Step | < 0.5秒 |
| 观测构建 | < 1ms |

## 迁移指南

### 从旧版本迁移

大多数代码无需修改，因为重构保持了向后兼容：

```python
# ✅ 旧代码仍然有效
net = TimedPetri()
obs, mask = net.reset()
mask, obs, time, done, reward = net.step(action)
```

### 使用新功能

```python
# ✨ 使用新的配置系统
from solutions.Td_petri.core.config import PetriConfig

config = PetriConfig.default()
config.history_length = 100  # 修改配置
net = TimedPetri(config)

# ✨ 访问新模块
path_registry = net.path_registry
all_paths = path_registry.get_all_paths()

# ✨ 使用资源管理器
utilization = net.resource_mgr.calculate_utilization(net.time)
```

## 常见问题

### Q: 重构后性能有变化吗？
A: 性能基本保持一致，部分操作因模块化略有优化。

### Q: 旧代码需要修改吗？
A: 不需要。重构保持了完全的向后兼容。

### Q: 如何自定义配置？
A: 使用 `PetriConfig` 类创建或加载配置，然后传递给 `TimedPetri(config)`。

### Q: 测试如何运行？
A: 使用 `pytest tests/td_petri/ -v` 运行所有测试。

## 贡献

如需修改路径定义、配置参数或添加新功能，请参考新的模块结构：

1. **配置修改** → `core/config.py`
2. **路径修改** → `rl/path_registry.py`
3. **资源管理** → `resources/resource_manager.py`
4. **观测/奖励** → `rl/observation.py`, `rl/reward.py`

---

**注意**: 原始文档 `td_petri.md` 的详细技术说明仍然适用，本文档仅补充重构相关信息。
