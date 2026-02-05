# Petri网环境配置说明

## 目录结构

```
data/petri_configs/
├── __init__.py                 # 模块初始化文件
├── env_config.py               # 配置类定义
├── default_config.json         # 默认配置
├── phase1_config.json          # 阶段1配置（仅报废惩罚）
├── phase2_config.json          # 阶段2配置（完整奖励）
├── usage_example.py            # 使用示例
└── README.md                   # 本说明文件
```

## 配置参数说明

### 环境基础参数
- `n_wafer`: 晶圆数量（默认: 4）

### 奖励参数
- `c_time`: 每秒时间成本，防止躺平（默认: 0.5）
- `R_done`: 每片完工奖励（默认: 100）
- `R_scrap`: 报废惩罚（默认: 100）

### 预警与安全裕量参数
- `T_warn`: 预警阈值（秒）（默认: 10）
- `a_warn`: 预警惩罚系数（默认: 0）
- `T_safe`: 安全裕量阈值（秒）（默认: 15）
- `b_safe`: 安全裕量奖励系数（默认: 0.5）

### WAIT 动作参数
- `MAX_WAIT_STEP`: WAIT 最大跳跃时间（默认: 20）

### 堵塞预防参数
- `c_congest`: 下游堵塞预测惩罚系数（默认: 50）

### 超时系数
- `D_Residual_time`: 运输位剩余时间（默认: 10）
- `P_Residual_time`: 加工腔室剩余时间（默认: 15）

### 释放时间违规惩罚参数
- `c_release_violation`: 违反最早释放时间的惩罚系数（默认: 30）
- `T_transport`: 运输时间（默认: 5）
- `T_load`: 装载时间（默认: 5）
- `T_pm1_to_pm2`: PM1 到 PM2 的必要运输时间（默认: 15）

### 停滞惩罚参数
- `idle_timeout`: 停滞超时阈值（秒）（默认: 100）
- `idle_penalty`: 停滞惩罚值（默认: 1000）

### 训练控制参数
- `stop_on_scrap`: 报废时是否停止（默认: true）
- `training_phase`: 训练阶段（1=仅报废惩罚，2=完整奖励）（默认: 2）

### 奖励开关配置
- `proc_reward`: 加工奖励（1=启用，0=禁用）
- `safe_reward`: 安全裕量奖励
- `penalty`: 加工腔室超时惩罚
- `warn_penalty`: 预警惩罚
- `transport_penalty`: 运输位超时惩罚（phase 1时为0）
- `congestion_penalty`: 堵塞预测惩罚
- `time_cost`: 时间成本
- `release_violation_penalty`: 释放时间违规惩罚

## 使用方法

### 方法1: 使用默认配置

```python
from solutions.Continuous_model.pn import Petri

# 使用默认配置
net = Petri()
```

### 方法2: 使用配置对象

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 创建配置
config = PetriEnvConfig(
    n_wafer=6,
    R_done=200,
    training_phase=1
)

# 使用配置创建环境
net = Petri(config=config)
```

### 方法3: 从配置文件加载

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 加载配置文件
config = PetriEnvConfig.load("data/petri_configs/phase1_config.json")

# 使用配置创建环境
net = Petri(config=config)
```

### 方法4: 向后兼容（旧代码仍可运行）

```python
from solutions.Continuous_model.pn import Petri

# 旧方式仍然有效
net = Petri(stop_on_scrap=True, training_phase=2)
```

## 保存和加载配置

### 保存配置

```python
from data.petri_configs.env_config import PetriEnvConfig

config = PetriEnvConfig(
    n_wafer=6,
    R_done=200,
    c_time=1.0
)

config.save("data/petri_configs/my_config.json")
```

### 加载配置

```python
from data.petri_configs.env_config import PetriEnvConfig

config = PetriEnvConfig.load("data/petri_configs/my_config.json")
print(config)
```

## 两阶段课程学习

### 阶段1：仅报废惩罚

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 加载阶段1配置
config = PetriEnvConfig.load("data/petri_configs/phase1_config.json")

# 创建环境
net = Petri(config=config)

# 验证配置
print(f"Training Phase: {net.training_phase}")
print(f"Transport Penalty: {net.reward_config['transport_penalty']}")
```

### 阶段2：完整奖励

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 加载阶段2配置
config = PetriEnvConfig.load("data/petri_configs/phase2_config.json")

# 创建环境
net = Petri(config=config)

# 验证配置
print(f"Training Phase: {net.training_phase}")
print(f"Transport Penalty: {net.reward_config['transport_penalty']}")
```

## 自定义奖励配置

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 创建自定义奖励配置
custom_reward_config = {
    'proc_reward': 1,
    'safe_reward': 0,          # 禁用安全裕量奖励
    'penalty': 1,
    'warn_penalty': 0,         # 禁用预警惩罚
    'transport_penalty': 1,
    'congestion_penalty': 1,   # 启用堵塞预测惩罚
    'time_cost': 1,
    'release_violation_penalty': 1,
}

config = PetriEnvConfig(
    training_phase=2,
    reward_config=custom_reward_config
)

net = Petri(config=config)
```

## 配置文件示例

### phase1_config.json
```json
{
  "n_wafer": 4,
  "training_phase": 1,
  "reward_config": {
    "proc_reward": 1,
    "safe_reward": 1,
    "penalty": 1,
    "warn_penalty": 1,
    "transport_penalty": 0,
    "congestion_penalty": 0,
    "time_cost": 1,
    "release_violation_penalty": 1
  }
}
```

### phase2_config.json
```json
{
  "n_wafer": 4,
  "training_phase": 2,
  "reward_config": {
    "proc_reward": 1,
    "safe_reward": 1,
    "penalty": 1,
    "warn_penalty": 1,
    "transport_penalty": 1,
    "congestion_penalty": 0,
    "time_cost": 1,
    "release_violation_penalty": 1
  }
}
```

## 与PPO训练集成

```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.PPO.enviroment import Env_PN

# 创建环境配置
petri_config = PetriEnvConfig.load("data/petri_configs/phase1_config.json")

# 创建训练环境（需要修改Env_PN以接受petri_config）
env = Env_PN(device="cpu", petri_config=petri_config)
```

## 最佳实践

1. **实验管理**: 为每个实验创建独立的配置文件
2. **版本控制**: 将配置文件纳入版本控制
3. **参数调优**: 复制预设配置文件，修改后保存为新文件
4. **文档记录**: 在配置文件中使用注释说明修改原因
5. **一致性**: 确保PPO训练配置和Petri环境配置的训练阶段一致
