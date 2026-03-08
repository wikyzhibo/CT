# 配置系统总览指南

本项目采用统一的配置管理系统，将所有训练和环境参数集中管理，便于实验追踪和复现。

## 📁 配置系统结构

```
data/
├── ppo_configs/              # PPO训练配置
│   ├── __init__.py
│   ├── training_config.py    # PPO训练配置类
│   ├── default_config.json   # 默认PPO配置
│   ├── phase1_config.json    # 阶段1 PPO配置
│   ├── phase2_config.json    # 阶段2 PPO配置
│   ├── training_runs/        # 训练运行配置记录
│   ├── usage_example.py      # 使用示例
│   └── README.md             # 详细说明
│
├── petri_configs/            # Petri网环境配置
│   ├── __init__.py
│   ├── env_config.py         # 环境配置类
│   ├── default_config.json   # 默认环境配置
│   ├── phase1_config.json    # 阶段1环境配置
│   ├── phase2_config.json    # 阶段2环境配置
│   ├── usage_example.py      # 使用示例
│   └── README.md             # 详细说明
│
└── CONFIG_SYSTEM_GUIDE.md    # 本文档
```

## 🎯 两大配置系统

### 1. PPO训练配置 (`ppo_configs/`)

管理PPO强化学习训练的所有超参数：

**主要参数:**
- 网络结构: `n_hidden`, `n_layer`
- 训练批次: `total_batch`, `sub_batch_size`, `num_epochs`
- PPO算法: `gamma`, `gae_lambda`, `clip_epsilon`, `lr`
- 熵系数: `entropy_start`, `entropy_end`
- 行为克隆: `lambda_bc0`, `bc_decay_batches`

**使用示例:**
```python
from data.ppo_configs.training_config import PPOTrainingConfig
from solutions.PPO.train import train

# 加载配置
config = PPOTrainingConfig.load("data/ppo_configs/s_train.json")

# 训练
log, policy = train(env, eval_env, config=config)
```

### 2. Petri网环境配置 (`petri_configs/`)

管理Petri网仿真环境的所有参数：

**主要参数:**
- 环境参数: `n_wafer`
- 奖励参数: `time_coef`, `R_done`, `R_scrap`
- 预警参数: `T_warn`, `a_warn`, `T_safe`, `b_safe`
- 超时系数: `D_Residual_time`, `P_Residual_time`
- 奖励开关: `reward_config`

**使用示例:**
```python
from data.petri_configs.env_config import PetriEnvConfig
from solutions.Continuous_model.pn import Petri

# 加载配置
config = PetriEnvConfig.load("data/petri_configs/s_train.json")

# 创建环境
net = Petri(config=config)
```

## 🔄 两阶段课程学习

本项目采用两阶段课程学习策略：

### 阶段1: 基础训练（仅报废惩罚）

**目标**: 让模型学习避免基本错误（加工腔室超时）

**配置:**
- PPO: `data/ppo_configs/phase1_config.json`
- Petri: `data/petri_configs/phase1_config.json`
- 特点: `training_phase=1`, `transport_penalty=0`

**运行:**
```bash
python solutions/PPO/run_ppo.py --phase 1
```

### 阶段2: 高级训练（完整奖励）

**目标**: 在阶段1基础上，优化运输位使用

**配置:**
- PPO: `data/ppo_configs/phase2_config.json`
- Petri: `data/petri_configs/phase2_config.json`
- 特点: `training_phase=2`, `transport_penalty=1`

**运行:**
```bash
python solutions/PPO/run_ppo.py --phase 2
# 自动加载阶段1的checkpoint继续训练
```

### 自动两阶段训练

```bash
python solutions/PPO/run_ppo.py --auto-phase2
# 自动执行 Phase 1 -> Phase 2
```

## 📝 配置文件管理最佳实践

### 1. 实验版本控制

每次训练会自动保存配置快照到 `training_runs/`：
```
data/ppo_configs/training_runs/
├── config_phase1_20260122_143000.json
├── config_phase2_20260122_150000.json
└── ...
```

这样可以追溯每次实验的确切配置。

### 2. 创建实验配置

```python
# 方法1: 修改现有配置
from data.ppo_configs.training_config import PPOTrainingConfig

config = PPOTrainingConfig.load("data/ppo_configs/s_train.json")
config.lr = 5e-4
config.total_batch = 300
config.save("data/ppo_configs/experiment_01.json")

# 方法2: 创建新配置
config = PPOTrainingConfig(
    n_hidden=256,
    total_batch=200,
    lr=1e-3,
    training_phase=1
)
config.save("data/ppo_configs/experiment_02.json")
```

### 3. 超参数调优

```python
# 创建参数扫描配置
for lr in [1e-3, 5e-4, 1e-4]:
    for n_hidden in [64, 128, 256]:
        config = PPOTrainingConfig(
            lr=lr,
            n_hidden=n_hidden,
            training_phase=1
        )
        config.save(f"data/ppo_configs/sweep_lr{lr}_h{n_hidden}.json")
```

## 🔧 命令行工具集成

### PPO训练命令行参数

```bash
# 基础用法
python solutions/PPO/run_ppo.py --phase 1

# 使用自定义配置
python solutions/PPO/run_ppo.py --config data/ppo_configs/my_config.json

# GPU训练
python solutions/PPO/run_ppo.py --phase 1 --device cuda

# 使用预训练
python solutions/PPO/run_ppo.py --phase 1 --with-pretrain

# 从checkpoint继续
python solutions/PPO/run_ppo.py \
    --phase 2 \
    --checkpoint solutions/PPO/saved_models/CT_phase1_latest.pt

# 完整示例
python solutions/PPO/run_ppo.py \
    --config data/ppo_configs/experiment_01.json \
    --device cuda \
    --with-pretrain
```

## 📊 配置对比工具

### 比较两个配置文件

```python
from data.ppo_configs.training_config import PPOTrainingConfig

config1 = PPOTrainingConfig.load("data/ppo_configs/s_train.json")
config2 = PPOTrainingConfig.load("data/ppo_configs/phase2_config.json")

# 打印配置
print(config1)
print(config2)

# 比较差异
import json
dict1 = config1.to_dict() if hasattr(config1, 'to_dict') else config1.__dict__
dict2 = config2.to_dict() if hasattr(config2, 'to_dict') else config2.__dict__

for key in dict1:
    if dict1[key] != dict2[key]:
        print(f"{key}: {dict1[key]} -> {dict2[key]}")
```

## 🎨 自定义配置示例

### 示例1: 快速训练配置

```json
{
  "n_hidden": 64,
  "n_layer": 3,
  "total_batch": 50,
  "sub_batch_size": 32,
  "num_epochs": 5,
  "lr": 1e-3,
  "training_phase": 1
}
```

### 示例2: 高精度训练配置

```json
{
  "n_hidden": 256,
  "n_layer": 6,
  "total_batch": 300,
  "sub_batch_size": 128,
  "num_epochs": 20,
  "lr": 5e-5,
  "training_phase": 2
}
```

### 示例3: 自定义奖励权重

```json
{
  "n_wafer": 4,
  "time_coef": 1.0,
  "R_done": 200,
  "R_scrap": 150,
  "training_phase": 2,
  "reward_config": {
    "proc_reward": 1,
    "safe_reward": 0,
    "penalty": 1,
    "warn_penalty": 0,
    "transport_penalty": 1,
    "congestion_penalty": 1,
    "time_cost": 1,
    "release_violation_penalty": 1
  }
}
```

## 🔍 配置验证和调试

### 验证配置有效性

```python
from data.ppo_configs.training_config import PPOTrainingConfig
from data.petri_configs.env_config import PetriEnvConfig

# 加载并验证PPO配置
try:
    ppo_config = PPOTrainingConfig.load("data/ppo_configs/my_config.json")
    print("✓ PPO配置有效")
    print(ppo_config)
except Exception as e:
    print(f"✗ PPO配置错误: {e}")

# 加载并验证环境配置
try:
    env_config = PetriEnvConfig.load("data/petri_configs/my_config.json")
    print("✓ 环境配置有效")
    print(env_config)
except Exception as e:
    print(f"✗ 环境配置错误: {e}")
```

### 检查配置一致性

```python
# 确保训练阶段一致
ppo_config = PPOTrainingConfig.load("data/ppo_configs/s_train.json")
env_config = PetriEnvConfig.load("data/petri_configs/s_train.json")

assert ppo_config.training_phase == env_config.training_phase, \
    "PPO和环境的训练阶段必须一致！"

print(f"✓ 配置一致性检查通过 (Phase {ppo_config.training_phase})")
```

## 📚 参考文档

- **PPO配置详细说明**: `data/ppo_configs/README.md`
- **Petri环境配置说明**: `data/petri_configs/README.md`
- **PPO运行指南**: `solutions/PPO/RUN_GUIDE.md`
- **使用示例**:
  - `data/ppo_configs/usage_example.py`
  - `data/petri_configs/usage_example.py`

## 🚀 快速开始

### 新手入门

```bash
# 1. 使用默认配置训练
python solutions/PPO/run_ppo.py

# 2. 查看配置
cat data/ppo_configs/default_config.json
cat data/petri_configs/default_config.json

# 3. 运行示例
python data/ppo_configs/usage_example.py
python data/petri_configs/usage_example.py
```

### 高级用户

```bash
# 1. 创建自定义配置
python -c "
from data.ppo_configs.training_config import PPOTrainingConfig
config = PPOTrainingConfig(lr=5e-4, total_batch=200)
config.save('data/ppo_configs/my_exp.json')
"

# 2. 使用自定义配置训练
python solutions/PPO/run_ppo.py --config data/ppo_configs/my_exp.json

# 3. 自动两阶段训练
python solutions/PPO/run_ppo.py --auto-phase2 --device cuda
```

## ⚠️ 常见问题

### Q1: 配置文件找不到？
**A**: 确保使用相对于项目根目录的路径，例如 `data/ppo_configs/xxx.json`

### Q2: 修改配置后没有生效？
**A**: 检查是否正确传递了配置对象/路径给训练函数

### Q3: 如何恢复之前的训练？
**A**: 查看 `data/ppo_configs/training_runs/` 找到对应时间戳的配置文件

### Q4: 两个配置系统如何协同？
**A**: PPO配置管理训练过程，Petri配置管理环境参数，确保两者的 `training_phase` 一致

## 💡 小贴士

1. **配置命名**: 使用描述性的文件名，如 `high_lr_large_batch.json`
2. **版本管理**: 重要配置文件加入Git版本控制
3. **注释记录**: 在实验笔记中记录配置文件路径和实验结果
4. **备份配置**: 定期备份 `training_runs/` 目录
5. **参数搜索**: 使用脚本批量生成配置文件进行超参数搜索
