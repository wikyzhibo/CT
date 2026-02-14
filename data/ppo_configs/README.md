# PPO训练配置文件说明

## 目录结构

```
data/ppo_configs/
├── __init__.py                 # 模块初始化文件
├── training_config.py          # 配置类定义
├── default_config.json         # 默认配置
├── phase1_config.json          # 阶段1配置（仅报废惩罚）
├── phase2_config.json          # 阶段2配置（完整奖励）
├── training_runs/              # 保存每次训练运行的配置
└── README.md                   # 本说明文件
```

## 配置参数说明

### 网络结构参数
- `n_hidden`: 隐藏层神经元数量（默认: 128）
- `n_layer`: 网络层数（默认: 4）

### 训练批次参数
- `total_batch`: 总批次数（默认: 150）
- `sub_batch_size`: 子批次大小（默认: 64）
- `num_epochs`: 每个批次的训练轮数（默认: 10）

### PPO算法参数
- `gamma`: 折扣因子（默认: 0.99）
- `gae_lambda`: GAE的λ参数（默认: 0.95）
- `clip_epsilon`: PPO裁剪参数（默认: 0.2）
- `lr`: 学习率（默认: 1e-4）

### 熵系数参数
- `entropy_start`: 初始熵系数（默认: 0.02）
- `entropy_end`: 最终熵系数（默认: 0.01）

### 行为克隆参数
- `lambda_bc0`: BC权重初始值（默认: 1.0）
- `bc_decay_batches`: BC权重衰减的批次数（默认: 200）
- `bc_weight_early`: 早期BC权重（默认: 2.0）
- `bc_weight_late`: 后期BC权重（默认: 0.1）
- `bc_switch_batch`: BC权重切换的批次数（默认: 100）

### 其他参数
- `device`: 计算设备（"cpu" 或 "cuda"）
- `seed`: 随机种子（默认: 42）
- `training_phase`: 训练阶段（1或2）
- `with_pretrain`: 是否使用预训练（默认: false）

## 使用方法

### 方法1: 使用配置对象

```python
from data.ppo_configs.training_config import PPOTrainingConfig

# 创建默认配置
config = PPOTrainingConfig()

# 或创建自定义配置
config = PPOTrainingConfig(
    n_hidden=256,
    total_batch=200,
    lr=5e-4,
    training_phase=1
)

# 训练
log, policy = train(env, eval_env, config=config)
```

### 方法2: 从配置文件加载

```python
from data.ppo_configs.training_config import PPOTrainingConfig

# 加载配置文件
config = PPOTrainingConfig.load("data/ppo_configs/phase1_config.json")

# 训练
log, policy = train(env, eval_env, config=config)
```

### 方法3: 直接指定配置文件路径

```python
# 训练函数会自动加载配置
log, policy = train(
    env, 
    eval_env, 
    config_path="data/ppo_configs/phase2_config.json"
)
```

### 保存配置

```python
config = PPOTrainingConfig(n_hidden=256, lr=5e-4)
config.save("data/ppo_configs/my_config.json")
```

## 配置文件管理

每次训练运行时，使用的配置会自动保存到 `training_runs/` 目录下，文件名格式为：
```
config_phase{训练阶段}_{时间戳}.json
```

这样可以追溯每次训练使用的具体配置参数。
