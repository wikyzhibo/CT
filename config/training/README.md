# PPO训练配置文件说明

## Abstract

- **What**：`PPOTrainingConfig` 为 Pydantic 模型；磁盘格式以 **YAML** 为主（`.yaml`/`.yml`），`load` 仍可读 **JSON**（`.json`）。
- **When**：调整 PPO 超参、配置多路线训练档位时编辑 `low.yaml` / `medium.yaml` / `high.yaml` / `s_train.yaml`，或通过 `PPOTrainingConfig.load(path)` 加载。
- **Not**：本类不包含行为克隆等未在 `training_config.py` 中声明的字段；多余键会被忽略。
- **Key rules**：根目录 `requirements.txt` 需包含 `pydantic>=2`、`PyYAML>=6`。

## 目录结构

```
config/training/
├── __init__.py
├── training_config.py          # PPOTrainingConfig（Pydantic）
├── low.yaml                    # 轻量路线训练档位
├── medium.yaml                 # 中等路线训练档位
├── high.yaml                   # 高难路线训练档位
├── s_train.yaml                # A 方案默认单路线训练档位
└── README.md
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

### 其他参数
- `device`: 计算设备（"cpu" 或 "cuda"）
- `seed`: 随机种子（默认: 42）

## 使用方法

### 方法1: 使用配置对象

```python
from config.training.training_config import PPOTrainingConfig

# 创建默认配置
config = PPOTrainingConfig()

# 或创建自定义配置
config = PPOTrainingConfig(
    n_hidden=256,
    total_batch=200,
    lr=5e-4
)

# 训练
log, policy = train(env, eval_env, config=config)
```

### 方法2: 从配置文件加载

```python
from config.training.training_config import PPOTrainingConfig

# 加载配置文件
config = PPOTrainingConfig.load("config/training/low.yaml")

# 训练
log, policy = train(env, eval_env, config=config)
```

### 方法3: 直接指定配置文件路径

```python
# 训练函数会自动加载配置
log, policy = train(
    env, 
    eval_env, 
    config_path="config/training/high.yaml"
)
```

### 保存配置

```python
config = PPOTrainingConfig(n_hidden=256, lr=5e-4)
config.save("data/ppo_configs/my_config.yaml")  # 或 .json，由后缀决定格式
```

## 配置文件管理

`solutions.A.eval.validate_all_routes` 会按 `ROUTE_PLAN` 中的路线配置选择 `low.yaml` / `medium.yaml` / `high.yaml`，并把训练汇总写入 `results/training_logs/validate_all_routes_summary.json`。`solutions/C/train.py` 在每次训练开始时仍会把当前配置写入 `results/training_logs/`，文件名形如 `config_ppo_{时间戳}.yaml`（后缀由 `save` 路径决定）。

## Related Docs

- `docs/training/training-guide.md`
