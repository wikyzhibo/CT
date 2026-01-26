# PPO 训练运行指南

## 快速开始

### 1. 使用默认配置训练（阶段2）

```bash
python solutions/PPO/run_ppo.py
```

### 2. 使用预设配置文件训练

```bash
# 阶段1训练
python solutions/PPO/run_ppo.py --phase 1

# 阶段2训练
python solutions/PPO/run_ppo.py --phase 2
```

### 3. 使用自定义配置文件

```bash
python solutions/PPO/run_ppo.py --config data/ppo_configs/my_config.json
```

### 4. 自动两阶段课程学习训练

```bash
python solutions/PPO/run_ppo.py --auto-phase2
```

## 命令行参数详解

### 基础参数

- `--phase {1,2}`: 训练阶段
  - `1`: 仅考虑报废惩罚（加工腔室超时）
  - `2`: 完整奖励（加工腔室超时 + 运输位超时）
  - 默认值: `2`

- `--device {cpu,cuda}`: 计算设备
  - `cpu`: 使用CPU训练
  - `cuda`: 使用GPU训练（需要CUDA支持）
  - 默认值: `cpu`

### 配置相关

- `--config PATH`: 指定配置文件路径
  - 示例: `--config data/ppo_configs/phase1_config.json`
  - 如果不指定，会自动尝试加载 `data/ppo_configs/phase{N}_config.json`
  - 如果预设配置也不存在，使用默认配置

### 训练控制

- `--checkpoint PATH`: 从checkpoint继续训练
  - 示例: `--checkpoint solutions/PPO/saved_models/CT_phase1_latest.pt`
  - 阶段2会自动检测并加载阶段1的checkpoint

- `--with-pretrain`: 启用行为克隆预训练
  - 默认: 不启用

- `--auto-phase2`: 自动执行两阶段训练
  - 先训练阶段1，保存模型
  - 再加载阶段1模型训练阶段2

## 使用示例

### 示例1: 快速测试（使用默认配置）

```bash
python solutions/PPO/run_ppo.py --phase 1
```

### 示例2: 使用GPU训练阶段1

```bash
python solutions/PPO/run_ppo.py --phase 1 --device cuda
```

### 示例3: 使用自定义配置和预训练

```bash
python solutions/PPO/run_ppo.py \
    --config data/ppo_configs/my_config.json \
    --with-pretrain \
    --device cuda
```

### 示例4: 阶段2训练（从阶段1checkpoint开始）

```bash
# 方法1: 自动检测checkpoint
python solutions/PPO/run_ppo.py --phase 2

# 方法2: 手动指定checkpoint
python solutions/PPO/run_ppo.py \
    --phase 2 \
    --checkpoint solutions/PPO/saved_models/CT_phase1_latest.pt
```

### 示例5: 自动两阶段课程学习

```bash
# 使用默认配置
python solutions/PPO/run_ppo.py --auto-phase2

# 使用自定义配置
python solutions/PPO/run_ppo.py \
    --auto-phase2 \
    --config data/ppo_configs/my_config.json \
    --device cuda
```

### 示例6: 继续之前中断的训练

```bash
python solutions/PPO/run_ppo.py \
    --phase 2 \
    --checkpoint solutions/PPO/saved_models/CT_phase2_20260122_143000.pt \
    --config data/ppo_configs/training_runs/config_phase2_20260122_143000.json
```

## 配置优先级

当同时指定多个参数时，优先级为：

1. **命令行参数** (最高优先级)
2. **配置文件参数**
3. **预设配置文件** (`data/ppo_configs/phase{N}_config.json`)
4. **默认配置** (最低优先级)

例如：
```bash
python solutions/PPO/run_ppo.py \
    --config data/ppo_configs/phase1_config.json \
    --phase 2 \
    --device cuda
```

这会：
- 从 `phase1_config.json` 加载基础配置
- 将 `training_phase` 覆盖为 `2`（命令行指定）
- 将 `device` 覆盖为 `cuda`（命令行指定）

## 训练输出

### 模型保存位置

- 时间戳模型: `solutions/PPO/saved_models/CT_phase{N}_{timestamp}.pt`
- 最新模型: `solutions/PPO/saved_models/CT_phase{N}_latest.pt`

### 配置保存位置

每次训练使用的配置会自动保存到:
- `data/ppo_configs/training_runs/config_phase{N}_{timestamp}.json`

### 训练日志

训练过程中会打印：
- 当前批次、帧数
- 累计奖励
- 完成周期数
- 平均makespan

示例输出：
```
batch 0001 | frames=640 | sum_reward=-1234.56| circle=5|makespan=8765.43
batch 0002 | frames=1280 | sum_reward=-1156.78| circle=6|makespan=8543.21
...
```

## 配置文件管理

### 创建自定义配置

```python
from data.ppo_configs.training_config import PPOTrainingConfig

# 创建配置
config = PPOTrainingConfig(
    n_hidden=256,
    n_layer=5,
    total_batch=300,
    lr=5e-4,
    training_phase=1
)

# 保存配置
config.save("data/ppo_configs/my_custom_config.json")
```

### 修改现有配置

```python
# 加载配置
config = PPOTrainingConfig.load("data/ppo_configs/phase1_config.json")

# 修改参数
config.total_batch = 300
config.lr = 5e-4

# 保存为新配置
config.save("data/ppo_configs/phase1_extended.json")
```

## 故障排查

### 问题1: CUDA out of memory

```bash
# 解决方案：使用CPU或减小batch size
python solutions/PPO/run_ppo.py --device cpu
```

或修改配置文件中的 `sub_batch_size` 和 `num_epochs`

### 问题2: 找不到配置文件

```bash
# 确保路径正确
ls data/ppo_configs/

# 或使用绝对路径
python solutions/PPO/run_ppo.py --config "C:/Users/.../data/ppo_configs/my_config.json"
```

### 问题3: Checkpoint加载失败

```bash
# 检查checkpoint是否存在
ls solutions/PPO/saved_models/

# 或不使用checkpoint从头训练
python solutions/PPO/run_ppo.py --phase 1
```

## 最佳实践

1. **首次训练**: 使用 `--auto-phase2` 进行完整的两阶段训练
2. **超参数调优**: 复制预设配置文件，修改参数后使用 `--config` 指定
3. **继续训练**: 使用 `--checkpoint` 加载之前的模型
4. **实验管理**: 每次训练会自动保存配置到 `training_runs/`，便于追踪
5. **GPU训练**: 如有GPU，使用 `--device cuda` 加速训练
