## pn.py 文档

本文件描述 `solutions/Continuous_model/pn.py` 的使用与整体设计。该环境模拟半导体制造调度，基于 Petri 网，支持双路线与双机械手协作。

### 系统概述

核心特性：
- 双路线加工：在 `s1` 处分流（color=1 走路线1，color=2 走路线2）
- 双机械手协作：`TM2` 与 `TM3` 分工
- 驻留约束：`s1/s3/s5` 有驻留时间限制，`s2/s4` 无驻留约束
- 释放时间追踪：预测下游释放时间并做违规惩罚
- 奖励塑形：支持密集奖励与安全裕量奖励
- 可视化：支持甘特图生成

### 拓扑结构

路线1：
`LP1 -> s1 -> s2 -> s3 -> s4 -> s5 -> LP_done`

路线2：
`LP2 -> s1 -> s5 -> LP_done`

### 腔室配置

| 库所 | 含义 | 容量 | 加工时间 | 驻留约束 |
| --- | --- | --- | --- | --- |
| s1 | PM7/PM8 | 2 | 70 | 有 |
| s2 | LLC | 1 | 0 | 无 |
| s3 | PM1/PM2/PM3/PM4 | 4 | 600 | 有 |
| s4 | LLD | 1 | 70 | 无 |
| s5 | PM9/PM10 | 2 | 200 | 有 |

### 机械手分工

- TM2：LP1/LP2/s1/s2/s4/s5/LP_done
- TM3：s2/s3/s4

### 快速开始

```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig
import numpy as np

config = PetriEnvConfig(n_wafer=10, stop_on_scrap=True, training_phase=1)
env = Petri(config=config)
env.reset()

while True:
    enabled = env.get_enable_t()
    if not enabled:
        break
    action = int(np.random.choice(enabled))
    done, reward, scrap = env.step(t=action, with_reward=True)
    if done:
        break

env.render_gantt("results/continuous_gantt.png")
```

### 关键概念

**驻留约束**  
有驻留约束的腔室必须在 `processing_time + P_Residual_time` 内离开，否则会触发报废或惩罚。

**释放时间追踪**  
当晶圆进入运输位时记录预计释放时间，并链式更新下游腔室的预计释放时间，提前检测容量冲突。

### 性能优化

可在配置中开启：
- `optimize_reward_calc`：向量化奖励计算
- `optimize_state_update`：批量更新 token 滞留时间
- `cache_indices`：缓存库所/变迁索引
- `optimize_data_structures`：按类型缓存库所列表

示例：
```python
config = PetriEnvConfig(
    optimize_reward_calc=True,
    optimize_state_update=True,
    cache_indices=True,
    optimize_data_structures=True,
)
```

### 统计与可视化

```python
stats = env.calc_wafer_statistics()
print(stats["system_avg"], stats["system_max"])
env.render_gantt("results/continuous_gantt.png")
```
