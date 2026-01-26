# API Contract: 性能优化配置

**Date**: 2026-01-25  
**Feature**: 002-sim-speedup

## Overview

本文档描述了性能优化相关的 API 变更和配置接口。由于本次优化主要针对内部实现，外部 API 保持不变，仅扩展了配置选项。

## Configuration API

### PetriEnvConfig

配置类扩展，添加性能优化选项。

**Location**: `data/petri_configs/env_config.py`

**New Fields**:

```python
class PetriEnvConfig:
    # ... 现有字段 ...
    
    # 新增性能优化字段
    turbo_mode: bool = False
    """极速模式开关。启用时禁用详细统计追踪以提升性能"""
    
    optimize_reward_calc: bool = True
    """是否优化奖励计算（使用 NumPy 向量化）"""
    
    optimize_enable_check: bool = True
    """是否优化使能条件检查"""
    
    optimize_state_update: bool = True
    """是否优化状态更新逻辑"""
    
    cache_indices: bool = True
    """是否缓存库所和变迁索引"""
```

**Backward Compatibility**: 
- 所有新字段都有默认值，现有代码无需修改
- 默认行为与优化前一致（除了性能提升）

## Public API (Unchanged)

### Petri Class

**Location**: `solutions/Continuous_model/pn.py`

**Methods**: 所有公共方法签名保持不变

```python
class Petri:
    def __init__(
        self, 
        config: Optional[PetriEnvConfig] = None,
        stop_on_scrap: Optional[bool] = None,
        training_phase: Optional[int] = None,
        reward_config: Optional[Dict[str, int]] = None
    ) -> None:
        """初始化方法签名不变"""
        pass
    
    def reset(self) -> None:
        """重置方法签名不变"""
        pass
    
    def step(
        self, 
        t: Optional[int] = None, 
        wait: bool = False, 
        with_reward: bool = False, 
        detailed_reward: bool = False
    ) -> Tuple[bool, Union[float, Dict[str, float]], bool]:
        """步骤方法签名不变，但内部实现优化"""
        pass
    
    def get_enable_t(self) -> List[int]:
        """获取可使能变迁方法签名不变，但内部实现优化"""
        pass
```

**Behavior Guarantees**:
- 所有方法的输入/输出格式保持不变
- 返回值类型和结构保持不变
- 功能行为保持一致（核心功能）

## Internal API Changes

### 内部方法优化

以下方法内部实现优化，但不改变调用方式：

1. **`calc_reward()`**: 使用 NumPy 向量化优化
2. **`_resource_enable()`**: 优化使能条件检查
3. **`_fire()`**: 优化状态更新逻辑
4. **`_update_stay_times()`**: 批量更新优化
5. **`_track_wafer_statistics()`**: 在极速模式下跳过

### 新增内部方法

```python
class Petri:
    def _build_cached_indices(self) -> None:
        """构建缓存的索引映射（如果 cache_indices=True）"""
        pass
    
    def _calc_reward_vectorized(
        self, 
        t1: int, 
        t2: int, 
        moving_pre_places: Optional[np.ndarray] = None,
        detailed: bool = False
    ) -> Union[float, Dict[str, float]]:
        """向量化版本的奖励计算（如果 optimize_reward_calc=True）"""
        pass
```

## Performance Characteristics

### 性能保证

1. **极速模式性能**: 
   - 在标准开发机器上，1 秒内执行超过 100,000 个模拟步数
   - 配置：`turbo_mode=True, with_reward=True, detailed_reward=False`

2. **常规优化性能**:
   - 执行 1000 个模拟步数的时间减少至少 20%
   - 完成一个 episode 的时间减少至少 15%

3. **内存使用**:
   - 内存使用增加不超过 10%

### 功能一致性保证

1. **核心功能**: 状态转换、奖励计算、报废检测完全一致
2. **可选功能**: 详细统计追踪在极速模式下可能缺失
3. **API 兼容性**: 所有公共 API 保持不变

## Usage Examples

### 启用极速模式

```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

# 创建极速模式配置
config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,
    turbo_mode=True  # 启用极速模式
)

# 创建环境
env = Petri(config=config)
env.reset()

# 使用方式与之前完全相同
enabled = env.get_enable_t()
if enabled:
    done, reward, scrap = env.step(
        t=enabled[0], 
        with_reward=True, 
        detailed_reward=False  # 极速模式建议不使用详细奖励
    )
```

### 禁用特定优化

```python
# 禁用奖励计算优化（用于调试）
config = PetriEnvConfig(
    n_wafer=4,
    optimize_reward_calc=False  # 禁用奖励计算优化
)

env = Petri(config=config)
```

### 完全禁用优化（向后兼容）

```python
# 禁用所有优化，行为与优化前完全一致
config = PetriEnvConfig(
    n_wafer=4,
    turbo_mode=False,
    optimize_reward_calc=False,
    optimize_enable_check=False,
    optimize_state_update=False,
    cache_indices=False
)

env = Petri(config=config)
```

## Migration Guide

### 对于现有代码

**无需修改**: 现有代码无需任何修改即可享受性能优化（默认启用）

### 对于新代码

**推荐配置**: 在强化学习训练中使用极速模式

```python
config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,
    turbo_mode=True  # 推荐启用
)
```

### 对于调试代码

**推荐配置**: 禁用优化以获取详细追踪信息

```python
config = PetriEnvConfig(
    n_wafer=4,
    turbo_mode=False,  # 保留详细追踪
    optimize_reward_calc=False  # 使用原始奖励计算逻辑
)
```

## Testing Contracts

### 性能测试

```python
def test_performance_turbo_mode():
    """测试极速模式性能"""
    config = PetriEnvConfig(n_wafer=4, turbo_mode=True)
    env = Petri(config=config)
    env.reset()
    
    import time
    start = time.time()
    for _ in range(100000):
        enabled = env.get_enable_t()
        if enabled:
            env.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"性能测试失败: {elapsed}秒 > 1秒"
```

### 功能一致性测试

```python
def test_functionality_consistency():
    """测试功能一致性"""
    import numpy as np
    np.random.seed(42)
    
    # 优化前版本（禁用所有优化）
    config_old = PetriEnvConfig(
        n_wafer=4,
        turbo_mode=False,
        optimize_reward_calc=False,
        optimize_enable_check=False,
        optimize_state_update=False
    )
    env_old = Petri(config=config_old)
    env_old.reset()
    
    # 优化后版本（启用所有优化）
    config_new = PetriEnvConfig(
        n_wafer=4,
        turbo_mode=True,
        optimize_reward_calc=True,
        optimize_enable_check=True,
        optimize_state_update=True
    )
    env_new = Petri(config=config_new)
    env_new.reset()
    
    # 执行相同动作序列
    actions = [0, 1, 2, 3, 4, 5]  # 示例动作序列
    for action in actions:
        enabled_old = env_old.get_enable_t()
        enabled_new = env_new.get_enable_t()
        assert enabled_old == enabled_new, "使能变迁不一致"
        
        if enabled_old:
            _, reward_old, _ = env_old.step(t=enabled_old[0], with_reward=True)
            _, reward_new, _ = env_new.step(t=enabled_new[0], with_reward=True)
            assert abs(reward_old - reward_new) < 1e-6, "奖励不一致"
```
