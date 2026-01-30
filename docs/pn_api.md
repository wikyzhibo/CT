## pn.py API 文档

本文档描述 `solutions/Continuous_model/pn.py` 的主要类与方法。

### Place

**说明**：Petri 网库所，支持 token 管理与释放时间追踪。

**属性**
- `name: str` 库所名
- `capacity: int` 容量
- `processing_time: int` 加工时间
- `type: int` 库所类型（1=加工腔室，2=运输库所，3=空闲库所，4=资源库所，5=无驻留约束腔室）
- `tokens: Deque[BasedToken]` token 队列
- `release_schedule: Deque[(token_id, release_time)]` 释放时间队列
- `last_machine: int` 加工腔室轮换分配记录

**方法**
- `clone() -> Place` 深拷贝库所
- `head() -> BasedToken` 读取队首 token
- `pop_head() -> BasedToken` 弹出队首 token
- `append(token) -> None` 追加 token
- `res_time(current_time, P_Residual_time=15, D_Residual_time=10) -> int` 计算剩余驻留时间
- `add_release(token_id, release_time)` 添加释放时间
- `update_release(token_id, new_release_time)` 更新释放时间
- `pop_release(token_id) -> Optional[int]` 移除释放时间记录
- `earliest_release() -> Optional[int]` 返回最早释放时间

### Petri

**说明**：Petri 网调度环境。

**构造**
```python
Petri(
    config: Optional[PetriEnvConfig] = None,
    stop_on_scrap: Optional[bool] = None,
    training_phase: Optional[int] = None,
    reward_config: Optional[Dict[str, int]] = None
)
```

**常用方法**
- `reset()` 重置环境
- `get_enable_t() -> List[int]` 获取当前可使能变迁
- `next_enable_time() -> int` 估计下一可使能时间
- `step(t=None, wait=False, with_reward=False, detailed_reward=False)` 执行一步
- `render_gantt(out_path)` 输出甘特图

**奖励计算**
- `calc_reward(t1, t2, moving_pre_places=None, detailed=False)`
- `_calc_reward_original(...)` 原始奖励计算
- `_calc_reward_vectorized(...)` 向量化奖励计算

**释放时间追踪**
- `_build_release_chain()` 构建链式映射
- `_record_initial_release(...)` 初始释放时间记录
- `_update_release(...)` 实际进入后的时间更新
- `_check_release_violation(...)` 释放时间违规检测

**统计**
- `_track_wafer_statistics(...)` 追踪统计
- `calc_wafer_statistics() -> Dict[str, Any]` 统计汇总

### BasedToken

来源：`solutions/Continuous_model/construct.py`

```python
@dataclass
class BasedToken:
    enter_time: int
    stay_time: int = 0
    token_id: int = -1
    machine: int = -1
    color: int = 0  # 1=路线1, 2=路线2
```

### PetriEnvConfig

来源：`data/petri_configs/env_config.py`  
用于配置奖励系数、驻留时间、最大晶圆数、优化开关等。

常用配置项：
- `n_wafer`
- `stop_on_scrap`
- `reward_config`
- `max_wafers_in_system`
- `optimize_reward_calc`
- `optimize_state_update`
- `cache_indices`
- `optimize_data_structures`
