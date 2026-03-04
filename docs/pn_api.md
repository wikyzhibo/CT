## pn.py API 文档

本文档描述 `solutions/Continuous_model/pn.py` 的主要类与方法。

### Place

**说明**：Petri 网库所，支持 token 管理。

**属性**
- `name: str` 库所名
- `capacity: int` 容量
- `processing_time: int` 加工时间
- `type: int` 库所类型（1=加工腔室，2=运输库所，3=空闲库所，4=资源库所，5=无驻留约束腔室）
- `tokens: Deque[BasedToken]` token 队列
- `last_machine: int` 加工腔室轮换分配记录

**方法**
- `clone() -> Place` 深拷贝库所
- `head() -> BasedToken` 读取队首 token
- `pop_head() -> BasedToken` 弹出队首 token
- `append(token) -> None` 追加 token
- `res_time(current_time, P_Residual_time=15, D_Residual_time=10) -> int` 计算剩余驻留时间

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
- `get_enable_t() -> Tuple[List[int], List[int]]` 获取当前可使能变迁（按 TM2/TM3 分组）
- `next_enable_time() -> int` 估计下一可使能时间
- `step(a1=None, a2=None, wait1=False, wait2=False, with_reward=False, detailed_reward=False, t=None, wait=None)` 执行一步（并发优先，t/wait 兼容）
- `render_gantt(out_path)` 输出甘特图

**奖励计算**
- `calc_reward(t1, t2, moving_pre_places=None, detailed=False)`
- `_calc_reward_original(...)` 原始奖励计算
- `_calc_reward_vectorized(...)` 向量化奖励计算

**事后追责**
- `_chamber_timeline` / `_chamber_active`：记录腔室实际进入/离开时间
- `blame_release_violations()`：按 `u_*` 动作链式前瞻并回填惩罚

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

---

### PetriSingleDevice

来源：`solutions/Continuous_model/pn_single.py`

**说明**：单设备 Petri 网实现（独立文件，不改原 `pn.py`），仅 1 个机械手，8 个腔体命名如下：
- `LP`, `LP_done`, `PM1`~`PM6`
- 其中 `PM2/PM6` 为展示腔体，不参与工艺流转

**构网来源**
- `solutions/Continuous_model/construct_single.py`
- 在初始化阶段生成：`pre/pst/net/m0/md/id2p_name/id2t_name/marks`

**工艺路线**
- `LP -> PM1(100s) -> PM3(300s) -> PM5(100s) -> LP_done`
- `LP -> PM1(100s) -> PM4(300s) -> PM5(100s) -> LP_done`

**主要接口**
- `reset()`：重置网状态
- `_get_enable_t() -> List[int]`：内部使能判定（单机械手）
- `get_enable_t() -> List[int]`：外部使能接口
- `step(t=None, wait=None, with_reward=False, detailed_reward=False, ...)`：执行单步（动作校验 -> 发射/等待 -> 时间推进 -> 奖励 -> done）
- `calc_reward(t1, t2, detailed=False)`：奖励计算（`detailed_reward=True` 时返回含 `total` 的字典）
- `calc_wafer_statistics()`：返回统计字典（供可视化左栏读取）
