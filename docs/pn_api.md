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
- 其中 `PM2` 为展示腔体，不参与工艺流转；`PM5` 为 UI 占位腔体（模型不参与流转）
- `PM6` 是否参与工艺由 `PetriEnvConfig.single_route_code` 决定

**构网来源**
- `solutions/Continuous_model/construct_single.py`
- 在初始化阶段生成：`pre/pst/net/m0/md/id2p_name/id2t_name/marks`
- `d_TM1` 在构网中设置 `processing_time=5s`（默认），用于约束“运输位停留后才能 `t_*` 进入腔室”
- 通过 `PetriEnvConfig.single_robot_capacity` 控制 `d_TM1` 容量：
  - `1`：Single Arm（单臂）
  - `2`：Dual Arm（双臂）
- 单设备清洗配置（当前训练简化版）：
  - `single_cleaning_enabled=true`
  - `single_cleaning_targets=["PM3","PM4"]`
  - `single_cleaning_trigger_wafers=2`
  - `single_cleaning_duration=150`

**工艺路线**
- `single_device_mode="cascade"`：`LP -> PM7/PM8 -> LLC -> PM1/PM2/PM3/PM4 -> LLD -> PM9/PM10 -> LP_done`
- `single_route_code=0`（默认）：`LP -> PM1(100s) -> PM3/PM4(300s) -> LP_done`
- `single_route_code=1`：`LP -> PM1(100s) -> PM3/PM4(300s) -> PM6(300s) -> LP_done`
- 双臂模式约束：`u_*` 仅在可解析到有效目标腔室时才允许发射；若下游层满导致目标不可解析，则该 `u_*` 必须禁用，避免出现 `LP -> d_TM1 -> LP_done` 的非法短路。
- 上述约束同样适用于 `d_TM1` 为空时：双臂“可任意取片”不等于可忽略目标可达性，目标不可解析时必须禁用 `u_*`。

**主要接口**
- `reset()`：重置网状态
- `_get_enable_t() -> List[int]`：内部使能判定（单机械手）
- `get_enable_t() -> List[int]`：外部使能接口
- `get_enable_actions(wait_action_start=None) -> List[int]`：返回完整离散动作使能（`transition + wait`）
- `get_action_mask(wait_action_start=None, n_actions=None) -> np.ndarray`：返回完整离散动作掩码（`transition + wait`）
- `step(t=None, wait=None, with_reward=False, detailed_reward=False, ...)`：执行单步（动作校验 -> 发射/等待 -> 时间推进 -> 奖励 -> done）
- `calc_reward(t1, t2, detailed=False)`：奖励计算（`detailed_reward=True` 时返回含 `total` 的字典）
- `blame_release_violations() -> Dict[int, float]`：基于 `_chamber_timeline` 的单设备事后追责，输出 `fire_log_index -> penalty`
  - `single_route_code=0`：追责站点 `s1=PM1`，`s2=PM3∪PM4`，`u_LP` 链路按 `s1 -> s2` 判定
  - `single_route_code=1`：在上述基础上增加 `s3=PM6`，`u_LP` 链路扩展为 `s1 -> s2 -> s3`
- 清洗事件日志会附加写入 `fire_log`（`event_type=cleaning_start|cleaning_end`），用于后续追责/复盘。
- `calc_wafer_statistics()`：返回统计字典（供可视化左栏读取）

**两阶段训练相关字段**
- `no_release_penalty: bool`：第一阶段采样时置 `True`，第二阶段回填惩罚时置 `False`
- `_chamber_timeline/_chamber_active`：按路径代号记录加工腔体进入离开时间线（`code=0` 为 PM1/PM3/PM4；`code=1` 额外包含 PM6）

**设备模式字段**
- `single_device_mode`: `single` 或 `cascade`。
  - `single`: 使用原单设备路径（可叠加 `single_route_code`）。
  - `cascade`: 使用级联路径模板，但接口仍保持单动作离散动作空间（`transition + wait`）。

**驻留时间更新规则（单设备）**
- `LP`（type=3）中的 token 不更新 `stay_time`，与 `pn.py` 保持一致

**双臂防死锁规则（`_get_enable_t`）**
- 当 `d_TM1` 为空时：允许任意满足前置条件的 `u_*` 取片。
- 当 `d_TM1` 非空且队首晶圆在“取片时”其 `dst` 层已满：仅允许继续取出该 `dst` 层中的晶圆，禁止取其它位置晶圆。
- `t_*` 始终遵循 FIFO 队首目标约束（队首 `_target_place`）与 `d_TM1` dwell 时间约束。
- 使能计算分为两阶段：
  - Stage1：`pre/pst` + 容量 + 防死锁规则（用于死锁判定，`u_*` 在该阶段忽略清洗目标过滤）
  - Stage2：加工完成、目标可达与 dwell/清洗过滤（`get_enable_t()` 最终返回）
- WAIT 掩码规则（单设备）：
  - 默认启用所有 wait 档位；
  - 若当前路径任一加工腔室存在 token 满足 `stay_time >= processing_time`（加工完成待取片），禁用 `WAIT>5s`，仅保留 `WAIT_5s`。
- `u_LP` 不再使用 Stage2 额外边界拦截；当前仅遵循通用使能规则（加工完成、目标可达、清洗过滤与运输位 dwell 约束）。
- 当 `t_*` 的目标腔室处于 `is_cleaning=True` 时，Stage2 才禁用该变迁（死锁判定仍基于未清洗过滤的 Stage1）。
- 死锁定义：`LP_done` 未完成且 Stage1 无任何使能变迁。

**观测补充（单设备）**
- `Env_PN_Single` 的观测由两部分组成（全部为 `float32`）：
  - `MAX_WAFERS` 个晶圆的特征：先收集所有 token，按 `token_id` 升序排序；每片晶圆使用“合法 `(place_idx, where)` 对全集 one-hot”（静态规则枚举，不是 `P×W` 笛卡尔积），并拼接
    - `present`：真实晶圆为 `1`，补零位为 `0`
    - `status_one_hot`（顺序固定）：`processing / done_waiting_pick / moving / waiting`
      - `waiting` 定义：晶圆位于运输位且 `stay_time > D_Residual_time`
    - `remaining_processing_norm`：剩余加工量归一化，`0` 表示可取
    - `time_to_scrap_norm`：先按阈值 `30` 裁剪再归一化，`0` 表示立刻报废；运输位晶圆固定为 `1`
  - 不足 `MAX_WAFERS` 时按单片特征长度补零；最后追加 3 维腔室计数：`PM1/PM3/PM4` 的 `processed_wafer_count`（用于让策略感知清洗触发临界状态）

**运输位 token 的机械臂标识**
- 每个 token 在进入 `d_TM1`（即 `u_*` 发射）时分配 `machine` 字段。
- 分配策略为轮换（round-robin）：
  - 单臂模式固定为 `1`
  - 双臂模式在 `1/2` 间交替

**停滞惩罚（单设备已接入）**
- 沿用 `pn.py` 思路：累计连续 WAIT 时间（`_consecutive_wait_time`）
- 当累计时间达到 `idle_timeout = max(processing_time)+30` 且未触发过时，施加一次 `idle_timeout_penalty`
- 运输位超时惩罚：`d_TM1`（type=2）内晶圆停留超过 `D_Residual_time` 后按超时秒数线性扣分，受 `reward_config["transport_penalty"]` 控制，系数为 `transport_overtime_coef`

**死锁终止语义（单设备）**
- 发生死锁时，episode 终止，增加 `deadlock_count`。
- 死锁惩罚量级与 `R_scrap` 等价（`deadlock_penalty = -abs(R_scrap)`）。
- 死锁不计入 `scrap`。
