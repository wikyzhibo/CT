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
- `get_obs() -> List[float]` 返回观测特征（基类默认 `[]`，子类覆写）

**Place 子类（单设备观测）**

| 子类 | 库所 | 特征维度 | 构造参数 |
|-----|------|---------|---------|
| SR  | LP, LP_done | 1 / 0 | n_wafer |
| TM  | d_TM1, d_TM2, d_TM3 | `4 + onehot_dim`（cascade 下 TM2/TM3 固定 `4+8`） | D_Residual_time, target_onehot_map, onehot_dim |
| PM  | PM1, PM3, PM4... | 9 | P_Residual_time, cleaning_* |
| LL  | LLC, LLD | 6（基础 4 维 + `in/out` 两维） | - |

构网时由 `construct_single.build_net(obs_config=...)` 创建。详见 `docs/架构.md`。

### Petri

**说明**：Petri 网调度环境。

**构造**
```python
Petri(
    config: Optional[PetriEnvConfig] = None,
    stop_on_scrap: Optional[bool] = None,
)
```

（奖励分项开关不再通过参数或 `PetriEnvConfig` 配置；运行时固定全开。）

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
- `blame_release_violations()`：按 `u_LP`/`u_LLC`/`u_LLD` 动作链式前瞻并回填惩罚（仅追责释放动作）

**统计**
- A 方案 `solutions/A/petri_net.py`（`ClusterTool`）已移除 `_token_stats` 与 `calc_wafer_statistics`；`visualization/petri_adapter.py`、`petri_single_adapter.py` 构造 `StateInfo.stats` 时，系统/腔室/TM 聚合时长为占位（0 或空 dict），`completed_count`/`in_progress_count` 与 `resident_violation_count`/`qtime_violation_count` 来自网对象字段。
- **A 方案 `ClusterTool` 甘特（2026-03-31）**：`render_gantt(out_path, title_suffix=None)` **Does**：方法存在，调用不报错。**Does NOT**：当前方法体为空，不生成 PNG；网内不再维护 `_chamber_timeline` / `_chamber_active`。构网仍可输出 `route_meta.timeline_chambers`，供后续重写复用。
- 若文档其他处仍指 `solutions/Continuous_model/pn.py` 的 `Petri`，是否保留 `_track_wafer_statistics` / `calc_wafer_statistics` 以该文件为准。

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
- `n_wafer1`
- `n_wafer2`
- `stop_on_scrap`
- `max_wafers1_in_system`
- `max_wafers2_in_system`
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
- `PM6` 是否参与工艺由当前 `single_route_config` 所选路线的 `routes[<name>].sequence` 决定

**构网来源**
- `solutions/Continuous_model/construct_single.py`
- 在初始化阶段生成：`pre/pst/net/m0/md/id2p_name/id2t_name/marks`；构网同时返回 `pre_place_indices`、`pst_place_indices`、`transport_pre_place_idx`（每变迁的 pre/pst 库所索引及运输位 pre 索引），供 `get_action_mask`、`_fire` 等复用以减少检索
- `d_TM1` 在构网中设置 `processing_time=5s`（默认），用于约束“运输位停留后才能 `t_*` 进入腔室”
- 机械手容量固定为 1；双臂模式仅改变 swap 规则，不改变 place 容量口径。
- 单设备清洗：`cleaning_enabled` 为总开关；每腔室阈值与时长统一为 per-chamber map：`cleaning_trigger_wafers_map`、`cleaning_duration_map`。若配置提供 `chambers`，会优先从每腔室块生成/覆盖上述 map。
- ClusterTool 内部使用 `_cleaning_trigger_map`、`_cleaning_duration_map`（per-chamber）；构网时通过 `obs_config` 将上述 map 传给 `build_net`，每个 PM 实例获得对应腔室的清洁参数。

**工艺路线**
- 级联（`device_mode=cascade`）下路线由 `PetriEnvConfig.single_route_name` 在 `single_route_config.routes` 中选取；具体拓扑与 stage 工时以 JSON 为准（仓库示例见 `config/cluster_tool/cascade_routes_1_star.json`）。不再使用数字别名 `route_code` / `single_route_code` 或 `route4_takt_interval`。
- 双臂模式约束：`u_*` 仅在可解析到有效目标腔室时才允许发射；若下游层满导致目标不可解析，则该 `u_*` 必须禁用，避免出现 `LP -> d_TM1 -> LP_done` 的非法短路。
- 上述约束同样适用于 `d_TM1` 为空时：双臂“可任意取片”不等于可忽略目标可达性，目标不可解析时必须禁用 `u_*`。
- 单/级联模式路由门控统一改为 token 队列：`route_queue + route_head_idx`。
  - 仅 `t_*` 使用路由门控码（如 `t_PM7=1, t_PM8=2, t_LLC=3 ...`）。
  - `u_*` 不做路由门控；但 token 每次 fire 都会推进一次队头，用 `-1` 作为 `u_*` 占位通配符。
  - 示例：`[-1, (1,2), -1, 3, ...]` 表示 `u_*` 后到达并行 `t_PM7/t_PM8` 阶段，再进入 `t_LLC` 阶段。

**主要接口**
- `reset()`：重置网状态；返回 `(None, enabled_transition_indices)`，第二项为当前使能变迁索引的有序列表，由 `get_action_mask` 的前 `T` 维派生（与 RL 掩码中变迁段一致）。`Env_PN_Single` 等封装通常忽略该返回值，另行调用 `get_action_mask` 构建 `action_mask`。
- `get_action_mask(wait_action_start=None, n_actions=None) -> np.ndarray`：返回完整离散动作掩码（`transition + wait`）。使能与 wait 规则在此统一完成；实现为判定每个 transition/wait 使能时直接写 `mask[idx]=True`，不先生成动作 id 列表再写入。
- `Env_PN_Single`：`eval_mode=True` 时仍启用 `detailed_reward` 与 `net.eval()`；动作合法性仅通过 TensorDict 的 `action_mask`（及 `ClusterTool.step` 返回值中的 mask）表达，不在环境上缓存逐步使能列表。
- `step(a1=None, detailed_reward=False, wait_duration=None)`：执行单步并返回 `(done, reward_result, scrap, action_mask)`（动作校验 -> 发射/等待 -> 时间推进 -> 奖励 -> mask）；`advance_time()` 内会同步完成 `scrap/qtime` 状态扫描，减少重复 token 遍历。补充：非 WAIT 路径下，若本步 `u_*` 已取走与 `scrap_info` 同 `token_id` 且同源腔室的 resident wafer，则撤销本步 scrap（不终止、不追加 `scrap_penalty`）。
- `get_next_event_delta() -> Optional[int]`：计算当前时刻到下一关键事件的时间差（秒）。通过扫描 `marks` 中运输位 d_TM* 与加工腔室的 token，用不同规则计算；节拍为「下一节拍时刻 − 当前时间」。用于 wait 时截断推进量，避免跨过取片或发片决策点。
- `calc_reward(t1, t2, detailed=False)`：奖励计算（`detailed_reward=True` 时返回含 `total` 的字典）
- `blame_release_violations() -> Dict[int, float]`：基于 `_chamber_timeline` 与 `fire_log` 中 `cleaning_start` 的单设备事后追责，输出 `fire_log_index -> penalty`。
  - **占用时间线**：晶圆加工区间（来自 `_chamber_timeline`）与腔室清洁区间（来自 `fire_log` 的 `cleaning_start`，`_cleaning_trigger_map` 中启用清洗的腔室）合并计算容量占用
  - **仅追责释放动作**：`u_LP`、`u_LLC`、`u_LLD`。`u_PM7`、`u_PM2` 等从加工腔卸载的动作不追责。
  - 追责链路由当前路线的 `release_chain_by_u` / `release_station_aliases`（来自构网 `route_meta`）决定，不再按固定 `single_route_code` 枚举。
- 清洗事件日志会附加写入 `fire_log`（`event_type=cleaning_start|cleaning_end`），用于后续追责/复盘。
- （A 方案 `ClusterTool` 已移除 `calc_wafer_statistics()`；左栏统计见上文「统计」与适配器。）

**max_wafers1_in_system / max_wafers2_in_system 门控（2026-03-31）**
- What changed：`ClusterTool` 在制品上限门控：`u_LP` 发片时 `entered_wafer_count += 1`，`t_LP_done` 完成时 `entered_wafer_count -= 1`；按类型维护 `_entered_wafer_count_by_type`。单路线：`entered_wafer_count < max_wafers1_in_system` 时才允许发片；双子路径：不再使用单一全局在制数卡总 WIP，`_allow_start_for_route_type` 对 `route_type` 1/2 分别要求 `_entered_wafer_count_by_type[type] + 1 <= max_wafers1_in_system` 或 `<= max_wafers2_in_system`。
- Why：上限由配置显式给定；`wafer_type_alloc` 不用于推导 WIP 上限。
- Impact / How to use：该限制只作用于入口发片动作 `u_LP*`，不改变 `n_wafer1+n_wafer2` 的完工判定。`max_wafers1_in_system=0` 时单路线从首步起禁止发片；双子路径下 `route_type` 非 1/2 会抛错。
- Example：
  ```python
  cfg = PetriEnvConfig(
      n_wafer1=12,
      n_wafer2=0,
      max_wafers1_in_system=5,
      max_wafers2_in_system=4,
  )
  ```

**临时执行模式（2026-03-17）**
- `pn_single` 当前默认按“单臂 + 非 FIFO + token 扫描使能”执行：
  - `robot_capacity` 固定为 1（暂不考虑双臂死锁锁定规则）；
  - `get_action_mask()` 内部使能判定先检查 `u_LP`，再扫描系统内 token 生成候选 `u_*/t_*` 并直接写 mask；
  - 运行时除 `LP/LP_done` 外库所按 unit-capacity（1）约束；
  - `check_scrap` 判定改为基于 token 剩余时间：`remaining < -P_Residual_time` 视为驻留违规。
  - 非 WAIT 路径保留“先时间推进再 fire”；若本步 fire 已同步取走触发 resident 违规的同一 wafer，则撤销该步 scrap 影响。

**事后追责相关字段**
- `_chamber_timeline/_chamber_active`：按路径代号记录加工腔体进入离开时间线（`code=0` 为 PM1/PM3/PM4；`code=1` 额外包含 PM6），供 episode 结束后 `blame_release_violations` 使用
- `blame_release_violations` 将 `fire_log` 中的 `cleaning_start` 转为清洁占位区间，与晶圆区间一并计入下游站点容量，故 PM3/PM4 同时清洁时对 s2 的释放会被正确追责

**设备模式字段**
- `single_device_mode`: 当前 A 方案 `ClusterTool` 仅支持 `cascade`；路线由 `single_route_name` + `single_route_config` 指定。

**驻留时间更新规则（单设备）**
- `LP`（type=3）中的 token 不更新 `stay_time`，与 `pn.py` 保持一致

**双臂防死锁规则（使能 / mask）**
- 当 `d_TM1` 为空时：允许任意满足前置条件的 `u_*` 取片。
- 当 `d_TM1` 非空且队首晶圆在“取片时”其 `dst` 层已满：仅允许继续取出该 `dst` 层中的晶圆，禁止取其它位置晶圆。
- `t_*` 始终遵循 FIFO 队首目标约束（队首 `_target_place`）与 `d_TM1` dwell 时间约束。
- 使能计算分为两阶段：
  - Stage1：`pre/pst` + 容量 + 防死锁规则（用于死锁判定，`u_*` 在该阶段忽略清洗目标过滤）
  - Stage2：加工完成、目标可达与 dwell/清洗过滤（`get_action_mask` 内部使能判定最终采用）
- 路由判定规则：仅 `t_*` 读取运输位队首 token 的 `route_queue[route_head_idx]`。
  - 队头为 `-1`：该步对 `t_*` 路由不设限；
  - 队头为 `int`：仅允许匹配该码的 `t_*`；
  - 队头为集合（tuple/list/set）：允许集合内任意码的 `t_*`。
- WAIT 掩码规则（单设备）：
  - 默认启用所有 wait 档位；
  - 若当前路径任一加工腔室存在 token 满足 `stay_time >= processing_time`（加工完成待取片），禁用 `WAIT>5s`，仅保留 `WAIT_5s`。
- `u_LP` 不再使用 Stage2 额外边界拦截；当前仅遵循通用使能规则（加工完成、目标可达、清洗过滤与运输位 dwell 约束）。
- 当 `t_*` 的目标腔室处于 `is_cleaning=True` 时，Stage2 才禁用该变迁（死锁判定仍基于未清洗过滤的 Stage1）。
- 死锁定义：`LP_done` 未完成且 Stage1 无任何使能变迁。

**观测补充（单设备）**
- `Env_PN_Single` 的 observation 统一由 `ClusterTool.get_obs()` 返回（`float32`），顺序为 `LP -> TM -> chamber*`，`LP_done` 不进入主体观测。
- 观测布局（库所顺序、`obs_dim`、各段 `offset`）在 `ClusterTool.__init__` 中确定；`reset()` 仅将 `_obs_places` 重新绑定到克隆后的库所对象，不重建 `obs_dim` 与 buffer。
- PM 库所固定 9 维，末尾第 9 维改为 `near_cleaning_norm`（语义位置不变）：
  - `near_cleaning_norm = (1-is_cleaning) * clip((2-r)/2, 0, 1)`
  - `r = max(N-c, 0)`，`N = max(1, cleaning_trigger_wafers)`，`c = max(0, processed_wafer_count)`
  - 分段语义：`r>=2 -> 0`，`r=1 -> 0.5`，`r=0 -> 1`；清洗中强制 `0`

**运输位 token 的机械臂标识**
- 每个 token 在进入 `d_TM1`（即 `u_*` 发射）时分配 `machine` 字段。
- 分配策略为轮换（round-robin）：
  - 单臂模式固定为 `1`
  - 双臂模式在 `1/2` 间交替

**停滞惩罚（单设备已接入）**
- 沿用 `pn.py` 思路：累计连续 WAIT 时间（`_consecutive_wait_time`）
- 当累计时间达到 `idle_timeout = max(processing_time)+30` 且未触发过时，施加一次 `idle_timeout_penalty`
- 运输位超时惩罚：`d_TM1`（type=2）内晶圆停留超过 `D_Residual_time` 后按超时秒数线性扣分；奖励分项固定启用，系数为 `transport_overtime_coef`（与 `ClusterTool` 内系数字段一致）

**死锁终止语义（单设备）**
- 发生死锁时，episode 终止，增加 `deadlock_count`。
- 死锁惩罚量级与 `scrap_event_penalty` 等价（`deadlock_penalty = -abs(scrap_event_penalty)`）。
- 死锁不计入 `scrap`。
