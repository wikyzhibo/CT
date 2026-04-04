# ClusterTool Reference

## Abstract
- What: 本文档说明 `solutions/A/petri_net.py` 中 `ClusterTool` 的变量、函数作用与关键调用链。
- When: 需要理解 `ClusterTool` 状态字段、动作掩码与 step 执行链时使用。
- Not: 不替代 `pn_api.md` 与 `continuous-model/pn-single.md` 的行为规则。
- Key rules:
  - 变量与函数清单以当前代码静态扫描结果为准。
  - 行号基于当前提交，后续重构后需要重新生成。
  - 本文只解释作用，不改变实现语义。

## When to use
- 需要定位某个 `self.*` 字段负责什么。
- 需要快速查看 `step`、`get_action_mask`、`_fire` 的调用关系。
- 需要补充注释或排查状态漂移时。

## When NOT to use
- 需要训练参数与入口命令时，请看 `docs/training/training-guide.md`。
- 需要完整行为规范时，请看 `docs/continuous-model/pn-single.md`。
- 需要 API 对外契约时，请看 `docs/pn_api.md`。

## Behavior / Rules
- `ClusterTool` 主循环只通过 `step` 推进时间、发射变迁、计算奖励并返回下一步掩码与观测。
- 主链：`step -> (_advance_and_compute_reward / _fire) -> get_action_mask -> get_obs`。
- 关键子链A（mask）：`get_action_mask -> _resolve_required_release_type_for_entry_heads -> _pick_tm1_from_mask -> _allow_t_by_use_count -> _select_min_use_count_target`。
- 关键子链B（release/takt）：`_fire -> _apply_entry_release_delay -> _arm_entry_head_with_takt_delay -> _takt_required_interval`。
- 关键子链C（并发 TM1）：`get_action_mask -> _pick_tm1_from_mask -> _cached_auto_tm1_action`。
- `render_gantt` 只消费评估态甘特缓存，不参与训练路径。

## Configuration / API
- 代码位置：`solutions/A/petri_net.py`。
- 变量总数：`121`（按 `self.*` 扫描）。
- 函数总数：`51`（含局部函数）。

### Variables (All `self.*`)
| 分组 | 变量名 | 初始化位置 | 作用 | 小例子 |
|---|---|---:|---|---|
| 其他内部状态 | `_concurrent` | 21 | 保存动作模式与调度执行开关。 | `bool(concurrent)` |
| 其他内部状态 | `_cached_auto_tm1_action` | 22 | 保存动作模式与调度执行开关。 | `None` |
| 其他内部状态 | `_dual_arm` | 41 | 保存动作模式与调度执行开关。 | `bool(getattr(config, 'dual_arm', False))` |
| 其他内部状态 | `_stride_single_wait_mode` | 58 | 保存 ClusterTool 运行所需的内部状态。 | `bool(self.stride) and len(self.wait_durations) == 1 and...` |
| 其他内部状态 | `max_wafers1_in_system` | 74 | 保存 ClusterTool 运行所需的内部状态。 | `route_entry.get("max_wafer1_in_system",12)` |
| 其他内部状态 | `max_wafers2_in_system` | 75 | 保存 ClusterTool 运行所需的内部状态。 | `route_entry.get("max_wafer2_in_system",0)` |
| 其他内部状态 | `_base_proc_time_map` | 87 | 保存 ClusterTool 运行所需的内部状态。 | `dict(info.get("process_time_map") or {})` |
| 其他内部状态 | `chambers` | 93 | 保存 ClusterTool 运行所需的内部状态。 | `tuple(route_meta.get("timeline_chambers") or route_meta.get("chambers", ()))` |
| 其他内部状态 | `_step_map` | 95 | 保存 ClusterTool 运行所需的内部状态。 | `dict(route_meta.get("step_map", {}))` |
| 其他内部状态 | `_has_repeat_syntax_reentry` | 96 | 保存 ClusterTool 运行所需的内部状态。 | `bool(route_meta.get("has_repeat_syntax_reentry", False))` |
| 其他内部状态 | `_mask_skip_places` | 115 | 保存 ClusterTool 运行所需的内部状态。 | `frozenset({"LP_done"})` |
| 其他内部状态 | `_ready_chambers` | 116 | 保存 ClusterTool 运行所需的内部状态。 | `route_meta.get("ready_chambers") or route_meta.get("chambers")` |
| 其他内部状态 | `_single_process_chambers` | 117 | 保存 ClusterTool 运行所需的内部状态。 | `self.chambers` |
| 其他内部状态 | `_initial_capacity` | 128 | 保存 ClusterTool 运行所需的内部状态。 | `np.array(info["capacity"], dtype=int)` |
| 其他内部状态 | `_qtime_violated_tokens` | 174 | 保存 ClusterTool 运行所需的内部状态。 | `set()` |
| 其他内部状态 | `_last_deadlock` | 193 | 保存 ClusterTool 运行所需的内部状态。 | `False` |
| 其他内部状态 | `_training` | 215 | 保存 ClusterTool 运行所需的内部状态。 | `True` |
| 其他内部状态 | `_last_state_scan` | 216 | 保存 ClusterTool 运行所需的内部状态。 | `{}` |
| 其他内部状态 | `_ready_chambers_set` | 219 | 保存 ClusterTool 运行所需的内部状态。 | `frozenset(self._ready_chambers)` |
| 奖励与约束参数 | `done_event_reward` | 28 | 保存奖励或惩罚相关参数/缓存。 | `int(config.done_event_reward)` |
| 奖励与约束参数 | `finish_event_reward` | 29 | 保存奖励或惩罚相关参数/缓存。 | `self.done_event_reward * 6` |
| 奖励与约束参数 | `scrap_event_penalty` | 30 | 保存奖励或惩罚相关参数/缓存。 | `int(config.scrap_event_penalty)` |
| 奖励与约束参数 | `idle_event_penalty` | 31 | 保存奖励或惩罚相关参数/缓存。 | `float(config.idle_event_penalty)` |
| 奖励与约束参数 | `warn_coef_penalty` | 33 | 保存奖励或惩罚相关参数/缓存。 | `float(config.warn_coef_penalty)` |
| 奖励与约束参数 | `processing_coef_reward` | 34 | 保存奖励或惩罚相关参数/缓存。 | `float(config.processing_coef_reward)` |
| 奖励与约束参数 | `transport_overtime_coef_penalty` | 35 | 保存奖励或惩罚相关参数/缓存。 | `float(config.transport_overtime_coef_penalty)` |
| 奖励与约束参数 | `time_coef_penalty` | 36 | 保存奖励或惩罚相关参数/缓存。 | `float(config.time_coef_penalty)` |
| 奖励与约束参数 | `P_Residual_time` | 39 | 保存 episode 级计数器与时间状态。 | `int(config.P_Residual_time)` |
| 奖励与约束参数 | `D_Residual_time` | 40 | 保存 episode 级计数器与时间状态。 | `int(config.D_Residual_time)` |
| 奖励与约束参数 | `_per_wafer_reward` | 173 | 保存奖励或惩罚相关参数/缓存。 | `0.0` |
| 奖励与约束参数 | `_idle_penalty_applied` | 175 | 保存奖励或惩罚相关参数/缓存。 | `False` |
| 拓扑与变迁索引 | `_u_targets` | 94 | 保存变迁索引或运输映射缓存。 | `dict(route_meta.get("u_targets", {}))` |
| 拓扑与变迁索引 | `m0` | 125 | 保存 Petri 网核心结构/矩阵信息。 | `info["m0"]` |
| 拓扑与变迁索引 | `m` | 126 | 保存 Petri 网核心结构/矩阵信息。 | `self.m0.copy()` |
| 拓扑与变迁索引 | `k` | 127 | 保存 Petri 网核心结构/矩阵信息。 | `info["capacity"]` |
| 拓扑与变迁索引 | `id2p_name` | 129 | 保存 Petri 网核心结构/矩阵信息。 | `info["id2p_name"]` |
| 拓扑与变迁索引 | `id2t_name` | 130 | 保存 Petri 网核心结构/矩阵信息。 | `info["id2t_name"]` |
| 拓扑与变迁索引 | `_t_target_place_map` | 134 | 保存变迁索引或运输映射缓存。 | `dict(info.get("t_target_place_map") or {})` |
| 拓扑与变迁索引 | `_t_code_to_place` | 139 | 保存变迁索引或运输映射缓存。 | `{}` |
| 拓扑与变迁索引 | `idle_idx` | 151 | 保存 ClusterTool 运行所需的内部状态。 | `info["idle_idx"]` |
| 拓扑与变迁索引 | `P` | 152 | 保存 Petri 网核心结构/矩阵信息。 | `info["P"]` |
| 拓扑与变迁索引 | `T` | 153 | 保存 Petri 网核心结构/矩阵信息。 | `info["T"]` |
| 拓扑与变迁索引 | `_pre_place_indices` | 155 | 保存 ClusterTool 运行所需的内部状态。 | `info["pre_place_indices"]` |
| 拓扑与变迁索引 | `_pst_place_indices` | 156 | 保存 ClusterTool 运行所需的内部状态。 | `info["pst_place_indices"]` |
| 拓扑与变迁索引 | `_transport_pre_place_idx` | 157 | 保存变迁索引或运输映射缓存。 | `info["transport_pre_place_idx"]` |
| 拓扑与变迁索引 | `_fixed_topology` | 158 | 保存 ClusterTool 运行所需的内部状态。 | `bool(info.get("fixed_topology", False))` |
| 拓扑与变迁索引 | `_u_transition_by_source` | 185 | 保存变迁索引或运输映射缓存。 | `{}` |
| 拓扑与变迁索引 | `_u_transition_by_source_transport` | 186 | 保存变迁索引或运输映射缓存。 | `{}` |
| 拓扑与变迁索引 | `_t_transitions_by_transport` | 187 | 保存变迁索引或运输映射缓存。 | `{}` |
| 拓扑与变迁索引 | `_tm1_transition_indices` | 188 | 保存变迁索引或运输映射缓存。 | `[]` |
| 拓扑与变迁索引 | `_tm2_transition_indices` | 189 | 保存变迁索引或运输映射缓存。 | `[]` |
| 拓扑与变迁索引 | `_tm3_transition_indices` | 190 | 保存变迁索引或运输映射缓存。 | `[]` |
| 清洗状态 | `_cleaning_enabled` | 45 | 保存清洗策略阈值与运行时清洗状态。 | `bool(config.cleaning_enabled)` |
| 清洗状态 | `_cleaning_default_trigger` | 47 | 保存清洗策略阈值与运行时清洗状态。 | `0` |
| 清洗状态 | `_cleaning_trigger_map` | 52 | 保存清洗策略阈值与运行时清洗状态。 | `{ str(name): max(0, int(value)) for name, value in dict(getattr(config,...` |
| 状态与观测缓存 | `marks` | 161 | 保存运行期库所对象与访问统计。 | `self._clone_marks(info["marks"])` |
| 状态与观测缓存 | `ori_marks` | 162 | 保存运行期库所对象与访问统计。 | `self._clone_marks(self.marks)` |
| 状态与观测缓存 | `_place_by_name` | 220 | 保存运行期库所对象与访问统计。 | `{}` |
| 状态与观测缓存 | `_obs_place_names` | 221 | 保存观测向量构建所需的索引与缓存。 | `[]` |
| 状态与观测缓存 | `obs_dim` | 222 | 保存观测向量构建所需的索引与缓存。 | `0` |
| 状态与观测缓存 | `_obs_return_copy` | 223 | 保存观测向量构建所需的索引与缓存。 | `True` |
| 状态与观测缓存 | `_lp_done` | 225 | 保存运行期库所对象与访问统计。 | `self._place_by_name.get("LP_done")` |
| 状态与观测缓存 | `_obs_places` | 235 | 保存观测向量构建所需的索引与缓存。 | `[self._place_by_name[name] for name in obs_names]` |
| 状态与观测缓存 | `_obs_offsets` | 242 | 保存观测向量构建所需的索引与缓存。 | `offsets` |
| 状态与观测缓存 | `_obs_buffer` | 244 | 保存观测向量构建所需的索引与缓存。 | `np.zeros(self.obs_dim, dtype=np.float32)` |
| 统计计数与时间 | `time` | 164 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `idle_timeout` | 165 | 保存 episode 级计数器与时间状态。 | `max((p.processing_time for p in self.marks), default=0) + 50` |
| 统计计数与时间 | `entered_wafer_count` | 166 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `done_count` | 167 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `scrap_count` | 168 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `deadlock_count` | 169 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `resident_violation_count` | 170 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `qtime_violation_count` | 171 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `_consecutive_wait_time` | 176 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `_place_use_count` | 177 | 保存 episode 级计数器与时间状态。 | `{ str(place.name): 0 for place in self.marks }` |
| 统计计数与时间 | `_last_u_entry_fire_time` | 205 | 保存 episode 级计数器与时间状态。 | `0` |
| 统计计数与时间 | `_entered_wafer_count_by_type` | 211 | 保存 episode 级计数器与时间状态。 | `{int(t): 0 for t in sorted(all_types)}` |
| 评估与甘特记录 | `fire_log` | 172 | 保存评估回放与甘特图记录数据。 | `[]` |
| 评估与甘特记录 | `_eval_gantt_place_to_sm` | 180 | 保存评估回放与甘特图记录数据。 | `{}` |
| 评估与甘特记录 | `_eval_gantt_lane_cursor` | 181 | 保存评估回放与甘特图记录数据。 | `{}` |
| 评估与甘特记录 | `_eval_gantt_slots` | 182 | 保存评估回放与甘特图记录数据。 | `{}` |
| 评估与甘特记录 | `_eval_gantt_closed_ops` | 183 | 保存评估回放与甘特图记录数据。 | `[]` |
| 路线与发片控制 | `swap_duration` | 42 | 保存节拍/比例发片控制状态。 | `10` |
| 路线与发片控制 | `_cleaning_default_duration` | 46 | 保存清洗策略阈值与运行时清洗状态。 | `0` |
| 路线与发片控制 | `_cleaning_duration_map` | 48 | 保存清洗策略阈值与运行时清洗状态。 | `{ str(name): max(0, int(value)) for name, value in dict(getattr(config,...` |
| 路线与发片控制 | `single_route_config` | 66 | 保存路线、子路径或发片入口映射。 | `config.single_route_config` |
| 路线与发片控制 | `single_route_name` | 67 | 保存路线、子路径或发片入口映射。 | `config.single_route_name` |
| 路线与发片控制 | `_gantt_route_stages` | 70 | 保存路线、子路径或发片入口映射。 | `[ [str(place_name) for place_name in list(stage or [])] for stage in...` |
| 路线与发片控制 | `_route_stages` | 92 | 保存路线、子路径或发片入口映射。 | `[list(stage) for stage in route_stages]` |
| 路线与发片控制 | `_multi_subpath` | 99 | 保存路线、子路径或发片入口映射。 | `bool(route_meta.get("multi_subpath", False))` |
| 路线与发片控制 | `_subpath_to_type` | 100 | 保存路线、子路径或发片入口映射。 | `route_meta.get("subpath_to_type")` |
| 路线与发片控制 | `_wafer_type_to_subpath` | 101 | 保存路线、子路径或发片入口映射。 | `route_meta.get("wafer_type_to_subpath")` |
| 路线与发片控制 | `_subpath_route_stages` | 103 | 保存路线、子路径或发片入口映射。 | `{ str(subpath_name): [ [str(place_name) for place_name in list(stage or [])]...` |
| 路线与发片控制 | `_takt_policy` | 110 | 保存节拍/比例发片控制状态。 | `str(route_meta.get("takt_policy", "") or "")` |
| 路线与发片控制 | `_wafer_type_to_load_port` | 111 | 保存路线、子路径或发片入口映射。 | `route_meta.get("wafer_type_to_load_port")` |
| 路线与发片控制 | `_load_port_names` | 112 | 保存路线、子路径或发片入口映射。 | `route_meta.get("load_port_names")` |
| 路线与发片控制 | `_wafer_type_to_release_place` | 113 | 保存路线、子路径或发片入口映射。 | `route_meta.get("wafer_type_to_release_place") or {}` |
| 路线与发片控制 | `_release_control_places` | 114 | 保存路线、子路径或发片入口映射。 | `tuple(route_meta.get("release_control_places") or self._load_port_names)` |
| 路线与发片控制 | `_shared_ratio_cycle_enabled` | 118 | 保存节拍/比例发片控制状态。 | `False` |
| 路线与发片控制 | `_shared_ratio_cycle_types` | 119 | 保存节拍/比例发片控制状态。 | `()` |
| 路线与发片控制 | `_shared_ratio_cycle_idx` | 120 | 保存节拍/比例发片控制状态。 | `0` |
| 路线与发片控制 | `_lp_pick_cycle_idx` | 121 | 保存节拍/比例发片控制状态。 | `0` |
| 路线与发片控制 | `_t_route_code_map` | 131 | 保存路线、子路径或发片入口映射。 | `dict(info.get("t_route_code_map") or {})` |
| 路线与发片控制 | `_token_route_queue_templates_by_type` | 132 | 保存路线、子路径或发片入口映射。 | `info.get("token_route_queue_templates_by_type")` |
| 路线与发片控制 | `_token_route_type_sequence` | 133 | 保存路线、子路径或发片入口映射。 | `info.get("token_route_type_sequence")` |
| 路线与发片控制 | `_route_source_target_transport` | 135 | 保存路线、子路径或发片入口映射。 | `info.get("route_source_target_transport")` |
| 路线与发片控制 | `_t_route_code_by_idx` | 136 | 保存路线、子路径或发片入口映射。 | `[ int(self._t_route_code_map.get(name, -1)) for name in self.id2t_name ]` |
| 路线与发片控制 | `_takt_result_by_type` | 196 | 保存节拍/比例发片控制状态。 | `{ int(k): v for k, v in dict(takt_payload.get("takt_result_by_type") or...` |
| 路线与发片控制 | `_takt_result` | 200 | 保存节拍/比例发片控制状态。 | `takt_payload.get("takt_result_default")` |
| 路线与发片控制 | `_u_entry_release_count` | 206 | 保存节拍/比例发片控制状态。 | `0` |
| 路线与发片控制 | `_u_entry_release_count_by_type` | 210 | 保存节拍/比例发片控制状态。 | `{int(t): 0 for t in sorted(all_types)}` |
| 路线与发片控制 | `_entry_release_ready_time_shared` | 212 | 保存节拍/比例发片控制状态。 | `0` |
| 路线与发片控制 | `_entry_release_ready_time_by_type` | 213 | 保存节拍/比例发片控制状态。 | `{int(t): 0 for t in sorted(all_types)}` |
| 配置与运行参数 | `config` | 20 | 保存环境配置对象。 | `config` |
| 配置与运行参数 | `MAX_TIME` | 25 | 保存 ClusterTool 运行所需的内部状态。 | `config.MAX_TIME` |
| 配置与运行参数 | `n_wafer` | 26 | 保存 ClusterTool 运行所需的内部状态。 | `config.n_wafer` |
| 配置与运行参数 | `wait_durations` | 56 | 保存节拍/比例发片控制状态。 | `_normalize_wait_durations(config.wait_durations)` |
| 配置与运行参数 | `stride` | 57 | 保存动作模式与调度执行开关。 | `config.stride` |
| 配置与运行参数 | `ttime` | 65 | 保存 ClusterTool 运行所需的内部状态。 | `5` |
| 配置与运行参数 | `n_wafer1` | 77 | 保存 ClusterTool 运行所需的内部状态。 | `int(self.n_wafer * ratio[0] / sum(ratio))` |
| 配置与运行参数 | `n_wafer2` | 78 | 保存 ClusterTool 运行所需的内部状态。 | `int(self.n_wafer - self.n_wafer1)` |

### Functions (All 51 defs)
| 函数名 | 定义行 | 作用 | 关键调用点 |
|---|---:|---|---|
| `ClusterTool.__init__` | 18 | 初始化配置、构网结果、状态容器与索引缓存。 | `_init_shared_ratio_release_cycle@122, _clone_marks@161, _clone_marks@162, _reset_eval_gantt_records@184, _init_cleaning_state@194` |
| `ClusterTool.train` | 251 | 切换为训练模式。 | `无 self.* 调用` |
| `ClusterTool.eval` | 256 | 切换为评估模式。 | `无 self.* 调用` |
| `ClusterTool.step` | 261 | 执行一步仿真推进并返回 done/reward/scrap/mask/obs。 | `get_action_mask@285, get_obs@286, get_next_event_delta@320, get_next_event_delta@328, _advance_and_compute_reward@337` |
| `ClusterTool.reset` | 404 | 重置网状态、统计计数与入口节拍游标。 | `_clone_marks@405, _init_cleaning_state@424, _reset_eval_gantt_records@436, get_action_mask@444` |
| `ClusterTool.get_obs` | 453 | 将当前库所观测写入缓冲并返回观测向量。 | `无 self.* 调用` |
| `ClusterTool._advance_and_compute_reward` | 459 | 推进时间并累计奖励、惩罚与驻留 scrap 扫描结果。 | `无 self.* 调用` |
| `ClusterTool._reset_eval_gantt_records` | 566 | 初始化评估态甘特图记录槽位。 | `无 self.* 调用` |
| `ClusterTool._record_eval_gantt_enter` | 588 | 记录晶圆进入腔室时的甘特开始片段。 | `无 self.* 调用` |
| `ClusterTool._record_eval_gantt_exit` | 608 | 闭合晶圆离开腔室时的甘特片段。 | `无 self.* 调用` |
| `ClusterTool._fire` | 629 | 执行一个或多个变迁发射并更新 token、计数与日志。 | `_transition_transport_place@641, _on_processing_unload@684, _token_current_stage_process_time@687, _is_swap_eligible@691, _on_processing_unload@695` |
| `ClusterTool._fire._transport_order` | 640 | 定义多变迁发射时的运输位优先级排序。 | `_transition_transport_place@641` |
| `ClusterTool._should_cancel_resident_scrap_after_fire` | 780 | 判断发射后是否应撤销同步 resident scrap。 | `_should_cancel_resident_scrap_after_fire@783` |
| `ClusterTool._token_next_target` | 819 | 从 token 路由队列推断下一目标腔室。 | `无 self.* 调用` |
| `ClusterTool._token_current_stage_process_time` | 841 | 读取 token 当前阶段工时配置。 | `无 self.* 调用` |
| `ClusterTool._gate_targets_from_tok_gate` | 854 | 把路由 gate 归一化为允许目标集合。 | `无 self.* 调用` |
| `ClusterTool._stage_targets_for_candidates` | 877 | 计算候选目标与 gate 的交集目标序列。 | `_gate_targets_from_tok_gate@881` |
| `ClusterTool._select_min_use_count_target` | 889 | 按 use_count 最小且稳定顺序选择目标腔室。 | `_stage_targets_for_candidates@896` |
| `ClusterTool._route_gate_allows_t` | 918 | 判定 gate 是否允许指定的 t_route_code。 | `无 self.* 调用` |
| `ClusterTool._allow_t_by_use_count` | 932 | 校验 t_* 目标是否满足并行选机一致性约束。 | `_gate_targets_from_tok_gate@945, _select_min_use_count_target@947` |
| `ClusterTool._entry_type_head_tokens` | 958 | 收集各 release 入口队首 wafer 的 route_type。 | `无 self.* 调用` |
| `ClusterTool._allow_start_for_route_type` | 971 | 按在制上限判断 route_type 是否允许发片。 | `无 self.* 调用` |
| `ClusterTool._build_release_ratio_cycle` | 987 | 将 ratio 配置展开为 release 轮转序列。 | `无 self.* 调用` |
| `ClusterTool._init_shared_ratio_release_cycle` | 997 | 初始化 shared+ratio 的 release 轮转状态。 | `_build_release_ratio_cycle@1006` |
| `ClusterTool._required_release_type` | 1020 | 返回当前 release 轮次要求的 route_type。 | `无 self.* 调用` |
| `ClusterTool._resolve_required_release_type_for_entry_heads` | 1030 | 结合入口队首状态解析当前可执行 route_type。 | `_required_release_type@1031, _entry_type_head_tokens@1037` |
| `ClusterTool._advance_release_ratio_cycle` | 1052 | 推进 release 轮转游标。 | `无 self.* 调用` |
| `ClusterTool._advance_lp_pick_cycle` | 1061 | 推进 TM1 从 LP 取料轮转游标。 | `无 self.* 调用` |
| `ClusterTool._required_lp_pick_type` | 1071 | 返回当前 LP 取料轮次要求的 route_type。 | `无 self.* 调用` |
| `ClusterTool._takt_required_interval` | 1082 | 返回指定 route_type 的节拍间隔需求。 | `无 self.* 调用` |
| `ClusterTool._entry_delay_remaining` | 1130 | 计算当前 route_type 还需等待的节拍时长。 | `无 self.* 调用` |
| `ClusterTool._apply_entry_release_delay` | 1140 | 将节拍延迟写回入口 token 的 stay_time。 | `_entry_delay_remaining@1142` |
| `ClusterTool._arm_entry_head_with_takt_delay` | 1149 | 发片后为下一入口队首挂载节拍延迟。 | `_takt_required_interval@1154, _entry_type_head_tokens@1160, _apply_entry_release_delay@1162, _entry_type_head_tokens@1169, _apply_entry_release_delay@1172` |
| `ClusterTool._build_transition_index` | 1180 | 构建 u/t 变迁到 source/transport 的索引缓存。 | `_transition_transport_place@1200` |
| `ClusterTool._clone_marks` | 1214 | 深拷贝 marks 与 token，避免共享引用。 | `无 self.* 调用` |
| `ClusterTool._transition_target_place` | 1222 | 通过后置索引解析 t_idx 对应目标库所名。 | `无 self.* 调用` |
| `ClusterTool._transition_transport_place` | 1232 | 通过前置索引解析 t_idx 对应运输位库所名。 | `无 self.* 调用` |
| `ClusterTool._transport_for_t_target` | 1244 | 按 source->target 映射选择运输位名称。 | `无 self.* 调用` |
| `ClusterTool._init_cleaning_state` | 1252 | 初始化 PM 清洗阈值与运行状态字段。 | `无 self.* 调用` |
| `ClusterTool.get_next_event_delta` | 1264 | 计算下一关键事件时间差并同步关键标记。 | `_entry_type_head_tokens@1289, _entry_delay_remaining@1292` |
| `ClusterTool._on_processing_unload` | 1302 | 处理加工腔卸片后的清洗计数与清洗状态。 | `无 self.* 调用` |
| `ClusterTool._is_swap_eligible` | 1333 | 判断目标位是否满足双臂 swap 条件。 | `无 self.* 调用` |
| `ClusterTool._will_swap` | 1350 | 预判当前 t_idx 是否会触发 swap。 | `_is_swap_eligible@1361` |
| `ClusterTool._is_next_stage_available` | 1364 | 判断 source 下游是否存在可执行目标并返回目标名。 | `_select_min_use_count_target@1375` |
| `ClusterTool._pick_tm1_from_mask` | 1390 | 从全量掩码按优先级自动挑选 TM1 动作。 | `_transition_target_place@1434, _allow_t_by_use_count@1437, _required_lp_pick_type@1491` |
| `ClusterTool._pick_tm1_from_mask._is_struct_enabled_pick` | 1410 | 缓存并判定 TM1 候选 t_idx 的结构性使能。 | `无 self.* 调用` |
| `ClusterTool._pick_tm1_from_mask._lp_head_route_type` | 1497 | 读取指定 LP 队首晶圆的 route_type。 | `无 self.* 调用` |
| `ClusterTool._tm_masks_from_full` | 1516 | 把全量动作掩码投影成 TM1/TM2/TM3 三段掩码。 | `无 self.* 调用` |
| `ClusterTool.get_action_mask` | 1535 | 生成全量或并发三头动作掩码并缓存自动 TM1 动作。 | `_resolve_required_release_type_for_entry_heads@1567, _entry_delay_remaining@1577, _allow_start_for_route_type@1587, _token_next_target@1589, _transport_for_t_target@1592` |
| `ClusterTool.get_action_mask._is_struct_enabled` | 1556 | 缓存并判定通用变迁的结构性使能。 | `无 self.* 调用` |
| `ClusterTool.render_gantt` | 1688 | 基于评估日志渲染腔室甘特图。 | `无 self.* 调用` |

## Examples
- 正例：排查 `step` 奖励异常时，先看 `step` 调用链，再查 `_advance_and_compute_reward` 与 `_fire` 对应变量。
- 正例：定位入口节拍问题时，优先查看 `_entry_release_ready_time_*` 与 `_takt_result_by_type`。
- 反例：把 `self.*` 变量当作跨文件公共 API 直接依赖。
- 反例：忽略 `_concurrent` 与 `_cached_auto_tm1_action` 就分析并发动作。

## Edge Cases / Gotchas
- `_cleaning_duration_map`、`_cleaning_trigger_map` 在初始化早期与 route_meta 合并后会被覆盖，阅读时要看最终赋值。
- `_cached_auto_tm1_action` 只在并发模式由 `get_action_mask` 写入，`step` 会消费后清空。
- `reset` 会重建 `marks` 与 `_place_by_name`，不要持有旧 `Place` 引用。
- 行号与表达式例子来自当前代码快照，不保证未来版本一致。

## Related Docs
- `docs/README.md`
- `docs/pn_api.md`
- `docs/continuous-model/pn-single.md`
- `docs/CHANGELOG.md`
