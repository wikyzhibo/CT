# PDR Route and DFS Search

## Abstract
- What: 定义 `solutions/PDR` 当前默认构网路线与深度优先搜索行为。
- When: 修改 `solutions/PDR/construct.py` 或 `solutions/PDR/net.py` 前必须先读。
- Not: 不包含 obs 构造与 RL 选点逻辑。
- Key rules:
  - 默认路线固定为 `LP -> PM7/PM8 -> LLC -> PM1/PM2 -> LLD -> LP_done`。
  - 变迁命名必须为 `u_<src>_<TMx>` 与 `t_<TMx>_<dst>`。
  - `search` 必须按深度限制 DFS（默认深度 5）并在驻留违规时剪枝。

## When to use
- 需要验证 `construct` 生成的网结构与命名是否符合新规则。
- 需要通过 `search` 收集固定深度叶子节点用于后续策略筛选。

## When NOT to use
- 需要训练可直接执行策略（当前不做 RL 选择）。
- 需要训练侧 PPO 推理导出链路（由 `export_inference_sequence.py` 覆盖）。

## Behavior / Rules
1. 默认构网模板为单路线分叉结构：
   - `LP -> (PM7 | PM8) -> LLC -> (PM1 | PM2) -> LLD -> LP_done`
2. 机械手职责固定：
   - TM2：`LP/PM7/PM8` 取送、`LLC` 送、`LLD` 取
   - TM3：`PM1/PM2` 取送、`LLC` 取、`LLD` 送
3. `search` 行为：
   - 每轮先执行深度受限 DFS（`search_depth=5`），并收集叶子
   - 叶子收集使用显式栈 `collect_leaves_iterative`，不使用递归
   - 通过 `select_node` 计算每个叶子的 `|leaf_clock - current_clock|`，选差值最小的叶子作为下一轮当前状态
   - 抽取后必须清空叶子集合，再继续下一轮 DFS
   - 遇到 resident scrap（`check_scrap=True`）立即剪枝
   - 直到到达终止标识 `self.md` 或无叶子可扩展
4. `takt_cycle` 仅约束 `u_LP_TM2`：
   - 路径状态显式携带 `lp_release_count`
   - 第 `k` 次 `u_LP_TM2` 的最早允许时刻为 `sum(takt_cycle[:k+1])`
   - `_earliest_enable_time` 在 `u_LP_TM2` 上执行 `max(earliest, takt_ready_time)`
5. 叶子采集容器使用模块级全局变量：
   - `LEAF_NODES`
   - `LEAF_CLOCKS`
   - `LEAF_PATHS`
   - `LEAF_PATH_RECORDS`
   - `LEAF_LP_RELEASE_COUNTS`
6. 全程记录变迁发射序列与发射时刻：
   - `Petri.full_transition_path` 保存从起点到终止状态的变迁名称序列
   - `Petri.full_transition_records` 保存从起点到终止状态的事件序列：`[{transition, fire_time}, ...]`
   - 不记录中间 `m/marks` 历史链路
7. PDR->UI 回放序列转换规则（single）：
   - 输入为按路径顺序的 `full_transition_records`
   - 输出遵循 `schema_version=2`，`device_mode=single`，`sequence[*]` 含 `step/time/action/actions`
   - 相邻真实动作时间差 `delta > 5` 时，按每 5 秒插入一个 `WAIT_5s`
   - 等待插入次数固定为 `floor((delta-1)/5)`，不会与下一真实动作同一时刻重叠
8. `Petri` 初始化必须从构网模块加载网络信息，不再依赖外部 `.txt` 拓扑文件解析：
   - `Petri.__init__` -> `construct.build_pdr_net(...)`
   - 初始化时读取 `pre/pst/m0/md/ptime/capacity/id2p_name/id2t_name/marks`

## Configuration / API
- `solutions/PDR/construct.py`
  - `build_default_pdr_info(n_wafer=7, d_ptime=3, default_ttime=2)`
- `solutions/PDR/net.py`
  - `check_scrap(t, firetime, marks) -> bool`
  - `_lp_takt_ready_time(lp_release_count) -> int`
  - `get_leaf_node(m, marks, clock, path, path_records, lp_release_count) -> None`
  - `collect_leaves_iterative(m, marks, clock, depth, lp_release_count) -> None`
  - `search() -> bool`
- `solutions/PDR/parse_sequences.py`
  - `build_single_replay_payload(full_transition_records) -> dict`
  - `export_single_replay_payload(full_transition_records, out_name) -> Path`
- `solutions/PDR/run_pdr.py`
  - `main()` 在 `search()` 完成后自动导出 `seq/pdr_sequence.json`

## Examples
- 正例：调用 `build_default_pdr_info()` 后检查 `id2t_name` 含 `u_PM7_TM2` 与 `t_TM3_PM1`。
- 反例：使用旧命名 `u_A_B` / `t_B` 进行新路线验证。

## Edge Cases / Gotchas
- 当前 `search` 深度不足时不会写入全局叶子容器。
- 若分支在深度到达前无可使能变迁，该分支不会进入深度叶子集合。

## Related Docs
- `docs/README.md`
- `docs/td-petri/td-petri-guide.md`
- `docs/routes.md`
