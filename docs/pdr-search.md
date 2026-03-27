# PDR Route and DFS Search

## Abstract
- What: 定义 `solutions/PDR` 当前默认构网路线与深度优先搜索行为。
- When: 修改 `solutions/PDR/construct.py`、`solutions/PDR/core.py`、`solutions/PDR/train.py` 或配套导出/绘图脚本前必须先读。
- Not: 不包含 obs 构造与 RL 选点逻辑。
- Key rules:
  - 默认路线固定为 `LP -> PM7/PM8 -> LLC -> PM1/PM2 -> LLD -> LP_done`。
  - 变迁命名必须为 `u_<src>_<TMx>_<i>`（同一 `src+TMx` 下递增）与 `t_<TMx>_<dst>`。
  - `search` 必须按深度限制 DFS（默认深度 5）并在驻留违规时剪枝。

## When to use
- 需要验证 `construct` 生成的网结构与命名是否符合新规则。
- 需要通过 `search` 收集固定深度叶子节点用于后续策略筛选。

## When NOT to use
- 需要在默认 `search()` 推理流程中直接启用 PPO 选点（默认仍为 SPT 规则，不自动加载策略模型）。
- 需要训练侧 PPO 推理导出链路（由 `export_inference_sequence.py` 覆盖）。

## Behavior / Rules
1. 默认构网模板为单路线分叉结构：
   - `LP -> (PM7 | PM8) -> LLC -> (PM1 | PM2) -> LLD -> LP_done`
2. 机械手职责固定：
   - TM2：`LP/PM7/PM8` 取送、`LLC` 送、`LLD` 取
   - TM3：`PM1/PM2` 取送、`LLC` 取、`LLD` 送
3. 构网弧结构固定为：
   - 每条模块边 `A->B` 必须映射为 `A -> u_A_TMx_i -> d_TMx_B -> t_TMx_B -> B`
   - `d_TMx_B` 按 `(TMx, dst)` 共享；同一 `TMx` 指向同一 `dst` 的多条上游边共用一个 `d` 库所
   - `TM2/TM3` 为显式资源库所，`u` 占用、`t` 释放（`TMx->u_A_TMx_i` 与 `t_TMx_B->TMx`）
4. `search` 行为：
   - 每轮先执行深度受限 DFS（`search_depth=5`），并收集叶子
   - 叶子收集使用显式栈 `collect_leaves_iterative`，不使用递归
   - 通过 `select_node` 计算每个叶子的 `|leaf_clock - current_clock|`，选差值最小的叶子作为下一轮当前状态
   - 抽取后必须清空叶子集合，再继续下一轮 DFS
   - 遇到 resident scrap（`check_scrap=True`）立即剪枝
  - 直到终止库所 `LP_done` 的 token 数量达到 `n_wafer`，或无叶子可扩展
5. `takt_cycle` 仅约束 `u_LP_TM2`：
   - 路径状态显式携带 `lp_release_count`
   - 第 `k` 次 `u_LP_TM2` 的最早允许时刻为 `sum(takt_cycle[:k+1])`
   - `_earliest_enable_time` 在 `u_LP_TM2` 上执行 `max(earliest, takt_ready_time)`
6. 叶子采集容器使用模块级全局变量：
   - `LEAF_NODES`
   - `LEAF_CLOCKS`
   - `LEAF_PATHS`
   - `LEAF_PATH_RECORDS`
   - `LEAF_LP_RELEASE_COUNTS`
7. 全程记录变迁发射序列与发射时刻：
   - `Petri.full_transition_path` 保存从起点到终止状态的变迁名称序列
   - `Petri.full_transition_records` 保存从起点到终止状态的事件序列：`[{transition, fire_time}, ...]`
   - 不记录中间 `m/marks` 历史链路
8. PDR->UI 回放序列转换规则（single）：
   - 输入为按路径顺序的 `full_transition_records`
   - 输出遵循 `schema_version=2`，`device_mode=single`，`sequence[*]` 含 `step/time/action/actions`
   - 相邻真实动作时间差 `delta > 5` 时，按每 5 秒插入一个 `WAIT_5s`
   - 等待插入次数固定为 `floor((delta-1)/5)`，不会与下一真实动作同一时刻重叠
  - `u_<src>_<TMx>_<i>` 在导出时归一化为 `u_<src>_<TMx>`
9. `Petri` 初始化必须从构网模块加载网络信息，不再依赖外部 `.txt` 拓扑文件解析：
   - `Petri.__init__` -> `construct.build_pdr_net(...)`
   - 初始化时读取 `pre/pst/m0/md/ptime/capacity/id2p_name/id2t_name/marks`
10. `Petri` 必须在初始化阶段缓存弧关系索引：
  - `pre_places_by_t[t]`：变迁 `t` 的前置库所索引列表
  - `pst_places_by_t[t]`：变迁 `t` 的后置库所索引列表
  - `mask_t/_earliest_enable_time/_tpn_fire/check_scrap` 复用缓存，不重复扫描 `pre/pst` 矩阵
11. 训练态 PPO 叶子选择规则（仅 `train.py`）：
  - 每个训练 step 必须调用 `collect_leaves_iterative` 生成候选叶子。
  - 候选叶子数量大于 `k`（默认 `k=8`）时，只保留 `abs(leaf_clock-current_clock)` 最小的前 `k` 个。
  - 候选叶子数量不足 `k` 时，`action_mask` 仅前 `valid_count` 位为 `True`，其余位为 `False`。
  - step 基础奖励固定为 `-delta_clock`（`delta_clock = abs(leaf_clock-current_clock)`）。
  - 若本步 `finish=True`，在基础奖励上额外追加 `+1000`。
  - 若本步未收集到任何叶子（原本会触发断言），训练路径不抛断言，改为 `scrap=True`、附加惩罚 `-1000`，并立即 `reset` 训练状态。
  - 训练日志必须输出 `makespan`（按 batch 统计 `finish=True` 样本的平均 `time`）。
  - 训练态与推理态分离：`search()` 仍按现有 SPT 规则选叶，不读取 PPO 模型。
12. `solutions/B` 的 `u_LP` 节拍口径：
  - 构网时仍会给 `LP` 队列各 wafer 写入初始 `token_enter_time` 前缀：第 `k` 片（`k` 从 0 开始）使用 `sum(takt_cycle[:k])`，第 1 片为 `0`。
  - 每次发射 `u_LP_TM2_*` 后，必须动态抬升 `LP` 新队头 wafer 的 `token_enter_time` 到 `max(old_enter_time, fire_time + 180)`，保证任意相邻两次 `u_LP` 实际发射时间差 `>= 180`。
  - `get_enable_t` 不追加 `u_LP` 专门分支，统一沿用通用 `max(head_enter_time + ptime)`；节拍约束通过上述 `token_enter_time` 动态更新生效。
  - DFS 不再传递 `lp_release_count/last_lp_release_time`。

## Configuration / API
- `solutions/PDR/construct.py`
  - `build_pdr_net(n_wafer=7) -> dict`
- `solutions/PDR/core.py`
  - `check_scrap(t, firetime, marks) -> bool`
  - `_lp_takt_ready_time(lp_release_count) -> int`
  - `get_leaf_node(m, marks, clock, path, path_records, lp_release_count) -> None`
  - `collect_leaves_iterative(m, marks, clock, depth, lp_release_count) -> None`
  - `prepare_train_candidates(candidate_k=8) -> dict`
  - `step(action_idx=None, mode="train") -> (obs, reward, done, next_mask, info)`
  - `search() -> bool`
- `solutions/PDR/train.py`
  - `python -m solutions.PDR.train`：PPO 训练入口
  - 默认训练指标输出：`results/training_metrics.json`
- `solutions/PDR/parse_sequences.py`
  - `build_single_replay_payload(full_transition_records) -> dict`
  - `export_single_replay_payload(full_transition_records, out_name) -> Path`，输出 `seq/<out_name>.json`
- `solutions/PDR/plot_train_metrics.py`
  - `python -m solutions.PDR.plot_train_metrics --input <training_metrics.json> --output <out.png>`
  - `python -m solutions.PDR.plot_train_metrics --compare-inputs <a.json> <b.json> --output <out.png>`
- `solutions/PDR/run_pdr.py`
  - `main()` 在 `search()` 完成后自动导出 `seq/pdr_sequence.json`

## Examples
- 正例：调用 `build_pdr_net()` 后检查存在 `TM2/TM3` 资源库所与 `d_TM2_LLC`、`d_TM3_LLD` 这类运输库所，且 `id2t_name` 含 `u_PM7_TM2_1` 与 `t_TM3_PM1`。
- 反例：让同一个 `t_TMx_B` 同时依赖多个 `d` 前置库所。

## Edge Cases / Gotchas
- 当前 `search` 深度不足时不会写入全局叶子容器。
- 若分支在深度到达前无可使能变迁，该分支不会进入深度叶子集合。

## Related Docs
- `docs/README.md`
- `docs/td-petri/td-petri-guide.md`
- `docs/routes.md`

## Change Notes
- 2026-03-27: 修复 `solutions/B` 的 `u_LP` 节拍漂移：在 `u_LP_TM2_*` 发射后动态更新 `LP` 新队头 `token_enter_time=max(old, fire_time+180)`，将节拍口径从“仅静态前缀时间”收敛为“严格相邻发射间隔 >=180”。
- 2026-03-27: `solutions/B` 中 `u_LP` 节拍门控改为在构造 `marks` 时写入 `LP token_enter_time`（前缀和口径 `sum(takt_cycle[:k])`，首片=0）；`get_enable_t` 删除 `u_LP` 专门门控，DFS 同步删除 `lp_release_count/last_lp_release_time` 传递链路。
- 2026-03-26: 修复 `solutions.PDR` 模块内引用为包内导入；`parse_sequences.py` 导出目录统一到仓库根 `seq/` 并收口文件名；`train.py` 默认训练指标输出改为 `results/training_metrics.json`；`plot_train_metrics.py` 单文件/对比模式统一使用 `--output` 输出 PNG。
