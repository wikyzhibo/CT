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
- 需要完整回放 JSON 导出链路（当前文档不覆盖）。

## Behavior / Rules
1. 默认构网模板为单路线分叉结构：
   - `LP -> (PM7 | PM8) -> LLC -> (PM1 | PM2) -> LLD -> LP_done`
2. 机械手职责固定：
   - TM2：`LP/PM7/PM8` 取送、`LLC` 送、`LLD` 取
   - TM3：`PM1/PM2` 取送、`LLC` 取、`LLD` 送
3. `search(m, marks, time, depth=5)` 行为：
   - 仅做 DFS 展开，不做 best-first 终止
   - 遇到 resident scrap（`check_scrap=True`）立即剪枝
   - 深度到达后通过 `get_leaf_node` 记录叶子
4. 叶子采集容器使用模块级全局变量：
   - `LEAF_NODES`
   - `LEAF_CLOCKS`
   - `LEAF_PATHS`
5. `Petri` 初始化必须从构网模块加载网络信息，不再依赖外部 `.txt` 拓扑文件解析：
   - `Petri.__init__` -> `construct.build_pdr_net(...)`
   - 初始化时读取 `pre/pst/m0/md/ptime/capacity/id2p_name/id2t_name/marks`

## Configuration / API
- `solutions/PDR/construct.py`
  - `build_default_pdr_info(n_wafer=7, d_ptime=3, default_ttime=2)`
- `solutions/PDR/net.py`
  - `check_scrap(t, firetime, marks) -> bool`
  - `get_leaf_node(m, marks, clock) -> None`
  - `search(m, marks, time, mode=0, depth=5) -> bool`

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
