# DeadlockSafeCollector 说明

`solutions/PPO/data_collector.py` 提供了一个轻量级的 `DeadlockSafeCollector`，用于在 PPO 训练中避免 `torchrl.collectors.SyncDataCollector` 遇到环境终止或死锁时提前截断批次。

## 基本流程
1. **初始化**：收集器接收环境 `env`、策略 `policy`、每个批次采样步数 `frames_per_batch`、总采样步数 `total_frames` 以及 `device`。迭代器状态会保存当前累积的步数 `_frames`、待续的时间步 `_pending_td`，以及环境快照栈 `_state_stack`，用于回退。【F:solutions/PPO/data_collector.py†L17-L117】
2. **重置与快照**：当没有“待续”步时，调用环境 `reset` 初始化，并在环境支持 `_snapshot` 时将初始状态压入栈，便于后续回退。【F:solutions/PPO/data_collector.py†L35-L47】
3. **采样循环**：每个时间步把上一步的状态复制给策略获得动作，再交给环境 `step`。当前步的张量字典会保存到 `steps` 列表，累计到 `frames_per_batch` 或达到 `total_frames` 后堆叠输出，保持与 `SyncDataCollector` 相似的结构供 GAE/PPO 使用。【F:solutions/PPO/data_collector.py†L62-L101】

## 死锁与终止处理
- **终止**：检测 `terminated`/`truncated` 标志时立即重置环境，并清空快照栈，保证批次继续采样直到凑够 `frames_per_batch`。【F:solutions/PPO/data_collector.py†L48-L101】
- **死锁回退**：环境在 `next_td["deadlock_type"]` 中上报死锁时，收集器会弹出当前死锁对应的快照，调用 `_backtrack` 恢复最近一个可行动作的标识。如果所有快照都无可行动作，则退回到环境重置状态继续采样。【F:solutions/PPO/data_collector.py†L82-L117】
- **快照持久化**：若环境实现 `_snapshot`/`_restore`（例如 `solutions/PPO/enviroment.py` 中的 `CT` 环境），每次 step 后都会把 Petri 网状态推入栈。这样在嵌套死锁时可以多次回退，直到找到有可行动作的标识再继续 rollout。【F:solutions/PPO/data_collector.py†L85-L114】【F:solutions/PPO/enviroment.py†L44-L66】

## 使用方式
在训练脚本中，把 `DeadlockSafeCollector` 当作可迭代对象传给采样循环即可，例如：

```python
collector = DeadlockSafeCollector(env, policy, frames_per_batch=64, total_frames=10_000, device="cuda")
for batch in collector:
    loss = algo.compute_loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

当环境发生死锁时，收集器会自动回退后继续生成批次，避免缺口或提前结束。
