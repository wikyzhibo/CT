# PPO 双模式执行计划（xxx_plan.md）

## 1. 目标概述

本计划用于在 `run_ppo.py` 中设计并落地两种运行模式：

- **激进模式（Aggressive）**
  - 目标：尽可能减小 makespan
  - 特点：允许少量软约束代价，更强探索
- **保守模式（Conservative）**
  - 目标：不违规（或极低违规）
  - 特点：严格动作可行性，策略更新更稳定

核心思想：  
**用动作可行性（mask / safety layer）保证“不会违规”，  
用 reward / cost / 预算控制“有多激进”。**

---

## 2. 约束建模规范

### 2.1 硬约束（Hard Constraints）

定义：任何情况下都不能违反。

示例：
- 机器容量约束（同一时间只能加工一个任务）
- 工序 precedence（前序未完成不能执行后序）
- 非法动作（越界 id、不可执行操作）

**执行策略（必须）：**
- 在 `env` 或 `policy` 层实现 **action mask**
- 对不可行动作直接屏蔽（logit = -inf）
- 不使用 reward penalty 学硬约束

---

### 2.2 软约束（Soft Constraints）

定义：允许违反，但需要付出代价。

示例：
- 迟交（tardiness）
- 超出安全裕度
- 额外切换/能耗/负载波动

**统一抽象：**
```text
c_t = violation_amount
C = Σ c_t   (episode cost)
```

---

## 3. Reward 设计（Makespan 可学习）

### 3.1 基础目标

- 最小化 makespan

### 3.2 推荐 shaping（至少选一种）

- **时间推进惩罚**
  ```text
  r_t += -Δtime
  ```

- **makespan 增量惩罚**
  ```text
  r_t += -(MS_after - MS_before)
  ```

- **终局奖励**
  ```text
  r_terminal = -makespan
  ```

### 3.3 组合奖励形式

```text
r_t = r_time
    - w_idle * idle_time
    - w_switch * setup_change
    - α * c_t
```

---

## 4. 两种模式的核心差异

### 4.1 激进模式（Aggressive）

- 硬约束：始终 mask
- 软约束：低权重惩罚
- 探索：强
- 策略目标：更小 makespan

### 4.2 保守模式（Conservative）

- 硬约束：mask + 推理兜底
- 软约束：高权重惩罚或 0 预算
- 探索：弱
- 策略目标：0 或极低违规

---

## 5. PPO + 拉格朗日（可选但推荐）

### 5.1 优化目标

```text
maximize   E[R]
subject to E[C] ≤ d
```

拉格朗日形式：
```text
L = E[R - λ(C - d)]
```

更新规则：
```text
λ ← max(0, λ + η(E[C] - d))
```

### 5.2 模式对应预算

| 模式 | 约束预算 d |
|----|-----------|
| 激进 | 小正数（允许少量软违规） |
| 保守 | 0 或接近 0 |

---

## 6. PPO 超参数建议

### 6.1 激进模式

- `ent_coef`: 高
- `clip_range`: 较大
- `learning_rate`: 略高或更快衰减
- `target_kl`: 宽松
- 推理：采样 / 高温度

### 6.2 保守模式

- `ent_coef`: 0 或很小
- `clip_range`: 小
- `learning_rate`: 略低
- 推理：argmax / 低温度
- 必须启用 fallback

---

## 7. Mode Config 工程实现

```python
def get_mode_cfg(mode):
    if mode == "aggressive":
        return dict(
            violation_weight=1.0,
            ent_coef=0.02,
            clip_range=0.3,
            temperature=1.2,
            hard_mask=True,
        )
    elif mode == "conservative":
        return dict(
            violation_weight=50.0,
            ent_coef=0.0,
            clip_range=0.1,
            temperature=0.7,
            hard_mask=True,
        )
```

**使用位置：**
1. Env reward 计算  
2. Policy action sampling  
3. Eval / checkpoint 筛选

---

## 8. 推理安全兜底（强烈建议）

推理阶段执行顺序：
1. 应用 hard action mask
2. 若动作仍不可行：
   - 从 top-k 中选择可行动作
   - 或 fallback 到启发式规则（如 earliest-ready）

---

## 9. Checkpoint 选择标准

### 保守模式 Best
1. violation_rate == 0（或极小）
2. 在满足 1 的前提下 makespan 最小

### 激进模式 Best
1. makespan 最小
2. violation_rate ≤ 可接受阈值（可选）

---

## 10. 可选进阶：单模型多偏好

- 将 `λ` 或违规权重作为 observation 输入
- 训练时随机采样 `λ`
- 推理时：
  - 激进：小 `λ`
  - 保守：大 `λ`

---

## 11. 执行检查清单（Checklist）

- [ ] 明确硬约束与软约束
- [ ] 硬约束全部用 action mask
- [ ] makespan 有过程级 reward shaping
- [ ] 软约束统一成 cost
- [ ] 两种 mode 的 config 可切换
- [ ] 推理阶段有安全兜底
- [ ] Checkpoint 选择标准区分模式

---

**本文件作为 `run_ppo.py` 双模式实现的执行蓝图，可直接按章节逐条落地。**
