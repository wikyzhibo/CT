# RL



本阶段完成了**彩色 Petri 网**形式的产线调度建模，并配套**可视化界面**，支撑对调度过程的可视分析与演示。已在 **A/B/C/D 四条路径（集创赛）**以及 **1-1～1-5 （前期项目）五条加工路线**上完成训练与结果验证。通过**模型与训练流程优化**，强化学习训练耗时**大幅缩短**：在 A–D 等路径上单次训练已降至约 **7～18 s**；复杂 **1-*** 类路线目前多在约 **34～53 s**。整体训练效率已**接近启发式基线**；

---

## 一、本阶段完成工作

1. **颜色 Petri 网建模**  
   建立与产线调度问题一致的 Petri 网模型，用颜色实现晶圆路由，支撑多路线、多约束下的离散事件仿真与策略训练。

2. **可视化界面**  
   提供图形化界面，用于加载模型、回放与分析调度序列，便于排查路线逻辑、展示实验现象与阶段性结论。

3. **场景覆盖**  
   - **路径 A～D**：无清洁与带清洁（Clean=Y）两类设定下均已开展实验（带清洁 A 路径仍有问题，见第三节）。  
   - **加工路线 1-1～1-5**：已跑通并完成表中记录的训练与 Makespan 等指标；**1-6** （有死锁问题）仍在修改。

4. **模型与训练加速**  
   通过算法与实现侧优化，显著压缩单次训练时间。简单路径（A–D）上训练时间与组内启发式算法**处于同一量级**；**1-*** 路线因结构更复杂，单次训练仍普遍需要四十秒，已列为后续加速重点（见第四节）。

---

## 二、关键实验数据

**表 1（路径 A～D）**  
说明：`Makespan(75)` 为表中记录的完工跨度指标；`Training wafers` 为训练时所采用的晶圆数量；`Training time` 为单次训练耗时。

| Route | Clean | ARM | Training wafers | Makespan(75) | Training time (s) |
| :---: | :---: | :-------------: | :----------: | :---------------: | ----- |
| A | N | S | 6 | 38535 | 14.9 |
| B | N | S | 8 | 12725 | 9.2 |
| C | N | S | 8 | 12690 | 7.0 |
| D | N | S | 8 | 9180 | 11.8 |
| A | Y | S | / | / | / |
| B | Y | S | 12 | 13325 | 15.9 |
| C | Y | S | 20 | 15828 | 17.2 |
| D | Y | S | 15 | 9795 | 11.6 |
| A | N | D | 6 | 38535 | / |
| B | N | D | 13 | 11985 | / |
| C | N | D | 13 | 12315 | / |
| D | N | D | 13 | 8565 | / |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

表 1 备注：

- 当前实验**未**考虑「闲置清洁」「切换工艺类型清洁」。  
- **带清洁的 A 路径**场景仍存在问题，表中用 `/` 表示暂无有效结果。  

---

**表 2（加工路线 1-x）**  
说明：`Makespan(13片/秒)` 为表中记录的完工跨度指标；`Batch` 为批设置；`Training time` 为单次训练耗时。

| Route | Makespan(13片/秒) | Batch | Training time (s) |
| :---: | :---------------: | :---: | :---------------: |
| 1-1 | 1865 | 80 | 42.1 |
| 1-2 | 2959 | 80 | 42.1 |
| 1-3 | 1493 | 120 | 53.1 |
| 1-4 | 1705 | 80 | 40.3 |
| 1-5 | 1577 | 80 | 33.8 |
| 1-6 | — | — | — |

表 2 备注：

- **1-6**：路线已规划，Makespan / Batch / 训练时间等指标**待补**。  
- **1-*** 类路线结构更复杂，单次训练多在约 **40～60 s** 区间，后续以训练加速与结构优化为重点。

---

## 三、已知问题与备注

1. 清洁相关：实验设定**不包含**「闲置清洁」「切换工艺类型清洁」。  
2. **带清洁的 A 路径**结果异常或未收敛，需单独排查。  
3. 部分路线速记中出现的逻辑/节拍问题（见附录），需在模型与规则层面对照验证，例如：  
   - PM3 有晶圆时仍从 PM1 取片，导致 PM3 无法正常出片。  
   - LLC 节拍与预期不一致。  
   - LLD 有晶圆时仍从 PM6 取片，导致 LLD 侧阻塞。  

---

## 四、后续工作计划

- **启发式算法优化**（组内协同任务）。  
- **双腔建模**：单腔室同时处理两片晶圆的模型扩展，以及**死锁预防与控制**策略。  
- **策略网络**：在现有 MLP 基础上，探索 **GNN** 等结构以更好利用图/拓扑信息。  
- **训练加速**：针对 **1-*** 等复杂路线，将单次训练从当前普遍的 **40～60 s** 进一步压低。  
- **物理约束**：在模型与奖励/约束中显式加入更贴近机台的物理与工艺限制。

---

## 附录：路线速记与示意图

![示意图](assets/image-20260319160731282.png)

##### **PM7/8(78s)->LLC->PM3/4(110s)[88s/1片]->LLD(75s)->PM9/10(80s)**

![1-1](assets\1-1-1774451173907-1.png)

![1-1-30](assets\1-1-30.png)

------

##### PM7/8(99s)->PM9/10(73s)->LLC(110s)->PM2(80s)[420s/20片]->PM1(128s)->PM3(126s)->LLD(75s)->LP_done

<img src="C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\training_metrics_plot12.png" alt="training_metrics_plot12" style="zoom:25%;" />

![1-2](C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\1-2.png)

------

##### PM7/8(87s)->LLC(41s)->PM1/6(60s)->PM2/5(105s)->LLD(39s)->PM9/10(54.6s)->LP_done

![training_metrics_plot13](assets\training_metrics_plot13-1774451320661-4.png)

![1-3-20](assets\1-3-20.png)

------

##### PM7/8(50s)->LLC(34s)->PM1/6(53s)[304s/50片]->PM2/3/5(107s)->LLD(34s)->LP_done

<img src="C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\training_metrics_plot14.png" alt="training_metrics_plot14" style="zoom:25%;" />

![1-4](C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\1-4.png)

------

**PM7/8(50s)->LLC/LLD(37s)->PM9/10(56s)[352s/25片]->LLB**

![training_metrics_plot15](assets\training_metrics_plot15-1774451386628-7.png)

![1-5_30](assets\1-5_30.png)

------

**PM7/8(101s)->LLC(65s)->PM1(41s)[420s/20片]->PM3(93s)**

**->LLD(65s)->PM6(55s)[335s/25片]->LLD(65s)->LLB**

1-6 暂无，单臂下存在死锁

------

##### PM7/8(70s)->LLC(0s)->PM1/2/3/4(600s)->LLD(70s)->PM9/10(200s)

<img src="C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\training_metrics_plot21.png" alt="training_metrics_plot21" style="zoom:25%;" />

![2-1](C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\2-1.png)

------

**PM7/8(70s)->LLC(0s)->PM1/2(300s)->LLD(70s)**

<img src="C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\training_metrics_plot22.png" alt="training_metrics_plot22" style="zoom:25%;" />

![2-2](C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\2-2.png)

------

**PM7/8(70s)->PM9/10(200s)**

<img src="C:\Users\khand\OneDrive\code\dqn\CT\plan\assets\training_metrics_plot24.png" alt="training_metrics_plot24" style="zoom:25%;" />

![2-4](assets\2-4.png)

