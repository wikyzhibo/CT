## 1. 输入和输出

#### 输入

1. 节点（operation）特征矩阵 `fea` （n_node, n_feature）
   - 特征1：当前完工时间下界 `LB`（按照当前已经知道的情况，这个工序**最早**什么时候能结束）
   - 特征2：是否已经完成 `finished_mark`
2. 弧（job中operation的先后关系，使用同一机器的操作先后关系）邻接矩阵 `adj`（n_node, n_node）
   - adj[i,j]=1，一条从node_i指向node_j的弧

3. 当前候选集合 `candidate`
   - `candidate[i]` 的含义是选择job_i “当前的operation”

4. 掩码 `mask`
   - 某个 job 如果已经没有后续工序了，就把它对应的位置屏蔽掉




#### 输出

根据mask，选择执行某个job"当前的operation"

---

## 2. 内部处理

#### 第一步

- 每个节点不只看自己
- 还按邻接矩阵去收集和自己相连节点的信息

所以这一步以后，每个节点的表示里就不再只有“我自己”，还混入了“邻居”的信息。

$$
\text{adj} \cdot h =
\begin{bmatrix}
1 & 1 & 0 & 0 \\
0 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
10 \\
20 \\
30 \\
40
\end{bmatrix}
=
\begin{bmatrix}
10 + 20 \\
20 + 30 \\
30 + 40 \\
40
\end{bmatrix}
=
\begin{bmatrix}
30 \\
50 \\
70 \\
40
\end{bmatrix}
$$


#### 第二步

`torch.mm(adj, h)` 之后，每个节点的特征已经混入了邻居信息。然后这个矩阵会送入 MLP，可以把 MLP 的作用理解成：

- 把“聚合过邻居信息的节点特征”**再做一次抽象和加工**



## 3. 有什么用

