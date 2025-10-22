import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

def get_pre_pst(net_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """把 -1 / +1 网络表转为 Pre / Pst 矩阵（P, T）。"""
    pre = (net_df == -1).to_numpy(dtype=np.int64)
    pst = (net_df ==  1).to_numpy(dtype=np.int64)
    return pre, pst

def init_marks_from_m(m: np.ndarray, t0: int = 0) -> List[np.ndarray]:
    """
    根据标识向量 m 初始化每个库所的 token 列表。
    返回 marks: list of arrays, each with shape=(2, n_tokens)
        row0: enter_time
        row1: token_id  (1..n_tokens)
    """
    marks: List[np.ndarray] = []
    for cnt in m.astype(int).tolist():
        if cnt > 0:
            enter = np.repeat(t0, cnt)
            ids   = np.arange(1, cnt + 1, dtype=int)
            marks.append(np.vstack([enter, ids]))
        else:
            marks.append(np.empty((2, 0), dtype=int))
    return marks

class Petri:
    """
    - self.m         : 当前标识 (P,)
    - self.marks[p]  : 库所 p 的 tokens，shape=(2, n_tokens) -> [enter_time; id]
    - self.time      : 当前离散时间（0基）
    - self.ttime     : 变迁执行时长（标量或 (T,) 向量）
    - self.k         : 库所容量（(P,) 向量；如无限容量可给很大值）
    """
    def __init__(self, csv_path: str = "Net.csv",
                 ttime: Optional[int | np.ndarray] = 2,
                 n_t: int = 16):
        df   = pd.read_csv(csv_path)
        net  = df.iloc[:, 1:n_t+1]

        self.pre, self.pst = get_pre_pst(net)                       # (P, T)
        self.k     = df["K"].to_numpy(dtype=np.int64)           # (P,)
        self.m0    = df["M0" ].to_numpy(dtype=np.int64)             # (P,)
        self.md    = df["Md" ].to_numpy(dtype=np.int64)             # (P,)
        self.m0.setflags(write=False)
        self.ptime = df["TIME_P"].to_numpy()                        # 可选字段
        self.m     = self.m0.copy()
        self.time  = 0

        # 执行时长
        if isinstance(ttime, np.ndarray):
            self.ttime = ttime.astype(int)
        else:
            # 标量 -> 为每个变迁复制
            self.ttime = np.full(self.pre.shape[1], int(ttime), dtype=int)

        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]

        self.marks: List[np.ndarray] = init_marks_from_m(self.m, t0=self.time)

    def reset(self):
        self.time = 0
        self.m = self.m0.copy()
        self.marks = init_marks_from_m(self.m)

    # ---------- 可使能判定 ----------
    def mask_t(self, m: Optional[np.ndarray] = None) -> np.ndarray:
        """
        返回 (T,) 布尔掩码：在标识 m（默认当前 self.m）下可触发的变迁。
        规则：对所有库所 p
            1) m[p] >= pre[p, t]
            2) m[p] + pst[p, t] <= k[p]
        """
        if m is None:
            m = self.m
        cond_pre = (self.pre <= m[:, None]).all(axis=0)                      # (T,)
        cond_cap = ((m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)  # (T,)
        return cond_pre & cond_cap

    def enabled_transitions(self) -> list[int]:
        return np.nonzero(self.mask_t(self.m))[0].tolist()

    # ---------- 计算最早可使能时间 ----------
    def _earliest_enable_time(self, t: int, start_from: Optional[int] = None) -> int:
        """
        返回最早时刻 τ（>= start_from，默认当前 self.time），使变迁 t 可触发：
          - 每个前置库所有至少一枚 token，且它们的 enter_time <= τ；
          - 触发后不会违反容量（这里按标准库所容量检查，执行期间不额外占用库所容量；
            如你需要“执行期间也占用后置/前置库所”的模型，可在此加入时间区间检查）。
        若不可触发，返回 -1。
        """
        tau = self.time if start_from is None else int(start_from)
        d   = int(self.ttime[t])

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 若任何前置库所无 token，直接不可触发
        for p in pre_places:
            if self.marks[p].shape[1] == 0:
                return -1

        # 由前置 token 的 enter_time 决定最早时刻下界
        # 取各前置库所“队头”（最早进入的）token 的 enter_time
        earliest = tau
        for p in pre_places:
            tok_enter = int(self.marks[p][0, 0])   # 队头
            tok_enter += self.ptime[p]
            earliest  = max(earliest, tok_enter)

        # 检查容量（触发之后，tau+d 时刻往后每个后置库所 +1 是否超容量）
        # 这里采用离散时刻点检查：完成时刻后的即时可用（enter = tau + d）
        # 只要静态容量允许即可（若你有更严格的在制约束，可在此拓展）
        # 由于容量是静态上限，简单检查“现在的 m + Pst[:,t] <= k”即可：
        if not ((self.m + self.pst[:, t]) <= self.k).all():
            return -1

        return earliest

    # ---------- 触发变迁：更新 m、marks、time ----------
    def fire(self, t: int, start_from: Optional[int] = None) -> Tuple[int, int]:
        """
        触发变迁 t，并更新：
          - self.m：m <- m + (Pst - Pre)[:, t]
          - self.marks：消费前置库所的队头 token；在后置库所加入新 token（enter_time = te + d）
          - self.time：前进到 te + d
        返回：(te_time, finish_time)
        """
        te = self._earliest_enable_time(t, start_from=start_from)
        if te < 0:
            raise RuntimeError(f"Transition {t} cannot be enabled (time={self.time}).")

        d  = int(self.ttime[t])
        tf = te + d - 1  # 完成时刻（含）
        enter_new = tf + 1

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 1) 消费前置库所 token（队头）
        for p in pre_places:
            tokens = self.marks[p]
            if tokens.shape[1] == 0:
                raise RuntimeError(f"Place {p} has no token to consume.")
            # 弹出队头（enter_time 最早）
            self.marks[p] = tokens[:, 1:]
            self.m[p]    -= 1

        # 2) 生成后置库所 token：enter_time = finish+1
        for p in pst_places:
            next_id = (self.marks[p].shape[1] + 1) if self.marks[p].size else 1
            new_tok = np.array([[enter_new], [next_id]], dtype=int)
            if self.marks[p].size == 0:
                self.marks[p] = new_tok
            else:
                self.marks[p] = np.concatenate([self.marks[p], new_tok], axis=1)
            self.m[p] += 1

        # 3) 时间推进到完成之后（下一时刻可用）
        self.time = enter_new
        return te, tf

    # ---------- 一步接口（若需要返回 mask / 完成标志） ----------
    def step(self, t: int):
        te, tf = self.fire(t)
        mask = self.mask_t(self.m)
        #if self.m[14] == 6:
            #print("pause")
        finish = bool((self.m == self.md).all())
        deadlock = (not finish) and  (not mask.any())
        res_load = self.residual_process_time(self.m)
        info = {"te_time": te, "finish_time": tf, "time": self.time}
        return self.m.copy(), mask, finish, deadlock, res_load, info

    # ---------- 辅助：查看某库所的 tokens（enter_time, id） ----------
    def place_tokens(self, p: int) -> np.ndarray:
        """返回 shape=(2, n_tokens) 的数组：第0行为 enter_time，第1行为 id。"""
        return self.marks[p].copy()

    # ---------- 辅助：当前所有 token 的（place, enter_time, id）清单 ----------
    def tokens_snapshot(self) -> np.ndarray:
        rows = []
        for p in range(self.P):
            tok = self.marks[p]
            for j in range(tok.shape[1]):
                rows.append([p, int(tok[0, j]), int(tok[1, j])])
        return np.array(rows, dtype=int) if rows else np.empty((0, 3), dtype=int)

    def rand_action(self):
        mask = self.mask_t()
        t_set = np.nonzero(mask)[0].tolist()
        id = np.random.randint(len(t_set))
        return t_set[id]

    # --当前状态的剩余加工时间 --
    def residual_process_time(self,m):
        # n1:未加工，n2:pm1数量，n3:pm3 数量
        n1 = m[0:2].sum()
        n2 = m[2:4].sum()
        n3 = m[4:6].sum()
        # t1: pm1 加工时间，t2：pm2加工时间，t3: pm3加工时间
        # res_time = n1*(t1+t2+t3) + n2*(t2+t3) + n3*(t3)
        t1, t2, t3 = self.ptime[2:7:2]
        return n1 * (t1 + t2 + t3) + n2 * (t2 + t3) + n3 * t3




