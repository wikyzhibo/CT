from collections import deque
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from typing import List, Optional, Tuple
from solutions.model.pn_models import Place, BasedToken
from solutions.model.guard import ZeroBufferWindowController
from solutions.PDR.construct import build_pdr_net

def get_pre_pst(net_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """把 -1 / +1 网络表转为 Pre / Pst 矩阵（P, T）。"""
    pre = (net_df == -1).to_numpy(dtype=np.int64)
    pst = (net_df ==  1).to_numpy(dtype=np.int64)
    return pre, pst



INF = 10**6
LEAF_NODES = []
LEAF_CLOCKS = []
LEAF_PATHS = []

class Petri:
    def __init__(self, **kwargs) -> None:


        self.over_time = 0
        self.qtime_violation = 0
        self.shot = "s"

        self.ttime = int(kwargs.get("ttime", 5))

        self.time = 1

        self.log = []

        n_wafer = int(kwargs.get("n_wafer", 7))
        info = build_pdr_net(n_wafer=n_wafer)
        self.pre = np.ascontiguousarray(info["pre"], dtype=np.int32)
        self.pst = np.ascontiguousarray(info["pst"], dtype=np.int32)
        self.net = np.ascontiguousarray(self.pst - self.pre, dtype=np.int32)

        self.id2p_name = list(info["id2p_name"])
        self.id2t_name = list(info["id2t_name"])
        self.place_names = list(self.id2p_name)
        self.P = int(len(self.id2p_name))
        self.T = int(len(self.id2t_name))
        self.m0 = np.ascontiguousarray(info["m0"], dtype=np.int32)
        self.m = self.m0.copy()
        self.md = np.ascontiguousarray(info["md"], dtype=np.int32)
        self.ptime = np.ascontiguousarray(info["ptime"], dtype=np.int32)
        self.k = np.ascontiguousarray(info["capacity"], dtype=np.int32)
        self.idle_idx = dict(info["idle_idx"])
        self._init_marks = [p.clone() for p in info["marks"]]
        self.marks = [p.clone() for p in self._init_marks]
        self.n_wafer = int(info.get("n_wafer", n_wafer))

        # search 函数服务变量
        self.makespan = 0
        self.transitions = []
        self.m_record = []
        self.marks_record = []
        self.time_record = []
        self.mask_record = []

        self.expand_mark = 0
        self.search_depth = 3

    def _init_marks_from_m(self, m: np.ndarray, two_mode=0) -> List[Place]:
        """
        根据标识向量 m 初始化每个库所的 token 列表。
        返回 marks: list of arrays, each with shape=(2, n_tokens)
            row0: enter_time
            row1: token_id  (1..n_tokens)
        """
        idle_place = self.idle_idx.get('L1', self.idle_idx.get('start', 0))
        marks: List[Place] = []
        start_from = 0
        for i, cnt in enumerate(m.astype(int).tolist()):
            match self.id2p_name[i][0]:
                case 'p':
                    ddd = 1
                case 'd':
                    ddd = 2
                case 'L':
                    ddd = 3
                case _:
                    ddd = 4
            place = Place(
                name=self.id2p_name[i],
                capacity=int(self.k[i]) if self.k is not None else 0,
                processing_time=int(self.ptime[i]),
                type = ddd
            )
            if cnt > 0:
                enter = np.repeat(start_from, cnt)
                ids = np.arange(1, cnt + 1, dtype=int)
                if  i == idle_place:
                    match two_mode:
                        case 1: #先完成A，再完成B
                            type = np.repeat(2, cnt)
                            w = int(cnt / 3)
                            type[:w] = 1
                        case 2: #交替完成
                            type = np.repeat(2, cnt)
                            type[::2] = 1
                        case 3: #先完成B，再完成A
                            type = np.repeat(2, cnt)
                            w = 2 * int(cnt / 3)
                            type[w:] = 1
                        case _:
                            # 单晶圆模式
                            type = np.repeat(1, cnt) #表示为晶圆
                else:
                    type = np.repeat(-1, cnt)
                for e, id_, tp in zip(enter, ids, type):
                    _ = id_
                    _ = tp
                    place.append(BasedToken(enter_time=int(e)))
            marks.append(place)
        return marks

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]

    def reset(self):
        self.time = 0
        self.m = self.m0.copy()
        self.marks = [p.clone() for p in self._init_marks]

        self.m_record = []
        self.marks_record = []
        self.mask_record = []
        self.visited = []
        self.transitions = []
        self.time_record = []

    def mask_t(self, m: np.ndarray,marks: Optional[List[Place]] = None, with_clf=False) -> np.ndarray[bool]:
        """
        返回 (T,) 布尔掩码：在标识 m下可触发的变迁。
        规则：对所有库所 p
            1) m[p] >= pre[p, t]
            2) m[p] + pst[p, t] <= k[p]
        若没有容量限制，则忽略规则2

        :param with_clf:
        :param with_controller: if True, call _controller()
        :param m: mark of petri net
        :return: the mask for enable transition set
        """

        cond_pre = (self.pre <= m[:, None]).all(axis=0)
        if self.k is not None:
            cond_cap = ((m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)
            mask = cond_pre & cond_cap
        else:
            mask = cond_pre

        return mask

    def enabled_transitions(self) -> list[int]:
        return np.nonzero(self.mask_t(self.m))[0].tolist()

    # ---------- 计算最早可使能时间 ----------
    def _earliest_enable_time(self,
                              t: int,
                              m,
                              marks,
                              start_from: Optional[int] = None) -> int:
        """
        返回最早时刻 τ（>= start_from，默认当前 self.time），使变迁 t 可触发：
          - 每个前置库所有至少一枚 token，且它们的 enter_time <= τ；
          - 触发后不会违反容量（这里按标准库所容量检查，执行期间不额外占用库所容量；
            如你需要“执行期间也占用后置/前置库所”的模型，可在此加入时间区间检查）。
        若不可触发，返回 -1。
        """
        tau = self.time if start_from is None else int(start_from)
        d   = int(self.ttime)

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 若任何前置库所无 token，直接不可触发
        for p in pre_places:
            if len(marks[p]) == 0:
                return -1

        # 由前置 token 的 enter_time 决定最早时刻下界
        # 取各前置库所“队头”（最早进入的）token 的 enter_time
        earliest = tau
        for p in pre_places:
            tok_enter = int(marks[p].head().enter_time)   # 队头
            tok_enter += self.ptime[p]
            earliest  = max(earliest, tok_enter)

        # 检查容量（触发之后，tau+d 时刻往后每个后置库所 +1 是否超容量）
        # 这里采用离散时刻点检查：完成时刻后的即时可用（enter = tau + d）
        # 只要静态容量允许即可（若你有更严格的在制约束，可在此拓展）
        # 由于容量是静态上限，简单检查“现在的 m + Pst[:,t] <= k”即可：
        if not ((m + self.pst[:, t]) <= self.k).all():
            return -1
        return earliest

    def _tpn_fire(self,
             t: int,
             m: np.ndarray,
             marks: Optional[List[Place]] = None,
             start_from: Optional[int] = None,
             ) -> Tuple[np.ndarray, List[Place], int]:

        new_m = m.copy()
        time = self.time if start_from is None else start_from
        new_marks = self._clone_marks(marks)

        te = time

        d = int(self.ttime)
        tf = te + d - 1  # 完成时刻（含）
        enter_new = tf + 1

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 1) 消费前置库所 token（队头）
        for p in pre_places:
            tokens = new_marks[p]
            if tokens.type in [1,2,3]:
                tok = tokens.pop_head()
            else:
                _ = tokens.pop_head()
            new_m[p] -= 1

        # 2) 生成后置库所 token：enter_time = finish+1
        for p in pst_places:
            pst_place_type = new_marks[p].type
            if pst_place_type in [1,2,3] :
                tok.enter_time = enter_new
                new_marks[p].append(tok)
            else:
                new_tok = BasedToken(enter_time=enter_new)
                new_marks[p].append(new_tok)
            new_m[p] += 1

        # 3) 时间推进到完成之后（下一时刻可用）
        time = enter_new

        return new_m, new_marks, enter_new

    def _if_violation(self,t,firetime,mark):

        """检查是否存在违反约束：运输时间约束返回2，驻留时间约束返回1，否则返回0"""
        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_places:
            place = mark[p]
            if place.type == 1:
                if firetime > place.head().enter_time + place.processing_time + 15:
                    return 1
            if place.type == 2:
                if firetime > place.head().enter_time + 30:
                    return 2
        return 0

    # ---------- 一步接口（若需要返回 mask / 完成标志） ----------
    def step(self,t: int):
        """
        use for the PPO training algorithm
        """
        t_name = self.id2t_name[t]
        marks = self.marks
        m = self.m
        et = self._earliest_enable_time(t,m,marks,self.time)

        if t_name == 'u2':
            ft = self.zb.peek_time('t3',et)
        else:
            ft = self.zb.peek_time(t_name,et)
        job_id = self._get_t_job_id(t,marks)
        self.zb.commit(t_name,ft,job_id)
        start_from = int(ft)

        new_m, new_marks, time = self._tpn_fire(t,m,marks,start_from=start_from)
        mask = self.mask_t(new_m,new_marks)
        finish = bool((new_m == self.md).all())
        deadlock = (not finish) and  (not mask.any())
        time_violation_type = self._if_violation(t,time,marks)


        self.m = new_m
        self.marks = new_marks
        self.time = time

        if t_name == 't8':
            finish_a_wafer = 1
        else:
            finish_a_wafer = 0

        info = {"m": new_m,
                "marks": new_marks,
                "mask": mask,
                "finish_a_wafer": finish_a_wafer,
                "finish": finish,
                "deadlock": deadlock,
                "time": time,
                "time_violation_type": time_violation_type}
        return info

    def _search_fire(self, t: int, m, marks, start_from: int):
        """
        Fire transition for search algorithm (no PPO-specific logic).
        Returns info dict with: m, marks, mask, finish, deadlock, time
        """
        # Fire the transition using core Petri net logic
        new_m, new_marks, time = self._tpn_fire(t, m, marks, start_from=start_from)

        # Compute mask, finish, and deadlock status
        mask = self.mask_t(new_m, new_marks)
        finish = bool((new_m == self.md).all())
        deadlock = (not finish) and (not mask.any())

        # Return info dict without PPO-specific fields like overtime
        info = {
            "m": new_m,
            "marks": new_marks,
            "mask": mask,
            "finish": finish,
            "deadlock": deadlock,
            "time": time
        }
        return info

    def _get_t_job_id(self,t,mark: List[Place]):
        pre_place = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_place:
            if mark[p].type == 1 or mark[p].type == 2 or mark[p].type == 3:
                tok = mark[p].head()
                if hasattr(tok, "job_id"):
                    return tok.job_id
                if hasattr(tok, "token_id"):
                    return tok.token_id
                return -1
        return -1

    def check_scrap(self, t: int, firetime: int, marks: List[Place]) -> bool:
        """
        检查当前动作是否触发驻留时间违规（resident scrap）。
        返回 True 表示该分支需要剪枝。
        """
        return self._if_violation(t, firetime, marks) == 1

    def get_leaf_node(self, m: np.ndarray, marks: List[Place], clock: int) -> None:
        """
        记录深度叶子节点（全局收集）。
        """
        global LEAF_NODES, LEAF_CLOCKS, LEAF_PATHS
        LEAF_NODES.append({
            "m": m.copy(),
            "marks": self._clone_marks(marks),
        })
        LEAF_CLOCKS.append(int(clock))
        LEAF_PATHS.append([self.id2t_name[t] for t in self.transitions])

    def _dfs_collect_leaves(self, m, marks, clock: int, depth: int) -> None:

        # 到达第五层，收集叶子节点信息并返回
        if depth == 0:
            self.get_leaf_node(m, marks, clock)
            return

        mask = self.mask_t(m, marks)
        enabled_transitions = np.nonzero(mask)[0]
        # 死锁剪枝
        if len(enabled_transitions) == 0:
            return

        transition_queue = []
        for t in enabled_transitions:
            et = self._earliest_enable_time(t, m, marks, clock)
            transition_queue.append((int(t), int(et), self.id2t_name[t]))

        for t, enable_time, _ in transition_queue:
            if self.check_scrap(t, enable_time, marks):
                self.over_time += 1
                continue

            info = self._search_fire(t, m, marks, enable_time)
            new_m = info["m"]
            new_marks = info["marks"]
            new_clock = int(info["time"])

            self.transitions.append(t)
            self.time_record.append(new_clock)
            self._dfs_collect_leaves(new_m, new_marks, new_clock, depth=depth - 1)
            self.transitions.pop(-1)
            self.time_record.pop(-1)

    def search(self, mode=0):
        """
        深度优先搜索：
        - 输入当前 mark + clock
        - 固定深度（默认 5）
        - 遇到驻留违规即剪枝
        - 用全局变量收集深度叶子与对应 clock
        """
        global LEAF_NODES, LEAF_CLOCKS, LEAF_PATHS
        LEAF_NODES = []
        LEAF_CLOCKS = []
        LEAF_PATHS = []

        self.transitions = []
        self.time_record = []
        m = self.m.copy()
        marks = self._clone_marks(self.marks)

        self._dfs_collect_leaves(m=m, marks=marks, clock=0, depth=self.search_depth)
        return len(LEAF_NODES) > 0

    def _select_mode(self,queue,mode: int=0) :
        '''
        根据mode选择不同的Te
        0：贪婪，动作按照时间顺序从小到大
        1：剪枝，动作按照时间顺序从小到大并且只保留2个动作
        2：随机
        '''
        if mode == 0: #贪婪
            queue.sort(key=lambda x: x[1])
        elif mode == 1: #剪枝
            queue.sort(key=lambda x: x[1])
            if len(queue) > 2:
                queue = queue[:2]
        elif mode == 2:
            np.random.shuffle(queue)
        elif mode == 3:
            DEADLINE_TH = 20

            def sort_key(x):
                _, deadline, earliest,t_name,job_id = x
                is_urgent = deadline <= DEADLINE_TH
                # is_urgent: True -> 0（排前），False -> 1
                return (
                    0 if is_urgent else 1,
                    deadline if is_urgent else 0,
                    earliest,
                    t_name
                )

            queue.sort(key=sort_key)
        return queue



if __name__ == "__main__":
    petri = Petri(n_wafer=7, ttime=5)
    petri.reset()
    petri.search()