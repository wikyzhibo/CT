import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import sys
import time
from solutions.model.pn_models import Place, WaferToken, BasedToken
from construct_net.run_supernet import load_petri_split
from solutions.v2.utils import (Interval,_first_free_time_at,_insert_interval_sorted,
                                Message,QueueItem)
from solutions.v2.net_v2 import ActionInfo
from visualization.plot import plot_gantt_hatched_residence,Op


INF_OCC = 10**18

def get_pre_pst(net_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """把 -1 / +1 网络表转为 Pre / Pst 矩阵（P, T）。"""
    pre = (net_df == -1).to_numpy(dtype=np.int64)
    pst = (net_df ==  1).to_numpy(dtype=np.int64)
    return pre, pst

from dataclasses import dataclass

DEFAULT_STAGE2ACT = {
    1: ("t3", "u3", "u31"),
    2: ("t4", "u4"),
    3: ("t5", "u5"),
    4: ("t6", "u6"),
    5: ("t7", "u7")
}

INF = 10**6

class PetriV3:
    """
    input:


    - self.m         : 当前标识 (P,)
    - self.m0        : 初始标识 (P,)
    - self.md        : 终止标识 (P,)
    - self.marks[p]  : 库所 p 的 tokens，Place 对象
    - self.time      : 当前离散时间（0基）
    - self.ttime     : 变迁执行时长（标量或 (T,) 向量）
    - self.k         : 库所容量（(P,) 向量；如无限容量可给很大值）

    - self.visited        : search过程中访问的标识
    - self.makespan = 0   : 完工时间
    - self.transitions    : search过程中发射的变迁序列
    - self.m_record       : search过程中标识序列
    - self.marks_record   : search过程中token记录序列
    - self.mark_bad       : search过程中遇到的坏标识
    - self.expand_mark    : 访问标识数 int
    """
    def __init__(self, with_controller=False,**kwargs) -> None:

        super_info = load_petri_split("petri_N7.npz")
        self.pre = super_info['pre']
        self.pst = super_info['pst']
        self.net = self.pst - self.pre
        self.m0 = super_info['m0']
        self.m = super_info['m0']
        self.k = super_info['capacity']
        self.ptime = super_info['ptime']
        self.ttime = 2
        self.id2p_name = super_info['id2p_name']
        self.id2t_name = super_info['id2t_name']
        self.idle_idx = super_info['idle_idx']
        self.marks: List[Place] = super_info['marks']
        self.marks_copy = self._clone_marks(self.marks)
        self._init_path()
        self.md = super_info['md']
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.n_wafer = super_info['n_wafer']

        self.pointer = {1:0,2:0,3:0,4:0,5:0}
        self.stage_act = {1:[3,5],2:[],3:[9,11,13,15],4:[],5:[17,19]}

        self.log = []

        # search 函数服务变量
        self.makespan = 0
        self.transitions = []
        self.m_record = []
        self.marks_record = []
        self.time_record = []
        self.mask_record = []
        self.visited = []
        self.expand_mark = 0
        self.back_time = 0
        self.time = 1
        self.over_time = 0
        self.qtime_violation = 0
        self.shot = "s"
        self.dead_mark = []
        self.bad_mark = []

        self.transition_times = [[] for _ in range(self.T)]
        self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]

        self.ops = []
        self.open_occ = {}  # key: (place_id, tok_key) -> Interval
        self._init_open_occ_from_marks()
        self._cache_action_info = {}

        self.stage_c = {1: 2, 2: 1, 3: 4, 4: 1, 5: 2}
        self.proc = {1: 70, 2: 0, 3: 600, 4: 70, 5: 200}

    def _init_path(self):
        '''
        单独腔体

        pathD = [[['u_LP2_AL','t_AL']],
                 [['u_AL_LLA', 't_LLA']],
                 [['u_LLA_PM7', 't_PM7','u_PM7_PM9','t_PM9','u_PM9_LLB','t_LLB'], ['u_LLA_PM8', 't_PM8','u_PM8_PM10','t_PM10','u_PM10_LLB','t_LLB']],
                 [['u_LLB_LP_done','t_LP_done']]]
        pathC = [[['u_LP1_AL', 't_AL']],
                 [['u_AL_LLA', 't_LLA']],
                 [['u_LLA_PM7', 't_PM7','u_PM7_LLC', 't_LLC'], ['u_LLA_PM8', 't_PM8','u_PM8_LLC', 't_LLC']],
                 [['u_LLC_PM1','t_PM1','u_PM1_LLD', 't_LLD'],['u_LLC_PM2','t_PM2','u_PM2_LLD', 't_LLD'],['u_LLC_PM3','t_PM3','u_PM3_LLD', 't_LLD'],['u_LLC_PM4','t_PM4','u_PM4_LLD', 't_LLD']],
                 [['u_LLD_PM9','t_PM9','u_PM9_LLB', 't_LLB'],['u_LLD_PM10','t_PM10','u_PM10_LLB', 't_LLB']],
                 [['u_LLB_LP_done', 't_LP_done']]]

        '''

        # 并行腔体
        pathD = [[['u_LP2_AL', 't_AL']],
                 [['u_AL_LLA', 't_LLA']],
                 [['u_LLA_PM7', 't_PM7', 'u_PM7_PM9', 't_PM9', 'u_PM9_LLB', 't_LLB']],
                 [['u_LLB_LP_done', 't_LP_done']]]
        pathC = [[['u_LP1_AL', 't_AL']],
                 [['u_AL_LLA', 't_LLA']],
                 [['u_LLA_PM7', 't_PM7', 'u_PM7_LLC', 't_LLC']],
                 [['u_LLC_PM1', 't_PM1', 'u_PM1_LLD', 't_LLD']],
                 [['u_LLD_PM9', 't_PM9', 'u_PM9_LLB', 't_LLB']],
                 [['u_LLB_LP_done', 't_LP_done']]]
        
        pathC_idx = []
        for stage in pathC:
            stage_idx = []
            for branch in stage:
                branch_idx = [self.id2t_name.index(name) for name in branch]
                stage_idx.append(branch_idx)
            pathC_idx.append(stage_idx)
        
        pathD_idx = []
        for stage in pathD:
            stage_idx = []
            for branch in stage:
                branch_idx = [self.id2t_name.index(name) for name in branch]
                stage_idx.append(branch_idx)
            pathD_idx.append(stage_idx)
        
        lp1_idx = self.idle_idx['start'][0]
        lp2_idx = self.idle_idx['start'][1]
        
        for token in self.marks[lp1_idx].tokens:
            token.path = [stage.copy() for stage in pathC_idx]
        
        for token in self.marks[lp2_idx].tokens:
            token.path = [stage.copy() for stage in pathD_idx]

    def reset(self):
        self.visited = []
        self.transitions = []
        self.time_record = []
        self.m_record = []
        self.marks_record = []
        self.back_time = 0

        self.time = 1
        self.m = self.m0.copy()
        self.marks = self._clone_marks(self.marks_copy)
        self._init_path()


        self.transition_times = [[] for _ in range(self.T)]
        self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]

        self.ops = []
        self.open_occ = {}  # key: (place_id, tok_key) -> Interval
        self._init_open_occ_from_marks()
        self._cache_action_info = {}

        tran_queue = self.get_enable_t(self.m, self.marks)
        mask = np.zeros(self.T, dtype=bool)
        for t in tran_queue:
            mask[t[0]] = True
            self._cache_action_info[t[0]] = ActionInfo(t=t[0], enable_time=t[1], t_name=t[2], chain=t[3])
        return self.m, mask


    def _resource_enable(self, m):
        """资源允许的使能变迁"""
        cond_pre = (self.pre <= m[:, None]).all(axis=0)
        if self.k is not None:
            cond_cap = ((m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)
            mask = cond_pre & cond_cap
        else:
            mask = cond_pre

        enable_t = np.nonzero(mask)[0]
        return enable_t

    def _color_enable(self, enable_t, mark):
        """根据wafer自带的路径为wafer分流"""
        cc = []
        for t in enable_t:
            pre_places = np.nonzero(self.pre[:, t] > 0)[0]
            for p in pre_places:
                for tok in mark[p].tokens:
                    if isinstance(tok, WaferToken):
                        for branch in tok.path[0]:
                            if t in branch:
                                cc.append((t, branch))
                                break
                        break
        return cc

    def _earliest_place_entry(self, p: int, t_enter: int) -> Tuple[int, int]:
        """
        在 place p 的多资源中，找最早能在 t_enter 时进入的资源与时刻
        返回 (best_r, best_time)
        """
        best_r, best_t = 0, INF_OCC
        for r in range(len(self.place_times[p])):
            t0 = _first_free_time_at(self.place_times[p][r], t_enter, t_enter + self.ptime[p])
            if t0 < best_t:
                best_t, best_r = t0, r
        return best_r, best_t

    def _init_open_occ_from_marks(self):
        """
        初始化：对初始 marks 里已经在资源库所(type<=3)的 token，开开放区间 [enter_time, INF_OCC)
        否则首次释放时 open_occ 找不到，会导致区间永远不收口。
        """

        for p in range(self.P):
            if self.marks[p].type > 3:
                continue
            # Place.tokens 是 deque
            for tok in list(self.marks[p].tokens):
                tok_key = tok.job_id
                t_enter = int(tok.enter_time)

                r, t0 = self._earliest_place_entry(p, t_enter)
                if t0 > t_enter:
                    # 初始 token 被迫等待资源（通常不该发生），这里直接把它推到可进入时刻
                    t_enter = t0

                itv = Interval(start=t_enter, end=INF_OCC, tok_key=tok_key)
                _insert_interval_sorted(self.place_times[p][r], itv)
                self.open_occ[(p, tok_key)] = itv

    def _dry_run_chain(self, chain_names, m, marks, max_retry=50):
        """
        Dry-run（无副作用）检查“强制连续链”是否可行，并计算链中每一步的最早发射时刻。
        额外能力：若发现链内等待过长，则整体延后 chain 起点并重跑，直到全部满足。
        """
        shift0 = 0  # 对 chain_names[0] 的整体延后量
        t1 = self.id2t_name.index(chain_names[0])
        base_start = self._earliest_enable_time(t1, m, marks)

        for _ in range(max_retry):
            tmp_m = m.copy()
            tmp_marks = self._clone_marks(marks)
            tmp_time = base_start + shift0

            times = []
            violated = False
            need_shift = 0

            for i, name in enumerate(chain_names):
                t_id = self.id2t_name.index(name)

                se = self._resource_enable(tmp_m)
                se_chain = self._color_enable(se, tmp_marks)
                if t_id not in se:
                    return False, [], -1, None, None

                # 1) 计算该步最早可发射时间
                if i == 0:
                    te0 = self._earliest_enable_time(t_id, tmp_m, tmp_marks, tmp_time)
                else:
                    te0 = self._earliest_enable_time(t_id, tmp_m, tmp_marks, tmp_time)

                if te0 < 0:
                    return False, [], -1, None, None

                # 强制连续：不得早于当前临时时间
                te = max(int(te0), int(tmp_time))

                pre = np.nonzero(self.pre[:, t_id])[0]
                pid = -1
                for p in pre:
                    if tmp_marks[p].type <= 3:
                        pid = p
                        break

                # 2) 检查等待过长：若违规，计算需要整体右移多少，并跳出重跑
                if i > 0 and pid != -1:
                    last_time = times[i - 1]
                    gap = te - last_time - self.ptime[pid]
                    if gap > 5:
                        # 你原本的 delta 规则（保留）
                        delta = gap - 3
                        need_shift = max(need_shift, int(delta))
                        violated = True
                        break

                times.append(int(te))

                tmp_m, tmp_marks, tmp_time = self._tpn_fire(
                    t_id, tmp_m, tmp_marks, start_from=int(te), with_effect=False
                )

            if not violated:
                return True, times, tmp_time, tmp_m, tmp_marks

            # 发现违规：整体延后 chain 起点，然后从头重跑
            shift0 += need_shift

        # 超出重试次数仍不满足
        return False, [], -1, None, None

    def _fire_chain(self, chain_names, m, marks, start_from):
        """
        真正连续发射链：依次_fire，每一步都用上一步结束后的时间推进
        返回 info（同 _search_fire 风格）
        """
        cur_m, cur_marks, cur_time = m, marks, int(start_from)

        # 删除wafer已经发射的链
        t0 = chain_names[0]
        t0_id = self.id2t_name.index(t0)
        t0_pre = np.nonzero(self.pre[:, t0_id])[0]
        for p in t0_pre:
            if marks[p].type > 3:
                continue
            if isinstance(marks[p].head(), WaferToken):
                tok = marks[p].head()
                tok.path.pop(0)

        for i, name in enumerate(chain_names):
            t_id = self.id2t_name.index(name)

            if i == 0:
                te = start_from
            else:
                te = self._earliest_enable_time(t_id, cur_m, cur_marks)

            info = self._search_fire(t_id, cur_m, cur_marks, te)
            cur_m, cur_marks, cur_time = info["m"], info["marks"], info["time"]

        # 链执行完后返回当前状态
        se = self._resource_enable(cur_m)
        se = self._color_enable(se, cur_marks)
        finish = bool((cur_m == self.md).all())
        deadlock = (not finish) and (len(se) == 0)
        return {"m": cur_m, "marks": cur_marks, "mask": se, "finish": finish, "deadlock": deadlock, "time": cur_time}

    # ---------- 计算最早可使能时间 ----------
    def _earliest_enable_time(self, t: int, m, marks, start_from=None) -> int:
        d = int(self.ttime)
        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 由前置 token 的 enter_time 决定最早时刻下界
        t1 = 0 if start_from is None else int(start_from)
        for p in pre_places:
            place = marks[p]
            if place.type > 3:
                continue
            tok_enter = int(place.head().enter_time) + int(self.ptime[p])
            t1 = max(t1, tok_enter)

        candidate_time = t1

        # 迭代收敛：保证所有 pst 的 token_enter = te+d 时刻都能进入（允许插空）
        max_iterations = 20
        for _ in range(max_iterations):
            updated = False
            for p in pst_places:
                if marks[p].type > 3:
                    continue
                token_enter = candidate_time + d  # enter_new
                _, t0 = self._earliest_place_entry(p, token_enter)
                if t0 > token_enter:
                    candidate_time = t0 - d
                    updated = True
            if not updated:
                break

        # 再检查变迁自身执行区间冲突（你原来的逻辑保留）
        # final_time = self._from_interval_find_time(candidate_time, d, self.transition_times[t])

        t_name = self.id2t_name[t]
        # if t_name == 'u0' and len(self.u0_record) > 0:
        # candidate_time = max(candidate_time, self.u0_record[-1]+153)

        return int(candidate_time)

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]

    def _tpn_fire(
            self,
            t: int,
            m: np.ndarray,
            marks: Optional[List[Place]] = None,
            start_from: Optional[int] = None,
            with_effect=True,
            **kwargs
    ) -> Tuple[np.ndarray, List[Place], int]:

        t_name = self.id2t_name[t]

        new_m = m.copy()
        time = self.time if start_from is None else int(start_from)
        new_marks = self._clone_marks(marks)

        te = int(time)
        d = int(self.ttime)
        tf = te + d - 1
        enter_new = tf + 1

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        consumed = []  # [(p, tok)]
        moved_tok = None  # 主 token

        # 1) 消费 pre
        for p in pre_places:
            tokens = new_marks[p]
            if tokens.type in [1, 2, 3]:
                tok = tokens.pop_head()
                consumed.append((p, tok))
                if moved_tok is None:
                    moved_tok = tok
            else:
                _ = tokens.pop_head()
            new_m[p] -= 1

        # 记录变迁执行时间段
        if t < len(self.transition_times) and with_effect:
            self.transition_times[t].append((te, te + d))

        # 2) 生成 pst + “进入即开开放占用区间”
        for p in pst_places:
            pst_place_type = new_marks[p].type

            if pst_place_type in [1, 2, 3]:
                if moved_tok is None:
                    raise RuntimeError(f"t={self.id2t_name[t]} needs moved_tok but none from pre_places")

                moved_tok.enter_time = enter_new
                new_marks[p].append(moved_tok)

                tok_key = moved_tok.job_id
                r, t0 = self._earliest_place_entry(p, enter_new)

                # _earliest_enable_time 已保证 t0 <= enter_new；这里再防御一下
                if t0 > enter_new:
                    raise RuntimeError(f"Place p{p} no resource at {enter_new}, got {t0}")

                if with_effect:
                    itv = Interval(start=enter_new, end=INF_OCC, tok_key=tok_key)
                    _insert_interval_sorted(self.place_times[p][r], itv)
                    self.open_occ[(p, tok_key)] = itv

            else:
                new_tok = BasedToken(enter_time=enter_new)
                new_marks[p].append(new_tok)

            new_m[p] += 1

        # 3) 动作结束（enter_new）才释放 pre_place：收口区间
        if with_effect:
            for p, tok in consumed:
                tok_key = tok.job_id
                itv = self.open_occ.pop((p, tok_key), None)
                if itv is not None:
                    itv.end = enter_new

            if self.time < enter_new:
                self.time = enter_new

        self.snapshot(new_m, True)
        return new_m, new_marks, enter_new

    def snapshot(self, mark: Optional[np.ndarray[int]] = None, partial_disp: bool = False) -> None:
        """
        transfer the ndarray mark to human look

        :param mark:
        :param partial_disp: if true, display the d-place or p-place
        """
        if mark is None:
            mark = self.m
        s = ''

        disp = ['L','d','P','p','A']

        for i in range(self.P):
            if mark[i] != 0:
                name = self.id2p_name[i]
                if partial_disp:
                    if name[0] in disp:
                        if mark[i] == 1:
                            s += f'{self.id2p_name[i]}\t'
                        else:
                            s += f'{self.id2p_name[i]}*{mark[i]}\t'
                else:
                    s += f'{self.id2p_name[i]}*{mark[i]}\t'
        self.shot = s

    def _get_t_info(self, t, mark: List[Place]):
        pre_place = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_place:
            if mark[p].type <= 3:
                job_id = mark[p].head().job_id
                enter_time = mark[p].head().enter_time
                proc_time = mark[p].processing_time
                return job_id, enter_time, proc_time
        return -1

    def _search_fire(self, t: int, m, marks, start_from: int):
        """
        Fire transition for search algorithm (no PPO-specific logic).
        Returns info dict with: m, marks, mask, finish, deadlock, time
        """
        # Fire the transition using core Petri net logic

        # 记录发射信息
        t_name = self.id2t_name[t]

        job_id, enter_time, proc_time = self._get_t_info(t, marks)

        def which_stage(t_name):
            for i, s in DEFAULT_STAGE2ACT.items():
                if t_name in s:
                    return i
            return -1

        i = which_stage(t_name)

        start = enter_time
        proc_end = start + proc_time
        end = start_from
        stage, machine = -1, 0
        if i != -1:
            if t_name == DEFAULT_STAGE2ACT[i][1] or t_name == "u31":  # 卸载
                stage = i
                machine = self.machine_pr[i] % self.stage_c[i]
                self.machine_pr[i] += 1

        if start is None:
            print('pause')
        self.ops.append(Op(job=job_id, stage=stage, machine=machine,
                           start=enter_time, proc_end=proc_end, end=end))

        data = {'start_time': start,
                'machine': machine}

        new_m, new_marks, time = self._tpn_fire(t, m, marks,
                                                start_from=start_from,
                                                with_effect=True, **data)

        # Compute mask, finish, and deadlock status
        # se = self._resource_enable(new_m)
        # se = self._color_enable(se, new_marks)
        finish = bool((new_m == self.md).all())
        # deadlock = (not finish) and (len(se)==0)

        # Return info dict without PPO-specific fields like overtime
        info = {
            "m": new_m,
            "marks": new_marks,
            # "mask": se,
            "finish": finish,
            # "deadlock": deadlock,
            "time": time
        }
        return info

    def get_enable_t(self, m, mark):
        se = self._resource_enable(m)
        se_chain = self._color_enable(se, mark)
        names = [self.id2t_name[t] for t in se]
        transition_queue = []
        for t, chain in se_chain:
            name = self.id2t_name[t]
            chain = [self.id2t_name[x] for x in chain]
            ok, times, end_time, _, _ = self._dry_run_chain(chain_names=chain, m=m, marks=mark)
            if not ok:
                continue
            key_time = times[0]
            transition_queue.append((t, key_time, name, chain))
        return transition_queue

    # ---------- phase 2: execute selected action ----------
    def step(self, action: int, record=False):

        ainfo = self._cache_action_info[action]
        t, enable_time, t_name, chain = ainfo.t, ainfo.enable_time, ainfo.t_name, ainfo.chain

        time_violation_type = self._if_violation(t, enable_time, self.marks)

        qtime_violation = 0
        over_time = 0
        if time_violation_type == 2:
            qtime_violation = 1
        elif time_violation_type == 1:
            over_time = 1

        info = self._fire_chain(chain, self.m, self.marks, enable_time)
        self.m = info['m']
        self.marks = info['marks']
        tran_queue = self.get_enable_t(self.m, self.marks)
        new_mask = np.zeros(self.T, dtype=bool)
        self._cache_action_info = {}
        for t in tran_queue:
            new_mask[t[0]] = True
            self._cache_action_info[t[0]] = ActionInfo(t=t[0], enable_time=t[1], t_name=t[2], chain=t[3])

        return new_mask, self.m, self.time, qtime_violation, over_time, info['finish']

    def _get_t_job_id(self,t,mark: List[Place]):
        pre_place = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_place:
            if mark[p].type == 1 or mark[p].type == 2 or mark[p].type == 3:
                return mark[p].head().job_id
        return -1

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

    def _if_violation(self,t,firetime,mark):

        """检查是否存在违反约束：运输时间约束返回2，驻留时间约束返回1，否则返回0"""
        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_places:
            place = mark[p]
            if place.name == 'p3' or place.name == 'p5':
                if firetime > place.head().enter_time + place.processing_time + 15:
                    return 1
            if place.type == 2:
                if firetime > place.head().enter_time + 30:
                    return 2
        return 0

    def search(self, m, marks, time, mode=0):

        self.expand_mark += 1
        transition_queue = self.get_enable_t(m, marks)

        self._select_mode(queue=transition_queue, mode=mode)
        while transition_queue:

            t, enable_time, t_name, chain = transition_queue.pop(0)
            job_id = self._get_t_job_id(t, marks)

            time_violation_type = self._if_violation(t, enable_time, marks)
            if time_violation_type == 2:
                self.qtime_violation += 1
            elif time_violation_type == 1:
                self.over_time += 1

            self.log.append(Message(step=len(self.log) + 1,
                                    tran_name=t_name,
                                    cur_time=int(enable_time),
                                    job=job_id,
                                    cand=[QueueItem(c, int(b)) for a, b, c, d in transition_queue]))

            info = self._fire_chain(chain, m, marks, enable_time)

            finish = info["finish"]
            new_m = info["m"]
            new_marks = info["marks"]
            time = info["time"]

            if finish:
                self.makespan = self.time
                return True

            if any(np.array_equal(new_m, mx) for mx in self.visited):
                continue

            self.visited.append(new_m)
            self.transitions.append(t)
            self.time_record.append(time)
            self.m_record.append(new_m)
            self.marks_record.append(new_marks)
            # self.mask_record.append(mask)

            if self.search(new_m, new_marks, time, mode=mode):
                return True

            self.back_time += 1
            self.transitions.pop(-1)
            self.time_record.pop(-1)
            self.m_record.pop(-1)
            self.marks_record.pop(-1)
            self.mask_record.pop(-1)
            self.bad_mark.append(new_m)

        self.expand_mark += 1
        return False










def main():
    sys.setrecursionlimit(10000)
    search_mode = 0
    start = time.time()
    net = PetriV3(with_controller=True)
    print(f'|p={net.P}|t={net.T}')

    for _ in range(2):
        net.reset()
        net.search(net.m,net.marks,0,search_mode)
        print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}|back_time={net.back_time}'
              f'|expand marks={net.expand_mark}|search mode={search_mode}|'
              f'residual_violation={net.over_time}|Q_time_violation={net.qtime_violation}')
        #print(net.log)

    out_path = "../../results/"
    ops = [op for op in net.ops if op.stage != -1]
    tmp_p = {1:['PM7', 'PM8'],
             2:['LLC'],
             3:['PM1', 'PM2', 'PM3', 'PM4'],
             4:['LLD'],
             5:['PM9', 'PM10']}

    ops2 = []
    for stage, pms in tmp_p.items():
            for j,pm in enumerate(pms):
                pm_id = net.id2p_name.index(pm)
                p_occ = net.place_times[pm_id][0]
                for oc in p_occ:
                    job = oc.tok_key
                    start = oc.start
                    end = oc.end
                    proc = net.proc[stage]
                    ops2.append(Op(job=job, stage=stage, machine=j, start=start, proc_end=start + proc, end=end))

    arm_info = {'ARM1': ["t3", "u3", "t4", "u6", "t7", "u31", 'u7', 't8'],
                'ARM2': ["u4", "t5", "u5", "t6"],
                'STAGE2ACT': {1: ("t3", ["u3", "u31"]), 2: ("t4", "u4"), 3: ("t5", "u5"), 4: ("t6", "u6"),
                              5: ('t7', 'u7')}}

    plot_gantt_hatched_residence(ops=ops2, proc_time=net.proc,
                                 capacity=net.stage_c, n_jobs=net.n_wafer,
                                 out_path=out_path, with_label=True,
                                 arm_info=arm_info, policy=1, no_arm=True)


if __name__ == '__main__':
    main()