import sys
import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from data.config.params_N7 import params_N7
from visualization.plot import plot_gantt_hatched_residence,Op
from solutions.model.based_petri_net import Petri
from solutions.model.pn_models import Place, BasedToken, WaferToken
from solutions.v2.utils import (Interval,_first_free_time_at,_insert_interval_sorted,
                                Message,QueueItem)

INF = 10**6

DEFAULT_STAGE2ACT = {
    1: ("t3", "u3", "u31"),
    2: ("t4", "u4"),
    3: ("t5", "u5"),
    4: ("t6", "u6"),
    5: ("t7", "u7")
}

INF_OCC = 10**18  # 必须足够大，防止“时间倒退”

     # 用 tok.uid

@dataclass
class ActionInfo:
    t: int
    enable_time: float
    t_name: str
    chain: List[str]  # or whatever your chain type is


class PetriNet(Petri):
    def __init__(self,with_controller=False,with_capacity_controller=False,**kwargs) -> None:
        super().__init__(with_controller,with_capacity_controller,**kwargs)

        self.machine_pr = {1:0,2:0,3:0,4:0,5:0}
        self.stage_c = {1:1,2:1,3:4,4:1,5:1}
        self.stage_start = {1:[],2:[],3:[],4:[],5:[]}
        self.proc = {1:70,2:5,3:600,4:70,5:200}


        self.over_time = 0
        self.qtime_violation = 0

        self.transition_times = [[] for _ in range(self.T)]
        self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]

        # 强制连续发射链（按你的变迁命名）
        """
        N8
        self.FORCE_CHAIN = {
            "u2": ["u2", "t3", "u3", "t4"],  # arm1 unload -> load -> unload -> load
            "u4": ["u4", "t5", "u5", "t6"],  # arm2 unload -> load -> unload -> load
            "u6": ["u6","t7"],
            "u0": ["u0","t1"],
            "u1": ["u1","t2"],
            "u7": ["u7","t8"]
        }
        """
        path1 = [['u0', 't1'], ['u1', 't2'], ['u2', 't3', 'u3', 't4'],
                 ['u4', 't5', 'u5', 't6'], ['u6', 't7', 'u7', 't8'], ['u8', 't9']]
        path2 = [['u01', 't1'], ['u1', 't2'],
                 ['u2', 't3', 'u31', 't7', 'u7', 't8'], ['u8', 't9']]

        self.ops = []
        self.open_occ = {}  # key: (place_id, tok_key) -> Interval
        self._init_open_occ_from_marks()
        self._cache_action_info = {}

    def reset(self):
        self.visited = []
        self.transitions = []
        self.time_record = []
        self.m_record = []
        self.marks_record = []
        self.back_time = 0


        self.time = 1
        self.m = self.m0.copy()
        self.marks = self._init_marks_from_m(self.m)

        self.machine_pr = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.stage_c = {1: 2, 2: 1, 3: 4, 4: 1, 5: 2}
        self.stage_start = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.proc = {1: 70, 2: 5, 3: 600, 4: 70, 5: 200}

        self.transition_times = [[] for _ in range(self.T)]
        self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]

        self.ops = []
        self.open_occ = {}  # key: (place_id, tok_key) -> Interval
        self._init_open_occ_from_marks()
        self._cache_action_info = {}

        tran_queue = self.get_enable_t(self.m,self.marks)
        mask = np.zeros(self.T, dtype=bool)
        for t in tran_queue:
            mask[t[0]] = True
            self._cache_action_info[t[0]] = ActionInfo(t=t[0], enable_time=t[1], t_name=t[2], chain=t[3])
        return self.m, mask


    def _init_marks_from_m(self, m: np.ndarray, two_mode=0) -> List[Place]:
        idle_place = self.idle_idx['start']

        job_offset = 1

        marks: List[Place] = []
        path1 = [['u0','t1'],['u1','t2'],['u2','t3','u3','t4'],
                 ['u4','t5','u5','t6'],['u6','t7','u7','t8'],['u8','t9']]
        path2 = [['u01', 't1'], ['u1', 't2'],
                 ['u2', 't3', 'u31', 't7', 'u7','t8'],['u8', 't9']]
        p1 = []
        p2 = []
        for p_t in path1:
            tmp = []
            for x in p_t:
                tmp.append(self.id2t_name.index(x))
            p1.append(tmp)
        for p_t in path2:
            tmp = []
            for x in p_t:
                tmp.append(self.id2t_name.index(x))
            p2.append(tmp)

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
                enter = np.repeat(0, cnt)
                ids = np.arange(1, cnt + 1, dtype=int)
                if  i == idle_place[0]:
                    path = p1
                elif i == idle_place[1]:
                    path = p2

                for e, id_ in zip(enter, ids):
                    if i not in idle_place:
                        place.append(BasedToken(enter_time=id_))
                    else:
                        place.append(WaferToken(enter_time=e,job_id=job_offset,path=path.copy()))
                        job_offset += 1
            marks.append(place)
        return marks

    def _resource_enable(self, m):
        """资源允许的使能变迁"""
        cond_pre = (self.pre <= m[:, None]).all(axis=0)
        if self.k is not None:
            cond_cap = ((m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)
            mask = cond_pre & cond_cap
        else:
            mask = cond_pre

        if self.with_capacity_controller:

            for p,t in self.capacity_xianzhi.items():
                p_id = self.id2p_name.index(p)
                t_id = self.id2t_name.index(t)
                if m[p_id] == 0:
                    mask[t_id] = False

        if self.with_controller:
            new_mask = self._controller(m,mask)
            mask = new_mask

        enable_t = np.nonzero(mask)[0]
        return enable_t

    def _color_enable(self,enable_t,mark):
        """根据wafer自带的路径为wafer分流"""
        cc = []
        for t in enable_t:
            pre_places = np.nonzero(self.pre[:, t] > 0)[0]
            for p in pre_places:
                for tok in mark[p].tokens:
                    if isinstance(tok, WaferToken):
                        if t in tok.path[0]:
                            cc.append((t, tok.path[0]))
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
        base_start = self._earliest_enable_time(t1,m,marks)

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
                    te0 = self._earliest_enable_time(t_id, tmp_m, tmp_marks,tmp_time)
                else:
                    te0 = self._earliest_enable_time(t_id, tmp_m, tmp_marks,tmp_time)

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
                    t_id, tmp_m, tmp_marks, start_from=int(te),with_effect=False
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

        #删除wafer已经发射的链
        t0 = chain_names[0]
        t0_id = self.id2t_name.index(t0)
        t0_pre = np.nonzero(self.pre[:, t0_id])[0]
        for p in t0_pre:
            if marks[p].type > 3:
                continue
            if isinstance(marks[p].head(),WaferToken):
                tok = marks[p].head()
                tok.path.pop(0)

        for i,name in enumerate(chain_names):
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
        #final_time = self._from_interval_find_time(candidate_time, d, self.transition_times[t])

        t_name = self.id2t_name[t]
        #if t_name == 'u0' and len(self.u0_record) > 0:
            #candidate_time = max(candidate_time, self.u0_record[-1]+153)

        return int(candidate_time)

    def _tpn_fire(
            self,
            t: int,
            m: np.ndarray,
            marks: Optional[List[Place]] = None,
            start_from: Optional[int] = None,
            with_effect = True,
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

    def _get_t_info(self,t,mark: List[Place]):
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

        job_id, enter_time, proc_time = self._get_t_info(t,marks)

        def which_stage(t_name):
            for i,s in DEFAULT_STAGE2ACT.items():
                if t_name in s:
                    return i
            return -1
        i = which_stage(t_name)

        start = enter_time
        proc_end = start + proc_time
        end = start_from
        stage,machine = -1,0
        if i != -1:
            if t_name == DEFAULT_STAGE2ACT[i][1] or t_name == "u31": #卸载
                stage = i
                machine = self.machine_pr[i] % self.stage_c[i]
                self.machine_pr[i] += 1

        if start is None:
            print('pause')
        self.ops.append(Op(job=job_id, stage=stage, machine=machine,
                           start=enter_time, proc_end=proc_end, end=end))

        data = {'start_time':start,
                'machine':machine}

        new_m, new_marks, time = self._tpn_fire(t, m, marks,
                                                start_from=start_from,
                                                with_effect=True,**data)

        # Compute mask, finish, and deadlock status
        #se = self._resource_enable(new_m)
        #se = self._color_enable(se, new_marks)
        finish = bool((new_m == self.md).all())
        #deadlock = (not finish) and (len(se)==0)



        # Return info dict without PPO-specific fields like overtime
        info = {
            "m": new_m,
            "marks": new_marks,
            #"mask": se,
            "finish": finish,
            #"deadlock": deadlock,
            "time": time
        }
        return info

    def get_enable_t(self,m,mark):
        se = self._resource_enable(m)
        se_chain = self._color_enable(se, mark)
        names = [self.id2t_name[t] for t in se]
        transition_queue = []
        for t,chain in se_chain:
            name = self.id2t_name[t]
            chain = [self.id2t_name[x] for x in chain]
            ok, times, end_time, _, _ = self._dry_run_chain(chain_names=chain, m=m, marks=mark)
            if not ok:
                continue
            key_time = times[0]
            transition_queue.append((t, key_time, name,chain))
        return transition_queue

    # ---------- phase 2: execute selected action ----------
    def step(self, action: int, record = False):

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
        new_mask = np.zeros(self.T,dtype=bool)
        self._cache_action_info = {}
        for t in tran_queue:
            new_mask[t[0]] = True
            self._cache_action_info[t[0]] = ActionInfo(t=t[0],enable_time=t[1],t_name=t[2],chain=t[3])

        return new_mask, self.m, self.time, qtime_violation, over_time, info['finish']


    def search(self, m, marks, time, mode=0):

        self.expand_mark+=1
        transition_queue = self.get_enable_t(m,marks)

        self._select_mode(queue=transition_queue,mode=mode)
        while transition_queue:

            t, enable_time,t_name,chain = transition_queue.pop(0)
            job_id = self._get_t_job_id(t, marks)

            time_violation_type = self._if_violation(t,enable_time,marks)
            if time_violation_type == 2:
                self.qtime_violation += 1
            elif time_violation_type == 1:
                self.over_time += 1

            self.log.append(Message(step=len(self.log)+1,
                                    tran_name=t_name,
                                    cur_time=int(enable_time),
                                    job = job_id,
                                    cand=[QueueItem(c,int(b)) for a,b,c,d in transition_queue]))

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
            #self.mask_record.append(mask)

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
    search_mode = 2
    start = time.time()
    #params_N7['n_wafer']=12
    net = PetriNet(with_controller=True,
                with_capacity_controller=True,
                with_zhiliu_controller = False,
                **params_N7)
    print(f'|p={net.P}|t={net.T}')

    for _ in range(1):
        net.reset()
        m = net.m0.copy()
        marks = net.marks.copy()
        net.search(m,marks,0,search_mode)
        print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}|back_time={net.back_time}'
              f'|expand marks={net.expand_mark}|search mode={search_mode}|'
              f'residual_violation={net.over_time}|Q_time_violation={net.qtime_violation}')

    #for m in net.log:
    #    print(m)
    out_path = "../../results/"
    ops = [op for op in net.ops if op.stage!=-1]
    tmp_p = ['p3','p4','p5','p6','p7']
    tmpid = [net.id2p_name.index(i) for i in tmp_p]
    ops2 = []
    for i,id in enumerate(tmpid):
        for j in range(net.stage_c[i+1]):
            machine = j
            p_occ = net.place_times[id][machine]
            for oc in p_occ:
                job = oc.tok_key
                start = oc.start
                end = oc.end
                proc = net.proc[i+1]
                ops2.append(Op(job=job,stage=i+1,machine=machine,start=start,proc_end=start+proc,end=end))

    arm_info = {'ARM1': ["t3", "u3", "t4", "u6","t7","u31",'u7','t8'],
                'ARM2': ["u4", "t5", "u5", "t6"],
                'STAGE2ACT': {1: ("t3", ["u3","u31"]), 2: ("t4", "u4"), 3: ("t5", "u5"), 4: ("t6", "u6"),5:('t7','u7')}}
    n_job = net.n_wafer[0]+net.n_wafer[1]
    plot_gantt_hatched_residence(ops=ops2,proc_time=net.proc,
                                 capacity=net.stage_c,n_jobs=n_job,
                                 out_path=out_path,with_label=True,
                                 arm_info=arm_info,policy=0,no_arm=False)

import cProfile
import pstats

def run():
    main()   # 你现在的主调度 / 仿真入口




if __name__ == '__main__':
    cProfile.runctx(
        "run()",
        globals(),
        locals(),
        "profile.out"
    )

    p = pstats.Stats("profile.out")
    p.sort_stats("cumtime").print_stats(30)
