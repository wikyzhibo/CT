from collections import deque
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
#from solutions.model.guard import ZeroBufferWindowController
from solutions.model.pn_models import Place, Token
from solutions.model.guard import ZeroBufferWindowController

def get_pre_pst(net_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """把 -1 / +1 网络表转为 Pre / Pst 矩阵（P, T）。"""
    pre = (net_df == -1).to_numpy(dtype=np.int64)
    pst = (net_df ==  1).to_numpy(dtype=np.int64)
    return pre, pst

from dataclasses import dataclass


INF = 10**6

class Petri:
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
    def __init__(self, use_super_net=False,
                       with_controller=False,
                       with_capacity_controller=False,
                        with_zhiliu_controller=False,
                        **kwargs) -> None:

        self.back_time = 0
        # 训练分段信号（代），由 environment 传入
        self.generation = kwargs.get('generation', 0)
        # 提供便捷接口以便在训练过程中更新代信号
        # 使用示例：petri.set_generation(new_gen)
        self.id2p_name = None
        self.id2t_name = None
        self.place_names = None
        self.pst = None
        self.pre = None
        self.md = None
        self.m0 = None
        self.T = None
        self.P = None
        self.idle_idx = None
        self.k = None
        self.marks = None
        self.low_dim_idx = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20]
        self.low_dim = len(self.low_dim_idx)
        self.blocking_record = {'p6':[],
                                'p5':[],
                                'p4':[],
                                'p3':[]}
        self.ready = [0,0,0]

        proc_time = {1: 70, 2: 300, 3:70}
        capacity = {1: 2, 2: 2, 3:1}
        self.over_time = 0
        self.qtime_violation = 0

        self.zb = ZeroBufferWindowController(
            proc_time=proc_time,
            tau=15,
            window={1:15,2:15,3:100},
            capacity=capacity,
        )
        self.shot = "s"


        # 变迁名 -> 工序编号（开始加工）
        self.t_start_stage = {
            "t3": 1,
            "t5": 2,
        }


        # 弹出参数
        if not use_super_net:
            path = kwargs.get('path', None)
            self.n_wafer = kwargs.get("n_wafer", None)
            self.process_time = kwargs.get("process_time",None)
            self.capacity = kwargs.get("capacity",None)
            self.branch_info = kwargs.get("branch_info",None)  # 双晶圆需要屏蔽某些变迁 dict, {pre:'place_name', 'branch':[b1,b2]}
            self.capacity_xianzhi = kwargs.get('capacity_xianzhi',
                                               None)  # 进入真空区的晶圆数量限制，dict {place_id:transition_id} int->int

            #解析.ndr文件
            info = self.parse_file(path)
            self.pre = np.ascontiguousarray(self.pre, dtype=np.int32)
            self.pst = np.ascontiguousarray(self.pst, dtype=np.int32)
            self.net = np.ascontiguousarray(self.pst - self.pre, dtype=np.int32)

            self.m0 = info['m0']
            self.m = info['m0']
            if self.process_time is not None:
                self._arrange_time(info)
                self.marks: List[Place] = self._init_marks_from_m(self.m, two_mode=kwargs.get('two_mode', 0))
                self.k = np.ascontiguousarray(self.k, dtype=np.int32)

            # 设置时间
            self.ttime = kwargs.get('ttime', 2)
        else:
            super_info = kwargs.get('super_info')
            self.pre = super_info['pre']
            self.pst = super_info['pst']
            self.net = self.pst - self.pre
            self.m0 = super_info['m0']
            self.m = super_info['m0']
            self.k = super_info['capacity']
            self.ptime = super_info['ptime']
            #self.ttime = super_info['ttime']
            self.ttime = 2
            self.id2p_name = super_info['id2p_name']
            self.id2t_name = super_info['id2t_name']
            self.idle_idx = super_info['idle_idx']
            self.marks: List[np.ndarray] = self._init_marks_from_m(self.m, two_mode=kwargs.get('two_mode', 0))
            self.md = super_info['md']
            self.capacity_xianzhi = kwargs.get('capacity_xianzhi',None)# 进入真空区的晶圆数量限制，dict {place_id:transition_id} int->int
            self.P = self.pre.shape[0]
            self.T = self.pre.shape[1]


        # petri网系统时钟
        self.time = 1

        # search 函数服务变量
        if with_capacity_controller:
            self.with_capacity_controller = True
        if with_controller:
            self.with_controller = True
            self.con = kwargs.get("controller", None)
        self.with_zhuliu_controller = with_zhiliu_controller

        self.log = []

        # search 函数服务变量
        self.makespan = 0
        self.transitions = []
        self.m_record = []
        self.marks_record = []
        self.time_record = []
        self.mask_record = []
        # 超时配对记录（对应之前 env 中的 record）
        self._overtime_record = {'pm1': [],'llc':[] ,'pm2': [],'time':0}

        self.rg_marks = []
        self.visited = []
        self.unvisited = deque()

        self.expand_mark = 0

        self._id_of = {}  # dict[bytes -> int]：标识 -> id
        self._seen = set()
        self.dead_mark = []
        self.bad_mark = []
        #bundle = joblib.load("rf_deadlock_v1.joblib")
        #self.clf = bundle["model"]

    def _init_marks_from_m(self, m: np.ndarray, two_mode=0) -> List[Place]:
        """
        根据标识向量 m 初始化每个库所的 token 列表。
        返回 marks: list of arrays, each with shape=(2, n_tokens)
            row0: enter_time
            row1: token_id  (1..n_tokens)
        """
        idle_place = self.idle_idx['L1']
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
                    if i != self.idle_idx['L1']:
                        id_ = -1
                    place.append(Token(int(id_),int(e), int(tp)))
            marks.append(place)
        return marks

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]

    def reset(self):
        self.time = 0
        self.m = self.m0.copy()
        self.marks: List[Place] = self._init_marks_from_m(self.m)

        self.m_record = []
        self.marks_record = []
        self.mask_record = []
        self.visited = []
        self.transitions = []
        self.time_record = []
        # 清除超时配对记录
        self._overtime_record = {'pm1': [], 'pm2': []}
        self.u5_record = []
        self.u3_record = []
        self.zb.reset()

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

        # 多类型晶圆情况下，根据晶圆类型选择加工路径
        '''
                if self.branch_info is not None:
            b1,b2 = self.branch_info['branch']
            pre_id = self.branch_info['pre']
            if mask[b1] and mask[b2]:
                token = marks[pre_id].head()
                wafer_type = token.wafer_type
                match wafer_type:
                    case 1:
                        mask[b2] = False
                    case 2:
                        mask[b1] = False
                    case _:
                        raise RuntimeError('wafer type not corrected')
        '''


        if with_clf:
            te = np.nonzero(mask)[0]
            new = []
            for t in te:
                new.append(m + self.net[:,t])
            new = np.array(new).reshape(len(te),-1)
            y = self.clf.predict(new)
            bad = np.isin(y,1)
            mask[te[bad]] = False

        if self.with_capacity_controller:
            # 路线B
            '''
            if m[22] == 0:
                new_mask[12] = False
            '''
            for p,t in self.capacity_xianzhi.items():
                p_id = self.id2p_name.index(p)
                t_id = self.id2t_name.index(t)
                if m[p_id] == 0:
                    mask[t_id] = False

        if self.with_controller:
            new_mask = self._controller(m,mask)
            return new_mask

        return mask


    def _controller(self,
                    m: np.ndarray[int],
                    mask: np.ndarray[bool])-> np.ndarray[bool]:
        """
        if there is controller that avoid deadlock, it updates the mask

        :param m: the current marks
        :param mask: the mask for the transition set
        :return: new_mask that generated from the controllers
        """
        controllers = self.con
        place_table = self.id2p_name
        tran_table = self.id2t_name

        def add_mask(mask, idx):



            if isinstance(idx, list):
                for id in idx:
                    if id > 16:
                        return
                    mask[id] = False
            else:
                if idx>16:
                    return
                mask[idx] = False

        for tmp in controllers:
            controller = controllers[tmp]
            # 冲突类型死锁
            if tmp[0] == 'f':
                u2_id = tran_table.index('u2')
                p3_id = place_table.index('p3')
                d3_id = place_table.index('d3')
                if m[p3_id] == 2 and m[d3_id] == 1:
                    mask[u2_id] = False

                p1_id = place_table.index('p1')
                u0_id = tran_table.index('u0')
                if m[p1_id] == 1:
                    mask[u0_id] = False

            # bm类型死锁控制器
            elif tmp[0:2] == 'bm':
                places = controller['p']
                trans = controller['t']
                id_ps = [place_table.index(p) for p in places]
                id_ts = []
                for t in trans:
                    tmp = [tran_table.index(tx) for tx in t ] if isinstance(t,list) else tran_table.index(t)
                    id_ts.append(tmp)

                if m[id_ps].sum() == 3:
                    add_mask(mask, id_ts[0])
                    add_mask(mask, id_ts[2])

                if m[id_ps[0]] == 1 and m[id_ps[3]] == 1:
                    add_mask(mask, id_ts[0])
                    add_mask(mask, id_ts[3])

                if m[id_ps[1]] == 1 and m[id_ps[2]] == 1:
                    add_mask(mask, id_ts[1])
                    add_mask(mask, id_ts[2])

                if m[id_ps[3]] == 2:
                    add_mask(mask, id_ts[0])
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

        t_name = self.id2t_name[t]
        #if t == 5:
        #    earliest = max(earliest,self.u5_record+50)
        #if t == 9 and len(self.u3_record)>0 and len(self.u5_record)==2:
        #    earliest = max(self.u5_record[0] + 250,earliest)
        #if t == 9 and len(self.u3_record)==2 and len(self.u5_record)==1:
        #    earliest = max(self.u5_record[0] + 250,earliest)
        #if t == 9 and len(self.u3_record)==2 and len(self.u5_record)==0:
        #    earliest = max(self.u3_record[-1] + 50,earliest)

        #if t_name == 'u2' and len(self.u2_record) >= 1:
        #    earliest = max(self.u2_record[-1]+ 120,earliest)
        #if t_name == 't3' and len(self.u3_record) >= 1:
        #    earliest = max(self.u3_record[-1] +50, earliest)
        return earliest

    def _fire(self,
              t: int,
              m: np.ndarray[int])-> np.ndarray[int]:
        """
        普通Petri网的变迁发射

        :param t: transition id
        :param m: current mark
        :return: new_m: new mark
        """
        return m + self.net[:, t]

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
                new_tok = Token(job_id=-1, enter_time=enter_new, wafer_type=-1)
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
            if place.name == 'p3' or place.name == 'p5':
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

    def _deadline_fire_time(self,t,mark: List[Place],start_from):
        """
        计算变迁最迟发射的时间
        - 针对u3和u5这种有驻留约束的变迁 (检查p_i places)
        - 针对t1-t8加载变迁 (检查d_i places的30秒传输约束)
        """
        tran = ['u3','u5']
        load_tran = ['t1','t2','t3','t4','t5','t6','t7','t8']
        t_name = self.id2t_name[t]

        deadline = INF

        # Check process chamber deadlines (unload transitions)
        if t_name in tran:
            pre_place = np.nonzero(self.pre[:,t]>0)[0]
            for p in pre_place:
                if self.id2p_name[p][0] == 'p':
                    deadline = mark[p].res_time(start_from)
                    return deadline

        # Check transport deadlines (load transitions)
        if t_name in load_tran:
            pre_place = np.nonzero(self.pre[:,t]>0)[0]
            for p in pre_place:
                if self.id2p_name[p][0] == 'd':
                    deadline = mark[p].res_time(start_from)
                    return deadline

        return deadline

    def _get_t_job_id(self,t,mark: List[Place]):
        pre_place = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_place:
            if mark[p].type == 1 or mark[p].type == 2 or mark[p].type == 3:
                return mark[p].head().job_id
        return -1

    def search(self, m, marks, time, mode=0):

        #self.snapshot(m,True)
        self.expand_mark+=1
        mask = self.mask_t(m,marks)
        enabled_transitions = np.nonzero(mask)[0]
        names = [self.id2t_name[x] for x in enabled_transitions]

        transition_queue = []
        for t in enabled_transitions:
            et = self._earliest_enable_time(t, m, marks, time)  # 逻辑最早
            dt = self._deadline_fire_time(t,marks,time)
            job_id = self._get_t_job_id(t,marks)
            name = self.id2t_name[t]
            if self.with_zhuliu_controller:
                if name == 'u2':
                    ft = self.zb.peek_time('t3', et)
                else:
                    ft = self.zb.peek_time(name, et)  # 加上零缓冲+30s的反推下界（无副作用）
            else:
                ft = et
            transition_queue.append((t, dt, ft, self.id2t_name[t],job_id))

        transition_queue = self._select_mode(transition_queue, mode=mode)

        while transition_queue:

            t, deadline, enable_time,t_name,job_id = transition_queue.pop(0)
            if t_name == 't5':
                self.log.append(enable_time)

            time_violation_type = 0
            time_violation_type = self._if_violation(t,enable_time,marks)
            if time_violation_type == 2:
                self.qtime_violation += 1
            elif time_violation_type == 1:
                self.over_time += 1
            #ready_time = self.cal_ready_time(t,enable_time)
            #mxx = self._fire(t,m)
            #self.snapshot(mxx,partial_disp=True)

            if self.with_zhuliu_controller :
                self.zb.commit(t_name, enable_time,job_id)

            info = self._search_fire(t,m,marks,enable_time)
            finish = info["finish"]
            new_m = info["m"]
            new_marks = info["marks"]
            time = info["time"]

            if finish:
                self.makespan = info["time"]
                return True

            if any(np.array_equal(new_m, mx) for mx in self.visited):
                continue

            self.visited.append(new_m)
            self.transitions.append(t)
            self.time_record.append(time)
            self.m_record.append(new_m)
            self.marks_record.append(new_marks)
            self.mask_record.append(mask)

            if self.search(new_m, new_marks, time, mode=mode):
                return True

            #self.snapshot(new_m,partial_disp=True)
            self.back_time += 1
            self.transitions.pop(-1)
            self.time_record.pop(-1)
            self.m_record.pop(-1)
            self.marks_record.pop(-1)
            self.mask_record.pop(-1)
            self.bad_mark.append(new_m)

        self.expand_mark += 1
        return False

    def collect_expert_data(self):
        import sys
        sys.setrecursionlimit(10000)
        m = self.m0.copy()
        marks = self.marks.copy()
        time = 0

        self.search(m, marks, time,mode=0)
        from solutions.PPO.enviroment import impress_m,low_dim
        press_m = [impress_m(mx,self.idle_idx) for mx in self.m_record]
        low_m = [low_dim(mx,self.low_dim_idx) for mx in press_m]
        return low_m, np.array(self.transitions),self.mask_record


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

    def parse_file(self,filename):
        """
        解析tina.ndr 类型文件为petri网
        filename: 文件名

        output:
        info = {'nodes','edges'}
        - m0 ndarray (n_p,)
        - pre ndarray (n_p,n_t)
        - pst ndarray (n_p,n_t)
        - nodes = {"t1":{"type","id","token","x","y"}}
        - edges = [(input_name, output_name, weight)] List
        """
        nodes = {}
        edges = []
        n_t = 0
        n_p = 0
        self.idle_idx = {'L1':0,'L2':0}
        self.id2p_name = []
        self.id2t_name = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == 't':  # 变迁
                    x, y = float(parts[1]), float(parts[2])
                    name = parts[3]
                    token_count = int(parts[4])
                    nodes[name] = {'type': 't', 'x': x,
                                   'y': y, 'tokens': token_count,
                                   'id': n_t}
                    n_t += 1
                    self.id2t_name.append(name)
                elif parts[0] == 'p':  # 库所
                    x, y = float(parts[1]), float(parts[2])
                    name = parts[3]
                    token_count = int(parts[4])
                    nodes[name] = {'type': 'p', 'x': x, 'y': y, 'tokens': token_count, 'id': n_p}
                    #读取闲置库所id
                    if name[0] == 'L':
                        if int(name[1])==1:
                            self.idle_idx['L1'] = n_p
                        else:
                            self.idle_idx['L2'] = n_p
                    n_p += 1
                    self.id2p_name.append(name)
                elif parts[0] == 'e':  # 弧
                    input_name = parts[1]
                    output_name = parts[2]
                    weight = int(parts[3])
                    edges.append((input_name, output_name, weight))
                elif parts[0] == 'h':
                    break
                else:
                    continue
        self.P = n_p
        self.T = n_t
        m0 = np.zeros(n_p, dtype=int)

        for name in nodes:
            node = nodes[name]
            if node['type'] == 'p':
                id = node['id']
                m0[id] = node['tokens']

        self.md = m0.copy()
        if self.n_wafer is not None:
            m0[self.idle_idx['L1']] = self.n_wafer
            self.md[self.idle_idx['L1']] = 0
            self.md[self.idle_idx['L2']] = self.n_wafer

        # 前置矩阵和后置矩阵
        self.pre = np.zeros(shape=(n_p, n_t),dtype=int)
        self.pst = np.zeros_like(self.pre,dtype=int)
        for input_name, output_name, weight in edges:
            input_node = nodes[input_name]
            output_node = nodes[output_name]
            in_id = input_node['id']
            out_id = output_node['id']
            if input_node['type'] == 'p':
                self.pre[in_id, out_id] = weight
            elif input_node['type'] == 't':
                self.pst[out_id, in_id] = weight

        info = {'m0':   m0,
                'nodes': nodes,
                'edges': edges}
        return info

    def _arrange_time(self, info):
        nodes = info['nodes']
        place_table = self.place_names
        self.ptime = np.zeros(self.P,dtype=int)
        self.k = np.zeros(self.P,dtype=int)
        for name in nodes:
            node = nodes[name]
            if node['type'] == 'p':
                type_of_place = name[0]
                node_id = node['id']
                tmp_id = int(name[1]) - 1
                match type_of_place:
                    case 'p':
                        self.ptime[node_id] = self.process_time[tmp_id]
                        self.k[node_id] = self.capacity['pm'][tmp_id]
                    case 'L':
                        self.k[node_id] = 10000
                    case 'k':
                        self.k[node_id] = self.capacity['bm'][tmp_id]
                    case 'd':
                        self.ptime[node_id] = 3
                        self.k[node_id] = 2
                    case 'r':
                        self.k[node_id] = self.capacity['robot'][tmp_id]
                    case 's':
                        self.k[node_id] = 20
                    case _:
                        print('error')


    def _key(self, m: np.ndarray) -> bytes:
        return np.ascontiguousarray(m).tobytes()

    def _get_id(self, m: np.ndarray) -> int:
        return self._id_of.get(self._key(m), -1)

    def get_rg(self, m: np.ndarray):
        self._id_of = {}  # dict[bytes -> int]：标识 -> id
        self._seen = set()

        start = train_time.time()

        k0 = self._key(m)
        if k0 not in self._seen:
            self._seen.add(k0)
            self._id_of.setdefault(k0, len(self._id_of))
            self.unvisited.append(np.ascontiguousarray(m))  # _fire 返回新数组的话可不再 copy

        while self.unvisited:
            cur_m = self.unvisited.popleft()
            k_cur = self._key(cur_m)
            node_id = self._id_of.setdefault(k_cur, len(self._id_of))

            self.visited.append(cur_m)  # 若担心外部 in-place，改成 cur_m.copy()

            mask = self.mask_t(cur_m)
            te = np.nonzero(mask)[0]

            if te.size == 0:
                self.dead_marks.append(node_id)

            for t in te:
                new_m = self._fire(t, cur_m)  # 确保返回新数组；不确定的话：new_m = new_m.copy()
                kn = self._key(new_m)
                if kn not in self._seen:
                    self._seen.add(kn)  # 先登记 seen
                    self._id_of.setdefault(kn, len(self._id_of))  # 同步登记 id
                    self.unvisited.append(new_m)

        end = train_time.time()
        print(f'|marks={len(self.visited)}|dead = {len(self.dead_marks)}|{end - start:.1f} seconds|')

    def find_bad_mark(self,
                      m: np.ndarray[int])->int:

        k_cur = self._key(m)
        self._seen.add(k_cur)
        self._id_of.setdefault(k_cur, len(self._id_of))
        self.visited.append(m)

        mask = self.mask_t(m, with_controller=False)
        enabled_transitions = np.nonzero(mask)[0]

        if enabled_transitions.size == 0: #死锁状态
            self.dead_mark.append(self._id_of.get(k_cur))
            return 0
        else:
            type_of_cur = 1

        next_record = np.zeros_like(enabled_transitions,dtype=int)
        for i,t in enumerate(enabled_transitions):
            new_m = self._fire(t, m)
            k1 = self._key(new_m)
            #终止标识认为是活的
            if k1 in self.bad_mark:
                flag = 2
            elif k1 in self.dead_mark:
                flag = 0
            elif k1 in self._seen:
                flag = 1
            else:
                flag = self.find_bad_mark(new_m)

            if flag == 0:
                next_record[i] = 0
            elif flag == 1:
                next_record[i] = 1
            elif flag == 2:
                next_record[i] = 0

        state = next_record==1
        if ~np.any(state): #子节点有一个是连通节点，则该节点是连通节点
            type_of_cur = 2
            self.bad_mark.append(self._id_of.get(k_cur, -1))

        return type_of_cur

    def low_dim_token_times(self, marks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """

        """
        if marks is None:
            marks = getattr(self, "marks", None)
        if marks is None:
            # 若没有 marks 可用，返回全 0
            return np.zeros(len(self.low_dim_idx), dtype=int)

        state = []
        for i, p in enumerate(self.low_dim_idx):
            tokens = marks[p]
            if tokens is None or tokens.size == 0:
                #state.append(0)
                continue
            else:
                state.extend(tokens[0,:])
        return state

import sys
import time
from data.config.params_N8 import params_N8

def main():
    sys.setrecursionlimit(10000)
    search_mode = 3
    start = time.time()
    params_N8['n_wafer']=75
    net = Petri(with_controller=True,
                with_capacity_controller=True,
                with_zhiliu_controller = False,
                **params_N8)
    print(f'|p={net.P}|t={net.T}')

    m = net.m.copy()
    marks = net.marks.copy()
    net.search(m,marks,0,search_mode)
    print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}|back_time={net.back_time}'
          f'|expand marks={net.expand_mark}|search mode={search_mode}|'
          f'residual_violation={net.over_time}|Q_time_violation={net.qtime_violation}')
    #print(net.log)

if __name__ == '__main__':
    main()