from collections import deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Deque, List, Optional, Tuple
from solutions.Continuous_model.construct import SuperPetriBuilder

INF = 10**6

@dataclass
class BasedToken:
    enter_time: int

    def clone(self):
        return BasedToken(
            enter_time=self.enter_time,
        )

@dataclass
class Place:
    name: str
    capacity: int
    processing_time: int
    type: int  # 1 for manipulator place, 2 for delivery place, 3 for idle place, 4 for source place

    tokens: Deque[BasedToken] = field(default_factory=deque)

    def clone(self) -> "Place":
        cloned = Place(name=self.name, capacity=self.capacity, processing_time=self.processing_time,type=self.type)
        cloned.tokens = deque(tok.clone() for tok in self.tokens)
        return cloned

    def head(self):
        return self.tokens[0]

    def pop_head(self):
        return self.tokens.popleft()


    def append(self, token) -> None:
        self.tokens.append(token)

    def res_time(self, current_time: int) -> int:
        """返回当前库所内wafer的剩余超时时间"""
        if len(self.tokens) == 0:
            return 10**5
        else:
            # Type 1 (p_i): Process chamber places with processing time constraint
            # Type 2 (d_i): Delivery/transport places with 30s transport constraint
            if self.type == 1:  # Process chamber (p_i)
                res_time = self.head().enter_time + self.processing_time + 20 - current_time
            elif self.type == 2:  # Transport place (d_i)
                res_time = self.head().enter_time + 30 - current_time
            else:
                # Type 3 (idle), Type 4 (other) - no deadline
                return 10**5

            return -1 if res_time < 0 else int(res_time)

    def __len__(self) -> int:
        return len(self.tokens)


class Petri:
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
        self.ready = [0,0,0]

        self.shot = "s"

        super_info = kwargs.get('super_info')
        if super_info is None:
            build_info = kwargs.get("build_info")
            if build_info is not None:
                builder = SuperPetriBuilder(
                    d_ptime=build_info.get("d_ptime", 3),
                    default_ttime=build_info.get("default_ttime", 2),
                )
                super_info = builder.build(
                    build_info["modules"],
                    build_info["robots"],
                    build_info["routes"],
                    build_info.get("shared_groups"),
                )
        self.pre = super_info['pre']
        self.pst = super_info['pst']
        self.net = self.pst - self.pre
        self.m0 = super_info['m0']
        self.m = super_info['m0']
        self.k = super_info['capacity']
        self.ptime = super_info['ptime']
        # self.ttime = super_info['ttime']
        self.ttime = 2
        self.id2p_name = super_info['id2p_name']
        self.id2t_name = super_info['id2t_name']
        self.idle_idx = super_info['idle_idx']
        self._init_marks = super_info.get("marks") if super_info is not None else None
        if self._init_marks is not None:
            self.marks = self._clone_marks(self._init_marks)
        else:
            raise ValueError("Missing marks in super_info; use SuperPetriBuilder to generate marks.")
        self.md = super_info['md']
        self.capacity_xianzhi = kwargs.get('capacity_xianzhi',
                                           None)  # 进入真空区的晶圆数量限制，dict {place_id:transition_id} int->int
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

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]

    def reset(self):
        self.time = 0
        self.m = self.m0.copy()
        if self._init_marks is not None:
            self.marks = self._clone_marks(self._init_marks)
        else:
            raise ValueError("Missing marks in super_info; use SuperPetriBuilder to generate marks.")

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
        d = int(self.ttime)

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
            tok_enter = int(marks[p].head().enter_time)  # 队头
            tok_enter += self.ptime[p]
            earliest = max(earliest, tok_enter)

        if not ((m + self.pst[:, t]) <= self.k).all():
            return -1

        return earliest

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

    def enabled_transitions_at(self,
                               time: int,
                               m: Optional[np.ndarray] = None,
                               marks: Optional[List[Place]] = None) -> list[int]:
        active_m = self.m if m is None else m
        active_marks = self.marks if marks is None else marks
        mask = self.mask_t(active_m, active_marks)
        enabled = np.nonzero(mask)[0].tolist()
        if not enabled:
            return []
        result = []
        for t in enabled:
            et = self._earliest_enable_time(t, active_m, active_marks, time)
            if et != -1 and et <= time:
                result.append(t)
        return result


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
                new_marks[p].append(BasedToken(enter_time=enter_new))
            new_m[p] += 1

        # 3) 时间推进到完成之后（下一时刻可用）
        time = enter_new

        return new_m, new_marks, enter_new

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
