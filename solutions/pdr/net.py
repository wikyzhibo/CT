from collections import deque
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def get_pre_pst(net_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """把 -1 / +1 网络表转为 Pre / Pst 矩阵（P, T）。"""
    pre = (net_df == -1).to_numpy(dtype=np.int64)
    pst = (net_df ==  1).to_numpy(dtype=np.int64)
    return pre, pst



# ---------- 将变迁序号映射为其名字 ----------
def mapping(t):
    tran = ['u1','t1', 'u2','t2', 'u3','t3', 'u4','t4',
            'u5','t5', 'u6','t6', 'u7','t7','u8','t8']
    return tran[t]

class Petri:
    """
    input:


    - self.m         : 当前标识 (P,)
    - self.m0        : 初始标识 (P,)
    - self.md        : 终止标识 (P,)
    - self.marks[p]  : 库所 p 的 tokens，shape=(2, n_tokens) -> [enter_time; id]
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
    def __init__(self, **kwargs) -> None:

        self.back_time = 0
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


        # 弹出参数
        path = kwargs.get('path', '../../eg4.txt')
        self.n_wafer = kwargs.get("n_wafer", None)
        self.process_time = kwargs.get("process_time",None)
        self.capacity = kwargs.get("capacity",None)
        self.con = kwargs.get("controller",None)
        self.branch_info = kwargs.get("branch_info",None)  # 双晶圆需要屏蔽某些变迁 dict, {pre:'place_name', 'branch':[b1,b2]}
        self.capacity_xianzhi = kwargs.get("capacity_xianzhi",None) #进入真空区的晶圆数量限制，dict {place_id:transition_id} int->int

        #解析.ndr文件
        info = self.parse_file(path)
        self.pre = np.ascontiguousarray(self.pre, dtype=np.int32)
        self.pst = np.ascontiguousarray(self.pst, dtype=np.int32)
        self.net = np.ascontiguousarray(self.pst - self.pre, dtype=np.int32)

        self.m0 = info['m0']
        self.m = info['m0']
        if self.process_time is not None:
            self._arrange_time(info)
            self.marks: List[np.ndarray] = self._init_marks_from_m(self.m, two_mode=kwargs.get('two_mode', 0))
            self.k = np.ascontiguousarray(self.k, dtype=np.int32)

        # 设置时间
        self.ttime = kwargs.get('ttime', 2)

        # petri网系统时钟
        self.time = 0

        # search 函数服务变量

        self.makespan = 0
        self.transitions = []
        self.m_record = []
        self.marks_record = []
        self.time_record = []

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

    def _init_marks_from_m(self, m: np.ndarray, two_mode=0) -> List[np.ndarray]:
        """
        根据标识向量 m 初始化每个库所的 token 列表。
        返回 marks: list of arrays, each with shape=(2, n_tokens)
            row0: enter_time
            row1: token_id  (1..n_tokens)
        """
        idle_place = self.idle_idx['L1']
        marks: List[np.ndarray] = []
        start_from = 0

        for i, cnt in enumerate(m.astype(int).tolist()):
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
                            type = np.repeat(-1, cnt)
                else:
                    type = np.repeat(-1, cnt)
                marks.append(np.vstack([enter, ids, type]))
            else:
                marks.append(np.empty((3, 0), dtype=int))
        return marks

    def reset(self):
        self.time = 0
        self.m = self.m0.copy()
        self.marks = self._init_marks_from_m(self.m, two_mode=1)

        self.m_record = []
        self.marks_record = []
        self.time_record = []

    def mask_t(self, m: np.ndarray[int],marks, with_controller: bool = False, with_clf=False) -> np.ndarray[bool]:
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
        with_capacity_controller = 1

        cond_pre = (self.pre <= m[:, None]).all(axis=0)
        if self.k is not None:
            cond_cap = ((m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)
            mask = cond_pre & cond_cap
        else:
            mask = cond_pre

        # 多类型晶圆情况下，根据晶圆类型选择加工路径
        if self.branch_info is not None:
            b1,b2 = self.branch_info['branch']
            pre_id = self.branch_info['pre']
            if mask[b1] and mask[b2]:
                token = marks[pre_id]
                wafer_type = token[2,0]
                match wafer_type:
                    case 1:
                        mask[b2] = False
                    case 2:
                        mask[b1] = False
                    case _:
                        raise RuntimeError('wafer type not corrected')

        if with_clf:
            te = np.nonzero(mask)[0]
            new = []
            for t in te:
                new.append(m + self.net[:,t])
            new = np.array(new).reshape(len(te),-1)
            y = self.clf.predict(new)
            bad = np.isin(y,1)
            mask[te[bad]] = False

        if with_capacity_controller:
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

        if with_controller:
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

        for tmp in controllers:
            controller = controllers[tmp]
            # 冲突类型死锁
            if tmp == 'f':
                for p,k,t in controller:
                    id_p = place_table.index(p)
                    id_t = tran_table.index(t)
                    if m[id_p] == k:
                        mask[id_t] = False

            # bm类型死锁控制器
            elif tmp[0:2] == 'bm':
                places = controller['p']
                trans = controller['t']
                id_ps = [place_table.index(p) for p in places]
                id_ts = [tran_table.index(t) for t in trans]

                if m[id_ps].sum() == 3:
                    mask[id_ts[0]] = False
                    mask[id_ts[2]] = False
                elif m[id_ps[0]] == 1 and m[id_ps[3]] == 1:
                    mask[id_ts[0]] = False
                    mask[id_ts[3]] = False
                elif m[id_ps[1]] == 1 and m[id_ps[2]] == 1:
                    mask[id_ts[1]] = False
                    mask[id_ts[2]] = False
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
            if marks[p].shape[1] == 0:
                return -1

        # 由前置 token 的 enter_time 决定最早时刻下界
        # 取各前置库所“队头”（最早进入的）token 的 enter_time
        earliest = tau
        for p in pre_places:
            tok_enter = int(marks[p][0, 0])   # 队头
            tok_enter += self.ptime[p]
            earliest  = max(earliest, tok_enter)

        # 检查容量（触发之后，tau+d 时刻往后每个后置库所 +1 是否超容量）
        # 这里采用离散时刻点检查：完成时刻后的即时可用（enter = tau + d）
        # 只要静态容量允许即可（若你有更严格的在制约束，可在此拓展）
        # 由于容量是静态上限，简单检查“现在的 m + Pst[:,t] <= k”即可：
        if not ((m + self.pst[:, t]) <= self.k).all():
            return -1

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
             m: np.ndarray[int],
             marks: Optional[List[np.ndarray]] = None,
             start_from: Optional[int] = None,
             ) -> Tuple[int, int, np.ndarray, List[np.ndarray], int]:
        """
        触发变迁 t，并更新：
          - self.m：m <- m + (Pst - Pre)[:, t]
          - self.marks：消费前置库所的队头 token；在后置库所加入新 token（enter_time = te + d）
          - self.time：前进到 te + d

        :param t transition id
        :param m: mark of petri net
        :param marks: token with clock set
        :param start_from: the clock now
        :returns te, tf, new_m, new_marks, time
        """

        new_m = m.copy()
        time = start_from
        new_marks = marks.copy()
        te = self._earliest_enable_time(t, m, marks, start_from=start_from)
        if te < 0:
            raise RuntimeError(f"Transition {t} cannot be enabled (time={self.time}).")

        d = int(self.ttime)
        tf = te + d - 1  # 完成时刻（含）
        enter_new = tf + 1

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 1) 消费前置库所 token（队头）
        type = -1
        for p in pre_places:
            tokens = new_marks[p]
            if tokens.shape[1] == 0:
                raise RuntimeError(f"Place {p} has no token to consume.")
            # 弹出队头（enter_time 最早）
            x = tokens[:,0]
            if x[2] != -1:
                type = x[2]
            new_marks[p] = tokens[:, 1:]
            new_m[p] -= 1

        # 2) 生成后置库所 token：enter_time = finish+1
        for p in pst_places:
            if self.id2p_name[p][0] in ['p','d']:
                tmp_type = type
            else:
                tmp_type = -1
            next_id = (marks[p].shape[1] + 1) if new_marks[p].size else 1
            new_tok = np.array([[enter_new], [next_id],[tmp_type]], dtype=int)
            if new_marks[p].size == 0:
                new_marks[p] = new_tok
            else:
                new_marks[p] = np.concatenate([new_marks[p], new_tok], axis=1)
            new_m[p] += 1

        # 3) 时间推进到完成之后（下一时刻可用）
        time = enter_new

        return te, tf, new_m, new_marks, time

    # ---------- 一步接口（若需要返回 mask / 完成标志） ----------
    def step(self,
             t: int,
             m = None,
             marks: Optional[List[np.ndarray]] = None,
             start_from: Optional[int] = None,
             update = False):
        """
        use for the PPO training algorithm


        :param t:
        :param m:
        :param marks:
        :param start_from:
        :return:
        """

        if m is None:
            m = self.m.copy()
            marks = self.marks.copy()
            start_from = self.time

        te, tf, new_m, new_marks, time = self._tpn_fire(t,m,marks,start_from=start_from)
        mask = self.mask_t(new_m,new_marks, with_controller=False)
        finish = bool((new_m == self.md).all())
        deadlock = (not finish) and  (not mask.any())

        if update:
            self.m = new_m
            self.marks = new_marks
            self.time = time

        info = {"te_time": te,
                "finish_time": tf,
                "m": new_m,
                "marks": new_marks,
                "mask": mask,
                "finish": finish,
                "deadlock": deadlock,
                "time": time}
        return info

    def back(self):
        self.m_record.pop(-1)
        self.marks_record.pop(-1)
        self.time_record.pop(-1)
        self.m = self.m_record.pop(-1)
        self.marks = self.marks_record.pop(-1)
        self.time = self.time_record.pop(-1)

    def search(self, m, marks, time, mode=0):

        #self.snapshot(m,True)
        self.expand_mark+=1
        mask = self.mask_t(m,marks, with_controller=True,with_clf=False)
        enabled_transitions = np.nonzero(mask)[0]

        transition_queue = [(t, self._earliest_enable_time(t,m,marks,time)) for t in enabled_transitions]
        transition_queue = self._select_mode(transition_queue, mode=mode)


        while transition_queue:

            t, enable_time = transition_queue.pop(0)
            t_name = self.id2t_name[t]
            mxx = self._fire(t,m)
            #self.snapshot(mxx,partial_disp=True)
            mxxx = np.array(mxx).reshape(1,self.P)
            #is_bad = self.clf.predict(mxxx)
            #if is_bad == 1:
            #    continue

            info = self.step(t,m,marks,time)
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
            self.m_record.append(new_m)
            self.marks_record.append(new_marks)

            if self.search(new_m, new_marks, time, mode=mode):
                return True

            #self.snapshot(new_m,partial_disp=True)
            self.back_time += 1
            self.transitions.pop(-1)
            self.m_record.pop(-1)
            self.marks_record.pop(-1)
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
        return queue

    def pdr(self,m):
        pass

    def snapshot(self, mark: Optional[np.ndarray[int]] = None, partial_disp: bool = False) -> None:
        """
        transfer the ndarray mark to human look

        :param mark:
        :param partial_disp: if true, display the d-place or p-place
        """
        if mark is None:
            mark = self.m
        s = ''

        for i in range(self.P):
            if mark[i] != 0:
                name = self.id2p_name[i]
                if partial_disp:
                    if name[0] == 'p' or name[0] == 'd':
                        if mark[i] == 1:
                            s += f'{self.id2p_name[i]}\t'
                        else:
                            s += f'{self.id2p_name[i]}*{mark[i]}\t'
                else:
                    s += f'{self.id2p_name[i]}*{mark[i]}\t'
        print(s)

    def rand_action(self,m):
        mask = self.mask_t(m)
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
                        self.k[node_id] = 100
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

            mask = self.mask_t(cur_m, with_controller=False)
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

    def aco_search(self):
        n_ant = 10
        iter = 10
        rho = 0.9
        m0 = self.m0.copy()
        md = self.md.copy()
        marks = self.marks.copy()
        time = 0

        for i in range(iter):
            for j in range(n_ant):
                self.search(m0,marks,0,mode=2)







