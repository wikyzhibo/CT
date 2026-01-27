import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Set, Optional
from solutions.model.pn_models import Place

# 支持路线中分叉：Stage = "PM7" 或 ["PM7","PM8"]
Stage = Union[str, List[str]]

INF = 10 ** 9

@dataclass
class BasedToken:
    enter_time: int
    stay_time: int = 0
    token_id: int = -1  # wafer 唯一标识，-1 表示未分配
    machine: int = -1   # 分配的机器编号，-1 表示未分配
    color: int = 0      # 晶圆颜色/路线类型：0=未分配, 1=路线1, 2=路线2

    def clone(self):
        return BasedToken(enter_time=self.enter_time, stay_time=self.stay_time, 
                          token_id=self.token_id, machine=self.machine, color=self.color)

@dataclass
class RobotSpec:
    """机械手资源库所 r__TMx"""
    tokens: int  # 初始 token 数量（TM1=1, TM2/TM3=2）
    reach: Set[str]  # 该机械手可以触达的模块集合（用于自动选 robot）


@dataclass
class ModuleSpec:
    """模块库所（LP/AL/LL*/PM*/BUF...）"""
    tokens: int = 0
    ptime: int = 0  # 你手动输入的模块 place time
    capacity: int = 1  # place capacity（PM一般=1；LL可=2；LP按需）


@dataclass
class SharedGroup:
    """共享容量组：组内多个库所共享一个 k 资源池，总和<=cap"""
    name: str
    places: Set[str]
    cap: int = 2


class SuperPetriBuilder:
    """
    输出字段：
      nodes[name] = {
          'type': 'p'/'t', 'id':..., 'tokens':..., 'time':..., 'capacity':..., 'x':..., 'y':...
      }
      edges = [(src_name, dst_name, w)]
      m0 (P,), pre(P,T), pst(P,T)
      ptime(P,), ttime(T,), capacity(P,)
      id2p_name, id2t_name
    """

    def __init__(self, d_ptime: int = 3, default_ttime: int = 2):
        self.d_ptime = int(d_ptime)
        self.default_ttime = int(default_ttime)

        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, int]] = []
        self.id2p_name: List[str] = []
        self.id2t_name: List[str] = []

        self._p_count = 0
        self._t_count = 0

        # 按 id 存
        self._ptime_by_pid: List[int] = []
        self._cap_by_pid: List[int] = []
        self._ttime_by_tid: List[int] = []


    # ----------------- 基础创建 -----------------
    def add_place(
            self,
            name: str,
            tokens: int = 0,
            ptime: int = 0,
            capacity: int = INF,
            x: float = 0.0,
            y: float = 0.0,
    ):
        if name in self.nodes:
            return
        pid = self._p_count
        self.nodes[name] = {
            "type": "p",
            "x": float(x),
            "y": float(y),
            "tokens": int(tokens),
            "id": pid,
            "time": int(ptime),
            "capacity": int(capacity),
        }
        self.id2p_name.append(name)
        self._ptime_by_pid.append(int(ptime))
        self._cap_by_pid.append(int(capacity))
        self._p_count += 1

    def add_transition(
            self,
            name: str,
            ttime: Optional[int] = None,
            x: float = 0.0,
            y: float = 0.0,
    ):
        if name in self.nodes:
            return
        tid = self._t_count
        if ttime is None:
            ttime = self.default_ttime
        self.nodes[name] = {
            "type": "t",
            "x": float(x),
            "y": float(y),
            "tokens": 0,
            "id": tid,
            "time": int(ttime),
        }
        self.id2t_name.append(name)
        self._ttime_by_tid.append(int(ttime))
        self._t_count += 1

    def add_arc(self, src: str, dst: str, w: int = 1):
        arc = (src, dst, int(w))
        if arc in self.edges:
            return
        self.edges.append((src, dst, int(w)))

    # ----------------- 路线展开（支持分叉） -----------------
    @staticmethod
    def _as_list(stage: Stage) -> List[str]:
        return stage if isinstance(stage, list) else [stage]

    @classmethod
    def expand_route_to_edges(cls, route: List[Stage]) -> Set[Tuple[str, str]]:
        """
        例如 [A, [B1,B2], C] =>
          A->B1, A->B2, B1->C, B2->C
        """
        out = set()
        for i in range(len(route) - 1):
            left = cls._as_list(route[i])
            right = cls._as_list(route[i + 1])
            for a in left:
                for b in right:
                    out.add((a, b))
        return out

    @staticmethod
    def pick_robot_for_edge(a: str, b: str, robots: Dict[str, RobotSpec]) -> str:
        cand = [r for r, spec in robots.items() if (a in spec.reach and b in spec.reach)]
        if len(cand) != 1:
            raise ValueError(f"Edge {a}->{b} robot ambiguous/none, candidates={cand}")
        return cand[0]

    # ----------------- 构建主流程 -----------------
    def build(
            self,
            modules: Dict[str, ModuleSpec],
            robots: Dict[str, RobotSpec],
            routes: List[List[Stage]],
            shared_groups: Optional[List[SharedGroup]] = None,
            edge_weight: int = 1,
    ):
        """
        modules: 所有模块库所（共享）
        robots: 机械手资源库所（r__TMx）
        routes: 多条加工路线（可分叉）
        shared_groups: 共享容量组（如 LLA/LLB, LLC/LLD）
        """
        shared_groups = shared_groups or []
        self._modules = modules

        for m, spec in modules.items():
            self.add_place(m, tokens=spec.tokens, ptime=spec.ptime, capacity=spec.capacity)

        # 1) 建机器人资源 place：capacity=tokens；ptime=0
        for rname, rspec in robots.items():
            self.add_place(name=f"r_{rname}", tokens=rspec.tokens, ptime=0, capacity=rspec.tokens)

        # 3) 汇总所有路线的相邻模块边
        all_edges: Set[Tuple[str, str]] = set()
        for route in routes:
            all_edges |= self.expand_route_to_edges(route)

        # 4) 为每条模块边 a->b 创建 u-d-t 子结构
        for a, b in sorted(all_edges):
            if a not in modules or b not in modules:
                raise KeyError(f"Unknown module in route: {a}->{b}")

            robot = self.pick_robot_for_edge(a, b, robots)
            r_place = f"r_{robot}"

            # 命名：u__A__B, d__B, t__B
            u = f"u_{a}_{b}"
            d = f"d_{b}"
            t = f"t_{b}"

            # u/t 变迁：ttime=2（统一）
            self.add_transition(u, ttime=self.default_ttime)
            self.add_transition(t, ttime=self.default_ttime)

            # d 库所：ptime=3（统一），capacity=INF
            self.add_place(d, tokens=0, ptime=self.d_ptime, capacity=2)

            # 基本弧：
            # A + r -> u -> d -> t -> B + r
            self.add_arc(a, u, edge_weight)
            self.add_arc(r_place, u, 1)  # 卸载消耗机械手token
            self.add_arc(u, d, edge_weight)
            self.add_arc(d, t, edge_weight)
            self.add_arc(t, b, edge_weight)
            self.add_arc(t, r_place, 1)  # 装载归还机械手token

        return self.finalize()

    # ----------------- 输出 pre/pst 等 -----------------
    def finalize(self):
        P = self._p_count
        T = self._t_count

        # m0
        m0 = np.zeros(P, dtype=int)
        for name, nd in self.nodes.items():
            if nd["type"] == "p":
                m0[nd["id"]] = nd["tokens"]

        # 支持双起点 LP1/LP2 或单起点 LP
        if 'LP1' in self.id2p_name:
            # 双起点模式
            idle_idx = {
                'start1': self.id2p_name.index('LP1'),
                'start2': self.id2p_name.index('LP2'),
                'end': self.id2p_name.index('LP_done'),
            }
            n_wafer_route1 = m0[self.id2p_name.index("LP1")]
            n_wafer_route2 = m0[self.id2p_name.index("LP2")]
            n_wafer = n_wafer_route1 + n_wafer_route2
            
            md = m0.copy()
            md[self.id2p_name.index("LP_done")] = n_wafer
            md[self.id2p_name.index("LP1")] = 0
            md[self.id2p_name.index("LP2")] = 0
        else:
            # 单起点模式（向后兼容）
            idle_idx = {'start': self.id2p_name.index('LP'),
                        'end': self.id2p_name.index('LP_done'), }
            n_wafer = m0[self.id2p_name.index("LP")]
            n_wafer_route1 = n_wafer
            n_wafer_route2 = 0
            
            md = m0.copy()
            md[self.id2p_name.index("LP_done")] = n_wafer
            md[self.id2p_name.index("LP")] = 0

        # pre/pst
        pre = np.zeros((P, T), dtype=int)
        pst = np.zeros((P, T), dtype=int)

        for src, dst, w in self.edges:
            sn = self.nodes[src]
            dn = self.nodes[dst]
            if sn["type"] == "p" and dn["type"] == "t":
                pre[sn["id"], dn["id"]] += w
            elif sn["type"] == "t" and dn["type"] == "p":
                pst[dn["id"], sn["id"]] += w
            else:
                raise ValueError(f"Illegal arc: {src}({sn['type']}) -> {dst}({dn['type']})")

        matrix = pst - pre
        module_x = {}
        for i in range(P):
            if self.id2p_name[i][0] == 'P':
                a = np.where(matrix[i, :] > 0)[0]
                b = np.where(matrix[i, :] < 0)[0]
                module_x[self.id2p_name[i]] = (a, b)

        ptime = np.array(self._ptime_by_pid, dtype=int)
        capacity = np.array(self._cap_by_pid, dtype=int)
        ttime = np.array(self._ttime_by_tid, dtype=int)

        marks = []
        # idle_places: 资源库所（如 r_TM2, r_TM3）的 token 不需要 ID
        # 通过名称判断而非硬编码索引
        token_id_counter = 0  # 全局 token ID 计数器
        
        for i in range(P):
            pname = self.id2p_name[i]
            if pname.startswith('LP') and pname in getattr(self, '_modules', {}):
                ptype = 3
            elif pname.startswith('d'):
                ptype = 2
            elif pname in getattr(self, '_modules', {}):
                ptype = 1
            else:
                ptype = 4

            place = Place(
                name=pname,
                capacity=int(capacity[i]),
                processing_time=int(ptime[i]),
                type=ptype
            )

            cnt = m0[i]
            if cnt > 0:
                # 资源库所（r_ 开头）的 token 不需要 ID
                if pname.startswith('r_'):
                    for _ in range(cnt):
                        place.append(BasedToken(enter_time=0))
                # LP1 库所的初始 token 分配唯一 ID 和颜色1（路线1）
                elif pname == "LP1":
                    for _ in range(cnt):
                        place.append(BasedToken(enter_time=0, token_id=token_id_counter, color=1))
                        token_id_counter += 1
                # LP2 库所的初始 token 分配唯一 ID 和颜色2（路线2）
                elif pname == "LP2":
                    for _ in range(cnt):
                        place.append(BasedToken(enter_time=0, token_id=token_id_counter, color=2))
                        token_id_counter += 1
                # 单起点 LP 库所（向后兼容）
                elif pname == "LP":
                    for tok_id in range(cnt):
                        place.append(BasedToken(enter_time=0, token_id=tok_id, color=0))
                else:
                    for _ in range(cnt):
                        place.append(BasedToken(enter_time=0))
            marks.append(place)

        return {
            "m0": m0,
            "md": md,
            "pre": pre,
            "pst": pst,
            "ptime": ptime,
            "ttime": ttime,
            "capacity": capacity,
            "id2p_name": self.id2p_name,
            "id2t_name": self.id2t_name,
            "idle_idx": idle_idx,
            "module_x": module_x,
            "marks": marks,
            "n_wafer": n_wafer,
            "n_wafer_route1": n_wafer_route1,
            "n_wafer_route2": n_wafer_route2,
        }

