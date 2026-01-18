import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Set, Optional
from solutions.model.pn_models import WaferToken, Place, BasedToken

# 支持路线中分叉：Stage = "PM7" 或 ["PM7","PM8"]
Stage = Union[str, List[str]]

INF = 10 ** 9

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
        self.init_mode = 0

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

    def set_init_mode(self, mode: int) -> None:
        self.init_mode = int(mode)

    def _init_wafer_types(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.array([], dtype=int)
        if self.init_mode == 1:  # 先完成A，再完成B
            types = np.repeat(2, count)
            w = int(count / 3)
            types[:w] = 1
            return types
        if self.init_mode == 2:  # 交替完成
            types = np.repeat(2, count)
            types[::2] = 1
            return types
        if self.init_mode == 3:  # 先完成B，再完成A
            types = np.repeat(2, count)
            w = 2 * int(count / 3)
            types[w:] = 1
            return types
        # 单晶圆模式
        return np.repeat(1, count)

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

        idle_idx = {'start': [self.id2p_name.index('LP1'), self.id2p_name.index('LP2')],
                    'end': self.id2p_name.index('LP_done'), }

        md = m0.copy()
        n_wafer = m0[self.id2p_name.index("LP1")] + m0[self.id2p_name.index("LP2")]
        md[self.id2p_name.index("LP_done")] = n_wafer
        md[self.id2p_name.index("LP1")] = 0
        md[self.id2p_name.index("LP2")] = 0

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
        job_id = 1
        idle_places = set(idle_idx.get("start", [])) if isinstance(idle_idx, dict) else set()
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
                if i in idle_places:
                    for wafer_type in self._init_wafer_types(cnt):
                        place.append(
                            WaferToken(
                                enter_time=0,
                                job_id=job_id,
                                path=[],
                                type=int(wafer_type),
                            )
                        )
                        job_id += 1
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
            "nodes": self.nodes,
            "edges": self.edges,
            "id2p_name": self.id2p_name,
            "id2t_name": self.id2t_name,
            "idle_idx": idle_idx,
            "module_x": module_x,
            "marks": marks,
            "n_wafer": n_wafer
        }


