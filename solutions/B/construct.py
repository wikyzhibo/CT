import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from .pn_models import FlatMarks

# 支持路线中分叉：Stage = "PM7" 或 ["PM7","PM8"]
Stage = Union[str, List[str]]

INF = 10 ** 9

@dataclass(slots=True)
class BasedToken:
    enter_time: int
    stay_time: int = 0
    token_id: int = -1
    machine: int = -1
    route_type: int = 0
    step: int = 0
    where: int = 0
    route_queue: Tuple[Any, ...] = ()
    route_head_idx: int = 0
    _target_place: Optional[str] = None
    _dst_level_targets: Optional[Tuple[str, ...]] = None
    _dst_level_full_on_pick: bool = False
    _place_idx: int = -1

    def clone(self):
        return BasedToken(
            enter_time=self.enter_time,
            stay_time=self.stay_time,
            token_id=self.token_id,
            machine=self.machine,
            route_type=self.route_type,
            step=self.step,
            where=self.where,
            route_queue=tuple(self.route_queue),
            route_head_idx=int(self.route_head_idx),
        )

@dataclass
class RobotSpec:
    """机械手配置（仅用于边归属与可达性约束）"""
    tokens: int  # 兼容字段：保留但在连续模型构图中不再显式建资源库所
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
        for robot_name, robot_spec in robots.items():
            robot_tokens = int(robot_spec.tokens)
            self.add_place(robot_name, tokens=robot_tokens, ptime=0, capacity=robot_tokens)

        # 3) 汇总所有路线的相邻模块边
        all_edges: Set[Tuple[str, str]] = set()
        downstream_block_map: Dict[str, Set[str]] = {}
        for route in routes:
            all_edges |= self.expand_route_to_edges(route)
            for i in range(len(route) - 1):
                left = self._as_list(route[i])
                right = self._as_list(route[i + 1])
                for src in left:
                    if src not in downstream_block_map:
                        downstream_block_map[src] = set()
                    downstream_block_map[src].update(right)
        self._downstream_block_map = downstream_block_map

        # 2) 运输库所按 (TMx, dst) 共享，避免共享 t 时出现多前置死锁
        # 命名规则：
        #   u_<src>_<TMx>_<i>   表示从 src 卸载到 TMx（按 src+TMx 递增）
        #   t_<TMx>_<dst>   表示从 TMx 装载到 dst
        u_idx_by_src_robot: Dict[Tuple[str, str], int] = {}
        d_place_by_robot_dst: Dict[Tuple[str, str], str] = {}
        for a, b in sorted(all_edges):
            if a not in modules or b not in modules:
                raise KeyError(f"Unknown module in route: {a}->{b}")

            robot = self.pick_robot_for_edge(a, b, robots)
            d_key = (robot, b)
            if d_key not in d_place_by_robot_dst:
                d_place = f"d_{robot}_{b}"
                d_place_by_robot_dst[d_key] = d_place
                self.add_place(d_place, tokens=0, ptime=self.d_ptime, capacity=1)
            else:
                d_place = d_place_by_robot_dst[d_key]

            # 命名：u_<src>_<TMx>_<i>, t_<TMx>_<dst>
            u_key = (a, robot)
            u_idx_by_src_robot[u_key] = int(u_idx_by_src_robot.get(u_key, 0)) + 1
            u = f"u_{a}_{robot}_{u_idx_by_src_robot[u_key]}"
            t = f"t_{robot}_{b}"

            # u/t 变迁：ttime=2（统一）
            self.add_transition(u, ttime=self.default_ttime)
            self.add_transition(t, ttime=self.default_ttime)

            # 基本弧：
            # A -> u -> d -> t -> B
            self.add_arc(a, u, edge_weight)
            self.add_arc(u, d_place, edge_weight)
            self.add_arc(d_place, t, edge_weight)
            self.add_arc(t, b, edge_weight)
            # 资源占用：u 占用 TMx，t 释放 TMx
            self.add_arc(robot, u, edge_weight)
            self.add_arc(t, robot, edge_weight)

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

        place_type_arr = np.zeros(P, dtype=np.int8)
        place_cat = np.zeros(P, dtype=np.int8)  # 0=cap1_wafer, 1=multi_wafer, 2=resource
        _multi_wafer_pids: set[int] = set()
        _resource_pids: set[int] = set()

        for i in range(P):
            pname = self.id2p_name[i]
            if pname.startswith("LP") and pname in getattr(self, "_modules", {}):
                ptype = 3
            elif pname.startswith("d"):
                ptype = 2
            elif pname in getattr(self, "_modules", {}):
                ptype = 1
            else:
                ptype = 4

            place_type_arr[i] = ptype
            if ptype == 4:
                place_cat[i] = 2
                _resource_pids.add(i)
            elif capacity[i] > 1:
                place_cat[i] = 1
                _multi_wafer_pids.add(i)
            else:
                place_cat[i] = 0

        token_place = np.full(n_wafer, -1, dtype=np.int32)
        token_enter_time = np.zeros(n_wafer, dtype=np.int32)
        place_token = np.full(P, -1, dtype=np.int32)
        wafer_queues: Dict[int, List[int]] = {p: [] for p in _multi_wafer_pids}
        resource_queues: Dict[int, List[int]] = {p: [] for p in _resource_pids}

        token_id_counter = 0

        def allocate_ids(cnt: int) -> List[int]:
            nonlocal token_id_counter
            nxt = token_id_counter + int(cnt)
            if nxt > n_wafer:
                raise ValueError(
                    f"Initial wafer token count exceeds n_wafer: {nxt} > {n_wafer}"
                )
            ids = list(range(token_id_counter, nxt))
            token_id_counter = nxt
            return ids

        for i in range(P):
            cnt = int(m0[i])
            if cnt <= 0:
                continue

            cat = int(place_cat[i])
            if cat == 2:
                resource_queues[i].extend([0] * cnt)
                continue

            tids = allocate_ids(cnt)
            if cat == 0:
                # cap=1 晶圆库所只记录单 token 映射
                place_token[i] = int(tids[-1])
            else:
                wafer_queues[i].extend(int(tid) for tid in tids)

            for tid in tids:
                token_place[tid] = i
                token_enter_time[tid] = 0

        if token_id_counter != n_wafer:
            raise ValueError(
                f"Initial wafer token mismatch: assigned={token_id_counter}, expected={n_wafer}"
            )

        marks = FlatMarks(
            token_place=token_place,
            token_enter_time=token_enter_time,
            place_token=place_token,
            wafer_queues=wafer_queues,
            resource_queues=resource_queues,
        )

        _pre_places_idx = []
        _pst_places_idx = []
        for t in range(T):
            pre_idx = np.nonzero(pre[:, t] > 0)[0].astype(np.int32)
            pst_idx = np.nonzero(pst[:, t] > 0)[0].astype(np.int32)
            _pre_places_idx.append(pre_idx)
            _pst_places_idx.append(pst_idx)

        return {
            "m0": m0,
            "md": md,
            "pre": pre,
            "pst": pst,
            "pre_place_cache": _pre_places_idx,
            "pst_place_cache": _pst_places_idx,
            "ptime": ptime,
            "ttime": ttime,
            "capacity": capacity,
            "id2p_name": self.id2p_name,
            "id2t_name": self.id2t_name,
            "idle_idx": idle_idx,
            "module_x": module_x,
            "marks": marks,
            "place_type_arr": place_type_arr,
            "place_cat": place_cat,
            "_multi_wafer_pids": _multi_wafer_pids,
            "_resource_pids": _resource_pids,
            "n_wafer": n_wafer,
            "n_wafer_route1": n_wafer_route1,
            "n_wafer_route2": n_wafer_route2,
            "downstream_block_map": {
                src: sorted(list(dsts))
                for src, dsts in getattr(self, "_downstream_block_map", {}).items()
            },
        }


def build_pdr_net(n_wafer: int = 7, takt_cycle: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    构造默认 PDR 路线：
      LP -> PM7/PM8(70s) -> LLC -> PM1/PM2(300s) -> LLD -> LP_done

    机械手分工：
      - TM2 管理 LP, PM7/PM8 的取送；LLC 的送；LLD 的取
      - TM3 管理 PM1/PM2 的取送；LLC 的取；LLD 的送
    """
    modules = {
        "LP": ModuleSpec(tokens=int(n_wafer), ptime=0, capacity=int(n_wafer)),
        "PM7": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "PM8": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "LLC": ModuleSpec(tokens=0, ptime=0, capacity=1),
        "PM1": ModuleSpec(tokens=0, ptime=300, capacity=1),
        "PM2": ModuleSpec(tokens=0, ptime=300, capacity=1),
        "LLD": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "LP_done": ModuleSpec(tokens=0, ptime=0, capacity=int(n_wafer)),
    }
    robots = {
        "TM2": RobotSpec(
            tokens=1,
            reach={"LP", "PM7", "PM8", "LLC", "LLD", "LP_done"},
        ),
        "TM3": RobotSpec(
            tokens=1,
            reach={"LLC", "PM1", "PM2", "LLD"},
        ),
    }
    route = ["LP", ["PM7", "PM8"], "LLC", ["PM1", "PM2"], "LLD", "LP_done"]
    builder = SuperPetriBuilder(d_ptime=5, default_ttime=5)
    info = builder.build(modules=modules, robots=robots, routes=[route])

    if takt_cycle is None:
        takt_intervals = [180] * max(0, int(n_wafer) - 1)
    else:
        takt_intervals = [int(v) for v in takt_cycle]
    takt_prefix = [0]
    for interval in takt_intervals:
        takt_prefix.append(takt_prefix[-1] + int(interval))

    lp_idx = info["id2p_name"].index("LP")
    lp_queue = info["marks"].wafer_queues[lp_idx]
    for pos, tid in enumerate(lp_queue):
        prefix_idx = min(int(pos), len(takt_prefix) - 1)
        info["marks"].token_enter_time[int(tid)] = int(takt_prefix[prefix_idx])
    return info

