import numpy as np
from typing import Dict, List, Tuple, Union, Set, Deque
from solutions.model.pn_models import WaferToken, Place, BasedToken
from dataclasses import dataclass,field
from collections import deque

Stage = Union[str, List[str]]
INF = 10**9


@dataclass
class Place:
    name: str
    type: int  # 1 for manipulator place, 2 for delivery place, 3 for idle place, 4 for source place

    tokens: Deque[BasedToken] = field(default_factory=deque)

    def clone(self) -> "Place":
        cloned = Place(name=self.name, type=self.type)
        cloned.tokens = deque(tok.clone() for tok in self.tokens)
        return cloned

    def head(self):
        return self.tokens[0]

    def pop_head(self):
        return self.tokens.popleft()

    def append(self, token) -> None:
        self.tokens.append(token)


# ========== 工艺/动作时间（秒） ==========
T_PICK = 5
T_MOVE = 3
T_LOAD = 5

PROC_TIME = {
    "AL": 10,
    "LLA_S2": 20,
    "LLB_S1": 20,
    "PM7": 70,
    "PM8": 70,
    "PM1": 600, "PM2": 600, "PM3": 600, "PM4": 600,
    "LLC": 5,
    "LLD": 70,
    "PM9": 200,
    "PM10": 200,
    "LP1": 0,
    "LP2": 0,
    "LP_done": 0,
}


@dataclass
class ModuleSpec:
    tokens: int = 0
    capacity: int = 1


class SuperPetriBuilderV3:
    """
    生成“只有变迁有时间”的 PN 结构：
    - place 只表达状态：READY / IN / 中间态
    - transition 才带 time（pick/move/load/proc）
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, int]] = []
        self.id2p_name: List[str] = []
        self.id2t_name: List[str] = []
        self._p_count = 0
        self._t_count = 0
        self._ptime_by_pid: List[int] = []
        self._cap_by_pid: List[int] = []
        self._ttime_by_tid: List[int] = []
        # 增加用于跟踪为 PM 创建的容量库所（c_*）
        self._cap_places: Set[str] = set()

    # ----------------- 基础创建 -----------------
    def add_place(self, name: str, tokens: int = 0, capacity: int = INF):
        if name in self.nodes:
            return
        pid = self._p_count
        self.nodes[name] = {
            "type": "p",
            "tokens": int(tokens),
            "id": pid,
        }
        self.id2p_name.append(name)
        self._ptime_by_pid.append(0)
        self._cap_by_pid.append(int(capacity))
        self._p_count += 1

    def add_transition(self, name: str, ttime: int):
        if name in self.nodes:
            return
        tid = self._t_count
        self.nodes[name] = {
            "type": "t",
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
        self.edges.append(arc)

    # ----------------- 路线展开（支持分叉） -----------------
    @staticmethod
    def _as_list(stage: Stage) -> List[str]:
        return stage if isinstance(stage, list) else [stage]

    @classmethod
    def expand_route_to_edges(cls, route: List[Stage]) -> Set[Tuple[str, str]]:
        out = set()
        for i in range(len(route) - 1):
            left = cls._as_list(route[i])
            right = cls._as_list(route[i+1])
            for a in left:
                for b in right:
                    out.add((a, b))
        return out

    # ----------------- 机械手选择（按你给的规则） -----------------
    @staticmethod
    def pick_robot_for_edge(a: str, b: str) -> str:
        pm78 = {"PM7", "PM8"}
        pm910 = {"PM9", "PM10"}
        pm14 = {"PM1", "PM2", "PM3", "PM4"}

        # ARM1：LPi->AL, AL->LLA_S2, LLB_S1->LP_done
        if (a in {"LP1", "LP2"} and b == "AL") or (a == "AL" and b == "LLA_S2") or (a == "LLB_S1" and b == "LP_done"):
            return "ARM1"

        # ARM3：LLC<->PM1-4, PM1-4<->LLD, LLC<->LLD（按你的描述：LLC取出、LLD送入、PM1-4取送）
        if (a == "LLC" and b in pm14) or (a in pm14 and b == "LLD") or (a == "LLD" and b in pm14) or (a in pm14 and b == "LLC"):
            return "ARM3"
        if (a == "LLC" and b == "LLD") or (a == "LLD" and b == "LLC"):
            return "ARM3"

        # ARM2：LLA_S2/LLD 取出；LLB_S1/LLC 送入；PM7/8/9/10 取送；以及它们之间的移动
        # 这基本覆盖除 ARM1/ARM3 之外的其余边
        return "ARM2"

    # ----------------- 辅助：模块 places 命名 -----------------
    @staticmethod
    def p_ready(m: str) -> str:
        # 在腔室内，已加工完成
        return f"P_READY__{m}"

    @staticmethod
    def p_in(m: str) -> str:
        #在腔室内（加工前）
        return f"P_IN__{m}"

    @staticmethod
    def p_mid_pick(a: str, b: str, arm: str) -> str:
        return f"P_HAND__{arm}__{a}__TO__{b}"

    @staticmethod
    def p_mid_at(b: str, arm: str) -> str:
        return f"P_AT__{arm}__{b}"

    # ----------------- 容量库所（c_*） -----------------
    @ staticmethod
    def c_cap(m: str) -> str:
        return f"c_{m}"

    def _ensure_cap_place(self, m: str, modules: Dict[str, ModuleSpec]):
        """
        除 LP1/LP2/LP_done 外，所有模块都创建容量库所 c_m：
          - tokens = modules[m].capacity
          - capacity = modules[m].capacity
        """
        if m in ("LP1", "LP2", "LP_done"):
            return
        cname = self.c_cap(m)
        if cname in self._cap_places:
            return
        spec = modules[m]
        self.add_place(cname, tokens=int(spec.capacity), capacity=int(spec.capacity))
        self._cap_places.add(cname)
    
    
    # ----------------- 主构建 -----------------
    def build(
        self,
        modules: Dict[str, ModuleSpec],
        routes: List[List[Stage]],
        edge_weight: int = 1,
    ):
        # 1) 先建模块 READY/IN places（IN 只对 proc>0 的模块需要，但建了也没坏处）
        for m, spec in modules.items():
            if m in ("LP1", "LP2"):
                self.add_place(self.p_ready(m), tokens=spec.tokens, capacity=spec.capacity)
                continue
            self.add_place(self.p_ready(m), tokens=spec.tokens, capacity=spec.capacity)
            self.add_place(self.p_in(m), tokens=0, capacity=spec.capacity)

        # 1.5) 除 LP1/LP2/LP_done 外，所有模块建立容量库所 c_*
        for m in modules.keys():
            self._ensure_cap_place(m, modules)

        # 2) 汇总所有路线的相邻边
        all_edges: Set[Tuple[str, str]] = set()
        for route in routes:
            all_edges |= self.expand_route_to_edges(route)

        # 3) 对每条边 A->B，生成 PICK/MOVE/LOAD 子结构
        for a, b in sorted(all_edges):
            if a not in modules or b not in modules:
                raise KeyError(f"Unknown module in route: {a}->{b}")

            arm = self.pick_robot_for_edge(a, b)

            # places

            pA = self.p_ready(a)  # 从 A_READY 取
            pH = self.p_mid_pick(a, b, arm) # 取完后晶圆在机械手上
            pAT = self.p_mid_at(b, arm) # 移动完成在 B 外等待

            self.add_place(pH, tokens=0, capacity=INF)
            self.add_place(pAT, tokens=0, capacity=INF)

            # transitions
            tr_pick = f"{arm}_PICK__{a}__TO__{b}"
            tr_move = f"{arm}_MOVE__{a}__TO__{b}"
            tr_load = f"{arm}_LOAD__{a}__TO__{b}"

            self.add_transition(tr_pick, T_PICK)
            self.add_transition(tr_move, T_MOVE)
            self.add_transition(tr_load, T_LOAD)

            # arcs: A_READY -> pick -> HAND -> move -> AT_B -> load -> B_IN
            self.add_arc(pA, tr_pick, edge_weight)
            self.add_arc(tr_pick, pH, edge_weight)

            self.add_arc(pH, tr_move, edge_weight)
            self.add_arc(tr_move, pAT, edge_weight)

            self.add_arc(pAT, tr_load, edge_weight)
            self.add_arc(tr_load, self.p_in(b), edge_weight)

            # ========= 容量控制（c_*）：除了 LP1/LP2/LP_done 外，所有模块都受控 =========
            #  pick_a -> move -> load -> proc -> pick_b

            if a not in ("LP1", "LP2"):
                self.add_arc(tr_pick, self.c_cap(a))  # 从a处举起一片晶圆时，若a不是开始模块，则需要恢复一个容量给a的容量库所
            if b != 'LP_done':
                self.add_arc(self.c_cap(b),tr_pick)       #从a处举起一片晶圆时，消耗b的一个容量

        # 4) 每个模块的加工：IN -> PROC -> READY
        #    若 proc_time==0：直接用一个零时长“转移”也可以；这里统一建 TR_PROC（0 时长也OK）
        for m in modules.keys():
            # 源头 LP1/LP2 不需要 IN->PROC->READY，直接 IN -> READY
            if m in ("LP1", "LP2"):
                continue
            pt = int(PROC_TIME.get(m, 0))
            tr_proc = f"PROC__{m}"
            self.add_transition(tr_proc, pt)
            self.add_arc(self.p_in(m), tr_proc, 1)
            self.add_arc(tr_proc, self.p_ready(m), 1)

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

        # 终止：所有 wafer 都到 LP_done 的 READY
        # 兼容你原来的 idle_idx 语义：LP1/LP2 是 start
        start_lp1 = self.p_ready("LP1")
        start_lp2 = self.p_ready("LP2")
        end_lp_done = self.p_ready("LP_done")

        idle_idx = {
            "start": [self.id2p_name.index(start_lp1), self.id2p_name.index(start_lp2)],
            "end": self.id2p_name.index(end_lp_done),
        }

        md = m0.copy()
        n_wafer = m0[self.id2p_name.index(start_lp1)] + m0[self.id2p_name.index(start_lp2)]
        md[self.id2p_name.index(end_lp_done)] = int(n_wafer)
        md[self.id2p_name.index(start_lp1)] = 0
        md[self.id2p_name.index(start_lp2)] = 0

        # pre/pst
        pre = np.zeros((P, T), dtype=int)
        pst = np.zeros((P, T), dtype=int)

        for src, dst, w in self.edges:
            sn = self.nodes[src]
            dn = self.nodes[dst]
            if sn["type"] == "p" and dn["type"] == "t":
                pre[sn["id"], dn["id"]] = w
            elif sn["type"] == "t" and dn["type"] == "p":
                pst[dn["id"], sn["id"]] = w
            else:
                raise ValueError(f"Illegal arc: {src}({sn['type']}) -> {dst}({dn['type']})")

        ttime = np.array(self._ttime_by_tid, dtype=int)

        # marks（place.type 仍然沿用你现有 Place/WaferToken 逻辑）
        marks: List[Place] = []
        job_id = 1
        for i in range(P):
            pname = self.id2p_name[i]

            if 'LP1' in pname:
                type = 1
            elif 'LP2' in pname:
                type = 2
            # 把所有 wafer 所在状态 place 设为 type=3（可被 _get_t_info 找到）
            # 中间 HAND/AT 也属于 wafer 状态，设为 type=2 也行；这里统一 <=3
            if pname.startswith("P_"):
                ptype = 3
            else:
                ptype = 4

            place = Place(
                name=pname,
                type=ptype,
            )

            cnt = int(m0[i])
            if cnt > 0:
                if pname in (start_lp1, start_lp2):
                    for _ in range(cnt):
                        place.append(WaferToken(enter_time=0, job_id=job_id, path=[],type=type))
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
            "t_time": ttime,
            "id2p_name": self.id2p_name,
            "id2t_name": self.id2t_name,
            "idle_idx": idle_idx,
            "marks": marks,
            "n_wafer": int(n_wafer),
        }

def main():
    modules = {
        "LP1": ModuleSpec(tokens=25, capacity=100),
        "LP2": ModuleSpec(tokens=50, capacity=100),
        "AL": ModuleSpec(tokens=0, capacity=1),
        "LLA_S2": ModuleSpec(tokens=0, capacity=2),
        "PM7": ModuleSpec(tokens=0, capacity=1),
        "PM8": ModuleSpec(tokens=0, capacity=1),
        "LLC": ModuleSpec(tokens=0, capacity=1),
        "PM1": ModuleSpec(tokens=0, capacity=1),
        "PM2": ModuleSpec(tokens=0, capacity=1),
        "PM3": ModuleSpec(tokens=0, capacity=1),
        "PM4": ModuleSpec(tokens=0, capacity=1),
        "LLD": ModuleSpec(tokens=0, capacity=2),
        "PM9": ModuleSpec(tokens=0, capacity=1),
        "PM10": ModuleSpec(tokens=0, capacity=1),
        "LLB_S1": ModuleSpec(tokens=0, capacity=2),
        "LP_done": ModuleSpec(tokens=0, capacity=100),
    }

    routes = [
        ["LP1", "AL", "LLA_S2", ["PM7", "PM8"], "LLC", ["PM1", "PM2", "PM3", "PM4"], "LLD",
         ["PM9", "PM10"], "LLB_S1", "LP_done"],
        ["LP2", "AL", "LLA_S2", ["PM7", "PM8"], ["PM9", "PM10"], "LLB_S1", "LP_done"],
    ]

    builder = SuperPetriBuilderV3()
    info = builder.build(modules=modules, routes=routes)

if __name__ == '__main__':
    main()
