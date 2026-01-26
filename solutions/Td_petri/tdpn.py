import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import sys
import time
from solutions.model.pn_models import Place, WaferToken, BasedToken
from construct_net.run_supernet import load_petri_split
from solutions.v2.utils import (Interval,_insert_interval_sorted,Message,QueueItem)
# from solutions.v2.utils import _first_free_time_at
#from solutions.v2.net_v2 import ActionInfo
from visualization.plot import plot_gantt_hatched_residence,Op
from solutions.Td_petri.construct import SuperPetriBuilderV3,ModuleSpec
from dataclasses import dataclass

INF_OCC = 10**18

PARALLEL_GROUPS = {
    "PM1_4": ["PM1", "PM2", "PM3", "PM4"],
    "PM7_8": ["PM7", "PM8"],
    "PM9_10": ["PM9", "PM10"],
}

@dataclass
class ActionInfo:
    t: int
    fire_times: List[float]
    t_name: str
    chain: List[str]

from typing import List

def _first_free_time_at(intervals: List[Interval], t: int, t2: int) -> int:
    """
    intervals: 按 start 升序、互不重叠
    返回最早 cur>=t，使得 [cur, cur+dur) 可插入，其中 dur = t2 - t
    """
    cur = int(t)
    dur = int(t2) - int(t)
    if dur <= 0 or not intervals:
        return cur

    # ---- 二分：找到第一个 start > cur 的位置 i ----
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid].start <= cur:
            lo = mid + 1
        else:
            hi = mid
    i = lo

    # ---- 先检查前一个区间是否覆盖 cur ----
    if i > 0 and intervals[i - 1].end > cur:
        cur = intervals[i - 1].end

    # ---- 扫描所有与 [cur, cur+dur) 有冲突的区间（只扫冲突段）----
    while i < len(intervals):
        itv = intervals[i]
        end = cur + dur

        if itv.start >= end:          # 后面的区间都在窗口右侧，不冲突
            break

        if itv.end <= cur:            # 当前区间在 cur 左侧（可能因为 cur 被推后），跳过
            i += 1
            continue

        # 有重叠：把 cur 推到该占用区间末尾，窗口整体右移
        cur = itv.end
        i += 1

    return cur


def _first_free_time_open(intervals: List[Interval], t: int) -> int:
    """
    返回最早 cur >= t，使得从 cur 开始没有已排期区间覆盖。
    用于开放区间检查（wafer 进入后离开时间不确定）。
    """
    if len(intervals) == 0:
        return t
    else:
        for itv in intervals:
            if itv.end >= t:
                t = itv.end
        return t


def res_occ_to_ops(res_occ: dict, proc: dict) -> List[Op]:
    """
    将 net.res_occ 转换为 Op 列表
    - 工艺资源：PM / LLC / LLD（is_arm=False）
    - 机械手资源：ARM2 / ARM3（is_arm=True，甘特图画成绿色）
    """

    ops: List[Op] = []

    def map_stage_machine(res: str) -> Tuple[int, int, bool]:
        # -------- 工艺 --------
        if res in ("PM7", "PM8"):
            return 1, 0 if res == "PM7" else 1, False
        if res == "LLC":
            return 2, 0, False
        if res.startswith("PM") and res[2:].isdigit() and 1 <= int(res[2:]) <= 4:
            return 3, int(res[2:]) - 1, False
        if res == "LLD":
            return 4, 0, False
        if res in ("PM9", "PM10"):
            return 5, 0 if res == "PM9" else 1, False

        # -------- 机械手 --------
        if res == "ARM2":
            return 6, 0, True
        if res == "ARM3":
            return 7, 0, True

        return -1, 0, False

    for res_name, intervals in res_occ.items():
        stage, machine, is_arm = map_stage_machine(res_name)
        if stage < 0:
            continue

        for iv in intervals:
            proc_end = iv.end
            if stage > 0:
                proc_end = proc[stage]
            ops.append(
                Op(
                    job=int(iv.tok_key),
                    stage=int(stage),
                    machine=int(machine),
                    start=float(iv.start),
                    proc_end=float(iv.start + proc_end),
                    end=float(iv.end),
                    is_arm=is_arm,
                    kind=iv.kind,
                    from_loc=getattr(iv, 'from_loc', ''),
                    to_loc=getattr(iv, 'to_loc', '')
                )
            )

    return ops




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

class TimedPetri:

    def __init__(self) -> None:

        modules = {
            "LP1": ModuleSpec(tokens=0, capacity=100),
            "LP2": ModuleSpec(tokens=25, capacity=100),
            "AL": ModuleSpec(tokens=0, capacity=1),
            "LLA_S2": ModuleSpec(tokens=0, capacity=1),
            "PM7": ModuleSpec(tokens=0, capacity=1),
            "PM8": ModuleSpec(tokens=0, capacity=1),
            "LLC": ModuleSpec(tokens=0, capacity=1),
            "PM1": ModuleSpec(tokens=0, capacity=1),
            "PM2": ModuleSpec(tokens=0, capacity=1),
            "PM3": ModuleSpec(tokens=0, capacity=1),
            "PM4": ModuleSpec(tokens=0, capacity=1),
            "LLD": ModuleSpec(tokens=0, capacity=1),
            "PM9": ModuleSpec(tokens=0, capacity=1),
            "PM10": ModuleSpec(tokens=0, capacity=1),
            "LLB_S1": ModuleSpec(tokens=0, capacity=1),
            "LP_done": ModuleSpec(tokens=0, capacity=100),
        }
        routes = [
            ["LP1", "AL", "LLA_S2", ["PM7", "PM8"], "LLC", ["PM1", "PM2", "PM3", "PM4"], "LLD",
             ["PM9", "PM10"], "LLB_S1", "LP_done"],
            ["LP2", "AL", "LLA_S2", ["PM7", "PM8"], ["PM9", "PM10"], "LLB_S1", "LP_done"],
        ]

        builder = SuperPetriBuilderV3()
        info = builder.build(modules=modules, routes=routes)

        self.pre = info['pre']
        self.pst = info['pst']
        self.net = self.pst - self.pre
        self.m0 = info['m0']
        self.m = self.m0.copy()
        self.t_duration = info['t_time']
        self.id2p_name = info['id2p_name']
        self.id2t_name = info['id2t_name']
        self.idle_idx = info['idle_idx']
        self.marks: List[Place] = info['marks']
        self.marks_copy = self._clone_marks(self.marks)
        self._init_path()
        self.md = info['md']
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.n_wafer = info['n_wafer']
        self.lp_done_idx = self.idle_idx['end']

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
        self.last_time = 1
        self.over_time = 0
        self.qtime_violation = 0
        self.shot = "s"
        self.dead_mark = []
        self.bad_mark = []

        self.transition_times = [[] for _ in range(self.T)]

        self.ops = []
        self._init_resources()
        self._cache_action_info = {}

        self.open_mod_occ = {}  # key: (module_name, job_id) -> Interval
        # 例如 ("PM7", 12) -> Interval(start=100, end=INF, tok_key=12, kind=...)

        self.stage_c = {1: 2, 2: 1, 3: 4, 4: 1, 5: 2, 6:1, 7:1}
        self.proc = {1: 70, 2: 0, 3: 600, 4: 70, 5: 200, 6:5, 7:5}
        self.debug = False

        # 记录“上次用到的机器在该组里的下标”
        # 例：self.rr_idx = {"PM7_8": 0} 表示上次用了 PM7
        self.rr_idx = {g: -1 for g in PARALLEL_GROUPS}  # -1 表示还没用过
        self._build_rl_action_space()

        self.obs_place_idx = []
        for i,name in enumerate(self.id2p_name):
             if name.startswith('P_READY'):
                 self.obs_place_idx.append(i)

        self.his_len = 50
        self.his_a = [0] * self.his_len  # 用 -1 代表"无历史"
        # his_par_a_by_stage 和 his_par_total_len 在 _build_rl_action_space 中初始化
        # 现在更新 obs_dim（需要 obs_place_idx 和 his_par_total_len）
        self.obs_dim = 2 * len(self.obs_place_idx) + self.his_len

    def _mark2obs(self):
        obs1 = np.zeros(16)
        obs2 = np.zeros(16)
        for k, idx in enumerate(self.obs_place_idx):
            p = self.marks[idx]
            for tok in p.tokens:
                if tok.type == 1:
                    obs1[k] += 1
                elif tok.type == 2:
                    obs2[k] += 1

        # 将所有stage的历史记录摊平成一维
        # 按pstage顺序排列（从小到大）
        #his_par_list = []
        #for pstage in sorted(self.his_par_a_by_stage.keys()):
        #    his_par_list.extend(self.his_par_a_by_stage[pstage])
        #his_par = np.array(his_par_list, dtype=np.float32)
        his_a = np.array(self.his_a)
        
        obs = np.concatenate([obs1, obs2, his_a])
        return obs



    def _build_rl_action_space(self):
        # ... 你的 pathD / pathC 保持不变 ...
        pathD = [[['ARM1_PICK__LP1__TO__AL', 'ARM1_MOVE__LP1__TO__AL', 'ARM1_LOAD__LP1__TO__AL', 'PROC__AL']],
                 # LP1->AL
                 [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2',
                   'PROC__LLA_S2']],  # AL->LLA
                 [['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7',
                   'PROC__PM7', 'ARM2_PICK__PM7__TO__PM10', 'ARM2_MOVE__PM7__TO__PM10', 'ARM2_LOAD__PM7__TO__PM10',
                   'PROC__PM10'],
                  ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8',
                   'PROC__PM8', 'ARM2_PICK__PM8__TO__PM9', 'ARM2_MOVE__PM8__TO__PM9', 'ARM2_LOAD__PM8__TO__PM9',
                   'PROC__PM9']],  # LLA->PM7/PM8->PM9/PM10
                 [['ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1',
                   'PROC__LLB_S1'],
                  ['ARM2_PICK__PM10__TO__LLB_S1', 'ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1',
                   'PROC__LLB_S1']],  # PM9-PM10->LLB
                 [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done',
                   'PROC__LP_done']]]  # LLB->LP_done

        pathC = [[['ARM1_PICK__LP2__TO__AL', 'ARM1_MOVE__LP2__TO__AL', 'ARM1_LOAD__LP2__TO__AL', 'PROC__AL']],
                 # LP2->AL
                 [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2',
                   'PROC__LLA_S2']],  # AL->LLA
                 [['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7',
                   'PROC__PM7', 'ARM2_PICK__PM7__TO__LLC', 'ARM2_MOVE__PM7__TO__LLC', 'ARM2_LOAD__PM7__TO__LLC',
                   'PROC__LLC'],
                  ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8',
                   'PROC__PM8', 'ARM2_PICK__PM8__TO__LLC', 'ARM2_MOVE__PM8__TO__LLC', 'ARM2_LOAD__PM8__TO__LLC',
                   'PROC__LLC']],  # LLA->PM7/PM8->LLC
                 [['ARM3_PICK__LLC__TO__PM1', 'ARM3_MOVE__LLC__TO__PM1', 'ARM3_LOAD__LLC__TO__PM1', 'PROC__PM1',
                   'ARM3_PICK__PM1__TO__LLD', 'ARM3_MOVE__PM1__TO__LLD', 'ARM3_LOAD__PM1__TO__LLD', 'PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM2', 'ARM3_MOVE__LLC__TO__PM2', 'ARM3_LOAD__LLC__TO__PM2', 'PROC__PM2',
                   'ARM3_PICK__PM2__TO__LLD', 'ARM3_MOVE__PM2__TO__LLD', 'ARM3_LOAD__PM2__TO__LLD', 'PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM3', 'ARM3_MOVE__LLC__TO__PM3', 'ARM3_LOAD__LLC__TO__PM3', 'PROC__PM3',
                   'ARM3_PICK__PM3__TO__LLD', 'ARM3_MOVE__PM3__TO__LLD', 'ARM3_LOAD__PM3__TO__LLD', 'PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM4', 'ARM3_MOVE__LLC__TO__PM4', 'ARM3_LOAD__LLC__TO__PM4', 'PROC__PM4',
                   'ARM3_PICK__PM4__TO__LLD', 'ARM3_MOVE__PM4__TO__LLD', 'ARM3_LOAD__PM4__TO__LLD', 'PROC__LLD']],
                 # LLC->PM1-PM4->LLD
                 [['ARM2_PICK__LLD__TO__PM10', 'ARM2_MOVE__LLD__TO__PM10', 'ARM2_LOAD__LLD__TO__PM10', 'PROC__PM10',
                   'ARM2_PICK__PM10__TO__LLB_S1', 'ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1',
                   'PROC__LLB_S1'],
                  ['ARM2_PICK__LLD__TO__PM9', 'ARM2_MOVE__LLD__TO__PM9', 'ARM2_LOAD__LLD__TO__PM9', 'PROC__PM9',
                   'ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1',
                   'PROC__LLB_S1']],  # LLD->PM9-PM10->LLB
                 [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done',
                   'PROC__LP_done']]]  # LLB->LP_done

        allowed = []  # [(tag, chain_tuple), ...] 仅用于收集
        chain_meta = {}  # ch -> {"is_parallel":bool, "pstage":int, "tags":set()}

        pstage_id = 0  # 并行阶段计数器（跨 C/D 共用，方便统一编码）

        def add_path(tag, path):
            nonlocal pstage_id
            for stage in path:
                is_parallel = (len(stage) > 1)
                cur_pstage = pstage_id if is_parallel else -1
                if is_parallel:
                    pstage_id += 1

                for chain in stage:
                    ch = tuple(chain)
                    allowed.append((tag, ch))

                    if ch not in chain_meta:
                        chain_meta[ch] = {"is_parallel": is_parallel,
                                          "pstage": cur_pstage,
                                          "tags": set([tag])}
                    else:
                        # 同一条 chain 可能在别的路线也出现：共用 meta，但 tags 合并
                        chain_meta[ch]["tags"].add(tag)
                        # 如果任一路线认为它是并行选择的一部分，就当并行（更安全）
                        if is_parallel and not chain_meta[ch]["is_parallel"]:
                            chain_meta[ch]["is_parallel"] = True
                            chain_meta[ch]["pstage"] = cur_pstage

        add_path("D", pathD)
        add_path("C", pathC)

        # 关键：按 chain 去重（相同 chain 共用 id）
        uniq_chains = []
        seen = set()
        for _, ch in allowed:
            if ch in seen:
                continue
            seen.add(ch)
            uniq_chains.append(ch)

        self.aid2chain = uniq_chains  # [chain_tuple,...]
        self.chain2aid = {ch: i for i, ch in enumerate(uniq_chains)}
        self.A = len(self.aid2chain)

        # 重要：建立 aid -> 是否并行/并行阶段
        self.aid_is_parallel = np.zeros(self.A, dtype=bool)
        self.aid_pstage = -np.ones(self.A, dtype=np.int32)
        self.aid2tags = [set() for _ in range(self.A)]  # 可选调试：属于哪些路线

        for aid, ch in enumerate(self.aid2chain):
            meta = chain_meta.get(ch, None)
            if meta is None:
                continue
            self.aid_is_parallel[aid] = bool(meta["is_parallel"])
            self.aid_pstage[aid] = int(meta["pstage"])
            self.aid2tags[aid] = meta["tags"]

        # 计算每个stage的并行chain数量，并初始化历史记录队列
        # pstage -> count of parallel chains in that stage
        stage_chain_count = {}
        for aid in range(self.A):
            if self.aid_is_parallel[aid]:
                pstage = self.aid_pstage[aid]
                if pstage >= 0:  # 有效的并行阶段
                    if pstage not in stage_chain_count:
                        stage_chain_count[pstage] = set()
                    # 记录该stage的所有并行chain的aid（用于去重）
                    stage_chain_count[pstage].add(aid)
        
        # 为每个stage创建历史记录队列，长度为并行chain数量+1
        self.his_par_a_by_stage = {}
        self.stage_chain_counts = {}
        for pstage, aid_set in stage_chain_count.items():
            chain_count = len(aid_set)
            self.stage_chain_counts[pstage] = chain_count
            queue_len = chain_count + 1
            self.his_par_a_by_stage[pstage] = [-1] * queue_len
        
        # 计算总的obs维度（所有stage的历史记录摊平后的长度）
        self.his_par_total_len = sum(len(q) for q in self.his_par_a_by_stage.values())

    def _init_path(self):
        # 并行腔体
        pathD = [[['ARM1_PICK__LP1__TO__AL', 'ARM1_MOVE__LP1__TO__AL','ARM1_LOAD__LP1__TO__AL','PROC__AL']], #LP1->AL
                 [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2', 'PROC__LLA_S2']],#AL->LLA
                 [['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7', 'PROC__PM7','ARM2_PICK__PM7__TO__PM10', 'ARM2_MOVE__PM7__TO__PM10', 'ARM2_LOAD__PM7__TO__PM10','PROC__PM10'],
                  ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8','PROC__PM8','ARM2_PICK__PM8__TO__PM9', 'ARM2_MOVE__PM8__TO__PM9', 'ARM2_LOAD__PM8__TO__PM9','PROC__PM9']],#LLA->PM7/PM8->PM9/PM10
                 [['ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1','PROC__LLB_S1'],
                  ['ARM2_PICK__PM10__TO__LLB_S1','ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1','PROC__LLB_S1']], #PM9-PM10->LLB
                 [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done','PROC__LP_done']]] #LLB->LP_done

        pathC = [[['ARM1_PICK__LP2__TO__AL', 'ARM1_MOVE__LP2__TO__AL','ARM1_LOAD__LP2__TO__AL','PROC__AL']], #LP2->AL
                 [['ARM1_PICK__AL__TO__LLA_S2', 'ARM1_MOVE__AL__TO__LLA_S2', 'ARM1_LOAD__AL__TO__LLA_S2', 'PROC__LLA_S2']], #AL->LLA
                 [['ARM2_PICK__LLA_S2__TO__PM7', 'ARM2_MOVE__LLA_S2__TO__PM7', 'ARM2_LOAD__LLA_S2__TO__PM7','PROC__PM7','ARM2_PICK__PM7__TO__LLC', 'ARM2_MOVE__PM7__TO__LLC', 'ARM2_LOAD__PM7__TO__LLC','PROC__LLC'],
                  ['ARM2_PICK__LLA_S2__TO__PM8', 'ARM2_MOVE__LLA_S2__TO__PM8', 'ARM2_LOAD__LLA_S2__TO__PM8','PROC__PM8','ARM2_PICK__PM8__TO__LLC', 'ARM2_MOVE__PM8__TO__LLC', 'ARM2_LOAD__PM8__TO__LLC','PROC__LLC']], #LLA->PM7/PM8->LLC
                 [['ARM3_PICK__LLC__TO__PM1', 'ARM3_MOVE__LLC__TO__PM1', 'ARM3_LOAD__LLC__TO__PM1','PROC__PM1','ARM3_PICK__PM1__TO__LLD', 'ARM3_MOVE__PM1__TO__LLD', 'ARM3_LOAD__PM1__TO__LLD','PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM2', 'ARM3_MOVE__LLC__TO__PM2', 'ARM3_LOAD__LLC__TO__PM2','PROC__PM2','ARM3_PICK__PM2__TO__LLD', 'ARM3_MOVE__PM2__TO__LLD', 'ARM3_LOAD__PM2__TO__LLD','PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM3', 'ARM3_MOVE__LLC__TO__PM3', 'ARM3_LOAD__LLC__TO__PM3','PROC__PM3','ARM3_PICK__PM3__TO__LLD', 'ARM3_MOVE__PM3__TO__LLD', 'ARM3_LOAD__PM3__TO__LLD','PROC__LLD'],
                  ['ARM3_PICK__LLC__TO__PM4', 'ARM3_MOVE__LLC__TO__PM4', 'ARM3_LOAD__LLC__TO__PM4','PROC__PM4','ARM3_PICK__PM4__TO__LLD', 'ARM3_MOVE__PM4__TO__LLD', 'ARM3_LOAD__PM4__TO__LLD','PROC__LLD']], #LLC->PM1-PM4->LLD
                 [['ARM2_PICK__LLD__TO__PM10','ARM2_MOVE__LLD__TO__PM10','ARM2_LOAD__LLD__TO__PM10','PROC__PM10','ARM2_PICK__PM10__TO__LLB_S1', 'ARM2_MOVE__PM10__TO__LLB_S1', 'ARM2_LOAD__PM10__TO__LLB_S1', 'PROC__LLB_S1'],
                  ['ARM2_PICK__LLD__TO__PM9', 'ARM2_MOVE__LLD__TO__PM9', 'ARM2_LOAD__LLD__TO__PM9','PROC__PM9','ARM2_PICK__PM9__TO__LLB_S1', 'ARM2_MOVE__PM9__TO__LLB_S1', 'ARM2_LOAD__PM9__TO__LLB_S1','PROC__LLB_S1']], #LLD->PM9-PM10->LLB
                 [['ARM1_PICK__LLB_S1__TO__LP_done', 'ARM1_MOVE__LLB_S1__TO__LP_done', 'ARM1_LOAD__LLB_S1__TO__LP_done','PROC__LP_done']]] #LLB->LP_done
        
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
        
        lp1_idx = self.idle_idx['start'][1]
        lp2_idx = self.idle_idx['start'][0]
        
        for token in self.marks[lp1_idx].tokens:
            token.path = [stage.copy() for stage in pathC_idx]
        
        for token in self.marks[lp2_idx].tokens:
            token.path = [stage.copy() for stage in pathD_idx]

    def reset(self):
        # --- 最小必要状态 ---
        self.time = 1
        self.m = self.m0.copy()
        self.marks = self._clone_marks(self.marks_copy)
        self._init_path()

        self.transition_times = [[] for _ in range(self.T)]
        self._init_resources()
        self._cache_action_info = {}

        self.open_mod_occ = {}

        self.his_a = [0] * self.his_len  # 用 -1 代表"无历史"
        # 重置所有stage的历史记录
        for pstage in self.his_par_a_by_stage:
            queue_len = len(self.his_par_a_by_stage[pstage])
            self.his_par_a_by_stage[pstage] = [-1] * queue_len

        # --- 可选：仅当你仍需要这些（否则删掉） ---
        # self.ops = []  # 训练不画图可删

        # --- RL 用缓存 ---
        if not hasattr(self, "_cache_action_info"):
            self._cache_action_info = {}
        else:
            self._cache_action_info.clear()

        mask = np.zeros(self.A, dtype=bool)

        self.rr_idx = {g: -1 for g in PARALLEL_GROUPS}

        tran_queue = self.get_enable_t(self.m, self.marks)
        for item in tran_queue:
            chain = tuple(item[3])
            fire_times = item[4]

            # 只保留在 allowed chain 里的动作
            aid = self.chain2aid.get(chain, None)
            if aid is None:
                continue

            mask[aid] = True
            self._cache_action_info[aid] = ActionInfo(
                t=item[0], fire_times=fire_times, t_name=item[2], chain=item[3]
            )

        obs = self._mark2obs()
        return obs, mask

    def step(self, action: int, record=False):


        self.his_a.pop(0)
        self.his_a.append(int(action))
        '''
                if self.aid_is_parallel[action]:
            pstage = self.aid_pstage[action]
            if pstage >= 0 and pstage in self.his_par_a_by_stage:
                # 更新对应stage的历史记录
                self.his_par_a_by_stage[pstage].pop(0)
                self.his_par_a_by_stage[pstage].append(int(action))
        '''


        #dense1 = self.calc_tool_utilization()
        reward1 = self.cal_reward(self.marks)

        ainfo = self._cache_action_info[action]
        chain, fire_times = ainfo.chain, ainfo.fire_times

        info = self._fire_chain(chain, fire_times, self.m, self.marks)
        self.m = info["m"]
        self.marks = info["marks"]
        #self.time = info["time"]  # 建议更新
        reward2 = self.cal_reward(self.marks)
        #dense2 = self.calc_tool_utilization()

        # 重新算 mask
        tran_queue = self.get_enable_t(self.m, self.marks)
        self._cache_action_info.clear()
        mask = np.zeros(self.A, dtype=bool)

        for item in tran_queue:
            chain = tuple(item[3])
            fire_times = item[4]
            aid = self.chain2aid.get(chain, None)
            if aid is None:
                continue
            mask[aid] = True
            self._cache_action_info[aid] = ActionInfo(
                t=item[0], fire_times=fire_times, t_name=item[2], chain=item[3]
            )

        obs = self._mark2obs()
        done = info["finish"]

        #返回：动作掩码，状态，系统当前时间，是否结束，奖励

        return mask, obs, self.time, done, reward2 - reward1

    def cal_reward(self, mark):
        work_finish = 0
        xx = [0, 10, 30, 100, 770, 970, 1000]
        for p in self.obs_place_idx:
            if p not in self.idle_idx['start']:
                place = mark[p]
                for tok in place.tokens:
                    if isinstance(tok, WaferToken):
                        ww = tok.where
                        if tok.type == 1:
                            return -1
                        elif tok.type == 2:
                            work_finish += xx[ww]
        return work_finish/self.time

    def _tool_keys(self):
        # 只算机台，不算机械手；按你 _init_resources 里的命名来
        return [k for k in self.res_occ.keys() if not k.startswith("ARM")]

    def calc_tool_utilization(self, window=None, tool_keys=None):
        """
        window=None: [0, self.time] 的累计利用率
        window=W   : [self.time-W, self.time] 的滑窗利用率
        返回:
          util_sys: 系统平均利用率（机台均值，按时间加权等价）
          util_per: dict[tool] = 该机台利用率
          busy_sum: 所有机台忙碌总时长
          denom   : 分母 = N_tools * (t1-t0)
        """
        t1 = float(self.time)
        if window is None:
            t0 = 0.0
        else:
            t0 = max(0.0, t1 - float(window))
        span = max(t1 - t0, 1e-9)

        if tool_keys is None:
            tool_keys = self._tool_keys()

        util_per = {}
        busy_sum = 0.0
        for k in tool_keys:
            busy = 0
            occ = self.res_occ.get(k, [])
            for itv in occ:
                if itv.end > 100000:
                    itvend = t1
                else:
                    itvend = itv.end
                busy_sum += itvend - itv.start
            util = busy / span
            util_per[k] = util
            busy_sum += busy

        denom = max(len(tool_keys) * span, 1e-9)
        util_sys = busy_sum / span
        return util_sys

    def _parse_to_machine(self, t_name: str):
        # ARM2_PICK__LLA_S2__TO__PM7 -> "PM7"
        if "__TO__" not in t_name:
            return None
        to_part = t_name.split("__TO__")[-1]
        return to_part if to_part.startswith("PM") else None

    def _machine_group(self, machine: str):
        for g, ms in PARALLEL_GROUPS.items():
            if machine in ms:
                return g
        return None

    def filter_by_round_robin(self, enable_ts):
        # group -> list[(t_id, machine)]
        bucket = {}
        for t_id in enable_ts:
            t_name = self.id2t_name[t_id]
            m = self._parse_to_machine(t_name)
            if not m:
                continue
            g = self._machine_group(m)
            if not g:
                continue
            bucket.setdefault(g, []).append((t_id, m))

        keep = set(enable_ts)

        for g, items in bucket.items():
            machines = PARALLEL_GROUPS[g]
            if len(items) <= 1:
                continue  # 该组只有一台可使能，不需要轮转裁剪

            # 当前可使能的机器集合
            enabled_machines = {m for _, m in items}

            # 从 last_idx+1 开始循环找第一个可使能机器
            last_idx = self.rr_idx.get(g, -1)
            chosen_machine = None
            for k in range(1, len(machines) + 1):
                cand = machines[(last_idx + k) % len(machines)]
                if cand in enabled_machines:
                    chosen_machine = cand
                    break

            if chosen_machine is None:
                continue  # 理论上不该发生

            # 同组只保留 chosen_machine 对应的变迁，其它屏蔽
            for t_id, m in items:
                if m != chosen_machine and t_id in keep:
                    keep.remove(t_id)

        return [t for t in enable_ts if t in keep]

    def update_rr_after_fire(self, t_id):
        t_name = self.id2t_name[t_id]
        m = self._parse_to_machine(t_name)
        if not m:
            return
        g = self._machine_group(m)
        if not g:
            return
        machines = PARALLEL_GROUPS[g]
        self.rr_idx[g] = machines.index(m)

    def _resource_enable(self, m):
        """资源允许的使能变迁"""
        mask = (self.pre <= m[:, None]).all(axis=0)
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

    def _dry_run_chain(self, chain_names, m, marks, max_retry=200):
        """
        极简 dry-run（最终版）：
        0) 先做一次轻量结构一致性检查（fire + enable）
        1) 再做段级资源区间检查（ARM / PROC）
        2) 不 clone、不 _tpn_fire
        """

        # ---------- 轻量结构工具 ----------
        def _struct_enable_single(t, m):
            pre = self.pre[:, t]
            idx = np.nonzero(pre > 0)[0]
            return bool(np.all(m[idx] >= pre[idx]))

        def _struct_fire(m, t):
            return m - self.pre[:, t] + self.pst[:, t]

        # ---------- 工具函数 ----------
        def is_arm(name: str):
            return name.startswith("ARM")

        def parse_arm(name: str):
            return name.split("_", 1)[0]

        def parse_proc_module(name: str):
            return name.split("__", 1)[1]

        def split_blocks(chain_names):
            blocks = []
            cur = []
            cur_type = None
            for name in chain_names:
                t_id = self.id2t_name.index(name)
                typ = "ARM" if is_arm(name) else "PROC"
                if cur_type is None or typ == cur_type:
                    cur.append(t_id)
                    cur_type = typ
                else:
                    blocks.append((cur_type, cur))
                    cur = [t_id]
                    cur_type = typ
            if cur:
                blocks.append((cur_type, cur))
            return blocks

        # ---------- chain / id ----------
        t_ids = [self.id2t_name.index(n) for n in chain_names]

        # ---------- (0) 结构性预检查 ----------
        tmp_m = m.copy()
        for t_id in t_ids:
            tmp_se = self._resource_enable(tmp_m)
            if t_id in tmp_se:
                tmp_m = _struct_fire(tmp_m, t_id)
            else:
                return False, [], -1, None, None

        # ---------- 初始起点 ----------
        t0 = t_ids[0]
        base_start = int(self._earliest_enable_time(t0, m, marks))
        shift0 = 0

        real_res_occ = getattr(self, "res_occ", {})

        # ---------- 预处理 ----------
        blocks = split_blocks(chain_names)

        block_dur = {
            id(block): sum(int(self.t_duration[tid]) for tid in block)
            for _, block in blocks
        }

        arm_resource = None
        for name in chain_names:
            if is_arm(name):
                arm_resource = parse_arm(name)
                break

        proc_resource = {}
        for typ, block in blocks:
            if typ == "PROC":
                name = self.id2t_name[block[0]]
                proc_resource[id(block)] = parse_proc_module(name)

        # 识别最后一个 PROC 块的索引（用于开放区间检查）
        last_proc_idx = None
        for i, (typ, _) in enumerate(blocks):
            if typ == "PROC":
                last_proc_idx = i

        # ---------- retry（段级资源检查） ----------
        for _ in range(max_retry):
            t0_time = int(base_start + shift0)
            cur_t = t0_time
            ok = True
            need_shift = 0

            for i, (typ, block) in enumerate(blocks):
                dur = block_dur[id(block)]
                s, e = cur_t, cur_t + dur

                if typ == "ARM":
                    if arm_resource:
                        occ = real_res_occ.get(arm_resource, [])
                        t_free = _first_free_time_at(occ, s, e)
                        if t_free != s:
                            ok = False
                            need_shift = max(need_shift, t_free - s)
                            break
                else:
                    module = proc_resource[id(block)]
                    occ = real_res_occ.get(module, [])
                    if i == last_proc_idx:
                        # 最后一个 PROC：使用开放区间检查
                        t_free = _first_free_time_open(occ, s)
                    else:
                        # 非最后一个 PROC：使用闭区间检查
                        t_free = _first_free_time_at(occ, s, e)
                    if t_free != s:
                        ok = False
                        need_shift = max(need_shift, t_free - s)
                        break

                cur_t = e

            if ok:
                times = []
                t = t0_time
                for tid in t_ids:
                    times.append(t)
                    t += int(self.t_duration[tid])
                return True, times, int(cur_t), None, None

            shift0 += max(1, int(need_shift))

        return False, [], -1, None, None

    def _fire_chain(self, chain_names, times, m, marks):
        """
        真正连续发射链：
        - times 由 _dry_run_chain 给出
        - 不再调用 _earliest_enable_time
        """
        cur_m, cur_marks = m, marks
        cur_time = int(times[0])

        # 预先转成 t_ids，避免反复 index
        t_ids = [self.id2t_name.index(n) for n in chain_names]

        # 删除 wafer 已发射的链（保持你原逻辑）
        t0_id = t_ids[0]
        t0_pre = np.nonzero(self.pre[:, t0_id])[0]

        self.update_rr_after_fire(t0_id)
        for p in t0_pre:
            if marks[p].type > 3:
                continue
            if isinstance(marks[p].head(), WaferToken):
                tok = marks[p].head()
                tok.path.pop(0)
                tok.where += 1

        # 严格按 times 执行
        for i, t_id in enumerate(t_ids):
            te = int(times[i])  # ⭐ 直接用 dry-run 给出的时间

            info = self._search_fire(t_id, cur_m, cur_marks, te)
            cur_m, cur_marks, cur_time = info["m"], info["marks"], info["time"]

        # 链执行完后返回当前状态
        se = self._resource_enable(cur_m)
        se = self._color_enable(se, cur_marks)
        finish = True if cur_m[self.lp_done_idx] == self.n_wafer else False
        deadlock = (not finish) and (len(se) == 0)

        return {
            "m": cur_m,
            "marks": cur_marks,
            "mask": se,
            "finish": finish,
            "deadlock": deadlock,
            "time": cur_time
        }

    def _init_resources(self):
        """
        初始化资源占用时间轴（只有变迁有时间）：
        - self.res_occ: Dict[str, List[Interval]]
          key 是资源名（机械手/机台），value 是按 start 排序的占用区间列表。

        你后续在 _t_resources(t, marks) 里返回的资源名，必须都能在这里出现；
        即使不预注册也行（_tpn_fire/_sync_start 会 setdefault），但预注册更安全。
        """
        self.res_occ = {}

        # 1) 机械手资源（按你的系统改名/数量）
        for arm in ("ARM1", "ARM2", "ARM3"):
            self.res_occ[arm] = []

        for m in ('PM7','PM8','LLA','LLB','LLC','LLD','AL','PM1','PM2','PM3','PM4'):
            self.res_occ[m] = []

    def _t_resources(self, t: int, marks) -> List[str]:
        """
        V3（配合你的 builder 命名）：
        - 加工：  PROC__PM7      -> ["PM7"]
                PROC__LLA_S2    -> ["LLA_S2"]
        - 搬运：  ARM2_PICK__A__TO__B / ARM3_MOVE__... / ARM1_LOAD__... -> ["ARM2"/"ARM3"/"ARM1"]
        - 其他：  []
        """
        t_name = self.id2t_name[t]

        # 1) 工艺加工：占用对应模块资源
        # 例：PROC__PM7 / PROC__LLD / PROC__LLA_S2
        if t_name.startswith("PROC__"):
            mod = t_name.split("__", 1)[1]
            return [mod]

        # 2) 搬运动作：占用机械手资源
        # 例：ARM2_PICK__LLA_S2__TO__PM7
        if t_name.startswith("ARM"):
            # arm 名在第一个 "__" 前：ARM1 / ARM2 / ARM3
            arm = t_name.split("_", 1)[0]  # "ARM2"
            return [arm]

        return []

    def _close_resources(self, t: int) -> List[str]:
        t_name = self.id2t_name[t]

        module_list = ['PM7','PM8','PM1','PM2','PM3','PM4','LLC','LLD','PM9','PM10']

        # 例：ARM2_PICK__LLA_S2__TO__PM7
        if t_name.startswith("ARM") and "__" in t_name:
            parts = t_name.split("__")
            # parts = ['ARM2_PICK', 'LLA_S2', 'TO', 'PM7']
            sub_part = parts[0].split("_")
            if sub_part[1] != 'PICK':
                return []
            from_chamber = parts[1]
            if from_chamber in module_list:
                return from_chamber
            else:
                return []

        return []

    def _sync_start(self, res_names: List[str], t0: int, dur: int) -> int:
        """
        从 t0 开始，寻找一个最早时刻 t，使得：
          对所有资源 r ∈ res_names，
          区间 [t, t+dur) 在 r 的占用时间轴上都是空闲的。

        - self.res_occ[r] : List[Interval]，按 start 升序排列
        - Interval 采用半开区间 [start, end)
        - 返回值一定 >= t0
        """
        # 没有资源约束，直接返回
        if not res_names or dur <= 0:
            return int(t0)

        t = int(t0)

        # 迭代同步（通常 1~3 轮即可收敛）
        for _ in range(50):
            t_new = t
            for r in res_names:
                occ = self.res_occ.get(r)
                if not occ:
                    continue

                # 在该资源上，找 >= t 的最早可插入起点
                t_r = _first_free_time_at(occ, t, t + dur)
                if t_r > t_new:
                    t_new = t_r

            # 若所有资源在同一 t 已满足，收敛
            if t_new == t:
                return t

            t = t_new

        # 极端情况下仍未收敛，返回当前 t（理论上不该发生）
        return t

    # ---------- 计算最早可使能时间 ----------
    def _earliest_enable_time(self, t: int, m, marks, start_from=None) -> int:
        d = int(self.t_duration[t])

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]

        # 1) 晶圆就绪时间下界
        t1 = 0 if start_from is None else int(start_from)
        for p in pre_places:
            place = marks[p]
            if place.type > 3:
                continue

            tok = place.head()
            if tok is None:
                # 理论上不该发生：pre 已满足时这里应有 token
                continue

            tok_ready = int(tok.enter_time)

            if tok_ready > t1:
                t1 = tok_ready

        # 2) 资源最早可用（可插入区间）
        res_names = self._t_resources(t, marks)
        t2 = int(self._sync_start(res_names, t1, d))

        return t2

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]


    def _tpn_fire(
            self,
            t: int,
            m: np.ndarray,
            marks: Optional[List[Place]] = None,
            start_from: Optional[int] = None,
            with_effect: bool = True,
    ) -> Tuple[np.ndarray, List[Place], int]:
        """
        V3：只有变迁有时间。
        - 不再维护 place_times / open_occ
        - 仅在资源时间轴 self.res_occ 上插入占用区间 [te, te+d)
        - token.enter_time 更新为变迁结束时刻 enter_new = te + d
        """
        new_m = m.copy()
        new_marks = marks


        te = start_from

        # 变迁持续时间（必须为 int）
        d = int(self.t_duration[t])

        enter_new = te + d  # 结束时刻（半开区间）

        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        consumed: List[Tuple[int, BasedToken]] = []
        moved_tok: Optional[BasedToken] = None

        # 1) 消费 pre
        for p in pre_places:
            place = new_marks[p]
            if place.type in [1, 2, 3]:  # wafer/资源相关库所
                tok = place.pop_head()
                consumed.append((p, tok))
                if moved_tok is None:
                    moved_tok = tok
            else:
                _ = place.pop_head()
            new_m[p] -= 1

        # 2) 占用资源时间轴（只有变迁占用资源）
        if with_effect:
            res_names = self._t_resources(t, new_marks)  # 你自己的资源映射函数
            # 尽量把 job_id 带上，后续画图/统计方便；没有就用 -1
            tok_key = getattr(moved_tok, "job_id", -1) if moved_tok is not None else -1

            for rn in res_names:
                occ = self.res_occ.setdefault(rn, [])
                t_name = self.id2t_name[t]
                xx = -1
                from_loc = ""
                to_loc = ""
                if t_name.startswith("ARM"):
                    head = t_name.split("__", 1)[0]
                    action = head.split("_", 1)[1]
                    match action:
                        case 'PICK':
                            xx = 0
                        case 'LOAD':
                            xx = 0
                        case 'MOVE':
                            xx = 1
                    # 解析取放位置：ARM2_PICK__LLA_S2__TO__PM7 -> from_loc="LLA_S2", to_loc="PM7"
                    parts = t_name.split("__")
                    if len(parts) >= 4:
                        from_loc = parts[1]
                        to_loc = parts[3]

                module_list = ['PM7', 'PM8', 'PM1', 'PM2', 'PM3', 'PM4', 'LLC', 'LLD', 'PM9', 'PM10']
                itv = Interval(start=int(te), end=int(enter_new), tok_key=tok_key, kind=xx, from_loc=from_loc, to_loc=to_loc)
                if rn in module_list:
                    itv.end = INF_OCC

                _insert_interval_sorted(occ, itv)
                self.open_mod_occ[(rn, tok_key)] = itv

        # 3) 生成 pst
        for p in pst_places:
            pst_place_type = new_marks[p].type

            if pst_place_type in [1, 2, 3]:
                if moved_tok is None:
                    raise RuntimeError(f"t={self.id2t_name[t]} needs moved_tok but none from pre_places")

                moved_tok.enter_time = int(enter_new)
                new_marks[p].append(moved_tok)
            else:
                # 非 wafer place：生成普通 token
                new_tok = BasedToken(enter_time=int(enter_new))
                new_marks[p].append(new_tok)

            new_m[p] += 1

        # 关闭区间
        if with_effect:
            leave_time = start_from
            from_chamber = self._close_resources(t)
            if len(from_chamber) > 0 :
                tok_key = getattr(moved_tok, "job_id", -1) if moved_tok is not None else -1
                self.open_mod_occ[(from_chamber, tok_key)].end = leave_time


        # 4) 更新全局时间（推进到该变迁结束）
        if with_effect and self.time < enter_new:
            self.time = int(enter_new)

        #self.snapshot(new_m, True)
        return new_m, new_marks, int(enter_new)


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
        V3：用于 search 的发射（无 PPO 逻辑）。

        与旧版区别：
        - 你的新 builder 变迁名是：
            * PROC__<module>                       （加工，占用 module 资源）
            * ARMx_PICK__A__TO__B / MOVE / LOAD     （搬运，占用 ARMx）
        - 只有变迁有时间：start_from 是该变迁开始时刻 te
        - 记录 ops：只记录“加工类 PROC__*”，其 start=te, proc_end=te+dur, end=te+dur
          （搬运也可记，但你原 plot 主要看工艺段）
        """
        t_name = self.id2t_name[t]
        te = int(start_from)
        dur = int(self.t_duration[t])
        tf = te + dur



        # ---- 记录 ops：仅记录加工段（PROC__*）----
        if t_name.startswith("PROC__"):
            module = t_name.split("__", 1)[1]

            # job_id：从该加工变迁的输入库所里找 wafer token（type<=3）
            job_id = -1
            pre_places = np.nonzero(self.pre[:, t] > 0)[0]
            for p in pre_places:
                if marks[p].type <= 3 and isinstance(marks[p].head(), WaferToken):
                    job_id = marks[p].head().job_id
                    break

            # stage/machine：若你仍要沿用原来的 1~5 stage 画图，这里做一个映射
            # 你可以按自己的 stage 定义修改这个映射
            stage = -1
            machine = 0
            if module in ("PM7", "PM8"):
                stage = 1
                machine = 0 if module == "PM7" else 1
            elif module == "LLC":
                stage = 2
                machine = 0
            elif module in ("PM1", "PM2", "PM3", "PM4"):
                stage = 3
                machine = int(module[2:]) - 1  # PM1->0 ... PM4->3
            elif module == "LLD":
                stage = 4
                machine = 0
            elif module in ("PM9", "PM10"):
                stage = 5
                machine = 0 if module == "PM9" else 1
            else:
                # 其他加工模块（AL/LLA_S2/LLB_S1 等）如果不想画在甘特图里，就保持 -1
                stage = -1
                machine = 0

            if stage != -1 and job_id != -1:
                self.ops.append(
                    Op(job=job_id, stage=stage, machine=machine,
                       start=te, proc_end=te + dur, end=tf)
                )

        # 先 fire（会更新 token.enter_time=tf，并占用资源到 res_occ）
        #tmp_mark = self._clone_marks(marks)
        new_m, new_marks, time = self._tpn_fire(
            t, m, marks, start_from=te, with_effect=True
        )

        finish = True if new_m[self.lp_done_idx] == self.n_wafer else False
        return {
            "m": new_m,
            "marks": new_marks,
            "finish": finish,
            "time": int(time),
        }

    def get_enable_t(self, m, mark):
        se = self._resource_enable(m)
        #se = self.filter_by_round_robin(se)
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
            transition_queue.append((t, key_time, name, chain, times))
        return transition_queue

    def render_gantt(self, out_path: str = "../../results/", policy: int = 1, with_label: bool = True, no_arm: bool = True):
        """
        基于 res_occ 生成 ops 并绘制甘特图
        """
        ops = res_occ_to_ops(self.res_occ, self.proc)
        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=self.proc,
            capacity=self.stage_c,
            n_jobs=self.n_wafer,
            out_path=out_path,
            with_label=with_label,
            arm_info={},
            policy=policy,
            no_arm=no_arm
        )

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

            t, enable_time, t_name, chain, fire_times = transition_queue.pop(0)


            time_violation_type = self._if_violation(t, enable_time, marks)
            if time_violation_type == 2:
                self.qtime_violation += 1
            elif time_violation_type == 1:
                self.over_time += 1

            if self.debug:
                job_id = self._get_t_job_id(t, marks)
                mes = Message(step=len(self.log) + 1,
                              tran_name=t_name,
                              cur_time=int(enable_time),
                              job=job_id,
                              cand=[QueueItem(c, int(b)) for a, b, c, d in transition_queue])

                self.log.append(mes)
                print(mes)

            info = self._fire_chain(chain,fire_times, m, marks)


            finish = info["finish"]
            new_m = info["m"]
            new_marks = info["marks"]
            time = info["time"]

            if finish:
                self.makespan = self.time
                return True

            #if any(np.array_equal(new_m, mx) for mx in self.visited):
            #    continue

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
    search_mode = 2
    start = time.time()


    net = TimedPetri()

    #net.reset()
    net.search(net.m, net.marks, 0, search_mode)
    print(f'makespan={net.makespan}|search time = {time.time() - start:.2f}|back_time={net.back_time}'
          f'|expand marks={net.expand_mark}|search mode={search_mode}|'
          f'residual_violation={net.over_time}|Q_time_violation={net.qtime_violation}')

    draw_grantt = True
    if not draw_grantt:
       return
    out_path = "../../results/"
    net.render_gantt(out_path=out_path, policy=1, with_label=True, no_arm=True)

import cProfile
import pstats

def run():
    main()   # 你现在的主调度 / 仿真入口




if __name__ == '__main__':
    run()
    '''
    cProfile.runctx(
        "run()",
        globals(),
        locals(),
        "profile.out"
    )

    p = pstats.Stats("profile.out")
    p.sort_stats("cumtime").print_stats(30)
    '''
