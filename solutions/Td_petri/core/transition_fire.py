"""
变迁发射执行器模块。

该模块封装了 Petri 网中变迁发射的所有逻辑，
包括单变迁发射、链式发射和 dry-run 验证。
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass

from solutions.model.pn_models import Place, WaferToken, BasedToken
from solutions.Td_petri.resources import Interval, _first_free_time_at, _first_free_time_open, _insert_interval_sorted
from solutions.Td_petri.resources import get_transition_resources, get_close_resources, DEFAULT_MODULE_LIST
from visualization.plot import Op

INF_OCC = 10**18


@dataclass
class FireResult:
    """发射结果"""
    m: np.ndarray
    marks: List[Place]
    time: int
    finish: bool = False
    deadlock: bool = False
    mask: List = None


class TransitionFireExecutor:
    """
    变迁发射执行器，封装所有变迁发射相关逻辑。
    
    该类负责：
    1. 单变迁发射 (fire) - 合并了原 _tpn_fire 和 _search_fire
    2. 链式发射 (fire_chain)
    3. Dry-run 验证 (dry_run_chain)
    
    Example:
        >>> executor = TransitionFireExecutor(...)
        >>> result = executor.fire(t_id, m, marks, start_from=100)
        >>> chain_result = executor.fire_chain(chain_names, times, m, marks)
    """
    
    def __init__(self,
                 pre: np.ndarray,
                 pst: np.ndarray,
                 t_duration: np.ndarray,
                 id2t_name: List[str],
                 lp_done_idx: int,
                 n_wafer: int,
                 action_checker,
                 resource_mgr,
                 time_getter: Callable[[], int],
                 time_setter: Callable[[int], None],
                 ops_list: List):
        """
        初始化变迁发射执行器。
        
        Args:
            pre: 前置矩阵 (P x T)
            pst: 后置矩阵 (P x T)
            t_duration: 变迁持续时间数组
            id2t_name: 变迁ID到名称映射
            lp_done_idx: 完成库所索引
            n_wafer: 总晶圆数
            action_checker: 动作使能检查器
            resource_mgr: 资源管理器
            time_getter: 获取当前时间的函数
            time_setter: 设置当前时间的函数
            ops_list: 操作记录列表（用于甘特图）
        """
        self.pre = pre
        self.pst = pst
        self.t_duration = t_duration
        self.id2t_name = id2t_name
        self.lp_done_idx = lp_done_idx
        self.n_wafer = n_wafer
        self.action_checker = action_checker
        self.resource_mgr = resource_mgr
        self.time_getter = time_getter
        self.time_setter = time_setter
        self.ops = ops_list
        
        # 资源占用表的引用
        self.res_occ = resource_mgr.res_occ
        self.open_mod_occ = resource_mgr.open_mod_occ

    def fire(self, t: int, m: np.ndarray, marks: List[Place], 
             start_from: int, record_ops: bool = True) -> FireResult:
        """
        发射单个变迁（合并 _tpn_fire 和 _search_fire）。
        
        Args:
            t: 变迁ID
            m: 当前标记向量
            marks: 当前库所标记列表
            start_from: 开始时刻
            record_ops: 是否记录到 ops 列表（用于甘特图）
        
        Returns:
            FireResult 包含新标记、时间和完成状态
        """
        t_name = self.id2t_name[t]
        te = int(start_from)
        d = int(self.t_duration[t])
        tf = te + d
        
        # 1) 记录 ops（如果需要，仅记录 PROC）
        if record_ops and t_name.startswith("PROC__"):
            self._record_op(t, marks, te, d, tf)
        
        # 2) 核心发射逻辑
        new_m = m.copy()
        new_marks = marks
        enter_new = te + d
        
        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]
        
        consumed: List[Tuple[int, BasedToken]] = []
        moved_tok: Optional[BasedToken] = None
        
        # 消费 pre
        for p in pre_places:
            place = new_marks[p]
            if place.type in [1, 2, 3]:
                tok = place.pop_head()
                consumed.append((p, tok))
                if moved_tok is None:
                    moved_tok = tok
            else:
                _ = place.pop_head()
            new_m[p] -= 1
        
        # 占用资源时间轴
        res_names = get_transition_resources(t_name)
        tok_key = getattr(moved_tok, "job_id", -1) if moved_tok is not None else -1
        w_type = getattr(moved_tok, "type", 0) if moved_tok is not None else 0
        
        for rn in res_names:
            occ = self.res_occ.setdefault(rn, [])
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
                        xx = 1
                    case 'MOVE':
                        xx = 2
                parts = t_name.split("__")
                if len(parts) >= 4:
                    from_loc = parts[1]
                    to_loc = parts[3]
            
            module_list = ['PM7', 'PM8', 'PM1', 'PM2', 'PM3', 'PM4', 'LLC', 'LLD', 'PM9', 'PM10']
            itv = Interval(start=int(te), end=int(enter_new), tok_key=tok_key, 
                          kind=xx, from_loc=from_loc, to_loc=to_loc, wafer_type=w_type)
            if rn in module_list:
                itv.end = INF_OCC
            
            _insert_interval_sorted(occ, itv)
            self.open_mod_occ[(rn, tok_key)] = itv
        
        # 生成 pst
        for p in pst_places:
            pst_place_type = new_marks[p].type
            
            if pst_place_type in [1, 2, 3]:
                if moved_tok is None:
                    raise RuntimeError(f"t={t_name} needs moved_tok but none from pre_places")
                moved_tok.enter_time = int(enter_new)
                new_marks[p].append(moved_tok)
            else:
                new_tok = BasedToken(enter_time=int(enter_new))
                new_marks[p].append(new_tok)
            
            new_m[p] += 1
        
        # 关闭区间
        leave_time = start_from
        from_chamber = get_close_resources(t_name, DEFAULT_MODULE_LIST)
        if len(from_chamber) > 0:
            tok_key = getattr(moved_tok, "job_id", -1) if moved_tok is not None else -1
            if "PICK" in t_name:
                leave_time += 5
            self.open_mod_occ[(from_chamber, tok_key)].end = leave_time
        
        # 更新全局时间
        if self.time_getter() < enter_new:
            self.time_setter(int(enter_new))
        
        finish = new_m[self.lp_done_idx] == self.n_wafer
        return FireResult(m=new_m, marks=new_marks, time=int(enter_new), finish=finish)
    
    def _record_op(self, t: int, marks: List[Place], te: int, dur: int, tf: int):
        """记录操作到 ops 列表（用于甘特图）"""
        t_name = self.id2t_name[t]
        module = t_name.split("__", 1)[1]
        
        # 获取 job_id
        job_id = -1
        pre_places = np.nonzero(self.pre[:, t] > 0)[0]
        for p in pre_places:
            if marks[p].type <= 3 and isinstance(marks[p].head(), WaferToken):
                job_id = marks[p].head().job_id
                break
        
        # 映射 stage/machine
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
            machine = int(module[2:]) - 1
        elif module == "LLD":
            stage = 4
            machine = 0
        elif module in ("PM9", "PM10"):
            stage = 5
            machine = 0 if module == "PM9" else 1
        
        if stage != -1 and job_id != -1:
            self.ops.append(Op(job=job_id, stage=stage, machine=machine,
                              start=te, proc_end=te + dur, end=tf))

    def fire_chain(self, chain_names: List[str], times: List[int], 
                   m: np.ndarray, marks: List[Place],
                   path_updater: Callable) -> FireResult:
        """
        连续发射一条链。
        
        Args:
            chain_names: 变迁名称列表
            times: 各变迁的开始时间
            m: 当前标记向量
            marks: 当前库所标记列表
            path_updater: 路径更新函数（更新 token.where）
        
        Returns:
            FireResult 包含最终状态
        """
        cur_m, cur_marks = m, marks
        cur_time = int(times[0])
        
        t_ids = [self.id2t_name.index(n) for n in chain_names]
        
        # 更新轮询索引
        t0_id = t_ids[0]
        t0_pre = np.nonzero(self.pre[:, t0_id] > 0)[0]
        
        self.action_checker.update_rr_after_fire(t0_id)
        
        # 更新 token 路径位置
        for p in t0_pre:
            if marks[p].type > 3:
                continue
            if isinstance(marks[p].head(), WaferToken):
                tok = marks[p].head()
                tok.where += 1
        
        # 按 times 逐个发射
        for i, t_id in enumerate(t_ids):
            te = int(times[i])
            result = self.fire(t_id, cur_m, cur_marks, te, record_ops=True)
            cur_m, cur_marks, cur_time = result.m, result.marks, result.time
        
        # 检查最终状态
        se = self.action_checker.resource_enable(cur_m)
        se = self.action_checker.color_enable(se, cur_marks)
        finish = cur_m[self.lp_done_idx] == self.n_wafer
        deadlock = (not finish) and (len(se) == 0)
        
        return FireResult(
            m=cur_m, marks=cur_marks, time=cur_time,
            finish=finish, deadlock=deadlock, mask=se
        )

    def dry_run_chain(self, chain_names: List[str], m: np.ndarray, 
                      marks: List[Place], earliest_time_func: Callable,
                      max_retry: int = 200) -> Tuple[bool, List[int], int, Any, Any]:
        """
        资源时间段验证（不实际发射）。
        
        Args:
            chain_names: 变迁名称列表
            m: 当前标记向量
            marks: 当前库所标记列表
            earliest_time_func: 获取最早使能时间的函数
            max_retry: 最大重试次数
        
        Returns:
            (ok, times, end_time, None, None)
        """
        # 工具函数
        def _struct_fire(m, t):
            return m - self.pre[:, t] + self.pst[:, t]
        
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
        
        t_ids = [self.id2t_name.index(n) for n in chain_names]
        
        # 结构性预检查
        tmp_m = m.copy()
        for t_id in t_ids:
            tmp_se = self.action_checker.resource_enable(tmp_m)
            if t_id in tmp_se:
                tmp_m = _struct_fire(tmp_m, t_id)
            else:
                return False, [], -1, None, None
        
        # 初始起点
        t0 = t_ids[0]
        base_start = int(earliest_time_func(t0, m, marks))
        shift0 = 0
        
        # 预处理
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
        
        last_proc_idx = None
        for i, (typ, _) in enumerate(blocks):
            if typ == "PROC":
                last_proc_idx = i
        
        # 重试循环
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
                        occ = self.res_occ.get(arm_resource, [])
                        t_free = _first_free_time_at(occ, s, e)
                        if t_free != s:
                            ok = False
                            need_shift = max(need_shift, t_free - s)
                            break
                else:
                    module = proc_resource[id(block)]
                    occ = self.res_occ.get(module, [])
                    if i == last_proc_idx:
                        t_free = _first_free_time_open(occ, s)
                    else:
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

    def earliest_enable_time(self, t: int, m, marks, start_from=None) -> int:
        """
        计算变迁的最早使能时间。
        
        Args:
            t: 变迁ID
            m: 当前标记向量
            marks: 当前库所标记列表
            start_from: 可选的起始时间下界
        
        Returns:
            最早使能时间
        """
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
                continue
            
            tok_ready = int(tok.enter_time)
            if tok_ready > t1:
                t1 = tok_ready
        
        # 2) 资源最早可用（可插入区间）
        t_name = self.id2t_name[t]
        res_names = get_transition_resources(t_name)
        t2 = int(self.resource_mgr.sync_start(res_names, t1, d))
        
        return t2

