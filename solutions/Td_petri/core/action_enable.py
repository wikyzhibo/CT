"""
动作使能检查模块。

该模块封装了 Petri 网中变迁使能检查的所有逻辑，
包括资源使能、颜色使能和轮询策略过滤。
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Any

from solutions.Td_petri.resources.machine_utils import PARALLEL_GROUPS, machine_group, parse_to_machine


class ActionEnableChecker:
    """
    动作使能检查器，封装所有变迁使能相关逻辑。
    
    该类负责：
    1. 基于标记的结构使能检查 (resource_enable)
    2. 基于 token 颜色的使能过滤 (color_enable)
    3. 轮询策略过滤并行组 (filter_by_round_robin)
    4. 发射后更新轮询索引 (update_rr_after_fire)
    
    Example:
        >>> checker = ActionEnableChecker(
        ...     pre=pre_matrix,
        ...     id2t_name=id2t_name,
        ...     parallel_groups={'PM1_4': ['PM1', 'PM2', 'PM3', 'PM4']},
        ...     path_getter=lambda tok: token_path
        ... )
        >>> enabled = checker.resource_enable(marking)
        >>> filtered = checker.filter_by_round_robin(enabled)
    """
    
    def __init__(self, 
                 pre: np.ndarray,
                 id2t_name: List[str],
                 parallel_groups: Dict[str, List[str]],
                 path_getter: Callable[[Any], List]):
        """
        初始化动作使能检查器。
        
        Args:
            pre: 前置矩阵 (P x T)
            id2t_name: 变迁 ID 到名称的映射列表
            parallel_groups: 并行组配置，如 {'PM1_4': ['PM1', 'PM2', ...]}
            path_getter: 获取 token 路径的函数，接受 token 返回路径列表
        """
        self.pre = pre
        self.id2t_name = id2t_name
        self.parallel_groups = parallel_groups
        self.path_getter = path_getter
        
        # 轮询索引
        self.rr_idx: Dict[str, int] = {g: -1 for g in parallel_groups}
    
    def reset(self):
        """重置轮询状态"""
        self.rr_idx = {g: -1 for g in self.parallel_groups}
    
    def resource_enable(self, m: np.ndarray) -> np.ndarray:
        """
        基于标记的结构使能检查。
        
        检查所有变迁，返回那些前置条件被满足的变迁。
        
        Args:
            m: 当前标记向量 (P,)
        
        Returns:
            使能变迁的 ID 数组
        """
        mask = (self.pre <= m[:, None]).all(axis=0)
        enable_t = np.nonzero(mask)[0]
        return enable_t
    
    def color_enable(self, se: np.ndarray, marks: List) -> List[Tuple[int, List]]:
        """
        基于 token 颜色的使能过滤。
        
        对每个资源使能的变迁 t，检查其前置库所里的 token 颜色（路径）
        是否允许该变迁。
        
        Args:
            se: 资源使能的变迁 ID 数组
            marks: 库所标记列表（Place 对象列表）
        
        Returns:
            [(t, chain), ...] 其中 chain 是该 token 的完整执行链
        """
        se_chain = []
        for t in se:
            pre_places = np.nonzero(self.pre[:, t] > 0)[0]
            for p in pre_places:
                place = marks[p]
                if place.type <= 3:
                    for tok in list(place.tokens):
                        # 使用 path_getter 获取路径
                        token_path = self.path_getter(tok)
                        if not token_path:
                            continue
                        
                        # 检查下一阶段是否包含当前变迁
                        for branch in token_path[0]:
                            if t in branch:
                                se_chain.append((t, branch))
                                break
        return se_chain
    
    def filter_by_round_robin(self, enable_ts: np.ndarray) -> List[int]:
        """
        使用轮询策略过滤使能变迁。
        
        确保同一并行组内的机台轮流使用。
        
        Args:
            enable_ts: 使能变迁的 ID 数组
        
        Returns:
            过滤后的使能变迁列表
        """
        # 组 -> 列表[(t_id, machine)]
        bucket: Dict[str, List[Tuple[int, str]]] = {}
        for t_id in enable_ts:
            t_name = self.id2t_name[t_id]
            m = parse_to_machine(t_name)
            if not m:
                continue
            g = machine_group(m)
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
    
    def update_rr_after_fire(self, t_id: int):
        """
        发射后更新轮询索引。
        
        Args:
            t_id: 刚刚发射的变迁 ID
        """
        t_name = self.id2t_name[t_id]
        m = parse_to_machine(t_name)
        if not m:
            return
        g = machine_group(m)
        if not g:
            return
        machines = PARALLEL_GROUPS[g]
        self.rr_idx[g] = machines.index(m)
