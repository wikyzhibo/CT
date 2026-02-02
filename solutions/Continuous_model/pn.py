"""
Petri 网调度环境（连续时间版本）。

该模块实现了一个双路线、双机械手协作的半导体制造调度 Petri 网环境，支持：
- 两条加工路线在 s1 分流（color=1 走路线1，color=2 走路线2）
- 有/无驻留约束腔室并存（s1/s3/s5 有驻留约束，s2/s4 无驻留约束）
- 释放时间追踪与链式违规检测（避免容量冲突）
- 奖励塑形与向量化优化计算
- 甘特图生成与晶圆滞留时间统计

主要入口：
- Petri: 环境类，支持 reset/step/get_enable_t/render_gantt
- Place: 库所类，支持 token 管理与释放时间追踪
"""

from collections import deque
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union
from visualization.plot import plot_gantt_hatched_residence, Op
from solutions.Continuous_model.construct import SuperPetriBuilder, ModuleSpec, RobotSpec, BasedToken
from data.petri_configs.env_config import PetriEnvConfig

INF = 10**6
MAX_TIME = 10000  # 例如 300s

# 双机械手变迁映射：TM2/TM3 各自控制的变迁名称
TM2_TRANSITIONS = frozenset({
    "u_LP1_s1", "u_LP2_s1", "u_s1_s2", "u_s1_s5", "u_s4_s5", "u_s5_LP_done",
    "t_s1", "t_s2", "t_s5", "t_LP_done"
})
TM3_TRANSITIONS = frozenset({
    "u_s2_s3", "u_s3_s4", "t_s3", "t_s4"
})


@dataclass(slots=False)  # 不能使用 slots=True，因为 tokens 和 release_schedule 是可变字段（deque）
class Place:
    """
    Petri 网库所。

    属性说明：
    - name: 库所名称
    - capacity: 容量
    - processing_time: 加工时间（库所驻留时间）
    - type:
        1=加工腔室（有驻留约束）
        2=运输库所
        3=空闲库所（LP）
        4=资源库所（机械手）
        5=无驻留约束腔室（如 s2/LLC、s4/LLD）
    - tokens: token 队列（FIFO）
    - release_schedule: 释放时间追踪队列 [(token_id, release_time), ...]
    - last_machine: 上次分配的机器编号（type=1 使用）
    """
    # 注意：不能使用 __slots__，因为 tokens 和 release_schedule 是可变字段（deque）
    # 但可以通过其他方式优化属性访问（如使用局部变量缓存）
    name: str
    capacity: int
    processing_time: int
    type: int  # 1 manipulator place, 2 delivery place, 3 idle place, 4 source/other
    tokens: Deque[BasedToken] = field(default_factory=deque)
    # 晶圆释放时间追踪队列：[(token_id, release_time), ...]
    release_schedule: Deque[Tuple[int, int]] = field(default_factory=deque)
    # 上次分配的机器编号（仅 type=1 使用），用于轮换分配机器
    last_machine: int = -1

    def clone(self) -> "Place":
        cloned = Place(
            name=self.name,
            capacity=self.capacity,
            processing_time=self.processing_time,
            type=self.type,
            last_machine=self.last_machine,
        )
        cloned.tokens = deque(tok.clone() for tok in self.tokens)
        cloned.release_schedule = deque(self.release_schedule)  # 复制释放时间队列
        return cloned

    def head(self) -> BasedToken:
        return self.tokens[0]

    def pop_head(self) -> BasedToken:
        return self.tokens.popleft()

    def append(self, token: BasedToken) -> None:
        self.tokens.append(token)

    def res_time(self, current_time: int, P_Residual_time: int = 15, D_Residual_time: int = 10) -> int:
        """返回当前库所内wafer的剩余超时时间（示例逻辑，和你的原实现一致）"""
        if len(self.tokens) == 0:
            return 10**5
        else:
            if self.type == 1:  # process chamber
                res_time = self.head().enter_time + self.processing_time + P_Residual_time - current_time
            elif self.type == 2:  # transport place
                res_time = self.head().enter_time + 5 + D_Residual_time - current_time
            else:
                return 10**5
            return -1 if res_time < 0 else int(res_time)

    def __len__(self) -> int:
        return len(self.tokens)
    
    # ========== 释放时间追踪方法 ==========
    def add_release(self, token_id: int, release_time: int) -> None:
        """添加晶圆的预估释放时间"""
        self.release_schedule.append((token_id, release_time))
    
    def update_release(self, token_id: int, new_release_time: int) -> None:
        """更新指定晶圆的释放时间"""
        for i, (tid, _) in enumerate(self.release_schedule):
            if tid == token_id:
                self.release_schedule[i] = (token_id, new_release_time)
                return
    
    def pop_release(self, token_id: int) -> Optional[int]:
        """晶圆离开时移除记录，返回释放时间"""
        for i, (tid, rt) in enumerate(self.release_schedule):
            if tid == token_id:
                del self.release_schedule[i]
                return rt
        return None

    def get_release(self, token_id: int) -> Optional[int]:
        """查询指定晶圆的释放时间（不移除）"""
        for tid, rt in self.release_schedule:
            if tid == token_id:
                return rt
        return None
    
    def earliest_release(self) -> Optional[int]:
        """
        返回队列中最早的释放时间
        
        优化：使用生成器表达式和 min()，对于小规模数据（<10个元素）这是最高效的方式。
        如果 release_schedule 元素较多，可以考虑使用堆（heapq）优化。
        """
        if not self.release_schedule:
            return None
        # 对于小规模数据，min() 已经足够高效
        # 如果 release_schedule 元素较多（>10），可以考虑使用 heapq
        return min(rt for _, rt in self.release_schedule)


class Petri:
    def __init__(self, config: Optional[PetriEnvConfig] = None,
                 stop_on_scrap: Optional[bool] = None,
                 training_phase: Optional[int] = None,
                 reward_config: Optional[Dict[str, int]] = None,
                 enable_statistics: bool = False) -> None:
        """
        初始化 Petri 网环境。

        架构要点：
        - 双路线：LP1 -> s1 -> s2 -> s3 -> s4 -> s5 -> LP_done
                  LP2 -> s1 -> s5 -> LP_done
        - 双机械手：TM2 负责 LP1/LP2/s1/s2/s4/s5/LP_done，TM3 负责 s2/s3/s4
        - color 路线分流：color=1 走路线1，color=2 走路线2
        
        Args:
            config: PetriEnvConfig配置对象（推荐使用）
            stop_on_scrap: 报废时是否停止（如果config为None时使用）
            training_phase: 训练阶段（如果config为None时使用）
            reward_config: 奖励配置（如果config为None时使用）
        """
        # -----------------------
        # 1) 加载或创建配置
        # -----------------------
        if config is None:
            # 使用默认配置或传入的参数
            kwargs = {}
            if stop_on_scrap is not None:
                kwargs['stop_on_scrap'] = stop_on_scrap
            if training_phase is not None:
                kwargs['training_phase'] = training_phase
            if reward_config is not None:
                kwargs['reward_config'] = reward_config
            config = PetriEnvConfig(**kwargs)
        
        # 保存配置
        self.config = config
        
        # 将配置参数设置为实例属性（保持向后兼容）
        self.n_wafer = config.n_wafer
        self.c_time = config.c_time
        self.R_done = config.R_done
        self.R_finish = getattr(config, 'R_finish', 800)  # 全部完工大奖励
        self.R_scrap = config.R_scrap
        self.T_warn = config.T_warn
        self.a_warn = config.a_warn
        self.T_safe = config.T_safe
        self.b_safe = config.b_safe
        self.MAX_WAIT_STEP = config.MAX_WAIT_STEP
        self.c_congest = config.c_congest
        self.D_Residual_time = config.D_Residual_time
        self.P_Residual_time = config.P_Residual_time
        self.c_release_violation = config.c_release_violation
        self.T_transport = config.T_transport
        self.T_load = config.T_load
        self.T_pm1_to_pm2 = config.T_pm1_to_pm2
        self.idle_timeout = config.idle_timeout
        self.idle_penalty = config.idle_penalty
        self.stop_on_scrap = config.stop_on_scrap
        self.training_phase = config.training_phase
        self.reward_config = config.reward_config
        
        # ============ 性能优化配置 ============
        self.optimize_reward_calc = getattr(config, 'optimize_reward_calc', False)
        self.optimize_enable_check = getattr(config, 'optimize_enable_check', False)
        self.optimize_state_update = getattr(config, 'optimize_state_update', False)
        self.cache_indices = getattr(config, 'cache_indices', False)
        self.optimize_data_structures = getattr(config, 'optimize_data_structures', False)
        
        # 内部状态
        self.scrap_count = 0  # 报废计数器
        self._idle_penalty_applied = False  # 标记是否已施加过停滞惩罚
        self._consecutive_wait_time = 0  # 连续执行 WAIT 动作的累计时间
        self._per_wafer_reward = 0.0  # 累积的单片完工奖励
        
        # 晶圆进入系统限制
        self.entered_wafer_count = 0  # 已进入系统的晶圆数
        self.done_count = 0           # 已完成的晶圆数 (用于进度条和统计)
        self.max_wafers_in_system = config.max_wafers_in_system  # 最大允许进入系统的晶圆数

        self.shot = "s"

        # -----------------------
        # 2) 构建 Petri 网
        # -----------------------
        # 双路线支持：
        # 路线1：LP1 -> s1(70) -> s2(0) -> s3(600) -> s4(70) -> s5(200) -> LP_done
        # 路线2：LP2 -> s1(70) -> s5(200) -> LP_done
        # 在 s1 处根据晶圆颜色分流：颜色1走 s2，颜色2走 s5
        # 双机械手协作：TM2 负责 LP1/LP2/s1/s2放入/s4取出/s5/LP_done，TM3 负责 s2取出/s3/s4放入
        
        # 晶圆数量分配（可配置）
        # 注意：config 中 n_wafer_route1/2 可能为 None，需要检查
        _route1 = getattr(config, 'n_wafer_route1', None)
        _route2 = getattr(config, 'n_wafer_route2', None)
        self.n_wafer_route1 = _route1 if _route1 is not None else self.n_wafer // 2
        self.n_wafer_route2 = _route2 if _route2 is not None else self.n_wafer - self.n_wafer // 2
        
        modules = {
            "LP1": ModuleSpec(tokens=self.n_wafer_route1, ptime=0, capacity=self.n_wafer_route1),  # 路线1晶圆
            "LP2": ModuleSpec(tokens=self.n_wafer_route2, ptime=0, capacity=self.n_wafer_route2),  # 路线2晶圆
            "LP_done": ModuleSpec(tokens=0, ptime=0, capacity=self.n_wafer),
            "s1": ModuleSpec(tokens=0, ptime=70, capacity=2),    # PM7/PM8，有驻留约束
            "s2": ModuleSpec(tokens=0, ptime=0, capacity=1),     # LLC，无驻留约束（缓冲）
            "s3": ModuleSpec(tokens=0, ptime=600, capacity=4),   # PM1/PM2，有驻留约束，容量增加到4
            "s4": ModuleSpec(tokens=0, ptime=70, capacity=1),    # LLD，有加工时间但无驻留约束
            "s5": ModuleSpec(tokens=0, ptime=200, capacity=2),   # PM9/PM10，有驻留约束
        }
        robots = {
            "TM2": RobotSpec(tokens=1, reach={"LP1", "LP2", "s1", "s2", "s4", "s5", "LP_done"}),
            "TM3": RobotSpec(tokens=1, reach={"s2", "s3", "s4"}),
        }
        # 两条路线：在 s1 处分流
        routes = [
            ["LP1", "s1", "s2", "s3", "s4", "s5", "LP_done"],  # 路线1：完整路径
            ["LP2", "s1", "s5", "LP_done"],                     # 路线2：跳过 s2/s3/s4
        ]

        builder = SuperPetriBuilder(d_ptime=5, default_ttime=5)
        info = builder.build(modules=modules, robots=robots, routes=routes)
        
        # ============ 路线配置：定义每条路线的目标腔室序列 ============
        # 用于 step-based 索引过滤：Token 的 (route_type, step) 决定下一个目标
        # step 表示已完成的步骤数，从 0 开始
        # step=0 表示在 LP，目标是 routes[route_type-1][1]（第一个非起点腔室）
        self.ROUTE_CONFIG = {
            1: ["s1", "s2", "s3", "s4", "s5", "LP_done"],  # 路线1: 6步
            2: ["s1", "s5", "LP_done"],                     # 路线2: 3步
        }

        # -----------------------
        # 2) 保存网络结构
        # -----------------------
        self.pre = info["pre"]
        self.pst = info["pst"]
        self.net = self.pst - self.pre

        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]

        self.m0 = info["m0"]
        self.m = self.m0.copy()

        self.md = info["md"]
        self.k = info["capacity"]
        self.ptime = info["ptime"]

        # 这里你原来写死 ttime=5
        self.ttime = 5

        self.id2p_name = info["id2p_name"]
        self.id2t_name = info["id2t_name"]
        self.idle_idx = info["idle_idx"]

        self.ori_marks = info["marks"]
        self.marks = self._clone_marks(self.ori_marks)
        
        # 修正无驻留约束腔室的 type 为 5（s2/LLC 和 s4/LLD）
        no_constraint_places = {"s2", "s4"}
        for place in self.marks:
            if place.name in no_constraint_places:
                place.type = 5
        for place in self.ori_marks:
            if place.name in no_constraint_places:
                place.type = 5

        # petri网系统时钟
        self.time = 1
        self.fire_log = []

        # 构建上下游关系映射（用于堵塞检测，当前已关闭）
        self.downstream_map = {}  # 堵塞检测已关闭，保留空映射

        # 错峰启动：记录 t_PM1 上次发射时间
        self._last_t_pm1_fire_time = -INF
        
        # ============ 性能优化：构建缓存的索引映射（必须在 _build_release_chain 之前）============
        if self.cache_indices:
            self._build_cached_indices()
        else:
            self._place_indices = None
            self._transition_indices = None
        
        # 缓存加工腔室列表（type=1）
        # 注意：如果启用了数据结构优化，这个缓存将被 _marks_by_type[1] 替代
        self._process_places_cache = [p for p in self.marks if p.type == 1]
        
        # 释放时间追踪：构建加工腔室链路映射
        # release_chain[place_idx] = (downstream_place_idx, transport_time)
        # 例如：PM1 -> PM2，运输时间为 T_pm1_to_pm2
        self._build_release_chain()
        
        # ============ 数据结构优化：构建按类型分组的 marks 列表缓存 ============
        if self.optimize_data_structures:
            self._build_marks_by_type_cache()
        
        # 累计的释放时间违规惩罚（每个 step 计算后清零）
        self._release_violation_penalty = 0.0
        
        # 晶圆滞留时间统计追踪
        # {token_id: {enter_system, exit_system, chambers: {name: {enter, exit}}, transports: {name: {enter, exit}}}}
        self.wafer_stats: Dict[int, Dict[str, Any]] = {}
        
        # 可视化统计开关（训练模式下为 False，避免性能开销）
        self.enable_statistics = enable_statistics

    @staticmethod
    def _clone_marks(marks: List["Place"]) -> List[Place]:
        """克隆库所列表，确保使用 pn.py 中的 Place 类（带 release_schedule）"""
        cloned_list = []
        for p in marks:
            # 使用 pn.py 中的 Place 类创建新对象
            cloned = Place(
                name=p.name,
                capacity=p.capacity,
                processing_time=p.processing_time,
                type=p.type,
                last_machine=getattr(p, 'last_machine', -1),
            )
            cloned.tokens = deque(tok.clone() for tok in p.tokens)
            cloned_list.append(cloned)
        return cloned_list

    def _build_downstream_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        构建上下游关系映射。
        
        通过分析 Petri 网结构，找出每个加工腔室(type=1)的下游腔室。
        路径：上游腔室 -> u_变迁 -> d_库所 -> t_变迁 -> 下游腔室
        
        Returns:
            Dict[upstream_idx, List[(downstream_idx, d_place_idx)]]
            例如: {PM1_idx: [(PM2_idx, d_PM2_idx)]}
        """
        downstream_map: Dict[int, List[Tuple[int, int]]] = {}
        
        # 找出所有加工腔室（type=1）
        process_places = [i for i, p in enumerate(self.marks) if p.type == 1]
        
        for up_idx in process_places:
            downstream_map[up_idx] = []
            
            # 找从 up_idx 出发的变迁（u_变迁）
            # pre[up_idx, t] > 0 表示 up_idx 是变迁 t 的前置库所
            out_trans = np.flatnonzero(self.pre[up_idx, :] > 0)
            
            for t in out_trans:
                t_name = self.id2t_name[t]
                # 只关注 u_ 开头的变迁（搬运变迁）
                if not t_name.startswith("u_"):
                    continue
                
                # 找 u_变迁 的后置库所（d_库所）
                d_places = np.flatnonzero(self.pst[:, t] > 0)
                
                for d_idx in d_places:
                    d_name = self.id2p_name[d_idx]
                    # 只关注 d_ 开头的运输库所
                    if not d_name.startswith("d_"):
                        continue
                    
                    # 找从 d_库所 出发的变迁（t_变迁）
                    t_trans = np.flatnonzero(self.pre[d_idx, :] > 0)
                    
                    for t2 in t_trans:
                        # 找 t_变迁 的后置库所（下游腔室）
                        down_places = np.flatnonzero(self.pst[:, t2] > 0)
                        
                        for down_idx in down_places:
                            # 检查是否是加工腔室或终点
                            if self.marks[down_idx].type == 1 or self.id2p_name[down_idx].endswith("_done"):
                                downstream_map[up_idx].append((int(down_idx), int(d_idx)))
        
        return downstream_map

    def _build_cached_indices(self) -> None:
        """
        构建缓存的库所和变迁索引映射，用于快速查找。
        
        优化：避免每次调用 id2p_name.index() 和 id2t_name.index() 的线性查找开销。
        """
        self._place_indices = {name: idx for idx, name in enumerate(self.id2p_name)}
        self._transition_indices = {name: idx for idx, name in enumerate(self.id2t_name)}
    
    def _get_place_index(self, name: str) -> int:
        """获取库所索引（使用缓存如果可用）"""
        if self.cache_indices and self._place_indices is not None:
            return self._place_indices[name]
        return self.id2p_name.index(name)
    
    def _get_transition_index(self, name: str) -> int:
        """获取变迁索引（使用缓存如果可用）"""
        if self.cache_indices and self._transition_indices is not None:
            return self._transition_indices[name]
        return self.id2t_name.index(name)
    
    def _build_marks_by_type_cache(self) -> None:
        """
        构建按类型分组的 marks 列表缓存。
        
        优化：避免频繁遍历所有库所查找特定类型的库所。
        
        注意：如果优化失败，会回退到空缓存，不影响功能。
        """
        try:
            self._marks_by_type: Dict[int, List[Place]] = {}
            for place in self.marks:
                place_type = place.type
                if place_type not in self._marks_by_type:
                    self._marks_by_type[place_type] = []
                self._marks_by_type[place_type].append(place)
        except Exception:
            # 如果构建缓存失败，使用空缓存，回退到遍历方式
            self._marks_by_type = {}
    
    def _get_marks_by_type(self, place_type: int) -> List[Place]:
        """
        获取指定类型的库所列表（使用缓存）。
        
        Args:
            place_type: 库所类型（1=加工腔室, 2=运输库所, 3=空闲库所, 4=资源库所, 5=无驻留约束腔室）
            
        Returns:
            指定类型的库所列表
        """
        # 优化：直接访问缓存，避免 hasattr 检查开销
        # 如果启用了数据结构优化，直接使用缓存（缓存在 __init__ 中保证存在）
        if self.optimize_data_structures:
            return self._marks_by_type.get(place_type, [])
        # 如果优化未启用，回退到遍历方式
        return [p for p in self.marks if p.type == place_type]
    
    def _get_next_target(self, route_type: int, step: int) -> Optional[str]:
        """
        根据 (route_type, step) 获取下一个目标腔室名称。
        
        Args:
            route_type: 路线类型 (1 或 2)
            step: 当前工序步骤索引（0-based，表示已完成的步骤数）
            
        Returns:
            目标腔室名称，如果步骤超出范围则返回 None
        """
        route = self.ROUTE_CONFIG.get(route_type, [])
        if step < len(route):
            return route[step]
        return None
    
    def _can_enter_target(self, tok: BasedToken, target: str) -> bool:
        """
        检查 Token 是否可以进入指定目标腔室。
        
        Args:
            tok: Token 对象
            target: 目标腔室名称
            
        Returns:
            True 如果 Token 的 (route_type, step) 允许进入该目标
        """
        expected_target = self._get_next_target(tok.route_type, tok.step)
        return expected_target == target
    
    def _build_release_chain(self) -> None:
        """
        构建释放时间链路映射。
        
        release_chain[place_idx] = (downstream_place_idx, transport_time)
        用于链式更新下游腔室的预估释放时间。
        
        双路线：
        路线1：LP1 -> s1 -> s2 -> s3 -> s4 -> s5 -> LP_done
        路线2：LP2 -> s1 -> s5 -> LP_done
        
        有驻留约束腔室：s1, s3, s5
        无驻留约束腔室：s2, s4（作为机械手交接点）
        
        注意：由于双路线在 s1 分流，release_chain 需要按颜色区分。
        路线1的链路：s1 -> s3 -> s4 -> s5（s4 已纳入释放链，接受链式惩罚）。
        路线2的链路：s1 -> s5（在运行时处理）。
        """
        self.release_chain: Dict[int, Tuple[int, int]] = {}
        
        # 获取腔室索引（如果存在）
        def get_idx(name: str) -> int:
            if name in self.id2p_name:
                return self._get_place_index(name) if self.cache_indices else self.id2p_name.index(name)
            return -1
        
        s1_idx = get_idx("s1")
        s3_idx = get_idx("s3")
        s4_idx = get_idx("s4")
        s5_idx = get_idx("s5")
        
        # 路线1的链路（有驻留约束的腔室之间）
        # s1 -> s3：经过 s2（交接点），预估运输时间
        if s1_idx >= 0 and s3_idx >= 0:
            # 运输时间 = s1卸载 + d_s2运输 + s2装载 + s2卸载 + d_s3运输 + s3装载
            transport_s1_to_s3 = self.T_transport * 2 + self.T_load * 4  # 约 30s
            self.release_chain[s1_idx] = (s3_idx, transport_s1_to_s3)
        
        # s3 -> s4 -> s5：将 s4 纳入释放链，使其接受链式惩罚
        # s3 -> s4：预估运输时间
        if s3_idx >= 0 and s4_idx >= 0:
            # 运输时间 = s3卸载 + d_s4运输 + s4装载
            transport_s3_to_s4 = self.ttime * 3  # 约 15s
            self.release_chain[s3_idx] = (s4_idx, transport_s3_to_s4)
        
        # s4 -> s5：预估运输时间
        # 注意：release_s4 已经是 s4 的释放时间（= 进入时间 + 加工时间 70s）
        # 所以运输时间只需要：s4卸载 + d_s5运输 + s5装载，不需要再加 70s
        if s4_idx >= 0 and s5_idx >= 0:
            # 运输时间 = s4卸载 + d_s5运输 + s5装载
            transport_s4_to_s5 = self.ttime * 3  # 约 15s
            self.release_chain[s4_idx] = (s5_idx, transport_s4_to_s5)
        
        # 路线2的链路：s1 -> s5（直接）
        # 存储为单独的映射，在运行时根据颜色选择
        self.release_chain_route2: Dict[int, Tuple[int, int]] = {}
        if s1_idx >= 0 and s5_idx >= 0:
            # 运输时间 = s1卸载 + d_s5运输 + s5装载
            transport_s1_to_s5 = self.ttime * 2  # 约 10s
            self.release_chain_route2[s1_idx] = (s5_idx, transport_s1_to_s5)

    def _check_release_violation(self, place_idx: int, expected_enter_time: int) -> Tuple[float, int]:
        """
        检查晶圆预计进入时间是否违反腔室的释放约束，并返回修正后的进入时间。
        
        容量检查逻辑：
        - release_schedule：记录**预估中**即将进入腔室的晶圆（已承诺但未到达）
        - tokens：记录**实际已在**腔室中的晶圆
        - 实际占用 = len(tokens) + len(release_schedule)（两者不重叠，因为进入腔室后会从 schedule 移除）
        
        违规条件：实际占用 >= capacity 且 expected_enter_time < earliest_release
        如果违规，将 expected_enter_time 修正为 earliest_release，确保后续晶圆看到正确的释放时间。
        
        Args:
            place_idx: 目标腔室索引
            expected_enter_time: 预计进入时间
            
        Returns:
            (penalty: float, corrected_enter_time: int)
            - penalty: 惩罚值（0 表示不违规）
            - corrected_enter_time: 修正后的进入时间（如果违规则为 earliest_release，否则为 expected_enter_time）
        """
        place = self.marks[place_idx]
        
        # 计算实际占用：已在腔室中的 + 预估中即将进入的
        # 注意：tokens 和 release_schedule 不重叠
        # - 晶圆进入运输位时加入 release_schedule
        # - 晶圆进入腔室时从 release_schedule 移除，同时加入 tokens
        # - 但存在时间窗口：晶圆在运输中时同时存在于两者？
        # 设计澄清：release_schedule 记录的是「预计进入」，晶圆实际进入腔室（t_s* 变迁）时
        # 会调用 _update_release 而非 _pop_release，所以不会重复计算
        # 实际上 release_schedule 的记录会在晶圆离开腔室时才 pop
        # 因此：当晶圆在腔室中时，它同时存在于 tokens 和 release_schedule 中！
        # 所以检查应该只用 release_schedule，因为它覆盖了所有承诺（包括已进入的）
        actual_occupancy = len(place.release_schedule)
        
        # 如果占用未满，不违规
        if actual_occupancy < place.capacity:
            return (0.0, expected_enter_time)
        
        earliest = place.earliest_release()
        if earliest is None or expected_enter_time >= earliest:
            return (0.0, expected_enter_time)  # 不违规
        
        # 违规：计算惩罚，并修正进入时间为 earliest_release
        violation_gap = earliest - expected_enter_time
        penalty = self.c_release_violation * 100
        corrected_enter_time = earliest
        return (penalty, corrected_enter_time)

    def _record_initial_release(self, token_id: int, enter_d_time: int,
                                 target_place_idx: int, wafer_color: int = 0,
                                 chain_downstream: bool = False) -> float:
        """
        晶圆进入运输位时，记录初步预估的释放时间；可选链式传播到下游腔室。

        按「离开才检查」规则：离开 LP 时只对 s1 记录并检查，不链式传播。
        
        Args:
            token_id: 晶圆编号
            enter_d_time: 进入运输位的时间
            target_place_idx: 目标腔室索引
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
            chain_downstream: 是否链式传播到下游（默认 False，仅记录 target）
            
        Returns:
            违规惩罚值
        """
        target_place = self.marks[target_place_idx]
        
        # 计算预估进入时间和释放时间
        expected_enter = enter_d_time + self.T_transport + self.T_load
        
        penalty = 0.0
        corrected_enter = expected_enter
        if self.reward_config.get('release_violation_penalty', 1):
            penalty, corrected_enter = self._check_release_violation(target_place_idx, expected_enter)
        
        # 使用修正后的进入时间重新计算释放时间
        release_time = corrected_enter + target_place.processing_time
        target_place.add_release(token_id, release_time)
        
        if chain_downstream:
            penalty += self._chain_record_release(token_id, target_place_idx, release_time, wafer_color)
        
        return penalty

    def _chain_record_release(self, token_id: int, start_place_idx: int, 
                               start_release_time: int, wafer_color: int = 0) -> float:
        """
        链式记录/更新下游腔室的预估释放时间。
        
        对链上的每个下游腔室（包括 s4）进行释放违规检查并累加惩罚。
        
        Args:
            token_id: 晶圆编号
            start_place_idx: 起始腔室索引
            start_release_time: 起始腔室的释放时间
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
            
        Returns:
            下游违规惩罚值
        """
        penalty = 0.0
        current_idx = start_place_idx
        current_release = start_release_time
        
        # 根据颜色选择释放链路
        # 路线1（color=1）：使用 release_chain（s1 -> s3 -> s4 -> s5）
        # 路线2（color=2）：使用 release_chain_route2（s1 -> s5）
        if wafer_color == 2 and hasattr(self, 'release_chain_route2'):
            chain = self.release_chain_route2
        else:
            chain = self.release_chain
        
        while current_idx in chain:
            downstream_idx, transport_time = chain[current_idx]
            downstream_place = self.marks[downstream_idx]
            
            # 下游预估进入时间 = 当前释放时间 + 运输时间
            downstream_enter = current_release + transport_time
            
            # 检查下游违规并获取修正后的进入时间
            corrected_downstream_enter = downstream_enter
            if self.reward_config.get('release_violation_penalty', 1):
                downstream_penalty, corrected_downstream_enter = self._check_release_violation(downstream_idx, downstream_enter)
                penalty += downstream_penalty
            
            # 使用修正后的进入时间重新计算释放时间
            downstream_release = corrected_downstream_enter + downstream_place.processing_time
            
            # 记录下游释放时间
            downstream_place.add_release(token_id, downstream_release)
            
            # 继续链式传播
            current_idx = downstream_idx
            current_release = downstream_release
        
        return penalty

    def _update_release(self, token_id: int, actual_enter_time: int,
                        place_idx: int, wafer_color: int = 0,
                        chain_downstream: bool = True) -> None:
        """
        晶圆实际进入腔室时，更新精确的释放时间；可选链式更新下游。
        
        注意：更新操作不进行违规检测。按「离开才检查」规则，t_s1 时下游尚未写入，只更新 s1。
        
        Args:
            token_id: 晶圆编号
            actual_enter_time: 实际进入时间
            place_idx: 腔室索引
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
            chain_downstream: 是否链式更新下游（默认 True；t_s1 时传 False）
        """
        place = self.marks[place_idx]
        new_release_time = actual_enter_time + place.processing_time
        place.update_release(token_id, new_release_time)
        if chain_downstream:
            self._chain_update_release(token_id, place_idx, new_release_time, wafer_color)

    def _chain_update_release(self, token_id: int, start_place_idx: int, 
                               start_release_time: int, wafer_color: int = 0) -> None:
        """
        链式更新下游腔室中指定晶圆的预估释放时间。
        
        重要：这是更新已有记录，不是添加新记录，因此：
        1. 不调用 _check_release_violation（当前晶圆自己的记录不应约束自己）
        2. 不施加惩罚
        
        Args:
            token_id: 晶圆编号
            start_place_idx: 起始腔室索引
            start_release_time: 起始腔室的新释放时间
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
        """
        current_idx = start_place_idx
        current_release = start_release_time
        
        # 根据颜色选择释放链路
        if wafer_color == 2 and hasattr(self, 'release_chain_route2'):
            chain = self.release_chain_route2
        else:
            chain = self.release_chain
        
        while current_idx in chain:
            downstream_idx, transport_time = chain[current_idx]
            downstream_place = self.marks[downstream_idx]
            
            # 下游预估进入时间 = 当前释放时间 + 运输时间
            downstream_enter = current_release + transport_time
            
            # 重要修复：更新已有记录时，不调用 _check_release_violation
            # 因为当前晶圆自己的记录已经在 release_schedule 中，
            # 把它自己的释放时间当作约束会导致错误的修正
            # 违规检查只在首次添加记录（add_release）时才有意义
            
            # 直接使用预估进入时间计算释放时间
            downstream_release = downstream_enter + downstream_place.processing_time
            
            # 更新下游释放时间
            downstream_place.update_release(token_id, downstream_release)
            
            # 继续链式传播
            current_idx = downstream_idx
            current_release = downstream_release

    def _pop_release(self, token_id: int, place_idx: int) -> None:
        """
        晶圆离开腔室时，从释放队列中移除记录。
        
        Args:
            token_id: 晶圆编号
            place_idx: 腔室索引
        """
        place = self.marks[place_idx]
        place.pop_release(token_id)

    def _track_wafer_statistics(self, t_name: str, wafer_id: int, 
                                 start_time: int, enter_new: int) -> None:
        """
        追踪晶圆在各腔室和运输位的滞留时间。
        
        双路线：
        路线1：LP1 -> s1(PM7/PM8) -> s2(LLC) -> s3(PM1/PM2) -> s4(LLD) -> s5(PM9/PM10) -> LP_done
        路线2：LP2 -> s1(PM7/PM8) -> s5(PM9/PM10) -> LP_done
        运输位：d_s1, d_s2, d_s3, d_s4, d_s5, d_LP_done
        
        Args:
            t_name: 变迁名称
            wafer_id: 晶圆编号
            start_time: 变迁开始时间
            enter_new: 变迁完成时间（token 进入后置库所的时间）
        """
        if wafer_id < 0:
            return
        
        # 初始化晶圆统计记录
        if wafer_id not in self.wafer_stats:
            self.wafer_stats[wafer_id] = {
                "enter_system": None,
                "exit_system": None,
                "chambers": {},  # {chamber_name: {"enter": int, "exit": int}}
                "transports": {},  # {transport_name: {"enter": int, "exit": int}}
            }
        
        stats = self.wafer_stats[wafer_id]
        
        # ========== 系统进入/退出追踪 ==========
        # 双路线：u_LP1_s1 和 u_LP2_s1 都进入系统
        if t_name in ("u_LP1_s1", "u_LP2_s1"):
            # 晶圆离开 LP1/LP2，进入系统（进入 d_s1 运输位）
            stats["enter_system"] = start_time
            stats["transports"]["d_s1"] = {"enter": enter_new, "exit": None}
        
        elif t_name == "t_LP_done":
            # 晶圆进入 LP_done，离开系统
            stats["exit_system"] = enter_new
        
        # ========== 腔室进入/退出追踪 ==========
        # s1 (PM7/PM8)
        elif t_name == "t_s1":
            # 晶圆进入 s1 腔室
            stats["chambers"]["s1"] = {"enter": enter_new, "exit": None}
            # 离开 d_s1 运输位
            if "d_s1" in stats["transports"]:
                stats["transports"]["d_s1"]["exit"] = start_time
        
        elif t_name == "u_s1_s2":
            # 路线1：晶圆离开 s1 腔室，进入 d_s2 运输位
            if "s1" in stats["chambers"]:
                stats["chambers"]["s1"]["exit"] = start_time
            stats["transports"]["d_s2"] = {"enter": enter_new, "exit": None}
        
        elif t_name == "u_s1_s5":
            # 路线2：晶圆离开 s1 腔室，直接进入 d_s5 运输位
            if "s1" in stats["chambers"]:
                stats["chambers"]["s1"]["exit"] = start_time
            stats["transports"]["d_s5"] = {"enter": enter_new, "exit": None}
        
        # s2 (LLC) - 缓冲区，无驻留约束（仅路线1）
        elif t_name == "t_s2":
            # 晶圆进入 s2 缓冲区
            stats["chambers"]["s2"] = {"enter": enter_new, "exit": None}
            if "d_s2" in stats["transports"]:
                stats["transports"]["d_s2"]["exit"] = start_time
        
        elif t_name == "u_s2_s3":
            # 晶圆离开 s2，进入 d_s3 运输位
            if "s2" in stats["chambers"]:
                stats["chambers"]["s2"]["exit"] = start_time
            stats["transports"]["d_s3"] = {"enter": enter_new, "exit": None}
        
        # s3 (PM1/PM2)（仅路线1）
        elif t_name == "t_s3":
            # 晶圆进入 s3 腔室
            stats["chambers"]["s3"] = {"enter": enter_new, "exit": None}
            if "d_s3" in stats["transports"]:
                stats["transports"]["d_s3"]["exit"] = start_time
        
        elif t_name == "u_s3_s4":
            # 晶圆离开 s3 腔室，进入 d_s4 运输位
            if "s3" in stats["chambers"]:
                stats["chambers"]["s3"]["exit"] = start_time
            stats["transports"]["d_s4"] = {"enter": enter_new, "exit": None}
        
        # s4 (LLD) - 有加工时间但无驻留约束（仅路线1）
        elif t_name == "t_s4":
            # 晶圆进入 s4 腔室
            stats["chambers"]["s4"] = {"enter": enter_new, "exit": None}
            if "d_s4" in stats["transports"]:
                stats["transports"]["d_s4"]["exit"] = start_time
        
        elif t_name == "u_s4_s5":
            # 路线1：晶圆离开 s4，进入 d_s5 运输位
            if "s4" in stats["chambers"]:
                stats["chambers"]["s4"]["exit"] = start_time
            stats["transports"]["d_s5"] = {"enter": enter_new, "exit": None}
        
        # s5 (PM9/PM10)（两条路线都经过）
        elif t_name == "t_s5":
            # 晶圆进入 s5 腔室
            stats["chambers"]["s5"] = {"enter": enter_new, "exit": None}
            if "d_s5" in stats["transports"]:
                stats["transports"]["d_s5"]["exit"] = start_time
        
        elif t_name == "u_s5_LP_done":
            # 晶圆离开 s5 腔室，进入 d_LP_done 运输位
            if "s5" in stats["chambers"]:
                stats["chambers"]["s5"]["exit"] = start_time
            stats["transports"]["d_LP_done"] = {"enter": enter_new, "exit": None}
        
        # 最终进入 LP_done
        # t_LP_done 已在上面处理 exit_system

    def calc_wafer_statistics(self) -> Dict[str, Any]:
        """
        计算晶圆滞留时间统计数据。
        
        Returns:
            Dict 包含以下键：
            - system_avg: 平均系统滞留时间（从进入到离开）
            - system_max: 最大系统滞留时间
            - completed_count: 已完成晶圆数
            - in_progress_count: 进行中晶圆数
            - chambers: {chamber_name: {"avg": float, "max": int, "count": int}}
            - transports: {"avg": float, "max": int, "count": int}  # 所有运输位合并
            - transports_detail: {transport_name: {"avg": float, "max": int, "count": int}}
        """
        result = {
            "system_avg": 0.0,
            "system_max": 0,
            "system_diff": 0.0,
            "completed_count": 0,
            "in_progress_count": 0,
            "chambers": {},
            "transports": {"avg": 0.0, "max": 0, "count": 0},
            "transports_detail": {},
        }
        
        # 腔室名称映射到显示名称
        chamber_display = {
            "s1": "PM7/8",
            "s2": "LLC",
            "s3": "PM1/2/3/4",
            "s4": "LLD",
            "s5": "PM9/10",
        }
        
        # 初始化腔室统计
        for chamber in chamber_display.values():
            result["chambers"][chamber] = {"avg": 0.0, "max": 0, "count": 0, "times": []}
        
        # 初始化运输位统计
        transport_names = ["d_s1", "d_s2", "d_s3", "d_s4", "d_s5", "d_LP_done"]
        for t_name in transport_names:
            result["transports_detail"][t_name] = {"avg": 0.0, "max": 0, "count": 0, "times": []}
        
        system_times = []
        all_transport_times = []
        
        for wafer_id, stats in self.wafer_stats.items():
            # 系统滞留时间
            enter = stats.get("enter_system")
            exit_time = stats.get("exit_system")
            
            if enter is not None:
                if exit_time is not None:
                    # 已完成的晶圆
                    system_time = exit_time - enter
                    system_times.append(system_time)
                    result["completed_count"] += 1
                else:
                    # 进行中的晶圆，使用当前时间计算
                    #system_time = self.time - enter
                    #system_times.append(system_time)
                    result["in_progress_count"] += 1
            
            # 腔室滞留时间
            for chamber_name, times in stats.get("chambers", {}).items():
                display_name = chamber_display.get(chamber_name, chamber_name)
                enter_t = times.get("enter")
                exit_t = times.get("exit")
                
                if enter_t is not None:
                    stay_time = 0
                    if exit_t is not None:
                        stay_time = exit_t - enter_t
                    #else:
                        # 还在腔室中
                    #    stay_time = self.time - enter_t
                    
                    if display_name in result["chambers"] and stay_time > 0:
                        result["chambers"][display_name]["times"].append(stay_time)
            
            # 运输位滞留时间
            for transport_name, times in stats.get("transports", {}).items():
                enter_t = times.get("enter")
                exit_t = times.get("exit")

                stay_time = 0
                if enter_t is not None:
                    if exit_t is not None:
                        stay_time = exit_t - enter_t
                    #else:
                        # 还在运输中
                    #    stay_time = self.time - enter_t
                    
                    if stay_time > 0:
                        all_transport_times.append(stay_time)
                        if transport_name in result["transports_detail"]:
                            result["transports_detail"][transport_name]["times"].append(stay_time)
        
        # 计算系统统计
        if system_times:
            result["system_avg"] = sum(system_times) / len(system_times)
            result["system_max"] = max(system_times)
            result["system_diff"] = max(system_times) - min(system_times)
        
        # 计算腔室统计
        for chamber_name, data in result["chambers"].items():
            times = data["times"]
            if times:
                data["avg"] = sum(times) / len(times)
                data["max"] = max(times)
                data["count"] = len(times)
            del data["times"]  # 清理临时数据
        
        # 计算运输位统计（合并）
        if all_transport_times:
            result["transports"]["avg"] = sum(all_transport_times) / len(all_transport_times)
            result["transports"]["max"] = max(all_transport_times)
            result["transports"]["count"] = len(all_transport_times)
        
        # 计算各运输位详细统计
        for transport_name, data in result["transports_detail"].items():
            times = data["times"]
            if times:
                data["avg"] = sum(times) / len(times)
                data["max"] = max(times)
                data["count"] = len(times)
            del data["times"]  # 清理临时数据
        
        return result

    def _next_accept_time(self, place, t_now: int) -> float:
        """计算某个 place 在 t_now 时刻之后，最早什么时候能接收一个新 token。"""
        cap = max(1, place.capacity)
        n = len(place.tokens)

        # 有空位：立刻可接收
        if n < cap:
            return float(t_now)

        # 已满：等最早完成的那个释放
        finishes = [tok.enter_time + place.processing_time for tok in place.tokens]
        return float(min(finishes))

    def reset(self):
        self.time = 1
        self.m = self.m0.copy()
        self.marks = self._clone_marks(self.ori_marks)
        self._update_stay_times()
        self.fire_log = []
        self._last_t_pm1_fire_time = -INF  # 重置错峰计时
        self.scrap_count = 0  # 重置报废计数器
        self._idle_penalty_applied = False  # 重置停滞惩罚标记
        self._consecutive_wait_time = 0  # 重置连续 WAIT 时间
        self._release_violation_penalty = 0.0  # 重置释放时间违规惩罚
        self._per_wafer_reward = 0.0  # 重置单片完工奖励
        self.wafer_stats = {}  # 重置晶圆滞留时间统计
        self.entered_wafer_count = 0  # 重置已进入系统的晶圆数
        # 清空所有库所的释放时间队列，并重置机器分配计数器
        for place in self.marks:
            place.release_schedule.clear()
            if place.type == 1:
                place.last_machine = -1  # 重置机器轮换计数器
        # 数据结构优化：更新 _marks_by_type 缓存（因为 marks 列表已重新克隆）
        if self.optimize_data_structures:
            try:
                self._build_marks_by_type_cache()
            except Exception:
                # 如果更新缓存失败，使用空缓存，回退到遍历方式
                self._marks_by_type = {}
        
        # 更新 _process_places_cache（即使未启用数据结构优化也需要更新）
        self._process_places_cache = [p for p in self.marks if p.type == 1]

    def _update_stay_times(self) -> None:
        """更新所有 token 的滞留时间（批量更新优化）"""
        current_time = self.time
        if self.optimize_state_update:
            # 批量更新：减少函数调用开销
            # 使用按类型分组的缓存，跳过 type=3 (LP)
            if self.optimize_data_structures:
                # 遍历所有非 type=3 的库所类型，直接使用缓存避免函数调用
                marks_by_type = self._marks_by_type
                for place_type in [1, 2, 4, 5]:  # 跳过 type=3 (LP)
                    for place in marks_by_type.get(place_type, []):
                        if len(place.tokens) > 0:
                            for tok in place.tokens:
                                tok.stay_time = int(current_time - tok.enter_time)
            else:
                # 原始优化方式
                for place in self.marks:
                    if place.type == 3:  # 跳过 LP 中的 wafer
                        continue
                    if len(place.tokens) > 0:
                        # 批量计算并更新
                        for tok in place.tokens:
                            tok.stay_time = int(current_time - tok.enter_time)
        else:
            # 原始实现
            for place in self.marks:
                if place.type == 3:  # 跳过 LP 中的 wafer
                    continue
                for tok in place.tokens:
                    tok.stay_time = int(current_time - tok.enter_time)

    def calc_reward(self, t1: int, t2: int, moving_pre_places: Optional[np.ndarray] = None,
                          detailed: bool = False) -> float | Dict[str, float]:
        """
        计算从时间 t1 到 t2 的奖励和惩罚。
        
        奖励结构：
        1. 加工奖励：type=1 腔室内 token 在 [enter, enter+proc_time) 每秒 +r
        2. 安全裕量奖励：下游腔室有 token 且 slack > T_safe 时，每秒 +b_safe
        3. 时间成本：-c_time * Δt（不做事也亏）
        4. 超时惩罚：
           - type=1: 超过 proc_time + P_Residual_time后每秒惩罚
           - type=2: 超过 5 + D_Residual_time后每秒惩罚
        5. 预警惩罚：slack < T_warn 时施加稠密惩罚
        6. 堵塞预测惩罚：上游多个 token 即将同时完成而下游容量不足
        7. 错峰启动奖励：同一腔室内 token 进入时间差足够大
        8. 下游空闲惩罚：下游空闲但上游有等待 token
        
        Args:
            t1: 起始时间
            t2: 结束时间
            moving_pre_places: 正在移动的前置库所（可选）
            detailed: 是否返回详细奖励分解
            
        Returns:
            如果 detailed=False: 返回总奖励 (float)
            如果 detailed=True: 返回奖励字典 Dict[str, float]
        """
        if t2 <= t1:
            if detailed:
                return {"total": 0.0, "proc_reward": 0.0, "safe_reward": 0.0, 
                        "penalty": 0.0, "warn_penalty": 0.0, "transport_penalty": 0.0,
                        "congestion_penalty": 0.0, "time_cost": 0.0}
            return 0.0

        # 根据配置选择优化或原始实现
        if self.optimize_reward_calc:
            return self._calc_reward_vectorized(t1, t2, moving_pre_places, detailed)
        else:
            return self._calc_reward_original(t1, t2, moving_pre_places, detailed)
    def _calc_reward_original(self, t1: int, t2: int, moving_pre_places: Optional[np.ndarray] = None,
                              detailed: bool = False) -> float | Dict[str, float]:
        """原始奖励计算实现（用于功能一致性验证）"""
        moving_set = set(moving_pre_places.tolist()) if moving_pre_places is not None else set()
        
        # 从配置读取惩罚/奖励系数
        transport_overtime_coef = self.config.transport_overtime_coef  # type=2 运输库所超时惩罚系数
        chamber_overtime_coef = self.config.chamber_overtime_coef      # type=1 加工腔室超时惩罚系数
        processing_reward_coef = self.config.processing_reward_coef    # 加工奖励系数
        
        delta_t = t2 - t1
        proc_reward = 0.0      # 加工奖励
        overtime_penalty = 0.0 # 超时惩罚（加工腔室）
        warn_penalty = 0.0     # 预警惩罚
        transport_penalty = 0.0 # 运输超时惩罚
        safe_reward = 0.0      # 安全裕量奖励
        congestion_penalty = 0.0 # 堵塞预测惩罚

        # ========== 基本奖励 ==========
        for p_idx, place in enumerate(self.marks):
            for tok_idx, tok in enumerate(place.tokens):
                # LP 库所不计入惩罚（源头）
                if place.name == "LP":
                    continue

                if place.type not in (1, 2, 5):
                    continue

                # 查询是否存在运输时间违规
                if place.type == 2:
                    if self.reward_config.get('transport_penalty', 1):
                        deadline = tok.enter_time + self.D_Residual_time
                        over_start = max(t1, deadline)
                        if t2 > over_start:
                            transport_penalty += (t2 - over_start) * transport_overtime_coef

                # type=5: 无驻留约束腔室（如 s2/LLC、s4/LLD）- 只计加工奖励，无惩罚
                elif place.type == 5:
                    if place.processing_time > 0 and self.reward_config.get('proc_reward', 1):
                        proc_end = tok.enter_time + place.processing_time
                        proc_overlap = min(t2, proc_end) - max(t1, tok.enter_time)
                        if proc_overlap > 0:
                            proc_reward += proc_overlap * processing_reward_coef

                # 查询加工腔室相关奖励/惩罚（type=1 有驻留约束）
                elif place.type == 1:
                    # 加工腔室
                    proc_start = tok.enter_time
                    proc_end = tok.enter_time + place.processing_time
                    
                    # 1) 加工奖励：在加工时间内每秒 +r
                    if self.reward_config.get('proc_reward', 1):
                        proc_overlap = min(t2, proc_end) - max(t1, proc_start)
                        if proc_overlap > 0:
                            proc_reward += proc_overlap * processing_reward_coef
                    
                    # 2) 超时惩罚：超过 proc_time+ P_Residual_time 后惩罚
                    if self.reward_config.get('penalty', 1):
                        start_pen = tok.enter_time + place.processing_time + self.P_Residual_time
                        start = max(t1, start_pen)
                        if t2 > start:
                            overtime_penalty += (t2 - start) * chamber_overtime_coef
                    
                    # 3) 预警惩罚：slack < T_warn 时施加稠密惩罚
                    if self.reward_config.get('warn_penalty', 1):
                        scrap_deadline = tok.enter_time + place.processing_time + 10
                        warn_start = scrap_deadline - self.T_warn
                        
                        warn_overlap_start = max(t1, warn_start)
                        warn_overlap_end = min(t2, scrap_deadline)
                        
                        if warn_overlap_end > warn_overlap_start:
                            dt = warn_overlap_end - warn_overlap_start
                            avg_gap = ((warn_overlap_start - warn_start) + (warn_overlap_end - warn_start)) / 2
                            warn_penalty += self.a_warn * avg_gap * dt
                    
                    # 4) 安全裕量奖励：slack > T_safe 时给正向奖励
                    if self.reward_config.get('safe_reward', 1):
                        safe_deadline = tok.enter_time + place.processing_time + 10 - self.T_safe
                        safe_start = max(t1, proc_end)
                        safe_end = min(t2, safe_deadline)
                        if safe_end > safe_start:
                            safe_reward += self.b_safe * (safe_end - safe_start)
        
        # ========== 堵塞预防奖励塑形 ==========
        for up_idx, downstream_list in self.downstream_map.items():
            up_place = self.marks[up_idx]
            if len(up_place.tokens) == 0:
                continue
            
            finish_times = []
            for tok in up_place.tokens:
                if up_place.type == 1:
                    finish_time = tok.enter_time + up_place.processing_time + 15
                elif up_place.type == 2:
                    finish_time = tok.enter_time + 55
                finish_times.append(finish_time)
            
            for down_idx, d_idx in downstream_list:
                down_place = self.marks[down_idx]
                d_place = self.marks[d_idx]
                
                if len(down_place.tokens) == 0 and len(d_place.tokens) == 0:
                    continue
                
                down_available = max(0, down_place.capacity - len(down_place.tokens) - len(d_place.tokens))
                down_finish = []
                for tok in down_place.tokens:
                    down_finish.append(tok.enter_time + down_place.processing_time)
                for tok in d_place.tokens:
                    down_finish.append(tok.enter_time + 10 + 80)
                
                overflow = 0
                for i, ft in enumerate(finish_times):
                    if i < down_available:
                        continue
                    if ft < down_finish[0]:
                        overflow += 1
                        break
                
                if overflow > 0:
                    congestion_penalty += self.c_congest * overflow
        
        if not self.reward_config.get('congestion_penalty', 1):
            congestion_penalty = 0.0
        
        time_cost = self.c_time * delta_t if self.reward_config.get('time_cost', 1) else 0.0
        total_penalty = overtime_penalty + warn_penalty + transport_penalty
        total = proc_reward + safe_reward - total_penalty - congestion_penalty - time_cost
        
        if detailed:
            return {
                "total": total,
                "proc_reward": proc_reward,
                "safe_reward": safe_reward,
                "penalty": overtime_penalty,
                "warn_penalty": warn_penalty,
                "transport_penalty": transport_penalty,
                "congestion_penalty": congestion_penalty,
                "time_cost": time_cost,
            }
        
        return total
    
    def _calc_reward_vectorized(self, t1: int, t2: int, moving_pre_places: Optional[np.ndarray] = None,
                                 detailed: bool = False) -> float | Dict[str, float]:
        """
        向量化版本的奖励计算（性能优化）。
        
        使用 NumPy 向量化操作替代 Python 循环，大幅提升性能。
        """
        # 惩罚/奖励系数
        Q1_p = 3      # type=2 运输库所超时惩罚系数
        Q2_p = 0.2    # type=1 加工腔室超时惩罚系数
        r = 2         # 加工奖励系数
        
        delta_t = t2 - t1
        proc_reward = 0.0
        overtime_penalty = 0.0
        warn_penalty = 0.0
        transport_penalty = 0.0
        safe_reward = 0.0
        congestion_penalty = 0.0
        
        # ========== 向量化计算基本奖励 ==========
        # 收集所有需要计算的 token 数据
        enter_times = []
        proc_times = []
        place_types = []
        place_names = []
        
        for p_idx, place in enumerate(self.marks):
            if place.name == "LP" or place.type not in (1, 2, 5):
                continue
            
            for tok in place.tokens:
                enter_times.append(tok.enter_time)
                proc_times.append(place.processing_time)
                place_types.append(place.type)
                place_names.append(place.name)
        
        if not enter_times:
            # 没有 token 需要计算
            time_cost = self.c_time * delta_t if self.reward_config.get('time_cost', 1) else 0.0
            if detailed:
                return {"total": -time_cost, "proc_reward": 0.0, "safe_reward": 0.0,
                        "penalty": 0.0, "warn_penalty": 0.0, "transport_penalty": 0.0,
                        "congestion_penalty": 0.0, "time_cost": time_cost}
            return -time_cost
        
        # 转换为 NumPy 数组
        enter_times = np.array(enter_times, dtype=np.int32)
        proc_times = np.array(proc_times, dtype=np.int32)
        place_types = np.array(place_types, dtype=np.int32)
        
        # 条件短路：提前退出不需要的计算分支
        need_proc_reward = self.reward_config.get('proc_reward', 1)
        need_penalty = self.reward_config.get('penalty', 1)
        need_warn_penalty = self.reward_config.get('warn_penalty', 1)
        need_safe_reward = self.reward_config.get('safe_reward', 1)
        need_transport_penalty = self.reward_config.get('transport_penalty', 1)
        
        # type=2: 运输库所超时惩罚
        if need_transport_penalty:
            type2_mask = (place_types == 2)
            if np.any(type2_mask):
                deadlines = enter_times[type2_mask] + self.D_Residual_time
                over_starts = np.maximum(t1, deadlines)
                over_mask = t2 > over_starts
                if np.any(over_mask):
                    transport_penalty = np.sum((t2 - over_starts[over_mask]) * Q1_p)
        
        # type=5: 无驻留约束腔室（只计加工奖励）
        if need_proc_reward:
            type5_mask = (place_types == 5) & (proc_times > 0)
            if np.any(type5_mask):
                proc_ends = enter_times[type5_mask] + proc_times[type5_mask]
                proc_overlaps = np.minimum(t2, proc_ends) - np.maximum(t1, enter_times[type5_mask])
                positive_overlaps = proc_overlaps[proc_overlaps > 0]
                if len(positive_overlaps) > 0:
                    proc_reward += np.sum(positive_overlaps * r)
        
        # type=1: 加工腔室（有驻留约束）
        type1_mask = (place_types == 1)
        if np.any(type1_mask):
            type1_enter = enter_times[type1_mask]
            type1_proc = proc_times[type1_mask]
            proc_starts = type1_enter
            proc_ends = type1_enter + type1_proc
            
            # 1) 加工奖励
            if need_proc_reward:
                proc_overlaps = np.minimum(t2, proc_ends) - np.maximum(t1, proc_starts)
                positive_overlaps = proc_overlaps[proc_overlaps > 0]
                if len(positive_overlaps) > 0:
                    proc_reward += np.sum(positive_overlaps * r)
            
            # 2) 超时惩罚
            if need_penalty:
                start_pens = type1_enter + type1_proc + self.P_Residual_time
                starts = np.maximum(t1, start_pens)
                over_mask = t2 > starts
                if np.any(over_mask):
                    overtime_penalty = np.sum((t2 - starts[over_mask]) * Q2_p)
            
            # 3) 预警惩罚
            if need_warn_penalty:
                scrap_deadlines = type1_enter + type1_proc + 10
                warn_starts = scrap_deadlines - self.T_warn
                warn_overlap_starts = np.maximum(t1, warn_starts)
                warn_overlap_ends = np.minimum(t2, scrap_deadlines)
                valid_mask = warn_overlap_ends > warn_overlap_starts
                if np.any(valid_mask):
                    dts = warn_overlap_ends[valid_mask] - warn_overlap_starts[valid_mask]
                    avg_gaps = ((warn_overlap_starts[valid_mask] - warn_starts[valid_mask]) + 
                               (warn_overlap_ends[valid_mask] - warn_starts[valid_mask])) / 2
                    warn_penalty = np.sum(self.a_warn * avg_gaps * dts)
            
            # 4) 安全裕量奖励
            if need_safe_reward:
                safe_deadlines = type1_enter + type1_proc + 10 - self.T_safe
                safe_starts = np.maximum(t1, proc_ends)
                safe_ends = np.minimum(t2, safe_deadlines)
                valid_mask = safe_ends > safe_starts
                if np.any(valid_mask):
                    safe_reward = np.sum(self.b_safe * (safe_ends[valid_mask] - safe_starts[valid_mask]))
        
        # ========== 堵塞预防奖励塑形（保持原始实现，因为逻辑复杂）==========
        for up_idx, downstream_list in self.downstream_map.items():
            up_place = self.marks[up_idx]
            if len(up_place.tokens) == 0:
                continue
            
            finish_times = []
            for tok in up_place.tokens:
                if up_place.type == 1:
                    finish_time = tok.enter_time + up_place.processing_time + 15
                elif up_place.type == 2:
                    finish_time = tok.enter_time + 55
                finish_times.append(finish_time)
            
            for down_idx, d_idx in downstream_list:
                down_place = self.marks[down_idx]
                d_place = self.marks[d_idx]
                
                if len(down_place.tokens) == 0 and len(d_place.tokens) == 0:
                    continue
                
                down_available = max(0, down_place.capacity - len(down_place.tokens) - len(d_place.tokens))
                down_finish = []
                for tok in down_place.tokens:
                    down_finish.append(tok.enter_time + down_place.processing_time)
                for tok in d_place.tokens:
                    down_finish.append(tok.enter_time + 10 + 80)
                
                overflow = 0
                for i, ft in enumerate(finish_times):
                    if i < down_available:
                        continue
                    if ft < down_finish[0]:
                        overflow += 1
                        break
                
                if overflow > 0:
                    congestion_penalty += self.c_congest * overflow
        
        # 应用奖励开关
        if not self.reward_config.get('congestion_penalty', 1):
            congestion_penalty = 0.0
        
        # 时间成本
        time_cost = self.c_time * delta_t if self.reward_config.get('time_cost', 1) else 0.0
        
        # 合并惩罚
        total_penalty = overtime_penalty + warn_penalty + transport_penalty
        total = proc_reward + safe_reward - total_penalty - congestion_penalty - time_cost
        
        if detailed:
            return {
                "total": total,
                "proc_reward": proc_reward,
                "safe_reward": safe_reward,
                "penalty": overtime_penalty,
                "warn_penalty": warn_penalty,
                "transport_penalty": transport_penalty,
                "congestion_penalty": congestion_penalty,
                "time_cost": time_cost,
            }
        
        return total



    # ---------- 计算最早可使能时间 ----------
    def _earliest_enable_time(self, t):
        pre_places = np.flatnonzero(self.pre[:, t] > 0)
        earliest = 0
        for p in pre_places:
            tok_enter = int(self.marks[p].head().enter_time)  # 队头
            tok_enter += int(self.ptime[p])
            earliest = max(earliest, tok_enter)
        
        return int(earliest)

    def _resource_enable(self):
        # 双路线变迁:
        # 路线1: ['u_LP1_s1', 't_s1', 'u_s1_s2', 't_s2', 'u_s2_s3', 't_s3', 
        #         'u_s3_s4', 't_s4', 'u_s4_s5', 't_s5', 'u_s5_LP_done', 't_LP_done']
        # 路线2: ['u_LP2_s1', 't_s1', 'u_s1_s5', 't_s5', 'u_s5_LP_done', 't_LP_done']
        # 库所: ['LP1', 'LP2', 'LP_done', 's1', 's2', 's3', 's4', 's5', 
        #        'r_TM2', 'r_TM3', 'd_s1', 'd_s2', 'd_s3', 'd_s4', 'd_s5', 'd_LP_done']
        
        # 基本使能条件：前置库所有足够 token 且后置库所不超容量
        cond_pre = (self.pre <= self.m[:, None]).all(axis=0)
        cond_cap = ((self.m[:, None] + self.pst) <= self.k[:, None]).all(axis=0)
        mask = cond_pre & cond_cap
        
        # 双机械手容量约束：防止向已满的加工腔室发送晶圆
        # s1 容量=2：当 s1 已满时，禁用 u_LP1_s1 和 u_LP2_s1
        # s3 容量=4：当 s3 已满时，禁用 u_s2_s3
        # s5 容量=2：当 s5 已满时，禁用 u_s4_s5 和 u_s1_s5
        capacity_constraints = [
            ("s1", "u_LP1_s1"), ("s1", "u_LP2_s1"),  # 双起点到 s1
            ("s3", "u_s2_s3"),                        # 路线1: s2 -> s3
            ("s5", "u_s4_s5"), ("s5", "u_s1_s5"),    # 路线1: s4 -> s5, 路线2: s1 -> s5
        ]
        for place_name, u_trans_name in capacity_constraints:
            if place_name in self.id2p_name and u_trans_name in self.id2t_name:
                p_idx = self._get_place_index(place_name)
                t_idx = self._get_transition_index(u_trans_name)
                if self.m[p_idx] >= self.marks[p_idx].capacity:
                    mask[t_idx] = False
        
        # ========== 限制进入系统的晶圆数量 ==========
        # 当已进入系统的晶圆数达到上限时，禁用 u_LP1_s1 和 u_LP2_s1
        if self.entered_wafer_count >= self.max_wafers_in_system:
            if "u_LP1_s1" in self.id2t_name:
                t_idx = self._get_transition_index("u_LP1_s1")
                mask[t_idx] = False
            if "u_LP2_s1" in self.id2t_name:
                t_idx = self._get_transition_index("u_LP2_s1")
                mask[t_idx] = False
        
        # ========== 步骤索引感知的路由约束 (u_变迁) ==========
        # 使用 (route_type, step) 决定晶圆下一个目标
        # u_变迁：检查源库所队头晶圆的 step 对应的下一目标是否匹配变迁的目标
        
        # 定义 u 变迁到其目标腔室的映射
        u_trans_targets = {
            "u_LP1_s1": "s1",
            "u_LP2_s1": "s1",
            "u_s1_s2": "s2",
            "u_s1_s5": "s5",
            "u_s2_s3": "s3",
            "u_s3_s4": "s4",
            "u_s4_s5": "s5",
            "u_s5_LP_done": "LP_done",
        }
        
        # 定义 u 变迁的源库所
        u_trans_sources = {
            "u_LP1_s1": "LP1",
            "u_LP2_s1": "LP2",
            "u_s1_s2": "s1",
            "u_s1_s5": "s1",
            "u_s2_s3": "s2",
            "u_s3_s4": "s3",
            "u_s4_s5": "s4",
            "u_s5_LP_done": "s5",
        }
        
        for u_name, target in u_trans_targets.items():
            if u_name not in self.id2t_name:
                continue
            t_idx = self._get_transition_index(u_name)
            if not mask[t_idx]:  # 已被禁用
                continue
            
            # 获取源库所
            source_name = u_trans_sources.get(u_name)
            if source_name not in self.id2p_name:
                continue
            
            source_idx = self._get_place_index(source_name)
            source_place = self.marks[source_idx]
            
            if len(source_place.tokens) == 0:
                continue
            
            # 检查队头晶圆的下一目标是否匹配
            head_tok = source_place.head()
            if not self._can_enter_target(head_tok, target):
                mask[t_idx] = False

        # ========== 运输库所 FIFO 约束 (t_变迁) ==========
        # 对所有 d_ 开头的运输库所 (d_TM2, d_TM3等) 应用 FIFO 策略
        # 确保从 d_xxx 取出的晶圆确实是去往该 t_变迁对应的目标腔室
        for p_name in self.id2p_name:
            if not p_name.startswith("d_"):
                continue
                
            d_idx = self._get_place_index(p_name)
            d_place = self.marks[d_idx]
            
            if len(d_place.tokens) > 0:
                head_tok = d_place.head()
                # 确保是 wafer token (有 route_type)
                if head_tok.route_type == 0:
                    continue
                    
                expected_target = self._get_next_target(head_tok.route_type, head_tok.step)
                
                # 找到所有消费此 d_place 的变迁
                consumers = np.where(self.pre[d_idx, :] > 0)[0]
                for t_idx in consumers:
                    t_name = self.id2t_name[t_idx]
                    if t_name.startswith("t_") and mask[t_idx]:
                        # t_name 格式为 t_{target}
                        target = t_name[2:]
                        if target != expected_target:
                            mask[t_idx] = False
        
        out_te = np.flatnonzero(mask)
        return out_te

    def next_enable_time(self) -> int:
        te = self._resource_enable()
        if len(te) == 0:
            return self.time + 1
        earliest = INF
        for t in te:
            earliest = min(earliest, self._earliest_enable_time(t))
        return int(earliest)

    def _fire(self, t):
        start_time = self.time
        enter_new = self.time + self.ttime
        t_name = self.id2t_name[t]

        pre_places = np.flatnonzero(self.pre[:, t] > 0)
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 1) 消费前置库所 token（队头），保存 token_id, route_type, step
        consumed_token_ids = []
        consumed_route_types = []
        consumed_steps = []
        for p in pre_places:
            place = self.marks[p]
            # 只有非资源库所的 token 才有 wafer id 和路线信息
            if place.type != 4:  # 非资源库所
                tok = place.head()
                consumed_token_ids.append(tok.token_id)
                consumed_route_types.append(tok.route_type)
                consumed_steps.append(tok.step)
            place.pop_head()
            self.m[p] -= 1

        # 2) 生成后置库所 token：enter_time = finish time，继承 token_id, route_type
        #    t_ 变迁（进入腔室）时递增 step
        # 保存后置库所信息用于 fire_log
        post_place_info = []  # [(chamber_name, machine_id, token_id), ...]
        
        token_id_idx = 0
        for p in pst_places:
            place = self.marks[p]
            # 只有非资源库所才传递 wafer id 和路线信息
            if place.type != 4 and token_id_idx < len(consumed_token_ids):
                tid = consumed_token_ids[token_id_idx]
                t_route_type = consumed_route_types[token_id_idx]
                t_step = consumed_steps[token_id_idx]
                token_id_idx += 1
                
                # t_ 变迁（进入非运输库所）时递增 step
                # 运输库所 (type=2, d_xxx) 不递增 step
                if t_name.startswith("t_") and place.type != 2:
                    t_step += 1
            else:
                tid = -1  # 资源库所的 token 无 wafer id
                t_route_type = 0
                t_step = 0
            
            # 为 type=1 的加工腔室分配机器（轮换策略）
            if place.type == 1:
                machine_id = (place.last_machine + 1) % place.capacity
                place.last_machine = machine_id
            else:
                machine_id = -1
            
            # 保存后置库所信息（用于 fire_log）
            if place.type != 4:  # 非资源库所
                post_place_info.append((place.name, machine_id, tid))
            
            self.marks[p].append(BasedToken(enter_time=enter_new, stay_time=1, token_id=tid, machine=machine_id, route_type=t_route_type, step=t_step))
            self.m[p] += 1

        # 3) 时间推进到完成之后
        self.time = enter_new
        self._update_stay_times()
        
        # 记录 t_s1 发射时间（用于错峰约束，如需要）
        if t_name == "t_s1":
            self._last_t_pm1_fire_time = start_time
        
        # ========== 4) 释放时间追踪逻辑 ==========
        # 获取晶圆 token_id 和 route_type（第一个非资源库所消费的 token）
        wafer_id = consumed_token_ids[0] if consumed_token_ids else -1
        wafer_route_type = consumed_route_types[0] if consumed_route_types else 0
        
        # 根据变迁类型处理释放时间（只追踪有驻留约束的腔室：s1, s3, s5）
        # 双路线：u_LP1_s1 和 u_LP2_s1 都进入 d_s1
        if t_name in ("u_LP1_s1", "u_LP2_s1"):
            # 递增已进入系统的晶圆计数
            self.entered_wafer_count += 1
            # 晶圆离开 LP 进入 d_s1：只对 s1 记录并检查，不链式传播（离开才检查）
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                penalty = self._record_initial_release(wafer_id, enter_new, s1_idx, wafer_route_type, chain_downstream=False)
                self._release_violation_penalty += penalty
            
        elif t_name == "t_s1":
            # 晶圆进入 s1：只更新 s1，不链式更新（下游 s3/s4/s5 在离开 s2/s4 时才写入）
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._update_release(wafer_id, enter_new, s1_idx, wafer_route_type, chain_downstream=False)
            
        elif t_name == "u_s1_s2":
            # 路线1：晶圆离开 s1 去 s2，从 s1 队列移除
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._pop_release(wafer_id, s1_idx)
        
        elif t_name == "u_s1_s5":
            # 路线2：晶圆离开 s1 直接去 s5，从 s1 队列移除；对 s5 做 add_release + 违规检查（离开才检查）
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._pop_release(wafer_id, s1_idx)
            if "s5" in self.id2p_name:
                s5_idx = self._get_place_index("s5")
                place_s5 = self.marks[s5_idx]
                # 晶圆进入 d_s5 时间 = enter_new，预计进入 s5 = enter_new + T_load
                expected_enter_s5 = enter_new + self.T_load
                
                penalty_s5 = 0.0
                corrected_enter_s5 = expected_enter_s5
                if self.reward_config.get('release_violation_penalty', 1):
                    penalty_s5, corrected_enter_s5 = self._check_release_violation(s5_idx, expected_enter_s5)
                    self._release_violation_penalty += penalty_s5
                
                # 使用修正后的进入时间重新计算释放时间
                release_s5 = corrected_enter_s5 + place_s5.processing_time
                place_s5.add_release(wafer_id, release_s5)
            
        elif t_name == "u_s2_s3":
            # 路线1：离开 s2(LLC)，对 s3、s4 做 add_release + 违规检查（离开才检查）
            if "s3" in self.id2p_name and "s4" in self.id2p_name:
                s3_idx = self._get_place_index("s3")
                s4_idx = self._get_place_index("s4")
                place_s3 = self.marks[s3_idx]
                place_s4 = self.marks[s4_idx]
                expected_enter_s3 = enter_new + self.T_load
                
                # 检查 s3 违规并获取修正后的进入时间
                penalty = 0.0
                corrected_enter_s3 = expected_enter_s3
                if self.reward_config.get('release_violation_penalty', 1):
                    penalty_s3, corrected_enter_s3 = self._check_release_violation(s3_idx, expected_enter_s3)
                    penalty += penalty_s3
                
                # 使用修正后的 s3 进入时间重新计算 s3 释放时间和 s4 进入时间
                release_s3 = corrected_enter_s3 + place_s3.processing_time
                transport_s3_to_s4 = self.ttime * 3
                expected_enter_s4 = release_s3 + transport_s3_to_s4
                
                # 检查 s4 违规并获取修正后的进入时间
                corrected_enter_s4 = expected_enter_s4
                if self.reward_config.get('release_violation_penalty', 1):
                    penalty_s4, corrected_enter_s4 = self._check_release_violation(s4_idx, expected_enter_s4)
                    penalty += penalty_s4
                
                # 使用修正后的 s4 进入时间重新计算 s4 释放时间
                release_s4 = corrected_enter_s4 + place_s4.processing_time
                
                place_s3.add_release(wafer_id, release_s3)
                place_s4.add_release(wafer_id, release_s4)
                self._release_violation_penalty += penalty
            
        elif t_name == "t_s3":
            # 晶圆进入 s3：更新精确释放时间 + 链式更新 s4（s4 已在 u_s2_s3 时加入）
            if "s3" in self.id2p_name:
                s3_idx = self._get_place_index("s3")
                self._update_release(wafer_id, enter_new, s3_idx, wafer_route_type)
        
        elif t_name == "t_s4":
            # 晶圆进入 s4：只更新 s4，不链式更新（s5 在 u_s4_s5 时才加入）
            if "s4" in self.id2p_name:
                s4_idx = self._get_place_index("s4")
                self._update_release(wafer_id, enter_new, s4_idx, wafer_route_type, chain_downstream=False)
            
        elif t_name == "u_s3_s4":
            # 晶圆离开 s3：从 s3 队列移除
            if "s3" in self.id2p_name:
                s3_idx = self._get_place_index("s3")
                self._pop_release(wafer_id, s3_idx)
        
        elif t_name == "u_s4_s5":
            # 路线1：离开 s4(LLD)，对 s5 做 add_release + 违规检查（离开才检查）；再从 s4 队列移除
            if "s4" in self.id2p_name and "s5" in self.id2p_name:
                s4_idx = self._get_place_index("s4")
                s5_idx = self._get_place_index("s5")
                place_s4 = self.marks[s4_idx]
                place_s5 = self.marks[s5_idx]
                release_s4 = place_s4.get_release(wafer_id)
                if release_s4 is not None:
                    # 修复：运输时间不需要再加 70s，因为 release_s4 已经包含了 s4 的加工时间
                    # 运输时间 = s4卸载 + d_s5运输 + s5装载
                    transport_s4_to_s5 = self.ttime * 3  # 约 15s
                    expected_enter_s5 = release_s4 + transport_s4_to_s5
                    
                    penalty_s5 = 0.0
                    corrected_enter_s5 = expected_enter_s5
                    if self.reward_config.get('release_violation_penalty', 1):
                        penalty_s5, corrected_enter_s5 = self._check_release_violation(s5_idx, expected_enter_s5)
                        self._release_violation_penalty += penalty_s5
                    
                    # 使用修正后的进入时间重新计算释放时间
                    release_s5 = corrected_enter_s5 + place_s5.processing_time
                    place_s5.add_release(wafer_id, release_s5)
                else:
                    # 防御性处理：release_s4 为 None 时使用当前时间估算
                    # 这种情况不应该发生，如果发生说明 u_s2_s3 时未正确写入 s4 的 release_schedule
                    import warnings
                    warnings.warn(
                        f"[释放时间异常] wafer_id={wafer_id} 在 u_s4_s5 时 release_s4 为 None，"
                        f"s4.release_schedule={list(place_s4.release_schedule)}，使用当前时间估算",
                        RuntimeWarning
                    )
                    # 使用当前时间 + 运输时间作为 fallback
                    transport_s4_to_s5 = self.ttime * 3
                    expected_enter_s5 = enter_new + transport_s4_to_s5
                    
                    penalty_s5 = 0.0
                    corrected_enter_s5 = expected_enter_s5
                    if self.reward_config.get('release_violation_penalty', 1):
                        penalty_s5, corrected_enter_s5 = self._check_release_violation(s5_idx, expected_enter_s5)
                        self._release_violation_penalty += penalty_s5
                    
                    release_s5 = corrected_enter_s5 + place_s5.processing_time
                    place_s5.add_release(wafer_id, release_s5)
                self._pop_release(wafer_id, s4_idx)
            
        elif t_name == "t_s5":
            # 晶圆进入 s5：更新精确释放时间
            if "s5" in self.id2p_name:
                s5_idx = self._get_place_index("s5")
                self._update_release(wafer_id, enter_new, s5_idx, wafer_route_type)
            
        elif t_name == "u_s5_LP_done":
            # 晶圆离开 s5：从 s5 队列移除
            if "s5" in self.id2p_name:
                s5_idx = self._get_place_index("s5")
                self._pop_release(wafer_id, s5_idx)
        
        elif t_name == "t_LP_done":
            # 晶圆完成加工，给予单片完工奖励
            self._per_wafer_reward += self.R_done
            self.entered_wafer_count -= 1
            self.done_count += 1
        
        # ========== 5) 晶圆滞留时间统计追踪 ==========
        self._track_wafer_statistics(t_name, wafer_id, start_time, enter_new)
        
        # ========== 6) 记录 fire_log，包含足够信息用于甘特图绘制 ==========
        log_entry = {
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(enter_new),
            "token_id": wafer_id,
        }
        
        # 解析变迁名称，提取额外信息
        # 进入腔室的变迁：t_s1, t_s2, t_s3, t_s4, t_s5
        if t_name.startswith("t_") and len(t_name) > 2:
            chamber_name = t_name[2:]  # 提取 "s1", "s2" 等
            if chamber_name in ["s1", "s2", "s3", "s4", "s5"]:
                # 从 post_place_info 中找到对应的 chamber_name 和 machine_id
                for p_name, m_id, tid in post_place_info:
                    if p_name == chamber_name and tid == wafer_id:
                        log_entry["chamber_name"] = chamber_name
                        log_entry["machine_id"] = m_id
                        break
                else:
                    # 如果没找到，设置默认值
                    log_entry["chamber_name"] = chamber_name
                    log_entry["machine_id"] = -1
        
        # 离开腔室的变迁：u_s1_s2, u_s1_s5, u_s2_s3, u_s3_s4, u_s4_s5, u_s5_LP_done
        elif t_name.startswith("u_") and "_" in t_name[2:]:
            parts = t_name[2:].split("_")  # 例如 "s1_s2" -> ["s1", "s2"], "s5_LP_done" -> ["s5", "LP", "done"]
            if len(parts) >= 2:
                from_chamber = parts[0]
                # 处理特殊情况：u_s5_LP_done -> to_chamber = "LP_done"
                if len(parts) > 2 and parts[1] == "LP" and parts[2] == "done":
                    to_chamber = "LP_done"
                else:
                    to_chamber = parts[1]
                log_entry["from_chamber"] = from_chamber
                log_entry["to_chamber"] = to_chamber
        
        # 机械手动作：所有 u_ 和 t_ 开头的变迁都涉及机械手动作
        # u_ 变迁：卸载变迁（从源库所取晶圆）
        # t_ 变迁：装载变迁（将晶圆放入目标库所）
        robot_map = {
            # u_ 变迁（卸载）
            "u_LP1_s1": "TM2",
            "u_LP2_s1": "TM2",
            "u_s1_s2": "TM2",
            "u_s1_s5": "TM2",
            "u_s2_s3": "TM3",
            "u_s3_s4": "TM3",
            "u_s4_s5": "TM2",
            "u_s5_LP_done": "TM2",
            # t_ 变迁（装载）
            "t_s1": "TM2",
            "t_s2": "TM2",
            "t_s3": "TM3",
            "t_s4": "TM3",
            "t_s5": "TM2",
            "t_LP_done": "TM2",
        }
        
        if t_name.startswith("u_") or t_name.startswith("t_"):
            robot_name = robot_map.get(t_name, "TM2")  # 默认 TM2
            log_entry["robot_name"] = robot_name
            
            # 提取 from_loc 和 to_loc
            if t_name.startswith("u_"):
                # u_ 变迁：u_A_B -> from_loc=A, to_loc=B
                if "_" in t_name[2:]:
                    parts = t_name[2:].split("_")
                    if len(parts) >= 2:
                        # 处理特殊情况：u_LP1_s1, u_LP2_s1 -> from_loc = "LP1"/"LP2"
                        if len(parts) >= 3 and parts[0] == "LP" and parts[1].isdigit():
                            log_entry["from_loc"] = f"{parts[0]}{parts[1]}"  # "LP1" 或 "LP2"
                            log_entry["to_loc"] = parts[2]
                        # 处理特殊情况：u_s5_LP_done -> to_loc = "LP_done"
                        elif len(parts) > 2 and parts[1] == "LP" and parts[2] == "done":
                            log_entry["from_loc"] = parts[0]
                            log_entry["to_loc"] = "LP_done"
                        else:
                            log_entry["from_loc"] = parts[0]
                            log_entry["to_loc"] = parts[1]
            elif t_name.startswith("t_"):
                # t_ 变迁：t_B -> from_loc=d_B (运输位), to_loc=B (目标库所)
                # 例如：t_s1 -> from_loc="d_s1", to_loc="s1"
                chamber_name = t_name[2:]  # 提取 "s1", "s2" 等
                if chamber_name in ["s1", "s2", "s3", "s4", "s5", "LP_done"]:
                    log_entry["from_loc"] = f"d_{chamber_name}"  # 运输位
                    log_entry["to_loc"] = chamber_name  # 目标库所
        
        self.fire_log.append(log_entry)

    def render_gantt(self, out_path: str = r"C:\Users\khand\OneDrive\code\dqn\CT\results\continuous_gantt.png"):
        """
        生成甘特图，追踪晶圆在腔室中的占用情况和机械手动作。
        
        腔室映射：
        - stage 1: s1 (PM7/PM8, capacity=2, processing_time=70)
        - stage 2: s2 (LLC, capacity=1, processing_time=0)
        - stage 3: s3 (PM1/PM2/PM3/PM4, capacity=4, processing_time=600)
        - stage 4: s4 (LLD, capacity=1, processing_time=70)
        - stage 5: s5 (PM9/PM10, capacity=2, processing_time=200)
        - stage 6: TM2 (机械手)
        - stage 7: TM3 (机械手)
        """
        # 腔室占用追踪：chamber_slots[chamber_name] = {token_id: (start_time, machine_id), ...}
        # 使用字典直接通过 token_id 查找，避免 FIFO 推断
        chamber_slots: Dict[str, Dict[int, Tuple[int, int]]] = {
            "s1": {},
            "s2": {},
            "s3": {},
            "s4": {},
            "s5": {},
        }
        
        # 腔室配置
        chamber_config = {
            "s1": {"stage": 1, "capacity": 2, "proc_time": 70},
            "s2": {"stage": 2, "capacity": 1, "proc_time": 0},
            "s3": {"stage": 3, "capacity": 4, "proc_time": 600},
            "s4": {"stage": 4, "capacity": 1, "proc_time": 70},
            "s5": {"stage": 5, "capacity": 2, "proc_time": 200},
        }
        
        # 机械手 stage 映射
        robot_stage_map = {
            "TM2": 6,
            "TM3": 7,
        }
        
        ops: List[Op] = []
        
        for log in self.fire_log:
            t_name = log["t_name"]
            t1 = log["t1"]
            t2 = log["t2"]
            token_id = log.get("token_id", -1)
            
            # ========== 处理进入腔室的变迁 ==========
            if "chamber_name" in log:
                chamber_name = log["chamber_name"]
                machine_id = log.get("machine_id", -1)
                # 记录晶圆进入腔室
                chamber_slots[chamber_name][token_id] = (t2, machine_id)
            
            # ========== 处理离开腔室的变迁 ==========
            if "from_chamber" in log:
                from_chamber = log["from_chamber"]
                if from_chamber in chamber_slots and token_id in chamber_slots[from_chamber]:
                    start_time, machine_id = chamber_slots[from_chamber].pop(token_id)
                    config = chamber_config[from_chamber]
                    proc_end = start_time + config["proc_time"]
                    # 确保 machine_id 在有效范围内
                    max_machine = config["capacity"]
                    if machine_id < 0:
                        machine_id = 0
                    elif machine_id >= max_machine:
                        machine_id = max_machine - 1
                    ops.append(Op(
                        job=token_id,
                        stage=config["stage"],
                        machine=machine_id,
                        start=start_time,
                        proc_end=proc_end,
                        end=t1,  # 离开时间
                    ))
            
            # ========== 处理机械手动作 ==========
            # 所有 u_ 和 t_ 变迁都涉及机械手动作
            if "robot_name" in log:
                robot_name = log["robot_name"]
                stage = robot_stage_map.get(robot_name, 6)
                from_loc = log.get("from_loc", "")
                to_loc = log.get("to_loc", "")
                # kind: 0=卸载(u_), 1=装载(t_)
                kind = 0 if t_name.startswith("u_") else 1
                ops.append(Op(
                    job=token_id,
                    stage=stage,
                    machine=0,  # 机械手只有一个
                    start=t1,
                    proc_end=t2,
                    end=t2,
                    is_arm=True,
                    kind=kind,
                    from_loc=from_loc,
                    to_loc=to_loc,
                ))
        
        # ========== 处理未完成的腔室占用（仍在加工中的晶圆）==========
        # 如果系统结束时还有晶圆在腔室中，使用当前时间作为结束时间
        current_time = self.time if hasattr(self, 'time') else max([log.get("t2", 0) for log in self.fire_log], default=0)
        for chamber_name, slots_dict in chamber_slots.items():
            config = chamber_config[chamber_name]
            for tid, (start_time, machine_id) in slots_dict.items():
                proc_end = start_time + config["proc_time"]
                # 确保 machine_id 在有效范围内
                max_machine = config["capacity"]
                if machine_id < 0:
                    machine_id = 0
                elif machine_id >= max_machine:
                    machine_id = max_machine - 1
                ops.append(Op(
                    job=tid,
                    stage=config["stage"],
                    machine=machine_id,
                    start=start_time,
                    proc_end=proc_end,
                    end=current_time,  # 使用当前时间
                ))
        
        # ========== 生成甘特图 ==========
        # 配置 proc_time 和 capacity
        proc_time = {
            1: 70,   # s1
            2: 0,    # s2
            3: 600,  # s3
            4: 70,   # s4
            5: 200,  # s5
            6: 0,    # TM2 (机械手，无加工时间)
            7: 0,    # TM3 (机械手，无加工时间)
        }
        capacity = {
            1: 2,  # s1
            2: 1,  # s2
            3: 4,  # s3
            4: 1,  # s4
            5: 2,  # s5
            6: 1,  # TM2
            7: 1,  # TM3
        }
        
        # ========== 数据验证和修正 ==========
        # 1. 检查空 ops
        if not ops:
            print("警告: 没有操作数据，跳过甘特图绘制")
            return
        
        # 2. 验证和修正 Op 数据
        valid_ops = []
        invalid_count = 0
        for op in ops:
            # 检查 stage 是否在 proc_time 中
            if op.stage not in proc_time:
                invalid_count += 1
                continue
            
            # 检查 machine 是否在有效范围内
            max_machine = capacity.get(op.stage, 0)
            if max_machine <= 0:
                invalid_count += 1
                continue
            
            # 修正 machine_id 范围
            if op.machine < 0:
                op.machine = 0
            elif op.machine >= max_machine:
                op.machine = max_machine - 1
            
            # 验证时间范围
            if op.start < 0 or op.proc_end < op.start or op.end < op.proc_end:
                invalid_count += 1
                continue
            
            valid_ops.append(op)
        
        if invalid_count > 0:
            print(f"警告: 过滤了 {invalid_count} 个无效的 Op")
        
        if not valid_ops:
            print("警告: 验证后没有有效的操作数据，跳过甘特图绘制")
            return
        
        # 3. 计算 n_jobs（唯一晶圆数量）
        unique_jobs = set()
        for op in valid_ops:
            if op.job >= 0:  # 排除资源 token（token_id=-1）
                unique_jobs.add(op.job)
        n_jobs = max(1, len(unique_jobs))
        
        # 4. 保存 ops 到文件以便检查
        import os
        import json
        output_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else "."
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存 ops 数据
        ops_data = []
        for op in valid_ops:
            ops_data.append({
                "job": op.job,
                "stage": op.stage,
                "machine": op.machine,
                "start": op.start,
                "proc_end": op.proc_end,
                "end": op.end,
                "is_arm": op.is_arm,
                "kind": op.kind,
                "from_loc": op.from_loc,
                "to_loc": op.to_loc,
            })
        
        # 保存到 JSON 文件
        ops_file = os.path.join(output_dir, "ops_debug.json")
        if out_path.endswith('.png'):
            base_name = os.path.splitext(os.path.basename(out_path))[0]
            ops_file = os.path.join(output_dir, f"{base_name}_ops.json")
        else:
            base_name = os.path.basename(out_path) if os.path.basename(out_path) else "gantt"
            ops_file = os.path.join(output_dir, f"{base_name}_ops.json")
        
        try:
            with open(ops_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "n_jobs": n_jobs,
                    "unique_jobs": sorted(list(unique_jobs)),
                    "ops_count": len(valid_ops),
                    "ops": ops_data
                }, f, indent=2, ensure_ascii=False)
            print(f"  Ops 数据已保存到: {ops_file}")
        except Exception as e:
            print(f"  警告: 保存 ops 数据失败: {e}")
        
        # 5. 确保输出目录存在并处理路径
        # plot_gantt_hatched_residence 会在路径后追加 "{policy}_job{n_jobs}.png"
        # 所以如果 out_path 已经包含 .png，需要去掉扩展名
        if out_path.endswith('.png'):
            base_path = out_path[:-4]  # 去掉 .png
        else:
            base_path = out_path
        
        # 6. 调用绘图函数
        arm_info = {"ARM1": [], "ARM2": [], "STAGE2ACT": {}}
        try:
            plot_gantt_hatched_residence(
                ops=valid_ops,
                proc_time=proc_time,
                capacity=capacity,
                n_jobs=n_jobs,
                out_path=base_path,  # 传入不带 .png 的路径
                arm_info=arm_info,
                with_label=True,
                no_arm=True,  # 显示机械手
                policy=2,
            )
            # 实际生成的文件名
            policy_dict = {0: 'pdr', 1: 'random', 2: 'rl'}
            actual_path = f"{base_path}{policy_dict[2]}_job{n_jobs}.png"
            print(f"甘特图已成功生成: {actual_path}")
        except Exception as e:
            print(f"绘制甘特图时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise


    def get_enable_t(self) -> List[int]:
        te = self._resource_enable()
        use_t = []
        for t in te:
            el = self._earliest_enable_time(t=t)
            if self.time >= el:
                use_t.append(int(t))
        return use_t

    def get_enable_t_by_robot(self) -> Tuple[List[int], List[int]]:
        """
        返回两个机械手各自可用的变迁列表。
        
        Returns:
            (tm2_enabled, tm3_enabled): 两个列表，分别是 TM2 和 TM3 当前可使能的变迁索引
        """
        all_enabled = self.get_enable_t()
        tm2_enabled = []
        tm3_enabled = []
        
        for t in all_enabled:
            t_name = self.id2t_name[t]
            if t_name in TM2_TRANSITIONS:
                tm2_enabled.append(t)
            elif t_name in TM3_TRANSITIONS:
                tm3_enabled.append(t)
        
        return tm2_enabled, tm3_enabled


    def _check_scrap(self, return_info: bool = False) -> bool | Tuple[bool, Optional[Dict]]:
        """
        检查是否有报废的 wafer。报废：晶圆在腔室停留的时间超过 processing_time + Residual_time 秒。
        
        Args:
            return_info: 是否返回报废详情
            
        Returns:
            如果 return_info=False: 返回 bool
            如果 return_info=True: 返回 (bool, scrap_info)
                scrap_info = {"place": 库所名, "enter_time": 进入时间, "overtime": 超时秒数}
        """
        for place in self.marks:
            if place.type == 1:
                for tok in place.tokens:
                    overtime = (self.time - tok.enter_time) - place.processing_time - self.P_Residual_time
                    if overtime > 0:
                        if return_info:
                            return True, {
                                "place": place.name,
                                "enter_time": tok.enter_time,
                                "stay_time": self.time - tok.enter_time,
                                "proc_time": place.processing_time,
                                "overtime": overtime,
                            }
                        return True
        if return_info:
            return False, None
        return False

    def _check_idle_timeout(self) -> bool:
        """
        检查是否连续执行 WAIT 动作超过阈值时间（停滞检测）
        
        Returns:
            True 如果连续 WAIT 时间超过 idle_timeout
        """
        return self._consecutive_wait_time >= self.idle_timeout

    def step(self, t: Optional[int] = None, wait: bool = False, with_reward: bool = False, 
             detailed_reward: bool = False):
        """
        执行一步动作。
        
        Args:
            t: 要执行的变迁索引（当 wait=False 时）
            wait: 是否执行 WAIT 动作
            with_reward: 是否返回奖励
            detailed_reward: 是否返回详细奖励分解（仅当 with_reward=True 时有效）
            
        Returns:
            如果 with_reward=True 且 detailed_reward=False: (done, reward, scrap)
            如果 with_reward=True 且 detailed_reward=True: (done, reward_dict, scrap)
            否则: (done, scrap)
            
            done=True 表示 episode 结束（完成或报废）
            scrap=True 表示因报废而结束
        """
        if self.time >= MAX_TIME:
            if with_reward:
                if detailed_reward:
                    return True, {"total": -100, "timeout": True}, True
                return True, -100, True  # 超时视为报废
            return True, True

        if wait:
            t1 = self.time

            enabled = self.get_enable_t()

            t2 = self.time + 5
            #if len(enabled) > 0:
            #    t2 = self.time + 5   # 有其他动作可用，小步等待 5
            #else:
                #t2 = self.next_enable_time()  # 没有任何动作，跳到下一可使能时间

            # 累计连续 WAIT 时间
            self._consecutive_wait_time += (t2 - t1)

            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.time = t2
            self._update_stay_times()
            
            # 检查报废
            is_scrap, scrap_info = self._check_scrap(return_info=True)
            if is_scrap:
                self.scrap_count += 1
                done = self.stop_on_scrap  # 根据设置决定是否结束
                if with_reward:
                    if detailed_reward:
                        reward_result["scrap_penalty"] = -self.R_scrap
                        reward_result["total"] -= self.R_scrap
                        reward_result["scrap_info"] = scrap_info
                        return done, reward_result, True
                    return done, reward_result - self.R_scrap, True
                return done, True
            
            # 检查停滞（连续执行 WAIT 超过阈值时间）
            if self._check_idle_timeout() and not self._idle_penalty_applied:
                self._idle_penalty_applied = True  # 只惩罚一次
                if with_reward:
                    if detailed_reward:
                        reward_result["idle_timeout_penalty"] = -self.idle_penalty
                        reward_result["total"] -= self.idle_penalty
                    else:
                        reward_result -= self.idle_penalty
            
            if with_reward:
                return False, reward_result, False
            return False, False

        # 执行变迁动作
        # 重置连续 WAIT 时间（执行非 WAIT 动作）
        self._consecutive_wait_time = 0
        # 重置释放时间违规惩罚累积器
        self._release_violation_penalty = 0.0
        
        t1 = self.time
        t2 = self.time + self.ttime
        pre_places = np.flatnonzero(self.pre[:, t] > 0)
        reward_result = self.calc_reward(t1, t2, moving_pre_places=pre_places, detailed=detailed_reward)
        self._fire(t=t)
        
        # 将释放时间违规惩罚加入奖励
        if self._release_violation_penalty > 0:
            if with_reward:
                if detailed_reward:
                    reward_result["release_violation_penalty"] = -self._release_violation_penalty
                    reward_result["total"] -= self._release_violation_penalty
                else:
                    reward_result -= self._release_violation_penalty
        
        # 添加单片完工奖励（在 _fire 中累积）
        if self._per_wafer_reward > 0:
            if with_reward:
                if detailed_reward:
                    reward_result["wafer_done_bonus"] = self._per_wafer_reward
                    reward_result["total"] += self._per_wafer_reward
                else:
                    reward_result += self._per_wafer_reward
            self._per_wafer_reward = 0.0  # 清零

        # 检查完成
        lp_done_idx = self._get_place_index("LP_done")
        finish = (int(self.m[lp_done_idx]) == self.n_wafer)
        
        # 全部完工大奖励
        if finish:
            if with_reward:
                if detailed_reward:
                    reward_result["finish_bonus"] = self.R_finish+7000
                    reward_result["total"] += self.R_finish
                else:
                    reward_result += self.R_finish
        
        # fire 后也检查报废
        is_scrap, scrap_info = self._check_scrap(return_info=True)
        if is_scrap:
            self.scrap_count += 1
            done = self.stop_on_scrap  # 根据设置决定是否结束
            if with_reward:
                if detailed_reward:
                    reward_result["scrap_penalty"] = -self.R_scrap
                    reward_result["total"] -= self.R_scrap
                    reward_result["scrap_info"] = scrap_info
                    return done, reward_result, True
                return done, reward_result - self.R_scrap, True
            return done, True

        if with_reward:
            return finish, reward_result, False
        return finish, False

    def _check_robot_conflict(self, a1: Optional[int], a2: Optional[int]) -> bool:
        """
        检查两个动作是否产生资源冲突（访问同一非资源库所）。
        
        Args:
            a1: TM2 的变迁索引，None 表示等待
            a2: TM3 的变迁索引，None 表示等待
            
        Returns:
            True 表示无冲突，False 表示有冲突
        """
        if a1 is None or a2 is None:
            return True  # 有一个等待，不冲突
        
        # 获取两个变迁的前置和后置库所
        pre1 = set(np.flatnonzero(self.pre[:, a1] > 0))
        pst1 = set(np.flatnonzero(self.pst[:, a1] > 0))
        pre2 = set(np.flatnonzero(self.pre[:, a2] > 0))
        pst2 = set(np.flatnonzero(self.pst[:, a2] > 0))
        
        # 获取资源库所索引（r_TM2, r_TM3）
        resource_indices = set()
        for i, name in enumerate(self.id2p_name):
            if name.startswith("r_"):
                resource_indices.add(i)
        
        # 检查是否有重叠（排除资源库所）
        affected1 = (pre1 | pst1) - resource_indices
        affected2 = (pre2 | pst2) - resource_indices
        
        return len(affected1 & affected2) == 0

    def step_concurrent(self, a1: Optional[int] = None, a2: Optional[int] = None,
                        wait1: bool = False, wait2: bool = False,
                        with_reward: bool = False, detailed_reward: bool = False):
        """
        并发执行两个机械手的动作。
        
        采用同步执行策略：两个动作同时开始，时间推进到两者都完成。
        
        Args:
            a1: TM2 的动作（变迁索引），或 None 表示等待
            a2: TM3 的动作（变迁索引），或 None 表示等待
            wait1: TM2 是否执行 WAIT（当 a1 为 None 时）
            wait2: TM3 是否执行 WAIT（当 a2 为 None 时）
            with_reward: 是否返回奖励
            detailed_reward: 是否返回详细奖励分解
            
        Returns:
            如果 with_reward=True: (done, reward, scrap)
            否则: (done, scrap)
        """
        # 检查资源冲突
        if not self._check_robot_conflict(a1, a2):
            # 冲突时，优先执行 TM2 的动作，TM3 等待
            a2 = None
            wait2 = True
        
        # 处理等待参数
        if a1 is None:
            wait1 = True
        if a2 is None:
            wait2 = True
        
        # 如果两个都是等待
        if wait1 and wait2:
            return self.step(wait=True, with_reward=with_reward, detailed_reward=detailed_reward)
        
        # 如果只有一个动作
        if wait1 and not wait2:
            return self.step(t=a2, wait=False, with_reward=with_reward, detailed_reward=detailed_reward)
        if wait2 and not wait1:
            return self.step(t=a1, wait=False, with_reward=with_reward, detailed_reward=detailed_reward)
        
        # 两个都有动作：同步执行
        # 策略：先执行 a1，再执行 a2（简化实现，后续可优化为真正并发）
        t1 = self.time
        
        # 执行第一个动作
        done1, result1, scrap1 = self.step(t=a1, wait=False, with_reward=True, detailed_reward=detailed_reward)
        if done1 or scrap1:
            if with_reward:
                return done1, result1, scrap1
            return done1, scrap1
        
        # 执行第二个动作
        done2, result2, scrap2 = self.step(t=a2, wait=False, with_reward=True, detailed_reward=detailed_reward)
        
        # 合并奖励
        if detailed_reward and isinstance(result1, dict) and isinstance(result2, dict):
            combined_reward = {}
            for key in set(result1.keys()) | set(result2.keys()):
                v1 = result1.get(key, 0) if isinstance(result1.get(key, 0), (int, float)) else 0
                v2 = result2.get(key, 0) if isinstance(result2.get(key, 0), (int, float)) else 0
                combined_reward[key] = v1 + v2
            reward = combined_reward
        else:
            r1 = result1.get("total", result1) if isinstance(result1, dict) else result1
            r2 = result2.get("total", result2) if isinstance(result2, dict) else result2
            reward = r1 + r2
        
        done = done1 or done2
        scrap = scrap1 or scrap2
        
        if with_reward:
            return done, reward, scrap
        return done, scrap


    def calc_wafer_statistics(self) -> Dict[str, Any]:
        """
        计算晶圆统计数据，用于可视化界面显示。
        
        改进：结合历史记录 (self.wafer_stats) 和当前状态 (place.tokens)，
        实现累计平均值和最大值，而非瞬时快照。
        """
        if not self.enable_statistics:
            return {}  # 训练模式下跳过统计计算
        
        # 收集数据容器
        chamber_stats = {}    # {place_name: [stay_times]}
        transport_stats = {}  # {place_name: [stay_times]}
        system_times = []     # 所有已完成晶圆的系统停留时间

        # 1. 从历史记录中提取已完成的停留时间
        for wafer_id, stats in self.wafer_stats.items():
            # 系统级统计 (只统计已离开系统的)
            if stats.get("enter_system") is not None and stats.get("exit_system") is not None:
                system_times.append(stats["exit_system"] - stats["enter_system"])
            
            # 腔室历史
            for ch_name, timing in stats.get("chambers", {}).items():
                if timing.get("exit") is not None:
                    duration = timing["exit"] - timing.get("enter", 0)
                    chamber_stats.setdefault(ch_name, []).append(duration)
            
            # 运输历史
            for tr_name, timing in stats.get("transports", {}).items():
                if timing.get("exit") is not None:
                    duration = timing["exit"] - timing.get("enter", 0)
                    transport_stats.setdefault(tr_name, []).append(duration)

        # 2. 从当前 token 补充进行中的停留时间 (可选：如果只想要“历史已完成”统计，可去掉这一步)
        # 用户需求是“恒定记录”，通常意味着包含历史。包含正在进行中的可能会拉低平均值（因为还在增加），
        # 但能反映实时状态。这里我们策略是：
        # - 对于 Avg/Max 计算，通常只算“已完成的工序”比较准确。
        # - 如果算上进行中的，会导致刚进入的 0s 数据拉低平均值。
        # - 因此：只统计已完成的工序（history）+ 当前晶圆在当前位置如果已经停留了很久（比历史均值大）？
        # - 简化策略：只统计已完成的工序记录 (Exit is not None)，这样数据稳步累积，不会跳变。
        # 
        # (稍等，如果只统计已完成，那刚开始没完成时是0。用户说“离开后归零”，说明之前是显示“进行中”的)
        # 结合方案：列表包含 [历史完成时间] + [当前进行中时间]。
        # 为防止 "刚进入" 的短时间拉低平均值，可以只取 max? 不，Avg 应该反映真实负载。
        # 工业控制面板通常显示 "Last N Valid Runs" 或 "Accumulated Sinc Reset".
        # 
        # 决定：包含当前 tokens 的 stay_time，因为这反映了当前的拥堵情况。
        # 只要 wafer 还在里面，它是 active 的。
        
        for place in self.marks:
            if place.name.startswith("r_"): continue
            
            # 获取容器引用
            target_dict = None
            if place.type == 1 or place.type == 5:
                target_dict = chamber_stats
            elif place.type == 2:
                target_dict = transport_stats
                
            if target_dict is not None:
                for token in place.tokens:
                    stay = float(getattr(token, "stay_time", 0))
                    # 只有当 stay_time > 0 才计入，避免刚进入的 0 拉低平均
                    if stay > 0:
                        target_dict.setdefault(place.name, []).append(stay)
            
            # 系统级在线统计不易准确计算（因为还没结束），暂只统计已完成的系统时间
            # 或者：对于进行中的 wafer，计算 (current_time - enter_system_time)
            # 但 pn.py 里 token 没直接存 enter_system_time，得去 wafer_stats 查
            # 暂时只统计已完成的系统周期
        
        # 3. 计算聚合指标
        
        # 系统级
        if system_times:
            sys_avg = float(np.mean(system_times))
            sys_max = float(np.max(system_times))
            sys_min = float(np.min(system_times))
            sys_diff = sys_max - sys_min
        else:
            sys_avg = 0.0
            sys_max = 0.0
            sys_diff = 0.0
            
        # 辅助函数
        def calc_metric(times):
            if not times: return {"avg": 0.0, "max": 0.0}
            return {
                "avg": float(np.mean(times)),
                "max": float(np.max(times))
            }

        # 腔室级
        chambers_result = {}
        # 确保所有腔室都有 key，即使没数据
        all_chambers = ["s1", "s2", "s3", "s4", "s5"] # 常用腔室
        # 加上 map 中存在的
        for p in self.marks:
            if (p.type==1 or p.type==5) and p.name not in all_chambers:
                all_chambers.append(p.name)
        
        for name in all_chambers:
            chambers_result[name] = calc_metric(chamber_stats.get(name, []))

        # 运输级详细
        transports_detail = {}
        for name, times in transport_stats.items():
            transports_detail[name] = calc_metric(times)
            
        # 机械手聚合 (TM2/TM3)
        tm2_places = ["d_s1", "d_s2", "d_s5", "d_LP_done"]
        tm3_places = ["d_s3", "d_s4"]
        
        tm2_times = []
        tm3_times = []
        
        for k, v in transport_stats.items():
            if k in tm2_places: tm2_times.extend(v)
            elif k in tm3_places: tm3_times.extend(v)
            
        transports_result = {
            "TM2": calc_metric(tm2_times),
            "TM3": calc_metric(tm3_times)
        }

        # 计数
        completed_count = int(getattr(self, "done_count", 0))
        in_progress_count = sum(len(p.tokens) for p in self.marks if not p.name.startswith("r_"))

        return {
            "system_avg": sys_avg,
            "system_max": sys_max,
            "system_diff": sys_diff,
            "completed_count": completed_count,
            "in_progress_count": in_progress_count,
            "chambers": chambers_result,
            "transports": transports_result,
            "transports_detail": transports_detail,
        }



def main():
    #np.random.seed(0)  # 可删，保证复现
    model = Petri()
    model.reset()

    wait_id = 10
    print("Transitions:", model.id2t_name)
    print("Places:", model.id2p_name)
    print("-" * 70)

    max_steps = 300
    for step_i in range(max_steps):
        enabled = model.get_enable_t()
        enabled.append(wait_id)

        t = int(np.random.choice(enabled))
        if t==wait_id:
           print(f"  -> fire t=wait  at time={model.time}")
        else:
            print(f"  -> fire t={t} ({model.id2t_name[t]})at time={model.time}")

        if t == wait_id:
            finish = model.step(wait=True)
        else:
            finish = model.step(t=t, wait=False)

        if finish:
            print(f"[DONE] Finished at time={model.time}, step={step_i}")
            model.render_gantt()
            break


if __name__ == "__main__":
    main()

