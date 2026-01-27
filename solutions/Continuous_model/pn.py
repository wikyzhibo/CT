from collections import deque
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union
from visualization.plot import plot_gantt_hatched_residence, Op
from solutions.Continuous_model.construct import SuperPetriBuilder, ModuleSpec, RobotSpec, BasedToken
from data.petri_configs.env_config import PetriEnvConfig

INF = 10**6
MAX_TIME = 5000  # 例如 300s

@dataclass(slots=False)  # 不能使用 slots=True，因为 tokens 和 release_schedule 是可变字段（deque）
class Place:
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
                 reward_config: Optional[Dict[str, int]] = None) -> None:
        """
        初始化Petri网环境。
        
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
        self.turbo_mode = getattr(config, 'turbo_mode', False)
        self.optimize_reward_calc = getattr(config, 'optimize_reward_calc', True)
        self.optimize_enable_check = getattr(config, 'optimize_enable_check', True)
        self.optimize_state_update = getattr(config, 'optimize_state_update', True)
        self.cache_indices = getattr(config, 'cache_indices', True)
        self.optimize_data_structures = getattr(config, 'optimize_data_structures', True)
        
        # 内部状态
        self.scrap_count = 0  # 报废计数器
        self._idle_penalty_applied = False  # 标记是否已施加过停滞惩罚
        self._consecutive_wait_time = 0  # 连续执行 WAIT 动作的累计时间
        self._per_wafer_reward = 0.0  # 累积的单片完工奖励
        
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
        self.n_wafer_route1 = getattr(config, 'n_wafer_route1', self.n_wafer // 2)
        self.n_wafer_route2 = getattr(config, 'n_wafer_route2', self.n_wafer - self.n_wafer // 2)
        
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
        # 新路线拓扑：LP -> s1 -> s2 -> s3 -> s4 -> s5 -> LP_done
        self.downstream_map = {}  # 堵塞检测已关闭，保留空映射

        # 错峰启动：记录 t_PM1 上次发射时间
        self._last_t_pm1_fire_time = -INF
        
        # ============ 性能优化：构建缓存的索引映射（必须在 _build_release_chain 之前）============
        if self.cache_indices:
            self._build_cached_indices()
        else:
            self._place_indices = None
            self._transition_indices = None
        
        # 极速模式缓存初始化
        self._pre_places_cache = {}
        self._pst_places_cache = {}
        
        # 缓存关键变迁索引
        self._t_LP_done_idx = self._get_transition_index("t_LP_done") if "t_LP_done" in self.id2t_name else -1
        self._lp_done_idx = self._get_place_index("LP_done") if "LP_done" in self.id2p_name else -1
        
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
        这里只构建路线1的链路（s1 -> s3 -> s5），路线2的 s1 -> s5 在运行时处理。
        """
        self.release_chain: Dict[int, Tuple[int, int]] = {}
        
        # 获取腔室索引（如果存在）
        def get_idx(name: str) -> int:
            if name in self.id2p_name:
                return self._get_place_index(name) if self.cache_indices else self.id2p_name.index(name)
            return -1
        
        s1_idx = get_idx("s1")
        s3_idx = get_idx("s3")
        s5_idx = get_idx("s5")
        
        # 路线1的链路（有驻留约束的腔室之间）
        # s1 -> s3：经过 s2（交接点），预估运输时间
        if s1_idx >= 0 and s3_idx >= 0:
            # 运输时间 = s1卸载 + d_s2运输 + s2停留 + d_s3运输 + s3装载
            transport_s1_to_s3 = self.ttime * 4  # 约 20s
            self.release_chain[s1_idx] = (s3_idx, transport_s1_to_s3)
        
        # s3 -> s5：经过 s4（交接点），预估运输时间
        if s3_idx >= 0 and s5_idx >= 0:
            # 运输时间 = s3卸载 + d_s4运输 + s4加工(70s) + d_s5运输 + s5装载
            transport_s3_to_s5 = self.ttime * 4 + 70  # 约 90s
            self.release_chain[s3_idx] = (s5_idx, transport_s3_to_s5)
        
        # 路线2的链路：s1 -> s5（直接）
        # 存储为单独的映射，在运行时根据颜色选择
        self.release_chain_route2: Dict[int, Tuple[int, int]] = {}
        if s1_idx >= 0 and s5_idx >= 0:
            # 运输时间 = s1卸载 + d_s5运输 + s5装载
            transport_s1_to_s5 = self.ttime * 2  # 约 10s
            self.release_chain_route2[s1_idx] = (s5_idx, transport_s1_to_s5)

    def _check_release_violation(self, place_idx: int, expected_enter_time: int) -> float:
        """
        检查晶圆预计进入时间是否违反腔室的释放约束。
        
        违规条件：队列已满（len >= capacity）且 expected_enter_time < earliest_release
        
        Args:
            place_idx: 目标腔室索引
            expected_enter_time: 预计进入时间
            
        Returns:
            惩罚值（0 表示不违规）
        """
        place = self.marks[place_idx]
        
        # 如果队列未满，不违规
        if len(place.release_schedule) < place.capacity:
            return 0.0
        
        earliest = place.earliest_release()
        if earliest is None or expected_enter_time >= earliest:
            return 0.0  # 不违规
        
        # 违规：计算惩罚（与违规时间差成正比）
        violation_gap = earliest - expected_enter_time
        return self.c_release_violation * violation_gap

    def _record_initial_release(self, token_id: int, enter_d_time: int, 
                                 target_place_idx: int, wafer_color: int = 0) -> float:
        """
        晶圆进入运输位时，记录初步预估的释放时间，并链式传播到下游腔室。
        
        Args:
            token_id: 晶圆编号
            enter_d_time: 进入运输位的时间
            target_place_idx: 目标腔室索引
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
            
        Returns:
            违规惩罚值
        """
        target_place = self.marks[target_place_idx]
        
        # 计算预估进入时间和释放时间
        # 预估进入时间 = enter_d_time + T_transport + T_load
        expected_enter = enter_d_time + self.T_transport + self.T_load
        # 预估释放时间 = expected_enter + processing_time
        release_time = expected_enter + target_place.processing_time
        
        # 检查违规
        penalty = 0.0
        if self.reward_config.get('release_violation_penalty', 1):
            penalty = self._check_release_violation(target_place_idx, expected_enter)
        
        # 记录释放时间
        target_place.add_release(token_id, release_time)
        
        # 链式传播到下游腔室（根据颜色选择链路）
        penalty += self._chain_record_release(token_id, target_place_idx, release_time, wafer_color)
        
        return penalty

    def _chain_record_release(self, token_id: int, start_place_idx: int, 
                               start_release_time: int, wafer_color: int = 0) -> float:
        """
        链式记录/更新下游腔室的预估释放时间。
        
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
        # 路线1（color=1）：使用 release_chain（s1 -> s3 -> s5）
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
            # 下游预估释放时间 = 进入时间 + 加工时间
            downstream_release = downstream_enter + downstream_place.processing_time
            
            # 检查下游违规
            if self.reward_config.get('release_violation_penalty', 1):
                penalty += self._check_release_violation(downstream_idx, downstream_enter)
            
            # 记录下游释放时间
            downstream_place.add_release(token_id, downstream_release)
            
            # 继续链式传播
            current_idx = downstream_idx
            current_release = downstream_release
        
        return penalty

    def _update_release(self, token_id: int, actual_enter_time: int, 
                        place_idx: int, wafer_color: int = 0) -> None:
        """
        晶圆实际进入腔室时，更新精确的释放时间，并链式更新下游。
        
        注意：更新操作不进行违规检测，因为这是更新已有晶圆的释放时间。
        
        Args:
            token_id: 晶圆编号
            actual_enter_time: 实际进入时间
            place_idx: 腔室索引
            wafer_color: 晶圆颜色（1=路线1, 2=路线2）
        """
        place = self.marks[place_idx]
        
        # 计算新的释放时间
        new_release_time = actual_enter_time + place.processing_time
        
        # 更新当前腔室的释放时间
        place.update_release(token_id, new_release_time)
        
        # 链式更新下游（不检查违规）
        self._chain_update_release(token_id, place_idx, new_release_time, wafer_color)

    def _chain_update_release(self, token_id: int, start_place_idx: int, 
                               start_release_time: int, wafer_color: int = 0) -> None:
        """
        链式更新下游腔室中指定晶圆的预估释放时间。
        
        注意：更新操作不进行违规检测，因为这是更新已有晶圆的释放时间。
        
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
            # 下游预估释放时间 = 进入时间 + 加工时间
            downstream_release = downstream_enter + downstream_place.processing_time
            
            # 更新下游释放时间（不检查违规）
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
        # 清空所有库所的释放时间队列，并重置机器分配计数器
        for place in self.marks:
            place.release_schedule.clear()
            if place.type == 1:
                place.last_machine = -1  # 重置机器轮换计数器
        # 性能优化：重置极速模式缓存（网络结构不变，但状态变了）
        # 注意：_pre_places_cache 和 _pst_places_cache 不需要重置，因为网络结构不变
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
        if self.turbo_mode:
            return self._calc_reward_turbo(t1, t2, detailed)
        elif self.optimize_reward_calc:
            return self._calc_reward_vectorized(t1, t2, moving_pre_places, detailed)
        else:
            return self._calc_reward_original(t1, t2, moving_pre_places, detailed)
    
    def _calc_reward_turbo(self, t1: int, t2: int, detailed: bool = False) -> float | Dict[str, float]:
        """
        极速版本的奖励计算（最大性能优化）。
        
        简化计算，只保留核心奖励/惩罚，跳过复杂的堵塞检测等。
        """
        proc_reward = 0.0
        overtime_penalty = 0.0
        p_res = self.P_Residual_time
        
        # 使用按类型分组的缓存，只遍历有驻留约束的加工腔室（type=1）
        # 优化：直接使用缓存字典，避免函数调用开销
        if self.optimize_data_structures:
            process_places = self._marks_by_type.get(1, [])
        else:
            process_places = self._process_places_cache
        for place in process_places:
            ptime = place.processing_time
            for tok in place.tokens:
                enter = tok.enter_time
                proc_end = enter + ptime
                
                # 加工奖励（内联 min/max）
                overlap_start = t1 if t1 > enter else enter
                overlap_end = t2 if t2 < proc_end else proc_end
                if overlap_end > overlap_start:
                    proc_reward += (overlap_end - overlap_start) * 2  # r=2
                
                # 超时惩罚
                pen_start = proc_end + p_res
                if t2 > pen_start:
                    pen_overlap_start = t1 if t1 > pen_start else pen_start
                    overtime_penalty += (t2 - pen_overlap_start) * 0.2  # Q2_p=0.2
        
        # 时间成本
        total = proc_reward - overtime_penalty - self.c_time * (t2 - t1)
        
        if detailed:
            return {
                "total": total,
                "proc_reward": proc_reward,
                "safe_reward": 0.0,
                "penalty": overtime_penalty,
                "warn_penalty": 0.0,
                "transport_penalty": 0.0,
                "congestion_penalty": 0.0,
                "time_cost": self.c_time * (t2 - t1),
            }
        
        return total
    
    def _calc_reward_original(self, t1: int, t2: int, moving_pre_places: Optional[np.ndarray] = None,
                              detailed: bool = False) -> float | Dict[str, float]:
        """原始奖励计算实现（用于功能一致性验证）"""
        moving_set = set(moving_pre_places.tolist()) if moving_pre_places is not None else set()
        
        # 惩罚/奖励系数
        Q1_p = 3      # type=2 运输库所超时惩罚系数
        Q2_p = 0.2      # type=1 加工腔室超时惩罚系数
        r = 2         # 加工奖励系数
        
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
                            transport_penalty += (t2 - over_start) * Q1_p

                # type=5: 无驻留约束腔室（如 s2/LLC、s4/LLD）- 只计加工奖励，无惩罚
                elif place.type == 5:
                    if place.processing_time > 0 and self.reward_config.get('proc_reward', 1):
                        proc_end = tok.enter_time + place.processing_time
                        proc_overlap = min(t2, proc_end) - max(t1, tok.enter_time)
                        if proc_overlap > 0:
                            proc_reward += proc_overlap * r

                # 查询加工腔室相关奖励/惩罚（type=1 有驻留约束）
                elif place.type == 1:
                    # 加工腔室
                    proc_start = tok.enter_time
                    proc_end = tok.enter_time + place.processing_time
                    
                    # 1) 加工奖励：在加工时间内每秒 +r
                    if self.reward_config.get('proc_reward', 1):
                        proc_overlap = min(t2, proc_end) - max(t1, proc_start)
                        if proc_overlap > 0:
                            proc_reward += proc_overlap * r
                    
                    # 2) 超时惩罚：超过 proc_time+ P_Residual_time 后惩罚
                    if self.reward_config.get('penalty', 1):
                        start_pen = tok.enter_time + place.processing_time + self.P_Residual_time
                        start = max(t1, start_pen)
                        if t2 > start:
                            overtime_penalty += (t2 - start) * Q2_p
                    
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
        if self.turbo_mode:
            return self._earliest_enable_time_turbo(t)
        
        pre_places = np.flatnonzero(self.pre[:, t] > 0)
        earliest = 0
        for p in pre_places:
            tok_enter = int(self.marks[p].head().enter_time)  # 队头
            tok_enter += int(self.ptime[p])
            earliest = max(earliest, tok_enter)
        
        return int(earliest)
    
    def _earliest_enable_time_turbo(self, t):
        """极速版本的最早使能时间计算"""
        # 使用缓存的前置库所索引
        pre_places = self._pre_places_cache.get(t)
        if pre_places is None:
            pre_places = np.flatnonzero(self.pre[:, t] > 0)
            self._pre_places_cache[t] = pre_places
        
        earliest = 0
        marks = self.marks
        ptime = self.ptime
        for p in pre_places:
            tok_enter = marks[p].tokens[0].enter_time + ptime[p]
            if tok_enter > earliest:
                earliest = tok_enter
        
        return earliest

    def _resource_enable(self):
        # 双路线变迁:
        # 路线1: ['u_LP1_s1', 't_s1', 'u_s1_s2', 't_s2', 'u_s2_s3', 't_s3', 
        #         'u_s3_s4', 't_s4', 'u_s4_s5', 't_s5', 'u_s5_LP_done', 't_LP_done']
        # 路线2: ['u_LP2_s1', 't_s1', 'u_s1_s5', 't_s5', 'u_s5_LP_done', 't_LP_done']
        # 库所: ['LP1', 'LP2', 'LP_done', 's1', 's2', 's3', 's4', 's5', 
        #        'r_TM2', 'r_TM3', 'd_s1', 'd_s2', 'd_s3', 'd_s4', 'd_s5', 'd_LP_done']
        
        if self.turbo_mode:
            return self._resource_enable_turbo()
        
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
        
        # ========== 颜色感知的分流约束 ==========
        # 在 s1 处根据晶圆颜色决定走哪条路线
        # u_s1_s2 只对 color=1 的晶圆使能（路线1）
        # u_s1_s5 只对 color=2 的晶圆使能（路线2）
        if "s1" in self.id2p_name:
            s1_idx = self._get_place_index("s1")
            s1_place = self.marks[s1_idx]
            
            if len(s1_place.tokens) > 0:
                head_color = s1_place.head().color
                
                # u_s1_s2: 只有颜色1的晶圆可以走 s2（路线1）
                if "u_s1_s2" in self.id2t_name:
                    t_s1_s2_idx = self._get_transition_index("u_s1_s2")
                    if head_color != 1:
                        mask[t_s1_s2_idx] = False
                
                # u_s1_s5: 只有颜色2的晶圆可以走 s5（路线2）
                if "u_s1_s5" in self.id2t_name:
                    t_s1_s5_idx = self._get_transition_index("u_s1_s5")
                    if head_color != 2:
                        mask[t_s1_s5_idx] = False

        out_te = np.flatnonzero(mask)
        return out_te
    
    def _resource_enable_turbo(self):
        """极速版本的使能检查（预计算优化）"""
        m = self.m
        
        # 使用缓存的布尔掩码计算
        if not hasattr(self, '_enable_cache_built'):
            self._build_enable_cache()
        
        # 基本使能条件：前置库所有足够 token 且后置库所不超容量
        # 使用纯 NumPy（避免 JIT 编译开销）
        cond_pre = np.all(self.pre <= m[:, None], axis=0)
        cond_cap = np.all((m[:, None] + self.pst) <= self.k[:, None], axis=0)
        mask = cond_pre & cond_cap
        
        # 使用缓存的索引进行容量约束检查
        for p_idx, t_idx, cap in self._capacity_constraints:
            if m[p_idx] >= cap:
                mask[t_idx] = False
        
        # ========== 颜色感知的分流约束（极速版本）==========
        if self._s1_idx >= 0:
            s1_place = self.marks[self._s1_idx]
            if len(s1_place.tokens) > 0:
                head_color = s1_place.tokens[0].color
                
                # u_s1_s2: 只有颜色1的晶圆可以走 s2
                if self._t_s1_s2_idx >= 0 and head_color != 1:
                    mask[self._t_s1_s2_idx] = False
                
                # u_s1_s5: 只有颜色2的晶圆可以走 s5
                if self._t_s1_s5_idx >= 0 and head_color != 2:
                    mask[self._t_s1_s5_idx] = False
        
        return np.flatnonzero(mask)
    
    def _build_enable_cache(self):
        """构建使能检查的缓存数据"""
        self._enable_cache_built = True
        self._capacity_constraints = []
        
        # 双路线容量约束
        capacity_constraints = [
            ("s1", "u_LP1_s1"), ("s1", "u_LP2_s1"),
            ("s3", "u_s2_s3"),
            ("s5", "u_s4_s5"), ("s5", "u_s1_s5"),
        ]
        for place_name, u_trans_name in capacity_constraints:
            if place_name in self.id2p_name and u_trans_name in self.id2t_name:
                p_idx = self._get_place_index(place_name)
                t_idx = self._get_transition_index(u_trans_name)
                cap = self.marks[p_idx].capacity
                self._capacity_constraints.append((p_idx, t_idx, cap))
        
        # 缓存分流相关索引
        self._s1_idx = self._get_place_index("s1") if "s1" in self.id2p_name else -1
        self._t_s1_s2_idx = self._get_transition_index("u_s1_s2") if "u_s1_s2" in self.id2t_name else -1
        self._t_s1_s5_idx = self._get_transition_index("u_s1_s5") if "u_s1_s5" in self.id2t_name else -1

    def next_enable_time(self) -> int:
        te = self._resource_enable()
        if len(te) == 0:
            return self.time + 1
        earliest = INF
        for t in te:
            earliest = min(earliest, self._earliest_enable_time(t))
        return int(earliest)

    def _fire(self, t):
        if self.turbo_mode:
            return self._fire_turbo(t)
        
        start_time = self.time
        enter_new = self.time + self.ttime
        t_name = self.id2t_name[t]

        pre_places = np.flatnonzero(self.pre[:, t] > 0)
        pst_places = np.nonzero(self.pst[:, t] > 0)[0]

        # 1) 消费前置库所 token（队头），保存 token_id 和 color
        consumed_token_ids = []
        consumed_colors = []
        for p in pre_places:
            place = self.marks[p]
            # 只有非资源库所的 token 才有 wafer id 和 color
            if place.type != 4:  # 非资源库所
                tok = place.head()
                consumed_token_ids.append(tok.token_id)
                consumed_colors.append(tok.color)
            place.pop_head()
            self.m[p] -= 1

        # 2) 生成后置库所 token：enter_time = finish time，继承 token_id 和 color
        token_id_idx = 0
        for p in pst_places:
            place = self.marks[p]
            # 只有非资源库所才传递 wafer id 和 color
            if place.type != 4 and token_id_idx < len(consumed_token_ids):
                tid = consumed_token_ids[token_id_idx]
                tcolor = consumed_colors[token_id_idx]
                token_id_idx += 1
            else:
                tid = -1  # 资源库所的 token 无 wafer id
                tcolor = 0
            
            # 为 type=1 的加工腔室分配机器（轮换策略）
            if place.type == 1:
                machine_id = (place.last_machine + 1) % place.capacity
                place.last_machine = machine_id
            else:
                machine_id = -1
            
            self.marks[p].append(BasedToken(enter_time=enter_new, stay_time=1, token_id=tid, machine=machine_id, color=tcolor))
            self.m[p] += 1

        # 3) 时间推进到完成之后
        self.time = enter_new
        self._update_stay_times()
        
        # 记录 t_s1 发射时间（用于错峰约束，如需要）
        if t_name == "t_s1":
            self._last_t_pm1_fire_time = start_time
        
        # ========== 4) 释放时间追踪逻辑 ==========
        # 获取晶圆 token_id 和 color（第一个非资源库所消费的 token）
        wafer_id = consumed_token_ids[0] if consumed_token_ids else -1
        wafer_color = consumed_colors[0] if consumed_colors else 0
        
        # 根据变迁类型处理释放时间（只追踪有驻留约束的腔室：s1, s3, s5）
        # 双路线：u_LP1_s1 和 u_LP2_s1 都进入 d_s1
        if t_name in ("u_LP1_s1", "u_LP2_s1"):
            # 晶圆进入 d_s1：记录初步预估释放时间 + 链式传播
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                penalty = self._record_initial_release(wafer_id, enter_new, s1_idx, wafer_color)
                self._release_violation_penalty += penalty
            
        elif t_name == "t_s1":
            # 晶圆进入 s1：更新精确释放时间 + 链式更新
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._update_release(wafer_id, enter_new, s1_idx, wafer_color)
            
        elif t_name == "u_s1_s2":
            # 路线1：晶圆离开 s1 去 s2，从 s1 队列移除
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._pop_release(wafer_id, s1_idx)
        
        elif t_name == "u_s1_s5":
            # 路线2：晶圆离开 s1 直接去 s5，从 s1 队列移除
            if "s1" in self.id2p_name:
                s1_idx = self._get_place_index("s1")
                self._pop_release(wafer_id, s1_idx)
            
        elif t_name == "t_s3":
            # 晶圆进入 s3：更新精确释放时间 + 链式更新
            if "s3" in self.id2p_name:
                s3_idx = self._get_place_index("s3")
                self._update_release(wafer_id, enter_new, s3_idx, wafer_color)
            
        elif t_name == "u_s3_s4":
            # 晶圆离开 s3：从 s3 队列移除
            if "s3" in self.id2p_name:
                s3_idx = self._get_place_index("s3")
                self._pop_release(wafer_id, s3_idx)
            
        elif t_name == "t_s5":
            # 晶圆进入 s5：更新精确释放时间
            if "s5" in self.id2p_name:
                s5_idx = self._get_place_index("s5")
                self._update_release(wafer_id, enter_new, s5_idx, wafer_color)
            
        elif t_name == "u_s5_LP_done":
            # 晶圆离开 s5：从 s5 队列移除
            if "s5" in self.id2p_name:
                s5_idx = self._get_place_index("s5")
                self._pop_release(wafer_id, s5_idx)
        
        elif t_name == "t_LP_done":
            # 晶圆完成加工，给予单片完工奖励
            self._per_wafer_reward += self.R_done
        
        # ========== 5) 晶圆滞留时间统计追踪 ==========
        self._track_wafer_statistics(t_name, wafer_id, start_time, enter_new)
        
        self.fire_log.append({
            "t_name": t_name,
            "t1": int(start_time),
            "t2": int(enter_new),
            "token_id": wafer_id,
        })
    
    def _fire_turbo(self, t):
        """极速版本的变迁执行（跳过所有追踪逻辑）"""
        enter_new = self.time + self.ttime
        
        # 使用缓存的前置/后置库所
        pre_places = self._pre_places_cache.get(t)
        if pre_places is None:
            pre_places = np.flatnonzero(self.pre[:, t] > 0)
            self._pre_places_cache[t] = pre_places
        
        pst_places = self._pst_places_cache.get(t)
        if pst_places is None:
            pst_places = np.flatnonzero(self.pst[:, t] > 0)
            self._pst_places_cache[t] = pst_places
        
        marks = self.marks
        m = self.m
        
        # 1) 消费前置库所 token，保存 token_id 和 color
        consumed_tid = -1
        consumed_color = 0
        for p in pre_places:
            place = marks[p]
            if place.type != 4 and consumed_tid == -1:
                tok = place.tokens[0]
                consumed_tid = tok.token_id
                consumed_color = tok.color
            place.tokens.popleft()
            m[p] -= 1
        
        # 2) 生成后置库所 token（继承 color）
        for p in pst_places:
            place = marks[p]
            if place.type != 4:
                tid = consumed_tid
                tcolor = consumed_color
                consumed_tid = -1  # 只传递一次
                consumed_color = 0
            else:
                tid = -1
                tcolor = 0
            
            # 简化机器分配
            if place.type == 1:
                machine_id = (place.last_machine + 1) % place.capacity
                place.last_machine = machine_id
            else:
                machine_id = -1
            
            place.tokens.append(BasedToken(enter_time=enter_new, stay_time=1, token_id=tid, machine=machine_id, color=tcolor))
            m[p] += 1
        
        # 3) 时间推进
        self.time = enter_new
        
        # 4) 检查完工奖励（只检查 t_LP_done）
        if t == self._t_LP_done_idx:
            self._per_wafer_reward += self.R_done

    def render_gantt(self, out_path: str = r"C:\Users\khand\OneDrive\code\dqn\CT\results\continuous_gantt.png"):
        pm1_slots = [None, None]  # PM1 双 slot 追踪，记录 (start_time, token_id)
        pm2_info = None  # (start_time, token_id)
        pm3_info = None  # (start_time, token_id)
        ops: List[Op] = []

        for log in self.fire_log:
            t_name = log["t_name"]
            t1 = log["t1"]
            t2 = log["t2"]
            token_id = log.get("token_id", -1)

            if t_name == "t_PM1":
                # 找空闲 slot 记录开始时间和 token_id
                for i in range(2):
                    if pm1_slots[i] is None:
                        pm1_slots[i] = (t2, token_id)
                        break
            elif t_name == "u_PM1_PM2":
                # 找最早进入的 slot（FIFO），生成 Op 并释放 slot
                occupied = [i for i in range(2) if pm1_slots[i] is not None]
                if occupied:
                    earliest_idx = min(occupied, key=lambda i: pm1_slots[i][0])
                    pm1_start, pm1_token_id = pm1_slots[earliest_idx]
                    proc_end = t1
                    ops.append(Op(job=pm1_token_id, stage=1, machine=earliest_idx, start=pm1_start, proc_end=pm1_start+30, end=proc_end))
                    pm1_slots[earliest_idx] = None

            if t_name == "t_PM2":
                pm2_info = (t2, token_id)
            elif t_name == "u_PM2_LP_done" and pm2_info is not None:
                pm2_start, pm2_token_id = pm2_info
                proc_end = t1
                ops.append(Op(job=pm2_token_id, stage=2, machine=0, start=pm2_start, proc_end=pm2_start+80, end=proc_end))
                pm2_info = None

            if t_name == "t_PM3":
                pm3_info = (t2, token_id)
            elif t_name == "u_PM3_LP_done" and pm3_info is not None:
                pm3_start, pm3_token_id = pm3_info
                proc_end = t1
                ops.append(Op(job=pm3_token_id, stage=2, machine=1, start=pm3_start, proc_end=pm3_start+80, end=proc_end))
                pm3_info = None

            # 机械手动作：除加工区间外的所有变迁时间段
            if t_name.startswith("u_"):
                kind = 1
            else:
                kind = 0
            ops.append(Op(job=token_id, stage=3, machine=0, start=t1, proc_end=t2, end=t2, is_arm=True, kind=kind))

        proc_time = {1: 0, 2: 0, 3: 0}
        capacity = {1: 2, 2: 1, 3: 1}
        n_jobs = max(1, len(ops))
        arm_info = {"ARM1": [], "ARM2": [], "STAGE2ACT": {}}
        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=proc_time,
            capacity=capacity,
            n_jobs=n_jobs,
            out_path=out_path,
            arm_info=arm_info,
            with_label=False,
            no_arm=True,
            policy=2,
        )


    def get_enable_t(self) -> List[int]:
        if self.turbo_mode:
            return self._get_enable_t_turbo()
        
        te = self._resource_enable()
        use_t = []
        for t in te:
            el = self._earliest_enable_time(t=t)
            if self.time >= el:
                use_t.append(int(t))
        return use_t
    
    def _get_enable_t_turbo(self) -> List[int]:
        """极速版本的使能变迁获取"""
        m = self.m
        
        # 使用缓存的布尔掩码计算
        if not hasattr(self, '_enable_cache_built'):
            self._build_enable_cache()
        
        # 基本使能条件（使用 NumPy 向量化）
        cond_pre = np.all(self.pre <= m[:, None], axis=0)
        cond_cap = np.all((m[:, None] + self.pst) <= self.k[:, None], axis=0)
        mask = cond_pre & cond_cap
        
        # 容量约束检查
        for p_idx, t_idx, cap in self._capacity_constraints:
            if m[p_idx] >= cap:
                mask[t_idx] = False
        
        te = np.flatnonzero(mask)
        if len(te) == 0:
            return []
        
        # 时间条件检查
        current_time = self.time
        result = []
        marks = self.marks
        ptime = self.ptime
        pre_cache = self._pre_places_cache
        pre_mat = self.pre
        
        for t in te:
            # 内联 _earliest_enable_time_turbo
            pre_places = pre_cache.get(t)
            if pre_places is None:
                pre_places = np.flatnonzero(pre_mat[:, t] > 0)
                pre_cache[t] = pre_places
            
            earliest = 0
            for p in pre_places:
                tok_enter = marks[p].tokens[0].enter_time + ptime[p]
                if tok_enter > earliest:
                    earliest = tok_enter
            
            if current_time >= earliest:
                result.append(int(t))
        
        return result

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
        # 极速模式：使用简化版本
        if self.turbo_mode:
            if with_reward and not detailed_reward:
                return self._step_turbo(t, wait)
            elif not with_reward:
                return self._step_turbo_no_reward(t, wait)

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
                    reward_result["finish_bonus"] = self.R_finish
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
    
    def _step_turbo(self, t: Optional[int], wait: bool):
        """
        极速版本的 step 函数（简化版本，只返回 (done, reward, scrap)）
        """
        current_time = self.time
        
        # 超时检查
        if current_time >= MAX_TIME:
            return True, -100.0, True
        
        if wait:
            # WAIT 动作
            t2 = current_time + 5
            
            # 简化奖励计算（内联）
            reward = self._calc_reward_turbo(current_time, t2, detailed=False)
            self.time = t2
            
            # 简化报废检查
            if self._check_scrap_turbo():
                self.scrap_count += 1
                return self.stop_on_scrap, reward - self.R_scrap, True
            
            return False, reward, False
        
        # 执行变迁动作
        ttime = self.ttime
        t2 = current_time + ttime
        
        # 简化奖励计算
        reward = self._calc_reward_turbo(current_time, t2, detailed=False)
        
        # 执行变迁
        self._fire_turbo(t)
        
        # 添加单片完工奖励
        per_wafer = self._per_wafer_reward
        if per_wafer > 0:
            reward += per_wafer
            self._per_wafer_reward = 0.0
        
        # 检查完成
        finish = (self.m[self._lp_done_idx] == self.n_wafer)
        if finish:
            reward += self.R_finish
            return True, reward, False
        
        # 简化报废检查
        if self._check_scrap_turbo():
            self.scrap_count += 1
            return self.stop_on_scrap, reward - self.R_scrap, True
        
        return False, reward, False
    
    def _check_scrap_turbo(self) -> bool:
        """极速版本的报废检查"""
        current_time = self.time
        p_res = self.P_Residual_time
        
        # 使用按类型分组的缓存，只检查有驻留约束的加工腔室（type=1）
        # 优化：直接使用缓存，避免函数调用开销
        if self.optimize_data_structures:
            process_places = self._marks_by_type.get(1, [])
        else:
            process_places = self._process_places_cache
        for place in process_places:
            deadline = place.processing_time + p_res
            for tok in place.tokens:
                if (current_time - tok.enter_time) > deadline:
                    return True
        return False
    
    def _step_turbo_no_reward(self, t: Optional[int], wait: bool):
        """极速版本的 step 函数（不计算奖励，只返回 (done, scrap)）"""
        current_time = self.time
        
        # 超时检查
        if current_time >= MAX_TIME:
            return True, True
        
        if wait:
            # WAIT 动作
            self.time = current_time + 5
            
            # 简化报废检查
            if self._check_scrap_turbo():
                self.scrap_count += 1
                return self.stop_on_scrap, True
            
            return False, False
        
        # 执行变迁动作
        self._fire_turbo(t)
        
        # 检查完成
        finish = (self.m[self._lp_done_idx] == self.n_wafer)
        if finish:
            return True, False
        
        # 简化报废检查
        if self._check_scrap_turbo():
            self.scrap_count += 1
            return self.stop_on_scrap, True
        
        return False, False


def main():
    #np.random.seed(0)  # 可删，保证复现
    model = Petri()
    model.reset()

    wait_id = 10
    print("Transitions:", model.id2t_name)
    print("Places:", model.id2p_name)
    print("-" * 70)

    max_steps = 60
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

