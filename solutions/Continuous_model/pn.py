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
from solutions.Td_petri.resources.interval_utils import Interval, _first_free_time_at, _insert_interval_sorted
from solutions.Continuous_model.construct import SuperPetriBuilder, ModuleSpec, RobotSpec, BasedToken
from data.petri_configs.env_config import PetriEnvConfig
import traceback

INF = 10**6

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


@dataclass
class TransitionFilter:
    """变迁过滤器：描述一个变迁的使能条件（在 construct 阶段预计算）"""
    t_idx: int                    # 变迁索引
    t_name: str                   # 变迁名称
    robot: str                    # 所属机械手 ("TM2" / "TM3")
    # 路由过滤（仅 t_ 变迁需要）：
    # 检查运输库所队头 token 的 (route_type, step) 是否在 allowed_route_steps 中
    check_source_idx: int = -1    # 需要检查的源运输库所索引，-1=不检查
    allowed_route_steps: Optional[Set[Tuple[int, int]]] = None  # 允许的 (route_type, step) 集合
    # 容量约束（仅 u_ 变迁到有限容量腔室时需要）
    capacity_place_idx: int = -1  # 需要检查容量的目标库所索引, -1=不检查
    # 系统入口约束：是否是入口变迁（u_LP1_s1, u_LP2_s1）
    is_entry: bool = False        # 是否受 entered_wafer_count 限制


class TransitionManager:
    """
    变迁使能管理器：在 construct 阶段构建过滤器，运行时高效查询。
    
    整合了原 _resource_enable + get_enable_t + get_enable_t_by_robot 的全部逻辑：
    1. 基本 pre/cap mask
    2. 容量约束（预计算哪些 u_ 变迁需要检查目标库所容量）
    3. 系统晶圆数限制（入口变迁标记）
    4. t_ 变迁路由过滤（预计算 allowed_route_steps）
    5. 最早使能时间过滤
    6. 按 TM2/TM3 分组返回
    """
    
    def __init__(self, petri: "Petri"):
        """从 Petri 网实例构建每个变迁的过滤器。"""
        self._petri = petri
        
        # ---------- 预计算 allowed_route_steps ----------
        # 对每个 t_ 变迁（装入腔室），从 ROUTE_CONFIG 推导允许的 (route_type, step) 组合
        # 例如 t_s1 的目标是 "s1"，在 ROUTE_CONFIG 中 route1[0]="s1", route2[0]="s1"
        # 所以 allowed = {(1,0), (2,0)}
        target_to_route_steps: Dict[str, Set[Tuple[int, int]]] = {}
        for route_type, route in petri.ROUTE_CONFIG.items():
            for step_idx, target_name in enumerate(route):
                if target_name not in target_to_route_steps:
                    target_to_route_steps[target_name] = set()
                target_to_route_steps[target_name].add((route_type, step_idx))
        
        # ---------- 预计算每个 u_ 变迁的容量约束目标 ----------
        # 只有目标库所容量有限（非 INF）的 u_ 变迁需要额外容量检查
        # 原 capacity_constraints: ("s1", "u_LP1_s1"), ("s1", "u_LP2_s1"), 
        #   ("s3", "u_s2_s3"), ("s5", "u_s4_s5"), ("s5", "u_s1_s5")
        # 这些都是 u_ 变迁将晶圆送往有限容量的加工腔室
        u_trans_to_capacity_target: Dict[str, str] = {}
        for t_name in petri.id2t_name:
            if not t_name.startswith("u_"):
                continue
            # 从名称解析：u_{source}_{target}
            parts = t_name.split("_", 2)  # ["u", source, target]
            if len(parts) == 3:
                target_name = parts[2]
            else:
                continue
            # 检查目标库所是否有有限容量且是加工腔室
            if target_name in petri.id2p_name:
                p_idx = petri._get_place_index(target_name)
                place = petri.marks[p_idx]
                if place.capacity < 100:  # 有限容量（非 INF）
                    u_trans_to_capacity_target[t_name] = target_name
        
        # ---------- 预计算 t_ 变迁的源运输库所索引 ----------
        # t_ 变迁消耗 d_TM2 或 d_TM3 中的 token
        t_trans_source: Dict[str, int] = {}
        for t_name in petri.id2t_name:
            if not t_name.startswith("t_"):
                continue
            t_idx = petri._get_transition_index(t_name)
            pre_places = np.flatnonzero(petri.pre[:, t_idx] > 0)
            for p_idx in pre_places:
                p_name = petri.id2p_name[p_idx]
                if p_name.startswith("d_"):  # 运输库所
                    t_trans_source[t_name] = p_idx
                    break
        
        # ---------- 构建所有过滤器 ----------
        self.tm2_filters: List[TransitionFilter] = []
        self.tm3_filters: List[TransitionFilter] = []
        
        for t_idx, t_name in enumerate(petri.id2t_name):
            # 判断机械手归属
            if t_name in TM2_TRANSITIONS:
                robot = "TM2"
            elif t_name in TM3_TRANSITIONS:
                robot = "TM3"
            else:
                continue  # 不属于任何机械手的变迁跳过
            
            f = TransitionFilter(t_idx=t_idx, t_name=t_name, robot=robot)
            
            # t_ 变迁：设置路由过滤
            if t_name.startswith("t_"):
                target_name = t_name[2:]  # t_s1 -> s1
                if target_name in target_to_route_steps:
                    f.allowed_route_steps = target_to_route_steps[target_name]
                if t_name in t_trans_source:
                    f.check_source_idx = t_trans_source[t_name]
            
            # u_ 变迁：设置容量约束
            if t_name.startswith("u_"):
                if t_name in u_trans_to_capacity_target:
                    cap_target = u_trans_to_capacity_target[t_name]
                    f.capacity_place_idx = petri._get_place_index(cap_target)
                
                # 入口变迁标记
                if t_name in ("u_LP1_s1", "u_LP2_s1"):
                    f.is_entry = True
            
            if robot == "TM2":
                self.tm2_filters.append(f)
            else:
                self.tm3_filters.append(f)
    
    def get_enable_t(self) -> Tuple[List[int], List[int]]:
        """
        返回 TM2/TM3 各自可用的变迁列表。
        
        Returns:
            (tm2_enabled, tm3_enabled): 两个列表，分别是 TM2 和 TM3 当前可使能的变迁索引
        """
        petri = self._petri
        
        # ===== 1. 基本 pre/cap mask（向量化） =====
        cond_pre = (petri.pre <= petri.m[:, None]).all(axis=0)
        cond_cap = ((petri.m[:, None] + petri.pst) <= petri.k[:, None]).all(axis=0)
        mask = cond_pre & cond_cap
        
        tm2_enabled = []
        tm3_enabled = []
        
        for filters, result_list in [(self.tm2_filters, tm2_enabled), 
                                     (self.tm3_filters, tm3_enabled)]:
            for f in filters:
                if not mask[f.t_idx]:
                    continue
                
                # ===== 2. 入口变迁：系统晶圆数限制 =====
                if f.is_entry and petri.entered_wafer_count >= petri.max_wafers_in_system:
                    continue
                
                # ===== 3. u_ 容量约束：目标库所已满则不使能 =====
                if f.capacity_place_idx >= 0:
                    p_idx = f.capacity_place_idx
                    if petri.m[p_idx] >= petri.marks[p_idx].capacity:
                        continue
                
                # ===== 4. t_ 路由过滤：检查运输库所队头 token =====
                if f.check_source_idx >= 0 and f.allowed_route_steps is not None:
                    d_place = petri.marks[f.check_source_idx]
                    if len(d_place.tokens) > 0:
                        head_tok = d_place.head()
                        if head_tok.route_type != 0:  # 跳过非晶圆 token
                            if (head_tok.route_type, head_tok.step) not in f.allowed_route_steps:
                                continue
                
                # ===== 5. 最早使能时间过滤 =====
                pre_places = np.flatnonzero(petri.pre[:, f.t_idx] > 0)
                earliest = 0
                for p in pre_places:
                    tok_enter = int(petri.marks[p].head().enter_time)
                    tok_enter += int(petri.ptime[p])
                    earliest = max(earliest, tok_enter)
                if petri.time < earliest:
                    continue
                
                result_list.append(f.t_idx)
        
        return tm2_enabled, tm3_enabled


class Petri:
    def __init__(self, config: Optional[PetriEnvConfig] = None,
                 enable_statistics: bool = False) -> None:
        """
        初始化 Petri 网环境。

        架构要点：
        - 双路线：LP1 -> s1 -> s2 -> s3 -> s4 -> s5 -> LP_done
                  LP2 -> s1 -> s5 -> LP_done
        - 双机械手：TM2 负责 LP1/LP2/s1/s2/s4/s5/LP_done，TM3 负责 s2/s3/s4
        - color 路线分流：color=1 走路线1，color=2 走路线2

        """
        # -----------------------
        # 1) 加载或创建配置
        # -----------------------
        if config is None:
            # 使用默认配置或传入的参数
            config = PetriEnvConfig()
        
        # 保存配置
        self.config = config
        print(config.format(detailed=True))
        
        # 将配置参数设置为实例属性（保持向后兼容）
        self.n_wafer_route1 = getattr(config, 'n_wafer_route1', None)
        self.n_wafer_route2 = getattr(config, 'n_wafer_route2', None)
        self.n_wafer = self.n_wafer_route1 + self.n_wafer_route2
        self.c_time = config.c_time
        self.R_done = config.R_done
        self.R_finish = getattr(config, 'R_finish', 800)  # 全部完工大奖励
        self.R_scrap = config.R_scrap
        self.T_warn = config.T_warn
        self.a_warn = config.a_warn
        self.T_safe = config.T_safe
        self.b_safe = config.b_safe
        self.D_Residual_time = config.D_Residual_time
        self.P_Residual_time = config.P_Residual_time
        self.c_release_violation = config.c_release_violation
        self.enable_release_penalty_detection = config.enable_release_penalty_detection
        self.T_transport = config.T_transport
        self.T_load = config.T_load
        self.idle_timeout = config.idle_timeout
        self.idle_penalty = config.idle_penalty
        self.stop_on_scrap = config.stop_on_scrap
        self.training_phase = config.training_phase
        self.reward_config = config.reward_config
        self.MAX_TIME = config.MAX_TIME
        self.Wait_time = config.Wait_time
        
        # ============ 性能优化配置 ============
        # 内部状态
        self.scrap_count = 0  # 报废计数器
        self.resident_violation_count = 0 # 驻留时间违规计数 (Type 1)
        self.qtime_violation_count = 0    # Q-time 违规计数 (Type 2)
        self.violated_tokens: Dict[int, Set[str]] = {} # 记录已违规的 token (token_id -> {type, ...})
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
        self.time = 0
        self.fire_log = []

        # 错峰启动：记录 t_PM1 上次发射时间
        self._last_t_pm1_fire_time = -INF
        
        # 释放时间追踪：构建加工腔室链路映射
        # release_chain[place_idx] = (downstream_place_idx, transport_time)
        # 例如：PM1 -> PM2，运输时间为 T_pm1_to_pm2
        self._build_release_chain()
        
        # 累计的释放时间违规惩罚（每个 step 计算后清零）
        self._release_violation_penalty = 0.0
        
        # 晶圆滞留时间统计追踪
        # {token_id: {enter_system, exit_system, chambers: {name: {enter, exit}}, transports: {name: {enter, exit}}}}
        self.wafer_stats: Dict[int, Dict[str, Any]] = {}
        
        # s5 首次预估释放时间追踪：{wafer_id: first_estimated_release_time}
        self._s5_first_estimate: Dict[int, int] = {}
        
        # 机械手时间轴：仅存储未来预估的操作，用于准确预估晶圆到达各腔室的时间
        self._tm2_timeline: List[Interval] = []  # TM2 机械手预估占用时间轴
        self._tm3_timeline: List[Interval] = []  # TM3 机械手预估占用时间轴
        
        # 可视化统计开关（训练模式下为 False，避免性能开销)
        self.enable_statistics = enable_statistics

        # 构建变迁使能管理器（预计算每个变迁的过滤器）
        self._tm = TransitionManager(self)

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


    def _get_place_index(self, name: str) -> int:
        """获取库所索引"""
        return self.id2p_name.index(name)
    
    def _get_transition_index(self, name: str) -> int:
        """获取变迁索引"""
        return self.id2t_name.index(name)

    # ==================== Release subsystem ====================
    #
    # Release tracks estimated wafer arrival times at chambers,
    # detects capacity conflicts early, and applies penalties to
    # prevent "wafer arrives but chamber is full" deadlocks.
    #
    # Design principles:
    #   1. Check-on-leave: violation detection only on first add
    #      (when wafer leaves current chamber), not on update.
    #   2. Estimate-then-refine: record with transport estimate,
    #      update with actual time when wafer enters chamber.
    #   3. Chain propagation: one operation propagates along the
    #      release chain to all downstream chambers.
    #   4. Dual-route separation: release_chain (route 1) and
    #      release_chain_route2 (route 2) are independent.
    #
    # Function index (lifecycle order):
    #   _build_release_chain     -> init: build chain mapping
    #   _record_initial_release  -> first record + violation check
    #   _chain_record_release    -> chain first-record downstream
    #   _check_release_violation -> detect capacity conflict
    #   _update_release          -> refine with actual enter time
    #   _chain_update_release    -> chain refine downstream (no check)
    #   _pop_release             -> remove on chamber exit
    #
    # Transition -> Release operation mapping:
    #   u_LP1_s1 / u_LP2_s1 -> _record_initial_release(s1);
    #                           route2 also reserves s5
    #   t_s1                -> _update_release(s1, chain=False)
    #   u_s1_s2             -> _pop_release(s1)
    #   u_s1_s5             -> _pop_release(s1) + update_release(s5)
    #   u_s2_s3             -> add_release(s3, s4, s5) + violation
    #   t_s3                -> _update_release(s3, chain=True)
    #                           -> chain updates s4, s5
    #   u_s3_s4             -> _pop_release(s3)
    #   t_s4                -> _update_release(s4, chain=False)
    #   u_s4_s5             -> update_release(s5) + _pop_release(s4)
    #   t_s5                -> _update_release(s5)
    #   u_s5_LP_done        -> _pop_release(s5)
    # ============================================================

    def _build_release_chain(self) -> None:
        """
        构建释放时间链路映射（初始化时调用一次）。
        
        release_chain[place_idx] = (downstream_place_idx, transport_time)
        用于链式更新下游腔室的预估释放时间。
        
        构建两条链路：
          路线1 (color=1): s1 -> s3 -> s4 -> s5
          路线2 (color=2): s1 -> s5
        运行时根据 wafer_color 选择对应链路进行传播。
        
        有驻留约束腔室：s1, s3, s5
        无驻留约束腔室：s2, s4（作为机械手交接点）
        """
        self.release_chain: Dict[int, Tuple[int, int]] = {}
        
        # 获取腔室索引（如果存在）
        def get_idx(name: str) -> int:
            if name in self.id2p_name:
                return self.id2p_name.index(name)
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
        
        被 _record_initial_release 和 _chain_record_release 调用；
        _update_release / _chain_update_release 不调用此方法（更新已有记录不应约束自己）。
        
        判定流程：
        1. 若 release_schedule 记录数 < 容量 -> 不违规
        2. 否则取第 (len - capacity) 早的释放时间作为约束
        3. 若预计进入时间早于该约束 -> 违规，返回惩罚并修正进入时间
        
        Args:
            place_idx: 目标腔室索引
            expected_enter_time: 预计进入时间
            
        Returns:
            (penalty: float, corrected_enter_time: int)
            - penalty: 惩罚值（0 表示不违规）
            - corrected_enter_time: 修正后的进入时间
        """
        place = self.marks[place_idx]
        actual_occupancy = len(place.release_schedule)
        
        # 如果占用未满，不违规
        if actual_occupancy < place.capacity:
            return (0.0, expected_enter_time)

        release = sorted(place.release_schedule, key=lambda x: x[1])  # 按释放时间排序
        _,earliest = release[-place.capacity]
        if earliest is None or expected_enter_time >= earliest:
            return (0.0, expected_enter_time)  # 不违规
        
        # 违规：计算惩罚，并修正进入时间为 earliest_release
        penalty = self.c_release_violation * 100
        corrected_enter_time = earliest
        return (penalty, corrected_enter_time)

    def _record_initial_release(self, token_id: int, enter_d_time: int,
                                 target_place_idx: int, wafer_color: int = 0,
                                 chain_downstream: bool = False) -> float:
        """
        晶圆进入运输位时，记录初步预估的释放时间；可选链式传播到下游腔室。
        
        触发时机：u_LP1_s1 / u_LP2_s1（晶圆离开 LP 进入 d_s1）。

        执行流程：
        1. 预估进入时间 = enter_d_time + T_transport + T_load
        2. _check_release_violation 检查违规并修正
        3. 释放时间 = 修正后进入时间 + processing_time
        4. add_release 记录到目标腔室
        5. 若 chain_downstream=True，调用 _chain_record_release 传播到下游
        
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
        if self.enable_release_penalty_detection and self.reward_config.get('release_violation_penalty', 1):
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
        链式首次记录下游腔室的预估释放时间。
        
        由 _record_initial_release（chain_downstream=True）触发。
        沿释放链逐个下游腔室：计算进入时间 -> 违规检查 -> add_release -> 继续传播。
        
        与 _chain_update_release 的区别：
        - 本方法是首次 add，会检查违规并施加惩罚
        - _chain_update_release 是更新已有记录，不检查违规
        
        根据 wafer_color 选择链路：
          color=1 -> release_chain (s1->s3->s4->s5)
          color=2 -> release_chain_route2 (s1->s5)
        
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
            if self.enable_release_penalty_detection and self.reward_config.get('release_violation_penalty', 1):
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
        晶圆实际进入腔室时，用精确时间替换先前的预估值，可选链式更新下游。
        
        触发时机：t_s1 / t_s3 / t_s4 / t_s5（晶圆从运输位进入腔室）。
        
        执行流程：
        1. 新释放时间 = actual_enter_time + processing_time
        2. update_release 更新该腔室的记录
        3. 若 chain_downstream=True，调用 _chain_update_release 更新下游
        
        重要：不调用 _check_release_violation（已有记录不应约束自己）。
        
        chain_downstream 使用场景：
          t_s1 -> False（下游在离开 s2 时才写入，还不存在）
          t_s3 -> True（链式更新 s4、s5）
          t_s4 -> False（s5 在 u_s4_s5 时单独更新）
          t_s5 -> True（默认）
        
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
        链式精确更新下游腔室中指定晶圆的预估释放时间。
        
        由 _update_release（chain_downstream=True）触发。
        典型场景：t_s3 时链式更新 s4、s5 的预估时间。
        
        与 _chain_record_release 的区别：
        - 本方法更新已有记录（update_release），不检查违规、不施加惩罚
        - _chain_record_release 是首次添加记录（add_release），会检查违规
        
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
        
        触发时机：u_s1_s2 / u_s1_s5 / u_s3_s4 / u_s4_s5 / u_s5_LP_done
        （所有 u_ 离开变迁）。
        
        移除后该腔室的 release_schedule 空出位置，
        后续晶圆的 _check_release_violation 将看到容量减少。
        
        Args:
            token_id: 晶圆编号
            place_idx: 腔室索引
        """
        place = self.marks[place_idx]
        place.pop_release(token_id)

    def _get_transition_robot(self, t_name: str) -> Optional[str]:
        """判断变迁使用哪个机械手，返回 'TM2'/'TM3' 或 None。"""
        if t_name in TM2_TRANSITIONS:
            return "TM2"
        elif t_name in TM3_TRANSITIONS:
            return "TM3"
        return None


    def _get_robot_timeline(self, robot_name: str) -> List[Interval]:
        """
        获取指定机械手的时间轴。
        
        Args:
            robot_name: 机械手名称（"TM2" 或 "TM3"）
            
        Returns:
            对应的时间轴列表
        """
        if robot_name == "TM2":
            return self._tm2_timeline
        elif robot_name == "TM3":
            return self._tm3_timeline
        else:
            raise ValueError(f"Unknown robot: {robot_name}")
    
    def _add_robot_estimate(self, robot_name: str, wafer_id: int, start: int, end: int, from_loc: str = "", to_loc: str = "") -> None:
        """
        向机械手时间轴添加预估的未来操作。
        
        Args:
            robot_name: 机械手名称（"TM2" 或 "TM3"）
            wafer_id: 晶圆编号
            start: 预估开始时间
            end: 预估结束时间
            from_loc: 起始位置（用于区分同一晶圆的不同段）
            to_loc: 目标位置
        """
        timeline = self._get_robot_timeline(robot_name)
        interval = Interval(start=int(start), end=int(end), tok_key=wafer_id, from_loc=from_loc, to_loc=to_loc)
        _insert_interval_sorted(timeline, interval)

    def _find_chained_robot_slot(self, robot_name: str, 
                                 start_search: int, 
                                 duration1: int, 
                                 gap: int, 
                                 duration2: int,
                                 max_search_step: int = 20) -> Tuple[int, int]:
        """
        查找满足两个时间段占用的机械手空闲时间。
        Constraint:
          Slot1 starts at t1 >= start_search
          Slot1 duration = duration1
          Slot2 starts at t2 = t1 + duration1 + gap
          Slot2 duration = duration2
          Both Slot1 and Slot2 must be free in robot_timeline.
        
        Returns:
            (t1, t2) start times for both slots.
        """
        timeline = self._get_robot_timeline(robot_name)
        t_start = start_search
        
        # 简单迭代查找
        for _ in range(max_search_step):
            # 1. 找第一个 slot
            t1 = _first_free_time_at(timeline, t_start, t_start + duration1)
            
            # 2. 计算对应的第二个 slot 时间
            t2 = t1 + duration1 + gap
            
            # 3. 检查第二个 slot 是否可用
            # _first_free_time_at 返回的是满足 [t, t+dur) 空闲的最早 t >= expected
            # 我们需要严格检查 t2 是否可用，即 check if [t2, t2+dur2) overlaps with any interval
            # 这里调用 _first_free_time_at(..., t2, ...) 如果返回 t2，说明 t2 可用
            t2_actual = _first_free_time_at(timeline, t2, t2 + duration2)
            
            if t2_actual == t2:
                # Slot2 也在预期位置可用 -> 找到解
                return t1, t2
            else:
                # Slot2 不可用 (t2_actual > t2)
                # 这意味着我们需要推迟 t1。
                # 简单策略：t_start = t1 + 1 (或者更激进：从 t2_actual 反推?)
                # 如果以 t2_actual 作为 Slot2 的开始，那么 new_t1 = t2_actual - gap - duration1
                # 但需要保证 new_t1 >= t1 + 1 (向前推进)
                new_t1_candidate = t2_actual - gap - duration1
                if new_t1_candidate > t1:
                     t_start = new_t1_candidate
                else:
                     t_start = t1 + 1
        return t_start, t_start + duration1 + gap
    
    def _remove_robot_estimate(self, robot_name: str, wafer_id: int, from_loc: str = "", to_loc: str = "") -> None:
        """
        从机械手时间轴移除指定晶圆的预估操作（当实际执行时）。
        由 (wafer_id, from_loc, to_loc) 唯一确定一个预估。
        
        Args:
            robot_name: 机械手名称（"TM2" 或 "TM3"）
            wafer_id: 晶圆编号
            from_loc: 起始位置
            to_loc: 目标位置
        """
        timeline = self._get_robot_timeline(robot_name)
        # 移除匹配的区间
        # 如果 from_loc/to_loc 为空，则移除所有该 wafer_id 的区间（兼容旧行为，但不建议）
        if not from_loc and not to_loc:
             timeline[:] = [itv for itv in timeline if itv.tok_key != wafer_id]
        else:
             timeline[:] = [itv for itv in timeline if not (itv.tok_key == wafer_id and itv.from_loc == from_loc and itv.to_loc == to_loc)]

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

    def reset(self):
        self.time = 0
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
        self._s5_first_estimate = {}  # 重置 s5 首次预估
        self._tm2_timeline = []  # 重置 TM2 时间轴
        self._tm3_timeline = []  # 重置 TM3 时间轴
        self.entered_wafer_count = 0  # 重置已进入系统的晶圆数
        self.done_count = 0           # 重置已完成的晶圆数
        self.resident_violation_count = 0 # 重置驻留违规计数
        self.qtime_violation_count = 0    # 重置Q-time违规计数
        self.violated_tokens = {}         # 重置违规记录
        # 清空所有库所的释放时间队列，并重置机器分配计数器
        for place in self.marks:
            place.release_schedule.clear()
            if place.type == 1:
                place.last_machine = -1  # 重置机器轮换计数器

        # 更新 _process_places_cache
        self._process_places_cache = [p for p in self.marks if p.type == 1]

    def _update_stay_times(self) -> None:
        """更新所有 token 的滞留时间"""
        current_time = self.time
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

        # 从配置读取惩罚/奖励系数
        transport_overtime_coef = self.config.transport_overtime_coef
        chamber_overtime_coef = self.config.chamber_overtime_coef
        processing_reward_coef = self.config.processing_reward_coef
        
        delta_t = t2 - t1
        proc_reward = 0.0
        overtime_penalty = 0.0
        warn_penalty = 0.0
        transport_penalty = 0.0
        safe_reward = 0.0

        for p_idx, place in enumerate(self.marks):
            for tok_idx, tok in enumerate(place.tokens):
                # LP 库所不计入惩罚（源头）
                if place.name == "LP" or place.type not in (1, 2, 5):
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
                    proc_end = tok.enter_time + place.processing_time
                    
                    # 1) 加工奖励：在加工时间内每秒 +r
                    if self.reward_config.get('proc_reward', 1):
                        proc_overlap = min(t2, proc_end) - max(t1, tok.enter_time)
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
        
        time_cost = self.c_time * delta_t if self.reward_config.get('time_cost', 1) else 0.0
        
        # 9. 停滞惩罚（连续 WAIT 超过阈值）
        idle_penalty_val = 0.0
        if (self._consecutive_wait_time >= self.idle_timeout) and not self._idle_penalty_applied:
            self._idle_penalty_applied = True  # 只惩罚一次
            idle_penalty_val = self.idle_penalty
        
        total = proc_reward + safe_reward - (overtime_penalty + warn_penalty + transport_penalty) - time_cost - idle_penalty_val
        
        if detailed:
            return {
                "total": total,
                "proc_reward": proc_reward,
                "safe_reward": safe_reward,
                "penalty": overtime_penalty,
                "warn_penalty": warn_penalty,
                "transport_penalty": transport_penalty,
                "time_cost": time_cost,
                "idle_timeout_penalty": -idle_penalty_val if idle_penalty_val > 0 else 0.0
            }
        
        return total

    def _fire(self, t: Union[int, List[int]]):
        start_time = self.time
        enter_new = self.time + self.ttime
        
        # 统一处理单个变迁和变迁列表
        transitions = [t] if isinstance(t, (int, np.integer)) else t
        
        for t_idx in transitions:
            t_name = self.id2t_name[t_idx]

            pre_places = np.flatnonzero(self.pre[:, t_idx] > 0)
            pst_places = np.nonzero(self.pst[:, t_idx] > 0)[0]

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
                # 晶圆离开 LP 进入 d_s1：对 s1 记录并检查
                if "s1" in self.id2p_name:
                    s1_idx = self._get_place_index("s1")
                    penalty = self._record_initial_release(wafer_id, enter_new, s1_idx, wafer_route_type, chain_downstream=False)
                    self._release_violation_penalty += penalty
                
                # 类型 2 晶圆（LP2 路线）：同时预约 s5（拉式预约）
                if wafer_route_type == 2 and "s5" in self.id2p_name:
                    s1_idx = self._get_place_index("s1")
                    s5_idx = self._get_place_index("s5")
                    place_s1 = self.marks[s1_idx]
                    place_s5 = self.marks[s5_idx]
                    
                    # 预估 s1 释放时间（基于 s1 预约）
                    expected_enter_s1 = enter_new + self.T_transport + self.T_load
                    release_s1 = expected_enter_s1 + place_s1.processing_time
                    
                    # 预估 s5 进入时间：考虑 TM2 时间轴（s1→s5 运输使用 TM2）
                    # 缺陷修复：需同时预约 s5->LP_done 的出料操作
                    transport_s1_to_s5 = self.T_load * 2 + self.T_transport
                    transport_s5_out = self.T_load * 2 + self.T_transport # s5 -> LP_done
                    
                    # Gap = s5 processing time
                    s5_proc_time = place_s5.processing_time
                    
                    # 查询 TM2 链式时间轴
                    # Slot1: s1 -> s5 (at t1, duration=transport_s1_to_s5)
                    # Gap: s5_proc_time
                    # Slot2: s5 -> LP_done (at t2, duration=transport_s5_out)
                    t1, t2 = self._find_chained_robot_slot(
                        "TM2",
                        start_search=release_s1,
                        duration1=transport_s1_to_s5,
                        gap=s5_proc_time,
                        duration2=transport_s5_out
                    )
                    
                    earliest_tm2_free = t1
                    
                    # 实际进入 s5 的时间
                    expected_enter_s5 = earliest_tm2_free + transport_s1_to_s5
                                        


                    penalty_s5, corrected_enter_s5 = self._check_release_violation(s5_idx, expected_enter_s5)

                    if self.reward_config.get('release_violation_penalty', 1):
                        self._release_violation_penalty += penalty_s5
                    
                    release_s5 = corrected_enter_s5 + place_s5.processing_time
                    place_s5.add_release(wafer_id, release_s5)
                    
                    # 将 TM2 预估占用写入时间轴 (Inbound)
                    self._add_robot_estimate("TM2", wafer_id, earliest_tm2_free, earliest_tm2_free + transport_s1_to_s5,
                                             from_loc="s1", to_loc="s5")
                    
                    # 将 TM2 预估占用写入时间轴 (Outbound)
                    # t2 是 Slot2 的开始时间
                    self._add_robot_estimate("TM2", wafer_id, t2, t2 + transport_s5_out,
                                             from_loc="s5", to_loc="LP_done")
                    
                    # 记录 s5 首次预估释放时间（路线2）
                    if wafer_id not in self._s5_first_estimate:
                        self._s5_first_estimate[wafer_id] = release_s5
                
            # 记录实际机械手占用（用于更新时间轴，使预估更准确）
            tr_robot = self._get_transition_robot(t_name)
            if tr_robot and wafer_id != -1:
                # 记录实际占用区间 [start_time, enter_new)
                # 不带 from_loc/to_loc，避免误删除
                self._add_robot_estimate(tr_robot, wafer_id, start_time, enter_new)

            if t_name == "t_s1":
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
                # 路线2：晶圆离开 s1 直接去 s5，从 s1 队列移除；检查实际进入时间是否与其他晶圆冲突
                # 清理 TM2 预估（实际执行时）
                self._remove_robot_estimate("TM2", wafer_id, from_loc="s1", to_loc="s5")
                
                if "s1" in self.id2p_name:
                    s1_idx = self._get_place_index("s1")
                    self._pop_release(wafer_id, s1_idx)
                if "s5" in self.id2p_name:
                    s5_idx = self._get_place_index("s5")
                    place_s5 = self.marks[s5_idx]
                    # 晶圆进入 d_s5 时间 = enter_new，预计进入 s5 = enter_new + T_load
                    expected_enter_s5 = enter_new + self.T_load
                    
                    # 检查实际进入是否与其他晶圆冲突（排除自己的预约）
                    other_releases = [(tid, rt) for tid, rt in place_s5.release_schedule if tid != wafer_id]
                    other_releases_sorted = sorted(other_releases, key=lambda x: x[1])
                    
                    if len(other_releases_sorted) >= place_s5.capacity:
                        # 其他晶圆占满了 s5，检查是否可以进入
                        constraint_time = other_releases_sorted[-place_s5.capacity][1]
                        if expected_enter_s5 < constraint_time:
                            # 实际进入与其他晶圆冲突，施加惩罚
                            penalty = self.c_release_violation * 100
                            self._release_violation_penalty += penalty
                    
                    # 更新 s5 的释放时间（用实际进入时间）
                    release_s5 = expected_enter_s5 + place_s5.processing_time
                    place_s5.update_release(wafer_id, release_s5)
                
            elif t_name == "u_s2_s3":
                # 路线1：离开 s2(LLC)，对 s3、s4、s5 做 add_release + 违规检查（离开才检查）
                if "s3" in self.id2p_name and "s4" in self.id2p_name and "s5" in self.id2p_name:
                    s3_idx = self._get_place_index("s3")
                    s4_idx = self._get_place_index("s4")
                    s5_idx = self._get_place_index("s5")
                    place_s3 = self.marks[s3_idx]
                    place_s4 = self.marks[s4_idx]
                    place_s5 = self.marks[s5_idx]
                    
                    # ========== s2 → s3 预估（使用 TM3） ==========
                    expected_enter_s3 = enter_new + self.T_load + self.T_transport
                    
                    # 检查 s3 违规并获取修正后的进入时间
                    penalty = 0.0
                    corrected_enter_s3 = expected_enter_s3
                    if self.reward_config.get('release_violation_penalty', 1):
                        penalty_s3, corrected_enter_s3 = self._check_release_violation(s3_idx, expected_enter_s3)
                        penalty += penalty_s3
                    
                    # 使用修正后的 s3 进入时间重新计算 s3 释放时间
                    release_s3 = corrected_enter_s3 + place_s3.processing_time
                    
                    # ========== s3 → s4 预估（使用 TM3） ==========
                    transport_s3_to_s4 = self.ttime * 3  # 合并的运输时间块
                    
                    # 查询 TM3 时间轴，获取最早可用时间
                    tm3_timeline = self._get_robot_timeline("TM3")
                    earliest_tm3_free_s3_s4 = _first_free_time_at(
                        tm3_timeline,
                        release_s3,  # 期望开始时间
                        release_s3 + transport_s3_to_s4  # 期望结束时间
                    )
                    
                    # 实际进入 s4 的时间 = TM3 可用时间 + 运输时间
                    expected_enter_s4 = earliest_tm3_free_s3_s4 + transport_s3_to_s4
                    
                    # 检查 s4 违规并获取修正后的进入时间
                    corrected_enter_s4 = expected_enter_s4
                    if self.reward_config.get('release_violation_penalty', 1):
                        penalty_s4, corrected_enter_s4 = self._check_release_violation(s4_idx, expected_enter_s4)
                        penalty += penalty_s4
                    
                    # 使用修正后的 s4 进入时间计算初步 s4 释放时间
                    release_s4 = corrected_enter_s4 + place_s4.processing_time
                    
                    # ========== s4 → s5 预估（使用 TM2） ==========
                    # 缺陷修复：需同时预约 s5->LP_done 的出料操作
                    transport_s4_to_s5 = self.ttime * 3  # 合并的运输时间块
                    transport_s5_out = self.ttime * 3 # s5 -> LP_done
                    
                    # Gap = s5 processing time
                    s5_proc_time = place_s5.processing_time
                    
                    # 查询 TM2 链式时间轴
                    # Slot1: s4 -> s5 (at t1, duration=transport_s4_to_s5)
                    # Gap: s5_proc_time
                    # Slot2: s5 -> LP_done (at t2, duration=transport_s5_out)
                    t1, t2 = self._find_chained_robot_slot(
                        "TM2",
                        start_search=release_s4,
                        duration1=transport_s4_to_s5,
                        gap=s5_proc_time,
                        duration2=transport_s5_out
                    )
                    
                    earliest_tm2_free_s4_s5 = t1
                    
                    # ========== 拉式预约：根据 s5 可用时间反推 s4 释放时间 ==========
                    s5_schedule = sorted(place_s5.release_schedule, key=lambda x: x[1])
                    if len(s5_schedule) >= place_s5.capacity:
                        # s5 满了，找第 (len - capacity) 个释放时间（即下一个可用 slot）
                        s5_available = s5_schedule[-place_s5.capacity][1]
                        expected_enter_s5 = earliest_tm2_free_s4_s5 + transport_s4_to_s5
                        
                        if expected_enter_s5 < s5_available:
                            # 反推 s4 应该何时释放（晶圆在 s4 多停留）
                            s4_should_release = s5_available - transport_s4_to_s5
                            if s4_should_release > release_s4:
                                release_s4 = s4_should_release
                                # 重新计算 TM2 可用时间 (链式)
                                t1, t2 = self._find_chained_robot_slot(
                                    "TM2",
                                    start_search=release_s4,
                                    duration1=transport_s4_to_s5,
                                    gap=s5_proc_time,
                                    duration2=transport_s5_out
                                )
                                earliest_tm2_free_s4_s5 = t1

                            # 修复：不能强制等于 s5_available，因为 robot 可能因为繁忙而导致实际到达时间晚于 s5_available
                            # 正确做法是基于重新计算的 t1 (earliest_tm2_free_s4_s5) 来计算预计到达时间
                            expected_enter_s5 = earliest_tm2_free_s4_s5 + transport_s4_to_s5
                    else:
                        # s5 有空位，正常计算
                        expected_enter_s5 = earliest_tm2_free_s4_s5 + transport_s4_to_s5
                    
                    # 计算 s5 释放时间
                    release_s5 = expected_enter_s5 + place_s5.processing_time
                    
                    place_s3.add_release(wafer_id, release_s3)
                    place_s4.add_release(wafer_id, release_s4)
                    place_s5.add_release(wafer_id, release_s5)

                    if wafer_id not in self._s5_first_estimate:
                        self._s5_first_estimate[wafer_id] = release_s5
                    
                    # 写入各阶段预估占用
                    self._add_robot_estimate("TM3", wafer_id, earliest_tm3_free_s3_s4, earliest_tm3_free_s3_s4 + transport_s3_to_s4,
                                             from_loc="s3", to_loc="s4")
                    self._add_robot_estimate("TM2", wafer_id, earliest_tm2_free_s4_s5, earliest_tm2_free_s4_s5 + transport_s4_to_s5,
                                             from_loc="s4", to_loc="s5")
                    
                    # 写入 s5->LP_done 预估 (Outbound)
                    # t2 是 Slot2 的开始时间
                    self._add_robot_estimate("TM2", wafer_id, t2, t2 + transport_s5_out,
                                             from_loc="s5", to_loc="LP_done")
                    


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
                # 清理 TM3 预估（实际执行时，对应 s3->s4）
                self._remove_robot_estimate("TM3", wafer_id, from_loc="s3", to_loc="s4")
                
                if "s3" in self.id2p_name:
                    s3_idx = self._get_place_index("s3")
                    self._pop_release(wafer_id, s3_idx)
            
            elif t_name == "u_s4_s5":
                # 路线1：离开 s4(LLD)，更新 s5 的释放时间（已在 u_s2_s3 时添加）；再从 s4 队列移除
                # 清理 TM2 预估（实际执行时，对应 s4->s5）
                self._remove_robot_estimate("TM2", wafer_id, from_loc="s4", to_loc="s5")
                
                if "s4" in self.id2p_name and "s5" in self.id2p_name:
                    s4_idx = self._get_place_index("s4")
                    s5_idx = self._get_place_index("s5")
                    place_s4 = self.marks[s4_idx]
                    place_s5 = self.marks[s5_idx]
                    release_s4 = place_s4.get_release(wafer_id)
                    if release_s4 is not None:
                        # 运输时间 = s4卸载 + d_s5运输 + s5装载
                        transport_s4_to_s5 = self.ttime * 3  # 约 15s
                        expected_enter_s5 = release_s4 + transport_s4_to_s5
                        
                        # 更新 s5 的释放时间（不再检查违规，因为已在 u_s2_s3 时检查）
                        release_s5 = expected_enter_s5 + place_s5.processing_time
                        place_s5.update_release(wafer_id, release_s5)
                    else:
                        # 防御性处理：release_s4 为 None 时使用当前时间估算
                        import warnings
                        warnings.warn(
                            f"[释放时间异常] wafer_id={wafer_id} 在 u_s4_s5 时 release_s4 为 None，"
                            f"s4.release_schedule={list(place_s4.release_schedule)}，使用当前时间估算",
                            RuntimeWarning
                        )
                        # 使用当前时间 + 运输时间作为 fallback
                        transport_s4_to_s5 = self.ttime * 3
                        expected_enter_s5 = enter_new + transport_s4_to_s5
                        release_s5 = expected_enter_s5 + place_s5.processing_time
                        place_s5.update_release(wafer_id, release_s5)
                    self._pop_release(wafer_id, s4_idx)
                
            elif t_name == "t_s5":
                # 晶圆进入 s5：更新精确释放时间
                if "s5" in self.id2p_name:
                    s5_idx = self._get_place_index("s5")
                    self._update_release(wafer_id, enter_new, s5_idx, wafer_route_type)
                
            elif t_name == "u_s5_LP_done":
                # 晶圆离开 s5：从 s5 队列移除
                # 清理 TM2 预估（实际执行时，对应 s5->LP_done）
                self._remove_robot_estimate("TM2", wafer_id, from_loc="s5", to_loc="LP_done")
                
                if "s5" in self.id2p_name:
                    s5_idx = self._get_place_index("s5")
                    self._pop_release(wafer_id, s5_idx)
                

            
            elif t_name == "t_LP_done":
                # 记录 s5 实际离开时间到 wafer_stats（已在 _track_wafer_statistics 中处理）
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
            
            # 机械手动作：使用模块级常量判断
            if t_name.startswith("u_") or t_name.startswith("t_"):
                robot_name = self._get_transition_robot(t_name) or "TM2"  # 默认 TM2
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

        # 3) 时间推进到完成之后 (只更新一次)
        self.time = enter_new
        self._update_stay_times()

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

    def get_enable_t_by_robot(self) -> Tuple[List[int], List[int]]:
        """返回两个机械手各自可用的变迁列表。"""
        return self._tm.get_enable_t()


    def _check_scrap(self, return_info: bool = False) -> bool | Tuple[bool, Optional[Dict]]:
        """
        检查是否有报废的 wafer。
        - 驻留时间违规 (Type 1): 停留时间 > processing_time + P_Residual_time
        - Q-time 违规 (Type 2): 停留时间 > 15s (用户定义)
        
        Args:
            return_info: 是否返回报废详情
            
        Returns:
            如果 return_info=False: 返回 bool
            如果 return_info=True: 返回 (bool, scrap_info)
                scrap_info = {"place": 库所名, "enter_time": 进入时间, "overtime": 超时秒数, "type": 违规类型}
        """
        for place in self.marks:
            # Type 1: 加工腔室 (驻留时间约束)
            if place.type == 1:
                # 注意：s1/s3/s5 是 type=1，s2/s4 是 type=5 (无约束)
                for tok in place.tokens:
                    overtime = (self.time - tok.enter_time) - place.processing_time - self.P_Residual_time
                    if overtime > 0:
                        self.resident_violation_count += 0.5
                        
                        if return_info:
                            return True, {
                                "token_id": getattr(tok, "token_id", -1),
                                "place": place.name,
                                "enter_time": tok.enter_time,
                                "stay_time": self.time - tok.enter_time,
                                "proc_time": place.processing_time,
                                "overtime": overtime,
                                "type": "resident"
                            }
                        return True
            
            # Type 2: 运输库所/机械手 (Q-time 约束 > 15s) - 已禁用
            elif place.type == 2:
                 for tok in place.tokens:
                     # Q-time limit = 15s
                     overtime = (self.time - tok.enter_time) - 15
                     if overtime > 0:
                         if return_info:
                             return False, {
                                 "token_id": getattr(tok, "token_id", -1),
                                 "place": place.name,
                                 "enter_time": tok.enter_time,
                                 "stay_time": self.time - tok.enter_time,
                                 "proc_time": 0,
                                 "overtime": overtime,
                                 "type": "qtime"
                             }
                         return True

                        
        if return_info:
            return False, None
        return False

    def step(self, t: Optional[Union[int, List[int]]] = None,
             wait: bool = False,
             detailed_reward: bool = False):
        """
        执行一步动作。
        
        Args:
            t: 要执行的变迁索引或变迁索引列表（当 wait=False 时）
            wait: 是否执行 WAIT 动作
            detailed_reward: 是否返回详细奖励分解（仅当 with_reward=True 时有效）
            
        Returns:
            detailed_reward=False: (done, reward, scrap)
            detailed_reward=True: (done, reward_dict, scrap)
            
            done=True 表示 episode 结束（完成或报废）
            scrap=True 表示因报废而结束
        """
        if self.time >= self.MAX_TIME:
            if detailed_reward:
                return True, {"total": -100, "timeout": True}, True
            return True, -100, True  # 超时视为报废

        if wait:
            t1 = self.time
            t2 = self.time + self.Wait_time
            self._consecutive_wait_time += self.Wait_time # 累计连续 WAIT 时间

            reward_result = self.calc_reward(t1, t2, detailed=detailed_reward)
            self.time = t2
            self._update_stay_times()
            
            # 检查报废
            is_scrap, scrap_info = self._check_scrap(return_info=True)
            if is_scrap:
                sType = scrap_info.get("type", "") if scrap_info else ""
                tid = scrap_info.get("token_id", -1)
                
                # 检查是否已记录过该 token 的该类型违规
                already_counted = False
                if tid != -1:
                    if tid not in self.violated_tokens:
                        self.violated_tokens[tid] = set()
                    if sType in self.violated_tokens[tid]:
                        already_counted = True
                    else:
                        self.violated_tokens[tid].add(sType)
                
                # 1. 驻留时间违规 (Resident Time) - 严重，视为 scrap，可能停止
                if sType == "resident":
                    if not already_counted:
                        self.scrap_count += 1
                        self.resident_violation_count += 1
                    
                    done = self.stop_on_scrap
                    if detailed_reward:
                        reward_result["scrap_penalty"] = -self.R_scrap
                        reward_result["total"] -= self.R_scrap
                        reward_result["scrap_info"] = scrap_info
                        return done, reward_result, True
                    return done, reward_result - self.R_scrap, True

                # 2. Q-time 违规 - 警告，不停止，不计入 scrap_count (以免触发外部停止)
                elif sType == "qtime":
                    if not already_counted:
                        self.qtime_violation_count += 1
                    
                    # 施加惩罚但不停止
                    if detailed_reward:
                        # 使用较小的惩罚或特定 Q-time 惩罚
                        reward_result["qtime_penalty"] = -self.R_scrap  # 或其他系数
                        reward_result["total"] -= self.R_scrap
                        reward_result["scrap_info"] = scrap_info  # 记录详情
                    else:
                        reward_result -= self.R_scrap
            
            return False, reward_result, False

        # 执行变迁动作
        # 重置连续 WAIT 时间和停滞标记（执行非 WAIT 动作）
        self._consecutive_wait_time = 0
        self._idle_penalty_applied = False
        # 重置释放时间违规惩罚累积器
        self._release_violation_penalty = 0.0
        
        t1 = self.time
        t2 = self.time + self.ttime
        
        # 处理单个变迁或变迁列表及其前置库所
        transitions = [t] if isinstance(t, (int, np.integer)) else t
        pre_places_indices = []
        for t_idx in transitions:
            pre_places_indices.extend(np.flatnonzero(self.pre[:, t_idx] > 0))
        pre_places = np.array(pre_places_indices)
        
        reward_result = self.calc_reward(t1, t2, moving_pre_places=pre_places, detailed=detailed_reward)
        self._fire(t=t)
        
        # 将释放时间违规惩罚加入奖励
        if self._release_violation_penalty > 0:
            if detailed_reward:
                reward_result["release_violation_penalty"] = -self._release_violation_penalty
                reward_result["total"] -= self._release_violation_penalty
            else:
                reward_result -= self._release_violation_penalty

        
        # 添加单片完工奖励（在 _fire 中累积）
        if self._per_wafer_reward > 0:
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
            if detailed_reward:
                reward_result["finish_bonus"] = self.R_finish
                reward_result["total"] += self.R_finish
            else:
                reward_result += self.R_finish

        
        # fire 后也检查报废
        is_scrap, scrap_info = self._check_scrap(return_info=True)
        if is_scrap:
            sType = scrap_info.get("type", "") if scrap_info else ""
            tid = scrap_info.get("token_id", -1)
            
            # 检查是否已记录过该 token 的该类型违规
            already_counted = False
            if tid != -1:
                if tid not in self.violated_tokens:
                    self.violated_tokens[tid] = set()
                if sType in self.violated_tokens[tid]:
                    already_counted = True
                else:
                    self.violated_tokens[tid].add(sType)

            # 1. 驻留时间违规 (Resident Time) - 严重，视为 scrap，可能停止
            if sType == "resident":
                if not already_counted:
                    self.scrap_count += 1
                    self.resident_violation_count += 1
                
                done = self.stop_on_scrap
                if detailed_reward:
                    reward_result["scrap_penalty"] = -self.R_scrap
                    reward_result["total"] -= self.R_scrap
                    reward_result["scrap_info"] = scrap_info
                    return done, reward_result, True
                return done, reward_result - self.R_scrap, True

            # 2. Q-time 违规 - 警告，不停止
            elif sType == "qtime":
                if not already_counted:
                    self.qtime_violation_count += 1
                
                # 施加惩罚但不停止
                if detailed_reward:
                    reward_result["qtime_penalty"] = -self.R_scrap
                    reward_result["total"] -= self.R_scrap
                    reward_result["scrap_info"] = scrap_info
                else:
                    reward_result -= self.R_scrap

        return finish, False

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
        # 处理等待参数
        if a1 is None:
            wait1 = True
        if a2 is None:
            wait2 = True
        
        # 如果两个都是等待
        if wait1 and wait2:
            return self.step(wait=True, with_reward=with_reward, detailed_reward=detailed_reward)
        
        # 收集所有非等待的动作
        actions = []
        if not wait1 and a1 is not None:
            actions.append(a1)
        if not wait2 and a2 is not None:
            actions.append(a2)
            
        # 调用 step 执行（单个或多个动作）
        return self.step(t=actions, wait=False, with_reward=with_reward, detailed_reward=detailed_reward)


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
            "resident_violation_count": self.resident_violation_count,
            "qtime_violation_count": self.qtime_violation_count,
        }



def main():
    #np.random.seed(0)  # 可删，保证复现
    model = Petri()
    model.reset()

    wait_id = 999
    print("Transitions:", model.id2t_name)
    print("Places:", model.id2p_name)
    print("-" * 70)

    max_steps = 1000
    for step_i in range(max_steps):
        enabled = model.get_enable_t()
        # print(f"Step {step_i} Enabled: {enabled}") 

        
        # 优先选择非 wait 变迁
        real_transitions = [t for t in enabled if t != wait_id]
        
        if real_transitions:
            t = int(np.random.choice(real_transitions))
            print(f"  -> fire t={t} ({model.id2t_name[t]})at time={model.time}")
            finish = model.step(t=t, wait=False)
        else:
            t = wait_id
            print(f"  -> fire t=wait  at time={model.time}")
            finish = model.step(wait=True)

        if finish:
            print(f"[DONE] Finished at time={model.time}, step={step_i}")
            model.render_gantt()
            break


if __name__ == "__main__":
    main()

