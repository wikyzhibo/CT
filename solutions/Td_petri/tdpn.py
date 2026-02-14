import numpy as np
from typing import List, Optional
import sys
import time
import torch
from torchrl.data import Bounded, Unbounded, Categorical, Composite, Binary
from torchrl.envs import EnvBase
from tensordict import TensorDict
from solutions.model.pn_models import Place
from solutions.Td_petri.construct import SuperPetriBuilderV3
from solutions.Td_petri.resources import ActionInfo
from solutions.Td_petri.core.config import PetriConfig
from solutions.Td_petri.core.action_enable import ActionEnableChecker
from solutions.Td_petri.core.transition_fire import TransitionFireExecutor
from solutions.Td_petri.resources import ResourceManager
from solutions.Td_petri.rl import PathRegistry,ActionSpaceBuilder,ObservationBuilder,RewardCalculator


class TimedPetri(EnvBase):
    """
    Timed Petri Net 环境，直接作为 TorchRL 兼容的 RL 环境使用。
    
    Features:
    - Configurable via PetriConfig
    - Action masking support
    - Time tracking
    - Reward shaping options
    
    Example:
        >>> config = PetriConfig.default()
        >>> env = TimedPetri(config=config, device='cpu')
        >>> td = env.reset()
        >>> action = env.action_spec.rand()
        >>> td_next = env.step(td.set("action", action))
    """
    
    metadata = {'render.modes': ['human', 'rgb_array'], "render_fps": 30}
    batch_locked = False

    def __init__(self, config: Optional['PetriConfig'] = None,
                 device: str = 'cpu', seed: Optional[int] = None,
                 reward_mode: str = 'progress') -> None:
        """
        初始化 Timed Petri Net.
        
        Args:
            config: 配置对象。如果为 None，使用默认配置。
            device: Device for tensors ('cpu' or 'cuda')
            seed: Random seed
            reward_mode: Reward calculation mode:
                - 'progress': Based on wafer progress (default)
                - 'time': Negative time penalty
                - 'combined': Progress + time penalty
        """
        # Initialize EnvBase if TorchRL is available
        super().__init__(device=device)

        
        self.reward_mode = reward_mode
        self._device = device

        # 使用提供的配置或创建默认配置
        if config is None:
            config = PetriConfig.default()
        self.config = config
        
        # 构建 Petri 网结构
        builder = SuperPetriBuilderV3()
        info = builder.build(modules=config.modules, routes=config.routes)

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
        
        # 初始化路径注册表（路径定义的唯一权威来源）
        self.path_registry = PathRegistry()
        self._init_path()  # 使用 PathRegistry 初始化 token 路径
        
        self.md = info['md']
        self.P = self.pre.shape[0]
        self.T = self.pre.shape[1]
        self.n_wafer = info['n_wafer']
        self.lp_done_idx = self.idle_idx['end']

        self.log = []

        # 搜索功能的服务变量
        self.makespan = 0
        self.transitions = []
        self.m_record = []
        self.marks_record = []
        self.time_record = []
        self.mask_record = []
        self.visited = []
        self.expand_mark = 0
        self.back_time = 0
        self.time = 0
        self.last_time = 0
        self.over_time = 0
        self.qtime_violation = 0
        self.shot = "s"
        self.dead_mark = []
        self.bad_mark = []

        self.transition_times = [[] for _ in range(self.T)]

        self.ops = []
        
        # 初始化资源管理器
        self.resource_mgr = ResourceManager()
        self.res_occ = self.resource_mgr.res_occ  # 向后兼容
        self.open_mod_occ = self.resource_mgr.open_mod_occ  # 向后兼容
        
        self._cache_action_info = {}

        # 阶段配置（来自 config）
        self.stage_c = config.stage_capacity
        self.proc = config.processing_time
        self.debug = False

        # 初始化动作使能检查器
        self.action_checker = ActionEnableChecker(
            pre=self.pre,
            id2t_name=self.id2t_name,
            parallel_groups=config.parallel_groups,
            path_getter=self.get_token_path
        )
        self.rr_idx = self.action_checker.rr_idx  # 向后兼容
        
        # 初始化变迁发射执行器
        self.fire_executor = TransitionFireExecutor(
            pre=self.pre,
            pst=self.pst,
            t_duration=self.t_duration,
            id2t_name=self.id2t_name,
            lp_done_idx=self.lp_done_idx,
            n_wafer=self.n_wafer,
            action_checker=self.action_checker,
            resource_mgr=self.resource_mgr,
            time_getter=lambda: self.time,
            time_setter=lambda t: setattr(self, 'time', t),
            ops_list=self.ops
        )
        
        # 使用 ActionSpaceBuilder 构建 RL 动作空间
        action_builder = ActionSpaceBuilder(self.path_registry)
        action_info = action_builder.get_action_space_info()
        self.aid2chain = action_info['aid2chain']
        self.chain2aid = action_info['chain2aid']
        self.aid_is_parallel = action_info['aid_is_parallel']
        self.aid_pstage = action_info['aid_pstage']
        self.aid2tags = action_info['aid2tags']
        self.A = action_info['A']

        # 识别可观测的库所（P_READY_*）
        self.obs_place_idx = []
        for i, name in enumerate(self.id2p_name):
            if name.startswith('P_READY'):
                self.obs_place_idx.append(i)

        # 初始化观测构建器
        self.his_len = config.history_length
        self.obs_builder = ObservationBuilder(self.obs_place_idx, self.his_len)
        self.obs_dim = self.obs_builder.get_observation_dim()
        
        # 初始化奖励计算器
        self.reward_calc = RewardCalculator(
            self.obs_place_idx,
            list(self.idle_idx['start']),
            config.reward_weights_map
        )
        
        # 创建 TorchRL specs（如果 TorchRL 可用）
        self._make_spec()
        # Set seed
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item() & 0xFFFFFFFF  # 限制在 0 到 2^32-1
        self.set_seed(seed)






    # ========== TorchRL EnvBase Interface Methods ==========
    
    def _make_spec(self):
        """Create observation, action, reward, and done specifications."""

        obs_dim = self.obs_dim
        act_dim = self.A
        
        # Observation spec
        self.observation_spec = Composite(
            observation=Bounded(
                low=0, high=100, 
                shape=(obs_dim,), 
                dtype=torch.int64, 
                device=self._device
            ),
            action_mask=Binary(
                n=act_dim, 
                dtype=torch.bool
            ),
            time=Unbounded(
                shape=(1,), 
                dtype=torch.int64, 
                device=self._device
            ),
            shape=()
        )
        
        # Action spec
        self.action_spec = Categorical(
            n=act_dim, 
            shape=(1,), 
            dtype=torch.int64
        )
        
        # Reward spec
        self.reward_spec = Unbounded(
            shape=(1,), 
            dtype=torch.float32
        )
        
        # State spec (empty for now)
        self.state_spec = Composite(shape=())
        
        # Done spec
        self.done_spec = Composite(
            terminated=Unbounded(
                shape=(1,), 
                dtype=torch.bool
            ),
            finish=Unbounded(
                shape=(1,), 
                dtype=torch.bool
            ),
        )
    
    def _reset(self, td_params: Optional['TensorDict'] = None) -> 'TensorDict':
        """
        TorchRL 标准 reset 接口
        
        Args:
            td_params: Optional parameters (unused)
        
        Returns:
            TensorDict with initial observation, action mask, and time
        """
        """重置环境到初始状态"""
        # 重置核心状态
        self.time = 0
        self.m = self.m0.copy()
        self.marks = self._clone_marks(self.marks_copy)
        self._init_path()

        self.transition_times = [[] for _ in range(self.T)]

        # 重置资源管理器
        self.resource_mgr.reset()

        self._cache_action_info = {}

        # 重置观测构建器历史
        self.obs_builder.reset_history()

        # 重置动作使能检查器
        self.action_checker.reset()

        # 构建初始动作掩码
        mask = np.zeros(self.A, dtype=bool)
        tran_queue = self.get_enable_t(self.m, self.marks)
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

        obs = self.obs_builder.build_observation(self.marks)

        return TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask, dtype=torch.bool),
            "time": torch.tensor([0], dtype=torch.int64),
        })
    
    def _step(self, tensordict: 'TensorDict') -> 'TensorDict':
        """
        TorchRL 标准 step 接口
        
        Args:
            tensordict: Must contain 'action' and 'time' keys
        
        Returns:
            TensorDict with next observation, reward, done flags
        """
        action = tensordict["action"].item()
        last_time = tensordict["time"].item()

        """Execute step in Petri net"""
        # 更新动作历史
        self.obs_builder.update_history(action)

        # 计算当前奖励
        reward1 = self.reward_calc.calculate_reward(self.marks, self.time)

        # 获取动作信息并执行链条
        ainfo = self._cache_action_info[action]
        chain, fire_times = ainfo.chain, ainfo.fire_times

        info = self.fire_executor.fire_chain(chain, fire_times, self.m, self.marks,path_updater=lambda tok: setattr(tok, 'where', tok.where + 1))
        self.m = info.m
        self.marks = info.marks

        # 计算新奖励
        reward2 = self.reward_calc.calculate_reward(self.marks, self.time)

        # 重建动作掩码
        tran_queue = self.get_enable_t(self.m, self.marks)
        self._cache_action_info.clear()
        mask_next = np.zeros(self.A, dtype=bool)

        for item in tran_queue:
            chain = tuple(item[3])
            fire_times = item[4]
            aid = self.chain2aid.get(chain, None)
            if aid is None:
                continue
            mask_next[aid] = True
            self._cache_action_info[aid] = ActionInfo(
                t=item[0], fire_times=fire_times, t_name=item[2], chain=item[3]
            )

        new_obs = self.obs_builder.build_observation(self.marks)
        finish = info.finish

        time_val = self.time
        reward_progress = reward2 - reward1

        # Calculate reward based on mode
        delta_time = time_val - last_time
        
        if self.reward_mode == 'progress':
            reward = reward_progress
        elif self.reward_mode == 'time':
            reward = -delta_time if delta_time > 0 else 0
        elif self.reward_mode == 'combined':
            reward = reward_progress - 0.1 * delta_time
        else:
            reward = reward_progress
        
        # Build next state
        next_state = TensorDict({
            "observation": torch.as_tensor(new_obs, dtype=torch.int64),
            "action_mask": torch.as_tensor(mask_next, dtype=torch.bool),
            "time": torch.tensor([time_val], dtype=torch.int64),
        })
        
        # Add reward
        next_state.set("reward", torch.tensor([reward], dtype=torch.float32))
        
        # Add done flags
        next_state.set("terminated", torch.tensor([finish], dtype=torch.bool))
        next_state.set("finish", torch.tensor([finish], dtype=torch.bool))
        next_state.set("done", torch.tensor([finish], dtype=torch.bool))
        
        return next_state
    
    def _set_seed(self, seed: Optional[int]):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)


    def _init_path(self):
        """
        初始化路径索引（优化版）

        不再在每个 token 中存储完整路径，而是通过 token 的 type 和 where 属性
        动态查找路径。这样可以节省内存并简化代码。

        路径信息存储在 PathRegistry 中，通过 get_token_path() 方法动态获取。
        """
        # 预先计算并缓存路径索引，供后续查找使用
        self.pathC_idx = self.path_registry.get_path_indices(self.id2t_name, 'C')
        self.pathD_idx = self.path_registry.get_path_indices(self.id2t_name, 'D')

        # 不再需要在 token 中存储路径
        # 路径将通过 get_token_path(token) 方法动态获取

    def get_token_path(self, token):
        """
        根据 token 的颜色（type）和 where 属性动态获取其路径

        Args:
            token: WaferToken 实例

        Returns:
            路径列表（transition ID 的嵌套列表）

        示例:
            >>> token = wafer_token  # type=2 (Route C), where=3
            >>> path = net.get_token_path(token)
            >>> # 返回从当前 where 位置开始的剩余路径
        """
        # 根据 token.type 确定使用哪条路径
        # type=1: Route D (LP1)
        # type=2: Route C (LP2)
        if token.type == 1:
            full_path = self.pathC_idx
        elif token.type == 2:
            full_path = self.pathD_idx
        else:
            return []

        # 根据 token.where 返回剩余路径
        # where 表示当前所在的阶段索引
        where = getattr(token, 'where', 0)

        # 返回从 where 开始的剩余路径
        return full_path[where:]

    def _clone_marks(self, marks: List[Place]) -> List[Place]:
        return [p.clone() for p in marks]

    def get_enable_t(self, m, mark):
        se = self.action_checker.resource_enable(m)
        se = self.action_checker.filter_by_round_robin(se)
        se_chain = self.action_checker.color_enable(se, mark)
        names = [self.id2t_name[t] for t in se]
        transition_queue = []
        for t, chain in se_chain:
            name = self.id2t_name[t]
            chain = [self.id2t_name[x] for x in chain]
            ok, times, end_time, _, _ = self.fire_executor.dry_run_chain(
                chain_names=chain, m=m, marks=mark,
                earliest_time_func=self.fire_executor.earliest_enable_time
            )
            if not ok:
                continue
            key_time = times[0]
            transition_queue.append((t, key_time, name, chain, times))
        return transition_queue
