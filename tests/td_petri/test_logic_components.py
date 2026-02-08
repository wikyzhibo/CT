"""
Petri 网核心组件的单元测试。

验证 ActionEnableChecker, TransitionFireExecutor 等。
"""

import pytest
import numpy as np
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig
from solutions.model.pn_models import WaferToken, Place

class TestActionEnableChecker:
    """测试 ActionEnableChecker 的逻辑"""
    
    @pytest.fixture
    def env(self):
        """初始化一个默认环境"""
        config = PetriConfig.default()
        # 为了测试方便，将 LP1 的晶圆设为 1
        config.modules["LP1"].tokens = 1
        return TimedPetri(config)

    def test_resource_enable(self, env):
        """测试结构使能判定"""
        checker = env.action_checker
        # 初始状态下，LP1 的 PICK 变迁应该是结构使能的
        se = checker.resource_enable(env.m)
        t_names = [env.id2t_name[t] for t in se]
        
        # 查找包含 LP1 PICK 的变迁
        lp1_pick = [n for n in t_names if "ARM1_PICK__LP1" in n]
        assert len(lp1_pick) > 0

    def test_color_enable(self, env):
        """测试基于颜色的使能过滤"""
        checker = env.action_checker
        m = env.m.copy()
        marks = env._clone_marks(env.marks)
        
        # 获取结构使能的变迁
        se = checker.resource_enable(m)
        
        # 运行颜色使能检查
        se_chain = checker.color_enable(se, marks)
        
        # 初始状态，应该会有 [(t_id, chain), ...]
        assert len(se_chain) > 0
        for t, chain in se_chain:
            assert isinstance(t, (int, np.integer))
            assert isinstance(chain, list)

    def test_round_robin(self, env):
        """测试并行的 Round-Robin 轮询逻辑"""
        checker = env.action_checker
        # 构造一个有多个并行变迁可用的场景
        # 例如：PM1, PM2 同时可用
        # 这里通过手动修改 enable_ts 来测试逻辑
        pm1_id = env.id2t_name.index("ARM3_PICK__LLC__TO__PM1")
        pm2_id = env.id2t_name.index("ARM3_PICK__LLC__TO__PM2")
        
        ts = np.array([pm1_id, pm2_id])
        
        # 第一次过滤
        filtered1 = checker.filter_by_round_robin(ts)
        # 应该只保留一个（通常是第一个 PM1）
        assert len(filtered1) == 1
        chosen1 = filtered1[0]
        
        # 模拟发射 chosen1 后的更新
        checker.update_rr_after_fire(chosen1)
        
        # 第二次过滤相同的 ts
        filtered2 = checker.filter_by_round_robin(ts)
        assert len(filtered2) == 1
        chosen2 = filtered2[0]
        
        # 此时应该选择了不同的机器 (PM2)
        assert chosen1 != chosen2

class TestTransitionFireExecutor:
    """测试变迁发射执行器"""

    @pytest.fixture
    def env(self):
        return TimedPetri(PetriConfig.default())

    def test_single_fire(self, env):
        """测试单变迁发射的 Marking 更新和资源记录"""
        executor = env.fire_executor
        m = env.m.copy()
        marks = env._clone_marks(env.marks)
        
        # 找一个具体的变迁：LP1 的 PICK
        t_id = env.id2t_name.index("ARM1_PICK__LP1__TO__AL")
        
        # 发射前 LP1 应该有 Token
        lp1_idx = env.id2p_name.index("P_READY__LP1")
        assert m[lp1_idx] > 0
        
        # 执行发射
        result = executor.fire(t_id, m, marks, start_from=0)
        
        # 验证发射后的 Marking
        assert result.m[lp1_idx] == m[lp1_idx] - 1
        
        # 验证资源占用记录 (ARM1 应该被占用)
        assert "ARM1" in env.resource_mgr.res_occ
        occ = env.resource_mgr.res_occ["ARM1"]
        assert len(occ) == 1
        assert occ[0].start == 0

    def test_dry_run_chain_timing(self, env):
        """测试 dry_run_chain 返回的时间准确性"""
        executor = env.fire_executor
        m = env.m.copy()
        marks = env._clone_marks(env.marks)
        
        # 定义一个简单的链条：PICK -> MOVE -> LOAD
        chain = [
            "ARM1_PICK__LP1__TO__AL",
            "ARM1_MOVE__LP1__TO__AL",
            "ARM1_LOAD__LP1__TO__AL"
        ]
        
        ok, times, end_time, _, _ = executor.dry_run_chain(
            chain, m, marks, 
            earliest_time_func=lambda t, m, marks: 0
        )
        
        assert ok
        assert len(times) == 3
        # 验证时间累加逻辑：0, 5, 10 (PICK=5, MOVE=5)
        assert times[0] == 0
        assert times[1] == 5
        assert times[2] == 10
        assert end_time == 15 # LOAD=5

if __name__ == "__main__":
    pytest.main([__file__])
