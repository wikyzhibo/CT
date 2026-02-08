"""
强化学习组件（Observation, Reward, ActionSpace）的单元测试。
"""

import pytest
import numpy as np
import torch
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig
from solutions.model.pn_models import WaferToken, Place

class TestObservationBuilder:
    """测试 ObservationBuilder 逻辑"""

    @pytest.fixture
    def env(self):
        return TimedPetri(PetriConfig.default())

    def test_observation_shape(self, env):
        """测试观测向量的维度"""
        obs_builder = env.obs_builder
        obs = obs_builder.build_observation(env.marks)
        
        expected_dim = 2 * len(env.obs_place_idx) + env.his_len
        assert obs.shape[0] == expected_dim

    def test_observation_content(self, env):
        """测试观测向量的具体内容（Marking 和 History）"""
        obs_builder = env.obs_builder
        # 初始状态，LP1 有晶圆 (type=1)
        obs = obs_builder.build_observation(env.marks)
        
        # 查找 LP1 在观测索引中的位置
        lp1_idx_in_obs = -1
        lp1_p_idx = env.id2p_name.index("P_READY__LP1")
        if lp1_p_idx in env.obs_place_idx:
            lp1_idx_in_obs = env.obs_place_idx.index(lp1_p_idx)
        
        if lp1_idx_in_obs != -1:
            # Type 1 晶圆在 obs1 段 (前 16 位)
            assert obs[lp1_idx_in_obs] > 0
            
        # 测试历史记录更新
        obs_builder.update_history(10) # 动作 ID = 10
        obs_new = obs_builder.build_observation(env.marks)
        # 历史记录在最后一段
        assert obs_new[-1] == 10

class TestRewardCalculator:
    """测试 RewardCalculator 逻辑"""

    @pytest.fixture
    def env(self):
        return TimedPetri(PetriConfig.default())

    def test_reward_progress(self, env):
        """测试进度奖惩逻辑"""
        calc = env.reward_calc
        marks = env._clone_marks(env.marks)
        
        # 初始状态 reward 可能为 0 (time=0)
        r0 = calc.calculate_reward(marks, 0)
        assert r0 == 0
        
        # 模拟一个 Wafer 推进到 Stage 3 (PM7/8)
        # 找到 P_READY__PM7 并放入一个 WaferToken
        pm7_idx = env.id2p_name.index("P_READY__PM7")
        token = WaferToken(enter_time=100, job_id=1, path=[], type=2) # Route D
        token.where = 3
        marks[pm7_idx].append(token)
        
        r1 = calc.calculate_reward(marks, 100)
        # 如果权重配置正确且 where=3 对应 stage_weights[3]
        # reward = weights[3] / time = 100 / 100 = 1.0 (根据默认权重)
        assert r1 > 0

class TestActionSpaceBuilder:
    """测试 ActionSpaceBuilder 逻辑"""

    @pytest.fixture
    def env(self):
        return TimedPetri(PetriConfig.default())

    def test_action_mapping(self, env):
        """测试动作 ID 与链条的映射一致性"""
        assert env.A > 0
        assert len(env.aid2chain) == env.A
        
        for aid, chain in enumerate(env.aid2chain):
            assert env.chain2aid[tuple(chain)] == aid

    def test_parallel_metadata(self, env):
        """测试并行动作的元数据"""
        # 找到属于 PM1-4 的动作，它们应该是并行的
        pm1_chain = None
        for chain in env.aid2chain:
            if "ARM3_PICK__LLC__TO__PM1" in chain:
                pm1_chain = chain
                break
        
        if pm1_chain:
            aid = env.chain2aid[tuple(pm1_chain)]
            assert env.aid_is_parallel[aid] == True
            assert env.aid_pstage[aid] != -1

if __name__ == "__main__":
    pytest.main([__file__])
