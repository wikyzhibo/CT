"""
黄金用例测试：3 晶圆全流程硬编码推理验证。

场景：
1. 3 个晶圆从 LP1 出发。
2. 依次经过 PICK -> MOVE -> LOAD 到 AL 腔。
3. 验证资源竞争（ARM1 同时只能搬运一个）和加工时间。
"""

import pytest
import numpy as np
import torch
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig

class TestGoldenCase3Wafers:
    """3 晶圆场景的硬编码推理验证"""

    @pytest.fixture
    def env(self):
        """配置 3 个晶圆的测试环境"""
        config = PetriConfig.default()
        config.modules["LP1"].tokens = 3
        config.modules["LP2"].tokens = 0
        return TimedPetri(config)

    def test_three_wafers_flow(self, env):
        """测试 3 个晶圆从进入到第一个腔室的逻辑推理"""
        # 初始重置
        td = env.reset()
        current_time = 0
        
        # --- 晶圆 1 的操作 ---
        # 1. 选择第一个动作：LP1 -> AL 的链条
        mask = td["action_mask"].numpy()
        lp1_to_al_aid = -1
        for aid, chain in enumerate(env.aid2chain):
            if "ARM1_PICK__LP1__TO__AL" in chain:
                lp1_to_al_aid = aid
                break
        
        assert lp1_to_al_aid != -1
        assert mask[lp1_to_al_aid] == True
        
        # 执行动作 1 (晶圆 1)
        action = torch.tensor([lp1_to_al_aid])
        td["action"] = action
        td_next = env.step(td)
        
        # 推理晶圆 1 时间：
        # PICK(5) + MOVE(5) + LOAD(5) = 15
        assert env.time == 15
        
        # --- 晶圆 2 的操作 ---
        # 此时晶圆 1 正在 AL 加工 (10s) 或准备下一步
        # 但我们关注晶圆 2 能否开始 PICK
        # ARM1 已经在 15s 释放，所以晶圆 2 应该在 15s 开始 PICK
        mask2 = td_next["action_mask"].numpy()
        assert mask2[lp1_to_al_aid] == True
        
        td_next["action"] = action # 再次执行相同的链条给晶圆 2
        td_next2 = env.step(td_next)
        
        # 推理晶圆 2 时间：
        # 开始时间 15 + PICK(5) + MOVE(5) + LOAD(5) = 30
        assert env.time == 30
        
        # --- 晶圆 3 的操作 ---
        # 晶圆 3 应该在 30s 开始 PICK
        mask3 = td_next2["action_mask"].numpy()
        assert mask3[lp1_to_al_aid] == True
        
        td_next2["action"] = action
        td_next3 = env.step(td_next2)
        
        # 推理晶圆 3 时间：
        # 开始时间 30 + 15 = 45
        assert env.time == 45
        
        # --- 验证 Marking ---
        # 此时 AL 应该有晶圆，或者已经加工完进入下一阶段了
        # 在这个简单的测试中，我们只验证这 3 次搬运是否成功计入了资源时间轴
        occ = env.resource_mgr.res_occ["ARM1"]
        assert len(occ) == 9 # 每个动作产生 3 个 Interval (PICK, MOVE, LOAD) -> 3 * 3 = 9
        
        # 验证第二个动作的第一个区间 (晶圆 2 的 PICK) 起始点是否为 15
        # 注意：occ 是按开始时间排序的，索引 3 应该是第二次操作的开始
        assert occ[3].start == 15
        assert occ[6].start == 30

    def test_full_process_done(self, env):
        """验证 3 个晶圆最终都能到达 LP_done 并保留标记"""
        # 这是一个集成性质的测试，模拟自动运行直到结束
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            # 简单策略：总是选第一个可用动作
            mask = env.action_checker.resource_enable(env.m)
            # 转为 RL 掩码逻辑
            tran_queue = env.get_enable_t(env.m, env.marks)
            if not tran_queue:
                break
                
            # 执行第一个可用动作
            chain = tran_queue[0][3]
            fire_times = tran_queue[0][4]
            env.fire_executor.fire_chain(chain, fire_times, env.m, env.marks, 
                                        path_updater=lambda tok: setattr(tok, 'where', tok.where + 1))
            
            steps += 1
            if env.m[env.lp_done_idx] == 3:
                done = True
        
        # 验证结束状态
        assert env.m[env.lp_done_idx] == 3
        # 验证 Token 是否保留
        lp_done_place = env.marks[env.lp_done_idx]
        assert len(lp_done_place.tokens) == 3
        for tok in lp_done_place.tokens:
            assert tok.enter_time <= env.time

if __name__ == "__main__":
    pytest.main([__file__])
