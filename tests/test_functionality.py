"""功能一致性测试：验证优化后功能保持一致"""
import numpy as np
import pytest
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig


class TestFunctionality:
    """功能一致性测试类"""
    
    def test_state_consistency(self):
        """
        测试状态一致性：验证优化前后状态转换一致
        
        使用相同随机种子和动作序列，比较优化前后的最终状态。
        """
        np.random.seed(42)
        
        # 优化前
        config_old = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False,
            cache_indices=False
        )
        env_old = Petri(config=config_old)
        env_old.reset()
        
        # 优化后
        config_new = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_reward_calc=True,
            optimize_enable_check=True,
            optimize_state_update=True,
            cache_indices=True
        )
        env_new = Petri(config=config_new)
        env_new.reset()
        
        # 执行相同动作序列（100 步）
        for step in range(100):
            enabled_old = env_old.get_enable_t()
            enabled_new = env_new.get_enable_t()
            
            assert enabled_old == enabled_new, f"步骤 {step}: 使能变迁不一致 (old: {enabled_old}, new: {enabled_new})"
            
            if enabled_old:
                # 选择相同的动作
                action = enabled_old[0] if enabled_old else None
                if action is not None:
                    _, _, _ = env_old.step(t=action, with_reward=True, detailed_reward=False)
                    _, _, _ = env_new.step(t=action, with_reward=True, detailed_reward=False)
            else:
                _, _, _ = env_old.step(wait=True, with_reward=True, detailed_reward=False)
                _, _, _ = env_new.step(wait=True, with_reward=True, detailed_reward=False)
            
            # 检查状态一致性
            assert np.array_equal(env_old.m, env_new.m), f"步骤 {step}: 标记向量不一致"
            assert env_old.time == env_new.time, f"步骤 {step}: 系统时间不一致"
        
        # 检查最终状态
        final_state_old = {
            'm': env_old.m.copy(),
            'time': env_old.time,
            'scrap_count': env_old.scrap_count
        }
        final_state_new = {
            'm': env_new.m.copy(),
            'time': env_new.time,
            'scrap_count': env_new.scrap_count
        }
        
        assert np.array_equal(final_state_old['m'], final_state_new['m']), "最终标记向量不一致"
        assert final_state_old['time'] == final_state_new['time'], "最终系统时间不一致"
        assert final_state_old['scrap_count'] == final_state_new['scrap_count'], "报废计数不一致"
        
        print("[PASS] 状态一致性测试通过")
    
    def test_reward_consistency(self):
        """
        测试奖励一致性：验证优化前后奖励计算一致
        
        使用相同随机种子和动作序列，比较优化前后的奖励序列和总奖励。
        注意：turbo_mode 使用简化的奖励计算，不参与一致性测试。
        """
        np.random.seed(42)
        
        # 优化前（禁用所有优化）
        config_old = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False
        )
        env_old = Petri(config=config_old)
        env_old.reset()
        
        # 优化后（启用向量化奖励计算，但不启用 turbo_mode）
        config_new = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=False,  # turbo_mode 使用简化奖励，不参与一致性测试
            optimize_reward_calc=True,
            optimize_enable_check=True,
            optimize_state_update=True
        )
        env_new = Petri(config=config_new)
        env_new.reset()
        
        rewards_old = []
        rewards_new = []
        
        # 执行相同动作序列（50 步）
        for step in range(50):
            enabled_old = env_old.get_enable_t()
            enabled_new = env_new.get_enable_t()
            
            if enabled_old:
                action = enabled_old[0]
                _, reward_old, _ = env_old.step(t=action, with_reward=True, detailed_reward=False)
                _, reward_new, _ = env_new.step(t=action, with_reward=True, detailed_reward=False)
            else:
                _, reward_old, _ = env_old.step(wait=True, with_reward=True, detailed_reward=False)
                _, reward_new, _ = env_new.step(wait=True, with_reward=True, detailed_reward=False)
            
            rewards_old.append(reward_old)
            rewards_new.append(reward_new)
            
            # 检查奖励一致性（允许小的浮点误差）
            assert abs(reward_old - reward_new) < 1e-6, f"步骤 {step}: 奖励不一致 (old: {reward_old}, new: {reward_new})"
        
        # 检查总奖励
        total_reward_old = sum(rewards_old)
        total_reward_new = sum(rewards_new)
        
        assert abs(total_reward_old - total_reward_new) < 1e-5, f"总奖励不一致 (old: {total_reward_old}, new: {total_reward_new})"
        
        print(f"[PASS] 奖励一致性测试通过 (总奖励: {total_reward_old:.2f})")
    
    def test_event_consistency(self):
        """
        测试事件一致性：验证优化前后核心事件日志一致
        
        比较关键事件（如变迁发射、状态变化）的一致性。
        """
        np.random.seed(42)
        
        # 优化前
        config_old = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False
        )
        env_old = Petri(config=config_old)
        env_old.reset()
        
        # 优化后
        config_new = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_reward_calc=True,
            optimize_enable_check=True,
            optimize_state_update=True
        )
        env_new = Petri(config=config_new)
        env_new.reset()
        
        events_old = []
        events_new = []
        
        # 执行相同动作序列（30 步）
        for step in range(30):
            enabled_old = env_old.get_enable_t()
            enabled_new = env_new.get_enable_t()
            
            assert enabled_old == enabled_new, f"步骤 {step}: 使能变迁不一致"
            
            if enabled_old:
                action = enabled_old[0]
                done_old, _, scrap_old = env_old.step(t=action, with_reward=True, detailed_reward=False)
                done_new, _, scrap_new = env_new.step(t=action, with_reward=True, detailed_reward=False)
                
                events_old.append({
                    'step': step,
                    'action': action,
                    'time': env_old.time,
                    'done': done_old,
                    'scrap': scrap_old
                })
                events_new.append({
                    'step': step,
                    'action': action,
                    'time': env_new.time,
                    'done': done_new,
                    'scrap': scrap_new
                })
                
                assert done_old == done_new, f"步骤 {step}: 完成状态不一致"
                assert scrap_old == scrap_new, f"步骤 {step}: 报废状态不一致"
                assert env_old.time == env_new.time, f"步骤 {step}: 系统时间不一致"
            else:
                done_old, _, scrap_old = env_old.step(wait=True, with_reward=True, detailed_reward=False)
                done_new, _, scrap_new = env_new.step(wait=True, with_reward=True, detailed_reward=False)
                
                assert done_old == done_new, f"步骤 {step}: 完成状态不一致"
                assert scrap_old == scrap_new, f"步骤 {step}: 报废状态不一致"
        
        # 检查事件序列一致性
        assert len(events_old) == len(events_new), "事件数量不一致"
        for i, (event_old, event_new) in enumerate(zip(events_old, events_new)):
            assert event_old['action'] == event_new['action'], f"事件 {i}: 动作不一致"
            assert event_old['time'] == event_new['time'], f"事件 {i}: 时间不一致"
            assert event_old['done'] == event_new['done'], f"事件 {i}: 完成状态不一致"
            assert event_old['scrap'] == event_new['scrap'], f"事件 {i}: 报废状态不一致"
        
        print(f"[PASS] 事件一致性测试通过 ({len(events_old)} 个事件)")
    
    def test_config_disabled_consistency(self):
        """
        测试配置禁用一致性：验证禁用所有优化时行为与未优化版本一致
        """
        np.random.seed(42)
        
        # 禁用所有优化
        config_disabled = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False,
            cache_indices=False
        )
        env_disabled = Petri(config=config_disabled)
        env_disabled.reset()
        
        # 未优化版本（使用默认配置，假设默认也是禁用优化）
        config_default = PetriEnvConfig(
            n_wafer=4,
            training_phase=2
        )
        env_default = Petri(config=config_default)
        env_default.reset()
        
        # 执行相同动作序列（20 步）
        for step in range(20):
            enabled_disabled = env_disabled.get_enable_t()
            enabled_default = env_default.get_enable_t()
            
            if enabled_disabled:
                action = enabled_disabled[0]
                _, reward_disabled, _ = env_disabled.step(t=action, with_reward=True, detailed_reward=False)
                _, reward_default, _ = env_default.step(t=action, with_reward=True, detailed_reward=False)
                
                assert abs(reward_disabled - reward_default) < 1e-6, f"步骤 {step}: 奖励不一致"
            else:
                _, reward_disabled, _ = env_disabled.step(wait=True, with_reward=True, detailed_reward=False)
                _, reward_default, _ = env_default.step(wait=True, with_reward=True, detailed_reward=False)
                
                assert abs(reward_disabled - reward_default) < 1e-6, f"步骤 {step}: 奖励不一致"
        
        print("[PASS] 配置禁用一致性测试通过")


    def test_data_structure_consistency(self):
        """
        测试数据结构优化一致性：验证优化后的数据结构与优化前产生相同的核心结果
        
        使用相同随机种子和动作序列，比较启用数据结构优化前后的最终状态、奖励和核心事件。
        """
        import random
        random.seed(42)
        
        # 仅启用 002-sim-speedup (baseline)
        config_baseline = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,  # 禁用数据结构优化
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,  # 启用数据结构优化
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        # 执行相同动作序列（100 步）
        rewards_baseline = []
        rewards_optimized = []
        
        for step in range(100):
            enabled_baseline = env_baseline.get_enable_t()
            enabled_optimized = env_optimized.get_enable_t()
            
            assert enabled_baseline == enabled_optimized, f"步骤 {step}: 使能变迁不一致"
            
            if enabled_baseline:
                actions = list(enabled_baseline) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    _, reward_b, _ = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                    _, reward_o, _ = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    _, reward_b, _ = env_baseline.step(t=selected, with_reward=True, detailed_reward=False)
                    _, reward_o, _ = env_optimized.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                _, reward_b, _ = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                _, reward_o, _ = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
            
            rewards_baseline.append(reward_b)
            rewards_optimized.append(reward_o)
            
            # 检查状态一致性
            assert np.array_equal(env_baseline.m, env_optimized.m), f"步骤 {step}: 标记向量不一致"
            assert env_baseline.time == env_optimized.time, f"步骤 {step}: 时间不一致"
        
        # 检查奖励一致性
        for step, (r_b, r_o) in enumerate(zip(rewards_baseline, rewards_optimized)):
            assert abs(r_b - r_o) < 1e-6, f"步骤 {step}: 奖励不一致 (baseline: {r_b}, optimized: {r_o})"
        
        # 检查最终状态
        assert np.array_equal(env_baseline.m, env_optimized.m), "最终标记向量不一致"
        assert env_baseline.time == env_optimized.time, "最终时间不一致"
        
        print(f"[PASS] 数据结构优化功能一致性测试通过")
    
    def test_compatibility_with_sim_speedup(self):
        """
        测试数据结构优化与 002-sim-speedup 的兼容性
        
        验证同时启用两种优化措施时，系统能正常工作且不产生冲突。
        """
        config = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,  # 启用 002-sim-speedup
            optimize_data_structures=True,  # 启用数据结构优化
        )
        env = Petri(config=config)
        env.reset()
        
        # 执行一些步骤，确保不报错
        for _ in range(50):
            enabled = env.get_enable_t()
            if enabled:
                env.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                env.step(wait=True, with_reward=True, detailed_reward=False)
        
        # 验证缓存存在
        assert hasattr(env, '_marks_by_type'), "_marks_by_type 缓存不存在"
        assert len(env._marks_by_type) > 0, "_marks_by_type 缓存为空"
        
        # 验证可以访问缓存
        process_places = env._get_marks_by_type(1)
        assert len(process_places) > 0, "无法获取 type=1 的库所"
        
        print(f"[PASS] 数据结构优化与 002-sim-speedup 兼容性测试通过")
    
    def test_state_consistency_with_data_structure_optimization(self):
        """
        测试状态一致性（启用数据结构优化）：验证状态转换一致性
        
        使用相同随机种子和动作序列，比较启用数据结构优化前后的状态转换。
        """
        import random
        random.seed(42)
        
        # 仅启用 002-sim-speedup
        config_baseline = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        # 执行相同动作序列（100 步），检查每个步骤的状态
        for step in range(100):
            enabled_baseline = env_baseline.get_enable_t()
            enabled_optimized = env_optimized.get_enable_t()
            
            assert enabled_baseline == enabled_optimized, f"步骤 {step}: 使能变迁不一致"
            assert np.array_equal(env_baseline.m, env_optimized.m), f"步骤 {step}: 标记向量不一致"
            assert env_baseline.time == env_optimized.time, f"步骤 {step}: 时间不一致"
            
            if enabled_baseline:
                actions = list(enabled_baseline) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                    env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    env_baseline.step(t=selected, with_reward=True, detailed_reward=False)
                    env_optimized.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
        
        # 检查最终状态
        assert np.array_equal(env_baseline.m, env_optimized.m), "最终标记向量不一致"
        assert env_baseline.time == env_optimized.time, "最终时间不一致"
        
        print(f"[PASS] 状态一致性测试通过（启用数据结构优化）")
    
    def test_reward_consistency_with_data_structure_optimization(self):
        """
        测试奖励一致性（启用数据结构优化）：验证奖励计算一致性
        
        使用相同随机种子和动作序列，比较启用数据结构优化前后的奖励序列。
        """
        import random
        random.seed(42)
        
        # 仅启用 002-sim-speedup
        config_baseline = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        # 执行相同动作序列（100 步），收集奖励
        rewards_baseline = []
        rewards_optimized = []
        
        for step in range(100):
            enabled_baseline = env_baseline.get_enable_t()
            enabled_optimized = env_optimized.get_enable_t()
            
            assert enabled_baseline == enabled_optimized, f"步骤 {step}: 使能变迁不一致"
            
            if enabled_baseline:
                actions = list(enabled_baseline) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    _, reward_b, _ = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                    _, reward_o, _ = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    _, reward_b, _ = env_baseline.step(t=selected, with_reward=True, detailed_reward=False)
                    _, reward_o, _ = env_optimized.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                _, reward_b, _ = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                _, reward_o, _ = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
            
            rewards_baseline.append(reward_b)
            rewards_optimized.append(reward_o)
        
        # 检查奖励一致性
        for step, (r_b, r_o) in enumerate(zip(rewards_baseline, rewards_optimized)):
            assert abs(r_b - r_o) < 1e-6, f"步骤 {step}: 奖励不一致 (baseline: {r_b}, optimized: {r_o})"
        
        print(f"[PASS] 奖励一致性测试通过（启用数据结构优化）")
    
    def test_event_consistency_with_data_structure_optimization(self):
        """
        测试核心事件一致性（启用数据结构优化）：验证核心事件一致性
        
        使用相同随机种子和动作序列，比较启用数据结构优化前后的核心事件（如完成、报废等）。
        """
        import random
        random.seed(42)
        
        # 仅启用 002-sim-speedup
        config_baseline = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        # 执行相同动作序列，检查核心事件
        done_baseline = False
        done_optimized = False
        scrap_baseline = False
        scrap_optimized = False
        
        step = 0
        while step < 1000 and not (done_baseline and done_optimized):
            enabled_baseline = env_baseline.get_enable_t()
            enabled_optimized = env_optimized.get_enable_t()
            
            if enabled_baseline:
                actions = list(enabled_baseline) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    done_b, _, scrap_b = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                    done_o, _, scrap_o = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    done_b, _, scrap_b = env_baseline.step(t=selected, with_reward=True, detailed_reward=False)
                    done_o, _, scrap_o = env_optimized.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                done_b, _, scrap_b = env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                done_o, _, scrap_o = env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
            
            done_baseline = done_b
            done_optimized = done_o
            scrap_baseline = scrap_b
            scrap_optimized = scrap_o
            
            # 检查事件一致性
            assert done_baseline == done_optimized, f"步骤 {step}: 完成状态不一致"
            assert scrap_baseline == scrap_optimized, f"步骤 {step}: 报废状态不一致"
            
            step += 1
        
        print(f"[PASS] 核心事件一致性测试通过（启用数据结构优化）")
    
    def test_simultaneous_optimizations_compatibility(self):
        """
        测试同时启用两种优化的兼容性：验证同时启用 002-sim-speedup 和数据结构优化时不产生冲突
        
        运行一个完整的 episode，确保两种优化措施可以协同工作。
        """
        config = PetriEnvConfig(
            n_wafer=4,
            training_phase=2,
            turbo_mode=True,  # 启用 002-sim-speedup
            optimize_data_structures=True,  # 启用数据结构优化
        )
        env = Petri(config=config)
        env.reset()
        
        # 验证两种优化都启用
        assert env.turbo_mode == True, "turbo_mode 未启用"
        assert env.optimize_data_structures == True, "optimize_data_structures 未启用"
        
        # 验证缓存存在
        assert hasattr(env, '_marks_by_type'), "_marks_by_type 缓存不存在"
        assert len(env._marks_by_type) > 0, "_marks_by_type 缓存为空"
        
        # 执行一个完整的 episode
        done = False
        step_count = 0
        max_steps = 1000
        
        while not done and step_count < max_steps:
            enabled = env.get_enable_t()
            if enabled:
                done, _, _ = env.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                done, _, _ = env.step(wait=True, with_reward=True, detailed_reward=False)
            step_count += 1
        
        # 验证没有错误发生
        assert step_count < max_steps or done, "Episode 未完成或超时"
        
        # 验证缓存仍然有效
        assert hasattr(env, '_marks_by_type'), "缓存在运行后丢失"
        
        print(f"[PASS] 同时启用两种优化的兼容性测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
