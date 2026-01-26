"""性能基准测试：验证优化效果"""
import time
import pytest
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig


class TestPerformance:
    """性能测试类"""
    
    def test_turbo_mode_performance(self):
        """
        测试极速模式性能：100k 步/秒
        
        在标准开发机器上，使用训练模式配置（with_reward=True, detailed_reward=False, 极速模式启用），
        系统应在 1 秒内执行超过 100,000 个模拟步数。
        
        动作选择策略：从所有可用动作（所有 enabled 动作 + wait）中均匀随机选择，每个动作的概率相等。
        如果有 n 个 enabled 动作，则每个动作（包括 wait）的概率为 1/(n+1)。
        如果没有 enabled 动作，则只能选择 wait。
        """
        import random
        random.seed(42)  # 固定随机种子以保证可重复性
        
        config = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True  # 启用极速模式
        )
        env = Petri(config=config)
        env.reset()
        
        # 执行 100,000 步
        start = time.time()
        step_count = 0
        target_steps = 100000
        
        while step_count < target_steps:
            enabled = env.get_enable_t()
            
            # 从所有可用动作中均匀随机选择（所有 enabled 动作 + wait）
            # 每个动作的概率相等：1/(n+1)，其中 n 是 enabled 动作的数量
            if enabled:
                # 构建动作列表：所有 enabled 动作 + wait
                actions = list(enabled) + ['wait']
                # 均匀随机选择一个动作
                selected = random.choice(actions)
                
                if selected == 'wait':
                    env.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    env.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                # 如果没有 enabled，只能 wait
                env.step(wait=True, with_reward=True, detailed_reward=False)
            
            step_count += 1
            
            # 如果完成，重置环境继续测试
            if env.m[env.id2p_name.index("LP_done")] == env.n_wafer:
                env.reset()
        
        elapsed = time.time() - start
        steps_per_sec = target_steps / elapsed
        
        print(f"\n性能测试结果:")
        print(f"  执行步数: {target_steps:,}")
        print(f"  执行时间: {elapsed:.3f} 秒")
        print(f"  步数/秒: {steps_per_sec:,.0f}")
        print(f"  目标: >100,000 步/秒")
        print(f"  是否达标: {'[PASS]' if steps_per_sec > 100000 else '[FAIL]'}")
        
        assert elapsed < 1.0, f"性能测试失败: {elapsed:.3f}秒 > 1秒 (步数/秒: {steps_per_sec:,.0f})"
        assert steps_per_sec > 100000, f"性能未达标: {steps_per_sec:,.0f} 步/秒 < 100,000 步/秒"
    
    def test_optimization_improvement(self):
        """
        测试优化效果：1000 步执行时间减少至少 20%
        
        比较优化前后执行 1000 步的时间，验证优化效果。
        """
        # 优化前（禁用所有优化）
        config_old = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False,
            cache_indices=False
        )
        env_old = Petri(config=config_old)
        env_old.reset()
        
        start = time.time()
        for _ in range(1000):
            enabled = env_old.get_enable_t()
            if enabled:
                env_old.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                env_old.step(wait=True, with_reward=True, detailed_reward=False)
        time_old = time.time() - start
        
        # 优化后（启用所有优化）
        config_new = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_reward_calc=True,
            optimize_enable_check=True,
            optimize_state_update=True,
            cache_indices=True
        )
        env_new = Petri(config=config_new)
        env_new.reset()
        
        start = time.time()
        for _ in range(1000):
            enabled = env_new.get_enable_t()
            if enabled:
                env_new.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                env_new.step(wait=True, with_reward=True, detailed_reward=False)
        time_new = time.time() - start
        
        improvement = (time_old - time_new) / time_old * 100
        
        print(f"\n优化效果测试结果:")
        print(f"  优化前时间: {time_old:.3f} 秒")
        print(f"  优化后时间: {time_new:.3f} 秒")
        print(f"  改进幅度: {improvement:.1f}%")
        print(f"  目标: >= 20%")
        print(f"  是否达标: {'[PASS]' if improvement >= 20 else '[FAIL]'}")
        
        assert improvement >= 20, f"优化效果不足: {improvement:.1f}% < 20%"
    
    def test_episode_performance(self):
        """
        测试 Episode 性能：完成一个 episode 的时间减少至少 15%
        """
        # 优化前
        config_old = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=False,
            optimize_reward_calc=False,
            optimize_enable_check=False,
            optimize_state_update=False
        )
        
        # 运行 10 个 episode 取平均
        episode_times_old = []
        for _ in range(10):
            env_old = Petri(config=config_old)
            env_old.reset()
            start = time.time()
            
            done = False
            while not done:
                enabled = env_old.get_enable_t()
                if enabled:
                    done, _, _ = env_old.step(t=enabled[0], with_reward=True, detailed_reward=False)
                else:
                    done, _, _ = env_old.step(wait=True, with_reward=True, detailed_reward=False)
            
            episode_times_old.append(time.time() - start)
        
        avg_time_old = sum(episode_times_old) / len(episode_times_old)
        
        # 优化后
        config_new = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_reward_calc=True,
            optimize_enable_check=True,
            optimize_state_update=True
        )
        
        episode_times_new = []
        for _ in range(10):
            env_new = Petri(config=config_new)
            env_new.reset()
            start = time.time()
            
            done = False
            while not done:
                enabled = env_new.get_enable_t()
                if enabled:
                    done, _, _ = env_new.step(t=enabled[0], with_reward=True, detailed_reward=False)
                else:
                    done, _, _ = env_new.step(wait=True, with_reward=True, detailed_reward=False)
            
            episode_times_new.append(time.time() - start)
        
        avg_time_new = sum(episode_times_new) / len(episode_times_new)
        improvement = (avg_time_old - avg_time_new) / avg_time_old * 100
        
        print(f"\nEpisode 性能测试结果:")
        print(f"  优化前平均时间: {avg_time_old:.3f} 秒")
        print(f"  优化后平均时间: {avg_time_new:.3f} 秒")
        print(f"  改进幅度: {improvement:.1f}%")
        print(f"  目标: >= 15%")
        print(f"  是否达标: {'[PASS]' if improvement >= 15 else '[FAIL]'}")
        
        assert improvement >= 15, f"Episode 优化效果不足: {improvement:.1f}% < 15%"
    
    def test_data_structure_optimization_performance(self):
        """
        测试数据结构优化性能：10000 步执行时间减少至少 5%
        
        比较仅启用 002-sim-speedup 与启用 002-sim-speedup + 数据结构优化的性能差异。
        """
        import random
        random.seed(42)
        
        target_steps = 10000
        
        # 仅启用 002-sim-speedup (baseline)
        config_baseline = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,  # 禁用数据结构优化
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        start = time.time()
        step_count = 0
        while step_count < target_steps:
            enabled = env_baseline.get_enable_t()
            if enabled:
                actions = list(enabled) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    env_baseline.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
            step_count += 1
            if env_baseline.m[env_baseline.id2p_name.index("LP_done")] == env_baseline.n_wafer:
                env_baseline.reset()
        elapsed_baseline = time.time() - start
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,  # 启用数据结构优化
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        start = time.time()
        step_count = 0
        while step_count < target_steps:
            enabled = env_optimized.get_enable_t()
            if enabled:
                actions = list(enabled) + ['wait']
                selected = random.choice(actions)
                if selected == 'wait':
                    env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
                else:
                    env_optimized.step(t=selected, with_reward=True, detailed_reward=False)
            else:
                env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
            step_count += 1
            if env_optimized.m[env_optimized.id2p_name.index("LP_done")] == env_optimized.n_wafer:
                env_optimized.reset()
        elapsed_optimized = time.time() - start
        
        improvement = (elapsed_baseline - elapsed_optimized) / elapsed_baseline * 100
        
        print(f"\n数据结构优化性能测试结果:")
        print(f"  执行步数: {target_steps:,}")
        print(f"  Baseline 时间: {elapsed_baseline:.3f} 秒")
        print(f"  Optimized 时间: {elapsed_optimized:.3f} 秒")
        print(f"  改进幅度: {improvement:.1f}%")
        print(f"  目标: >= 5%")
        print(f"  是否达标: {'[PASS]' if improvement >= 5 else '[FAIL]'}")
        
        assert improvement >= 5, f"数据结构优化效果不足: {improvement:.1f}% < 5%"
    
    def test_frequent_access_optimization(self):
        """
        测试频繁访问操作优化：频繁访问操作执行时间减少至少 8%
        
        通过多次调用奖励计算、状态更新等频繁访问数据结构的操作来测试。
        """
        import random
        random.seed(42)
        
        # 仅启用 002-sim-speedup (baseline)
        config_baseline = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=False,
        )
        env_baseline = Petri(config=config_baseline)
        env_baseline.reset()
        
        # 执行频繁访问操作（奖励计算、状态更新等）
        start = time.time()
        for _ in range(1000):
            # 触发奖励计算
            enabled = env_baseline.get_enable_t()
            if enabled:
                env_baseline.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
            # 触发状态更新（通过 step 自动触发）
        elapsed_baseline = time.time() - start
        
        # 启用 002-sim-speedup + 数据结构优化
        config_optimized = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,
            optimize_data_structures=True,
        )
        env_optimized = Petri(config=config_optimized)
        env_optimized.reset()
        
        start = time.time()
        for _ in range(1000):
            enabled = env_optimized.get_enable_t()
            if enabled:
                env_optimized.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
        elapsed_optimized = time.time() - start
        
        improvement = (elapsed_baseline - elapsed_optimized) / elapsed_baseline * 100
        
        print(f"\n频繁访问操作优化测试结果:")
        print(f"  操作次数: 1000")
        print(f"  Baseline 时间: {elapsed_baseline:.3f} 秒")
        print(f"  Optimized 时间: {elapsed_optimized:.3f} 秒")
        print(f"  改进幅度: {improvement:.1f}%")
        print(f"  目标: >= 8%")
        print(f"  是否达标: {'[PASS]' if improvement >= 8 else '[FAIL]'}")
        
        assert improvement >= 8, f"频繁访问操作优化效果不足: {improvement:.1f}% < 8%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
