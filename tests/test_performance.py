"""性能基准测试：验证优化效果"""
import time
import pytest
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig


def measure_step_rate(env, num_steps=10000):
    """
    测量步速的工具函数
    
    Args:
        env: Petri网环境实例
        num_steps: 要执行的步数
        
    Returns:
        tuple: (steps_per_second, elapsed_time)
    """
    import random
    random.seed(42)
    
    start = time.time()
    step_count = 0
    
    while step_count < num_steps:
        enabled = env.get_enable_t()
        
        if enabled:
            actions = list(enabled) + ['wait']
            selected = random.choice(actions)
            
            if selected == 'wait':
                env.step(wait=True, with_reward=True, detailed_reward=False)
            else:
                env.step(t=selected, with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
        
        step_count += 1
        
        # 如果完成，重置环境继续测试
        if env.m[env._get_place_index("LP_done")] == env.n_wafer:
            env.reset()
    
    elapsed = time.time() - start
    steps_per_sec = num_steps / elapsed
    
    return steps_per_sec, elapsed


def measure_episode_time(env, num_episodes=10):
    """
    测量episode执行时间的工具函数
    
    Args:
        env: Petri网环境实例
        num_episodes: 要运行的episode数量
        
    Returns:
        tuple: (average_time_per_episode, total_time, episode_times_list)
    """
    episode_times = []
    
    for _ in range(num_episodes):
        env.reset()
        start = time.time()
        
        done = False
        while not done:
            enabled = env.get_enable_t()
            if enabled:
                done, _, _ = env.step(t=enabled[0], with_reward=True, detailed_reward=False)
            else:
                done, _, _ = env.step(wait=True, with_reward=True, detailed_reward=False)
        
        episode_times.append(time.time() - start)
    
    avg_time = sum(episode_times) / len(episode_times)
    total_time = sum(episode_times)
    
    return avg_time, total_time, episode_times


class TestPerformance:
    """性能测试类"""
    
    def test_turbo_80000_steps_per_second(self):
        """
        测试极速模式性能：80,000 步/秒（User Story 1目标）
        
        在标准开发机器上，使用训练模式配置（with_reward=True, detailed_reward=False, 极速模式启用），
        系统应在 1 秒内执行至少 80,000 个模拟步数。
        
        动作选择策略：从所有可用动作（所有 enabled 动作 + wait）中均匀随机选择，每个动作的概率相等。
        如果有 n 个 enabled 动作，则每个动作（包括 wait）的概率为 1/(n+1)。
        如果没有 enabled 动作，则只能选择 wait。
        """
        import random
        random.seed(42)  # 固定随机种子以保证可重复性
        
        config = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,  # 启用极速模式
            optimize_state_update=True,
            cache_indices=True,
            optimize_data_structures=True
        )
        env = Petri(config=config)
        env.reset()
        
        # 执行 80,000 步
        start = time.time()
        step_count = 0
        target_steps = 800000
        
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
            if env.m[env._get_place_index("LP_done")] == env.n_wafer:
                env.reset()
        
        elapsed = time.time() - start
        steps_per_sec = target_steps / elapsed
        
        print(f"\n性能测试结果 (User Story 1):")
        print(f"  执行步数: {target_steps:,}")
        print(f"  执行时间: {elapsed:.3f} 秒")
        print(f"  步数/秒: {steps_per_sec:,.0f}")
        print(f"  目标: >= 80,000 步/秒")
        print(f"  是否达标: {'[PASS]' if steps_per_sec >= 80000 else '[FAIL]'}")
        
        #assert elapsed <= 1.0, f"性能测试失败: {elapsed:.3f}秒 > 1秒 (步数/秒: {steps_per_sec:,.0f})"
        assert steps_per_sec >= 80000, f"性能未达标: {steps_per_sec:,.0f} 步/秒 < 80,000 步/秒"
    
    def test_episode_time_improvement(self):
        """
        测试 Episode 性能改进：100个episode的总时间减少至少15%（User Story 2目标）
        """
        # 优化前（基准配置）
        config_old = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=False,
            optimize_state_update=False,
            cache_indices=False,
            optimize_data_structures=False
        )
        
        # 运行 100 个 episode 测量总时间
        print("\n运行基准配置（100个episode）...")
        start_old = time.time()
        for _ in range(100):
            env_old = Petri(config=config_old)
            env_old.reset()
            
            done = False
            while not done:
                enabled = env_old.get_enable_t()
                if enabled:
                    done, _, _ = env_old.step(t=enabled[0], with_reward=True, detailed_reward=False)
                else:
                    done, _, _ = env_old.step(wait=True, with_reward=True, detailed_reward=False)
        
        total_time_old = time.time() - start_old
        
        # 优化后（turbo模式）
        config_new = PetriEnvConfig(
            n_wafer=24,
            training_phase=2,
            turbo_mode=True,  # 启用极速模式
            optimize_state_update=True,
            cache_indices=True,
            optimize_data_structures=True
        )
        
        # 运行 100 个 episode 测量总时间
        print("运行turbo模式配置（100个episode）...")
        start_new = time.time()
        for _ in range(100):
            env_new = Petri(config=config_new)
            env_new.reset()
            
            done = False
            while not done:
                enabled = env_new.get_enable_t()
                if enabled:
                    done, _, _ = env_new.step(t=enabled[0], with_reward=True, detailed_reward=False)
                else:
                    done, _, _ = env_new.step(wait=True, with_reward=True, detailed_reward=False)
        
        total_time_new = time.time() - start_new
        improvement = (total_time_old - total_time_new) / total_time_old * 100
        
        print(f"\nEpisode 性能测试结果 (User Story 2):")
        print(f"  优化前总时间（100个episode）: {total_time_old:.3f} 秒")
        print(f"  优化后总时间（100个episode）: {total_time_new:.3f} 秒")
        print(f"  改进幅度: {improvement:.1f}%")
        print(f"  目标: >= 15%")
        print(f"  是否达标: {'[PASS]' if improvement >= 15 else '[FAIL]'}")
        
        assert improvement >= 15, f"Episode 优化效果不足: {improvement:.1f}% < 15%"
    
    def test_turbo_mode_consistency(self):
        """
        测试极速模式与标准模式的一致性（User Story 3）
        
        使用相同的随机种子和动作序列，验证两种模式产生相同的结果。
        """
        import random
        
        # 测试配置
        n_wafer = 24
        random_seed = 42
        
        # 标准模式
        # 标准模式：不使用任何优化，使用原始实现
        config_standard = PetriEnvConfig(
            n_wafer=n_wafer,
            training_phase=2,
            turbo_mode=False,
            optimize_state_update=False,
            cache_indices=False,
            optimize_data_structures=False
        )
        
        # Turbo模式：启用所有优化
        config_turbo = PetriEnvConfig(
            n_wafer=n_wafer,
            training_phase=2,
            turbo_mode=True,
            optimize_state_update=True,
            cache_indices=True,
            optimize_data_structures=True
        )
        
        # 运行标准模式
        random.seed(random_seed)
        env_standard = Petri(config=config_standard)
        env_standard.reset()
        
        rewards_standard = []
        done_standard = False
        step_count = 0
        max_steps = 500  # 减少步数以加快测试
        
        # 预先生成动作序列，确保两个模式使用完全相同的动作序列
        random.seed(random_seed)
        action_sequence = []
        temp_env = Petri(config=config_standard)
        temp_env.reset()
        for _ in range(max_steps * 2):  # 生成足够多的动作
            enabled = temp_env.get_enable_t()
            if enabled:
                actions = list(enabled) + ['wait']
                selected = random.choice(actions)
                action_sequence.append(selected)
                if selected == 'wait':
                    result = temp_env.step(wait=True, with_reward=False)
                    done = result[0]
                else:
                    result = temp_env.step(t=selected, with_reward=False)
                    done = result[0]
                if done or temp_env.m[temp_env._get_place_index("LP_done")] == temp_env.n_wafer:
                    break
            else:
                action_sequence.append('wait')
                result = temp_env.step(wait=True, with_reward=False)
                done = result[0]
                if done or temp_env.m[temp_env._get_place_index("LP_done")] == temp_env.n_wafer:
                    break
        
        # 运行标准模式（使用预生成的动作序列）
        random.seed(random_seed)  # 重置随机种子
        env_standard = Petri(config=config_standard)
        env_standard.reset()
        
        rewards_standard = []
        done_standard = False
        step_count = 0
        
        for action in action_sequence:
            if done_standard or step_count >= max_steps:
                break
            if action == 'wait':
                done_standard, reward, scrap = env_standard.step(wait=True, with_reward=True, detailed_reward=False)
            else:
                done_standard, reward, scrap = env_standard.step(t=action, with_reward=True, detailed_reward=False)
            
            rewards_standard.append(reward)
            step_count += 1
            
            if env_standard.m[env_standard._get_place_index("LP_done")] == env_standard.n_wafer:
                break
        
        # 运行Turbo模式（使用相同的动作序列）
        random.seed(random_seed)  # 重置随机种子
        env_turbo = Petri(config=config_turbo)
        env_turbo.reset()
        
        rewards_turbo = []
        done_turbo = False
        step_count = 0
        
        for action in action_sequence:
            if done_turbo or step_count >= max_steps:
                break
            if action == 'wait':
                done_turbo, reward, scrap = env_turbo.step(wait=True, with_reward=True, detailed_reward=False)
            else:
                done_turbo, reward, scrap = env_turbo.step(t=action, with_reward=True, detailed_reward=False)
            
            rewards_turbo.append(reward)
            step_count += 1
            
            if env_turbo.m[env_turbo._lp_done_idx] == env_turbo.n_wafer:
                break
        
        # 比较结果
        total_reward_standard = sum(rewards_standard)
        total_reward_turbo = sum(rewards_turbo)
        reward_diff = abs(total_reward_standard - total_reward_turbo)
        reward_diff_pct = (reward_diff / abs(total_reward_standard)) * 100 if total_reward_standard != 0 else 0
        
        print(f"\n一致性测试结果 (User Story 3):")
        print(f"  标准模式总奖励: {total_reward_standard:.2f}")
        print(f"  Turbo模式总奖励: {total_reward_turbo:.2f}")
        print(f"  奖励差异: {reward_diff:.2f} ({reward_diff_pct:.2f}%)")
        print(f"  目标: < 1% 差异")
        print(f"  是否达标: {'[PASS]' if reward_diff_pct < 1.0 else '[FAIL]'}")
        
        # 验证奖励差异 < 1%
        assert reward_diff_pct < 1.0, f"奖励差异过大: {reward_diff_pct:.2f}% >= 1%"
    
    def test_reward_calculation_consistency(self):
        """
        测试奖励计算一致性（User Story 3）
        
        验证turbo模式和标准模式在相同状态下计算相同的奖励值。
        """
        import random
        random.seed(42)
        
        config_standard = PetriEnvConfig(n_wafer=24, training_phase=2, turbo_mode=False)
        config_turbo = PetriEnvConfig(
            n_wafer=24, training_phase=2, turbo_mode=True,
            optimize_state_update=True, cache_indices=True, optimize_data_structures=True
        )
        
        env_standard = Petri(config=config_standard)
        env_turbo = Petri(config=config_turbo)
        
        # 同步运行几步
        for _ in range(10):
            random.seed(42 + _)
            enabled_s = env_standard.get_enable_t()
            enabled_t = env_turbo.get_enable_t()
            
            if enabled_s and enabled_t:
                action = random.choice(list(enabled_s))
                env_standard.step(t=action, with_reward=True, detailed_reward=False)
                env_turbo.step(t=action, with_reward=True, detailed_reward=False)
            else:
                env_standard.step(wait=True, with_reward=True, detailed_reward=False)
                env_turbo.step(wait=True, with_reward=True, detailed_reward=False)
        
        # 计算相同时间段的奖励
        t1, t2 = 100, 200
        reward_standard = env_standard.calc_reward(t1, t2, detailed=False)
        reward_turbo = env_turbo.calc_reward(t1, t2, detailed=False)
        
        diff = abs(reward_standard - reward_turbo)
        diff_pct = (diff / abs(reward_standard)) * 100 if reward_standard != 0 else 0
        
        print(f"\n奖励计算一致性测试:")
        print(f"  标准模式奖励: {reward_standard:.2f}")
        print(f"  Turbo模式奖励: {reward_turbo:.2f}")
        print(f"  差异: {diff:.2f} ({diff_pct:.2f}%)")
        
        assert diff_pct < 1.0, f"奖励计算差异过大: {diff_pct:.2f}% >= 1%"
    
    def test_scrap_detection_consistency(self):
        """
        测试报废检测一致性（User Story 3）
        
        验证turbo模式和标准模式在相同条件下检测到相同的报废状态。
        """
        config_standard = PetriEnvConfig(n_wafer=24, training_phase=2, turbo_mode=False)
        config_turbo = PetriEnvConfig(
            n_wafer=24, training_phase=2, turbo_mode=True,
            optimize_state_update=True, cache_indices=True, optimize_data_structures=True
        )
        
        env_standard = Petri(config=config_standard)
        env_turbo = Petri(config=config_turbo)
        
        # 同步运行到可能产生报废的状态
        import random
        random.seed(42)
        for _ in range(500):
            enabled_s = env_standard.get_enable_t()
            enabled_t = env_turbo.get_enable_t()
            
            if enabled_s and enabled_t:
                action = random.choice(list(enabled_s))
                env_standard.step(t=action, with_reward=True, detailed_reward=False)
                env_turbo.step(t=action, with_reward=True, detailed_reward=False)
            else:
                env_standard.step(wait=True, with_reward=True, detailed_reward=False)
                env_turbo.step(wait=True, with_reward=True, detailed_reward=False)
        
        # 检查报废状态
        scrap_standard, _ = env_standard._check_scrap(return_info=True)
        scrap_turbo = env_turbo._check_scrap_turbo()
        
        print(f"\n报废检测一致性测试:")
        print(f"  标准模式检测到报废: {scrap_standard}")
        print(f"  Turbo模式检测到报废: {scrap_turbo}")
        
        # 两种模式应该检测到相同的报废状态
        assert scrap_standard == scrap_turbo, f"报废检测不一致: 标准={scrap_standard}, Turbo={scrap_turbo}"
    
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
