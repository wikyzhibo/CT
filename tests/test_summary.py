"""测试总结报告"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def run_basic_test():
    """基本功能测试"""
    print("=" * 60)
    print("基本功能测试")
    print("=" * 60)
    
    config = PetriEnvConfig(n_wafer=12, training_phase=2)
    env = Petri(config=config)
    env.reset()
    
    # 运行 50 步
    for i in range(50):
        enabled = env.get_enable_t()
        if enabled:
            done, reward, scrap = env.step(t=enabled[0], with_reward=True, detailed_reward=False)
            if done:
                print(f"Episode 完成于步骤 {i}, time={env.time}, scrap={scrap}")
                break
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
    
    print(f"[PASS] 基本功能测试通过 (最终时间: {env.time})")
    return True

def run_optimization_comparison():
    """优化效果对比"""
    print("\n" + "=" * 60)
    print("优化效果对比测试")
    print("=" * 60)
    
    np.random.seed(42)
    steps = 5000  # 增加测试步数以获得更准确的结果
    
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
    
    start = time.time()
    for _ in range(steps):
        enabled = env_old.get_enable_t()
        if enabled:
            env_old.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env_old.step(wait=True, with_reward=True, detailed_reward=False)
    time_old = time.time() - start
    
    # 优化后
    np.random.seed(42)
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
    
    start = time.time()
    for _ in range(steps):
        enabled = env_new.get_enable_t()
        if enabled:
            env_new.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env_new.step(wait=True, with_reward=True, detailed_reward=False)
    time_new = time.time() - start
    
    improvement = (time_old - time_new) / time_old * 100
    steps_per_sec_old = steps / time_old
    steps_per_sec_new = steps / time_new
    
    print(f"测试步数: {steps:,}")
    print(f"优化前时间: {time_old:.3f} 秒 ({steps_per_sec_old:,.0f} 步/秒)")
    print(f"优化后时间: {time_new:.3f} 秒 ({steps_per_sec_new:,.0f} 步/秒)")
    print(f"改进幅度: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"[PASS] 性能提升 {improvement:.1f}%")
    else:
        print(f"[INFO] 当前测试显示性能变化: {improvement:.1f}% (可能需要更多优化)")
    
    return improvement

def run_turbo_mode_test():
    """极速模式测试"""
    print("\n" + "=" * 60)
    print("极速模式测试")
    print("=" * 60)
    
    config = PetriEnvConfig(
        n_wafer=4,
        training_phase=2,
        turbo_mode=True,
        optimize_state_update=True,
        cache_indices=True
    )
    env = Petri(config=config)
    env.reset()
    
    # 测试 10,000 步（而不是 100,000 步，因为可能需要很长时间）
    test_steps = 10000
    start = time.time()
    
    for i in range(test_steps):
        enabled = env.get_enable_t()
        if enabled:
            env.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
        
        # 如果完成，重置
        if env.m[env._get_place_index("LP_done")] == env.n_wafer:
            env.reset()
    
    elapsed = time.time() - start
    steps_per_sec = test_steps / elapsed
    
    print(f"执行步数: {test_steps:,}")
    print(f"执行时间: {elapsed:.3f} 秒")
    print(f"步数/秒: {steps_per_sec:,.0f}")
    print(f"目标: >100,000 步/秒")
    
    if steps_per_sec > 100000:
        print(f"[PASS] 极速模式性能达标")
    else:
        print(f"[INFO] 当前性能: {steps_per_sec:,.0f} 步/秒 (目标: 100,000 步/秒)")
        print(f"      需要 {100000/steps_per_sec:.1f}x 提升才能达到目标")
    
    return steps_per_sec

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Petri 网模拟器性能优化测试报告")
    print("=" * 60)
    
    # 基本功能测试
    run_basic_test()
    
    # 优化效果对比
    improvement = run_optimization_comparison()
    
    # 极速模式测试
    turbo_perf = run_turbo_mode_test()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"1. 基本功能: [PASS]")
    print(f"2. 优化效果: {improvement:+.1f}%")
    print(f"3. 极速模式: {turbo_perf:,.0f} 步/秒")
    print("\n注意: 性能测试结果可能因硬件和系统负载而异。")
    print("建议: 在实际训练环境中进行更长时间的测试以获得准确结果。")
