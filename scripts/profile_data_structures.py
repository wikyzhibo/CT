"""性能分析脚本：验证数据结构优化的效果"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import io
import time
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def profile_data_structure_optimization():
    """运行性能分析，对比优化前后的效果"""
    print("=" * 70)
    print("数据结构优化性能分析")
    print("=" * 70)
    
    # 配置参数
    n_wafer = 24
    target_steps = 10000
    
    # ========== 测试1: 仅启用 002-sim-speedup ==========
    print(f"\n[测试1] 仅启用 002-sim-speedup (baseline)")
    config_baseline = PetriEnvConfig(
        n_wafer=n_wafer,
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
            env_baseline.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env_baseline.step(wait=True, with_reward=True, detailed_reward=False)
        step_count += 1
        if env_baseline.m[env_baseline.id2p_name.index("LP_done")] == env_baseline.n_wafer:
            env_baseline.reset()
    elapsed_baseline = time.time() - start
    steps_per_sec_baseline = target_steps / elapsed_baseline
    
    print(f"  执行步数: {target_steps:,}")
    print(f"  执行时间: {elapsed_baseline:.3f} 秒")
    print(f"  步数/秒: {steps_per_sec_baseline:,.0f}")
    
    # ========== 测试2: 启用 002-sim-speedup + 数据结构优化 ==========
    print(f"\n[测试2] 启用 002-sim-speedup + 数据结构优化")
    config_optimized = PetriEnvConfig(
        n_wafer=n_wafer,
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
            env_optimized.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env_optimized.step(wait=True, with_reward=True, detailed_reward=False)
        step_count += 1
        if env_optimized.m[env_optimized.id2p_name.index("LP_done")] == env_optimized.n_wafer:
            env_optimized.reset()
    elapsed_optimized = time.time() - start
    steps_per_sec_optimized = target_steps / elapsed_optimized
    
    print(f"  执行步数: {target_steps:,}")
    print(f"  执行时间: {elapsed_optimized:.3f} 秒")
    print(f"  步数/秒: {steps_per_sec_optimized:,.0f}")
    
    # ========== 性能对比 ==========
    print(f"\n[性能对比]")
    improvement = (elapsed_baseline - elapsed_optimized) / elapsed_baseline * 100
    speedup = elapsed_baseline / elapsed_optimized
    print(f"  时间改进: {improvement:+.1f}%")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  目标: >= 5% 改进")
    print(f"  状态: {'[PASS]' if improvement >= 5 else '[FAIL]'}")
    
    # ========== 详细性能分析（使用 cProfile）==========
    print(f"\n[详细性能分析] 运行 cProfile...")
    config_prof = PetriEnvConfig(
        n_wafer=n_wafer,
        training_phase=2,
        turbo_mode=True,
        optimize_data_structures=True,
    )
    env_prof = Petri(config=config_prof)
    env_prof.reset()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行少量步数进行详细分析
    for i in range(1000):
        enabled = env_prof.get_enable_t()
        if enabled:
            env_prof.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env_prof.step(wait=True, with_reward=True, detailed_reward=False)
        if env_prof.m[env_prof.id2p_name.index("LP_done")] == env_prof.n_wafer:
            env_prof.reset()
    
    profiler.disable()
    
    # 分析结果
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 显示前20个最耗时的函数
    
    result = s.getvalue()
    print(result)
    
    # 保存到文件
    with open('profile_data_structures_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("数据结构优化性能分析结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Baseline (仅 002-sim-speedup):\n")
        f.write(f"  执行时间: {elapsed_baseline:.3f} 秒\n")
        f.write(f"  步数/秒: {steps_per_sec_baseline:,.0f}\n\n")
        f.write(f"Optimized (002-sim-speedup + 数据结构优化):\n")
        f.write(f"  执行时间: {elapsed_optimized:.3f} 秒\n")
        f.write(f"  步数/秒: {steps_per_sec_optimized:,.0f}\n\n")
        f.write(f"性能改进: {improvement:+.1f}%\n")
        f.write(f"加速比: {speedup:.2f}x\n\n")
        f.write("=" * 70 + "\n")
        f.write("详细性能分析 (cProfile):\n")
        f.write("=" * 70 + "\n\n")
        f.write(result)
    
    print("\n性能分析结果已保存到 profile_data_structures_results.txt")

if __name__ == "__main__":
    profile_data_structure_optimization()
