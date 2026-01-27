"""性能分析脚本：识别极速模式的性能瓶颈"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import io
import time
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def profile_turbo_mode():
    """运行极速模式性能分析"""
    print("=" * 60)
    print("极速模式性能分析")
    print("=" * 60)
    
    # 创建极速模式配置
    config = PetriEnvConfig(
        n_wafer=24,
        training_phase=2,
        turbo_mode=True,
        optimize_state_update=True,
        cache_indices=True,
        optimize_data_structures=True
    )
    
    # 创建环境
    env = Petri(config=config)
    env.reset()
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行模拟（执行 10,000 步进行详细分析）
    step_count = 0
    target_steps = 10000
    
    start_time = time.time()
    while step_count < target_steps:
        enabled = env.get_enable_t()
        if enabled:
            env.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
        
        step_count += 1
        
        # 如果完成，重置环境继续测试
        if env.m[env._get_place_index("LP_done")] == env.n_wafer:
            env.reset()
    
    elapsed = time.time() - start_time
    profiler.disable()
    
    # 分析结果
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    
    print(f"\n执行统计:")
    print(f"  执行步数: {step_count:,}")
    print(f"  执行时间: {elapsed:.3f} 秒")
    print(f"  步数/秒: {step_count / elapsed:,.0f}")
    
    # 按累计时间排序
    stats.sort_stats('cumulative')
    print(f"\n按累计时间排序（前30个函数）：")
    stats.print_stats(30)
    
    result = s.getvalue()
    
    # 保存到文件
    output_file = 'profile_turbo_mode_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"极速模式性能分析结果\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"执行步数: {step_count:,}\n")
        f.write(f"执行时间: {elapsed:.3f} 秒\n")
        f.write(f"步数/秒: {step_count / elapsed:,.0f}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(result)
    
    print(f"\n性能分析结果已保存到 {output_file}")
    
    # 按总时间排序
    stats.sort_stats('tottime')
    print(f"\n按总时间排序（前20个函数）：")
    stats.print_stats(20)
    
    # 识别热点函数
    print(f"\n{'=' * 60}")
    print("性能瓶颈识别:")
    print(f"{'=' * 60}")
    
    # 获取最耗时的函数
    stats.sort_stats('tottime')
    top_functions = []
    for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
        if tt > 0.01:  # 只关注耗时超过10ms的函数
            top_functions.append((func_name, tt, ct))
    
    top_functions.sort(key=lambda x: x[1], reverse=True)
    
    print("\n最耗时的函数（总时间 > 10ms）:")
    for i, (func_name, tt, ct) in enumerate(top_functions[:10], 1):
        print(f"  {i}. {func_name[2]}: {tt:.3f}s (累计: {ct:.3f}s)")

if __name__ == "__main__":
    profile_turbo_mode()
