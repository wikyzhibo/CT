"""性能分析脚本：识别 Petri 网模拟器的性能瓶颈"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import io
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def profile_petri_simulation():
    """运行性能分析"""
    # 创建配置
    config = PetriEnvConfig(n_wafer=4, training_phase=2)
    
    # 创建环境
    env = Petri(config=config)
    env.reset()
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行模拟（执行 1000 步）
    for i in range(1000):
        enabled = env.get_enable_t()
        if enabled:
            env.step(t=enabled[0], with_reward=True, detailed_reward=False)
        else:
            env.step(wait=True, with_reward=True, detailed_reward=False)
    
    profiler.disable()
    
    # 分析结果
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # 显示前30个最耗时的函数
    
    result = s.getvalue()
    print(result)
    
    # 保存到文件
    with open('profile_results.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    print("\n性能分析结果已保存到 profile_results.txt")
    
    # 按总时间排序
    stats.sort_stats('tottime')
    print("\n按总时间排序（前20个函数）：")
    stats.print_stats(20)

if __name__ == "__main__":
    profile_petri_simulation()
