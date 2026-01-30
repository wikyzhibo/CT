"""
调试脚本：重现 s4 释放时间计算错误的场景
"""
import sys
sys.path.insert(0, r"c:\Users\khand\OneDrive\code\dqn\CT")

from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

def main():
    config = PetriEnvConfig(n_wafer=8, training_phase=2)
    env = Petri(config=config)
    env.reset()
    
    print("=" * 80)
    print("开始运行，观察 t_s3 时 s4 释放时间的更新")
    print("=" * 80)
    
    # 运行直到 t_s3 发生
    step = 0
    max_steps = 50
    
    while step < max_steps:
        enabled = env.get_enable_t()
        if not enabled:
            env.step(wait=True, with_reward=True)
            step += 1
            continue
        
        # 选择第一个可用动作
        t = enabled[0]
        t_name = env.id2t_name[t]
        
        # 在 t_s3 之前和之后打印 s3, s4 的 release_schedule
        if t_name == "t_s3":
            print(f"\n>>> 步骤 {step}, TIME={env.time}: 即将执行 {t_name}")
            s3_idx = env._get_place_index("s3")
            s4_idx = env._get_place_index("s4")
            print(f"    执行前 s3.release_schedule = {list(env.marks[s3_idx].release_schedule)}")
            print(f"    执行前 s4.release_schedule = {list(env.marks[s4_idx].release_schedule)}")
            
            env.step(t=t, with_reward=True)
            
            print(f"    执行后 s3.release_schedule = {list(env.marks[s3_idx].release_schedule)}")
            print(f"    执行后 s4.release_schedule = {list(env.marks[s4_idx].release_schedule)}")
            print()
            
            # 执行完 t_s3 后退出
            break
        else:
            # 静默执行其他动作
            env.step(t=t, with_reward=True)
        
        step += 1
    
    print("=" * 80)
    print("调试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
