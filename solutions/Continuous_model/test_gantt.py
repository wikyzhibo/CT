"""
甘特图绘制测试脚本

测试 render_gantt 方法，使用训练好的模型运行一个完整的 episode 并生成甘特图。

用法:
    python test_gantt.py --model_path CT_phase2_best.pt
    python test_gantt.py --model_path solutions/PPO/saved_models/CT_phase2_best.pt
"""

import os
import sys
import argparse
import torch
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from torchrl.envs.utils import set_exploration_type, ExplorationType
from solutions.PPO.enviroment import Env_PN
from solutions.PPO.network.models import MaskedPolicyHead


def load_policy(model_path, env, device="cpu"):
    """加载训练好的策略模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    
    n_actions = env.action_spec.space.n
    n_obs = env.observation_spec["observation"].shape[0]
    
    policy_backbone = MaskedPolicyHead(hidden=256, n_obs=n_obs, n_actions=n_actions, n_layers=4)
    td_module = TensorDictModule(
        policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=td_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    ).to(device)
    
    policy.load_state_dict(state_dict)
    policy.eval()
    print("模型加载成功")
    return policy


def run_episode(env, policy, max_steps=5000):
    """
    运行一个完整的 episode
    
    Returns:
        (done, makespan, step_count): 是否完成、完成时间、步数
    """
    print(f"\n开始运行 episode (最大步数: {max_steps})...")
    
    policy.eval()
    with torch.no_grad():
        with set_exploration_type(ExplorationType.RANDOM):
            out = env.rollout(max_steps, policy)
    
    # 检查是否完成任务
    finish_mask = out["next", "finish"]
    if finish_mask.sum().item() > 0:
        # 找到第一个完成的时间
        finish_indices = torch.where(finish_mask)[0]
        makespan = out["next", "time"][finish_indices[0]].item()
        step_count = finish_indices[0].item() + 1
        print(f"✓ Episode 完成!")
        print(f"  Makespan: {makespan:.2f}s")
        print(f"  步数: {step_count}")
        return True, makespan, step_count
    else:
        makespan = out["next", "time"][-1].item()
        step_count = len(out)
        print(f"✗ Episode 未完成 (达到最大步数)")
        print(f"  当前时间: {makespan:.2f}s")
        print(f"  步数: {step_count}")
        return False, makespan, step_count


def test_gantt(model_path, output_dir="./results", max_steps=2000, device="cpu"):
    """
    测试甘特图绘制
    
    Args:
        model_path: 模型文件路径
        output_dir: 输出目录
        max_steps: 最大步数
        device: 计算设备
    """
    print("=" * 70)
    print("甘特图绘制测试")
    print("=" * 70)
    
    # 1. 创建环境
    print("\n[1/4] 创建环境...")
    base_env = Env_PN(device=device)
    
    transform = Compose([
        ActionMask(),
        DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.float32,
            in_keys="observation",
            out_keys="observation_f"
        ),
    ])
    env = TransformedEnv(base_env, transform)
    print(f"  环境创建成功")
    print(f"  动作空间大小: {env.action_spec.space.n}")
    print(f"  观测维度: {env.observation_spec['observation'].shape[0]}")
    
    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    policy = load_policy(model_path, env, device=device)
    
    # 3. 运行 episode
    print("\n[3/4] 运行 episode...")
    done, makespan, step_count = run_episode(env, policy, max_steps=max_steps)
    
    # 4. 生成甘特图
    print("\n[4/4] 生成甘特图...")
    petri_net = base_env.net  # 获取 Petri 网对象
    
    # 检查 fire_log 是否为空
    if not petri_net.fire_log:
        print("✗ 错误: fire_log 为空，无法生成甘特图")
        return False
    
    print(f"  fire_log 条目数: {len(petri_net.fire_log)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"gantt_test_{timestamp}.png")
    
    try:
        # 调用 render_gantt
        petri_net.render_gantt(out_path=output_path)
        
        # 验证文件是否存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ 甘特图生成成功!")
            print(f"  文件路径: {output_path}")
            print(f"  文件大小: {file_size / 1024:.2f} KB")
            return True
        else:
            print(f"✗ 错误: 文件未生成")
            return False
            
    except Exception as e:
        print(f"✗ 生成甘特图时发生错误:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="测试甘特图绘制功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认路径查找模型
  python test_gantt.py
  
  # 指定模型路径
  python test_gantt.py --model_path CT_phase2_best.pt
  
  # 指定完整路径
  python test_gantt.py --model_path solutions/PPO/saved_models/CT_phase2_best.pt
  
  # 指定输出目录
  python test_gantt.py --model_path CT_phase2_best.pt --output_dir ./test_results
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="CT_phase2_best.pt",
        help="模型文件路径（默认: CT_phase2_best.pt）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录（默认: ./results）"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="最大步数（默认: 2000）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备（默认: cpu）"
    )
    
    args = parser.parse_args()
    
    # 处理模型路径
    model_path = args.model_path
    if not os.path.isabs(model_path):
        # 尝试多个可能的路径
        possible_paths = [
            model_path,  # 当前目录
            os.path.join("..", "PPO", "saved_models", model_path),
            os.path.join("..", "PPO", model_path),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            print(f"错误: 找不到模型文件")
            print(f"尝试过的路径:")
            for path in possible_paths:
                print(f"  - {path}")
            return 1
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 运行测试
    success = test_gantt(
        model_path=model_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=device
    )
    
    print("\n" + "=" * 70)
    if success:
        print("测试完成: 成功")
        return 0
    else:
        print("测试完成: 失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
