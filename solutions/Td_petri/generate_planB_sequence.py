
import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torchrl.envs import Compose, DTypeCastTransform, ActionMask
from torchrl.envs.utils import set_exploration_type, ExplorationType

from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.rl import load_policy
from solutions.Td_petri.utils import TDPNParser, res_occ_to_event
from solutions.Td_petri.vis_utils import render_gantt_from_petri


def execute_custom_sequence(sequence: List[int], env):
    """
    执行自定义的动作序列
    
    Args:
        sequence: 动作ID序列列表，每个元素是一个动作ID（链条ID）
        env: TimedPetri 环境实例
    
    Returns:
        执行完序列后的环境实例
    """
    print(f"开始执行自定义动作序列，共 {len(sequence)} 步")
    
    # 重置环境
    td = env.reset()
    
    for step_idx, action_id in enumerate(sequence):
        # 检查动作ID是否有效
        if action_id < 0 or action_id >= env.A:
            print(f"警告: 步骤 {step_idx + 1} 的动作ID {action_id} 超出范围 [0, {env.A-1}]")
            continue
        
        # 获取对应的链条信息
        chain = env.aid2chain[action_id]
        chain_str = " -> ".join(chain)
        
        # 检查动作是否可用
        action_mask = td["action_mask"].numpy()
        if not action_mask[action_id]:
            print(f"警告: 步骤 {step_idx + 1} 的动作 {action_id} ({chain_str}) 当前不可用")
            print(f"当前可用动作: {np.where(action_mask)[0].tolist()}")
            
            # 如果动作不可用，打印当前状态并尝试继续（虽然可能会失败）
            continue
        
        # 构建执行动作所需的 TensorDict
        # 注意：TorchRL 的 step 需要输入包含 action 的 TensorDict
        td.set("action", torch.tensor([action_id], dtype=torch.int64))
        
        # 执行动作
        td = env.step(td)
        
        print(f"步骤 {step_idx + 1}/{len(sequence)}: 动作 {action_id} ({chain_str}) - 完成")
        
        # 检查是否完成
        # TorchRL 的 done 标志通常在 "done", "terminated", 或 "truncated" 中
        done = td.get("done", torch.tensor([False])).item()
        terminated = td.get("terminated", torch.tensor([False])).item()

        td = td['next']

        if done or terminated:
            print(f"环境在第 {step_idx + 1} 步完成")
            break
            
    return env


def generate_with_policy(env, policy, max_steps: int = 2000):
    """
    使用训练好的策略生成动作序列
    
    Args:
        env: TransformedEnv 环境
        policy: 训练好的策略网络
        max_steps: 最大步数
    
    Returns:
        执行完成后的 petri_net 实例
    """
    policy.eval()
    with torch.no_grad():
        with set_exploration_type(ExplorationType.MODE):
            _ = env.rollout(max_steps, policy)
    
    return env.base_env


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成或执行 Plan B 动作序列")
    parser.add_argument(
        "--custom-sequence",
        type=str,
        default=None,
        help="自定义动作序列的 JSON 文件路径（如果提供，将执行该序列而不是使用策略生成）"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="planB",
        help="输出文件的名称前缀（默认: planB）"
    )
    args = parser.parse_args()
    
    device = "cpu"
    
    # 初始化环境
    base_env = TimedPetri()
    
    # 判断执行模式
    if args.custom_sequence:
        # 模式1: 执行自定义序列
        print(f"模式: 执行自定义动作序列")
        print(f"序列文件: {args.custom_sequence}")
        
        # 加载自定义序列
        ROOT_DIR = Path(__file__).resolve().parents[2]
        series_path = os.path.join(ROOT_DIR, "data","action_series", args.custom_sequence)

        with open(series_path, "r") as f:
            custom_sequence = json.load(f)
        
        # 执行自定义序列
        petri_net = execute_custom_sequence(custom_sequence, base_env)
        
    else:
        # 模式2: 使用策略生成序列
        print(f"模式: 使用训练好的策略生成序列")
        
        # 获取模型路径
        ROOT_DIR = Path(__file__).resolve().parents[1]
        model_path = os.path.join(ROOT_DIR, "PPO", "syc_model", "taskD", "best_8_8.pt")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        # 创建 TransformedEnv
        transform = Compose([
            ActionMask(),
            DTypeCastTransform(
                dtype_in=torch.int64,
                dtype_out=torch.float32,
                in_keys="observation",
                out_keys="observation_f"
            ),
        ])
        from torchrl.envs import TransformedEnv
        env = TransformedEnv(base_env, transform)
        
        # 加载策略
        policy = load_policy(model_path, env, device)
        
        # 运行 Rollout
        petri_net = generate_with_policy(env, policy)

    # 生成事件流
    events = res_occ_to_event(petri_net.res_occ)

    # 解析事件流为动作序列
    parser_obj = TDPNParser()
    sequence = parser_obj.parse(events)
    
    # 保存结果
    output_dir = Path(__file__).parent
    output_file = output_dir / f"{args.output_name}_sequence.json"

    with open(output_file, "w") as f:
        json.dump(sequence, f, indent=2)
        
    print(f"Generated {output_file} with {len(sequence)} steps.")

    # 生成甘特图
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    out_file = str(results_dir / f"{args.output_name}_")
    # Instead of env.net.render_gantt, use our modular renderer
    render_gantt_from_petri(petri_net, out_path=out_file, policy=3)
    print(f"Gantt chart generated at {out_file}")

if __name__ == "__main__":
    main()
