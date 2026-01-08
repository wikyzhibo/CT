import os
import torch
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from torchrl.envs.utils import set_exploration_type, ExplorationType
from solutions.PPO.enviroment import CT2, CT_v2
from solutions.PPO.network.models import MaskedPolicyHead
from visualization.plot import plot_gantt_hatched_residence,Op

def load_policy(model_path, env, device="cpu"):
    state_dict = torch.load(model_path, map_location=device)

    n_actions = env.action_spec.space.n
    n_m = env.observation_spec["observation"].shape[0]

    policy_backbone = MaskedPolicyHead(hidden=256, n_obs=n_m, n_actions=n_actions)
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
    return policy





def ev(env, policy=None, max_steps=1500):
    """
    评估策略在环境中的表现
    
    Args:
        env: 环境对象
        policy: 策略模型
        max_steps: 最大步数
    
    Returns:
        makespan: 完成时间（如果任务完成），否则返回 None
    """
    if policy is None:
        raise ValueError("policy 不能为 None")
    
    policy.eval()
    with torch.no_grad():
        with set_exploration_type(ExplorationType.RANDOM):
            out = env.rollout(max_steps, policy)
    
    # 检查是否完成任务
    if out["next", "finish"].sum().item() <= 0:
        print("警告: 任务未完成")
        env.reset()
        return None
    
    # 计算 makespan
    makespan = torch.where(
        out["next", "finish"],
        out["next", "time"],
        100000
    ).min().item()

    print(f"评估结果 - Makespan: {makespan:.2f}", end="")
    print(f" | 步数: {len(out)}")
    
    return makespan


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证训练好的PPO模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/CT_latest.pt",
        help="模型文件路径（默认: saved_models/CT_latest.pt）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="N7",
        choices=["N2", "N5", "N6", "N7", "N8"],
        help="使用的配置文件（默认: N8）"
    )
    parser.add_argument(
        "--n_wafer",
        type=int,
        default=75,
        help="晶圆数量（默认: 75）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备（默认: cpu）"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1500,
        help="最大评估步数（默认: 1500）"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=50,
        help="运行次数（默认: 5）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 导入配置
    config_module = f"config.params_{args.config}"
    if args.config == "N8":
        from data.config.params_N8 import params_N8 as params
    elif args.config == "N7":
        from data.config.params_N7 import params_N7 as params
    elif args.config == "N6":
        from data.config.params_N6 import params_N6 as params
    elif args.config == "N5":
        from data.config.params_N5 import params_N5 as params
    elif args.config == "N2":
        from data.config.params_N2 import params_N2 as params
    else:
        raise ValueError(f"不支持的配置: {args.config}")
    
    # 设置晶圆数量
    #params['n_wafer'] = args.n_wafer
    
    # 创建环境
    print(f"创建环境 - 配置: {args.config}, 晶圆数: {args.n_wafer}")
    base_env = CT_v2(device=device)
    
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
    
    # 加载模型
    model_path = args.model_path
    if not os.path.isabs(model_path):
        # 相对路径，从当前文件所在目录计算
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
    
    print(f"加载模型: {model_path}")
    policy = load_policy(model_path, env, device=device)
    
    # 运行评估
    print(f"\n开始评估 (运行 {args.n_runs} 次)...")
    makespans = []

    min_makespan = 10**5
    for run in range(args.n_runs):
        print(f"\n--- 运行 {run + 1}/{args.n_runs} ---")
        env.reset()
        makespan = ev(env, policy, max_steps=args.max_steps)
        if makespan < min_makespan:
            min_makespan = makespan
            ops2 = []
            net = env.net
            tmp_p = ['PM7', 'LLC', 'PM1', 'LLD', 'PM9']
            tmpid = [net.id2p_name.index(i) for i in tmp_p]
            for i, id in enumerate(tmpid):
                for j in range(net.stage_c[i + 1]):
                    machine = j
                    p_occ = net.place_times[id][machine]
                    for oc in p_occ:
                        job = oc.tok_key
                        start = oc.start
                        end = oc.end
                        proc = net.proc[i + 1]
                        ops2.append(
                            Op(job=job, stage=i + 1, machine=machine, start=start, proc_end=start + proc, end=end))
        if makespan is not None:
            makespans.append(makespan)

    # 绘制甘特图

    net = env.net
    out_path = r"C:\Users\khand\OneDrive\code\dqn\CT\results\\"


    arm_info = {'ARM1': ["t3", "u3", "t4", "u6", "t7", "u31", 'u7', 't8'],
                'ARM2': ["u4", "t5", "u5", "t6"],
                'STAGE2ACT': {1: ("t3", "u3"), 2: ("t4", "u4"), 3: ("t5", "u5"), 4: ("t6", "u6"), 5: ('t7', 'u7')}}
    n_job = net.n_wafer
    plot_gantt_hatched_residence(ops=ops2, proc_time=net.proc,
                                 capacity=net.stage_c, n_jobs=n_job,
                                 out_path=out_path, with_label=True,
                                 arm_info=arm_info,policy=2,no_arm=True)

    # 统计结果
    if makespans:
        print(f"\n{'='*50}")
        print(f"评估完成!")
        print(f"成功运行次数: {len(makespans)}/{args.n_runs}")
        print(f"平均 Makespan: {sum(makespans)/len(makespans):.2f}")
        if len(makespans) > 1:
            print(f"最佳 Makespan: {min(makespans):.2f}")
            print(f"最差 Makespan: {max(makespans):.2f}")
        print(f"{'='*50}")
    else:
        print("\n警告: 所有运行都未完成任务")