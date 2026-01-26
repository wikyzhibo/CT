import matplotlib.pyplot as plt
import numpy as np
import torch
from enviroment import CT2,CT_v2
from train import train
from torchrl.envs import (Compose, DTypeCastTransform,TransformedEnv, ActionMask)
import time

from data.config.params_N8 import params_N8

if __name__ == "__main__":

<<<<<<< Updated upstream
=======
    train_mode = "auto-two"  # 固定为两阶段训练


>>>>>>> Stashed changes
    # 环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env1 = CT_v2(device=device)
    base_env2 = CT_v2(device=device)

    transform = Compose([ActionMask(),
                         DTypeCastTransform(dtype_in=torch.int64,dtype_out=torch.float32,
                                            in_keys="observation",out_keys="observation_f"),])
    train_env = TransformedEnv(base_env1,transform)
    eval_env = TransformedEnv(base_env2,transform)

    #check_env_specs(env)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

<<<<<<< Updated upstream
    start_time = time.time()
    log,policy = train(
        train_env,
        eval_env,
        device="cpu",
        with_pretrain=False
    )
    print("training time:", time.time() - start_time)
=======
    # 自动两阶段训练模式
    if train_mode == "auto-two":
        print("=" * 60)
        print("自动两阶段课程学习训练")
        print("=" * 60)

        # Phase 1: 仅报废惩罚
        print("\n[Phase 1] 仅考虑报废惩罚...")
        train_env, eval_env = create_env(device, training_phase=1)

        # 创建或加载 Phase 1 配置
        config1 = PPOTrainingConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\ppo_configs\phase1_config.json")


        start_time = time.time()
        log1, policy1 = train(train_env, eval_env, config=config1)
        print(f"Phase 1 training time: {time.time() - start_time:.2f}s")

        # 获取 Phase 1 最佳模型路径
        saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
        phase1_checkpoint = os.path.join(saved_models_dir, "CT_phase1_best.pt")

        # Phase 2: 完整奖励（加载 Phase 1 checkpoint）
        print("\n[Phase 2] 完整奖励（加载 Phase 1 模型）...")
        train_env, eval_env = create_env(device, training_phase=2)

        # 创建 Phase 2 配置
        config2 = PPOTrainingConfig.load(r"C:\Users\khand\OneDrive\code\dqn\CT\data\ppo_configs\phase2_config.json")

        start_time = time.time()
        log2, policy2 = train(
            train_env, eval_env,
            config=config2,
            checkpoint_path=phase1_checkpoint
        )
        print(f"Phase 2 training time: {time.time() - start_time:.2f}s")

        log = {"phase1": log1, "phase2": log2}
        policy = policy2
>>>>>>> Stashed changes

    y1 = np.asarray(log["phase2"]["reward"], dtype=float)
    x = np.arange(1, len(y1)+1)  # 更稳：按实际长度画

    #out = eval_env.rollout(max_steps=1300,policy=policy)
    #time = out['next','time'][-1].item()
    #n_wafer = out['next','reward'].sum().item()/200
    #print('step',len(out))
    #print("makespan",time)
    #print("n_wafer:",n_wafer)
    #env.reset()
    #log,_ = train(env,eval_env,device="cpu",with_pretrain=False)

    #y2 = np.asarray(log["makespan"], dtype=float)


    plt.plot(x, y1, marker="o")
    #plt.plot(x, y2, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("reward")
    plt.title("PPO")

    #best_makespan = 12000  # <- 这里填你想对比的“最优/基准”值
    #plt.axhline(best_makespan, linestyle="--", linewidth=1)
    #plt.text(x[-1], best_makespan, f"  best={best_makespan:.0f}", va="center")

    #plt.legend(["bc","without bc", "GREEDY"])
    #plt.savefig(r"C:\User\khand\OneDrive\code\dqn\CT\results\PPO without BC.png")
    plt.show()
