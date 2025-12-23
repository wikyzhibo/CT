import matplotlib.pyplot as plt
import numpy as np

import torch

from enviroment import CT
from Train_CT import train,ev
import time
from torchrl.envs import (Compose, StepCounter, DTypeCastTransform,
                          TransformedEnv,ActionMask,RewardSum)


if __name__ == "__main__":

    # 任务b
    from config.params_N8 import params_N8

    # 环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env1 = CT(device=device,allow_idle=False,**params_N8)
    params_N8['n_wafer'] = 75
    base_env2 = CT(device=device,allow_idle=False, **params_N8)

    transform = Compose([ActionMask(),
                         DTypeCastTransform(dtype_in=torch.int64,dtype_out=torch.float32,
                                            in_keys="observation",out_keys="observation_f"),])
    env = TransformedEnv(base_env1,transform)

    eval_env = TransformedEnv(base_env2,transform)

    #check_env_specs(env)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    start_time = time.time()
    log,policy = train(
        env,
        eval_env,
        device="cpu",
        with_pretrain=True
    )
    print("training time:", time.time() - start_time)

    y1 = np.asarray(log["reward"], dtype=float)
    x = np.arange(1, len(y1)+1)  # 更稳：按实际长度画

    #env.reset()
    #log,_ = train(env,eval_env,device="cpu",with_pretrain=False)

    #y2 = np.asarray(log["makespan"], dtype=float)


    plt.plot(x, y1, marker="o")
    #plt.plot(x, y2, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Makespan")
    plt.title("PPO")

    #best_makespan = 12000  # <- 这里填你想对比的“最优/基准”值
    #plt.axhline(best_makespan, linestyle="--", linewidth=1)
    #plt.text(x[-1], best_makespan, f"  best={best_makespan:.0f}", va="center")

    #plt.legend(["bc","without bc", "GREEDY"])
    #plt.savefig(r"C:\User\khand\OneDrive\code\dqn\CT\res\PPO without BC.png")
    plt.show()
