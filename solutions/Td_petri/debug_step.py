import numpy as np
import random
from solutions.Td_petri.tdpn import TimedPetri

def rollout(tp, max_steps=200, seed=0):
    """
    tp: TimedPetri 实例
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=== TimedPetri RESET ===")
    _ , mask=tp.reset()


    step = 0
    done = False
    cur_time = 0

    while step < max_steps and not done:

        se = np.nonzero(mask)[0]
        t = random.choice(se)
        #print(f"step{step}:",tp.id2t_name[t])
        mask, _ , cur_time, done = tp.step(t)

        step += 1

    print("=== END ===")
    print(f"Total steps={step}, finish={done}, time={cur_time}")





if __name__ == "__main__":
    env = TimedPetri()

    for i in range(10):
        print(f"=== iter{i} ===")
        rollout(env, max_steps=500, seed=42)
