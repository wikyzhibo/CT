import time
import sys

from model.super_net import ModuleSpec, RobotSpec, SharedGroup, SuperPetriBuilder
from PDR.net import Petri
from utils.draw_gantt import build_pm_intervals, plot_gantt

if __name__ == "__main__":
    # 模块库所：ptime 你手动填；capacity 你手动填（PM=1，LL可=2，LP按需）
    modules = {
        "LP1": ModuleSpec(tokens=75, ptime=0, capacity=100),
        "AL":  ModuleSpec(tokens=0,  ptime=8, capacity=1),

        # 缓冲区1：LLA(in) / LLB(out) 缓冲区2：LLC(in) / LLD(out)
        "LLA": ModuleSpec(tokens=0, ptime=20, capacity=2),
        "LLB": ModuleSpec(tokens=0, ptime=20, capacity=2),
        "LLC": ModuleSpec(tokens=0, ptime=0, capacity=2),
        "LLD": ModuleSpec(tokens=0, ptime=70, capacity=2),

        "PM7": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "PM8": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "PM1": ModuleSpec(tokens=0, ptime=300, capacity=1),
        "PM2": ModuleSpec(tokens=0, ptime=300, capacity=1),

        "LP_done": ModuleSpec(tokens=0, ptime=0, capacity=100),
    }

    # 机械手资源（r__TMx）：TM1=1，TM2/TM3=2
    robots = {
        "TM1": RobotSpec(tokens=1, reach={"LP1", "AL", "LLA", "LLB","LP_done"}),
        "TM2": RobotSpec(tokens=2, reach={"LLA", "LLB", "LLC", "LLD", "PM7", "PM8"}),
        "TM3": RobotSpec(tokens=2, reach={"LLC", "LLD", "PM1", "PM2"}),
    }

    # 两个共享缓冲区组
    shared_groups = [
        SharedGroup(name="LLAB", places={"LLA", "LLB"}, cap=2),
        SharedGroup(name="LLCD", places={"LLC", "LLD"}, cap=2),
    ]

    # 多路径（含分叉）：示例：LP1->AL->(LLB)->(PM7/PM8)->LLD->(PM1/PM2)->LLA->LP_done
    # 这里用 LLB(进来) 和 LLA(出去)，LLD(进来) 和 LLC(出去) 由你路线来表达
    routes = [
        ["LP1", "AL", "LLA", ["PM7", "PM8"], "LLC", ["PM1", "PM2"], "LLD","LLB", "LP_done"]
    ]

    builder = SuperPetriBuilder(d_ptime=3, default_ttime=2)
    info = builder.build(modules, robots, routes, shared_groups=shared_groups)

    print("P,T =", info["pre"].shape[0], info["pre"].shape[1])
    print("edges =", len(info["edges"]))
    print("ptime len =", len(info["ptime"]), "ttime len =", len(info["ttime"]))
    print("capacity len =", len(info["capacity"]))

    param = {'super_info': info,
             "capacity_xianzhi": {'w1': 'u_AL_LLA'},
             "controller": {
                 'bm1': {'p': ['d_LLA','LLA','d_LLB','LLB'],
                         't': ['u_AL_LLA', 't_LLA', 'u_LLD_LLB', 't_LLB']},
                 'bm2': {'p': ['d_LLC','LLC', 'd_LLD','LLD'],
                         't': [['u_PM7_LLC','u_PM8_LLC'], 't_LLC', ['u_PM1_LLD','u_PM2_LLD'], 't_LLD']},
                 'f': [('AL', 1, 'u_LP1_AL'),('PM7', 1, 'u_LLA_PM7'),
                       ('PM8', 1, 'u_LLA_PM8'),('PM1', 1, 'u_LLC_PM1'),
                       ('PM2', 1, 'u_LLC_PM2')]
                            }
             }
    net = Petri(use_super_net=True,
                with_controller=True,
                with_capacity_controller=True,
                **param)

    sys.setrecursionlimit(10000)
    start = time.time()
    search_mode = 0
    net.search(net.m.copy(), net.marks.copy(),0,mode=search_mode)

    print(
        f'makespan={net.makespan}|search time = {time.time() - start:.2f}|back_time={net.back_time}'
        f'|expand marks={net.expand_mark}|search mode={search_mode}')

    #s=''
    #for n in net.transitions[:20]:
    #    s += f"{net.id2t_name[n]}|"
    #print(s)

    import pandas as pd
    df = pd.DataFrame(net.id2p_name,columns=['Place'])
    df2 = pd.DataFrame(net.net,columns=net.id2t_name)
    df3 = pd.concat([df,df2], axis=1).to_csv('../supernet_output.csv', index=False)


    actions = net.transitions
    times = net.time_record
    pm_intervals = build_pm_intervals(actions, times, info['module_x'])
    fig, ax = plot_gantt(pm_intervals, title="PM Utilization Gantt")
