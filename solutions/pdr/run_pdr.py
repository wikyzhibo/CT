import numpy as np
import sys

from pdr.net import Petri
from utils.draw_gantt import build_pm_intervals, plot_gantt
from config.params_N8 import params_N8
from pdr.cal_remaining_time import train_remaining_model,predict_remaining
import time

tmp = ['t7', 'u6', 't6', 'u4', 't5', 'u5', 'u2', 't8', 'u7', 't3', 't2', 't1', 'u1', 'u0', 't4', 'u3']

def main():
    sys.setrecursionlimit(10000)
    search_mode = 0
    start = time.time()
    params_N8['n_wafer']=75
    net = Petri(with_controller=True,
                with_capacity_controller=True,
                with_zhiliu_controller = True,
                **params_N8)
    print(f'|p={net.P}|t={net.T}')

    m = net.m.copy()
    marks = net.marks.copy()
    net.search(m,marks,0,search_mode)
    print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}|back_time={net.back_time}'
          f'|expand marks={net.expand_mark}|search mode={search_mode}|over={net.over_time}')
    print(net.log)

    """"
    # 1️⃣ 训练模型
    model, dataset = train_remaining_model(
        net,
        epochs=50,
        batch_size=64
    )

    for i in range(len(net.m_record)):
        m_now = net.m_record[i]  # 当前 marking

        remaining = predict_remaining(
            model,
            dataset,
            m_now
        )[0]

        if remaining[0] > 1 or remaining[1] >1:
            print( f"({remaining[0]:.1f},{remaining[1]:.1f}) ")
    """


    x = []
    #for i,v in enumerate(net.transitions):
    #    t_name = net.id2t_name[v]
    #    if t_name == 'u2':
    #        print(f'u2: {int(net.time_record[i])}')


    for i,v in enumerate(net.transitions):
        t_name = net.id2t_name[v]
        if v == 9:
            cur = net.time_record[i]
            x.append(int(cur))
        elif v == 15:
            cur = net.time_record[i]
            start = x.pop(0)
            print(f'pm1: {int(cur)-start}')



    #print(net.u3_record)
    for i,v in enumerate(net.transitions):
        if v == 4:
            cur = net.time_record[i]
            x.append(int(cur))
        elif v == 5:
            cur = net.time_record[i]
            start = x.pop(0)
            print(f'pm2: {int(cur)-start}')

    '''
    for i in range(0,10):
    mark = net.marks_record[i]
    out = net.low_dim_token_times(marks=mark)
    net.snapshot(net.m_record[i], True)
    print(len(out),out)
    '''

main()