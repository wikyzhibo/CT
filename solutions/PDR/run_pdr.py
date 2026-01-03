import sys
import time

from solutions.PDR.net import Petri
from data.config.params_N8 import params_N8


tmp = ['t7', 'u6', 't6', 'u4', 't5', 'u5', 'u2', 't8', 'u7', 't3', 't2', 't1', 'u1', 'u0', 't4', 'u3']

def main():
    sys.setrecursionlimit(10000)
    search_mode = 3
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
          f'|expand marks={net.expand_mark}|search mode={search_mode}|'
          f'residual_violation={net.over_time}|Q_time_violation={net.qtime_violation}')
    #print(net.log)

if __name__ == '__main__':
    main()