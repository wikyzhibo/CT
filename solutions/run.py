from pdr.net import Petri
import time

params_N2 = {'path':'../Net/N2.txt',
          'n_wafer':25,
          'process_time':[20,10,2790,10,104,20,2790],
          'capacity':{'pm':[2,2,4,2,1,1,1],
                      'bm':[2,2],
                      'robot':[1,1,1],},}
'''

'''

params_N3 = {'path':'../Net/N3.txt',
             'n_wafer':2}
params_N4 = {'path':'Net/N4.txt'}

# C加工路径参数
params_N5 = {'path':'../Net/N5.txt',
          'n_wafer':75,
          'process_time':[8,20,70,0,600,70,200,20],
          'capacity':{'pm':[1,2,2,2,4,2,2,2],
                      'bm':[2,2],
                      'robot':[1,2,2]},
          'controller':{'bm1':{'p':['d2','p2','d8','p8'],
                                't':['u1', 't2', 'u7', 't8']},
                        'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                't': ['u3', 't4', 'u5', 't6']},
                        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                              ('p5', 4, 'u4'), ('p7', 2, 'u6')]}
                        }

# 任务b
params_N6 = {'path':'../Net/N6.txt',
          'n_wafer':75,
          'process_time':[8,20,70,0,300,70,20],
          'capacity':{'pm':[1,2,2,2,2,2,2],
                      'bm':[2,2],
                      'robot':[1,2,2]},
          'capacity_xianzhi':{'s1':'u1'},
          'controller':{'bm1':{'p':['d2','p2','d7','p7'],
                                't':['u1', 't2', 'u6', 't7']},
                        'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                't': ['u3', 't4', 'u5', 't6']},
                        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                              ('p5', 2, 'u4')]}
                        }

# 任务c
params_N7 = {'path':'../Net/N7.txt',
          'n_wafer':75,
          'two_mode':2,
          'process_time':[8,20,70,0,600,70,200,20],
          'capacity':{'pm':[1,2,2,2,4,2,2,2],
                      'bm':[2,2],
                      'robot':[1,2,2]},
          'branch_info':{'branch':[17,18],'pre':13},
          'capacity_xianzhi':{'s1':'u3','s2':'u1'},
          'controller':{'bm1':{'p':['d2','p2','d7','p7'],
                                't':['u1', 't2', 'u6', 't7']},
                        'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                't': ['u3', 't4', 'u5', 't6']},
                        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                              ('p5', 4, 'u4'), ('p7', 2, 'u31')]}
                        }

def main():
    import sys
    sys.setrecursionlimit(10000)
    search_mode = 0
    start = time.time()
    net = Petri(**params_N6)
    print(f'|p={net.P}|t={net.T}')
    m = net.m.copy()
    marks = net.marks.copy()
    net.search(m,marks,0,search_mode)
    print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}|back_time={net.back_time}|expand marks={net.expand_mark}|search mode={search_mode}')

main()