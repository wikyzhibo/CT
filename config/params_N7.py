# 任务c
params_N7 = {'path':'../Net/N7.txt',
          'n_wafer':12,
          'two_mode':1,
          'process_time':[8,20,70,0,600,70,200,20],
          'capacity':{'pm':[1,2,2,2,4,2,2,2],
                      'bm':[2,2],
                      'robot':[1,2,2]},
          'branch_info':{'branch':[17,18],'pre':13},
          'capacity_xianzhi':{'s1':'u3','s2':'u1'},
          'controller':{'bm1':{'p':['d2','p2','d8','p8'],
                                't':['u1', 't2', 'u7', 't8']},
                        'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                                't': ['u3', 't4', 'u5', 't6']},
                        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                              ('p5', 4, 'u4'), ('p7', 2, 'u31'),
                              ('p7', 2, 'u6')]}
                        }