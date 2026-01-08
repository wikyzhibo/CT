# 任务c
params_N7 = {'path':r'C:\Users\khand\OneDrive\code\dqn\CT\data\ndr_data\N7.txt',
          'n_wafer':[25,50],
          'idle_place':{'start':['L1','L10'],'end':['L2']},
          'process_time':[10,20,70,5,600,70,200,20],
          'capacity':{'pm':[1,1,2,1,4,1,2,1],
                      'robot':[1,2,2]},
          'branch_info':{'branch':[17,18],'pre':13},
          'capacity_xianzhi':{'s2':'u1'},
          'controller':{
                        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'),
                              ('p5', 4, 'u4'), ('p7', 2, 'u31'),
                              ('p7', 2, 'u6')]}
                        }