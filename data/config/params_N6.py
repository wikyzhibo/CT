# params_N6 configuration extracted from solutions/PPO/run_ppo.py
# Keep this module importable as: from config.params_N6 import params_N6

params_N6 = {
    'path': r'C:\Users\khand\OneDrive\code\dqn\CT\Net\N6.txt',
    'n_wafer': 10,
    'process_time': [8, 20, 70, 0, 300, 70, 20],
    'capacity': {
        'pm': [1, 2, 2, 2, 2, 2, 2],
        'bm': [2, 2],
        'robot': [1, 2, 2]
    },
    'capacity_xianzhi': {'s1': 'u1'},
    'controller': {
        'bm1': {'p': ['d2', 'p2', 'd7', 'p7'],
                't': ['u1', 't2', 'u6', 't7']},
        'bm2': {'p': ['d4', 'p4', 'd6', 'p6'],
                't': ['u3', 't4', 'u5', 't6']},
        'f': [('p1', 1, 'u0'), ('p3', 2, 'u2'), ('p5', 2, 'u4')]
    }
}

