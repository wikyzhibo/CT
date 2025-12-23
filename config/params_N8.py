# params_N6 configuration extracted from solutions/PPO/run_ppo.py
# Keep this module importable as: from config.params_N6 import params_N6

params_N8 = {
    'path': r'C:\Users\khand\OneDrive\code\dqn\CT\Net\N8.txt',
    'n_wafer': 7,
    'process_time': [8, 20, 70, 0, 300, 70, 20],
    'capacity': {
        'pm': [1, 2, 2, 1, 2, 1, 2],
        'bm': [2],
        'robot': [1, 2, 2]
    },
    'capacity_xianzhi': {'s1': 'u1'},
    'controller': {
        'bm1': {'p': ['d2', 'p2', 'd7', 'p7'],
                't': ['u1', 't2', 'u6', 't7']},
        'f1': [('p3','r2',2,'u2')]
    }
}