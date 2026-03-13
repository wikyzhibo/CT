import numpy as np
import torch

from solutions.Continuous_model.env_single import Env_PN_Single


def test_observation_is_float32_and_present_padding():
    env = Env_PN_Single()
    td = env.reset()
    assert td["observation"].dtype == torch.float32

    obs = env._build_obs()
    assert obs.dtype == np.float32
    assert int(obs.shape[-1]) == int(env.observation_spec["observation"].shape[-1])

    # reset 后运输位默认无晶圆，TM 特征为 8 维：时间 [0,0,0,1] + 去向 one-hot [0,0,0,0]
    tm_features = obs[1:9]
    assert np.allclose(tm_features, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_transport_status_and_time_to_scrap_semantics():
    env = Env_PN_Single()
    env.reset()

    lp = env.net._get_place("LP")
    d_tm1 = env.net._get_place("d_TM1")

    tok = lp.pop_head()
    tok.stay_time = int(env.net.D_Residual_time) + 1
    tok.where = 1
    d_tm1.append(tok)

    for p in env.net.marks:
        if p.name not in {"d_TM1"}:
            p.tokens.clear()

    obs = env._build_obs()
    tm_features = obs[1:9]
    transport_complete, wafer_stay_over_long, wafer_stay_time_norm, distance_to_penalty_norm = tm_features[:4]

    assert float(transport_complete) == 1.0
    assert float(wafer_stay_over_long) == 1.0
    assert 0.0 < float(wafer_stay_time_norm) <= 1.0
    assert float(distance_to_penalty_norm) == 0.0
