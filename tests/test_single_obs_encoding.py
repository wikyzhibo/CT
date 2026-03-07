import numpy as np
import torch

from solutions.Continuous_model.env_single import Env_PN_Single


def _feature_slice(env: Env_PN_Single, obs: np.ndarray, wafer_idx: int) -> np.ndarray:
    start = wafer_idx * env.wafer_feat_dim
    end = start + env.wafer_feat_dim
    return obs[start:end]


def _status_slice(env: Env_PN_Single, feat: np.ndarray) -> np.ndarray:
    status_start = 1 + env.pair_dim
    return feat[status_start : status_start + 4]


def test_observation_is_float32_and_present_padding():
    env = Env_PN_Single()
    td = env.reset()
    assert td["observation"].dtype == torch.float32

    lp = env.net._get_place("LP")
    keep_tok = lp.head()
    for p in env.net.marks:
        p.tokens.clear()
    lp.append(keep_tok)

    obs = env._build_obs()
    assert obs.dtype == np.float32

    feat0 = _feature_slice(env, obs, 0)
    present = float(feat0[0])
    assert present == 1.0

    feat_last = _feature_slice(env, obs, env.MAX_WAFERS - 1)
    assert float(feat_last[0]) == 0.0


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
    feat0 = _feature_slice(env, obs, 0)

    status = _status_slice(env, feat0)
    # status 顺序: processing / done_waiting_pick / moving / waiting
    assert np.allclose(status, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    rem_processing = float(feat0[1 + env.pair_dim + 4])
    rem_scrap = float(feat0[1 + env.pair_dim + 5])
    assert rem_processing == 0.0
    assert rem_scrap == 1.0
