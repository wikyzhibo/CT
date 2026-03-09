"""Verification script for TM observation with wafer destination one-hot."""
from solutions.Continuous_model.env_single import Env_PN_Single

# single mode: TM=4
env_s = Env_PN_Single(device_mode="single", seed=0)
td_s = env_s.reset()
obs_s = td_s["observation"]
pm_names_s = env_s._get_place_obs_pm_names()
expected_s = 1 + 4 + 9 * len(pm_names_s)
assert obs_s.shape[-1] == expected_s
print("single OK, obs_dim =", expected_s)

# cascade mode: TM=14
env_c = Env_PN_Single(device_mode="cascade", seed=0)
td_c = env_c.reset()
obs_c = td_c["observation"]
pm_names_c = env_c._get_place_obs_pm_names()
expected_c = 1 + 14 + 9 * len(pm_names_c)
assert obs_c.shape[-1] == expected_c
print("cascade OK, obs_dim =", expected_c)
print("all checks passed")
