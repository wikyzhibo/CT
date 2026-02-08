from solutions.Td_petri.tdpn import TimedPetri
import torch

env = TimedPetri()
td = env._reset()
print("Reset OK")

valid = torch.where(td['action_mask'])[0]
td.set('action', valid[0].unsqueeze(0))
td2 = env._step(td)
print("Step OK, reward:", td2['reward'].item())
