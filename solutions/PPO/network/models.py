import torch.nn as nn
import torch

# --------------- 策略头 ---------------
class MaskedPolicyHead(nn.Module):
    def __init__(self, hidden=128,n_obs=None, n_actions=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs,hidden), nn.ReLU(),
            nn.Linear(hidden,hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden,n_actions)
        )
    def forward(self, obs):
        logits = self.net(obs.to(torch.float32))
        return logits