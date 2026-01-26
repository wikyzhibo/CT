import torch.nn as nn
import torch

# --------------- 策略头 ---------------
class MaskedPolicyHead(nn.Module):
    def __init__(self, hidden=128, n_obs=None, n_actions=None, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_obs, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.net(obs.to(torch.float32))
        return logits