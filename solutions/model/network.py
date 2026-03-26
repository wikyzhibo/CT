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


class DualHeadPolicyNet(nn.Module):
    """
    双输出头策略网络，分别输出 TM2 和 TM3 的动作概率。
    
    共享 backbone 提取特征，然后通过两个独立的输出头预测各自的动作。
    """
    def __init__(self, n_obs, n_hidden=256, n_actions_tm2=11, n_actions_tm3=5, n_layers=4):
        """
        Args:
            n_obs: 观测维度
            n_hidden: 隐藏层维度
            n_actions_tm2: TM2 动作空间大小（10 变迁 + 1 WAIT）
            n_actions_tm3: TM3 动作空间大小（4 变迁 + 1 WAIT）
            n_layers: backbone 层数
        """
        super().__init__()
        
        # 共享 backbone
        backbone_layers = [nn.Linear(n_obs, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            backbone_layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 双输出头
        self.head_tm2 = nn.Linear(n_hidden, n_actions_tm2)
        self.head_tm3 = nn.Linear(n_hidden, n_actions_tm3)
        
    def forward(self, obs):
        """
        Args:
            obs: 观测张量 (batch, n_obs)
            
        Returns:
            dict: {"logits_tm2": (batch, n_actions_tm2), "logits_tm3": (batch, n_actions_tm3)}
        """
        features = self.backbone(obs.to(torch.float32))
        logits_tm2 = self.head_tm2(features)
        logits_tm3 = self.head_tm3(features)
        return {"logits_tm2": logits_tm2, "logits_tm3": logits_tm3}
