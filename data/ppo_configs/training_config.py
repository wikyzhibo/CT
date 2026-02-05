"""PPO训练配置类"""
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class PPOTrainingConfig:
    """PPO训练配置参数"""
    
    # 网络结构参数
    n_hidden: int = 128
    n_layer: int = 4
    
    # 训练批次参数
    total_batch: int = 150
    sub_batch_size: int = 64
    num_epochs: int = 10
    
    # PPO算法参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    lr: float = 1e-4
    
    # 熵系数参数
    entropy_start: float = 0.02
    entropy_end: float = 0.01
    
    # 行为克隆参数
    lambda_bc0: float = 1.0
    bc_decay_batches: int = 200
    bc_weight_early: float = 2.0  # 前100个batch的BC权重
    bc_weight_late: float = 0.1   # 100个batch后的BC权重
    bc_switch_batch: int = 100    # 切换BC权重的batch数
    
    # 设备和随机种子
    device: str = "cpu"
    seed: int = 42
    
    # 训练阶段和预训练
    training_phase: int = 2
    with_pretrain: bool = False
    
    @property
    def frames_per_batch(self) -> int:
        """计算每个batch的帧数"""
        return self.sub_batch_size * self.num_epochs
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PPOTrainingConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __str__(self):
        """打印配置信息"""
        lines = ["=" * 50, "PPO训练配置:"]
        lines.append(f"  网络: hidden={self.n_hidden}, layers={self.n_layer}")
        lines.append(f"  批次: total={self.total_batch}, sub_batch={self.sub_batch_size}, epochs={self.num_epochs}")
        lines.append(f"  PPO: gamma={self.gamma}, lambda={self.gae_lambda}, clip={self.clip_epsilon}, lr={self.lr}")
        lines.append(f"  熵系数: {self.entropy_start} -> {self.entropy_end}")
        lines.append(f"  BC: lambda={self.lambda_bc0}, decay_batches={self.bc_decay_batches}")
        lines.append(f"  阶段: {self.training_phase}, 预训练: {self.with_pretrain}")
        lines.append(f"  设备: {self.device}, 种子: {self.seed}")
        lines.append("=" * 50)
        return "\n".join(lines)
