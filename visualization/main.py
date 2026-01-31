"""
PySide6 可视化入口
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from solutions.PPO.enviroment import Env_PN

from .petri_adapter import PetriAdapter
from .viewmodel import PetriViewModel
from .main_window import PetriMainWindow


def build_adapter(adapter_name: str) -> PetriAdapter:
    if adapter_name != "petri":
        raise ValueError(f"不支持的适配器: {adapter_name}")
    env = Env_PN(detailed_reward=True)
    return PetriAdapter(env)


def load_model(model_path: str, adapter: PetriAdapter):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        adapter: Petri 适配器
        
    Returns:
        模型动作获取函数
    """
    import torch
    from tensordict import TensorDict
    from torchrl.modules import ProbabilisticActor, MaskedCategorical
    from tensordict.nn import TensorDictModule
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from solutions.PPO.network.models import MaskedPolicyHead
    
    try:
        # 获取环境参数
        n_actions = adapter.env.n_actions
        n_obs = adapter.env.observation_spec["observation"].shape[0]
        
        # 构建模型架构（与训练时相同）
        policy_backbone = MaskedPolicyHead(
            hidden=256, 
            n_obs=n_obs, 
            n_actions=n_actions, 
            n_layers=4
        )
        td_module = TensorDictModule(
            policy_backbone, 
            in_keys=["observation_f"], 
            out_keys=["logits"]
        )
        policy = ProbabilisticActor(
            module=td_module,
            in_keys={"logits": "logits", "mask": "action_mask"},
            out_keys=["action"],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
        )
        
        # 加载 state_dict
        state_dict = torch.load(model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy.eval()
        
        print(f"✓ 模型加载成功: {model_path}")
        
        # 创建模型动作获取函数
        def get_model_action() -> int:
            """使用模型预测动作"""
            # 获取观察和动作掩码
            obs = adapter.env._build_obs()
            action_mask_indices = adapter.net.get_enable_t()
            action_mask = torch.zeros(n_actions, dtype=torch.bool)
            action_mask[action_mask_indices] = True
            action_mask[adapter.net.T] = True  # WAIT 动作
            
            # 构建 TensorDict
            td = TensorDict({
                "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0),
                "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                "action_mask": action_mask.unsqueeze(0),
            }, batch_size=[1])
            
            # 使用模型预测
            with torch.no_grad():
                with set_exploration_type(ExplorationType.MODE):
                    td = policy(td)
                    action = td["action"].item()
            
            return int(action)
        
        return get_model_action
        
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None



def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 Petri 可视化")
    parser.add_argument("--adapter", default="petri", choices=["petri"], help="算法适配器")
    parser.add_argument("--model", "-m", type=str, help="模型文件路径")
    parser.add_argument("--no-model", action="store_true", help="不加载模型")
    args = parser.parse_args()

    adapter = build_adapter(args.adapter)
    viewmodel = PetriViewModel(adapter)

    app = QApplication(sys.argv)
    window = PetriMainWindow(viewmodel)
    
    # 加载模型（如果指定）
    if not args.no_model:
        if args.model:
            model_path = args.model
        else:
            # 默认模型路径
            default_model = Path("result/best_policy.pt")
            if default_model.exists():
                model_path = str(default_model)
                print(f"使用默认模型: {model_path}")
            else:
                model_path = None
                print("未找到默认模型，将以手动模式运行")
        
        if model_path:
            model_handler = load_model(model_path, adapter)
            if model_handler:
                window.set_model_handler(model_handler)
    else:
        print("已禁用模型加载")
    
    window.show()

    # 初始化状态
    viewmodel.reset()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

