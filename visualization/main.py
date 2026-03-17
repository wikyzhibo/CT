"""
PySide6 可视化入口

改进：
- 正确设置应用图标（任务栏 + 窗口）
- Windows 下设置 AppUserModelID
"""

from __future__ import annotations

import argparse
import sys
import ctypes
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from solutions.Continuous_model.env_single import Env_PN_Single

from .petri_single_adapter import PetriSingleAdapter
from .viewmodel import PetriViewModel
from .main_window import PetriMainWindow


def set_app_icon(app: QApplication) -> QIcon | None:
    """设置应用图标，返回图标对象或 None"""
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    icon_candidates = ["app.ico", "app.png", "icon.ico", "icon.png"]
    
    for icon_name in icon_candidates:
        icon_path = assets_dir / icon_name
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            app.setWindowIcon(icon)
            print(f"✓ 图标加载成功: {icon_path}")
            return icon
    
    print("⚠ 未找到应用图标文件，使用默认图标")
    return None


def set_windows_app_id():
    """设置 Windows AppUserModelID，使任务栏图标正确显示"""
    if sys.platform == "win32":
        try:
            app_id = "ct.visualization.wafer.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass  # 忽略错误


def build_adapter(
    adapter_name: str,
    device_mode: str = "cascade",
    robot_capacity: int = 1,
    route_code: int | None = None,
    env_overrides: dict | None = None,
    step_verbose: bool = True,
):
    if adapter_name != "petri":
        raise ValueError(f"不支持的适配器: {adapter_name}")
    env_overrides = dict(env_overrides or {})
    selected_route_code = env_overrides.get("single_route_code", route_code)
    effective_robot_capacity = int(env_overrides.get("single_robot_capacity", robot_capacity))
    env = Env_PN_Single(
        detailed_reward=True,
        device_mode=device_mode,
        robot_capacity=effective_robot_capacity,
        route_code=None if selected_route_code is None else int(selected_route_code),
        process_time_map=env_overrides.get("single_process_time_map"),
        proc_time_rand_enabled=env_overrides.get("proc_rand_enabled")
        or env_overrides.get("single_proc_time_rand_enabled"),
        proc_time_rand_scale_map=env_overrides.get("single_proc_time_rand_scale_map"),
    )
    return PetriSingleAdapter(env, step_verbose=step_verbose)






def load_model(model_path: str, adapter: PetriSingleAdapter):
    """
    加载训练好的模型
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
        # Ensure we get the correct obs dim
        n_obs = adapter.env.observation_spec["observation"].shape[0]
        
        print(f"[DEBUG] Model Params: n_actions={n_actions}, n_obs={n_obs}")

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
        
        # 加载 state_dict（兼容多种历史保存格式）
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        loaded = False
        actor_error: Exception | None = None
        try:
            # 格式1：直接保存 ProbabilisticActor.state_dict()
            policy.load_state_dict(state_dict)
            loaded = True
        except Exception as e:
            actor_error = e

        if not loaded:
            # 格式2/3：保存 policy_module.state_dict() 或 policy_backbone.state_dict()
            # 需要把 key 归一到 MaskedPolicyHead 的参数命名。
            candidate_state_dicts = []

            # identity
            candidate_state_dicts.append(state_dict)

            # strip "backbone." 前缀（train_single 中 policy_module 保存）
            candidate_state_dicts.append(
                {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
            )

            # strip "module.0.module." 前缀（部分旧可视化封装保存）
            candidate_state_dicts.append(
                {k[len("module.0.module."):]: v for k, v in state_dict.items() if k.startswith("module.0.module.")}
            )

            # strip "backbone.module.0.module." 前缀（双重封装的极端情况）
            candidate_state_dicts.append(
                {
                    k[len("backbone.module.0.module."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("backbone.module.0.module.")
                }
            )

            for cand in candidate_state_dicts:
                if not cand:
                    continue
                try:
                    policy_backbone.load_state_dict(cand)
                    loaded = True
                    break
                except Exception:
                    continue

        if not loaded:
            raise RuntimeError(
                f"无法识别的单设备模型权重格式: {model_path}. "
                f"原始加载错误: {actor_error}"
            )
        policy.eval()
        
        print(f"✓ 模型加载成功: {model_path}")
        
        # 创建模型动作获取函数
        def get_model_action() -> int:
            """使用模型预测动作"""
            try:
                # 获取观察和动作掩码
                obs = adapter.env.net.get_obs()
                # 单设备多档 wait 下，直接复用 env 输出的离散动作掩码。
                action_mask = torch.as_tensor(adapter.env._mask(), dtype=torch.bool)
                
                # 构建 TensorDict
                # MaskedPolicyHead expects 'observation_f' (float) based on models.py
                td = TensorDict({
                    "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0),
                    "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                    "action_mask": action_mask.unsqueeze(0),
                }, batch_size=[1])
                
                # 使用模型预测
                with torch.no_grad():
                    # explicitly set mode to MODE (ArgMax) to match viz.py manual fix
                    with set_exploration_type(ExplorationType.RANDOM):
                        td = policy(td)
                        action = td["action"].item()
                
                return int(action)
            except Exception as e:
                print(f"[ERROR] Inference Failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return get_model_action
        
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return None


def load_concurrent_model(model_path: str, adapter: PetriSingleAdapter):
    """
    加载双机械手并发动作模型。
    
    返回一个函数，调用时返回 (a1, a2) 双动作。
    """
    import torch
    from torchrl.modules import MaskedCategorical
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from solutions.PPO.network.models import DualHeadPolicyNet
    from solutions.Continuous_model.train_concurrent import DualActionPolicyModule
    from solutions.Continuous_model.pn import TM2_TRANSITIONS, TM3_TRANSITIONS
    
    try:
        n_obs = adapter.env.observation_spec["observation"].shape[0]
        
        # 构建变迁名称到索引的映射
        t_name_to_idx = {name: i for i, name in enumerate(adapter.net.id2t_name)}
        
        # TM2/TM3 动作空间
        TM2_TRANSITION_NAMES = [
            "u_LP1_s1", "u_LP2_s1", "u_s1_s2", "u_s1_s5", "u_s4_s5", "u_s5_LP_done",
            "t_s1", "t_s2", "t_s5", "t_LP_done"
        ]
        TM3_TRANSITION_NAMES = [
            "u_s2_s3", "u_s3_s4", "t_s3", "t_s4"
        ]
        
        tm2_indices = [t_name_to_idx[n] for n in TM2_TRANSITION_NAMES if n in t_name_to_idx]
        tm3_indices = [t_name_to_idx[n] for n in TM3_TRANSITION_NAMES if n in t_name_to_idx]
        n_actions_tm2 = len(tm2_indices) + 1  # +1 for WAIT
        n_actions_tm3 = len(tm3_indices) + 1
        
        print(f"[Concurrent Model] n_obs={n_obs}, TM2={n_actions_tm2}, TM3={n_actions_tm3}")
        
        # 构建模型
        backbone = DualHeadPolicyNet(
            n_obs=n_obs,
            n_hidden=256,
            n_actions_tm2=n_actions_tm2,
            n_actions_tm3=n_actions_tm3,
            n_layers=4,
        )
        policy_module = DualActionPolicyModule(backbone)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        policy_module.load_state_dict(state_dict)
        policy_module.eval()
        
        print(f"✓ 并发模型加载成功: {model_path}")
        
        def get_model_actions():
            """返回 (a1, a2) 双动作（Petri 网变迁索引，-1 表示 WAIT）"""
            try:
                obs = adapter.env.net.get_obs()
                tm2_enabled, tm3_enabled = adapter.net.get_enable_t()
                tm2_enabled_set = set(tm2_enabled)
                tm3_enabled_set = set(tm3_enabled)
                
                # 构建掩码
                mask_tm2 = torch.zeros(n_actions_tm2, dtype=torch.bool)
                for i, t_idx in enumerate(tm2_indices):
                    mask_tm2[i] = (t_idx in tm2_enabled_set)
                mask_tm2[-1] = True  # WAIT
                
                mask_tm3 = torch.zeros(n_actions_tm3, dtype=torch.bool)
                for i, t_idx in enumerate(tm3_indices):
                    mask_tm3[i] = (t_idx in tm3_enabled_set)
                mask_tm3[-1] = True
                
                obs_f = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    with set_exploration_type(ExplorationType.MODE):
                        out = policy_module.backbone(obs_f)
                        dist_tm2 = MaskedCategorical(logits=out["logits_tm2"], mask=mask_tm2.unsqueeze(0))
                        dist_tm3 = MaskedCategorical(logits=out["logits_tm3"], mask=mask_tm3.unsqueeze(0))
                        a1_idx = dist_tm2.mode.item()
                        a2_idx = dist_tm3.mode.item()
                
                # 转换为 Petri 网变迁索引
                a1 = -1 if a1_idx == len(tm2_indices) else tm2_indices[a1_idx]
                a2 = -1 if a2_idx == len(tm3_indices) else tm3_indices[a2_idx]
                
                return (a1, a2)
            except Exception as e:
                print(f"[ERROR] Concurrent Inference Failed: {e}")
                import traceback
                traceback.print_exc()
                return (-1, -1)  # 都等待
        
        return get_model_actions
        
    except Exception as e:
        print(f"✗ 加载并发模型失败: {e}")
        return None


def apply_model_for_mode(model_path: str, device_mode: str, window: PetriMainWindow) -> tuple[bool, str]:
    """按设备模式加载模型并回填到窗口 handler。"""
    adapter = window.viewmodel.adapter
    if not isinstance(adapter, PetriSingleAdapter):
        window.set_model_handler(None)
        return False, "当前适配器与 pn_single 不一致，请先切换设备后重试。"
    handler = load_model(model_path, adapter)
    if handler is None:
        window.set_model_handler(None)
        return False, f"{device_mode} 模式模型加载失败，请确认权重与当前代码版本匹配: {model_path}"
    window.set_model_handler(handler)
    return True, f"{device_mode} 模式模型加载成功: {model_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 Petri 可视化")
    parser.add_argument("--adapter", default="petri", choices=["petri"], help="算法适配器")
    parser.add_argument("--device", type=str, default="single", choices=["single", "cascade"], help="设备模式")
    parser.add_argument("--device-mode", type=str, choices=["single", "cascade"], help="已弃用，等价于 --device")
    parser.add_argument("--single-route-code", type=int, default=None, choices=[0, 1], help="单设备路径代号（不传则使用 single.json）")
    parser.add_argument("--model", "-m", type=str, help="模型文件路径")
    parser.add_argument("--no-model", action="store_true", help="不加载模型")
    parser.add_argument("--debug", action="store_true", help="显示变迁按钮（用于调试）")
    parser.add_argument("--quiet", "-q", action="store_true", help="关闭每步使能/奖励的后台打印")
    args = parser.parse_args()
    selected_device = args.device_mode if args.device_mode else args.device

    # Windows 任务栏图标 fix
    set_windows_app_id()

    adapter = build_adapter(
        args.adapter,
        device_mode=selected_device,
        robot_capacity=1,
        route_code=args.single_route_code,
        step_verbose=not args.quiet,
    )
    viewmodel = PetriViewModel(adapter)

    app = QApplication(sys.argv)
    
    # 设置应用图标（在创建窗口之前）
    app_icon = set_app_icon(app)
    
    window = PetriMainWindow(viewmodel, debug=args.debug)
    window.set_adapter_factory(
        lambda mode, robot_capacity=1, env_overrides=None: build_adapter(
            args.adapter,
            device_mode=mode,
            robot_capacity=robot_capacity,
            route_code=args.single_route_code,
            env_overrides=env_overrides,
            step_verbose=not args.quiet,
        )
    )
    if selected_device in {"single", "cascade"}:
        window._device_mode = selected_device
        window.center_canvas.set_device_mode(selected_device)
        window._action_device_cascade.setChecked(selected_device == "cascade")
        window._action_device_single.setChecked(selected_device == "single")
        window._refresh_status_message()
    window.set_model_apply_callback(lambda path, mode: apply_model_for_mode(path, mode, window))
    
    # 窗口也设置图标
    if app_icon:
        window.setWindowIcon(app_icon)
    
    # 加载模型（如果指定）
    if not args.no_model:
        if args.model:
            model_path = args.model
        else:
            # 默认模型路径
            default_model = Path("solutions/Continuous_model/saved_models/CT_single_phase2_best.pt")
            if default_model.exists():
                model_path = str(default_model)
                print(f"使用默认模型: {model_path}")
            else:
                model_path = None
                print("未找到默认模型，将以手动模式运行")
        
        if model_path:
            ok, msg = apply_model_for_mode(model_path, selected_device, window)
            print(("✓ " if ok else "✗ ") + msg)
    else:
        print("已禁用模型加载")
        print("提示: 即使不加载模型，也可在回放菜单选择 JSON 后使用 Model B 进行离线回放。")
    
    window.show()

    # 初始化状态
    viewmodel.reset()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
