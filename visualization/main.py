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
import re
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from solutions.A.rl_env import Env_PN_Single, Env_PN_Concurrent, make_env

from .petri_single_adapter import PetriSingleAdapter
from .petri_adapter import PetriAdapter
from .viewmodel import PetriViewModel
from .main_window import PetriMainWindow
from results.paths import model_output_path


def set_app_icon(app: QApplication) -> QIcon | None:
    """设置应用图标，返回图标对象或 None"""
    assets_dir = Path(__file__).resolve().parent.parent / "results" / "image"
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
    env_overrides: dict | None = None,
    step_verbose: bool = True,
    concurrent: bool = False,
):
    if adapter_name != "petri":
        raise ValueError(f"不支持的适配器: {adapter_name}")
    env_overrides = dict(env_overrides or {})
    runtime_mode = str(env_overrides.get("runtime_mode", "concurrent" if concurrent else "single")).lower()
    env = make_env(
        runtime_mode=runtime_mode,
        device_mode=device_mode,
        device="cpu",
        env_overrides=env_overrides,
    )
    if isinstance(env, Env_PN_Concurrent):
        return PetriAdapter(env, step_verbose=step_verbose)
    if isinstance(env, Env_PN_Single):
        return PetriSingleAdapter(env, step_verbose=step_verbose)
    raise TypeError(f"不支持的环境类型: {type(env)}")


def _load_raw_state_dict(model_path: str):
    import torch

    return torch.load(model_path, map_location="cpu", weights_only=True)


def _detect_model_kind(state_dict: dict[str, object]) -> str:
    for key in state_dict.keys():
        key_str = str(key)
        if ("head_tm2" in key_str and "head_tm3" in key_str) or "head_tm1" in key_str:
            return "concurrent"
    return "single"


def _iter_single_candidate_state_dicts(state_dict: dict):
    return [
        state_dict,
        {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")},
        {k[len("module.0.module."):]: v for k, v in state_dict.items() if k.startswith("module.0.module.")},
        {
            k[len("backbone.module.0.module."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.module.0.module.")
        },
    ]


def _infer_single_model_shape(state_dict: dict) -> tuple[int, int]:
    first_linear = state_dict.get("net.0.weight")
    if first_linear is None:
        raise KeyError("权重缺少 net.0.weight，无法推断单动作策略网络结构")
    hidden = int(first_linear.shape[0])
    linear_key_pattern = re.compile(r"^net\.(\d+)\.weight$")
    linear_count = sum(1 for k in state_dict.keys() if linear_key_pattern.match(str(k)))
    if linear_count <= 1:
        raise ValueError("单动作策略网络线性层数量异常，无法推断 n_layers")
    return hidden, linear_count - 1


def _iter_concurrent_candidate_state_dicts(state_dict: dict):
    return [
        state_dict,
        {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")},
    ]


def _infer_concurrent_dual_head_shape(state_dict: dict) -> tuple[int, int, int, int, int]:
    """与 `export_inference_sequence._infer_dual_head_shape` 同构：DualHeadPolicyNet（仅 TM2/TM3）。"""
    w0 = state_dict.get("backbone.backbone.0.weight")
    if w0 is None:
        raise KeyError("权重缺少 backbone.backbone.0.weight")
    n_hidden, n_obs = int(w0.shape[0]), int(w0.shape[1])
    h2 = state_dict.get("backbone.head_tm2.weight")
    h3 = state_dict.get("backbone.head_tm3.weight")
    if h2 is None or h3 is None:
        raise KeyError("权重缺少 head_tm2 / head_tm3，无法推断 DualHeadPolicyNet 结构")
    n_actions_tm2 = int(h2.shape[0])
    n_actions_tm3 = int(h3.shape[0])
    linear_count = 0
    for k in state_dict:
        m = re.match(r"^backbone\.backbone\.(\d+)\.weight$", str(k))
        if m and int(m.group(1)) % 2 == 0:
            linear_count += 1
    n_layers = linear_count + 1
    return n_obs, n_hidden, n_actions_tm2, n_actions_tm3, n_layers


def _ensure_runtime_adapter(
    window: PetriMainWindow,
    device_mode: str,
    runtime_mode: str,
) -> tuple[bool, str]:
    runtime_mode = str(runtime_mode).lower()
    target_mode = "cascade" if runtime_mode == "concurrent" else str(device_mode).lower()
    if runtime_mode == "concurrent" and target_mode != "cascade":
        return False, "并发模型仅支持 cascade 设备模式。"
    adapter = window.viewmodel.adapter
    is_concurrent_adapter = isinstance(adapter, PetriAdapter)
    want_concurrent = runtime_mode == "concurrent"
    if is_concurrent_adapter == want_concurrent and window._device_mode == target_mode:
        window._concurrent_runtime = want_concurrent
        window._action_config_cascade_route.setEnabled(target_mode == "cascade")
        return True, ""
    if window._adapter_factory is None:
        return False, "当前窗口未注入适配器工厂，无法切换运行模式。"
    overrides = {"runtime_mode": runtime_mode} if want_concurrent else None
    try:
        new_adapter = window._adapter_factory(target_mode, window._robot_capacity, overrides)
        window.apply_runtime_adapter(
            new_adapter,
            target_mode,
            concurrent_runtime=want_concurrent,
            reset=True,
        )
        return True, ""
    except Exception as exc:
        return False, f"切换到 {runtime_mode} 可视化后端失败: {exc}"


def load_model(model_path: str, adapter: PetriSingleAdapter):
    """
    加载训练好的模型
    """
    import torch
    from tensordict import TensorDict
    from torchrl.modules import ProbabilisticActor, MaskedCategorical
    from tensordict.nn import TensorDictModule
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from solutions.model.network import MaskedPolicyHead
    
    try:
        n_actions = adapter.env.n_actions
        n_obs = adapter.env.observation_spec["observation"].shape[0]

        print(f"[DEBUG] Model Params: n_actions={n_actions}, n_obs={n_obs}")

        raw_state_dict = _load_raw_state_dict(model_path)
        hidden = 256
        n_layers = 4
        actor_error: Exception | None = None
        for cand in _iter_single_candidate_state_dicts(raw_state_dict):
            if not cand:
                continue
            try:
                hidden, n_layers = _infer_single_model_shape(cand)
                break
            except Exception:
                continue

        policy_backbone = MaskedPolicyHead(
            hidden=hidden,
            n_obs=n_obs,
            n_actions=n_actions,
            n_layers=n_layers,
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
        
        loaded = False
        try:
            # 格式1：直接保存 ProbabilisticActor.state_dict()
            policy.load_state_dict(raw_state_dict)
            loaded = True
        except Exception as e:
            actor_error = e

        if not loaded:
            for cand in _iter_single_candidate_state_dicts(raw_state_dict):
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


def load_concurrent_model(model_path: str, adapter: PetriAdapter):
    """
    加载并发双头策略（TM2/TM3）；TM1 由 `ClusterTool` 规则自动执行。
    返回的步进回调为 `(a1, a2, a3)`，其中 `a1` 恒为 `-1`（占位，仿真侧忽略）。
    """
    import torch
    from solutions.model.network import DualHeadPolicyNet

    try:
        env = adapter.env
        if not isinstance(env, Env_PN_Concurrent):
            raise TypeError("并发模型要求 Env_PN_Concurrent 适配器")

        n_obs = env.observation_spec["observation"].shape[0]
        n_actions_tm2 = int(env.n_actions_tm2)
        n_actions_tm3 = int(env.n_actions_tm3)
        raw_state_dict = _load_raw_state_dict(model_path)
        n_hidden = 256
        n_layers = 4
        inferred_tm2 = n_actions_tm2
        inferred_tm3 = n_actions_tm3
        inferred_obs = n_obs
        for cand in _iter_concurrent_candidate_state_dicts(raw_state_dict):
            if not cand:
                continue
            try:
                inferred_obs, n_hidden, inferred_tm2, inferred_tm3, n_layers = _infer_concurrent_dual_head_shape(
                    cand
                )
                break
            except Exception:
                continue
        if inferred_obs != n_obs or inferred_tm2 != n_actions_tm2 or inferred_tm3 != n_actions_tm3:
            raise ValueError(
                f"并发模型动作空间与当前环境不匹配: "
                f"weights=(obs={inferred_obs}, tm2={inferred_tm2}, tm3={inferred_tm3}) "
                f"env=(obs={n_obs}, tm2={n_actions_tm2}, tm3={n_actions_tm3})"
            )

        print(
            f"[Concurrent Model] n_obs={n_obs}, TM2={n_actions_tm2}, TM3={n_actions_tm3} (TM1 自动)"
        )

        backbone = DualHeadPolicyNet(
            n_obs=n_obs,
            n_hidden=n_hidden,
            n_actions_tm2=n_actions_tm2,
            n_actions_tm3=n_actions_tm3,
            n_layers=n_layers,
        )

        loaded = False
        for cand in _iter_concurrent_candidate_state_dicts(raw_state_dict):
            if not cand:
                continue
            try:
                backbone.load_state_dict(cand)
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            raise RuntimeError(f"无法识别的并发模型权重格式: {model_path}")
        backbone.eval()

        print(f"✓ 并发模型加载成功: {model_path}")

        def get_model_actions():
            """返回 `(a1, a2, a3)`：`a1=-1` 占位；仿真仅使用 TM2/TM3 全局变迁索引（-1 表示 WAIT）。"""
            try:
                obs_f = torch.as_tensor(env._build_obs(), dtype=torch.float32).unsqueeze(0)
                _mask_tm1_np, mask_tm2_np, mask_tm3_np = env.net.get_action_mask(
                    wait_action_start=int(env.net.T),
                    n_actions=int(env.net.T + len(env.wait_durations)),
                )
                mask_tm2 = torch.as_tensor(mask_tm2_np, dtype=torch.bool).unsqueeze(0)
                mask_tm3 = torch.as_tensor(mask_tm3_np, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    out = backbone(obs_f)
                    logits_tm2 = out["logits_tm2"].masked_fill(~mask_tm2, -1e9)
                    logits_tm3 = out["logits_tm3"].masked_fill(~mask_tm3, -1e9)
                    a2_idx = int(logits_tm2.argmax(dim=-1).item())
                    a3_idx = int(logits_tm3.argmax(dim=-1).item())

                a2 = -1 if a2_idx == env.tm2_wait_action else int(env.tm2_transition_indices[a2_idx])
                a3 = -1 if a3_idx == env.tm3_wait_action else int(env.tm3_transition_indices[a3_idx])

                return (-1, a2, a3)
            except Exception as e:
                print(f"[ERROR] Concurrent Inference Failed: {e}")
                import traceback
                traceback.print_exc()
                return (-1, -1, -1)

        return get_model_actions

    except Exception as e:
        print(f"✗ 加载并发模型失败: {e}")
        return None


def apply_model_for_mode(model_path: str, device_mode: str, window: PetriMainWindow) -> tuple[bool, str]:
    """按权重类型加载单动作或并发模型，并在需要时切换运行时适配器。"""
    try:
        raw_state_dict = _load_raw_state_dict(model_path)
    except Exception as exc:
        return False, f"读取模型失败: {exc}"

    model_kind = _detect_model_kind(raw_state_dict)
    runtime_mode = "concurrent" if model_kind == "concurrent" else "single"
    ok, reason = _ensure_runtime_adapter(window, device_mode, runtime_mode)
    if not ok:
        if model_kind == "concurrent":
            window.set_concurrent_model_handler(None)
        else:
            window.set_model_handler(None)
        return False, reason

    adapter = window.viewmodel.adapter
    if model_kind == "concurrent":
        if not isinstance(adapter, PetriAdapter):
            window.set_concurrent_model_handler(None)
            return False, "并发适配器切换失败，当前窗口仍不是并发后端。"
        handler = load_concurrent_model(model_path, adapter)
        if handler is None:
            window.set_concurrent_model_handler(None)
            return False, f"并发模型加载失败，请确认权重与当前代码版本匹配: {model_path}"
        window.set_concurrent_model_handler(handler)
        return True, f"并发模型加载成功: {model_path}"

    if not isinstance(adapter, PetriSingleAdapter):
        window.set_model_handler(None)
        return False, "单动作适配器切换失败，当前窗口仍不是 pn_single 后端。"
    handler = load_model(model_path, adapter)
    if handler is None:
        window.set_model_handler(None)
        return False, f"{device_mode} 模式模型加载失败，请确认权重与当前代码版本匹配: {model_path}"
    window.set_model_handler(handler)
    return True, f"{device_mode} 模式模型加载成功: {model_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 Petri 可视化")
    parser.add_argument("--adapter", default="petri", choices=["petri"], help="算法适配器")
    parser.add_argument("--device", type=str, default="cascade", choices=["single", "cascade"], help="设备模式")
    parser.add_argument("--device-mode", type=str, choices=["single", "cascade"], help="已弃用，等价于 --device")
    parser.add_argument("--model", "-m", type=str, help="模型文件路径")
    parser.add_argument("--concurrent", action="store_true", help="启用并发三动作可视化（仅支持 cascade）")
    parser.add_argument("--no-model", action="store_true", help="不加载模型")
    parser.add_argument("--debug", action="store_true", help="显示变迁按钮（用于调试）")
    parser.add_argument("--quiet", "-q", action="store_true", help="关闭每步使能/奖励的后台打印")
    args = parser.parse_args()
    selected_device = args.device_mode if args.device_mode else args.device
    concurrent_mode = bool(args.concurrent)
    if concurrent_mode and selected_device != "cascade":
        parser.error("--concurrent 仅支持 --device cascade")

    # Windows 任务栏图标 fix
    set_windows_app_id()

    adapter = build_adapter(
        args.adapter,
        device_mode=selected_device,
        robot_capacity=1,
        step_verbose=not args.quiet,
        concurrent=concurrent_mode,
    )
    viewmodel = PetriViewModel(adapter)

    app = QApplication(sys.argv)
    
    # 设置应用图标（在创建窗口之前）
    app_icon = set_app_icon(app)
    
    window = PetriMainWindow(viewmodel, debug=args.debug)
    window._cascade_route_name = getattr(viewmodel.adapter.env.net, "single_route_name", None)

    def adapter_factory(mode, robot_capacity=1, env_overrides=None):
        ov = dict(env_overrides or {})
        if mode == "cascade":
            rn = getattr(window, "_cascade_route_name", None)
            if rn:
                ov.setdefault("single_route_name", str(rn))
        return build_adapter(
            args.adapter,
            device_mode=mode,
            robot_capacity=robot_capacity,
            env_overrides=ov,
            step_verbose=not args.quiet,
            concurrent=window._concurrent_runtime,
        )

    window.set_adapter_factory(adapter_factory)
    if selected_device in {"single", "cascade"}:
        window._device_mode = selected_device
        window._concurrent_runtime = concurrent_mode
        window.center_canvas.set_device_mode(selected_device)
        window._action_device_cascade.setChecked(selected_device == "cascade")
        window._action_device_single.setChecked(selected_device == "single")
        window._action_config_cascade_route.setEnabled(selected_device == "cascade")
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
            default_model_name = "CT_concurrent_best.pt" if concurrent_mode else "CT_single_best.pt"
            default_model = model_output_path(default_model_name)
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
