"""
PySide6 可视化入口

改进：
- 正确设置应用图标（任务栏 + 窗口）
- Windows 下设置 AppUserModelID
"""

from __future__ import annotations

import argparse
import re
import sys
import ctypes
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from solutions.A.rl_env import Env_PN_Single, Env_PN_Concurrent, make_env

from .petri_single_adapter import PetriSingleAdapter
from .petri_adapter import PetriAdapter
from .viewmodel import PetriViewModel
from .main_window import PetriMainWindow


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


def _to_dual_head_inner_state_dict(raw: dict) -> dict:
    """`ppo_trainer` 并发保存 `DualActionPolicyModule.state_dict()`，键前缀多为 `backbone.backbone.*`；剥掉一层 `backbone.` 得到 `DualHeadPolicyNet` 键。"""
    keys = [str(k) for k in raw.keys()]
    if any(k.startswith("backbone.backbone.") for k in keys):
        prefix = "backbone."
        return {str(k)[len(prefix) :]: v for k, v in raw.items() if str(k).startswith(prefix)}
    return raw


def _infer_dual_head_inner_shape(state_dict: dict) -> tuple[int, int, int, int, int]:
    """与 `export_inference_sequence._infer_dual_head_shape` 一致，但针对内层键（`backbone.0.weight` 等）。"""
    w0 = state_dict.get("backbone.0.weight")
    if w0 is None:
        raise KeyError("并发策略权重缺少 backbone.0.weight")
    n_hidden, n_obs = int(w0.shape[0]), int(w0.shape[1])
    h2 = state_dict["head_tm2.weight"]
    h3 = state_dict["head_tm3.weight"]
    n_actions_tm2 = int(h2.shape[0])
    n_actions_tm3 = int(h3.shape[0])
    linear_count = 0
    for k in state_dict:
        m = re.match(r"^backbone\.(\d+)\.weight$", k)
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
        return True, ""
    if window._adapter_factory is None:
        return False, "当前窗口未注入适配器工厂，无法切换运行模式。"
    overrides = {"runtime_mode": runtime_mode} if want_concurrent else None
    try:
        new_adapter = window._adapter_factory(target_mode, overrides)
        window.apply_runtime_adapter(
            new_adapter,
            target_mode,
            concurrent_runtime=want_concurrent,
            reset=True,
        )
        return True, ""
    except Exception as exc:
        return False, f"切换到 {runtime_mode} 可视化后端失败: {exc}"


def load_concurrent_model(model_path: str, adapter: PetriAdapter) -> tuple[object | None, str]:
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
        inner_state_dict = _to_dual_head_inner_state_dict(raw_state_dict)
        ckpt_n_obs, n_hidden, ckpt_na2, ckpt_na3, n_layers = _infer_dual_head_inner_shape(inner_state_dict)

        if int(ckpt_n_obs) != int(n_obs):
            raise ValueError(f"权重 n_obs={ckpt_n_obs} 与当前环境观测维 {n_obs} 不一致")
        if int(ckpt_na2) != n_actions_tm2 or int(ckpt_na3) != n_actions_tm3:
            raise ValueError(
                f"权重动作维 ({ckpt_na2},{ckpt_na3}) 与 Env ({n_actions_tm2},{n_actions_tm3}) 不一致"
            )

        print(
            f"[Concurrent Model] n_obs={n_obs}, n_hidden={n_hidden}, n_layers={n_layers}, "
            f"TM2={n_actions_tm2}, TM3={n_actions_tm3} (TM1 自动)"
        )

        backbone = DualHeadPolicyNet(
            n_obs=n_obs,
            n_hidden=n_hidden,
            n_actions_tm2=n_actions_tm2,
            n_actions_tm3=n_actions_tm3,
            n_layers=n_layers,
        )
        backbone.load_state_dict(inner_state_dict, strict=True)
        backbone.eval()

        print(f"✓ 并发模型加载成功: {model_path}")

        def get_model_actions():
            """返回 `(a1, a2, a3)`：`a1=-1` 占位；仿真仅使用 TM2/TM3 全局变迁索引（-1 表示 WAIT）。"""
            try:
                obs_f = torch.as_tensor(env._build_obs(), dtype=torch.float32).unsqueeze(0)
                _mask_tm1_np, mask_tm2_np, mask_tm3_np = env.net.get_action_mask(
                    wait_action_start=int(env.net.T),
                    n_actions=int(env.net.T + 1),
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

        return get_model_actions, ""

    except Exception as e:
        print(f"✗ 加载并发模型失败: {e}")
        return None, str(e)


def apply_model_for_mode(model_path: str, device_mode: str, window: PetriMainWindow) -> tuple[bool, str]:
    """仅加载并发模型，并在需要时切到并发运行时适配器。"""
    ok, reason = _ensure_runtime_adapter(window, device_mode, "concurrent")
    if not ok:
        window.set_concurrent_model_handler(None)
        return False, reason

    adapter = window.viewmodel.adapter
    if not isinstance(adapter, PetriAdapter):
        window.set_concurrent_model_handler(None)
        return False, "并发适配器切换失败，当前窗口仍不是并发后端。"
    handler, error_msg = load_concurrent_model(model_path, adapter)
    if handler is None:
        window.set_concurrent_model_handler(None)
        return False, f"并发模型加载失败: {error_msg}"
    window.set_concurrent_model_handler(handler)
    return True, f"并发模型加载成功: {model_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 Petri 可视化")
    parser.add_argument("--adapter", default="petri", choices=["petri"], help="算法适配器")
    parser.add_argument("--device", type=str, default="cascade", choices=["cascade"], help="设备模式（仅支持 cascade）")
    parser.add_argument("--device-mode", type=str, choices=["cascade"], help="已弃用，等价于 --device（仅支持 cascade）")
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
        step_verbose=not args.quiet,
        concurrent=concurrent_mode,
    )
    viewmodel = PetriViewModel(adapter)

    app = QApplication(sys.argv)
    
    # 设置应用图标（在创建窗口之前）
    app_icon = set_app_icon(app)
    
    window = PetriMainWindow(viewmodel, debug=args.debug)
    window._cascade_route_name = getattr(viewmodel.adapter.env.net, "single_route_name", None)

    def adapter_factory(mode, env_overrides=None):
        ov = dict(env_overrides or {})
        if mode == "cascade":
            rn = getattr(window, "_cascade_route_name", None)
            if rn:
                ov.setdefault("single_route_name", str(rn))
        return build_adapter(
            args.adapter,
            device_mode=mode,
            env_overrides=ov,
            step_verbose=not args.quiet,
            concurrent=window._concurrent_runtime,
        )

    window.set_adapter_factory(adapter_factory)
    if selected_device == "cascade":
        window._device_mode = selected_device
        window._concurrent_runtime = concurrent_mode
        window.center_canvas.set_device_mode(selected_device)
        window._action_device_cascade.setChecked(selected_device == "cascade")
        window._refresh_status_message()
    window.set_model_apply_callback(lambda path, mode: apply_model_for_mode(path, mode, window))
    
    # 窗口也设置图标
    if app_icon:
        window.setWindowIcon(app_icon)
    
    # 加载模型（仅在显式传 --model 时自动应用）
    if not args.no_model:
        if args.model:
            ok, msg = apply_model_for_mode(args.model, selected_device, window)
            print(("✓ " if ok else "✗ ") + msg)
        else:
            print("未指定 --model，启动手动模式（可在回放菜单中选择模型文件）。")
    else:
        print("已禁用模型加载")
        print("提示: 即使不加载模型，也可在回放菜单选择 JSON 后使用 Model B 进行离线回放。")
    
    window.show()

    # 初始化状态
    viewmodel.reset()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
