"""
级联 / 并发双动作推理序列导出工具。

默认在 `Env_PN_Single`（级联拓扑、`MaskedPolicyHead`）上 roll out，导出 JSON 中每步为双机械臂占位
`actions: [动作名, "WAIT"]`。`--concurrent` 时在 `Env_PN_Concurrent` 上加载 `DualHeadPolicyNet` 权重 rollout。

将 `sequence`、`replay_env_overrides`、`reward_report` 等写入仓库根目录
`results/action_sequences/<out_name>(W<晶圆数>-M<makespan>).json`：`out_name` 经 `safe_name` 清洗后，
后缀取自 rollout 结束时的 `env.net.n_wafer` 与 `env.net.time`（仿真时刻）。

Rollout 最大步数固定为模块常量 `MAX_STEPS`（10000）。

`rollout_and_export(..., gantt_png_path=..., gantt_title_suffix=...)` 可在导出 JSON 后
对本次 rollout 的 `env.net` 调用 `render_gantt` 写出甘特 PNG（与 `ClusterTool.render_gantt` 行为一致）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical, ProbabilisticActor

from solutions.model.network import DualHeadPolicyNet, MaskedPolicyHead
from solutions.A.rl_env import Env_PN_Concurrent, Env_PN_Single
from results.paths import ACTION_SEQUENCES_DIR, ensure_results_dirs, model_output_path, safe_name

MAX_STEPS = 10000


def _action_sequence_export_path(out_name: str, env: Env_PN_Single | Env_PN_Concurrent) -> Path:
    """`results/action_sequences/<safe(out_name)>(W{n_wafer}-M{time}).json`。"""
    base = safe_name(str(out_name), "export")
    net = env.net
    w = int(net.n_wafer)
    m = int(net.time)
    stem = f"{base}(W{w}-M{m})"
    ensure_results_dirs()
    return ACTION_SEQUENCES_DIR / f"{stem}.json"


class _DualActionPolicyModule(nn.Module):
    """与 `ppo_trainer.DualActionPolicyModule` 同构，避免循环 import。"""

    def __init__(self, backbone: DualHeadPolicyNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, observation_f: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(observation_f)


def _to_step_state(td_next: Any) -> Any:
    if "next" in td_next.keys():
        return td_next["next"].clone()
    return td_next.clone()


def _to_terminated(td_next: Any) -> bool:
    if "next" in td_next.keys() and "terminated" in td_next["next"].keys():
        t = td_next["next", "terminated"]
    else:
        t = td_next["terminated"]
    return bool(t.item() if hasattr(t, "item") else t)


def _to_finish(td_next: Any) -> bool:
    if "next" in td_next.keys() and "finish" in td_next["next"].keys():
        f = td_next["next", "finish"]
    else:
        f = td_next.get("finish", False)
    return bool(f.item() if hasattr(f, "item") else f)


def _to_scrap(td_next: Any) -> bool:
    """从 step 返回值中提取 scrap 标志。"""
    for key in [("scrap",), ("next", "scrap")]:
        try:
            v = td_next
            for k in key:
                v = v[k]
            return bool(v.item() if hasattr(v, "item") else v)
        except (KeyError, TypeError):
            continue
    return False


def _to_time(td_state: Any) -> int:
    t = td_state["time"]
    return int(t.item() if hasattr(t, "item") else t)


def _decode_single_action_name(env: Env_PN_Single, action_idx: int) -> str:
    if hasattr(env, "get_action_name"):
        return str(env.get_action_name(int(action_idx)))
    if action_idx == env.net.T:
        return "WAIT"
    return env.net.id2t_name[action_idx]


def _iter_single_candidate_state_dicts(state_dict: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    candidates: list[dict[str, torch.Tensor]] = [state_dict]
    candidates.append(
        {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    )
    candidates.append(
        {k[len("module.0.module."):]: v for k, v in state_dict.items() if k.startswith("module.0.module.")}
    )
    candidates.append(
        {
            k[len("backbone.module.0.module."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.module.0.module.")
        }
    )
    return [c for c in candidates if c]


def _infer_single_model_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    first_linear = state_dict.get("net.0.weight")
    if first_linear is None:
        raise KeyError("权重缺少 net.0.weight，无法推断单设备策略网络结构")
    hidden = int(first_linear.shape[0])
    linear_key_pattern = re.compile(r"^net\.(\d+)\.weight$")
    linear_count = sum(1 for k in state_dict.keys() if linear_key_pattern.match(k))
    if linear_count <= 1:
        raise ValueError("单设备策略网络线性层数量异常，无法推断 n_layers")
    n_layers = linear_count - 1
    return hidden, n_layers


def _build_single_policy(
    env: Env_PN_Single,
    model_path: Path,
    device: torch.device,
) -> ProbabilisticActor:
    n_obs = int(env.observation_spec["observation"].shape[0])
    raw_state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    load_error: Exception | None = None

    candidates = _iter_single_candidate_state_dicts(raw_state_dict)

    for idx, candidate in enumerate(candidates):
        try:
            hidden, n_layers = _infer_single_model_shape(candidate)
            policy_backbone = MaskedPolicyHead(
                hidden=hidden,
                n_obs=n_obs,
                n_actions=env.n_actions,
                n_layers=n_layers,
            ).to(device)
            td_module = TensorDictModule(
                policy_backbone,
                in_keys=["observation_f"],
                out_keys=["logits"],
            )
            policy = ProbabilisticActor(
                module=td_module,
                in_keys={"logits": "logits", "mask": "action_mask"},
                out_keys=["action"],
                distribution_class=MaskedCategorical,
                return_log_prob=True,
            ).to(device)

            # 兼容直接保存 actor.state_dict() 的格式
            try:
                policy.load_state_dict(raw_state_dict)
                policy.eval()
                return policy
            except Exception:
                pass

            # 兼容保存 backbone/policy_module.state_dict() 的格式
            policy_backbone.load_state_dict(candidate)
            policy.eval()
            return policy
        except Exception as e:
            load_error = e
            continue

    raise RuntimeError(f"无法识别的单设备模型权重格式: {model_path}. 原始错误: {load_error}")


def _rollout_single_sequence(
    model_path: Path,
    seed: int,
    exploration: str = "mode",
) -> tuple[list[dict[str, Any]], bool, dict[str, Any], dict[str, Any], Env_PN_Single]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    env = Env_PN_Single(seed=seed, eval_mode=True)
    policy = _build_single_policy(env, model_path=model_path, device=device)

    td = env.reset()
    sequence: list[dict[str, Any]] = []
    finished = False
    scrap_steps: list[int] = []
    release_steps: list[int] = []
    idle_steps: list[int] = []

    with torch.no_grad():
        for step in range(1, MAX_STEPS + 1):
            try:
                obs = td["observation"]
                td_model = TensorDict(
                    {
                        "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0).to(device),
                        "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
                        "action_mask": td["action_mask"].unsqueeze(0).bool().to(device),
                    },
                    batch_size=[1],
                )

                explore_type = ExplorationType.MODE if exploration == "mode" else ExplorationType.RANDOM
                with set_exploration_type(explore_type):
                    td_out = policy(td_model)
                action_idx = int(td_out["action"].squeeze(0).item())

                step_td = td.clone()
                step_td["action"] = torch.tensor(action_idx, dtype=torch.int64)
                td_next = env.step(step_td)
            except Exception:
                raise

            if _to_scrap(td_next):
                scrap_steps.append(step)

            td_after = _to_step_state(td_next)
            current_time = _to_time(td_after)
            action_name = _decode_single_action_name(env, action_idx)
            terminated = _to_terminated(td_next)
            finished = _to_finish(td_next)

            sequence.append(
                {
                    "step": step,
                    "time": current_time,
                    "action": action_name,
                    "actions": [action_name],
                }
            )

            if terminated:
                break
            td = td_after

    reward_report = {
        "scrap_penalty": {"count": len(scrap_steps), "steps": scrap_steps},
        "release_penalty": {"count": len(release_steps), "steps": release_steps},
        "idle_timeout_penalty": {"count": len(idle_steps), "steps": idle_steps},
    }
    replay_env_overrides = {
        # 回放时固定为本次 episode 的实际工序时长，避免可视化重启后与导出时不一致。
        "process_time_map": dict(getattr(env.net, "_base_proc_time_map", {})),
        "robot_capacity": 1,
        "device_mode": "cascade",
        # 配置驱动路线信息：用于可视化重建与导出时一致的路径拓扑。
        "single_route_name": getattr(env.net, "single_route_name", None),
    }
    single_route_cfg = getattr(env.net, "single_route_config", None)
    if single_route_cfg is not None:
        replay_env_overrides["single_route_config"] = dict(single_route_cfg)
    return sequence, finished, replay_env_overrides, reward_report, env


def _rollout_single_sequence_with_retry(
    model_path: Path,
    seed: int,
    max_retries: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool, Env_PN_Single]:
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")

    last_sequence: list[dict[str, Any]] = []
    last_overrides: dict[str, Any] = {}
    last_report: dict[str, Any] = {}
    last_finished = False
    last_env: Env_PN_Single | None = None
    for attempt in range(max_retries):
        attempt_seed = seed + attempt
        sequence, finished, replay_env_overrides, reward_report, env = _rollout_single_sequence(
            model_path=model_path,
            seed=attempt_seed,
            exploration="mode" if attempt == 0 else "random",
        )
        last_sequence = sequence
        last_overrides = replay_env_overrides
        last_report = reward_report
        last_finished = finished
        last_env = env
        if finished:
            if attempt > 0:
                print(f"[INFO] 级联模式第 {attempt + 1} 次推理达到 finish。")
            return sequence, replay_env_overrides, reward_report, finished, env

    print(f"[WARN] 级联模式重试 {max_retries} 次后仍未 finish，导出最后一次序列。")
    assert last_env is not None
    return last_sequence, last_overrides, last_report, last_finished, last_env


def _infer_dual_head_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
    w0 = state_dict.get("backbone.backbone.0.weight")
    if w0 is None:
        raise KeyError("并发策略权重缺少 backbone.backbone.0.weight")
    n_hidden, n_obs = int(w0.shape[0]), int(w0.shape[1])
    h2 = state_dict["backbone.head_tm2.weight"]
    h3 = state_dict["backbone.head_tm3.weight"]
    n_actions_tm2 = int(h2.shape[0])
    n_actions_tm3 = int(h3.shape[0])
    linear_count = 0
    for k in state_dict:
        m = re.match(r"^backbone\.backbone\.(\d+)\.weight$", k)
        if m and int(m.group(1)) % 2 == 0:
            linear_count += 1
    n_layers = linear_count + 1
    return n_obs, n_hidden, n_actions_tm2, n_actions_tm3, n_layers


def _build_concurrent_policy(
    env: Env_PN_Concurrent,
    model_path: Path,
    device: torch.device,
) -> _DualActionPolicyModule:
    raw_state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    n_obs, n_hidden, n_actions_tm2, n_actions_tm3, n_layers = _infer_dual_head_shape(raw_state_dict)
    if n_obs != int(env.observation_spec["observation"].shape[0]):
        raise ValueError(f"权重 n_obs={n_obs} 与 Env 观测维 {env.observation_spec['observation'].shape[0]} 不一致")
    if n_actions_tm2 != int(env.n_actions_tm2) or n_actions_tm3 != int(env.n_actions_tm3):
        raise ValueError(
            f"权重动作维 ({n_actions_tm2},{n_actions_tm3}) 与 Env ({env.n_actions_tm2},{env.n_actions_tm3}) 不一致"
        )
    backbone = DualHeadPolicyNet(
        n_obs=n_obs,
        n_hidden=n_hidden,
        n_actions_tm2=n_actions_tm2,
        n_actions_tm3=n_actions_tm3,
        n_layers=n_layers,
    ).to(device)
    policy_module = _DualActionPolicyModule(backbone).to(device)
    policy_module.load_state_dict(raw_state_dict)
    policy_module.eval()
    return policy_module


def _decode_concurrent_actions(env: Env_PN_Concurrent, a_tm2: int, a_tm3: int) -> tuple[str, str]:
    if int(a_tm2) == int(env.tm2_wait_action):
        n2 = "WAIT"
    else:
        n2 = str(env.net.id2t_name[int(env.tm2_transition_indices[int(a_tm2)])])
    if int(a_tm3) == int(env.tm3_wait_action):
        n3 = "WAIT"
    else:
        n3 = str(env.net.id2t_name[int(env.tm3_transition_indices[int(a_tm3)])])
    return n2, n3


def _rollout_concurrent_sequence(
    model_path: Path,
    seed: int,
    exploration: str = "mode",
) -> tuple[list[dict[str, Any]], bool, dict[str, Any], dict[str, Any], Env_PN_Concurrent]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    env = Env_PN_Concurrent(device="cpu")
    env.net.eval()
    policy_module = _build_concurrent_policy(env, model_path=model_path, device=device)

    td = env.reset()
    sequence: list[dict[str, Any]] = []
    finished = False
    scrap_steps: list[int] = []
    release_steps: list[int] = []
    idle_steps: list[int] = []

    with torch.no_grad():
        for step in range(1, MAX_STEPS + 1):
            obs_f = td["observation"].unsqueeze(0).to(device).float()
            mask_tm2 = td["action_mask_tm2"].unsqueeze(0).to(device).bool()
            mask_tm3 = td["action_mask_tm3"].unsqueeze(0).to(device).bool()

            out = policy_module(obs_f)
            logits_tm2 = out["logits_tm2"]
            logits_tm3 = out["logits_tm3"]
            if exploration == "mode":
                m2 = logits_tm2.masked_fill(~mask_tm2, -1e9)
                m3 = logits_tm3.masked_fill(~mask_tm3, -1e9)
                a_tm2 = torch.argmax(m2, dim=-1)
                a_tm3 = torch.argmax(m3, dim=-1)
            else:
                masked_logits_tm2 = logits_tm2.masked_fill(~mask_tm2, -1e9)
                masked_logits_tm3 = logits_tm3.masked_fill(~mask_tm3, -1e9)
                probs2 = torch.softmax(masked_logits_tm2, dim=-1)
                probs3 = torch.softmax(masked_logits_tm3, dim=-1)
                a_tm2 = torch.multinomial(probs2, 1).squeeze(-1)
                a_tm3 = torch.multinomial(probs3, 1).squeeze(-1)

            i2 = int(a_tm2.squeeze(0).item())
            i3 = int(a_tm3.squeeze(0).item())
            n2, n3 = _decode_concurrent_actions(env, i2, i3)

            step_td = td.clone()
            step_td["action_tm2"] = torch.tensor(i2, dtype=torch.int64)
            step_td["action_tm3"] = torch.tensor(i3, dtype=torch.int64)
            td_next = env.step(step_td)

            if _to_scrap(td_next):
                scrap_steps.append(step)

            td_after = _to_step_state(td_next)
            current_time = _to_time(td_after)
            terminated = _to_terminated(td_next)
            finished = _to_finish(td_next)

            sequence.append(
                {
                    "step": step,
                    "time": current_time,
                    "actions": [n2, n3],
                    "action_tm2": i2,
                    "action_tm3": i3,
                }
            )

            if terminated:
                break
            td = td_after

    reward_report = {
        "scrap_penalty": {"count": len(scrap_steps), "steps": scrap_steps},
        "release_penalty": {"count": len(release_steps), "steps": release_steps},
        "idle_timeout_penalty": {"count": len(idle_steps), "steps": idle_steps},
    }
    replay_env_overrides = {
        "process_time_map": dict(getattr(env.net, "_base_proc_time_map", {})),
        "robot_capacity": 1,
        "device_mode": "concurrent",
        "single_route_name": getattr(env.net, "single_route_name", None),
    }
    single_route_cfg = getattr(env.net, "single_route_config", None)
    if single_route_cfg is not None:
        replay_env_overrides["single_route_config"] = dict(single_route_cfg)
    return sequence, finished, replay_env_overrides, reward_report, env


def _rollout_concurrent_sequence_with_retry(
    model_path: Path,
    seed: int,
    max_retries: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool, Env_PN_Concurrent]:
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")

    last_sequence: list[dict[str, Any]] = []
    last_overrides: dict[str, Any] = {}
    last_report: dict[str, Any] = {}
    last_finished = False
    last_env: Env_PN_Concurrent | None = None
    for attempt in range(max_retries):
        attempt_seed = seed + attempt
        sequence, finished, replay_env_overrides, reward_report, env = _rollout_concurrent_sequence(
            model_path=model_path,
            seed=attempt_seed,
            exploration="mode" if attempt == 0 else "random",
        )
        last_sequence = sequence
        last_overrides = replay_env_overrides
        last_report = reward_report
        last_finished = finished
        last_env = env
        if finished:
            if attempt > 0:
                print(f"[INFO] concurrent 模式第 {attempt + 1} 次推理达到 finish。")
            return sequence, replay_env_overrides, reward_report, finished, env

    print(f"[WARN] concurrent 模式重试 {max_retries} 次后仍未 finish，导出最后一次序列。")
    assert last_env is not None
    return last_sequence, last_overrides, last_report, last_finished, last_env


def rollout_and_export(
    model_path: Path,
    seed: int,
    out_name: str,
    concurrent: bool,
    retry: int,
    gantt_png_path: Path | None = None,
    gantt_title_suffix: str | None = None,
) -> dict[str, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if concurrent:
        sequence, replay_env_overrides, reward_report, _finished, env = _rollout_concurrent_sequence_with_retry(
            model_path=model_path,
            seed=seed,
            max_retries=retry,
        )
        payload: dict[str, Any] = {
            "reward_report": reward_report,
            "schema_version": 2,
            "device_mode": "concurrent",
            "sequence": sequence,
            "replay_env_overrides": replay_env_overrides,
        }
    else:
        sequence, replay_env_overrides, reward_report, _finished, env = _rollout_single_sequence_with_retry(
            model_path=model_path,
            seed=seed,
            max_retries=retry,
        )
        sequence_dual = []
        for item in sequence:
            action_name = item.get("action")
            sequence_dual.append(
                {
                    "step": item.get("step"),
                    "time": item.get("time"),
                    "actions": [action_name, "WAIT"],
                }
            )
        payload = {
            "reward_report": reward_report,
            "schema_version": 2,
            "device_mode": "cascade",
            "sequence": sequence_dual,
            "replay_env_overrides": replay_env_overrides,
        }

    action_series_path = _action_sequence_export_path(out_name, env)

    with action_series_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if gantt_png_path is not None:
        env.net.render_gantt(str(gantt_png_path), title_suffix=gantt_title_suffix)

    return {"action_series_path": action_series_path}


def main() -> None:
    default_model = model_output_path("CT_single_best.pt")
    parser = argparse.ArgumentParser(description="导出推理动作序列（默认级联；--concurrent 为双头并发模型）")
    parser.add_argument("--model", type=Path, default=default_model, help="模型权重路径")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--out-name",
        type=str,
        default="tmp",
        help="JSON 主文件名前缀（不含 .json）；实际文件名为 <前缀>(W晶圆数-M时刻).json",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="使用 DualHeadPolicyNet（Env_PN_Concurrent）；默认使用级联 MaskedPolicyHead",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=10,
        help="未 finish 时的最大重试次数",
    )
    args = parser.parse_args()
    out_name = args.out_name
    raw_model = args.model
    cand = Path(raw_model)
    if cand.is_file():
        model_path = cand.resolve()
    else:
        model_path = model_output_path(str(raw_model)).resolve()
    out = rollout_and_export(
        model_path=model_path,
        seed=args.seed,
        out_name=out_name,
        concurrent=bool(args.concurrent),
        retry=int(args.retry),
    )

    print(f"[INFO] 已导出 action_series: {out['action_series_path']}")


if __name__ == "__main__":
    main()
