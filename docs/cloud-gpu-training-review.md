# 云端 GPU 训练代码审查

针对 `pn_single.py`、`train_single.py`、`env_single.py` 在云端 GPU 上训练时的审查结论与修改建议。

---

## 1. 设备与数据流

| 模块 | 设备 | 说明 |
|------|------|------|
| `pn_single.py` (ClusterTool) | CPU / NumPy | 无 GPU 依赖，状态与观测均为 NumPy，适合保持现状。 |
| `env_single.py` (Env_PN_Single) | **必须 CPU** | `_step` 使用 `torch.from_numpy()`，输出均在 CPU。训练时应显式 `device="cpu"`，避免与 spec 的 device 混淆。 |
| `train_single.py` | 由 config/CLI 决定 | 策略与 value 在 `config.device`（可为 `cuda`/`cuda:0`）；rollout 在 CPU 采集后 `rollout.to(device)` 再计算，数据流正确。 |

**结论**：策略与价值网络可放在 GPU，环境与 rollout 采集保持在 CPU，当前设计合理。需保证 env 构造时始终使用 `device="cpu"`。

---

## 2. 已检查无问题的点

- **checkpoint 加载**：`torch.load(..., map_location=device, weights_only=True)` 正确，GPU 训练时权重会落到当前 device。
- **rollout 写入**：预分配 buffer 在 CPU，`collect_rollout_single` 中 `.cpu()` 仅在非 CPU 时调用，逻辑正确。
- **TensorDict.to(device)**：`rollout.to(device)` 会把整批数据搬到 GPU，PPO 更新在 GPU 上执行，无跨设备误用。
- **env_single 常量**：`_TRUE_T` / `_FALSE_T` 在 CPU，仅用于 env 返回的 TensorDict，之后被 collect 写入 CPU buffer，无 device 冲突。

---

## 3. 需修改或建议（面向云端 GPU）

### 3.1 训练设备与 env 设备（必做）

- **CLI 未暴露计算设备**：当前仅能从 `s_train.json` 的 `device` 改为 `cuda` 或 `cuda:0`。建议在 `train_single.py` 中增加 `--device` 或 `--gpu` 参数，便于在云端用 `python -m ... --device cuda` 启动，无需改 JSON。
- **env 设备显式固定**：在 `train_single` 中创建 env 时显式传入 `device="cpu"`，避免日后误传 `config.device` 导致 env 与 spec 不一致。

### 3.2 随机种子（建议）

- 使用 GPU 时，除 `torch.manual_seed(config.seed)` 外，建议增加 `torch.cuda.manual_seed_all(config.seed)`，以便多 GPU 或 CUDA 层面复现。

### 3.3 日志与缓冲（建议）

- 云端常通过 stdout 采集日志，建议对关键 `print` 使用 `flush=True`，避免缓冲导致日志延迟或丢失。
- 若有统一日志系统（如 TensorBoard），可后续将 `log` 字典中的指标写入，便于监控与对比。

### 3.4 路径与 checkpoint（按需）

- `best_model_path` 与 `saved_models_dir` 当前为基于项目路径的固定位置。云端若需写入挂载卷或对象存储，可增加 `--output-dir` 或从环境变量读取输出根目录，再在此目录下建 `models/`、`saved_models/`。
- 若训练时长较大，可考虑每 N 个 batch 保存一次 checkpoint（例如 `{output_dir}/checkpoint_latest.pt`），便于断点续训或故障恢复。

### 3.5 显存与精度（按需）

- 当前无混合精度与梯度累积。若 batch 较大导致 OOM，可考虑：减小 `sub_batch_size` / `frames_per_batch`，或后续引入 `torch.cuda.amp` 与梯度累积。
- 当前实现未使用多 GPU；若需多卡，需在数据或模型层面做分布式封装，超出本次单卡审查范围。

---

## 4. 修改清单（本次实施）

1. **train_single.py**
   - 增加 `--device` 参数，覆盖 `config.device`，支持 `cuda`、`cuda:0`、`cpu`。
   - 创建 env 时显式传入 `device="cpu"`。
   - 当 `device` 为 cuda 时调用 `torch.cuda.manual_seed_all(config.seed)`。
   - 关键 `print` 增加 `flush=True`（batch 日志与训练结束汇总）。
2. **env_single.py**
   - 在类或 `__init__` 文档中说明：训练时推荐/约定使用 `device="cpu"`，因 env 内部为 NumPy/CPU。

---

## 5. 使用示例（修改后）

```bash
# 本地 CPU
python -m solutions.Continuous_model.train_single

# 云端 GPU（单卡）
python -m solutions.Continuous_model.train_single --device cuda

# 指定 GPU 与 checkpoint
python -m solutions.Continuous_model.train_single --device cuda:0 --checkpoint best.pt
```

配置文件中保留 `"device": "cpu"` 时，仍可通过命令行 `--device cuda` 覆盖为 GPU，便于同一份配置在本地与云端复用。
