# Driving GRPO Clean Entrypoint

这个目录是整理后的独立入口，不修改旧的 `driving_utils/`、`my_project/`、`my_project2/` 和 `my_data/`。

## 文件职责

- `src/`: 训练与奖励相关 Python 代码（入口、参数、预处理、reward plugin）。
- `scripts/`: 运行脚本（仅 `.sh`）。
- `configs/`: 配置与模板（`json`/`txt`）。

## 运行

```bash
bash driving_grpo_clean/scripts/run_driving_grpo_manifest.sh
```

可覆盖的常用环境变量：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash driving_grpo_clean/scripts/run_driving_grpo_manifest.sh
```

如果你的环境只有 `python` 没有 `python3`：

```bash
PYTHON=python bash driving_grpo_clean/scripts/run_driving_grpo_manifest.sh
```

## 当前默认策略

默认使用 no-think 数据和 reward：

- `preprocess_type=driving_no_think`
- `reward_funcs=driving_no_think_format driving_decision_accuracy`

这样不会出现 mixed prompt 要求输出 `<think>`，但 reward 又惩罚 `<think>` 的冲突。

## 路径处理

`configs/datasets_manifest.json` 使用相对路径 `my_data/driving_sample_32.jsonl`。样本 JSONL 中如果还有旧路径 `/home/linux/project/v3.12-ms-swift/...`，预处理器会在读取样本时映射到当前仓库根目录。

## 如果要跑 mixed

把 `configs/datasets_manifest.json` 里的 `preprocess_type` 改成 `driving_mixed`，并把脚本里的 reward 改成：

```bash
--reward_funcs driving_mixed_format driving_decision_accuracy
```

## 接入小模型作为 Reward Model（Rubric 评分）

默认脚本不启用 RM。若设置环境变量 `DRIVING_RM_MODEL`，会自动追加：

- `--external_plugins driving_grpo_clean/src/driving_rm_plugin.py`
- `--reward_model <DRIVING_RM_MODEL>`
- `--reward_model_plugin driving_rubric_rm`
- `--reward_weights <w1> <w2> <w3>`（对应：格式奖励、准确率奖励、RM奖励）

示例：

```bash
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
DRIVING_REWARD_W1=0.3 DRIVING_REWARD_W2=0.3 DRIVING_REWARD_W3=0.4 \
bash driving_grpo_clean/scripts/run_driving_grpo_manifest.sh
```

可选：自定义 rubric（JSON 字符串）：

```bash
DRIVING_RM_RUBRIC_JSON='[
  {"name":"horizontal_decision","desc":"横向决策是否一致","weight":0.4},
  {"name":"vertical_decision","desc":"纵向决策是否一致","weight":0.4},
  {"name":"safety_actionability","desc":"动作是否安全且可执行","weight":0.2}
]' \
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash driving_grpo_clean/scripts/run_driving_grpo_manifest.sh
```
