# Driving GRPO Clean Entrypoint

这个目录是整理后的独立入口，不修改旧的 `driving_utils/`、`my_project/`、`my_project2/` 和 `my_data/`。

## 文件职责

- `run_driving_grpo_manifest.sh`: 唯一推荐运行入口。
- `driving_manifest_main.py`: 训练主入口，负责读取 manifest、绑定 train/eval dataset、加载 reward。
- `driving_manifest_args.py`: GRPO 默认参数。
- `manifest_dataset_register.py`: manifest 注册和样本预处理。
- `driving_reward_funcs.py`: reward 注册，包含格式奖励和决策准确率奖励。
- `datasets_manifest.json`: 当前默认 no-think 的 train/eval 数据配置。

## 运行

```bash
bash driving_grpo_clean/run_driving_grpo_manifest.sh
```

可覆盖的常用环境变量：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash driving_grpo_clean/run_driving_grpo_manifest.sh
```

如果你的环境只有 `python` 没有 `python3`：

```bash
PYTHON=python bash driving_grpo_clean/run_driving_grpo_manifest.sh
```

## 当前默认策略

默认使用 no-think 数据和 reward：

- `preprocess_type=driving_no_think`
- `reward_funcs=driving_no_think_format driving_decision_accuracy`

这样不会出现 mixed prompt 要求输出 `<think>`，但 reward 又惩罚 `<think>` 的冲突。

## 路径处理

`datasets_manifest.json` 使用相对路径 `my_data/driving_sample_32.jsonl`。样本 JSONL 中如果还有旧路径 `/home/linux/project/v3.12-ms-swift/...`，预处理器会在读取样本时映射到当前仓库根目录。

## 如果要跑 mixed

把 `datasets_manifest.json` 里的 `preprocess_type` 改成 `driving_mixed`，并把脚本里的 reward 改成：

```bash
--reward_funcs driving_mixed_format driving_decision_accuracy
```
