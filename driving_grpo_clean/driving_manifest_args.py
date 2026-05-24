import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm.argument.rlhf_args import RLHFArguments


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class DrivingManifestRLHFArguments(RLHFArguments):
    """Driving GRPO arguments for the cleaned manifest entrypoint."""

    dataset_manifest_path: str = os.path.join(project_root(), 'driving_grpo_clean', 'datasets_manifest.json')
    manifest_register_file: str = os.path.join(project_root(), 'driving_grpo_clean', 'manifest_dataset_register.py')
    dataset_name: Optional[str] = None
    use_manifest_as_dataset: bool = True
    reward_mode: Literal['no_think', 'mixed'] = 'no_think'
    dataset: List[str] = field(default_factory=lambda: ['placeholder_dataset_for_init'])

    driving_think_ratio: float = 0.3
    driving_think_seed: int = 42

    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'] = 'grpo'
    model: str = 'Qwen/Qwen3-VL-2B-Instruct'
    tuner_type: str = 'full'
    torch_dtype: str = 'bfloat16'
    load_from_cache_file: bool = True
    max_length: int = 1024
    max_completion_length: int = 1024
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-7
    gradient_accumulation_steps: int = 4
    eval_steps: int = 8
    save_steps: int = 100
    save_total_limit: int = 10
    logging_steps: int = 10
    output_dir: str = os.path.join(project_root(), 'output', 'GRPO_DRIVING_MANIFEST_CLEAN')
    warmup_ratio: float = 0.01
    dataloader_num_workers: int = 2
    num_generations: int = 4
    temperature: float = 1.0
    beta: float = 0.1
    num_iterations: int = 1
    remove_unused_columns: bool = False

    inject_gt_on_all_wrong: bool = False
    inject_gt_reward_threshold: float = 0.0
