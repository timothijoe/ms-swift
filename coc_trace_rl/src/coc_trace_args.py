from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from swift.llm.argument.rlhf_args import RLHFArguments


@dataclass
class CocTraceDrivingArguments(RLHFArguments):
    dataset_manifest_path: str = 'coc_trace_rl/configs/driving_video_manifest.json'
    dataset: List[str] = field(default_factory=lambda: ['placeholder_dataset_for_init'])
    rlhf_type: str = 'grpo'
    coc_trace_enabled: bool = False
    coc_trace_reward_threshold: float = 0.8
    coc_trace_probability: float = 0.3
    coc_trace_num_generations: int = 3
    coc_trace_seed: int = 42
    coc_guidance_level_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    reward_funcs: Optional[List[str]] = field(
        default_factory=lambda: ['driving_decision_accuracy', 'coc_trace_score'])

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.coc_trace_reward_threshold <= 1.0:
            raise ValueError('coc_trace_reward_threshold must be in [0, 1]')
        if not 0.0 <= self.coc_trace_probability <= 1.0:
            raise ValueError('coc_trace_probability must be in [0, 1]')
        if self.coc_trace_enabled and self.coc_trace_num_generations < 3:
            raise ValueError('coc_trace_num_generations must be at least 3 when CoC Trace is enabled')
        if len(self.coc_guidance_level_weights) != 3 or any(weight < 0 for weight in self.coc_guidance_level_weights):
            raise ValueError('coc_guidance_level_weights must contain three non-negative values')
