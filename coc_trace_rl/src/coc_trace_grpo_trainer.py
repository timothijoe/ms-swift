from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Sequence

import torch
from accelerate.utils import gather_object

from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
from swift.trainers.rlhf_trainer.utils import get_even_process_data

from .coc_trace_schema import (
    TraceSchema,
    build_guidance_variants,
    parse_trace_schema,
    should_trigger_coc_trace,
)


class CocTraceGRPOTrainer(GRPOTrainer):
    """GRPO with schema-driven CoC guidance for low-quality reasoning groups."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.coc_trace_enabled = bool(kwargs.pop('coc_trace_enabled', False))
        self.coc_trace_reward_threshold = float(kwargs.pop('coc_trace_reward_threshold', 0.8))
        self.coc_trace_probability = float(kwargs.pop('coc_trace_probability', 0.3))
        self.coc_trace_num_generations = int(kwargs.pop('coc_trace_num_generations', 3))
        self.coc_trace_seed = int(kwargs.pop('coc_trace_seed', 42))
        super().__init__(*args, **kwargs)
        if self.coc_trace_enabled and self.coc_trace_num_generations < 3:
            raise ValueError('coc_trace_num_generations must be at least 3 when CoC Trace is enabled')

    def _generate_and_score_completions(self, inputs):
        if not self.coc_trace_enabled:
            return super()._generate_and_score_completions(inputs)

        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)
        ordinary_inputs = self._generate_completions(inputs)
        ordinary_local_rewards, rewards_per_func = self._score_local_and_total(ordinary_inputs)
        guided_inputs = self._build_guided_inputs(ordinary_inputs, ordinary_local_rewards)
        if guided_inputs:
            guided_inputs = self._generate_completions(guided_inputs)
            self._restore_plain_prompts(guided_inputs)
            guided_local_rewards, _ = self._score_local_and_total(guided_inputs)
            ordinary_inputs, local_rewards = self._replace_group_worst(
                ordinary_inputs, ordinary_local_rewards, guided_inputs, guided_local_rewards)
            rewards_per_func = self._gather_rewards(local_rewards)
            self._metrics['train']['coc_trace/guided_rollouts'].append(float(len(guided_inputs)))

        # Use dynamic_num_samples for advantage computation, but restore afterwards
        # so the next call to _generate_completions uses the normal (non-dynamic) path.
        orig_dynamic = self.dynamic_num_samples
        self.dynamic_num_samples = True
        try:
            batches = self._prepare_batch_inputs(ordinary_inputs)
            advantages = self._compute_advantages(ordinary_inputs, rewards_per_func, batches)
        finally:
            self.dynamic_num_samples = orig_dynamic
        local_advantages = get_even_process_data(self, advantages)
        for item, advantage in zip(ordinary_inputs, local_advantages):
            item['advantages'] = advantage
        self._logs['advantages'].extend(advantages.tolist())
        for batch, encoded in zip(self.split_by_mini_batches(ordinary_inputs), batches):
            encoded['advantages'] = torch.stack([item['advantages'] for item in batch])
        return batches

    def _score_local_and_total(self, inputs):
        local_rewards = self._compute_rewards_per_func(inputs)
        return local_rewards, self._gather_rewards(local_rewards)

    def _gather_rewards(self, local_rewards: torch.Tensor) -> torch.Tensor:
        rows = gather_object([row.tolist() for row in local_rewards])
        return torch.tensor(rows, dtype=torch.float32, device=self.accelerator.device)

    def _replace_group_worst(self, ordinary_inputs, ordinary_rewards, guided_inputs, guided_rewards):
        coc_idx = self._coc_reward_index()
        ordinary_by_prompt = defaultdict(list)
        for index, item in enumerate(ordinary_inputs):
            ordinary_by_prompt[str(item['prompt_id'])].append(index)
        guided_by_prompt = defaultdict(list)
        for index, item in enumerate(guided_inputs):
            guided_by_prompt[str(item['_coc_source_prompt_id'])].append(index)

        merged_inputs = list(ordinary_inputs)
        merged_rewards = ordinary_rewards.clone()
        for prompt_id, guided_indices in guided_by_prompt.items():
            candidate_indices = ordinary_by_prompt[prompt_id]
            candidate_indices.sort(key=lambda index: float(ordinary_rewards[index, coc_idx].item()))
            for replacement_index, guided_index in zip(candidate_indices, guided_indices):
                guided = guided_inputs[guided_index]
                replaced = ordinary_inputs[replacement_index]
                guided['prompt_id'] = replaced['prompt_id']
                guided['request_id'] = replaced['request_id']
                merged_inputs[replacement_index] = guided
                merged_rewards[replacement_index] = guided_rewards[guided_index]
        return merged_inputs, merged_rewards

    def _build_guided_inputs(self, inputs: Sequence[dict[str, Any]], rewards: torch.Tensor) -> list[dict[str, Any]]:
        coc_index = self._coc_reward_index()
        grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
        for index, item in enumerate(inputs):
            score = float(rewards[index, coc_index].item())
            grouped[str(item['prompt_id'])].append((item, score))

        guided: list[dict[str, Any]] = []
        for prompt_id, group in grouped.items():
            plain = group[0][0]
            schema_data = plain.get('coc_trace_schema')
            if not isinstance(schema_data, dict):
                continue
            scores = [score for _, score in group]
            if not should_trigger_coc_trace(
                    enabled=True,
                    group_scores=scores,
                    threshold=self.coc_trace_reward_threshold,
                    probability=self.coc_trace_probability,
                    prompt_id=prompt_id,
                    global_step=self.state.global_step,
                    seed=self.coc_trace_seed):
                continue
            candidates = [parse_trace_schema(self._completion_text(item)) for item, _ in group]
            variants = build_guidance_variants(TraceSchema.from_dict(schema_data), candidates)
            num_guided = self.coc_trace_num_generations
            for offset in range(num_guided):
                variant = variants[offset % len(variants)]
                guided.append(self._guided_copy(plain, variant.level, variant.text, offset))
        self._metrics['train']['coc_trace/triggered_prompts'].append(float(bool(guided)))
        return guided

    def _coc_reward_index(self) -> int:
        targets = ('coc_trace_score', 'CocTraceScoreReward')
        for index, name in enumerate(self.reward_func_names):
            if any(target in name for target in targets):
                return index
        raise ValueError(f'coc_trace_score not found in reward_func_names: {self.reward_func_names}')

    @staticmethod
    def _completion_text(item: dict[str, Any]) -> str:
        return str(item['messages'][-1].get('content', ''))

    def _guided_copy(self, item: dict[str, Any], level: int, guidance: str, offset: int) -> dict[str, Any]:
        output = deepcopy(item)
        plain_messages = deepcopy(item['messages'][:-1])
        prompt_messages = deepcopy(plain_messages)
        for message in reversed(prompt_messages):
            if message.get('role') == 'user':
                message['content'] = f"{message.get('content', '')}\n\n[推理提示]\n{guidance}"
                break
        else:
            raise ValueError('CoC guidance requires a user message')
        output['messages'] = prompt_messages
        output['_coc_plain_messages'] = plain_messages
        output['_coc_source_prompt_id'] = item['prompt_id']
        output['guidance_level'] = level
        output['request_id'] = f"{item['request_id']}:coc:{level}:{offset}"
        return output

    @staticmethod
    def _restore_plain_prompts(inputs: Sequence[dict[str, Any]]) -> None:
        for item in inputs:
            response = deepcopy(item['messages'][-1])
            item['messages'] = deepcopy(item.pop('_coc_plain_messages')) + [response]
