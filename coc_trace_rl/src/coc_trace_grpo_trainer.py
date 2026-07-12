from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Sequence

import torch

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
        if self.coc_trace_enabled and self.coc_trace_num_generations < 1:
            raise ValueError('coc_trace_num_generations must be at least 1 when CoC Trace is enabled')

    def _generate_and_score_completions(self, inputs):
        if not self.coc_trace_enabled:
            return super()._generate_and_score_completions(inputs)

        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)
        ordinary_inputs = self._generate_completions(inputs)
        ordinary_rewards = self._score_completions(ordinary_inputs)
        guided_inputs = self._build_guided_inputs(ordinary_inputs, ordinary_rewards)
        rewards_per_func = ordinary_rewards
        if guided_inputs:
            # Guided inputs reuse the first ordinary sample's completion (assistant response).
            guided_inputs = self._populate_guided_completions(guided_inputs, ordinary_inputs)
            self._restore_plain_prompts(guided_inputs)
            guided_rewards = self._score_completions(guided_inputs)
            # Replace the worst ordinary samples (lowest coc_trace_score) with guided samples.
            coc_idx = self._coc_reward_index()
            ordinary_scores = [(i, float(ordinary_rewards[i, coc_idx].item())) for i in range(len(ordinary_inputs))]
            ordinary_scores.sort(key=lambda x: x[1])  # ascending: worst first
            num_replace = len(guided_inputs)
            replace_indices = {idx for idx, _ in ordinary_scores[:num_replace]}
            kept_indices = [i for i in range(len(ordinary_inputs)) if i not in replace_indices]
            # Borrow prompt_id/request_id from replaced samples so the group advantage
            # computation sees guided samples as part of the same prompt group.
            replaced_items = [ordinary_inputs[i] for i, _ in ordinary_scores[:num_replace]]
            for guided_item, replaced_item in zip(guided_inputs, replaced_items):
                guided_item['prompt_id'] = replaced_item.get('prompt_id')
                guided_item['request_id'] = replaced_item.get('request_id')
            ordinary_inputs = [ordinary_inputs[i] for i in kept_indices] + guided_inputs
            rewards_per_func = torch.cat(
                (ordinary_rewards[[i for i in kept_indices]], guided_rewards), dim=0)
            self._metrics['train']['coc_trace/guided_rollouts'].append(float(len(guided_inputs)))

        # Use dynamic_num_samples for advantage computation, but restore afterwards
        # so the next call to _generate_completions uses the normal (non-dynamic) path.
        orig_dynamic = self.dynamic_num_samples
        self.dynamic_num_samples = True
        batches = self._prepare_batch_inputs(ordinary_inputs)
        advantages = self._compute_advantages(ordinary_inputs, rewards_per_func, batches)
        self.dynamic_num_samples = orig_dynamic
        local_advantages = get_even_process_data(self, advantages)
        for item, advantage in zip(ordinary_inputs, local_advantages):
            item['advantages'] = advantage
        self._logs['advantages'].extend(advantages.tolist())
        for batch, encoded in zip(self.split_by_mini_batches(ordinary_inputs), batches):
            encoded['advantages'] = torch.stack([item['advantages'] for item in batch])
        return batches

    @staticmethod
    def _populate_guided_completions(guided_inputs, ordinary_inputs):
        """Copy completions from the first ordinary sample into guided inputs.

        The guided copy only modifies the prompt — we attach an existing completion
        so we can re-score without a second inference call.
        """
        if not ordinary_inputs:
            return guided_inputs
        src = ordinary_inputs[0]
        src_messages = src.get('messages', [])
        src_assistant = deepcopy(src_messages[-1]) if src_messages and src_messages[-1].get('role') == 'assistant' else None
        for guided in guided_inputs:
            guided_messages = guided.get('messages', [])
            if src_assistant and (not guided_messages or guided_messages[-1].get('role') != 'assistant'):
                guided_messages = guided_messages + [deepcopy(src_assistant)]
                guided['messages'] = guided_messages
            for key in ('rollout_logprobs', 'is_truncated', 'finish_reason', 'add_eos'):
                if key in src:
                    guided[key] = deepcopy(src[key])
        return guided_inputs

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
            num_guided = min(self.coc_trace_num_generations, len(variants), len(group))
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
        output['guidance_level'] = level
        output['request_id'] = f"{item['request_id']}:coc:{level}:{offset}"
        return output

    @staticmethod
    def _restore_plain_prompts(inputs: Sequence[dict[str, Any]]) -> None:
        for item in inputs:
            response = deepcopy(item['messages'][-1])
            item['messages'] = deepcopy(item.pop('_coc_plain_messages')) + [response]
