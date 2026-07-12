from types import SimpleNamespace

import pytest
import torch

from coc_trace_rl.src.coc_trace_grpo_trainer import CocTraceGRPOTrainer


def test_replace_guided_samples_within_their_original_prompt_group():
    trainer = object.__new__(CocTraceGRPOTrainer)
    trainer._coc_reward_index = lambda: 1
    ordinary = [
        {'prompt_id': 'a', 'request_id': 'a-0'},
        {'prompt_id': 'a', 'request_id': 'a-1'},
        {'prompt_id': 'b', 'request_id': 'b-0'},
        {'prompt_id': 'b', 'request_id': 'b-1'},
    ]
    guided = [
        {'_coc_source_prompt_id': 'a', 'request_id': 'guided-a'},
        {'_coc_source_prompt_id': 'b', 'request_id': 'guided-b'},
    ]
    ordinary_rewards = torch.tensor([[1.0, .9], [1.0, .1], [1.0, .8], [1.0, .2]])
    guided_rewards = torch.tensor([[1.0, .7], [1.0, .6]])

    samples, rewards = trainer._replace_group_worst(ordinary, ordinary_rewards, guided, guided_rewards)

    assert [sample['request_id'] for sample in samples] == ['a-0', 'a-1', 'b-0', 'b-1']
    assert samples[1]['_coc_source_prompt_id'] == 'a'
    assert samples[3]['_coc_source_prompt_id'] == 'b'
    assert rewards[:, 1].tolist() == pytest.approx([.9, .7, .8, .6])


def test_guided_samples_keep_a_source_prompt_id():
    trainer = object.__new__(CocTraceGRPOTrainer)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.coc_trace_reward_threshold = .8
    trainer.coc_trace_probability = 1.0
    trainer.coc_trace_seed = 1
    trainer.coc_trace_num_generations = 3
    trainer._metrics = {'train': {'coc_trace/triggered_prompts': []}}
    trainer._coc_reward_index = lambda: 0
    sample = {
        'prompt_id': 'a',
        'request_id': 'a-0',
        'coc_trace_schema': {'task_category': '交通参与者', 'task_subcategory': '盲区', 'evidence': [], 'risk_causes': [], 'constraints': []},
        'messages': [{'role': 'user', 'content': 'scene'}, {'role': 'assistant', 'content': '<think>bad</think>{}'}],
    }

    guided = trainer._build_guided_inputs([sample], torch.tensor([[0.0]]))

    assert len(guided) == 3
    assert {item['_coc_source_prompt_id'] for item in guided} == {'a'}
