from coc_trace_rl.src.coc_trace_schema import (
    build_guidance_variants,
    parse_trace_schema,
    should_trigger_coc_trace,
)
from coc_trace_rl.src.driving_dataset import prepare_jsonl

import json


def test_guidance_has_three_levels_and_omits_actions():
    reference = parse_trace_schema('左前方遮挡形成盲区，影响可见性和安全裕度，应减速谨慎通过。')
    candidate = parse_trace_schema('前方有车辆。')

    variants = build_guidance_variants(reference, [candidate])

    assert [item.level for item in variants] == [1, 2, 3]
    assert all('减速' not in item.text for item in variants)
    assert all('换道' not in item.text for item in variants)


def test_trigger_requires_flag_threshold_and_probability():
    params = dict(
        group_scores=[0.2, 0.7],
        threshold=0.8,
        probability=1.0,
        prompt_id='scene-1',
        global_step=4,
        seed=9,
    )

    assert should_trigger_coc_trace(enabled=True, **params)
    assert not should_trigger_coc_trace(enabled=False, **params)
    assert not should_trigger_coc_trace(enabled=True, group_scores=[0.8], **{
        key: value for key, value in params.items() if key != 'group_scores'
    })


def test_prepare_jsonl_persists_schema(tmp_path):
    source = tmp_path / 'source.jsonl'
    target = tmp_path / 'prepared.jsonl'
    source.write_text('{"think":"前方盲区影响可见性。"}\n', encoding='utf-8')

    assert prepare_jsonl(source, target) == 1

    row = json.loads(target.read_text(encoding='utf-8'))
    assert row['coc_trace_schema']['task_subcategory'] == '盲区'
