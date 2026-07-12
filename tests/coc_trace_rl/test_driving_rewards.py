from coc_trace_rl.src.driving_rewards import CocTraceScoreReward, DrivingDecisionAccuracyReward
from coc_trace_rl.src.coc_trace_schema import parse_trace_schema


def test_coc_score_uses_only_think_and_persisted_schema():
    schema = parse_trace_schema('前方盲区影响可见性。').to_dict()
    reward = CocTraceScoreReward()

    scores = reward(
        ['<think>前方盲区导致可见性受限。</think>{"横向决策":"车道居中","纵向决策":"减速"}'],
        coc_trace_schema=[schema],
    )

    assert scores == [1.0]


def test_decision_reward_requires_both_decisions():
    reward = DrivingDecisionAccuracyReward()

    scores = reward(
        ['{"横向决策":"车道居中","纵向决策":"减速"}', '{"横向决策":"车道居中"}'],
        gt_answer=[{'横向决策': '车道居中', '纵向决策': '减速'}, {'横向决策': '车道居中', '纵向决策': '减速'}],
    )

    assert scores == [1.0, 0.0]
