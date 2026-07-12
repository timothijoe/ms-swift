from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .coc_trace_schema import TraceSchema, parse_trace_schema, score_trace_schema


def _extract_think(completion: Any) -> str:
    match = re.search(r'<think>(.*?)</think>', str(completion), flags=re.DOTALL)
    return match.group(1).strip() if match else ''


def _extract_decision(completion: Any) -> dict[str, str]:
    match = re.search(r'\{.*\}', str(completion), flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {key: str(value[key]) for key in ('横向决策', '纵向决策') if key in value}


class CocTraceScoreReward:
    def __call__(self, completions: Sequence[Any], coc_trace_schema: Sequence[dict[str, object]], **_: Any) -> list[float]:
        scores = []
        for completion, schema in zip(completions, coc_trace_schema):
            trace = _extract_think(completion)
            if not trace or not isinstance(schema, dict):
                scores.append(0.0)
                continue
            scores.append(score_trace_schema(TraceSchema.from_dict(schema), parse_trace_schema(trace)))
        return scores


class DrivingDecisionAccuracyReward:
    def __call__(self, completions: Sequence[Any], gt_answer: Sequence[dict[str, Any]], **_: Any) -> list[float]:
        scores = []
        for completion, target in zip(completions, gt_answer):
            decision = _extract_decision(completion)
            is_correct = all(decision.get(key) == str(target.get(key)) for key in ('横向决策', '纵向决策'))
            scores.append(float(is_correct))
        return scores


try:
    from swift.plugin import orms
except ModuleNotFoundError:
    pass
else:
    orms['driving_decision_accuracy'] = DrivingDecisionAccuracyReward
    orms['coc_trace_score'] = CocTraceScoreReward
