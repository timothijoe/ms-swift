import json
import os
import re
from typing import Dict, List, Optional, Sequence

from swift.plugin import ORM, orms


HORIZONTAL_ENUM = {'向左转向', '向左快速转向', '向右转向', '向右快速转向', '左倒车', '右倒车', '直行'}
VERTICAL_ENUM = {'加速', '急加速', '减速', '急减速', '倒车', '保持', '停车等待', '蠕行'}
HORIZONTAL_ALIASES = {
    '保持': '直行',
    '左避障': '向左转向',
    '左侧绕行': '向左转向',
    '左绕': '向左转向',
    '左变道': '向左转向',
    '右变道': '向右转向',
    '右转': '向右转向',
}
VERTICAL_ALIASES = {
    '刹停': '停车等待',
    '停车': '停车等待',
    '让行': '减速',
    '跟车': '保持',
    '起步': '加速',
}

_DEBUG = os.getenv('DRIVING_REWARD_DEBUG', '0') == '1'
_DEBUG_N = int(os.getenv('DRIVING_REWARD_DEBUG_N', '2'))


def _extract_json(text: str) -> Optional[Dict[str, str]]:
    if text is None:
        return None
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_target(target: str) -> Optional[Dict[str, str]]:
    if target is None:
        return None
    try:
        return json.loads(target)
    except Exception:
        return _extract_json(target)


def _extract_decision_fields(obj: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Support both flat format and nested answer format."""
    if not obj or not isinstance(obj, dict):
        return None
    if 'answer' in obj and isinstance(obj.get('answer'), dict):
        obj = obj['answer']
    horizontal = obj.get('横向决策')
    vertical = obj.get('纵向决策')
    if horizontal is None or vertical is None:
        return None
    horizontal = HORIZONTAL_ALIASES.get(horizontal, horizontal)
    vertical = VERTICAL_ALIASES.get(vertical, vertical)
    return {'横向决策': horizontal, '纵向决策': vertical}


def _requires_think(messages) -> bool:
    if messages is None or not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get('content', '')
        if '<think>' in content:
            return True
    return False


def _as_list(value, n: int):
    return value if isinstance(value, list) else [value] * n


def _is_supported_dtype(dtype: Optional[str], supported: Sequence[str]) -> bool:
    return not supported or dtype in supported


class DrivingNoThinkFormatReward(ORM):
    def __init__(self, supported_data_types: Optional[List[str]] = None):
        self.supported_data_types = supported_data_types or ['driving_decision']

    def __call__(self, completions, data_type=None, **kwargs) -> List[Optional[float]]:
        dtypes = _as_list(data_type, len(completions))
        rewards = []
        for i, (completion, dtype) in enumerate(zip(completions, dtypes)):
            dtype = dtype or 'driving_decision'
            if not _is_supported_dtype(dtype, self.supported_data_types):
                rewards.append(None)
                continue
            has_think = re.search(r'<think>.*?</think>', completion, flags=re.DOTALL) is not None
            reward = 0.0 if has_think else 1.0
            rewards.append(reward)
            if _DEBUG and i < _DEBUG_N:
                print(f'[DRIVING_REWARD] no_think_format dtype={dtype} reward={reward}')
        return rewards


class DrivingMixedFormatReward(ORM):
    def __init__(self, supported_data_types: Optional[List[str]] = None):
        self.supported_data_types = supported_data_types or ['driving_decision']

    def __call__(self, completions, messages=None, data_type=None, **kwargs) -> List[Optional[float]]:
        msg_list = _as_list(messages, len(completions))
        dtypes = _as_list(data_type, len(completions))
        rewards = []
        for i, (completion, msgs, dtype) in enumerate(zip(completions, msg_list, dtypes)):
            dtype = dtype or 'driving_decision'
            if not _is_supported_dtype(dtype, self.supported_data_types):
                rewards.append(None)
                continue
            need_think = _requires_think(msgs)
            has_think = re.search(r'<think>.*?</think>', completion, flags=re.DOTALL) is not None
            reward = 1.0 if need_think == has_think else 0.0
            rewards.append(reward)
            if _DEBUG and i < _DEBUG_N:
                print(f'[DRIVING_REWARD] mixed_format dtype={dtype} need_think={need_think} reward={reward}')
        return rewards


class DrivingDecisionAccuracyReward(ORM):
    def __init__(self, supported_data_types: Optional[List[str]] = None):
        self.supported_data_types = supported_data_types or ['driving_decision']

    def __call__(self, completions, label=None, data_type=None, **kwargs) -> List[Optional[float]]:
        targets = _as_list(label, len(completions))
        dtypes = _as_list(data_type, len(completions))
        rewards = []
        for i, (completion, target, dtype) in enumerate(zip(completions, targets, dtypes)):
            dtype = dtype or 'driving_decision'
            if not _is_supported_dtype(dtype, self.supported_data_types):
                rewards.append(None)
                continue

            pred = _extract_json(completion)
            tgt = _normalize_target(target)
            if not pred or not tgt:
                rewards.append(0.0)
                continue

            pred_decision = _extract_decision_fields(pred)
            tgt_decision = _extract_decision_fields(tgt)
            if not pred_decision or not tgt_decision:
                rewards.append(0.0)
                continue

            horizontal = pred_decision.get('横向决策')
            vertical = pred_decision.get('纵向决策')
            if horizontal not in HORIZONTAL_ENUM or vertical not in VERTICAL_ENUM:
                rewards.append(0.0)
                continue

            reward = 0.5 * ((horizontal == tgt_decision.get('横向决策')) + (vertical == tgt_decision.get('纵向决策')))
            rewards.append(reward)
            if _DEBUG and i < _DEBUG_N:
                print(f'[DRIVING_REWARD] accuracy dtype={dtype} reward={reward} pred={pred_decision} tgt={tgt_decision}')
        return rewards


orms['driving_no_think_format'] = DrivingNoThinkFormatReward
orms['driving_mixed_format'] = DrivingMixedFormatReward
orms['driving_decision_accuracy'] = DrivingDecisionAccuracyReward
