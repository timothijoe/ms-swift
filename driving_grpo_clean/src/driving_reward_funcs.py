import json
import os
import re
from typing import Dict, List, Optional, Sequence

from swift.plugin import ORM, orms


# New canonical enums for the latest dataset format.
HORIZONTAL_ENUM = {'左换道', '右换道', '左避让', '右避让', '车道居中', '路口掉头', '路口左转', '路口顺行', '路口右转'}
VERTICAL_ENUM = {'保持', '加速', '减速', '停车'}
HORIZONTAL_ALIASES = {
    # Legacy to new format
    '直行': '车道居中',
    '保持': '车道居中',
    '向左转向': '左换道',
    '向右转向': '右换道',
    '左避障': '左避让',
    '右避障': '右避让',
    '左侧绕行': '左避让',
    '右侧绕行': '右避让',
    '左绕': '左避让',
    '右绕': '右避让',
    '左变道': '左换道',
    '右变道': '右换道',
    '右转': '路口右转',
    '左转': '路口左转',
}
VERTICAL_ALIASES = {
    # Legacy to new format
    '停车等待': '停车',
    '刹停': '停车',
    '停车': '停车',
    '让行': '减速',
    '跟车': '保持',
    '起步': '加速',
    '急加速': '加速',
    '急减速': '减速',
    '蠕行': '减速',
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

    def __call__(self, completions, label=None, gt_answer=None, data_type=None, **kwargs) -> List[Optional[float]]:
        targets = _as_list(gt_answer if gt_answer is not None else label, len(completions))
        dtypes = _as_list(data_type, len(completions))
        rewards = []
        for i, (completion, target, dtype) in enumerate(zip(completions, targets, dtypes)):
            dtype = dtype or 'driving_decision'
            if not _is_supported_dtype(dtype, self.supported_data_types):
                rewards.append(None)
                continue

            pred = _extract_json(completion)
            tgt = target if isinstance(target, dict) else _normalize_target(target)
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
