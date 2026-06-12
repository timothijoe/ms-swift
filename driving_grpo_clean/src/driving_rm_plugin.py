import json
import os
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from swift.llm import PtEngine, RequestConfig
from swift.plugin import rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()


POSITIONS = {
    '左前方',
    '前方',
    '右前方',
    '左侧',
    '右侧',
    '左后方',
    '后方',
    '右后方',
    '当前车道',
    '路口',
    '左侧车道',
    '右侧车道',
}

CATEGORIES = {
    '盲区',
    '障碍物',
    '交通信号',
    '车辆行为',
    '路况',
    '交通管制',
    '跟车',
    '换道条件',
}

DETAIL_SCORE_VALUES = {0.0, 0.5, 1.0}
_SAVE_LOCK = threading.Lock()


EXTRACTION_SYSTEM_PROMPT = """
你是一个自动驾驶场景结构化标注助手。任务是把驾驶场景描述转换为 JSON。

核心原则：
- 只允许抽取原文中明确表达的信息。
- 严禁补充、推理、常识扩展或新增原文不存在的因素。
- 一个因素必须只对应一个位置、一个大类、一个细节、一个原文片段。
- 如果一句话中存在多个因素，必须拆分，禁止合并。

必须严格只输出 JSON，不要输出 markdown，不要输出 ```json。

输出结构：
{
  "因素": [
    {
      "位置": "",
      "大类": "",
      "细节": "",
      "原文片段": ""
    }
  ],
  "动作": {
    "横向决策": [],
    "纵向决策": [],
    "执行策略": ""
  }
}

位置只能从以下值中选择：
左前方、前方、右前方、左侧、右侧、左后方、后方、右后方、当前车道、路口、左侧车道、右侧车道

大类只能从以下值中选择：
盲区、障碍物、交通信号、车辆行为、路况、交通管制、跟车、换道条件

横向决策只能从以下值中选择：
换道、避让、保持、转弯
归一化规则：
- 左换道 / 右换道 / 变道 / 变更车道 / 并线 -> 换道
- 左避障 / 右避障 / 左避让 / 右避让 / 绕行避让 / 避障 -> 避让
- 左转 / 右转 / 掉头 / 转弯 -> 转弯
- 车道居中 / 直行 / 保持 / 未明确描述横向动作时 -> 保持

纵向决策只能从以下值中选择：
保持、加速、减速、停车
归一化规则：
- 减速 / 慢行 / 让行 / 蠕行 -> 减速
- 起步 / 跟车起步 / 加速 -> 加速
- 刹停 / 停车等待 / 停止 / 停车 -> 停车
- 未明确描述纵向动作时 -> 保持
- 连续动作，例如“减速后停车”，输出为 ["减速", "停车"]

执行策略只能从以下值中选择：
直接执行、条件满足后执行
判定规则：
- 直接执行：当前可以立即执行动作，例如减速通过盲区、积水、弯道或正常转弯。
- 条件满足后执行：需要等待车辆、行人、信号、交警指令或安全条件满足后再执行。
"""


DETAIL_SCORING_SYSTEM_PROMPT = """
你是一个自动驾驶场景语义细节评分模型。
只根据输入的 detail pair 给出相似度分数 score 和简短关系描述。
score 只能是 0.0、0.5、1.0。
必须严格只输出 JSON，不要输出 markdown，不要输出 ```json。
"""


def _safe_str(value: Any) -> str:
    return '' if value is None else str(value).strip()


def _safe_json_obj(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_list(value: Any, default: str = '保持') -> List[str]:
    if value is None or value == '':
        return [default]
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    out = [_safe_str(item) for item in values if _safe_str(item)]
    return out or [default]


def _normalize_lateral(value: Any) -> List[str]:
    out = []
    for item in _as_list(value, default='保持'):
        if any(k in item for k in ('换道', '变道', '变更车道', '并线')):
            out.append('换道')
        elif any(k in item for k in ('避让', '避障', '绕行')):
            out.append('避让')
        elif any(k in item for k in ('转弯', '左转', '右转', '掉头')):
            out.append('转弯')
        else:
            out.append('保持')
    return sorted(set(out)) or ['保持']


def _normalize_longitudinal(value: Any) -> List[str]:
    out = []
    for item in _as_list(value, default='保持'):
        if any(k in item for k in ('减速', '慢行', '让行', '蠕行')):
            out.append('减速')
        if any(k in item for k in ('停车', '刹停', '停止', '等待')):
            out.append('停车')
        if any(k in item for k in ('加速', '起步')):
            out.append('加速')
        if not out:
            out.append('保持')
    return sorted(set(out)) or ['保持']


def _normalize_strategy(value: Any, text: str = '') -> str:
    value = _safe_str(value)
    if value in {'直接执行', '条件满足后执行'}:
        return value
    conditional_keywords = ('等待', '让行', '信号', '行人', '无车', '安全', '条件', '切入', '变道', '交警')
    return '条件满足后执行' if any(keyword in text for keyword in conditional_keywords) else '直接执行'


def _normalize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    factors = []
    for factor in scene.get('因素', []) or []:
        if not isinstance(factor, dict):
            continue
        position = _safe_str(factor.get('位置'))
        category = _safe_str(factor.get('大类') or factor.get('事件类型'))
        detail = _safe_str(factor.get('细节') or factor.get('具体描述') or factor.get('concept'))
        fragment = _safe_str(factor.get('原文片段') or detail)
        if not position and not detail:
            continue
        factors.append({'位置': position or '前方', '大类': _normalize_category(category or detail), '细节': detail, '原文片段': fragment})

    action = scene.get('动作', {}) or {}
    text_for_strategy = json.dumps(scene, ensure_ascii=False)
    return {
        '因素': factors,
        '动作': {
            '横向决策': _normalize_lateral(action.get('横向决策') or action.get('横向动作')),
            '纵向决策': _normalize_longitudinal(action.get('纵向决策') or action.get('纵向动作')),
            '执行策略': _normalize_strategy(action.get('执行策略') or action.get('行为目标'), text_for_strategy),
        },
    }


def _normalize_category(text: str) -> str:
    text = _safe_str(text)
    if text in CATEGORIES:
        return text
    if any(k in text for k in ('盲区', '遮挡', '豁口')):
        return '盲区'
    if any(k in text for k in ('障碍', '占道', '停靠', '锥桶', '施工', '事故车')):
        return '障碍物'
    if any(k in text for k in ('信号', '红灯', '绿灯', '黄灯')):
        return '交通信号'
    if any(k in text for k in ('交警', '交通管制', '管制')):
        return '交通管制'
    if any(k in text for k in ('空闲', '无车', '无来车', '安全')):
        return '换道条件'
    if any(k in text for k in ('积水', '湿滑', '弯道', '上坡', '下坡', '坑洼', '井盖', '路面')):
        return '路况'
    if any(k in text for k in ('跟车', )):
        return '跟车'
    return '车辆行为'


def _infer_category(fragment: str) -> str:
    return _normalize_category(fragment)


def _extract_detail(fragment: str, position: str) -> str:
    detail = fragment.replace(position, '', 1)
    detail = re.sub(r'^(存在|有|一辆|一名|一个|的)', '', detail)
    detail = re.sub(r'(自车|应|应该|需要).*$','', detail)
    detail = re.sub(r'[，。；、\s]+', '', detail)
    return detail or fragment


def _infer_actions(summary: str) -> Dict[str, Any]:
    return {
        '横向决策': _normalize_lateral(summary),
        '纵向决策': _normalize_longitudinal(summary),
        '执行策略': _normalize_strategy('', summary),
    }


def _extract_scene_locally(summary: str) -> Dict[str, Any]:
    summary = _safe_str(summary)
    factors = []
    position_pattern = '|'.join(sorted((re.escape(item) for item in POSITIONS), key=len, reverse=True))
    last_position = ''
    for fragment in re.split(r'[，。；;]', summary):
        fragment = _safe_str(fragment)
        if not fragment or '自车' in fragment:
            continue
        match = re.search(position_pattern, fragment)
        if match is not None:
            position = match.group(0)
            last_position = position
        elif last_position:
            position = last_position
        else:
            continue
        factors.append({
            '位置': position,
            '大类': _infer_category(fragment),
            '细节': _extract_detail(fragment, position),
            '原文片段': fragment,
        })

    if not factors and summary:
        factors.append({
            '位置': '前方',
            '大类': _infer_category(summary),
            '细节': _extract_detail(summary, '前方'),
            '原文片段': summary,
        })
    return _normalize_scene({'因素': factors, '动作': _infer_actions(summary)})


def build_reference_rm_schema(reference: str) -> Dict[str, Any]:
    """Build the dataset-side rm_schema from one reference summary.

    This is intended for offline dataset preparation. The returned object is the
    canonical schema consumed by DrivingFormalRMPlugin, so it can be written
    directly to each JSONL sample as `rm_schema`.
    """
    return _extract_scene_locally(reference)


def _scene_from_rm_schema(rm_schema: Any, gt_answer: Any = None, reference_text: str = '') -> Optional[Dict[str, Any]]:
    if not isinstance(rm_schema, dict):
        return None

    if isinstance(rm_schema.get('因素'), list):
        scene = {'因素': rm_schema.get('因素') or [], '动作': rm_schema.get('动作') or {}}
        if not scene['动作'] and gt_answer:
            scene['动作'] = _action_from_answer(gt_answer, reference_text)
        return _normalize_scene(scene)

    factors = []
    for key in ('cause_groups', 'decision_groups'):
        for group in rm_schema.get(key, []) or []:
            if not isinstance(group, dict):
                continue
            concept = _safe_str(group.get('concept'))
            aliases = group.get('aliases') or []
            text = concept or (_safe_str(aliases[0]) if aliases else '')
            if not text:
                continue
            position = _first_position(text) or _first_position(' '.join(map(str, aliases))) or '前方'
            factors.append({
                '位置': position,
                '大类': _infer_category(text),
                '细节': text,
                '原文片段': text,
            })

    if not factors:
        return None
    return _normalize_scene({'因素': factors, '动作': _action_from_answer(gt_answer, reference_text)})


def _first_position(text: str) -> str:
    for position in sorted(POSITIONS, key=len, reverse=True):
        if position in text:
            return position
    return ''


def _action_from_answer(answer: Any, reference_text: str = '') -> Dict[str, Any]:
    obj = _safe_json_obj(answer) or (answer if isinstance(answer, dict) else {})
    if not isinstance(obj, dict):
        obj = {}
    if isinstance(obj.get('answer'), dict):
        obj = obj['answer']
    return {
        '横向决策': _normalize_lateral(obj.get('横向决策') or obj.get('横向动作') or reference_text),
        '纵向决策': _normalize_longitudinal(obj.get('纵向决策') or obj.get('纵向动作') or reference_text),
        '执行策略': _normalize_strategy(obj.get('执行策略') or obj.get('行为目标'), reference_text),
    }


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return default


def _strip_decision_json(content: str) -> str:
    """Return text before the final decision JSON to avoid leaking answer fields into RM scoring."""
    content = _safe_str(content)
    for match in re.finditer(r'\{.*?\}', content, flags=re.DOTALL):
        obj = _safe_json_obj(match.group(0))
        if not isinstance(obj, dict):
            continue
        answer_obj = obj.get('answer') if isinstance(obj.get('answer'), dict) else obj
        if isinstance(answer_obj, dict) and ('横向决策' in answer_obj or '纵向决策' in answer_obj):
            prefix = content[:match.start()].strip()
            return prefix or content.strip()
    return content.strip()


def _extract_pred_text(sample: Dict[str, Any]) -> str:
    messages = sample.get('messages') if isinstance(sample, dict) else None
    if not isinstance(messages, list) or not messages:
        return ''
    content = _safe_str(messages[-1].get('content') if isinstance(messages[-1], dict) else messages[-1])
    default_extract_think = _to_bool(os.getenv('DRIVING_FORMAL_RM_EXTRACT_THINK', '1'), default=True)
    extract_think = _to_bool(sample.get('rm_extract_think'), default=default_extract_think) if isinstance(sample, dict) else default_extract_think
    if not extract_think:
        return _strip_decision_json(content)
    match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    return match.group(1).strip() if match else content


def _extract_reference_text(sample: Dict[str, Any]) -> str:
    think = _safe_str(sample.get('think') or sample.get('gt_think'))
    if think:
        return think
    label = sample.get('label')
    label_obj = _safe_json_obj(label)
    if isinstance(label_obj, dict):
        return _safe_str(label_obj.get('think'))
    return _safe_str(label)


def _position_score(reference_pos: str, candidate_pos: str) -> float:
    reference_pos = _safe_str(reference_pos)
    candidate_pos = _safe_str(candidate_pos)
    if not reference_pos or not candidate_pos:
        return 0.0
    if reference_pos == candidate_pos:
        return 1.0
    direction_groups = [
        {'前方', '左前方', '右前方', '正前方'},
        {'左方', '左侧', '左侧车道', '左前方', '左后方'},
        {'右方', '右侧', '右侧车道', '右前方', '右后方'},
        {'后方', '左后方', '右后方', '正后方'},
    ]
    for group in direction_groups:
        if reference_pos in group and candidate_pos in group:
            return 0.5
    return 0.0


def _category_score(reference_category: str, candidate_category: str) -> float:
    return 1.0 if _normalize_category(reference_category) == _normalize_category(candidate_category) else 0.0


def _coarse_anchor_score(reference_factor: Dict[str, Any], candidate_factor: Dict[str, Any]) -> float:
    if _category_score(reference_factor.get('大类'), candidate_factor.get('大类')) <= 0:
        return 0.0
    return _position_score(reference_factor.get('位置'), candidate_factor.get('位置'))


def _detail_sort_score(reference_detail: str, candidate_detail: str) -> float:
    reference_detail = _safe_str(reference_detail)
    candidate_detail = _safe_str(candidate_detail)
    if not reference_detail or not candidate_detail:
        return 0.0
    if reference_detail == candidate_detail:
        return 1.0
    if reference_detail in candidate_detail or candidate_detail in reference_detail:
        return 0.7
    reference_chars = set(reference_detail)
    candidate_chars = set(candidate_detail)
    union = reference_chars | candidate_chars
    if not union:
        return 0.0
    return 0.3 if len(reference_chars & candidate_chars) / len(union) >= 0.4 else 0.0


def _match_factor_pairs_one_to_one(
    reference_factors: Sequence[Dict[str, Any]],
    candidate_factors: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranked_by_reference: List[List[Dict[str, Any]]] = [[] for _ in reference_factors]
    all_ranked_pairs = []
    for reference_index, reference_factor in enumerate(reference_factors):
        for candidate_index, candidate_factor in enumerate(candidate_factors):
            coarse_score = _coarse_anchor_score(reference_factor, candidate_factor)
            if coarse_score <= 0:
                continue
            sort_score = _detail_sort_score(reference_factor.get('细节'), candidate_factor.get('细节'))
            ranked_item = {
                'candidate_factor': candidate_factor,
                'coarse_score': coarse_score,
                'sort_score': sort_score,
                'rank': coarse_score * 10 + sort_score,
            }
            ranked_by_reference[reference_index].append(ranked_item)
            all_ranked_pairs.append({
                'reference_index': reference_index,
                'candidate_index': candidate_index,
                'reference_factor': reference_factor,
                'candidate_factor': candidate_factor,
                'coarse_score': coarse_score,
                'rank': ranked_item['rank'],
            })

    for ranked_candidates in ranked_by_reference:
        ranked_candidates.sort(key=lambda item: item['rank'], reverse=True)

    matched_reference_indexes = set()
    matched_candidate_indexes = set()
    matched_pairs: Dict[int, Dict[str, Any]] = {}
    all_ranked_pairs.sort(key=lambda item: item['rank'], reverse=True)
    for pair in all_ranked_pairs:
        reference_index = pair['reference_index']
        candidate_index = pair['candidate_index']
        if reference_index in matched_reference_indexes or candidate_index in matched_candidate_indexes:
            continue
        matched_reference_indexes.add(reference_index)
        matched_candidate_indexes.add(candidate_index)
        matched_pairs[reference_index] = {
            'reference_factor': pair['reference_factor'],
            'candidate_factor': pair['candidate_factor'],
            'coarse_score': pair['coarse_score'],
            'candidate_rank': ranked_by_reference[reference_index],
        }

    factor_pairs = []
    for reference_index, reference_factor in enumerate(reference_factors):
        factor_pairs.append(matched_pairs.get(reference_index, {
            'reference_factor': reference_factor,
            'candidate_factor': None,
            'coarse_score': 0.0,
            'candidate_rank': ranked_by_reference[reference_index],
        }))
    return factor_pairs


def _score_action_list(reference_list: Any, candidate_list: Any) -> float:
    reference_set = {_safe_str(item) for item in (reference_list or []) if _safe_str(item)}
    candidate_set = {_safe_str(item) for item in (candidate_list or []) if _safe_str(item)}
    if not reference_set or not candidate_set:
        return 0.0
    if reference_set == candidate_set:
        return 1.0
    if reference_set & candidate_set:
        return 0.5
    return 0.0


def _score_action_text(reference_text: Any, candidate_text: Any) -> float:
    reference_text = _safe_str(reference_text)
    candidate_text = _safe_str(candidate_text)
    if not reference_text or not candidate_text:
        return 0.0
    if reference_text == candidate_text:
        return 1.0
    if reference_text in candidate_text or candidate_text in reference_text:
        return 0.5
    return 0.0


def _score_actions(reference_action: Dict[str, Any], candidate_action: Dict[str, Any]) -> Dict[str, float]:
    return {
        'lat': _score_action_list(reference_action.get('横向决策'), candidate_action.get('横向决策')),
        'lon': _score_action_list(reference_action.get('纵向决策'), candidate_action.get('纵向决策')),
        'strategy': _score_action_text(reference_action.get('执行策略'), candidate_action.get('执行策略')),
    }


def _build_detail_pairs(factor_pairs: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    detail_pairs = []
    for pair in factor_pairs:
        candidate_factor = pair.get('candidate_factor')
        if candidate_factor is None:
            continue
        detail_pairs.append((_safe_str(pair.get('reference_factor', {}).get('细节')), _safe_str(candidate_factor.get('细节'))))
    return detail_pairs


def _extract_detail_scores(detail_score_result: Dict[str, Any], expected_len: int) -> List[float]:
    scores = []
    for item in detail_score_result.get('results', []) or []:
        try:
            score = float(item.get('score', 0.0))
        except Exception:
            score = 0.0
        if score not in DETAIL_SCORE_VALUES:
            if score >= 0.75:
                score = 1.0
            elif score >= 0.25:
                score = 0.5
            else:
                score = 0.0
        scores.append(score)
    if len(scores) < expected_len:
        scores.extend([0.0] * (expected_len - len(scores)))
    return scores[:expected_len]


def _score_detail_pairs_locally(detail_pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    results = []
    for reference_detail, candidate_detail in detail_pairs:
        score = _detail_sort_score(reference_detail, candidate_detail)
        if score >= 0.75:
            normalized_score = 1.0
            relation = '本地规则判断为同一细节'
        elif score >= 0.25:
            normalized_score = 0.5
            relation = '本地规则判断为同类相关'
        else:
            normalized_score = 0.0
            relation = '本地规则判断为不匹配'
        results.append({
            'reference_detail': reference_detail,
            'candidate_detail': candidate_detail,
            'relation': relation,
            'score': normalized_score,
        })
    return {'results': results}


def _merge_detail_scores(factor_pairs: Sequence[Dict[str, Any]], detail_scores: Sequence[float]) -> List[Dict[str, Any]]:
    merged = []
    score_index = 0
    for pair in factor_pairs:
        item = dict(pair)
        if item.get('candidate_factor') is None:
            item['fine_score'] = 0.0
        else:
            item['fine_score'] = float(detail_scores[score_index]) if score_index < len(detail_scores) else 0.0
            score_index += 1
        merged.append(item)
    return merged


def _calculate_final_result(
    factor_pairs: Sequence[Dict[str, Any]],
    action_score: Dict[str, float],
    coarse_weight: float = 0.5,
    fine_weight: float = 0.5,
) -> Dict[str, Any]:
    scored_pairs = []
    factor_scores = []
    for pair in factor_pairs:
        coarse_score = float(pair.get('coarse_score', 0.0))
        fine_score = float(pair.get('fine_score', 0.0))
        if pair.get('candidate_factor') is None:
            fine_score = 0.0
        factor_score = coarse_weight * coarse_score + fine_weight * fine_score
        item = dict(pair)
        item['combined_score'] = round(factor_score, 4)
        scored_pairs.append(item)
        factor_scores.append(factor_score)

    action_detail = {
        'lat': float(action_score.get('lat', 0.0)),
        'lon': float(action_score.get('lon', 0.0)),
        'strategy': float(action_score.get('strategy', 0.0)),
    }
    factor_total = sum(factor_scores)
    action_total = sum(action_detail.values())
    max_score = len(factor_scores) + 3
    normalized_total = (factor_total + action_total) / max_score if max_score else 0.0
    return {
        'factor_pairs': scored_pairs,
        'factor_scores': [round(score, 4) for score in factor_scores],
        'action_scores': action_detail,
        'final_reward': round(max(0.0, min(1.0, normalized_total)), 4),
    }


def _count_answer_chars(text: str) -> int:
    return len(re.sub(r'\s+', '', text or ''))


def _length_reward(text: str, min_chars: int, target_chars: int, max_chars: int) -> float:
    char_count = _count_answer_chars(text)
    if char_count <= 0:
        score = 0.0
    elif char_count < min_chars:
        score = char_count / min_chars
    elif char_count <= target_chars:
        score = 1.0
    elif char_count <= max_chars:
        score = 1.0 - 0.5 * ((char_count - target_chars) / max(1, max_chars - target_chars))
    else:
        score = 0.2
    return max(0.0, min(1.0, score))


def _repetition_reward(text: str, ngram_size: int = 6, bad_repeat_ratio: float = 0.35, terrible_repeat_ratio: float = 0.55) -> float:
    compact = re.sub(r'\s+', '', text or '')
    if len(compact) < ngram_size:
        return 1.0
    ngrams = [compact[index:index + ngram_size] for index in range(len(compact) - ngram_size + 1)]
    repeat_ratio = 1.0 - len(set(ngrams)) / len(ngrams)
    if repeat_ratio >= terrible_repeat_ratio:
        return 0.0
    if repeat_ratio >= bad_repeat_ratio:
        return 0.3
    return 1.0


def _text_quality_reward(candidate: str) -> float:
    min_chars = int(os.getenv('DRIVING_FORMAL_RM_MIN_CHARS', '20'))
    target_chars = int(os.getenv('DRIVING_FORMAL_RM_TARGET_CHARS', '80'))
    max_chars = int(os.getenv('DRIVING_FORMAL_RM_MAX_CHARS', '180'))
    return round(_length_reward(candidate, min_chars, target_chars, max_chars) * _repetition_reward(candidate), 4)


def _compact_factor(factor: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if factor is None:
        return None
    return {
        '位置': _safe_str(factor.get('位置')),
        '大类': _safe_str(factor.get('大类')),
        '细节': _safe_str(factor.get('细节')),
    }


class DrivingFormalRMPlugin(DefaultRMPlugin):
    """Formal driving reward model: schema extraction, deterministic matching, and detail-pair scoring."""

    def __init__(self, model, template):
        super().__init__(model, template)
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)
        self.request_config = RequestConfig(max_tokens=int(os.getenv('DRIVING_FORMAL_RM_MAX_TOKENS', '512')), temperature=0)
        self.task_weight = float(os.getenv('DRIVING_FORMAL_RM_TASK_WEIGHT', '0.85'))
        self.text_weight = float(os.getenv('DRIVING_FORMAL_RM_TEXT_WEIGHT', '0.15'))
        weight_sum = self.task_weight + self.text_weight
        if weight_sum <= 0:
            self.task_weight, self.text_weight = 1.0, 0.0
        else:
            self.task_weight /= weight_sum
            self.text_weight /= weight_sum
        self.save_details = os.getenv('DRIVING_FORMAL_RM_SAVE', '1') != '0'
        self.save_path = Path(os.getenv('DRIVING_FORMAL_RM_SAVE_PATH', 'output/driving_formal_rm_scores.jsonl'))
        self.debug = os.getenv('DRIVING_FORMAL_RM_DEBUG', '0') == '1'
        self.debug_n = int(os.getenv('DRIVING_FORMAL_RM_DEBUG_N', '2'))

    def __call__(self, inputs, **kwargs):
        rewards = []
        score_items = []
        for index, sample in enumerate(inputs):
            item = self._score_sample(sample)
            rewards.append(float(item['final_reward']))
            score_items.append(item)
            if self.debug and index < self.debug_n:
                print(f"[DRIVING_FORMAL_RM] reward={item['final_reward']} item={json.dumps(item, ensure_ascii=False)}")
        self._save_score_items(score_items)
        return torch.tensor(rewards, dtype=torch.float32)

    def _score_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = sample if isinstance(sample, dict) else {}
        reference_text = _extract_reference_text(sample)
        candidate_text = _extract_pred_text(sample)
        gt_answer = sample.get('gt_answer')
        label_obj = _safe_json_obj(sample.get('label'))
        if gt_answer is None and isinstance(label_obj, dict):
            gt_answer = label_obj.get('answer')

        reference_scene = (
            _scene_from_rm_schema(sample.get('rm_schema'), gt_answer=gt_answer, reference_text=reference_text)
            or self._extract_scene(reference_text)
        )
        candidate_scene = self._extract_scene(candidate_text)
        final = self._evaluate_scenes(reference_scene, candidate_scene)

        task_score = final['final_reward']
        text_score = _text_quality_reward(candidate_text)
        reward = self.task_weight * task_score + self.text_weight * text_score

        item = {
            'reference': reference_text,
            'candidate': candidate_text,
            'factor_scores': final['factor_scores'],
            'action_scores': final['action_scores'],
            'task_score': round(task_score, 4),
            'text_quality_score': round(text_score, 4),
            'final_reward': round(max(0.0, min(1.0, reward)), 4),
            'reference_scene': reference_scene,
            'candidate_scene': candidate_scene,
            'factor_matches': [{
                'index': idx + 1,
                'status': 'matched' if pair.get('candidate_factor') is not None else 'unmatched',
                'reference': _compact_factor(pair.get('reference_factor')),
                'candidate': _compact_factor(pair.get('candidate_factor')),
                'scores': {
                    'coarse': round(float(pair.get('coarse_score', 0.0)), 4),
                    'fine': round(float(pair.get('fine_score', 0.0)), 4),
                    'combined': round(float(pair.get('combined_score', 0.0)), 4),
                },
            } for idx, pair in enumerate(final['factor_pairs'])],
        }
        return item

    def _extract_scene(self, text: str) -> Dict[str, Any]:
        text = _safe_str(text)
        if not text:
            return _normalize_scene({'因素': [], '动作': {}})
        request = {
            'messages': [
                {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
                {'role': 'user', 'content': text},
            ]
        }
        try:
            result = self.engine.infer([request], self.request_config, use_tqdm=False)[0]
            obj = _safe_json_obj(result.choices[0].message.content)
            if isinstance(obj, dict):
                return _normalize_scene(obj)
        except Exception as exc:
            if self.debug:
                logger.warning(f'Formal RM extraction failed, fallback to local extraction: {exc}')
        return _extract_scene_locally(text)

    def _score_detail_pairs(self, detail_pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
        if not detail_pairs:
            return {'results': []}
        pairs_text = '\n'.join(
            f'{index + 1}. ("{reference_detail}", "{candidate_detail}")'
            for index, (reference_detail, candidate_detail) in enumerate(detail_pairs)
        )
        prompt = f"""
现在输入如下 detail pair：
{pairs_text}

评分规则：
1.0 = 描述的是同一个细节、同一种风险来源或同一个对象状态
0.5 = 同类相关但关键细节不完全一致；或一个泛化、一个具体；或数量不同
0.0 = 不是同一件事，关键对象、机制或风险来源不同

严格只输出 JSON：
{{
  "results": [
    {{
      "reference_detail": "...",
      "candidate_detail": "...",
      "relation": "...",
      "score": 0.0
    }}
  ]
}}
"""
        request = {
            'messages': [
                {'role': 'system', 'content': DETAIL_SCORING_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ]
        }
        try:
            result = self.engine.infer([request], self.request_config, use_tqdm=False)[0]
            obj = _safe_json_obj(result.choices[0].message.content)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            if self.debug:
                logger.warning(f'Formal RM detail scoring failed, fallback to local scoring: {exc}')
        return _score_detail_pairs_locally(detail_pairs)

    def _evaluate_scenes(self, reference_scene: Dict[str, Any], candidate_scene: Dict[str, Any]) -> Dict[str, Any]:
        reference_scene = _normalize_scene(reference_scene)
        candidate_scene = _normalize_scene(candidate_scene)
        factor_pairs = _match_factor_pairs_one_to_one(reference_scene.get('因素', []), candidate_scene.get('因素', []))
        action_score = _score_actions(reference_scene.get('动作', {}), candidate_scene.get('动作', {}))
        detail_pairs = _build_detail_pairs(factor_pairs)
        detail_score_result = self._score_detail_pairs(detail_pairs)
        detail_scores = _extract_detail_scores(detail_score_result, len(detail_pairs))
        merged_pairs = _merge_detail_scores(factor_pairs, detail_scores)
        return _calculate_final_result(merged_pairs, action_score)

    def _save_score_items(self, items: Sequence[Dict[str, Any]]) -> None:
        if not self.save_details or not items:
            return
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with _SAVE_LOCK:
                with self.save_path.open('a', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as exc:
            logger.warning(f'Failed to save formal RM score details to {self.save_path}: {exc}')


rm_plugins['driving_formal_rm'] = DrivingFormalRMPlugin
