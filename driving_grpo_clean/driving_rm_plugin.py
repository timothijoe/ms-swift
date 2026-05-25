import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Tuple

import torch

from swift.llm import PtEngine, RequestConfig
from swift.plugin import rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()


def _load_rubric_items() -> List[Dict]:
    """Load rubric from env var DRIVING_RM_RUBRIC_JSON or use a safe default."""
    rubric_text = os.getenv('DRIVING_RM_RUBRIC_JSON', '').strip()
    if rubric_text:
        try:
            rubric_items = json.loads(rubric_text)
            if isinstance(rubric_items, list) and rubric_items:
                return rubric_items
        except Exception:
            logger.warning('Failed to parse DRIVING_RM_RUBRIC_JSON, fallback to default rubric.')

    return [
        {'name': 'horizontal_decision', 'desc': '横向决策是否与目标一致', 'weight': 0.4},
        {'name': 'vertical_decision', 'desc': '纵向决策是否与目标一致', 'weight': 0.4},
        {'name': 'format_validity', 'desc': '输出是否为可解析JSON且字段完整', 'weight': 0.2},
    ]


def _normalize_weights(items: List[Dict]) -> List[Tuple[str, str, float]]:
    values = []
    total = 0.0
    for item in items:
        name = str(item.get('name', '')).strip()
        desc = str(item.get('desc', '')).strip()
        weight = float(item.get('weight', 0.0))
        if not name or weight <= 0:
            continue
        values.append((name, desc, weight))
        total += weight
    if not values:
        return [('overall', '整体质量评分', 1.0)]
    if total <= 0:
        return [(name, desc, 1.0 / len(values)) for name, desc, _ in values]
    return [(name, desc, weight / total) for name, desc, weight in values]


class DrivingRubricRMPlugin(DefaultRMPlugin):
    """
    Generative RM plugin for driving imitation-style rubric scoring.

    Expected model output JSON:
    {
      "sub_scores": {"horizontal_decision": 0.9, "vertical_decision": 0.8, "format_validity": 1.0},
      "overall": 0.88,
      "reason": "..."
    }
    """

    def __init__(self, model, template):
        super().__init__(model, template)
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)
        self.request_config = RequestConfig(max_tokens=256, temperature=0)
        self.rubric = _normalize_weights(_load_rubric_items())
        self.system = (
            '你是自动驾驶决策评审器。请根据给定打分点对模型输出进行评分。'
            '每个分项分数范围[0,1]，返回严格JSON，不要输出额外文本。'
        )

    def __call__(self, inputs, **kwargs):
        rm_inputs = self._build_rm_inputs(inputs)
        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        rewards = [self._extract_reward(result.choices[0].message.content) for result in results]
        return torch.tensor(rewards, dtype=torch.float32)

    def _build_rm_inputs(self, inputs: List[Dict]) -> List[Dict]:
        rubric_lines = '\n'.join([f'- {n}: {d} (weight={w:.3f})' for n, d, w in self.rubric])
        rm_inputs = []
        for infer_request in inputs:
            request = deepcopy(infer_request)
            messages = request.get('messages', [])
            label = request.get('label', '')
            prompt = (
                '任务: 比较【模型输出】与【目标标签】的匹配程度。\n'
                f'打分点:\n{rubric_lines}\n\n'
                f'目标标签:\n{label}\n\n'
                f'模型输出对话:\n{self._messages_to_text(messages)}\n\n'
                '输出JSON格式:\n'
                '{"sub_scores": {"<item_name>": 0~1}, "overall": 0~1, "reason": "..."}'
            )
            request['messages'] = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': prompt}]
            rm_inputs.append(request)
        return rm_inputs

    @staticmethod
    def _messages_to_text(messages: List[Dict]) -> str:
        lines = []
        for message in messages:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            if content:
                lines.append(f'{role}: {content}')
        return '\n'.join(lines)

    def _extract_reward(self, text: str) -> float:
        obj = self._extract_json(text)
        if obj is None:
            return 0.0
        sub_scores = obj.get('sub_scores', {})
        if not isinstance(sub_scores, dict):
            sub_scores = {}

        weighted = 0.0
        for name, _desc, weight in self.rubric:
            score = sub_scores.get(name, 0.0)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))
            weighted += weight * score

        overall = obj.get('overall')
        if overall is not None:
            try:
                overall = max(0.0, min(1.0, float(overall)))
                weighted = 0.5 * weighted + 0.5 * overall
            except Exception:
                pass
        return float(max(0.0, min(1.0, weighted)))

    @staticmethod
    def _extract_json(text: str):
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


rm_plugins['driving_rubric_rm'] = DrivingRubricRMPlugin
