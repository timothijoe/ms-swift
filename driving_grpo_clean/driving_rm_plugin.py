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
_DEBUG_WAIT_DONE = False


def _append_debug_file(msg: str):
    path = os.getenv('DRIVING_RM_DEBUG_FILE', '/tmp/driving_rm_debug.log')
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass


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
        {
            'name': 'semantic_alignment',
            'desc': '与目标驾驶意图的语义一致性。允许用词不同、措辞不同，只要含义一致即可高分',
            'weight': 0.7
        },
        {
            'name': 'decision_completeness',
            'desc': '是否同时表达了横向与纵向决策意图，且不存在明显冲突',
            'weight': 0.3
        },
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
            '你是自动驾驶决策评审器。请根据打分点对模型输出评分。'
            '每个分项分数范围[0,1]。你必须只输出严格JSON，禁止输出任何额外文本。'
        )
        logger.warning('[DRIVING_RM_DEBUG] DrivingRubricRMPlugin initialized')
        _append_debug_file('[init] DrivingRubricRMPlugin initialized')

    def __call__(self, inputs, **kwargs):
        self._maybe_wait_for_debugger()
        debug_mode = os.getenv('DRIVING_RM_DEBUG', '0') == '1'
        if os.getenv('DRIVING_RM_DEBUG_FORCE_FAIL', '0') == '1':
            raise RuntimeError('DRIVING_RM_DEBUG_FORCE_FAIL=1, plugin is confirmed reachable.')
        if debug_mode:
            logger.warning(f'[DRIVING_RM_DEBUG] plugin called. batch_size={len(inputs)}')
            _append_debug_file(f'[call] batch_size={len(inputs)}')
        rm_inputs = self._build_rm_inputs(inputs)
        if debug_mode and rm_inputs:
            logger.warning(f'[DRIVING_RM_DEBUG] first_rm_prompt={rm_inputs[0]["messages"][-1]["content"][:300]}')
            _append_debug_file(f'[prompt] {rm_inputs[0]["messages"][-1]["content"][:120]}')
        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        rewards = [self._extract_reward(result.choices[0].message.content) for result in results]
        if debug_mode:
            logger.warning(f'[DRIVING_RM_DEBUG] rewards_preview={rewards[:3]}')
            _append_debug_file(f'[reward] {rewards[:3]}')
        return torch.tensor(rewards, dtype=torch.float32)

    @staticmethod
    def _maybe_wait_for_debugger():
        global _DEBUG_WAIT_DONE
        if _DEBUG_WAIT_DONE:
            return
        if os.getenv('DRIVING_RM_DEBUG_WAIT', '0') != '1':
            return
        _DEBUG_WAIT_DONE = True
        try:
            import debugpy
            host = os.getenv('DRIVING_RM_DEBUG_HOST', '127.0.0.1')
            port = int(os.getenv('DRIVING_RM_DEBUG_PORT', '5678'))
            debugpy.listen((host, port))
            logger.warning(f'[DRIVING_RM_DEBUG] waiting debugger attach at {host}:{port}')
            debugpy.wait_for_client()
            logger.warning('[DRIVING_RM_DEBUG] debugger attached')
        except Exception as e:
            logger.warning(f'[DRIVING_RM_DEBUG] debug wait failed: {e}')

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
                '{"sub_scores": {"<item_name>": 0~1}, "overall": 0~1, "reason": "..."}\n'
                '要求:\n'
                '1) sub_scores 必须包含所有打分点key；\n'
                '2) 分数必须在0到1之间；\n'
                '3) reason 用一句话说明扣分主因；\n'
                '4) 不要求字面一致，重点看语义是否等价。'
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
