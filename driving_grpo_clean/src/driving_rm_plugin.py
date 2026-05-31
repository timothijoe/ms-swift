import json
import os
import re
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

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
        {
            'name': 'think_semantic_alignment',
            'desc': '模型think与标准think在风险识别和动作依据上的语义一致性',
            'weight': 0.8
        },
        {
            'name': 'think_conciseness',
            'desc': 'think是否简洁且聚焦关键约束',
            'weight': 0.2
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


def _safe_json_obj(text: str):
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _plugin_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _configs_dir() -> str:
    return os.path.abspath(os.path.join(_plugin_dir(), '..', 'configs'))


def _default_template_store() -> Dict:
    return {
        'default_group': 'default',
        'scene_type_to_group': {
            'blocked_lane_with_oncoming': 'interaction_risk',
            'traffic_signal_stop': 'traffic_signal',
        },
        'groups': {
            'default': [{
                'name': 'default_v1',
                'system_prompt': '你是自动驾驶决策评审器。请根据打分点对模型输出评分。每个分项分数范围[0,1]。你必须只输出严格JSON，禁止输出任何额外文本。',
                'user_prompt_template': '任务: 重点比较【模型输出中的think】与【标准think】的一致性。\\n打分点:\\n{rubric_lines}\\n\\n目标answer:\\n{gt_answer}\\n\\n标准think(语义真值):\\n{gt_think}\\n\\n模型预测think:\\n{pred_think}\\n\\n模型预测answer:\\n{pred_answer}\\n\\n场景判分约束schema(逐条参考):\\n{schema_text}\\n\\n模型输出对话:\\n{message_text}\\n\\n输出JSON格式:\\n{\\\"sub_scores\\\": {\\\"<item_name>\\\": 0~1}, \\\"overall\\\": 0~1, \\\"reason\\\": \\\"...\\\"}'
            }],
            'interaction_risk': [{
                'name': 'interaction_risk_v1',
                'system_prompt': '你是交互风险驾驶决策评审器。只返回JSON分数。',
                'user_prompt_template': '请重点评估think中是否覆盖交互风险识别、让行/避让依据以及动作约束。\\n打分点:\\n{rubric_lines}\\n\\n标准think:\\n{gt_think}\\n\\n模型think:\\n{pred_think}\\n\\n场景schema:\\n{schema_text}\\n\\n参考answer:\\nGT={gt_answer}\\nPRED={pred_answer}\\n\\n返回JSON: {\\\"sub_scores\\\": {\\\"<item_name>\\\": 0~1}, \\\"overall\\\": 0~1, \\\"reason\\\": \\\"...\\\"}'
            }],
            'traffic_signal': [{
                'name': 'traffic_signal_v1',
                'system_prompt': '你是交通信号场景评审器。只输出JSON。',
                'user_prompt_template': '请评估think是否准确使用了信号灯/交警约束并给出正确动作依据。\\n打分点:\\n{rubric_lines}\\n\\n标准think:\\n{gt_think}\\n\\n模型think:\\n{pred_think}\\n\\n场景schema:\\n{schema_text}\\n\\nGT answer={gt_answer}\\nPRED answer={pred_answer}\\n\\n返回JSON: {\\\"sub_scores\\\": {\\\"<item_name>\\\": 0~1}, \\\"overall\\\": 0~1, \\\"reason\\\": \\\"...\\\"}'
            }]
        }
    }


def _load_template_store() -> Dict:
    """Load ONE template file. Fallback to built-in defaults when missing/invalid."""
    template_file = os.getenv('DRIVING_RM_TEMPLATES_FILE',
                              os.path.join(_configs_dir(), 'rm_templates.json'))
    default_store = _default_template_store()
    if not os.path.exists(template_file):
        logger.warning(f'Template file not found: {template_file}. Using built-in defaults.')
        return default_store
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f'Invalid template file format: {template_file}. Using built-in defaults.')
            return default_store
        merged = dict(default_store)
        merged.update({k: v for k, v in data.items() if k in {'default_group', 'scene_type_to_group', 'groups'}})
        return merged
    except Exception as e:
        logger.warning(f'Failed to load template file {template_file}: {e}. Using built-in defaults.')
        return default_store


def _load_prefix_store() -> Dict:
    """Load short prefix blocks by type. Fallback to {'default': ...}."""
    prefix_file = os.getenv('DRIVING_RM_PREFIX_FILE',
                            os.path.join(_configs_dir(), 'rm_templates_prefix.json'))
    default_store = {
        'default': {
            'system_prompt_prefix': '你是自动驾驶语义评分器。仅输出JSON评分结果。',
            'instruction_prefix': '请根据标准 rm_schema 和候选回答 candidate，输出结构化语义评分。',
            'few_shot_prefix': ''
        }
    }
    if not os.path.exists(prefix_file):
        logger.warning(f'Prefix file not found: {prefix_file}. Using built-in defaults.')
        return default_store
    try:
        data = json.load(open(prefix_file, 'r', encoding='utf-8'))
        if not isinstance(data, dict):
            logger.warning(f'Invalid prefix file format: {prefix_file}. Using built-in defaults.')
            return default_store
        merged = dict(default_store)
        merged.update(data)
        return merged
    except Exception as e:
        logger.warning(f'Failed to load prefix file {prefix_file}: {e}. Using built-in defaults.')
        return default_store


def _extract_think_and_answer_from_label(label_text: str):
    """Parse label in format: <think>...</think>\\n{...json...}."""
    think = ''
    answer = {}
    if not isinstance(label_text, str):
        return think, answer
    match = re.search(r'<think>(.*?)</think>', label_text, flags=re.DOTALL)
    if match:
        think = match.group(1).strip()
    obj = _safe_json_obj(label_text)
    if isinstance(obj, dict):
        answer = obj.get('answer', obj)
    return think, answer if isinstance(answer, dict) else {}


def _extract_pred_think_and_answer(messages: List[Dict]):
    """Extract predicted think/answer from generated assistant content."""
    if not messages or not isinstance(messages, list):
        return '', {}
    content = messages[-1].get('content', '')
    if not isinstance(content, str):
        return '', {}
    match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    pred_think = match.group(1).strip() if match else ''
    obj = _safe_json_obj(content)
    pred_answer = obj.get('answer', obj) if isinstance(obj, dict) else {}
    if not isinstance(pred_answer, dict):
        pred_answer = {}
    return pred_think, pred_answer


def _compact_prompt_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # collapse 3+ blank lines -> 2 blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # trim trailing spaces per line
    text = '\n'.join([line.rstrip() for line in text.splitlines()])
    return text.strip()


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
        self.prefix_store = _load_prefix_store()
        self.template_store = _load_template_store()
        self.template_seed = int(os.getenv('DRIVING_RM_TEMPLATE_SEED', '42'))
        self.template_rng = random.Random(self.template_seed)

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
            label_obj = _safe_json_obj(label) if isinstance(label, str) else (label if isinstance(label, dict) else None)
            label_think = label_obj.get('think', '') if isinstance(label_obj, dict) else ''
            label_answer = label_obj.get('answer', {}) if isinstance(label_obj, dict) else {}
            if not label_think and isinstance(label, str):
                parsed_think, parsed_answer = _extract_think_and_answer_from_label(label)
                label_think = parsed_think or label_think
                label_answer = parsed_answer or label_answer
            think = request.get('think', '') or label_think
            gt_answer = request.get('gt_answer', {}) or label_answer
            pred_think, pred_answer = _extract_pred_think_and_answer(messages)
            candidate_text = messages[-1].get('content', '') if messages else ''
            rm_schema = request.get('rm_schema')
            schema_text = json.dumps(rm_schema, ensure_ascii=False) if rm_schema is not None else '无'
            template_type, prefix = self._select_prefix(request, rm_schema)
            prompt = self._render_prefix_prompt(
                prefix=prefix,
                rubric_lines=rubric_lines,
                gt_answer=gt_answer,
                gt_think=think,
                pred_think=pred_think,
                pred_answer=pred_answer,
                schema_text=schema_text,
                candidate_text=candidate_text,
                message_text=self._messages_to_text(messages))

            # Keep legacy template route as fallback only when prefix is empty.
            if not prompt.strip():
                template = self._select_template(rm_schema)
                prompt = self._render_template(
                    template=template,
                    rubric_lines=rubric_lines,
                    gt_answer=gt_answer,
                    gt_think=think,
                    pred_think=pred_think,
                    pred_answer=pred_answer,
                    schema_text=schema_text,
                    message_text=self._messages_to_text(messages))
                system_prompt = template.get('system_prompt') if isinstance(template, dict) else None
            else:
                system_prompt = prefix.get('system_prompt_prefix')
            request['messages'] = [{
                'role': 'system',
                'content': system_prompt or self.system
            }, {
                'role': 'user',
                'content': prompt
            }]
            rm_inputs.append(request)
        return rm_inputs

    def _select_prefix(self, request: Dict, rm_schema: Optional[Dict]) -> Tuple[str, Dict]:
        t = str(request.get('rm_template_type', '') or request.get('template_type', '')).strip()
        if not t and isinstance(rm_schema, dict):
            # Backward-compatible fallback for old datasets.
            t = str(rm_schema.get('template_type', '') or rm_schema.get('type', '')).strip()
        if not t:
            t = 'default'
        prefix = self.prefix_store.get(t) or self.prefix_store.get('default', {})
        return t, prefix

    @staticmethod
    def _render_prefix_prompt(prefix: Dict,
                              *,
                              rubric_lines: str,
                              gt_answer: Dict,
                              gt_think: str,
                              pred_think: str,
                              pred_answer: Dict,
                              schema_text: str,
                              candidate_text: str,
                              message_text: str) -> str:
        if not isinstance(prefix, dict):
            return ''
        instruction = str(prefix.get('instruction_prefix', '') or '')
        few_shot = str(prefix.get('few_shot_prefix', '') or '')
        if not instruction and not few_shot:
            return ''
        content = '\n'.join([
            instruction,
            few_shot,
            '',
            '【待评分输入】',
            f'标准JSON:\n{schema_text}',
            '',
            '【候选回答】',
            candidate_text or message_text,
            '',
            '【补充上下文】',
            f'打分点:\n{rubric_lines}',
            f'标准think:\n{gt_think}',
            f'模型think:\n{pred_think}',
            f'标准answer:\n{json.dumps(gt_answer, ensure_ascii=False)}',
            f'模型answer:\n{json.dumps(pred_answer, ensure_ascii=False)}',
            '',
            '请严格只输出JSON。'
        ])
        if os.getenv('DRIVING_RM_COMPACT_PROMPT', '1') == '1':
            content = _compact_prompt_text(content)
        return content

    def _select_template(self, rm_schema: Optional[Dict]) -> Dict:
        scene_type = ''
        if isinstance(rm_schema, dict):
            scene_type = str(rm_schema.get('scene_type', '')).strip()
        scene_map = self.template_store.get('scene_type_to_group', {}) or {}
        group = scene_map.get(scene_type, self.template_store.get('default_group', 'default'))
        groups = self.template_store.get('groups', {}) or {}
        templates = groups.get(group) or groups.get('default') or []
        if not templates:
            return {}
        # deterministic but varied
        return self.template_rng.choice(templates)

    @staticmethod
    def _render_template(template: Dict,
                         *,
                         rubric_lines: str,
                         gt_answer: Dict,
                         gt_think: str,
                         pred_think: str,
                         pred_answer: Dict,
                         schema_text: str,
                         message_text: str) -> str:
        prompt_tpl = None if not isinstance(template, dict) else template.get('user_prompt_template')
        context = {
            'rubric_lines': rubric_lines,
            'gt_answer': json.dumps(gt_answer, ensure_ascii=False),
            'gt_think': gt_think,
            'pred_think': pred_think,
            'pred_answer': json.dumps(pred_answer, ensure_ascii=False),
            'schema_text': schema_text,
            'message_text': message_text,
            'candidate_text': message_text,
            'rm_schema_json': schema_text,
        }
        # New structured template format:
        # {instruction_prefix, few_shots, output_format, footer}
        if isinstance(template, dict) and any(k in template for k in ('instruction_prefix', 'few_shots', 'output_format')):
            blocks = []
            instruction_prefix = template.get('instruction_prefix', '')
            if instruction_prefix:
                rendered_prefix = str(instruction_prefix)
                # support explicit placeholders in large raw templates
                rendered_prefix = rendered_prefix.replace('{RM_SCHEMA}', context['rm_schema_json'])
                rendered_prefix = rendered_prefix.replace('{CANDIDATE}', context['candidate_text'])
                # keep backward compatibility with python format placeholders
                try:
                    rendered_prefix = rendered_prefix.format(**context)
                except Exception:
                    pass
                blocks.append(rendered_prefix)
            few_shots = template.get('few_shots', [])
            if isinstance(few_shots, list):
                for idx, fs in enumerate(few_shots, start=1):
                    if not isinstance(fs, dict):
                        continue
                    blocks.append(f'【Few-shot {idx}】')
                    if fs.get('schema'):
                        blocks.append('标准JSON:')
                        blocks.append(json.dumps(fs['schema'], ensure_ascii=False))
                    if fs.get('candidate'):
                        blocks.append(f'候选回答: {fs["candidate"]}')
                    if fs.get('output'):
                        blocks.append(f'输出: {json.dumps(fs["output"], ensure_ascii=False)}')
            output_format = template.get('output_format')
            if output_format:
                blocks.append('【最终输出格式】')
                blocks.append(str(output_format))
            footer = template.get('footer', '')
            if footer:
                blocks.append(str(footer).format(**context))
            full_text = '\n'.join(blocks)
            # If the large template already embeds placeholders, do not append duplicated blocks.
            if ('{RM_SCHEMA}' not in str(instruction_prefix)) and ('{CANDIDATE}' not in str(instruction_prefix)):
                blocks.append('【待评分输入】')
                blocks.append(context['rm_schema_json'])
                blocks.append('【候选回答】')
                blocks.append(context['candidate_text'])
                full_text = '\n'.join(blocks)
            if os.getenv('DRIVING_RM_COMPACT_PROMPT', '1') == '1':
                full_text = _compact_prompt_text(full_text)
            return full_text

        if prompt_tpl and isinstance(prompt_tpl, str):
            try:
                rendered = prompt_tpl.format(**context)
                if os.getenv('DRIVING_RM_COMPACT_PROMPT', '1') == '1':
                    rendered = _compact_prompt_text(rendered)
                return rendered
            except Exception:
                pass
        fallback = (
            '任务: 重点比较【模型输出中的think】与【标准think】的一致性。\n'
            f'打分点:\n{rubric_lines}\n\n'
            f'目标answer:\n{context["gt_answer"]}\n\n'
            f'标准think(语义真值):\n{gt_think}\n\n'
            f'模型预测think:\n{pred_think}\n\n'
            f'模型预测answer:\n{context["pred_answer"]}\n\n'
            f'场景判分约束schema(逐条参考):\n{schema_text}\n\n'
            f'模型输出对话:\n{message_text}\n\n'
            '输出JSON格式:\n'
            '{"sub_scores": {"<item_name>": 0~1}, "overall": 0~1, "reason": "..."}\n'
            '要求:\n'
            '1) sub_scores 必须包含所有打分点key；\n'
            '2) 分数必须在0到1之间；\n'
            '3) reason 用一句话说明扣分主因；\n'
            '4) 不要求字面一致，重点看think语义是否等价；\n'
            '5) 若模型缺少有效think，应显著扣分。')
        if os.getenv('DRIVING_RM_COMPACT_PROMPT', '1') == '1':
            fallback = _compact_prompt_text(fallback)
        return fallback

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
