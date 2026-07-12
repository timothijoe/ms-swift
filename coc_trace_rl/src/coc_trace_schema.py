from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence


ACTION_TERMS = ('左换道', '右换道', '左避让', '右避让', '车道居中', '减速', '加速', '停车', '保持')
TASK_TYPES = {
    '盲区': ('交通参与者', '盲区'),
    '遮挡': ('交通参与者', '盲区'),
    '变道': ('交通参与者', '车辆切入'),
    '切入': ('交通参与者', '车辆切入'),
    '障碍': ('道路风险', '障碍绕行'),
    '事故': ('道路风险', '障碍绕行'),
    '信号': ('交通管制', '信号灯'),
    '红灯': ('交通管制', '信号灯'),
    '路口': ('道路结构', '路口通行'),
}
EVIDENCE_TERMS = {
    '相对位置': ('前方', '左前方', '右前方', '左侧', '右侧', '后方'),
    '可见性': ('盲区', '遮挡', '视野'),
    '运动趋势': ('变道', '切入', '靠近', '驶入', '行驶'),
    '时序': ('即将', '正在', '前方存在', '当前'),
    '安全裕度': ('安全', '风险', '让行', '谨慎'),
}


@dataclass(frozen=True)
class TraceSchema:
    task_category: str
    task_subcategory: str
    evidence: tuple[str, ...]
    risk_causes: tuple[str, ...]
    constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'task_category': self.task_category,
            'task_subcategory': self.task_subcategory,
            'evidence': list(self.evidence),
            'risk_causes': list(self.risk_causes),
            'constraints': list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'TraceSchema':
        return cls(
            task_category=str(data.get('task_category', '驾驶场景')),
            task_subcategory=str(data.get('task_subcategory', '常规风险')),
            evidence=tuple(str(item) for item in data.get('evidence', [])),
            risk_causes=tuple(str(item) for item in data.get('risk_causes', [])),
            constraints=tuple(str(item) for item in data.get('constraints', [])),
        )


@dataclass(frozen=True)
class GuidanceVariant:
    level: int
    text: str
    gaps: tuple[str, ...]


def _without_actions(text: str) -> str:
    for term in ACTION_TERMS:
        text = text.replace(term, '')
    return text


def _matching_labels(text: str, terms: dict[str, Sequence[str]]) -> tuple[str, ...]:
    return tuple(label for label, keywords in terms.items() if any(keyword in text for keyword in keywords))


def parse_trace_schema(trace: str) -> TraceSchema:
    text = _without_actions(trace or '')
    category, subcategory = '驾驶场景', '常规风险'
    for keyword, value in TASK_TYPES.items():
        if keyword in text:
            category, subcategory = value
            break
    evidence = _matching_labels(text, EVIDENCE_TERMS)
    causes = tuple(term for term in ('遮挡', '盲区', '占道', '切入', '信号冲突') if term in text)
    constraints = tuple(term for term in ('可见性', '安全裕度', '路权', '交通规则') if term in text)
    return TraceSchema(category, subcategory, evidence, causes, constraints)


def score_trace_schema(reference: TraceSchema, candidate: TraceSchema) -> float:
    fields = (
        float(reference.task_category == candidate.task_category),
        float(reference.task_subcategory == candidate.task_subcategory),
        _overlap(reference.evidence, candidate.evidence),
        _overlap(reference.risk_causes, candidate.risk_causes),
        _overlap(reference.constraints, candidate.constraints),
    )
    return sum(fields) / len(fields)


def _overlap(reference: Iterable[str], candidate: Iterable[str]) -> float:
    expected = set(reference)
    if not expected:
        return 1.0
    return len(expected & set(candidate)) / len(expected)


def build_guidance_variants(reference: TraceSchema, candidate_schemas: Sequence[TraceSchema]) -> list[GuidanceVariant]:
    covered = set().union(*(set(schema.evidence) for schema in candidate_schemas)) if candidate_schemas else set()
    gaps = tuple(item for item in reference.evidence if item not in covered)
    gap_text = '、'.join(gaps) if gaps else '风险因果与安全约束'
    locator = f'该场景属于“{reference.task_category}/{reference.task_subcategory}”任务。请基于视频独立推理。'
    frame = f'该场景属于“{reference.task_category}/{reference.task_subcategory}”。请重点核查：{gap_text}，并说明风险因果。'
    checks = '、'.join(gaps[:3] or ('相对位置', '可见性', '风险因果'))
    checklist = f'请选择需要优先核查的推理维度：{checks}。请基于视频独立完成推理，不要引用本提示。'
    return [
        GuidanceVariant(level=1, text=locator, gaps=gaps),
        GuidanceVariant(level=2, text=frame, gaps=gaps),
        GuidanceVariant(level=3, text=checklist, gaps=gaps),
    ]


def should_trigger_coc_trace(*, enabled: bool, group_scores: Sequence[float], threshold: float,
                             probability: float, prompt_id: str, global_step: int, seed: int) -> bool:
    if not enabled or not group_scores or max(group_scores) >= threshold:
        return False
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    digest = sha256(f'{seed}:{global_step}:{prompt_id}'.encode()).digest()
    draw = int.from_bytes(digest[:8], 'big') / 2**64
    return draw < probability
