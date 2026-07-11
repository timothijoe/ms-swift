from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .coc_trace_schema import TraceSchema, parse_trace_schema


def prepare_jsonl(input_path: Path, output_path: Path) -> int:
    """Persist a normalized CoC Trace schema for every JSONL row."""
    count = 0
    with input_path.open('r', encoding='utf-8') as source, output_path.open('w', encoding='utf-8') as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            think = str(row.get('think') or _extract_assistant_think(row.get('messages')))
            if not think:
                raise ValueError(f'row {line_number} does not contain a CoC trace in `think`')
            row['think'] = think
            row['coc_trace_schema'] = parse_trace_schema(think).to_dict()
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


class DrivingTracePreprocessor:
    """Normalize an in-memory driving row without framework dependencies."""

    def preprocess(self, row: dict[str, Any]) -> Optional[dict[str, Any]]:
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            return None
        output = dict(row)
        assistant = messages[-1] if messages[-1].get('role') == 'assistant' else None
        content = assistant.get('content', '') if assistant else ''
        output['messages'] = messages[:-1] if assistant else messages
        output['think'] = str(output.get('think') or _extract_think(content))
        output['gt_answer'] = output.get('answer') or _extract_json(content)
        output['label'] = json.dumps({'think': output['think'], 'answer': output['gt_answer']}, ensure_ascii=False)
        output['data_type'] = 'driving_decision'
        schema = output.get('coc_trace_schema')
        if not isinstance(schema, dict):
            return None
        TraceSchema.from_dict(schema)
        return output


def _extract_assistant_think(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return ''
    last = messages[-1]
    return _extract_think(last.get('content', '')) if isinstance(last, dict) else ''


def _extract_think(content: Any) -> str:
    match = re.search(r'<think>(.*?)</think>', str(content), flags=re.DOTALL)
    return match.group(1).strip() if match else ''


def _extract_json(content: Any) -> dict[str, Any]:
    match = re.search(r'\{.*\}', str(content), flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
