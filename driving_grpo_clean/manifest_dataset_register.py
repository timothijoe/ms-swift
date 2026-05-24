import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from swift.llm.dataset.preprocessor import AutoPreprocessor, RowPreprocessor
from swift.llm.dataset.register import DatasetMeta, register_dataset


THINK_INSTRUCTION = '格式要求：先输出 <think>...</think>，然后直接输出JSON，包含键“横向决策”和“纵向决策”。'
OLD_PROJECT_PREFIX = '/home/linux/project/v3.12-ms-swift'


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _resolve_path(path: str, *, base_dir: str) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    candidates = [
        os.path.abspath(os.path.join(base_dir, path)),
        os.path.abspath(os.path.join(_repo_root(), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _remap_media_path(value):
    if isinstance(value, str):
        if value.startswith(OLD_PROJECT_PREFIX):
            return os.path.join(_repo_root(), value[len(OLD_PROJECT_PREFIX):].lstrip('/'))
        return value
    if isinstance(value, list):
        return [_remap_media_path(item) for item in value]
    return value


@dataclass
class ManifestEntry:
    name: str
    path: str
    preprocess_type: str = 'driving_no_think'
    annotation_path: Optional[str] = None
    sample_num: Optional[int] = None
    think_ratio: float = 0.0
    data_type: str = 'driving_decision'
    split: str = 'train'

    @staticmethod
    def from_dict(data: Dict[str, Any], *, base_dir: str) -> 'ManifestEntry':
        name = data.get('name')
        annotation_path = data.get('annotation_path')
        path = data.get('path') or annotation_path
        if not name or not path:
            raise ValueError(f'Each manifest item requires name/(path or annotation_path), got: {data}')

        sample_num = data.get('sample_num')
        if sample_num is not None:
            sample_num = int(sample_num)
            if sample_num <= 0:
                raise ValueError(f'sample_num must be > 0, got: {sample_num}')

        think_ratio = float(data.get('think_ratio', 0.0))
        if not 0 <= think_ratio <= 1:
            raise ValueError(f'think_ratio must be in [0, 1], got: {think_ratio}')

        split = str(data.get('split', 'train')).strip().lower() or 'train'
        if split not in {'train', 'eval'}:
            raise ValueError(f'split must be train or eval, got: {split}')

        return ManifestEntry(
            name=str(name),
            path=_resolve_path(path, base_dir=base_dir),
            preprocess_type=str(data.get('preprocess_type') or 'driving_no_think'),
            annotation_path=(_resolve_path(annotation_path, base_dir=base_dir) if annotation_path else None),
            sample_num=sample_num,
            think_ratio=think_ratio,
            data_type=str(data.get('data_type', 'driving_decision')).strip() or 'driving_decision',
            split=split,
        )


class DrivingDecisionNoThinkPreprocessor(RowPreprocessor):
    def __init__(self, *, data_type: str = 'driving_decision', **kwargs):
        super().__init__(**kwargs)
        self.data_type = data_type

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = row.get('messages')
        if not messages or not isinstance(messages, list) or messages[-1].get('role') != 'assistant':
            return
        label = messages[-1].get('content')
        if label is None:
            return
        row['messages'] = messages[:-1]
        row['label'] = label
        row['data_type'] = row.get('data_type') or self.data_type
        if 'images' in row:
            row['images'] = _remap_media_path(row['images'])
        if 'videos' in row:
            row['videos'] = _remap_media_path(row['videos'])
        return row


class DrivingDecisionMixedPreprocessor(RowPreprocessor):
    def __init__(self, *, think_ratio: float = 0.0, seed: int = 42,
                 data_type: str = 'driving_decision', **kwargs):
        super().__init__(**kwargs)
        self.think_ratio = max(0.0, min(1.0, think_ratio))
        self.random = random.Random(seed)
        self.data_type = data_type

    def _inject_think(self, messages):
        if messages and messages[0].get('role') == 'system':
            messages = [dict(messages[0])] + [dict(message) for message in messages[1:]]
            messages[0]['content'] = f"{messages[0].get('content', '')}\n{THINK_INSTRUCTION}".strip()
            return messages
        return [{'role': 'system', 'content': THINK_INSTRUCTION}] + [dict(message) for message in messages]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = row.get('messages')
        if not messages or not isinstance(messages, list) or messages[-1].get('role') != 'assistant':
            return
        label = messages[-1].get('content')
        if label is None:
            return

        prompt_messages = messages[:-1]
        if self.think_ratio > 0 and self.random.random() < self.think_ratio:
            prompt_messages = self._inject_think(prompt_messages)

        row['messages'] = prompt_messages
        row['label'] = label
        row['data_type'] = row.get('data_type') or self.data_type
        if 'images' in row:
            row['images'] = _remap_media_path(row['images'])
        if 'videos' in row:
            row['videos'] = _remap_media_path(row['videos'])
        return row


def _build_preprocessor(entry: ManifestEntry, seed: int, default_think_ratio: float = 0.0):
    common_kwargs = {'dataset_sample': entry.sample_num, 'random_state': seed}
    preprocess_type = (entry.preprocess_type or 'driving_no_think').strip()
    if preprocess_type == 'driving_no_think':
        return DrivingDecisionNoThinkPreprocessor(data_type=entry.data_type, **common_kwargs)
    if preprocess_type == 'driving_mixed':
        think_ratio = entry.think_ratio if entry.think_ratio > 0 else default_think_ratio
        return DrivingDecisionMixedPreprocessor(
            think_ratio=think_ratio, seed=seed, data_type=entry.data_type, **common_kwargs)
    if preprocess_type == 'auto':
        return AutoPreprocessor()
    raise ValueError(
        f'Unsupported preprocess_type: {entry.preprocess_type}. '
        'Supported: driving_no_think, driving_mixed, auto')


def _load_manifest(path: str) -> List[ManifestEntry]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Manifest root must be a list.')
    base_dir = os.path.dirname(path)
    return [ManifestEntry.from_dict(item, base_dir=base_dir) for item in data]


def register_from_manifest_path(manifest_path: str, *, seed: int = 42, default_think_ratio: float = 0.0):
    manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
    if not os.path.exists(manifest_path):
        raise ValueError(f'Manifest file not found: {manifest_path}')

    entries = _load_manifest(manifest_path)
    for entry in entries:
        if not os.path.exists(entry.path):
            raise ValueError(f'Dataset path not found: {entry.path}')
        register_dataset(
            DatasetMeta(
                dataset_name=entry.name,
                dataset_path=entry.path,
                preprocess_func=_build_preprocessor(entry, seed, default_think_ratio=default_think_ratio),
            ),
            exist_ok=True,
        )

    print(f'[DRIVING_MANIFEST] registered {len(entries)} datasets from {manifest_path}')
    return entries


def register_from_env():
    manifest_path = os.getenv('DATASET_MANIFEST_PATH')
    if not manifest_path:
        raise ValueError('DATASET_MANIFEST_PATH is required for env-based registration.')
    seed = int(os.getenv('DRIVING_THINK_SEED', '42'))
    default_think_ratio = float(os.getenv('DRIVING_THINK_RATIO', '0.0'))
    return register_from_manifest_path(manifest_path, seed=seed, default_think_ratio=default_think_ratio)


if os.getenv('DATASET_MANIFEST_PATH'):
    register_from_env()
