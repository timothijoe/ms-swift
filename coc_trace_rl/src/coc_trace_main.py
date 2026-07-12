from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Union

from swift.llm.dataset.preprocessor import RowPreprocessor
from swift.llm.dataset.register import DatasetMeta, register_dataset
from swift.llm.train.rlhf import SwiftRLHF
from swift.utils import get_model_parameter_info

from .coc_trace_args import CocTraceDrivingArguments
from .coc_trace_grpo_trainer import CocTraceGRPOTrainer
from .driving_dataset import DrivingTracePreprocessor


class _DatasetPreprocessor(RowPreprocessor):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._delegate = DrivingTracePreprocessor()

    def preprocess(self, row: dict[str, Any]):
        return self._delegate.preprocess(row)


class CocTraceDrivingRLHF(SwiftRLHF):
    args_class = CocTraceDrivingArguments

    def run(self):
        self._register_local_plugins()
        self._configure_manifest()
        return self._run_with_coc_trainer()

    def _register_local_plugins(self) -> None:
        from . import driving_formal_rm, driving_rewards
        assert driving_formal_rm and driving_rewards

    def _configure_manifest(self) -> None:
        manifest_path = Path(self.args.dataset_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        entries = json.loads(manifest_path.read_text(encoding='utf-8'))
        train, evaluation = [], []
        for entry in entries:
            path = (manifest_path.parent / entry['path']).resolve()
            if not path.exists():
                raise FileNotFoundError(f'dataset file not found: {path}')
            register_dataset(DatasetMeta(
                dataset_name=entry['name'], dataset_path=str(path), preprocess_func=_DatasetPreprocessor()), exist_ok=True)
            (evaluation if entry.get('split') == 'eval' else train).append(entry['name'])
        if not train:
            raise ValueError('manifest must contain one train dataset')
        self.args.dataset = train
        self.args.val_dataset = evaluation

    def _run_with_coc_trainer(self):
        train_dataset, val_dataset = self._prepare_dataset()
        self.args.save_args()
        data_collator = self._get_data_collator()
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        self.train_msg['model_parameter_info'] = get_model_parameter_info(self.model)
        trainer = CocTraceGRPOTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            coc_trace_enabled=self.args.coc_trace_enabled,
            coc_trace_reward_threshold=self.args.coc_trace_reward_threshold,
            coc_trace_probability=self.args.coc_trace_probability,
            coc_trace_num_generations=self.args.coc_trace_num_generations,
            coc_trace_seed=self.args.coc_trace_seed,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def coc_trace_driving_main(args: Optional[Union[List[str], CocTraceDrivingArguments]] = None):
    return CocTraceDrivingRLHF(args).main()


if __name__ == '__main__':
    coc_trace_driving_main()
