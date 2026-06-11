import os
from typing import List, Optional, Union

from swift.llm.train.rlhf import SwiftRLHF
from swift.utils import import_external_file

try:
    from .driving_manifest_args import DrivingManifestRLHFArguments, project_root
except ImportError:
    from driving_manifest_args import DrivingManifestRLHFArguments, project_root


class DrivingManifestRLHF(SwiftRLHF):
    args_class = DrivingManifestRLHFArguments
    args: args_class

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(project_root(), path))

    def _sync_eval_training_args(self):
        args = self.args
        if not getattr(args, 'val_dataset', None):
            return
        if not hasattr(args, 'training_args') or args.training_args is None:
            return

        ta = args.training_args
        ta.do_eval = True
        if getattr(ta, 'eval_steps', None) in (None, 0) and getattr(args, 'eval_steps', None):
            ta.eval_steps = args.eval_steps

        desired_eval_strategy = getattr(args, 'eval_strategy', None) or 'steps'
        if desired_eval_strategy == 'no':
            desired_eval_strategy = 'steps'

        eval_strategy_cls = type(getattr(ta, 'evaluation_strategy', 'steps'))
        if hasattr(eval_strategy_cls, 'STEPS'):
            if desired_eval_strategy == 'steps':
                ta.evaluation_strategy = eval_strategy_cls.STEPS
            elif desired_eval_strategy == 'epoch' and hasattr(eval_strategy_cls, 'EPOCH'):
                ta.evaluation_strategy = eval_strategy_cls.EPOCH
            elif hasattr(eval_strategy_cls, 'NO'):
                ta.evaluation_strategy = eval_strategy_cls.NO
        else:
            ta.evaluation_strategy = desired_eval_strategy
        ta.eval_strategy = desired_eval_strategy

        print(
            f'[DRIVING_EVAL] do_eval={ta.do_eval}, '
            f'evaluation_strategy={ta.evaluation_strategy}, eval_steps={ta.eval_steps}')

    def run(self):
        args = self.args
        manifest_path = self._resolve_path(args.dataset_manifest_path)
        register_file = self._resolve_path(args.manifest_register_file)
        reward_file = self._resolve_path('driving_grpo_clean/src/driving_reward_funcs.py')
        rm_plugin_file = self._resolve_path('driving_grpo_clean/src/driving_rm_plugin.py')

        if not os.path.exists(register_file):
            raise FileNotFoundError(f'manifest_register_file not found: {register_file}')
        if not os.path.exists(reward_file):
            raise FileNotFoundError(f'reward plugin not found: {reward_file}')

        register_module = import_external_file(register_file)
        entries = register_module.register_from_manifest_path(
            manifest_path, seed=args.driving_think_seed, default_think_ratio=args.driving_think_ratio)
        import_external_file(reward_file)

        # Make RM plugin loading explicit and observable in debug mode.
        if getattr(args, 'reward_model', None):
            rm_plugins = getattr(args, 'reward_model_plugin', None)
            if rm_plugins and 'driving_formal_rm' in rm_plugins:
                if not os.path.exists(rm_plugin_file):
                    raise FileNotFoundError(f'rm plugin file not found: {rm_plugin_file}')
                import_external_file(rm_plugin_file)
                print(f'[DRIVING_RM] imported rm plugin file: {rm_plugin_file}')

        if args.use_manifest_as_dataset:
            train_dataset = []
            eval_dataset = []
            for entry in entries:
                dataset_path = getattr(entry, 'annotation_path', None) or entry.path
                dataset_ref = f'{dataset_path}#{entry.sample_num}' if entry.sample_num is not None else dataset_path
                if entry.split == 'eval':
                    eval_dataset.append(dataset_ref)
                else:
                    train_dataset.append(dataset_ref)

            if not train_dataset:
                raise ValueError('No train entries found in manifest.')
            args.dataset = train_dataset
            args.val_dataset = eval_dataset
        else:
            if not args.dataset_name:
                raise ValueError('dataset_name is required when use_manifest_as_dataset is False.')
            args.dataset = [args.dataset_name]

        if not args.reward_funcs:
            # In RM-only mode we keep reward_funcs empty and rely on reward_model.
            if getattr(args, 'reward_model', None):
                args.reward_funcs = []
            else:
                args.reward_funcs = (
                    ['driving_mixed_format', 'driving_decision_accuracy']
                    if args.reward_mode == 'mixed'
                    else ['driving_no_think_format', 'driving_decision_accuracy'])

        if hasattr(args, 'training_args') and args.training_args is not None:
            setattr(args.training_args, 'inject_gt_on_all_wrong', args.inject_gt_on_all_wrong)
            setattr(args.training_args, 'inject_gt_reward_threshold', args.inject_gt_reward_threshold)

        self._sync_eval_training_args()
        print(f'[DRIVING_DATASET] train={args.dataset}')
        print(f'[DRIVING_DATASET] eval={getattr(args, "val_dataset", [])}')
        print(f'[DRIVING_REWARD] reward_funcs={args.reward_funcs}')
        print(f'[DRIVING_RM] reward_model={getattr(args, "reward_model", None)}')
        print(f'[DRIVING_RM] reward_model_plugin={getattr(args, "reward_model_plugin", None)}')
        print(f'[DRIVING_RM] external_plugins={getattr(args, "external_plugins", None)}')
        return super().run()


def driving_manifest_main(args: Optional[Union[List[str], DrivingManifestRLHFArguments]] = None):
    return DrivingManifestRLHF(args).main()


if __name__ == '__main__':
    driving_manifest_main()
