import os
import sys

try:
    from .driving_manifest_main import driving_manifest_main
except ImportError:
    from driving_manifest_main import driving_manifest_main


def build_args():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    args = [
        '--dataset_manifest_path',
        os.path.join(script_dir, 'datasets_manifest.json'),
        '--manifest_register_file',
        os.path.join(script_dir, 'manifest_dataset_register.py'),
        '--use_manifest_as_dataset',
        'true',
        '--reward_funcs',
        'driving_no_think_format',
        'driving_decision_accuracy',
        '--output_dir',
        os.path.join(project_root, 'output', 'GRPO_DRIVING_MANIFEST_CLEAN'),
        '--max_length',
        '1024',
        '--eval_strategy',
        'steps',
        '--eval_on_start',
        'true',
        '--eval_steps',
        '8',
        '--do_eval',
        'true',
        '--beta',
        '0.1',
    ]

    rm_model = os.getenv('DRIVING_RM_MODEL', '').strip()
    if rm_model:
        args += [
            '--external_plugins',
            os.path.join(script_dir, 'driving_rm_plugin.py'),
            '--reward_model',
            rm_model,
            '--reward_model_plugin',
            'driving_rubric_rm',
            '--reward_weights',
            os.getenv('DRIVING_REWARD_W1', '0.4'),
            os.getenv('DRIVING_REWARD_W2', '0.4'),
            os.getenv('DRIVING_REWARD_W3', '0.2'),
        ]
    return args


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('NPROC_PER_NODE', '1')
    cli_args = build_args()
    print('[DEBUG_ENTRY] argv:', ' '.join([sys.executable, __file__] + cli_args))
    driving_manifest_main(cli_args)
