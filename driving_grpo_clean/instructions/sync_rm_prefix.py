#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_text(path: str) -> str:
    return Path(path).read_text(encoding='utf-8').strip()


def main():
    parser = argparse.ArgumentParser(description='Safely sync long RM prefix texts into JSON')
    parser.add_argument('--type', required=True, help='template type key, e.g. default')
    parser.add_argument('--system', required=True, help='path to system prompt txt')
    parser.add_argument('--instruction', required=True, help='path to instruction txt')
    parser.add_argument('--fewshot', required=True, help='path to few-shot txt')
    parser.add_argument(
        '--json',
        default='driving_grpo_clean/instructions/rm_templates_prefix.json',
        help='target json file')
    args = parser.parse_args()

    json_path = Path(args.json)
    obj = json.loads(json_path.read_text(encoding='utf-8'))

    if args.type not in obj:
        obj[args.type] = {
            'system_prompt_prefix': '',
            'instruction_prefix': '',
            'few_shot_prefix': ''
        }

    obj[args.type]['system_prompt_prefix'] = read_text(args.system)
    obj[args.type]['instruction_prefix'] = read_text(args.instruction)
    obj[args.type]['few_shot_prefix'] = read_text(args.fewshot)

    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'updated {json_path}')
    print('type=', args.type)
    print('len(system)=', len(obj[args.type]['system_prompt_prefix']))
    print('len(instruction)=', len(obj[args.type]['instruction_prefix']))
    print('len(few_shot)=', len(obj[args.type]['few_shot_prefix']))


if __name__ == '__main__':
    main()
