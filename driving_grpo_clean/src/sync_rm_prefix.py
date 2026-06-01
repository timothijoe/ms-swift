#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def read_text(path: str) -> str:
    return Path(path).read_text(encoding='utf-8').strip()


def compact_text(text: str, mode: str) -> str:
    if mode == 'none':
        return text.strip()
    # common cleanup
    text = '\n'.join([line.rstrip() for line in text.splitlines()])
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if mode == 'light':
        return text
    # aggressive: reduce blank lines further and trim repeated spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    # rough estimate for CJK/mixed text
    return max(1, int(len(text) / 2.2))


def build_report(label: str, text: str) -> str:
    chars = len(text)
    lines = len(text.splitlines())
    tokens = estimate_tokens(text)
    has_output = ('输出格式' in text) or ('只输出JSON' in text) or ('strictly output' in text.lower())
    has_rules = ('规则' in text) or ('评分' in text) or ('score' in text.lower())
    has_fewshot = ('few-shot' in text.lower()) or ('示例' in text) or ('candidate' in text.lower())
    flags = f'output={has_output}, rules={has_rules}, fewshot={has_fewshot}'
    return f'[{label}] chars={chars}, lines={lines}, est_tokens={tokens}, {flags}'


def main():
    parser = argparse.ArgumentParser(description='Safely sync long RM prefix texts into JSON')
    parser.add_argument('--type', required=True, help='template type key, e.g. default')
    parser.add_argument('--system', required=True, help='path to system prompt txt')
    parser.add_argument('--instruction', required=True, help='path to instruction txt')
    parser.add_argument('--fewshot', required=True, help='path to few-shot txt')
    parser.add_argument(
        '--compact',
        choices=['none', 'light', 'aggressive'],
        default='light',
        help='compact mode when writing into json')
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='only print quality/size report, do not write json')
    parser.add_argument(
        '--json',
        default='driving_grpo_clean/configs/rm_templates_prefix.json',
        help='target json file')
    args = parser.parse_args()

    system_raw = read_text(args.system)
    instruction_raw = read_text(args.instruction)
    fewshot_raw = read_text(args.fewshot)

    system_text = compact_text(system_raw, args.compact)
    instruction_text = compact_text(instruction_raw, args.compact)
    fewshot_text = compact_text(fewshot_raw, args.compact)

    print(build_report('system(raw)', system_raw))
    print(build_report('instruction(raw)', instruction_raw))
    print(build_report('fewshot(raw)', fewshot_raw))
    print(build_report(f'system({args.compact})', system_text))
    print(build_report(f'instruction({args.compact})', instruction_text))
    print(build_report(f'fewshot({args.compact})', fewshot_text))

    if args.check_only:
        return

    json_path = Path(args.json)
    obj = json.loads(json_path.read_text(encoding='utf-8'))

    if args.type not in obj:
        obj[args.type] = {
            'system_prompt_prefix': '',
            'instruction_prefix': '',
            'few_shot_prefix': ''
        }

    obj[args.type]['system_prompt_prefix'] = system_text
    obj[args.type]['instruction_prefix'] = instruction_text
    obj[args.type]['few_shot_prefix'] = fewshot_text

    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'updated {json_path}')
    print('type=', args.type)
    print('len(system)=', len(obj[args.type]['system_prompt_prefix']))
    print('len(instruction)=', len(obj[args.type]['instruction_prefix']))
    print('len(few_shot)=', len(obj[args.type]['few_shot_prefix']))


if __name__ == '__main__':
    main()
