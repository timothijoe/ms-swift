#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coc_trace_rl.src.driving_dataset import prepare_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Persist CoC Trace schemas in a driving JSONL dataset.')
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_path', type=Path)
    args = parser.parse_args()
    print(f'prepared {prepare_jsonl(args.input_path, args.output_path)} rows')


if __name__ == '__main__':
    main()
