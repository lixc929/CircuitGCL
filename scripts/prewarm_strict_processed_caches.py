#!/usr/bin/env python3
"""Serially materialize and validate processed caches before parallel runs."""

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sram_dataset import performat_SramDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ssram+digtime+timing_ctrl+array_128_32_8t',
    )
    parser.add_argument('--relation_sample_seed', type=int, default=20260711)
    parser.add_argument('--small_dataset_sample_rates', type=float, default=1.0)
    parser.add_argument('--large_dataset_sample_rates', type=float, default=0.1)
    args = parser.parse_args()

    dataset = performat_SramDataset(
        name=args.dataset,
        dataset_dir='./datasets/',
        neg_edge_ratio=0.0,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level='edge',
        net_only=True,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
        relation_sample_seed=args.relation_sample_seed,
    )
    print(json.dumps(
        dataset.processed_cache_provenance,
        indent=2,
        sort_keys=True,
    ))


if __name__ == '__main__':
    main()
