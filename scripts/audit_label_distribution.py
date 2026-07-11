#!/usr/bin/env python3
"""Audit source/transfer label imbalance with fixed normalized-label bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_train import regression_bin_metrics
from sram_dataset import performat_SramDataset


QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def distribution(values, bin_edges, class_boundaries):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    counts, _ = np.histogram(values, bins=bin_edges)
    class_ids = np.digitize(values, class_boundaries, right=False)
    class_counts = np.bincount(
        class_ids,
        minlength=len(class_boundaries) + 1,
    )
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'quantiles': {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in QUANTILES
        },
        'histogram': {
            'bin_edges': [float(value) for value in bin_edges],
            'counts': [int(value) for value in counts],
            'fractions': [float(value / values.size) for value in counts],
        },
        'class_boundaries': [float(value) for value in class_boundaries],
        'class_counts': [int(value) for value in class_counts],
        'class_fractions': [float(value / values.size) for value in class_counts],
    }


def jensen_shannon(source_counts, target_counts):
    source = np.asarray(source_counts, dtype=np.float64) + 1e-12
    target = np.asarray(target_counts, dtype=np.float64) + 1e-12
    source /= source.sum()
    target /= target.sum()
    midpoint = 0.5 * (source + target)
    return float(0.5 * (
        np.sum(source * np.log(source / midpoint))
        + np.sum(target * np.log(target / midpoint))
    ))


def compare_to_source(source_values, target_values, bin_edges):
    source_values = np.asarray(source_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    source_counts, _ = np.histogram(source_values, bins=bin_edges)
    target_counts, _ = np.histogram(target_values, bins=bin_edges)
    low = float(np.quantile(source_values, 0.1))
    high = float(np.quantile(source_values, 0.9))
    positive_counts = source_counts[source_counts > 0]
    rare_threshold = float(np.quantile(positive_counts, 0.25))
    rare_bins = np.flatnonzero(source_counts <= rare_threshold)
    target_bins = np.clip(
        np.digitize(target_values, bin_edges[1:-1], right=False),
        0,
        len(bin_edges) - 2,
    )
    return {
        'jensen_shannon_10bin': jensen_shannon(source_counts, target_counts),
        'source_q10': low,
        'source_q90': high,
        'target_fraction_below_source_q10': float(np.mean(target_values <= low)),
        'target_fraction_above_source_q90': float(np.mean(target_values >= high)),
        'rare_source_bin_count_threshold': rare_threshold,
        'rare_source_bins': [int(value) for value in rare_bins],
        'target_fraction_in_rare_source_bins': float(
            np.mean(np.isin(target_bins, rare_bins))
        ),
    }


def load_labels(args):
    dataset = performat_SramDataset(
        name=args.dataset,
        dataset_dir=args.dataset_dir,
        neg_edge_ratio=0.0,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level='edge',
        net_only=True,
        class_boundaries=args.class_boundaries,
    )
    dataset.norm_nfeat([0, 1])
    labels = {}
    for index, name in enumerate(dataset.names):
        edge_label = dataset[index].edge_label.detach().cpu().numpy()
        labels[name] = edge_label[:, 0] if edge_label.ndim == 2 else edge_label
    return labels


def load_predictions(path):
    if path is None:
        return {}
    archive = np.load(path)
    circuits = {}
    for key in archive.files:
        if key.endswith('_labels'):
            name = key[:-len('_labels')]
            prediction_key = f'{name}_predictions'
            if prediction_key not in archive:
                raise ValueError(f'Missing {prediction_key} in {path}.')
            circuits[name] = {
                'labels': archive[key],
                'predictions': archive[prediction_key],
            }
    return circuits


def write_tsv(payload, path):
    lines = [
        'circuit\tcount\tmean\tstd\tq10\tq50\tq90\tjs_from_source'
    ]
    for name, summary in payload['circuits'].items():
        comparison = payload['comparisons_to_source'].get(name, {})
        quantiles = summary['quantiles']
        lines.append('\t'.join(str(value) for value in (
            name,
            summary['count'],
            summary['mean'],
            summary['std'],
            quantiles['0.1'],
            quantiles['0.5'],
            quantiles['0.9'],
            comparison.get('jensen_shannon_10bin', 0.0),
        )))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_distributions(labels, bin_edges, path):
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, values in labels.items():
        counts, _ = np.histogram(values, bins=bin_edges)
        fractions = counts / counts.sum()
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        axis.plot(centers, fractions, marker='o', linewidth=1.5, label=name)
    axis.set_xlabel('Normalized capacitance label')
    axis.set_ylabel('Fraction per bin')
    axis.set_title('Circuit label distributions')
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ssram+digtime+timing_ctrl+array_128_32_8t',
    )
    parser.add_argument('--dataset_dir', default='./datasets/')
    parser.add_argument('--small_dataset_sample_rates', type=float, default=1.0)
    parser.add_argument('--large_dataset_sample_rates', type=float, default=0.1)
    parser.add_argument(
        '--class_boundaries',
        type=float,
        nargs='+',
        default=[0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--predictions_npz', type=Path)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('logs/label_distribution_audit_20260711'),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args)
    source_name = next(iter(labels))
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    payload = {
        'source_circuit': source_name,
        'bin_edges': [float(value) for value in bin_edges],
        'circuits': {
            name: distribution(values, bin_edges, args.class_boundaries)
            for name, values in labels.items()
        },
        'comparisons_to_source': {
            name: compare_to_source(labels[source_name], values, bin_edges)
            for name, values in labels.items()
            if name != source_name
        },
        'prediction_bin_metrics': {},
    }
    for name, arrays in load_predictions(args.predictions_npz).items():
        payload['prediction_bin_metrics'][name] = regression_bin_metrics(
            arrays['labels'],
            arrays['predictions'],
            bin_edges,
        )

    json_path = args.output_dir / 'label_distribution.json'
    tsv_path = args.output_dir / 'label_distribution.tsv'
    plot_path = args.output_dir / 'label_distribution.png'
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_tsv(payload, tsv_path)
    plot_distributions(labels, bin_edges, plot_path)
    print(json.dumps({
        'source_circuit': source_name,
        'circuit_counts': {
            name: int(values.size) for name, values in labels.items()
        },
        'comparisons_to_source': payload['comparisons_to_source'],
    }, indent=2, sort_keys=True))
    print(f'JSON: {json_path}')
    print(f'TSV: {tsv_path}')
    print(f'Plot: {plot_path}')


if __name__ == '__main__':
    main()
