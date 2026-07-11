#!/usr/bin/env python3
"""Summarize isolated CircuitGCL artifacts without parsing console logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


RUNTIME_ARGS = {
    'git_commit',
    'gpu',
    'log_dir',
    'run_artifact_dir',
    'run_log_path',
}
METRICS = ('val_mse', 'digtime', 'timing_ctrl', 'array')


def _json_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]


def scientific_args(args, include_seed):
    ignored = set(RUNTIME_ARGS)
    if not include_seed:
        ignored.add('seed')
    return {
        key: value
        for key, value in args.items()
        if key not in ignored
    }


def result_mse(result):
    if not result:
        return None
    if result.get('mse_raw') is not None:
        return float(result['mse_raw'])
    if result.get('mse') is not None:
        return float(result['mse'])
    return None


def method_name(args):
    mode = 'no_gcl' if not args.get('sgrl', 0) else args.get('sgrl_mode', 'static')
    parts = [mode, f"loss={args.get('regress_loss', 'unknown')}"]
    if mode == 'joint_shared':
        parts.extend([
            f"lora_r={args.get('joint_lora_rank', 0)}",
            f"layer={args.get('joint_lora_layer', -1)}",
            f"lambda={args.get('joint_gcl_lambda', 0.0)}",
            f"backbone_lr={args.get('joint_backbone_lr', 'default')}",
        ])
    elif mode == 'partial_shared':
        parts.extend([
            f"shared_k={args.get('shared_gnn_layers')}",
            f"fusion={args.get('partial_shared_stats_fusion')}",
            f"freeze={args.get('partial_shared_freeze_epochs')}",
        ])
    return '|'.join(str(part) for part in parts)


def infer_run_name(config_path, config, root=None, prefix_root=False):
    relative_name = None
    if root is not None:
        try:
            relative = config_path.relative_to(root)
            relative_name = (
                relative.parts[0]
                if len(relative.parts) > 1
                else root.name
            )
        except ValueError:
            relative_name = None
    log_dir = config.get('args', {}).get('log_dir')
    name = relative_name or (
        Path(log_dir).name if log_dir else config_path.parent.parent.name
    )
    return f'{root.name}/{name}' if prefix_root and root is not None else name


def load_run(config_path, root=None, prefix_root=False):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metrics_path = config_path.with_name('metrics.json')
    metrics = (
        json.loads(metrics_path.read_text(encoding='utf-8'))
        if metrics_path.is_file()
        else None
    )
    args = config.get('args', {})
    tests = (metrics or {}).get('test_results', {})
    checkpoint = (metrics or {}).get('best_checkpoint')
    anomalies = []
    config_status = config.get('status', 'unknown')
    metric_status = (metrics or {}).get('status')
    if metrics is None:
        anomalies.append('missing_metrics')
    if metric_status and config_status != metric_status:
        anomalies.append(
            f'status_mismatch:{config_status}!={metric_status}'
        )
    if config_status == 'completed':
        if not checkpoint:
            anomalies.append('missing_checkpoint_path')
        elif not Path(checkpoint).is_file():
            anomalies.append('missing_checkpoint_file')

    run_name = infer_run_name(config_path, config, root, prefix_root)
    row = {
        'run_name': run_name,
        'method': method_name(args),
        'seed': int(args.get('seed', -1)),
        'status': config_status,
        'metrics_status': metric_status,
        'started_at': config.get('started_at'),
        'ended_at': config.get('ended_at'),
        'git_commit': config.get('git_commit'),
        'artifact_dir': str(config_path.parent),
        'config_path': str(config_path),
        'checkpoint': checkpoint,
        'best_epoch': (metrics or {}).get('best_epoch'),
        'val_mse': (
            float(metrics['best_val_mse'])
            if metrics and metrics.get('best_val_mse') is not None
            else None
        ),
        'digtime': result_mse(tests.get('digtime')),
        'timing_ctrl': result_mse(tests.get('timing_ctrl')),
        'array': result_mse(tests.get('array_128_32_8t')),
        'loss': args.get('regress_loss'),
        'sgrl_mode': args.get('sgrl_mode'),
        'joint_gcl_lambda': args.get('joint_gcl_lambda'),
        'joint_backbone_lr': args.get('joint_backbone_lr'),
        'joint_lora_rank': args.get('joint_lora_rank'),
        'joint_lora_layer': args.get('joint_lora_layer'),
        'config_fingerprint': _json_fingerprint(
            scientific_args(args, include_seed=True)
        ),
        'group_fingerprint': _json_fingerprint(
            scientific_args(args, include_seed=False)
        ),
        'canonical': False,
        'anomalies': anomalies,
    }
    return row


def discover_runs(roots):
    roots = [Path(root).resolve() for root in roots]
    path_roots = {}
    for root in roots:
        if root.is_file() and root.name == 'run_config.json':
            path_roots[root] = root.parent
        elif root.exists():
            for path in root.rglob('run_config.json'):
                path_roots[path.resolve()] = root
    prefix_root = len(roots) > 1
    rows = [
        load_run(path, path_roots[path], prefix_root)
        for path in sorted(path_roots)
    ]
    mark_canonical_and_duplicates(rows)
    return rows


def mark_canonical_and_duplicates(rows):
    exact = {}
    by_group_seed = {}
    for row in rows:
        exact.setdefault(row['config_fingerprint'], []).append(row)
        by_group_seed.setdefault(
            (row['group_fingerprint'], row['seed']), []
        ).append(row)

    for duplicate_rows in exact.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                row['anomalies'].append(
                    f'duplicate_config:{len(duplicate_rows)}'
                )

    for candidate_rows in by_group_seed.values():
        completed = [row for row in candidate_rows if row['status'] == 'completed']
        pool = completed or candidate_rows
        selected = sorted(
            pool,
            key=lambda row: (row['started_at'] or '', row['artifact_dir']),
        )[0]
        selected['canonical'] = True


def metric_stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {'count': 0, 'mean': None, 'pstdev': None, 'values': []}
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'pstdev': statistics.pstdev(values),
        'values': values,
    }


def summarize_groups(rows):
    groups_by_fingerprint = {}
    for row in rows:
        if not row['canonical'] or row['status'] != 'completed':
            continue
        groups_by_fingerprint.setdefault(row['group_fingerprint'], []).append(row)

    method_counts = {}
    for group_rows in groups_by_fingerprint.values():
        method = group_rows[0]['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    summaries = {}
    for fingerprint, group_rows in sorted(groups_by_fingerprint.items()):
        group_rows.sort(key=lambda row: row['seed'])
        method = group_rows[0]['method']
        name = (
            method
            if method_counts[method] == 1
            else f'{method}@{fingerprint}'
        )
        summaries[name] = {
            'method': method,
            'seeds': [row['seed'] for row in group_rows],
            'num_runs': len(group_rows),
            'group_fingerprint': fingerprint,
            'run_names': sorted({row['run_name'] for row in group_rows}),
            'metrics': {
                metric: metric_stats([row[metric] for row in group_rows])
                for metric in METRICS
            },
        }
    return summaries


def paired_differences(rows, baseline):
    summaries = summarize_groups(rows)
    matching_baselines = [
        name for name, summary in summaries.items()
        if name == baseline or summary['method'] == baseline
    ]
    if len(matching_baselines) != 1:
        raise ValueError(
            f'Baseline {baseline!r} matched {len(matching_baselines)} groups: '
            f'{matching_baselines}'
        )
    baseline_name = matching_baselines[0]
    group_by_fingerprint = {
        summary['group_fingerprint']: name
        for name, summary in summaries.items()
    }
    canonical = {}
    for row in rows:
        if row['canonical'] and row['status'] == 'completed':
            name = group_by_fingerprint[row['group_fingerprint']]
            canonical[(name, row['seed'])] = row
    baseline_seeds = {
        seed: row for (name, seed), row in canonical.items()
        if name == baseline_name
    }
    result = {}
    for name in sorted({name for name, _seed in canonical} - {baseline_name}):
        pairs = []
        for seed, baseline_row in baseline_seeds.items():
            candidate = canonical.get((name, seed))
            if candidate is not None:
                pairs.append((seed, baseline_row, candidate))
        if not pairs:
            continue
        result[name] = {
            'baseline': baseline_name,
            'seeds': [seed for seed, _base, _candidate in pairs],
            'metrics': {},
        }
        for metric in METRICS:
            differences = [
                candidate[metric] - base[metric]
                for _seed, base, candidate in pairs
                if candidate[metric] is not None and base[metric] is not None
            ]
            result[name]['metrics'][metric] = metric_stats(differences)
    return result


def write_runs_tsv(rows, path):
    columns = [
        'run_name', 'method', 'seed', 'status', 'canonical', 'best_epoch',
        *METRICS, 'loss', 'sgrl_mode', 'joint_gcl_lambda',
        'joint_backbone_lr', 'joint_lora_rank', 'joint_lora_layer',
        'git_commit', 'artifact_dir', 'anomalies',
    ]
    lines = ['\t'.join(columns)]
    for row in sorted(rows, key=lambda item: (
        item['run_name'], item['seed'], item['started_at'] or ''
    )):
        values = []
        for column in columns:
            value = row[column]
            if column == 'anomalies':
                value = ';'.join(value)
            values.append('' if value is None else str(value))
        lines.append('\t'.join(values))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Summarize CircuitGCL run_config.json and metrics.json artifacts.'
    )
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument(
        '--baseline',
        help='Unique method/group name used for paired seed differences.',
    )
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.roots[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = discover_runs(args.roots)
    groups = summarize_groups(rows)
    paired = paired_differences(rows, args.baseline) if args.baseline else {}
    payload = {
        'roots': [str(root) for root in args.roots],
        'baseline': args.baseline,
        'num_artifacts': len(rows),
        'num_completed': sum(row['status'] == 'completed' for row in rows),
        'num_running': sum(row['status'] == 'running' for row in rows),
        'num_anomalous': sum(bool(row['anomalies']) for row in rows),
        'groups': groups,
        'paired_differences': paired,
        'runs': rows,
    }
    json_path = output_dir / 'experiment_summary.json'
    tsv_path = output_dir / 'experiment_runs.tsv'
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_runs_tsv(rows, tsv_path)
    print(json.dumps({
        key: payload[key]
        for key in ('num_artifacts', 'num_completed', 'num_running', 'num_anomalous')
    }, indent=2))
    print(f'JSON: {json_path}')
    print(f'TSV: {tsv_path}')


if __name__ == '__main__':
    main()
