#!/usr/bin/env python3
"""Build the canonical teacher-facing summary for strict reuse experiments.

The report is intentionally tied to the four preregistered 2026-07-12 roots.
It validates the completed artifacts before combining seeds, then emits one
JSON file for traceability and one TSV file for convenient inspection.  Raw
checkpoints and run artifacts remain in their original directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile


DATASET = 'ssram+digtime+timing_ctrl+array_128_32_8t'
TRANSFER_NAMES = ('digtime', 'timing_ctrl', 'array_128_32_8t')
METRICS = ('val_mse', 'digtime', 'timing_ctrl', 'array', 'transfer_mean')
FIXED_VIEW_SEED = 20260711
EXPECTED_AUDIT_EPOCHS = list(range(0, 160, 10))
GATES = {
    'val_mse': 0.05,
    'transfer_mean': 0.10,
    'digtime': 0.25,
    'timing_ctrl': 0.25,
    'array': 0.25,
}

# root -> run directory -> (stable method id, seed)
EXPECTED_MATRIX = {
    'strict_seed0_seven_20260712_v2': {
        'static_dual': ('static_dual', 0),
        'static_ema': ('static_ema_only', 0),
        'no_gcl': ('no_gcl', 0),
        'init_reuse': ('init_reuse', 0),
        'joint_lambda0': ('joint_rank0_lambda0', 0),
        'lora_r8_lambda0': ('joint_lora_r8_lambda0', 0),
        'lora_r8_lambda005': ('joint_lora_r8_lambda005', 0),
    },
    'strict_selection_seeds12_20260712': {
        'static_dual_seed1': ('static_dual', 1),
        'static_dual_seed2': ('static_dual', 2),
        'no_gcl_seed1': ('no_gcl', 1),
        'no_gcl_seed2': ('no_gcl', 2),
        'init_reuse_seed1': ('init_reuse', 1),
        'init_reuse_seed2': ('init_reuse', 2),
        'joint_lambda0_seed1': ('joint_rank0_lambda0', 1),
        'joint_lambda0_seed2': ('joint_rank0_lambda0', 2),
        'lora_r8_lambda005_seed1': ('joint_lora_r8_lambda005', 1),
        'lora_r8_lambda005_seed2': ('joint_lora_r8_lambda005', 2),
    },
    'strict_joint_rank0_lambda005_seed0_20260712': {
        'joint_rank0_lambda005_seed0': ('joint_rank0_lambda005', 0),
    },
    'strict_joint_rank0_lambda005_seeds12_audit_20260712': {
        'joint_rank0_lambda005_seed1': ('joint_rank0_lambda005', 1),
        'joint_rank0_lambda005_seed2': ('joint_rank0_lambda005', 2),
    },
}

EXPECTED_METHOD_SEEDS = {
    'static_dual': {0, 1, 2},
    'static_ema_only': {0},
    'no_gcl': {0, 1, 2},
    'init_reuse': {0, 1, 2},
    'joint_rank0_lambda0': {0, 1, 2},
    'joint_lora_r8_lambda0': {0},
    'joint_lora_r8_lambda005': {0, 1, 2},
    'joint_rank0_lambda005': {0, 1, 2},
}

COMMON_EXPECTATIONS = {
    'task_level': 'edge',
    'task': 'regression',
    'dataset': DATASET,
    'neg_edge_ratio': 0.0,
    'net_only': True,
    'protocol': 'strict_inductive',
    'sgrl_graph_scope': 'source',
    'normalization_scope': 'source',
    'small_dataset_sample_rates': 1.0,
    'large_dataset_sample_rates': 0.1,
    'num_hops': 2,
    'num_neighbors': 8,
    'num_workers': 0,
    'epochs': 160,
    'early_stopping_patience': 0,
    'early_stopping_min_delta': 0.0,
    'batch_size': 512,
    'lr': 0.0001,
    'momentum': 0.99,
    'weight_decay': 0.0,
    'cl_model': 'clustergcn',
    'cl_act_fn': 'tanh',
    'cl_epochs': 5,
    'cl_gnn_layers': 2,
    'cl_hid_dim': 64,
    'cl_batch_size': 32768,
    'cl_num_neighbors': 8,
    'cl_dropout': 0.3,
    'model': 'clustergcn',
    'num_gnn_layers': 2,
    'num_head_layers': 2,
    'dropout': 0.1,
    'use_bn': 0,
    'act_fn': 'prelu',
    'use_stats': 1,
    'src_dst_agg': 'concat',
    'regress_loss': 'mse',
    'class_loss': 'bsmCE',
    'num_classes': 5,
    'class_boundaries': [0.2, 0.4, 0.6, 0.8],
    'noise_sigma': 0.001,
    'lds_kernel': 'gaussian',
    'lds_ks': 9,
    'lds_sigma': 0.02,
    'sgrl_online_lr': 1e-6,
    'sgrl_reuse_stats': 1,
    'sgrl_reuse_stats_fusion': 'concat',
    'joint_shared_audit': 1,
    'joint_shared_audit_interval': 10,
}

JOINT_COMMON_EXPECTATIONS = {
    'joint_shared_gnn_layers': 2,
    'joint_lora_layer': -1,
    'joint_lora_alpha': None,
    'joint_backbone_lr': None,
}

METHOD_EXPECTATIONS = {
    'static_dual': {
        'sgrl': 1, 'sgrl_mode': 'static', 'hid_dim': 63,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
    'static_ema_only': {
        'sgrl': 1, 'sgrl_mode': 'static', 'hid_dim': 63,
        'sgrl_pretrain_target_update': 'circuitgcl_text_ema_only',
    },
    'no_gcl': {'sgrl': 0, 'sgrl_mode': 'static', 'hid_dim': 64},
    'init_reuse': {
        'sgrl': 1, 'sgrl_mode': 'init_reuse', 'hid_dim': 64,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
    'joint_rank0_lambda0': {
        'sgrl': 1, 'sgrl_mode': 'joint_shared', 'hid_dim': 64,
        'joint_lora_rank': 0,
        'joint_gcl_lambda': 0.0,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
    'joint_lora_r8_lambda0': {
        'sgrl': 1, 'sgrl_mode': 'joint_shared', 'hid_dim': 64,
        'joint_lora_rank': 8, 'joint_gcl_lambda': 0.0,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
    'joint_lora_r8_lambda005': {
        'sgrl': 1, 'sgrl_mode': 'joint_shared', 'hid_dim': 64,
        'joint_lora_rank': 8, 'joint_gcl_lambda': 0.05,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
    'joint_rank0_lambda005': {
        'sgrl': 1, 'sgrl_mode': 'joint_shared', 'hid_dim': 64,
        'joint_lora_rank': 0,
        'joint_gcl_lambda': 0.05,
        'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    },
}

PAIR_PROVENANCE_FIELDS = (
    'sgrl_cache_key',
    'sgrl_checkpoint_sha256',
    'split_fingerprint',
    'normalization_state_sha256',
    'downstream_initial_model_fingerprint',
    'train_sampler_fingerprint',
    'eval_sampler_fingerprints',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path, repo_root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def result_mse(result):
    if not isinstance(result, dict):
        raise ValueError('Expected a metric-result object.')
    value = result.get('mse_raw', result.get('mse'))
    if value is None or not math.isfinite(float(value)):
        raise ValueError('Metric result is missing a finite raw MSE.')
    return float(value)


def quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError('Cannot compute a quantile of an empty sequence.')
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def summary_stats(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Summary values must be a non-empty finite sequence.')
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'pstdev': statistics.pstdev(values),
        'min': min(values),
        'q10': quantile(values, 0.10),
        'q25': quantile(values, 0.25),
        'median': statistics.median(values),
        'q75': quantile(values, 0.75),
        'q90': quantile(values, 0.90),
        'max': max(values),
        'negative_fraction': sum(value < 0.0 for value in values) / len(values),
        'values': values,
    }


def _expect(args, expectations, context):
    for key, expected in expectations.items():
        if key not in args:
            raise ValueError(f'{context} is missing required argument {key}.')
        actual = args[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f'{context} has mismatched {key}: {actual!r} != {expected!r}'
            )


def _verify_hashed_paths(payload, seen, context):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.endswith('_path') and isinstance(value, str):
                sha_key = f'{key[:-5]}_sha256'
                if sha_key in payload:
                    path = Path(value)
                    expected = payload[sha_key]
                    identity = (str(path.resolve()), expected)
                    if identity not in seen:
                        if not path.is_file():
                            raise ValueError(f'{context} is missing provenance file {path}.')
                        actual = file_sha256(path)
                        if actual != expected:
                            raise ValueError(
                                f'{context} provenance SHA mismatch for {path}.'
                            )
                        seen.add(identity)
            _verify_hashed_paths(value, seen, context)
    elif isinstance(payload, list):
        for value in payload:
            _verify_hashed_paths(value, seen, context)


def _validate_seed_protocol(args, seed, context):
    if args.get('seed') != seed:
        raise ValueError(f'{context} has mismatched seed.')
    for key in (
        'pretraining_seed', 'downstream_seed', 'train_sampler_seed', 'split_seed'
    ):
        if args.get(key) != seed:
            raise ValueError(f'{context} has mismatched paired stage seed {key}.')
    for key in ('embedding_inference_seed', 'relation_sample_seed', 'eval_seed'):
        if args.get(key) != FIXED_VIEW_SEED:
            raise ValueError(f'{context} has mismatched fixed view seed {key}.')


def _expected_resolved_seeds(seed):
    return {
        'pretraining_seed': seed,
        'embedding_inference_seed': FIXED_VIEW_SEED,
        'downstream_seed': seed,
        'train_sampler_seed': seed,
        'relation_sample_seed': FIXED_VIEW_SEED,
        'split_seed': seed,
        'eval_seed': FIXED_VIEW_SEED,
    }


def _artifact_for_run(root, run_name):
    paths = sorted((root / run_name).rglob('run_config.json'))
    if len(paths) != 1:
        raise ValueError(
            f'{root.name}/{run_name} must contain exactly one run_config.json; '
            f'found {len(paths)}.'
        )
    return paths[0]


def load_run(root, run_name, method, seed, repo_root, hashed_paths):
    config_path = _artifact_for_run(root, run_name)
    metrics_path = config_path.with_name('metrics.json')
    context = f'{root.name}/{run_name}'
    if not metrics_path.is_file():
        raise ValueError(f'{context} is missing metrics.json.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if config.get('status') != 'completed' or metrics.get('status') != 'completed':
        raise ValueError(f'{context} is not completed.')
    args = config.get('args')
    if not isinstance(args, dict):
        raise ValueError(f'{context} has no argument object.')
    _expect(args, COMMON_EXPECTATIONS, context)
    _expect(args, METHOD_EXPECTATIONS[method], context)
    if args.get('sgrl_mode') == 'joint_shared':
        _expect(args, JOINT_COMMON_EXPECTATIONS, context)
    _validate_seed_protocol(args, seed, context)
    if 'sp8192w' in args['dataset'].lower():
        raise ValueError(f'{context} illegally includes the blind circuit.')

    runtime = config.get('runtime_metadata')
    if not isinstance(runtime, dict):
        raise ValueError(f'{context} is missing runtime provenance.')
    if runtime.get('source_graph_names') != ['ssram']:
        raise ValueError(f'{context} has mismatched source graphs.')
    if runtime.get('transfer_graph_names') != list(TRANSFER_NAMES):
        raise ValueError(f'{context} has mismatched transfer graphs.')
    if runtime.get('resolved_seeds') != _expected_resolved_seeds(seed):
        raise ValueError(f'{context} has mismatched resolved seed provenance.')
    if args.get('sgrl', 0) and runtime.get('sgrl_train_graph_names') != ['ssram']:
        raise ValueError(f'{context} did not fit SGRL on source only.')
    _verify_hashed_paths(runtime, hashed_paths, context)

    audit_root = 'strict_joint_rank0_lambda005_seeds12_audit_20260712'
    if root.name == audit_root:
        _expect(args, {
            'joint_gradient_audit': 1,
            'joint_gradient_audit_interval': 10,
        }, context)
    elif args.get('joint_gradient_audit', 0):
        raise ValueError(f'{context} unexpectedly enabled gradient auditing.')

    checkpoint = Path(metrics.get('best_checkpoint', ''))
    if not checkpoint.is_file():
        raise ValueError(f'{context} is missing its best checkpoint.')
    val_mse = float(metrics.get('best_val_mse'))
    if not math.isfinite(val_mse):
        raise ValueError(f'{context} has a non-finite best validation MSE.')
    best_epoch = metrics.get('best_epoch')
    if (
        type(best_epoch) is not int
        or best_epoch < 0
        or best_epoch >= args['epochs']
    ):
        raise ValueError(f'{context} has an invalid best epoch: {best_epoch!r}.')
    restored_val = result_mse(metrics.get('validation_results'))
    if not math.isclose(val_mse, restored_val, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f'{context} best validation MSE is not reproducible.')
    tests = metrics.get('test_results')
    if not isinstance(tests, dict) or set(tests) != set(TRANSFER_NAMES):
        raise ValueError(f'{context} has an unexpected transfer-result set.')
    transfer = {name: result_mse(tests[name]) for name in TRANSFER_NAMES}
    values = {
        'val_mse': val_mse,
        'digtime': transfer['digtime'],
        'timing_ctrl': transfer['timing_ctrl'],
        'array': transfer['array_128_32_8t'],
    }
    values['transfer_mean'] = statistics.mean(
        values[name] for name in ('digtime', 'timing_ctrl', 'array')
    )

    return {
        'method': method,
        'seed': seed,
        'root': root.name,
        'run_name': run_name,
        'git_commit': config.get('git_commit'),
        'best_epoch': best_epoch,
        **values,
        'artifact_dir': repo_relative(config_path.parent, repo_root),
        'config_path': repo_relative(config_path, repo_root),
        'config_sha256': file_sha256(config_path),
        'metrics_path': repo_relative(metrics_path, repo_root),
        'metrics_sha256': file_sha256(metrics_path),
        'checkpoint_path': repo_relative(checkpoint, repo_root),
        'checkpoint_sha256': file_sha256(checkpoint),
        'resolved_seeds': runtime.get('resolved_seeds'),
        'provenance': {
            key: runtime.get(key)
            for key in PAIR_PROVENANCE_FIELDS
        },
        '_runtime_metadata': runtime,
        '_metrics': metrics,
    }


def validate_matrix(roots):
    roots_by_name = {Path(root).resolve().name: Path(root).resolve() for root in roots}
    if set(roots_by_name) != set(EXPECTED_MATRIX):
        raise ValueError(
            'Strict reuse report requires exactly these roots: '
            f'{sorted(EXPECTED_MATRIX)}; got {sorted(roots_by_name)}.'
        )
    for root_name, expected_runs in EXPECTED_MATRIX.items():
        root = roots_by_name[root_name]
        if not root.is_dir():
            raise ValueError(f'Missing strict root: {root}')
        discovered = {
            path.relative_to(root).parts[0]
            for path in root.rglob('run_config.json')
        }
        if discovered != set(expected_runs):
            raise ValueError(
                f'{root_name} run matrix mismatch: expected '
                f'{sorted(expected_runs)}, found {sorted(discovered)}.'
            )
    return roots_by_name


def group_runs(runs):
    by_method = {}
    for run in runs:
        by_method.setdefault(run['method'], []).append(run)
    if set(by_method) != set(EXPECTED_METHOD_SEEDS):
        raise ValueError('Completed method set does not match the strict matrix.')
    summaries = {}
    for method, method_runs in sorted(by_method.items()):
        method_runs.sort(key=lambda run: run['seed'])
        seeds = {run['seed'] for run in method_runs}
        if seeds != EXPECTED_METHOD_SEEDS[method]:
            raise ValueError(
                f'{method} seed set mismatch: {sorted(seeds)} != '
                f'{sorted(EXPECTED_METHOD_SEEDS[method])}.'
            )
        summaries[method] = {
            'seeds': [run['seed'] for run in method_runs],
            'metrics': {
                metric: summary_stats([run[metric] for run in method_runs])
                for metric in METRICS
            },
        }
    return summaries, by_method


def compare_methods(candidate, reference, by_method):
    candidate_runs = {run['seed']: run for run in by_method[candidate]}
    reference_runs = {run['seed']: run for run in by_method[reference]}
    seeds = sorted(set(candidate_runs) & set(reference_runs))
    if seeds != [0, 1, 2]:
        raise ValueError(
            f'{candidate} vs {reference} requires paired seeds 0,1,2.'
        )
    metrics = {}
    for metric in METRICS:
        candidate_values = [candidate_runs[seed][metric] for seed in seeds]
        reference_values = [reference_runs[seed][metric] for seed in seeds]
        relative = [
            candidate_runs[seed][metric] / reference_runs[seed][metric] - 1.0
            for seed in seeds
        ]
        metrics[metric] = {
            'candidate_mean': statistics.mean(candidate_values),
            'reference_mean': statistics.mean(reference_values),
            'delta_of_means': (
                statistics.mean(candidate_values) /
                statistics.mean(reference_values) - 1.0
            ),
            'paired_relative': {
                'seeds': seeds,
                **summary_stats(relative),
            },
        }
    return {'candidate': candidate, 'reference': reference, 'metrics': metrics}


def promotion_gate(comparison):
    observed = {
        metric: comparison['metrics'][metric]['delta_of_means']
        for metric in GATES
    }
    checks = {
        metric: {
            'observed_degradation': observed[metric],
            'maximum_degradation': threshold,
            'passes': observed[metric] <= threshold,
        }
        for metric, threshold in GATES.items()
    }
    paired_seeds = comparison['metrics']['val_mse']['paired_relative']['seeds']
    per_seed = {}
    for index, seed in enumerate(paired_seeds):
        seed_checks = {
            metric: {
                'observed_degradation': comparison['metrics'][metric][
                    'paired_relative'
                ]['values'][index],
                'maximum_degradation': threshold,
                'passes': comparison['metrics'][metric][
                    'paired_relative'
                ]['values'][index] <= threshold,
            }
            for metric, threshold in GATES.items()
        }
        per_seed[str(seed)] = {
            'checks': seed_checks,
            'passes': all(check['passes'] for check in seed_checks.values()),
        }
    return {
        'candidate': comparison['candidate'],
        'reference': comparison['reference'],
        'checks': checks,
        'per_seed': per_seed,
        'passes': all(check['passes'] for check in checks.values()),
    }


def validate_lambda_pairing(by_method):
    baseline = {run['seed']: run for run in by_method['joint_rank0_lambda0']}
    candidate = {run['seed']: run for run in by_method['joint_rank0_lambda005']}
    per_seed = {}
    for seed in (0, 1, 2):
        mismatches = []
        for field in PAIR_PROVENANCE_FIELDS:
            baseline_value = baseline[seed]['provenance'][field]
            candidate_value = candidate[seed]['provenance'][field]
            if (
                baseline_value is None
                or candidate_value is None
                or baseline_value != candidate_value
            ):
                mismatches.append(field)
        if mismatches:
            raise ValueError(
                f'Rank-0 lambda pairing seed {seed} mismatches: {mismatches}.'
            )
        per_seed[str(seed)] = {
            'passes': True,
            'matched_fields': list(PAIR_PROVENANCE_FIELDS),
        }
    return {'passes': True, 'per_seed': per_seed}


def _audit_cosine(record, scope):
    value = (
        record['global']['cosine']
        if scope == 'global'
        else record['groups'][scope]['cosine']
    )
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f'Audit scope {scope} has a non-finite cosine.')
    return float(value)


def summarize_gradient_audit(by_method, repo_root):
    runs = {
        run['seed']: run for run in by_method['joint_rank0_lambda005']
        if run['seed'] in {1, 2}
    }
    if set(runs) != {1, 2}:
        raise ValueError('Gradient audit requires rank-0 lambda=.05 seeds 1 and 2.')
    scopes = ('global', 'embeddings', 'layers_0', 'layers_1')
    records_by_seed = {}
    sources = []
    for seed, run in sorted(runs.items()):
        runtime = run['_runtime_metadata']
        path = Path(runtime.get('joint_gradient_audit_path', ''))
        if not path.is_file():
            raise ValueError(f'Seed {seed} is missing gradient-audit JSONL.')
        records = [
            json.loads(line)
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        epochs = [record.get('epoch') for record in records]
        if epochs != EXPECTED_AUDIT_EPOCHS:
            raise ValueError(f'Seed {seed} has unexpected audit epochs: {epochs}.')
        if any(record.get('batch_index') != 0 for record in records):
            raise ValueError(f'Seed {seed} audit did not use batch zero.')
        if any(record.get('joint_gcl_lambda') != 0.05 for record in records):
            raise ValueError(f'Seed {seed} audit has mismatched lambda.')
        seed_summary = {
            scope: summary_stats([_audit_cosine(record, scope) for record in records])
            for scope in scopes
        }
        recorded = run['_metrics'].get('joint_gradient_audit_summary', {})
        if recorded.get('record_count') != len(records):
            raise ValueError(f'Seed {seed} metric audit summary count mismatch.')
        if not math.isclose(
            seed_summary['global']['mean'],
            recorded.get('global_cosine', {}).get('mean', math.nan),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f'Seed {seed} metric audit summary mean mismatch.')
        records_by_seed[seed] = records
        sources.append({
            'seed': seed,
            'path': repo_relative(path, repo_root),
            'sha256': file_sha256(path),
            'record_count': len(records),
            'summary': seed_summary,
        })
    pooled_records = records_by_seed[1] + records_by_seed[2]
    return {
        'seeds': [1, 2],
        'epochs': EXPECTED_AUDIT_EPOCHS,
        'record_count': len(pooled_records),
        'sources': sources,
        'pooled': {
            scope: summary_stats([
                _audit_cosine(record, scope) for record in pooled_records
            ])
            for scope in scopes
        },
    }


def _public_run(run):
    return {key: value for key, value in run.items() if not key.startswith('_')}


def build_report(roots, repo_root):
    roots_by_name = validate_matrix(roots)
    hashed_paths = set()
    runs = []
    for root_name, expected_runs in EXPECTED_MATRIX.items():
        root = roots_by_name[root_name]
        for run_name, (method, seed) in expected_runs.items():
            runs.append(load_run(
                root, run_name, method, seed, repo_root, hashed_paths
            ))
    runs.sort(key=lambda run: (run['method'], run['seed']))
    summaries, by_method = group_runs(runs)
    comparisons = {
        'joint_rank0_lambda005_vs_static_dual': compare_methods(
            'joint_rank0_lambda005', 'static_dual', by_method
        ),
        'joint_rank0_lambda005_vs_joint_rank0_lambda0': compare_methods(
            'joint_rank0_lambda005', 'joint_rank0_lambda0', by_method
        ),
    }
    gates = {
        method: promotion_gate(compare_methods(method, 'static_dual', by_method))
        for method, seeds in EXPECTED_METHOD_SEEDS.items()
        if seeds == {0, 1, 2}
    }
    lambda_pairing = validate_lambda_pairing(by_method)
    gradient_audit = summarize_gradient_audit(by_method, repo_root)
    commits = sorted({run['git_commit'] for run in runs})
    return {
        'schema_version': 1,
        'report_kind': 'strict_reuse_teacher_report',
        'roots': [repo_relative(roots_by_name[name], repo_root) for name in EXPECTED_MATRIX],
        'protocol': {
            'source': 'ssram',
            'transfer': list(TRANSFER_NAMES),
            'blind_circuit_present_in_formal_artifacts': False,
            'epochs': 160,
            'early_stopping': False,
            'fixed_view_seed': FIXED_VIEW_SEED,
            'promotion_thresholds': GATES,
        },
        'inventory': {
            'expected_artifacts': 20,
            'completed_artifacts': len(runs),
            'commits': commits,
            'verified_hashed_provenance_files': len(hashed_paths),
            'anomalies': [],
        },
        'runs': [_public_run(run) for run in runs],
        'method_summaries': summaries,
        'comparisons': comparisons,
        'promotion_gates_vs_static_dual': gates,
        'rank0_lambda_pairing': lambda_pairing,
        'gradient_audit': gradient_audit,
    }


def _tsv_rows(report):
    rows = []
    for run in report['runs']:
        rows.append({
            'record_type': 'run', 'id': f"{run['method']}/seed{run['seed']}",
            'method': run['method'], 'seed': run['seed'],
            **{metric: run[metric] for metric in METRICS},
            'best_epoch': run['best_epoch'], 'git_commit': run['git_commit'],
            'artifact_dir': run['artifact_dir'],
        })
    for method, summary in report['method_summaries'].items():
        row = {
            'record_type': 'method_summary', 'id': method, 'method': method,
            'count': len(summary['seeds']), 'seeds': ','.join(map(str, summary['seeds'])),
        }
        for metric in METRICS:
            row[metric] = summary['metrics'][metric]['mean']
            row[f'{metric}_pstdev'] = summary['metrics'][metric]['pstdev']
        rows.append(row)
    for name, comparison in report['comparisons'].items():
        row = {
            'record_type': 'comparison_delta_of_means', 'id': name,
            'method': comparison['candidate'], 'reference': comparison['reference'],
            'count': 3, 'seeds': '0,1,2',
        }
        for metric in METRICS:
            row[metric] = comparison['metrics'][metric]['delta_of_means']
        rows.append(row)
    for method, gate in report['promotion_gates_vs_static_dual'].items():
        row = {
            'record_type': 'promotion_gate', 'id': method, 'method': method,
            'reference': 'static_dual', 'passes': gate['passes'],
        }
        for metric, check in gate['checks'].items():
            row[metric] = check['observed_degradation']
        rows.append(row)
    for source in report['gradient_audit']['sources']:
        for scope, stats in source['summary'].items():
            rows.append({
                'record_type': 'gradient_audit',
                'id': f"seed{source['seed']}/{scope}", 'seed': source['seed'],
                'scope': scope, 'count': stats['count'], 'mean': stats['mean'],
                'pstdev': stats['pstdev'], 'median': stats['median'],
                'q10': stats['q10'], 'q25': stats['q25'], 'q75': stats['q75'],
                'q90': stats['q90'], 'negative_fraction': stats['negative_fraction'],
                'artifact_dir': source['path'],
            })
    for scope, stats in report['gradient_audit']['pooled'].items():
        rows.append({
            'record_type': 'gradient_audit_pooled', 'id': scope, 'scope': scope,
            'count': stats['count'], 'mean': stats['mean'],
            'pstdev': stats['pstdev'], 'median': stats['median'],
            'q10': stats['q10'], 'q25': stats['q25'], 'q75': stats['q75'],
            'q90': stats['q90'], 'negative_fraction': stats['negative_fraction'],
        })
    rows.append({
        'record_type': 'provenance_check', 'id': 'strict_inventory',
        'count': report['inventory']['completed_artifacts'],
        'passes': not report['inventory']['anomalies'],
        'mean': report['inventory']['verified_hashed_provenance_files'],
    })
    return rows


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent, delete=False
    ) as output:
        output.write(content)
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / 'strict_reuse_summary.json'
    tsv_path = output_dir / 'strict_reuse_summary.tsv'
    json_content = json.dumps(report, indent=2, sort_keys=True) + '\n'
    atomic_write_text(json_path, json_content)
    rows = _tsv_rows(report)
    columns = [
        'record_type', 'id', 'method', 'reference', 'seed', 'seeds', 'scope',
        'count', *METRICS, *[f'{metric}_pstdev' for metric in METRICS],
        'mean', 'pstdev', 'median', 'q10', 'q25', 'q75', 'q90',
        'negative_fraction', 'passes', 'best_epoch', 'git_commit', 'artifact_dir',
    ]
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='', dir=output_dir, delete=False
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(output.name)
    temporary_path.replace(tsv_path)
    return json_path, tsv_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Validate and consolidate the strict CircuitGCL reuse results.'
    )
    parser.add_argument('roots', nargs='*', type=Path)
    parser.add_argument(
        '--output_dir', type=Path,
        default=Path('logs/strict_reuse_report_20260712'),
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    roots = args.roots or [repo_root / 'logs' / name for name in EXPECTED_MATRIX]
    report = build_report(roots, repo_root)
    json_path, tsv_path = write_report(report, args.output_dir)
    print(json.dumps({
        'completed_artifacts': report['inventory']['completed_artifacts'],
        'verified_hashed_provenance_files': (
            report['inventory']['verified_hashed_provenance_files']
        ),
        'rank0_lambda005_gate_passes': report[
            'promotion_gates_vs_static_dual'
        ]['joint_rank0_lambda005']['passes'],
        'gradient_audit_records': report['gradient_audit']['record_count'],
        'json': str(json_path),
        'tsv': str(tsv_path),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
