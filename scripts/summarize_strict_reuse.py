#!/usr/bin/env python3
"""Build the canonical teacher-facing summary for strict reuse experiments.

The report is intentionally tied to the four preregistered multi-seed roots
and the three later seed-0 screen roots.  It validates completed artifacts
before combining seeds, preserves seed-0 screens as a separate evidence scope,
then emits one JSON file for traceability and one TSV file for convenient
inspection.  Raw checkpoints and run artifacts remain in their original
directories.
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
DEPLOYMENT_PARAMETER_COUNT = 29378

SEED0_SCREEN_ROOT_NAMES = {
    'linear_lambda': (
        'strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712'
    ),
    'pcgrad': 'strict_joint_rank0_lambda005_pcgrad_seed0_audit_20260713',
    'graphsage': 'strict_joint_sage_rank0_seed0_screen_20260714',
}

SEED0_SCREEN_SUMMARY_FILES = {
    'linear_lambda': ('schedule_seed0_summary.json', 'schedule_seed0_summary.tsv'),
    'pcgrad': ('pcgrad_seed0_summary.json', 'pcgrad_seed0_summary.tsv'),
    'graphsage': ('sage_seed0_summary.json', 'sage_seed0_summary.tsv'),
}

EXPECTED_SEED0_SCREEN_COMMITS = {
    'linear_lambda': '4375c9f2bf2a27e20ccba2a0f3a644cd6745f8b8',
    'pcgrad': '95cb94834bf7696fc3223cabb5d3be5997ecbad0',
    'graphsage': '62894d9d353a5eb0f3e19a113afe7224eaa02f04',
}

EXPECTED_SEED0_SCREEN_HASHES = {
    'linear_lambda': {
        'json': 'f95b1e87a0307f35583885a32ee489300f1ef415146be7216b95f3d3192b9141',
        'tsv': '96061154d46b492753dde7d1e8ffb2e5992fffb598487d3ac5eb983b70dbd24e',
        'manifest': 'e89f793e3d7a12425359f27b6ad16efa398aa51311681d81403c33f116c24ace',
    },
    'pcgrad': {
        'json': 'f90c7cb42435c11195a3e368d3934a2f11e612929208c46629f464b3fabbfc7f',
        'tsv': 'add618609b45b67cd7dcd8f366e4882f04d616c7aecfed53d86677111669e435',
        'manifest': 'fa473620f2852acf113ea70f09987c57bed673fa47762c857ce791661c023d35',
    },
    'graphsage': {
        'json': 'c78fc01f369784f2f4844f261abb3044578b5265b692692fe67a0c4125dcaa6e',
        'tsv': '485ab1c0e27875c85104ece3c20ce4ac894ed63dbfc2af9a28b18a23654cc6f5',
        'manifest': 'eb1b0ee022c039e83521b85690197bc5e21ab1ec2daa5feb63b889b4daa0d2b1',
    },
}

FORBIDDEN_BLIND_IDENTIFIERS = ('sp8192w', 'sp_8192w', 'sram_sp_8192w')
BLIND_SCAN_SUFFIXES = {'.json', '.jsonl', '.tsv', '.txt', '.log'}

THREE_SEED_TEACHER_ORDER = (
    'static_dual',
    'no_gcl',
    'init_reuse',
    'joint_rank0_lambda0',
    'joint_rank0_lambda005',
    'joint_lora_r8_lambda005',
)

THREE_SEED_METHOD_METADATA = {
    'static_dual': {
        'display_name': 'static dual',
        'continuous_gcl': False,
        'decision': 'reference',
    },
    'no_gcl': {
        'display_name': 'no-GCL',
        'continuous_gcl': False,
        'decision': 'diagnostic_control',
    },
    'init_reuse': {
        'display_name': 'init_reuse',
        'continuous_gcl': False,
        'decision': 'engineering_fallback',
    },
    'joint_rank0_lambda0': {
        'display_name': 'ClusterGCN rank0 lambda=0',
        'continuous_gcl': False,
        'decision': 'passes_but_not_selected',
    },
    'joint_lora_r8_lambda005': {
        'display_name': 'ClusterGCN LoRA r8 lambda=0.05',
        'continuous_gcl': True,
        'decision': 'rejected_by_per_circuit_gate',
    },
    'joint_rank0_lambda005': {
        'display_name': 'ClusterGCN rank0 lambda=0.05',
        'continuous_gcl': True,
        'decision': 'selected_shared_incumbent',
    },
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


def validate_seed0_screen_roots(screen_roots, repo_root):
    if set(screen_roots) != set(SEED0_SCREEN_ROOT_NAMES):
        raise ValueError(
            'Seed-0 screen report requires exactly these roots: '
            f'{sorted(SEED0_SCREEN_ROOT_NAMES)}.'
        )
    resolved = {}
    for screen, expected_name in SEED0_SCREEN_ROOT_NAMES.items():
        root = Path(screen_roots[screen]).resolve()
        expected_root = (Path(repo_root).resolve() / 'logs' / expected_name)
        if root != expected_root:
            raise ValueError(
                f'{screen} root must be {expected_root}; got {root}.'
            )
        if not root.is_dir():
            raise ValueError(f'Missing seed-0 screen root: {root}')
        resolved[screen] = root
    return resolved


def validate_canonical_core_roots(roots, repo_root):
    if len(roots) != len(EXPECTED_MATRIX):
        raise ValueError('Canonical report requires four distinct core roots.')
    roots_by_name = validate_matrix(roots)
    for root_name, root in roots_by_name.items():
        expected = Path(repo_root).resolve() / 'logs' / root_name
        if root != expected:
            raise ValueError(
                f'Canonical core root must be {expected}; got {root}.'
            )
    return roots_by_name


def _scan_text_artifacts_for_blind(root, context):
    paths = [
        path for path in sorted(Path(root).rglob('*'))
        if path.is_file() and path.suffix.lower() in BLIND_SCAN_SUFFIXES
    ]
    for path in paths:
        content = path.read_text(encoding='utf-8').lower()
        matched = [
            identifier for identifier in FORBIDDEN_BLIND_IDENTIFIERS
            if identifier in content
        ]
        if matched:
            raise ValueError(
                f'{context} contains blind identifier {matched[0]} in {path}.'
            )
    return len(paths)


def _validate_linear_screen_integrity(root, report):
    manifest_path = root / 'selection_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _expect(manifest, {
        'schema_version': 1,
        'method': 'joint_rank0_lambda005_to0005_linear_seed0',
        'protocol': 'strict_inductive',
        'seeds': [0],
        'epochs': 160,
        'automatic_multiseed_expansion': False,
        'blind_circuits': [],
        'physical_gpu': 4,
        'child_cuda_visible_devices': '4',
        'child_logical_gpu': 0,
        'comparison_roots': [
            'logs/strict_joint_rank0_lambda005_seed0_20260712',
            'logs/strict_seed0_seven_20260712_v2',
        ],
        'gpu_gate': {
            'compute_processes': 'record_only',
            'minimum_free_mb': 6500,
            'poll_seconds': 60,
            'required_consecutive_capacity_samples': 1,
            'utilization': 'record_only',
        },
        'gradient_audit': {
            'batch_index': 0,
            'enabled': True,
            'interval': 10,
            'observational_only': True,
        },
        'selection': {
            'minimum_source_val_improvement_to_expand': 2e-5,
            'primary': 'source validation raw MSE',
            'promotion_gates': {
                'any_transfer_circuit_degradation_max': 0.25,
                'source_validation_degradation_max': 0.05,
                'transfer_mean_degradation_max': 0.1,
            },
            'transfer_role': 'promotion gates and reporting only',
        },
    }, 'linear-lambda manifest')
    if manifest.get('git_commit') != EXPECTED_SEED0_SCREEN_COMMITS['linear_lambda']:
        raise ValueError('Linear-lambda manifest commit changed.')
    config_path = Path(report.get('sources', {}).get('candidate_config', ''))
    metrics_path = Path(report.get('sources', {}).get('candidate_metrics', ''))
    for path, label in ((config_path, 'config'), (metrics_path, 'metrics')):
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f'Linear-lambda {label} escaped its root.') from error
        if not path.is_file():
            raise ValueError(f'Linear-lambda {label} is missing.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    runtime = config.get('runtime_metadata')
    if not isinstance(runtime, dict):
        raise ValueError('Linear-lambda runtime provenance is missing.')
    if runtime.get('resolved_seeds') != _expected_resolved_seeds(0):
        raise ValueError('Linear-lambda resolved seeds changed.')
    _verify_hashed_paths(
        runtime, set(), 'linear-lambda supplemental runtime provenance'
    )


def _validate_materialized_screen_report(
        screen, report, root, writer, repo_root):
    json_name, tsv_name = SEED0_SCREEN_SUMMARY_FILES[screen]
    json_path = root / json_name
    tsv_path = root / tsv_name
    if not json_path.is_file() or not tsv_path.is_file():
        raise ValueError(f'{screen} is missing its materialized JSON/TSV summary.')
    manifest_path = root / 'selection_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'{screen} is missing selection_manifest.json.')
    expected_hashes = EXPECTED_SEED0_SCREEN_HASHES[screen]
    actual_hashes = {
        'json': file_sha256(json_path),
        'tsv': file_sha256(tsv_path),
        'manifest': file_sha256(manifest_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            f'{screen} immutable summary/manifest hashes changed: '
            f'{actual_hashes}.'
        )
    blind_scanned_text_files = _scan_text_artifacts_for_blind(root, screen)
    with tempfile.TemporaryDirectory() as temporary_dir:
        generated_json, generated_tsv = writer(report, Path(temporary_dir))
        if generated_json.read_bytes() != json_path.read_bytes():
            raise ValueError(f'{screen} JSON summary is stale or inconsistent.')
        if generated_tsv.read_bytes() != tsv_path.read_bytes():
            raise ValueError(f'{screen} TSV summary is stale or inconsistent.')
    return {
        'json_path': repo_relative(json_path, repo_root),
        'json_sha256': file_sha256(json_path),
        'tsv_path': repo_relative(tsv_path, repo_root),
        'tsv_sha256': actual_hashes['tsv'],
        'manifest_path': repo_relative(manifest_path, repo_root),
        'manifest_sha256': actual_hashes['manifest'],
        'blind_scanned_text_files': blind_scanned_text_files,
    }


def rebuild_seed0_screen_reports(
        screen_roots, core_roots, repo_root):
    # Lazy imports avoid a module-level cycle because the dedicated validators
    # share utility functions from this module.
    from summarize_joint_lambda_schedule_seed0 import (
        build_report as build_schedule_report,
        write_report as write_schedule_report,
    )
    from summarize_joint_pcgrad_seed0 import (
        build_report as build_pcgrad_report,
        write_report as write_pcgrad_report,
    )
    from summarize_joint_sage_seed0 import (
        build_report as build_sage_report,
        write_report as write_sage_report,
    )

    screen_roots = validate_seed0_screen_roots(screen_roots, repo_root)
    static_root = core_roots['strict_seed0_seven_20260712_v2']
    fixed_root = core_roots['strict_joint_rank0_lambda005_seed0_20260712']
    schedule_root = screen_roots['linear_lambda']
    reports = {
        'linear_lambda': build_schedule_report(
            schedule_root, fixed_root, static_root
        ),
        'pcgrad': build_pcgrad_report(
            screen_roots['pcgrad'], fixed_root, static_root, schedule_root
        ),
        'graphsage': build_sage_report(
            screen_roots['graphsage'], static_root, fixed_root,
            repo_root=repo_root,
        ),
    }
    for screen, expected_commit in EXPECTED_SEED0_SCREEN_COMMITS.items():
        if reports[screen].get('git_commit') != expected_commit:
            raise ValueError(f'{screen} immutable commit changed.')
    _validate_linear_screen_integrity(schedule_root, reports['linear_lambda'])
    writers = {
        'linear_lambda': write_schedule_report,
        'pcgrad': write_pcgrad_report,
        'graphsage': write_sage_report,
    }
    sources = {
        screen: _validate_materialized_screen_report(
            screen, reports[screen], screen_roots[screen], writers[screen],
            repo_root,
        )
        for screen in SEED0_SCREEN_ROOT_NAMES
    }
    return reports, sources, screen_roots


def _validated_metric_values(payload, context):
    if not isinstance(payload, dict):
        raise ValueError(f'{context} must be a metric object.')
    values = {}
    for metric in METRICS:
        value = payload.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f'{context} has invalid {metric}.')
        values[metric] = float(value)
    expected_transfer = statistics.mean(
        values[metric] for metric in ('digtime', 'timing_ctrl', 'array')
    )
    if not math.isclose(
            values['transfer_mean'], expected_transfer,
            rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f'{context} has inconsistent transfer mean.')
    return values


def _core_seed0_metrics(report, method):
    matches = [
        run for run in report['runs']
        if run['method'] == method and run['seed'] == 0
    ]
    if len(matches) != 1:
        raise ValueError(f'Canonical core is missing {method}/seed0.')
    return {metric: float(matches[0][metric]) for metric in METRICS}


def _require_metric_match(candidate, reference, context):
    candidate = _validated_metric_values(candidate, f'{context} candidate')
    reference = _validated_metric_values(reference, f'{context} reference')
    mismatches = [
        metric for metric in METRICS
        if not math.isclose(
            candidate[metric], reference[metric], rel_tol=1e-12, abs_tol=1e-15
        )
    ]
    if mismatches:
        raise ValueError(f'{context} metric mismatch: {mismatches}.')


def _screen_gate(candidate, static):
    relative = {
        metric: candidate[metric] / static[metric] - 1.0
        for metric in METRICS
    }
    checks = {
        metric: {
            'observed_degradation': relative[metric],
            'maximum_degradation': GATES[metric],
            'passes': relative[metric] <= GATES[metric],
        }
        for metric in GATES
    }
    return {
        'reference': 'static_dual/seed0',
        'relative_delta': relative,
        'checks': checks,
        'passes': all(check['passes'] for check in checks.values()),
    }


def _normalized_screen_path(path, screen_root, repo_root, context):
    if path in (None, ''):
        return None
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    path = path.resolve()
    try:
        path.relative_to(Path(screen_root).resolve())
    except ValueError as error:
        raise ValueError(f'{context} escaped its registered screen root.') from error
    return repo_relative(path, repo_root)


def _simple_screen_best_epoch(report, context):
    best_epoch = report.get('best_epoch')
    if best_epoch is None:
        metrics_path = Path(
            report.get('sources', {}).get('candidate_metrics', '')
        )
        if not metrics_path.is_file():
            raise ValueError(f'{context} candidate metrics are missing.')
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        best_epoch = metrics.get('best_epoch')
    if type(best_epoch) is not int or not 0 <= best_epoch < 160:
        raise ValueError(f'{context} has an invalid best epoch.')
    return best_epoch


def _simple_screen_record(
        screen, report, source, architecture, variant, continuous_gcl,
        static, incumbent, screen_root=None, repo_root=None):
    candidate = _validated_metric_values(
        report.get('candidate'), f'{screen} candidate'
    )
    gate = _screen_gate(candidate, static)
    recorded_gate = report.get('promotion_gate', {}).get('passes')
    selection = report.get('selection', {})
    advances = selection.get('advances_to_seeds12')
    recorded_improvement = selection.get(
        'source_val_improvement_over_fixed'
    )
    minimum_improvement = selection.get('minimum_improvement_to_expand')
    improvement = incumbent['val_mse'] - candidate['val_mse']
    if minimum_improvement != 2e-5:
        raise ValueError(f'{screen} expansion threshold changed.')
    if (
        recorded_improvement is None
        or not math.isclose(
            float(recorded_improvement), improvement,
            rel_tol=1e-12, abs_tol=1e-15,
        )
    ):
        raise ValueError(f'{screen} source improvement is inconsistent.')
    expected_advances = gate['passes'] and improvement > minimum_improvement
    if recorded_gate is not gate['passes'] or advances is not expected_advances:
        raise ValueError(f'{screen} selection fields are inconsistent.')
    if not gate['passes'] or advances is not False:
        raise ValueError(f'{screen} immutable selection conclusion changed.')
    artifact_dir = report.get('artifact_dir')
    checkpoint_path = report.get('checkpoint')
    best_epoch = _simple_screen_best_epoch(report, screen)
    if screen_root is not None and repo_root is not None:
        artifact_dir = _normalized_screen_path(
            artifact_dir, screen_root, repo_root, f'{screen} artifact'
        )
        checkpoint_path = _normalized_screen_path(
            checkpoint_path, screen_root, repo_root, f'{screen} checkpoint'
        )
    return {
        'screen': screen,
        'method': report.get('method'),
        'seed': 0,
        'evidence_scope': 'seed0_screen',
        'architecture': architecture,
        'variant': variant,
        'continuous_gcl': continuous_gcl,
        'planned_deployment_gnns': 1,
        'shared_model_trainable_parameters': DEPLOYMENT_PARAMETER_COUNT,
        **candidate,
        'best_epoch': best_epoch,
        'gate_reference': 'static_dual/seed0',
        'selection_reference': 'joint_rank0_lambda005/seed0',
        'relative_to_gate_reference': gate['relative_delta'],
        'relative_to_selection_reference': {
            metric: candidate[metric] / incumbent[metric] - 1.0
            for metric in METRICS
        },
        'source_val_improvement_over_incumbent': improvement,
        'minimum_improvement_to_expand': minimum_improvement,
        'promotion_gate': gate,
        'promotion_gate_passes': gate['passes'],
        'advances_to_seeds12': False,
        'selection_status': 'not_advanced',
        'selection_reason': 'source_improvement_not_over_2e-5',
        'git_commit': report.get('git_commit'),
        'artifact_dir': artifact_dir,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': report.get('checkpoint_sha256'),
        'summary_source': source,
    }


def normalize_seed0_screen_records(
        core_report, reports, sources, screen_roots=None, repo_root=None):
    if set(reports) != set(SEED0_SCREEN_ROOT_NAMES):
        raise ValueError('Seed-0 screen report set is incomplete.')
    if set(sources) != set(SEED0_SCREEN_ROOT_NAMES):
        raise ValueError('Seed-0 screen source set is incomplete.')
    serialized = json.dumps(reports, sort_keys=True).lower()
    if any(identifier in serialized for identifier in FORBIDDEN_BLIND_IDENTIFIERS):
        raise ValueError('Seed-0 screen reports contain a blind identifier.')

    static = _core_seed0_metrics(core_report, 'static_dual')
    cluster0 = _core_seed0_metrics(core_report, 'joint_rank0_lambda0')
    cluster005 = _core_seed0_metrics(core_report, 'joint_rank0_lambda005')
    linear = reports['linear_lambda']
    pcgrad = reports['pcgrad']
    sage = reports['graphsage']
    if (
        linear.get('schema_version') != 1
        or linear.get('method') != (
            'joint_rank0_lambda005_to0005_linear_seed0'
        )
    ):
        raise ValueError('Unexpected linear-lambda summary schema or method.')
    if (
        pcgrad.get('schema_version') != 1
        or pcgrad.get('method') != 'joint_rank0_lambda005_pcgrad_seed0'
    ):
        raise ValueError('Unexpected PCGrad summary schema or method.')
    expected_sage_runs = {
        'joint_sage_rank0_lambda0_seed0',
        'joint_sage_rank0_lambda005_seed0',
    }
    if (
        sage.get('schema_version') != 1
        or sage.get('report_kind') != 'compact_shared_sage_seed0_screen'
        or set(sage.get('runs', {})) != expected_sage_runs
    ):
        raise ValueError('Unexpected GraphSAGE summary schema or run set.')
    _require_metric_match(linear.get('static_dual'), static, 'linear/static')
    _require_metric_match(
        linear.get('fixed_lambda005'), cluster005, 'linear/fixed'
    )
    _require_metric_match(pcgrad.get('static_dual'), static, 'PCGrad/static')
    _require_metric_match(
        pcgrad.get('fixed_lambda005'), cluster005, 'PCGrad/fixed'
    )
    _require_metric_match(
        pcgrad.get('linear_lambda'), linear.get('candidate'),
        'PCGrad/linear prerequisite',
    )
    _require_metric_match(
        sage.get('controls', {}).get('canonical_static_dual'),
        static, 'GraphSAGE/static',
    )
    _require_metric_match(
        sage.get('controls', {}).get('cluster_rank0_lambda0'),
        cluster0, 'GraphSAGE/Cluster lambda=0',
    )
    _require_metric_match(
        sage.get('controls', {}).get('cluster_rank0_lambda005'),
        cluster005, 'GraphSAGE/Cluster lambda=0.05',
    )
    if linear.get('paired_provenance', {}).get('passes') is not True:
        raise ValueError('Linear-lambda paired provenance failed.')
    if pcgrad.get('paired_provenance', {}).get('passes') is not True:
        raise ValueError('PCGrad paired provenance failed.')
    if sage.get('paired_provenance', {}).get('passes') is not True:
        raise ValueError('GraphSAGE paired provenance failed.')

    records = [
        _simple_screen_record(
            'linear_lambda', linear, sources['linear_lambda'], 'clustergcn',
            'linear_lambda_0.05_to_0.005', True, static, cluster005,
            screen_roots['linear_lambda'] if screen_roots else None, repo_root,
        ),
        _simple_screen_record(
            'pcgrad', pcgrad, sources['pcgrad'], 'clustergcn',
            'pcgrad_lambda_0.05', True, static, cluster005,
            screen_roots['pcgrad'] if screen_roots else None, repo_root,
        ),
    ]
    sage_selection = sage.get('selection', {})
    if (
        sage_selection.get('advances_to_seeds12') is not False
        or sage_selection.get('selected_candidate') is not None
        or sage_selection.get('selection_reason') != (
            'no_candidate_met_source_improvement'
        )
    ):
        raise ValueError('GraphSAGE immutable selection conclusion changed.')
    advancement_eligible = set(
        sage_selection.get('advancement_eligible_candidates', [])
    )
    if sage_selection.get('minimum_improvement_to_expand') != 2e-5:
        raise ValueError('GraphSAGE expansion threshold changed.')
    sage_specs = (
        ('joint_sage_rank0_lambda0_seed0', 'constant_lambda_0', False),
        ('joint_sage_rank0_lambda005_seed0', 'constant_lambda_0.05', True),
    )
    for method, variant, continuous_gcl in sage_specs:
        run = sage.get('runs', {}).get(method)
        metrics = _validated_metric_values(run, f'{method} metrics')
        best_epoch = run.get('best_epoch')
        if type(best_epoch) is not int or not 0 <= best_epoch < 160:
            raise ValueError(f'{method} has an invalid best epoch.')
        gate = _screen_gate(metrics, static)
        recorded_gate = sage.get(
            'promotion_gates_vs_canonical_static', {}
        ).get(method, {})
        improvement = cluster005['val_mse'] - metrics['val_mse']
        expected_eligible = gate['passes'] and improvement > 2e-5
        if recorded_gate.get('passes') is not gate['passes']:
            raise ValueError(f'{method} gate result is inconsistent.')
        if (method in advancement_eligible) is not expected_eligible:
            raise ValueError(f'{method} advancement eligibility is inconsistent.')
        if not gate['passes'] or expected_eligible:
            raise ValueError(f'{method} immutable promotion conclusion changed.')
        artifact = run.get('artifact', {})
        artifact_dir = (
            str(Path(artifact.get('metrics_path', '')).parent)
            if artifact.get('metrics_path') else None
        )
        checkpoint_path = artifact.get('checkpoint_path')
        if screen_roots is not None and repo_root is not None:
            artifact_dir = _normalized_screen_path(
                artifact_dir, screen_roots['graphsage'], repo_root,
                f'{method} artifact',
            )
            checkpoint_path = _normalized_screen_path(
                checkpoint_path, screen_roots['graphsage'], repo_root,
                f'{method} checkpoint',
            )
        records.append({
            'screen': 'graphsage',
            'method': method,
            'seed': 0,
            'evidence_scope': 'seed0_screen',
            'architecture': 'sage',
            'variant': variant,
            'continuous_gcl': continuous_gcl,
            'planned_deployment_gnns': 1,
            'shared_model_trainable_parameters': DEPLOYMENT_PARAMETER_COUNT,
            **metrics,
            'best_epoch': best_epoch,
            'gate_reference': 'static_dual/seed0',
            'selection_reference': 'joint_rank0_lambda005/seed0',
            'relative_to_gate_reference': gate['relative_delta'],
            'relative_to_selection_reference': {
                metric: metrics[metric] / cluster005[metric] - 1.0
                for metric in METRICS
            },
            'source_val_improvement_over_incumbent': improvement,
            'minimum_improvement_to_expand': 2e-5,
            'promotion_gate': gate,
            'promotion_gate_passes': gate['passes'],
            'advances_to_seeds12': False,
            'selection_status': 'not_advanced',
            'selection_reason': sage_selection['selection_reason'],
            'git_commit': sage.get('git_commit'),
            'artifact_dir': artifact_dir,
            'checkpoint_path': checkpoint_path,
            'checkpoint_sha256': artifact.get('checkpoint_sha256'),
            'summary_source': sources['graphsage'],
        })
    if len(records) != 4:
        raise ValueError('Expected exactly four seed-0 screening records.')
    return records


def build_three_seed_teacher_rows(report):
    static = report['method_summaries']['static_dual']['metrics']
    rows = []
    for method in THREE_SEED_TEACHER_ORDER:
        summary = report['method_summaries'].get(method)
        if summary is None or summary.get('seeds') != [0, 1, 2]:
            raise ValueError(f'{method} lacks the required three-seed summary.')
        gate = report['promotion_gates_vs_static_dual'].get(method)
        if gate is None:
            raise ValueError(f'{method} lacks its promotion-gate summary.')
        metadata = THREE_SEED_METHOD_METADATA[method]
        per_seed = gate.get('per_seed', {})
        passed_seed_count = sum(
            seed_gate.get('passes') is True for seed_gate in per_seed.values()
        )
        if set(per_seed) != {'0', '1', '2'}:
            raise ValueError(f'{method} lacks complete per-seed gate results.')
        if (
            metadata['decision'] == 'rejected_by_per_circuit_gate'
            and gate['passes'] is not False
        ):
            raise ValueError(f'{method} rejection no longer matches its gate.')
        if (
            metadata['decision'] == 'selected_shared_incumbent'
            and (gate['passes'] is not True or passed_seed_count != 3)
        ):
            raise ValueError(f'{method} selection no longer has passing evidence.')
        rows.append({
            'method': method,
            **metadata,
            'evidence_scope': 'three_seed',
            'seeds': [0, 1, 2],
            'metrics': {
                metric: {
                    'mean': summary['metrics'][metric]['mean'],
                    'pstdev': summary['metrics'][metric]['pstdev'],
                }
                for metric in METRICS
            },
            'relative_to_static': {
                metric: (
                    summary['metrics'][metric]['mean'] /
                    static[metric]['mean'] - 1.0
                )
                for metric in METRICS
            },
            'promotion_gate_passes': gate['passes'],
            'per_seed_gate_pass_count': passed_seed_count,
            'all_seed_gates_pass': passed_seed_count == 3,
        })
    return rows


def build_seed0_teacher_rows(core_report, screen_records):
    fixed = _core_seed0_metrics(core_report, 'joint_rank0_lambda005')
    gate = core_report['promotion_gates_vs_static_dual'][
        'joint_rank0_lambda005'
    ]['per_seed']['0']['passes']
    control = {
        'screen': 'registered_control',
        'method': 'joint_rank0_lambda005_seed0',
        'seed': 0,
        'evidence_scope': 'seed0_control',
        'architecture': 'clustergcn',
        'variant': 'constant_lambda_0.05',
        'continuous_gcl': True,
        **fixed,
        'gate_reference': 'static_dual/seed0',
        'selection_reference': 'self',
        'relative_to_incumbent': {metric: 0.0 for metric in METRICS},
        'promotion_gate_passes': gate,
        'advances_to_seeds12': None,
        'selection_status': 'selected_shared_incumbent',
    }
    rows = [control]
    for record in screen_records:
        rows.append({
            **record,
            'relative_to_incumbent': {
                metric: record[metric] / fixed[metric] - 1.0
                for metric in METRICS
            },
        })
    return rows


def extend_report_with_seed0_screens(
        report, reports, sources, screen_roots, repo_root,
        core_blind_scanned_text_files):
    if report.get('inventory', {}).get('completed_artifacts') != 20:
        raise ValueError('Canonical core inventory must contain 20 artifacts.')
    if (
        type(core_blind_scanned_text_files) is not int
        or core_blind_scanned_text_files <= 0
    ):
        raise ValueError('Core blind scan must cover at least one text artifact.')
    report = json.loads(json.dumps(report))
    screen_records = normalize_seed0_screen_records(
        report, reports, sources, screen_roots, repo_root
    )
    core_roots = list(report['roots'])
    report['schema_version'] = 2
    report['report_scope'] = 'reuse_selection_pre_deployment_pre_blind'
    report['protocol'].update({
        'blind_identifier_present_in_included_text_artifacts': False,
        'blind_evaluation_completed': False,
        'blind_claim_scope': 'included_formal_text_artifacts_only',
    })
    report['core_roots'] = core_roots
    report['seed0_screen_roots'] = [
        repo_relative(screen_roots[screen], repo_root)
        for screen in SEED0_SCREEN_ROOT_NAMES
    ]
    report['roots'] = core_roots + report['seed0_screen_roots']
    report['inventory'].update({
        'core_artifacts': 20,
        'seed0_screen_artifacts': len(screen_records),
        'total_expected_artifacts': 20 + len(screen_records),
        'total_completed_artifacts': 20 + len(screen_records),
        'validated_seed0_screen_reports': len(reports),
        'verified_screen_summary_files': 2 * len(sources),
        'verified_screen_manifest_files': len(sources),
        'verified_hashed_provenance_files_scope': 'core_20_artifacts_only',
        'blind_scanned_core_text_files': core_blind_scanned_text_files,
        'blind_scanned_screen_text_files': sum(
            source['blind_scanned_text_files'] for source in sources.values()
        ),
    })
    screen_commits = [
        screen_report.get('git_commit') for screen_report in reports.values()
    ]
    if not all(isinstance(commit, str) and commit for commit in screen_commits):
        raise ValueError('Seed-0 screen reports have invalid commit provenance.')
    report['inventory']['commits'] = sorted(set(
        report['inventory']['commits'] + screen_commits
    ))
    incumbent_gate = report['promotion_gates_vs_static_dual'].get(
        'joint_rank0_lambda005', {}
    )
    incumbent_summary = report['method_summaries'].get(
        'joint_rank0_lambda005', {}
    )
    if (
        incumbent_summary.get('seeds') != [0, 1, 2]
        or incumbent_gate.get('passes') is not True
        or not all(
            value.get('passes') is True
            for value in incumbent_gate.get('per_seed', {}).values()
        )
        or set(incumbent_gate.get('per_seed', {})) != {'0', '1', '2'}
    ):
        raise ValueError('The locked incumbent lacks complete passing evidence.')
    report['seed0_screens'] = {
        'records': screen_records,
        'source_reports': sources,
        'validated_reports': reports,
        'selection': {
            'advancement_eligible_candidates': [],
            'advances_to_seeds12': False,
            'locked_method': 'joint_rank0_lambda005',
            'locked_architecture': 'clustergcn',
            'locked_variant': 'constant_lambda_0.05',
            'reason': 'all_seed0_screens_failed_registered_source_improvement',
        },
    }
    report['teacher_tables'] = {
        'three_seed_formal_comparison': build_three_seed_teacher_rows(report),
        'seed0_controlled_screens': build_seed0_teacher_rows(
            report, screen_records
        ),
        'scope_note': (
            'Seed-0 screens are not pooled into three-seed means. Deployment '
            'and blind measurements are pending.'
        ),
    }
    report['final_shared_selection'] = {
        'method': 'joint_rank0_lambda005',
        'architecture': 'clustergcn',
        'variant': 'constant_lambda_0.05',
        'rank': 0,
        'continuous_gcl': True,
        'planned_deployment_gnns': 1,
        'shared_model_trainable_parameters': DEPLOYMENT_PARAMETER_COUNT,
        'evidence_scope': 'three_seed',
        'status': 'locked_for_deployment_export',
        'deployment_export_validated': False,
    }
    return report


def build_canonical_report(roots, screen_roots, repo_root):
    core_roots = validate_canonical_core_roots(roots, repo_root)
    core_blind_scanned_text_files = sum(
        _scan_text_artifacts_for_blind(root, f'core/{root_name}')
        for root_name, root in core_roots.items()
    )
    report = build_report(roots, repo_root)
    reports, sources, resolved_screen_roots = rebuild_seed0_screen_reports(
        screen_roots, core_roots, repo_root
    )
    return extend_report_with_seed0_screens(
        report, reports, sources, resolved_screen_roots, repo_root,
        core_blind_scanned_text_files,
    )


def _tsv_rows(report):
    rows = []
    teacher_rows = {
        row['method']: row
        for row in report.get('teacher_tables', {}).get(
            'three_seed_formal_comparison', []
        )
    }
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
        teacher = teacher_rows.get(method)
        if teacher is not None:
            row.update({
                'evidence_scope': teacher['evidence_scope'],
                'continuous_gcl': teacher['continuous_gcl'],
                'passes': teacher['promotion_gate_passes'],
                'per_seed_gate_pass_count': (
                    teacher['per_seed_gate_pass_count']
                ),
                'all_seed_gates_pass': teacher['all_seed_gates_pass'],
                'selection_status': teacher['decision'],
            })
        rows.append(row)
    seed0_teacher_rows = report.get('teacher_tables', {}).get(
        'seed0_controlled_screens', []
    )
    if seed0_teacher_rows:
        control = seed0_teacher_rows[0]
        if control.get('selection_status') != 'selected_shared_incumbent':
            raise ValueError('Seed-0 teacher control row is invalid.')
        core_control = next(
            run for run in report['runs']
            if run['method'] == 'joint_rank0_lambda005' and run['seed'] == 0
        )
        rows.append({
            'record_type': 'seed0_screen_control',
            'id': control['method'],
            'method': control['method'],
            'reference': control['selection_reference'],
            'seed': 0,
            'screen': control['screen'],
            'evidence_scope': control['evidence_scope'],
            'architecture': control['architecture'],
            'variant': control['variant'],
            'continuous_gcl': control['continuous_gcl'],
            **{metric: control[metric] for metric in METRICS},
            'passes': control['promotion_gate_passes'],
            'source_val_improvement_over_incumbent': 0.0,
            'selection_status': control['selection_status'],
            'best_epoch': core_control['best_epoch'],
            'git_commit': core_control['git_commit'],
            'artifact_dir': core_control['artifact_dir'],
            'checkpoint_path': core_control['checkpoint_path'],
            'checkpoint_sha256': core_control['checkpoint_sha256'],
        })
    for record in report.get('seed0_screens', {}).get('records', []):
        source = record['summary_source']
        rows.append({
            'record_type': 'seed0_screen_run',
            'id': record['method'],
            'method': record['method'],
            'reference': record['selection_reference'],
            'seed': record['seed'],
            'screen': record['screen'],
            'evidence_scope': record['evidence_scope'],
            'architecture': record['architecture'],
            'variant': record['variant'],
            'continuous_gcl': record['continuous_gcl'],
            **{metric: record[metric] for metric in METRICS},
            'passes': record['promotion_gate_passes'],
            'source_val_improvement_over_incumbent': (
                record['source_val_improvement_over_incumbent']
            ),
            'minimum_improvement_to_expand': (
                record['minimum_improvement_to_expand']
            ),
            'advances_to_seeds12': record['advances_to_seeds12'],
            'selection_status': record['selection_status'],
            'selection_reason': record['selection_reason'],
            'best_epoch': record.get('best_epoch'),
            'git_commit': record['git_commit'],
            'artifact_dir': record['artifact_dir'],
            'checkpoint_path': record['checkpoint_path'],
            'checkpoint_sha256': record['checkpoint_sha256'],
            'source_summary_path': source['json_path'],
            'source_summary_sha256': source['json_sha256'],
        })
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
        'count': report['inventory'].get(
            'total_completed_artifacts',
            report['inventory']['completed_artifacts'],
        ),
        'passes': not report['inventory']['anomalies'],
        'mean': report['inventory']['verified_hashed_provenance_files'],
    })
    if 'final_shared_selection' in report:
        selection = report['final_shared_selection']
        rows.append({
            'record_type': 'final_selection',
            'id': 'shared_backbone',
            'method': selection['method'],
            'architecture': selection['architecture'],
            'variant': selection['variant'],
            'evidence_scope': selection['evidence_scope'],
            'continuous_gcl': selection['continuous_gcl'],
            'planned_deployment_gnns': selection['planned_deployment_gnns'],
            'shared_model_trainable_parameters': (
                selection['shared_model_trainable_parameters']
            ),
            'deployment_export_validated': (
                selection['deployment_export_validated']
            ),
            'selection_status': selection['status'],
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
        'screen', 'evidence_scope', 'architecture', 'variant',
        'continuous_gcl', 'count', *METRICS,
        *[f'{metric}_pstdev' for metric in METRICS],
        'mean', 'pstdev', 'median', 'q10', 'q25', 'q75', 'q90',
        'negative_fraction', 'passes', 'per_seed_gate_pass_count',
        'all_seed_gates_pass', 'source_val_improvement_over_incumbent',
        'minimum_improvement_to_expand', 'advances_to_seeds12',
        'planned_deployment_gnns', 'shared_model_trainable_parameters',
        'deployment_export_validated',
        'selection_status', 'selection_reason', 'best_epoch', 'git_commit',
        'artifact_dir', 'checkpoint_path', 'checkpoint_sha256',
        'source_summary_path', 'source_summary_sha256',
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
    parser.add_argument('--schedule_root', type=Path)
    parser.add_argument('--pcgrad_root', type=Path)
    parser.add_argument('--sage_root', type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    roots = args.roots or [repo_root / 'logs' / name for name in EXPECTED_MATRIX]
    screen_roots = {
        'linear_lambda': args.schedule_root or (
            repo_root / 'logs' / SEED0_SCREEN_ROOT_NAMES['linear_lambda']
        ),
        'pcgrad': args.pcgrad_root or (
            repo_root / 'logs' / SEED0_SCREEN_ROOT_NAMES['pcgrad']
        ),
        'graphsage': args.sage_root or (
            repo_root / 'logs' / SEED0_SCREEN_ROOT_NAMES['graphsage']
        ),
    }
    report = build_canonical_report(roots, screen_roots, repo_root)
    json_path, tsv_path = write_report(report, args.output_dir)
    print(json.dumps({
        'core_completed_artifacts': report['inventory']['completed_artifacts'],
        'seed0_screen_artifacts': report['inventory']['seed0_screen_artifacts'],
        'total_completed_artifacts': report['inventory'][
            'total_completed_artifacts'
        ],
        'validated_seed0_screen_reports': report['inventory'][
            'validated_seed0_screen_reports'
        ],
        'verified_hashed_provenance_files': (
            report['inventory']['verified_hashed_provenance_files']
        ),
        'rank0_lambda005_gate_passes': report[
            'promotion_gates_vs_static_dual'
        ]['joint_rank0_lambda005']['passes'],
        'gradient_audit_records': report['gradient_audit']['record_count'],
        'locked_shared_method': report['final_shared_selection']['method'],
        'json': str(json_path),
        'tsv': str(tsv_path),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
