#!/usr/bin/env python3
"""Validate and summarize the compact shared-GraphSAGE seed-0 screen."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

from summarize_strict_reuse import (
    COMMON_EXPECTATIONS,
    JOINT_COMMON_EXPECTATIONS,
    METHOD_EXPECTATIONS,
    atomic_write_text,
    file_sha256,
    result_mse,
)


SCREEN_NAME = 'compact_shared_sage_seed0_screen'
LAMBDA0_NAME = 'joint_sage_rank0_lambda0_seed0'
LAMBDA005_NAME = 'joint_sage_rank0_lambda005_seed0'
RUN_NAMES = (LAMBDA0_NAME, LAMBDA005_NAME)
TRANSFER_NAMES = ('digtime', 'timing_ctrl', 'array_128_32_8t')
METRICS = ('val_mse', 'digtime', 'timing_ctrl', 'array', 'transfer_mean')
GATES = {
    'val_mse': 0.05,
    'transfer_mean': 0.10,
    'digtime': 0.25,
    'timing_ctrl': 0.25,
    'array': 0.25,
}
MINIMUM_SOURCE_IMPROVEMENT = 2e-5
SOURCE_VAL_TIE_TOLERANCE = 2e-5
EXPECTED_SAGE_CACHE_KEY = '21a24d76e338d82eaf86'
REGISTERED_STATIC_ROW = {
    'val_mse': 0.007809748407453299,
    'digtime': 0.016486799344420433,
    'timing_ctrl': 0.010408373549580574,
    'array': 0.01004203874617815,
    'transfer_mean': 0.012312403880059719,
}
REGISTERED_CLUSTER_LAMBDA005_VAL = 0.008035913109779358
REGISTERED_UPPER_BOUNDS = {
    'val_mse': 0.008200235827825964,
    'digtime': 0.02060849918052554,
    'timing_ctrl': 0.013010466936975718,
    'array': 0.012552548432722688,
    'transfer_mean': 0.013543644268065692,
}
REGISTERED_REQUIRED_VAL = 0.008015913109779358
STAGE_SEED_EXPECTATIONS = {
    'seed': 0,
    'pretraining_seed': 0,
    'embedding_inference_seed': 20260711,
    'downstream_seed': 0,
    'train_sampler_seed': 0,
    'relation_sample_seed': 20260711,
    'split_seed': 0,
    'eval_seed': 20260711,
}
RESOLVED_SEED_EXPECTATIONS = {
    key: value for key, value in STAGE_SEED_EXPECTATIONS.items()
    if key != 'seed'
}

SAGE_COMMON_EXPECTATIONS = {
    **COMMON_EXPECTATIONS,
    'cl_model': 'sage',
    'model': 'sage',
    'e1_lr': 1e-6,
    'e2_lr': 2e-7,
    'gpu': 0,
    'hid_dim': 64,
    'sgrl': 1,
    'sgrl_mode': 'joint_shared',
    'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    'joint_shared_gnn_layers': 2,
    'joint_lora_rank': 0,
    'joint_lora_layer': -1,
    'joint_lora_alpha': None,
    'joint_backbone_lr': None,
    'joint_gcl_lambda_schedule': 'constant',
    'joint_gcl_lambda_final': None,
    'joint_gradient_strategy': 'none',
    'joint_gradient_audit_interval': 10,
    **STAGE_SEED_EXPECTATIONS,
}
SAGE_METHOD_EXPECTATIONS = {
    LAMBDA0_NAME: {
        'joint_gcl_lambda': 0.0,
        'joint_gradient_audit': 0,
    },
    LAMBDA005_NAME: {
        'joint_gcl_lambda': 0.05,
        'joint_gradient_audit': 0,
    },
}
PAIR_FIELDS = (
    'sgrl_cache_key',
    'sgrl_checkpoint_sha256',
    'sgrl_checkpoint_metadata_sha256',
    'processed_caches',
    'split_fingerprint',
    'split_indices_sha256',
    'normalization_state_sha256',
    'downstream_initial_model_fingerprint',
    'train_sampler_fingerprint',
    'eval_sampler_fingerprints',
)
CROSS_ARCH_PROVENANCE_FIELDS = (
    'processed_caches',
    'split_fingerprint',
    'split_indices_sha256',
    'normalization_state_sha256',
    'train_sampler_fingerprint',
    'eval_sampler_fingerprints',
)
EMBEDDING_RUNTIME_KEYS = (
    'embedding_cache_key',
    'embedding_path',
    'embedding_sha256',
    'embedding_metadata_path',
    'embedding_metadata_sha256',
    'embedding_sampler_fingerprint',
)
EXPECTED_REPRESENTATION_TRAIN_EPOCHS = (-1, *range(0, 160, 10))
REPRESENTATION_FINITE_FIELDS = (
    'base_cosine_to_initial',
    'base_norm_mean',
    'base_relative_l2_to_initial',
    'base_weight_relative_l2',
    'task_cosine_to_base',
    'task_norm_mean',
    'task_relative_l2_to_base',
    'stats_residual_scale',
)


def _expect(payload, expectations, context):
    if not isinstance(payload, dict):
        raise ValueError(f'{context} is not an object.')
    for key, expected in expectations.items():
        if key not in payload:
            raise ValueError(f'{context} is missing {key}.')
        actual = payload[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f'{context} mismatched {key}: {actual!r} != {expected!r}.'
            )


def _single_config(root, run_name):
    paths = sorted((Path(root) / run_name).rglob('run_config.json'))
    if len(paths) != 1:
        raise ValueError(
            f'{root}/{run_name} must contain exactly one run_config.json; '
            f'found {len(paths)}.'
        )
    return paths[0]


def _load_completed(config_path, context):
    metrics_path = config_path.with_name('metrics.json')
    if not metrics_path.is_file():
        raise ValueError(f'{context} is missing metrics.json.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if config.get('status') != 'completed' or metrics.get('status') != 'completed':
        raise ValueError(f'{context} is not completed.')
    checkpoint = Path(metrics.get('best_checkpoint', ''))
    if not checkpoint.is_file():
        raise ValueError(f'{context} is missing its best checkpoint.')
    return config, metrics, metrics_path, checkpoint


def _metric_row(metrics, context):
    value = metrics.get('best_val_mse')
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f'{context} has invalid best_val_mse.')
    val_mse = float(value)
    restored = result_mse(metrics.get('validation_results'))
    if not math.isclose(val_mse, restored, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f'{context} restored validation MSE does not match.')
    tests = metrics.get('test_results')
    if not isinstance(tests, dict) or set(tests) != set(TRANSFER_NAMES):
        raise ValueError(f'{context} must contain exactly three transfer results.')
    row = {
        'val_mse': val_mse,
        'digtime': result_mse(tests['digtime']),
        'timing_ctrl': result_mse(tests['timing_ctrl']),
        'array': result_mse(tests['array_128_32_8t']),
    }
    if not all(math.isfinite(float(value)) for value in row.values()):
        raise ValueError(f'{context} has a non-finite metric.')
    row['transfer_mean'] = (
        row['digtime'] + row['timing_ctrl'] + row['array']
    ) / 3.0
    return row


def _validate_representation_audit(path, best_epoch, context):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{context} is missing its representation audit.')
    records = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    train_records = [
        record for record in records if record.get('phase') == 'train'
    ]
    best_records = [
        record for record in records if record.get('phase') == 'best'
    ]
    if [record.get('epoch') for record in train_records] != list(
            EXPECTED_REPRESENTATION_TRAIN_EPOCHS):
        raise ValueError(f'{context} representation train epochs are invalid.')
    if any(record.get('split') != 'source_val' for record in train_records):
        raise ValueError(f'{context} representation train split is invalid.')
    expected_best_splits = {'source_val', *TRANSFER_NAMES}
    if (
        len(best_records) != len(expected_best_splits)
        or {record.get('split') for record in best_records}
        != expected_best_splits
        or any(record.get('epoch') != best_epoch for record in best_records)
    ):
        raise ValueError(f'{context} representation best records are invalid.')
    if len(records) != (
            len(EXPECTED_REPRESENTATION_TRAIN_EPOCHS)
            + len(expected_best_splits)):
        raise ValueError(f'{context} representation record count is invalid.')
    for record in records:
        for field in REPRESENTATION_FINITE_FIELDS:
            value = record.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(
                    f'{context} representation {field} is invalid.'
                )
        if record.get('lora_modules') != 0:
            raise ValueError(f'{context} representation audit used LoRA.')
    return {
        'path': path,
        'sha256': file_sha256(path),
        'record_count': len(records),
    }


def _validate_runtime(runtime, context):
    _expect(runtime, {
        'source_graph_names': ['ssram'],
        'transfer_graph_names': list(TRANSFER_NAMES),
        'sgrl_train_graph_names': ['ssram'],
        'resolved_seeds': RESOLVED_SEED_EXPECTATIONS,
        'normalization_scope': 'source',
    }, f'{context} runtime')
    for path_key, sha_key in (
        ('sgrl_checkpoint_path', 'sgrl_checkpoint_sha256'),
        ('sgrl_checkpoint_metadata_path', 'sgrl_checkpoint_metadata_sha256'),
        ('normalization_state_path', 'normalization_state_sha256'),
        ('split_indices_path', 'split_indices_sha256'),
    ):
        path = Path(runtime.get(path_key, ''))
        expected_sha = runtime.get(sha_key)
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(
                f'{context} has invalid {path_key}/{sha_key} provenance.'
            )


def _canonical_sgrl_processed_refs(processed_caches):
    refs = []
    for cache in processed_caches:
        if cache.get('graph_name') != 'ssram':
            continue
        refs.append({
            'graph_name': cache.get('graph_name'),
            'cache_key': cache.get('cache_key'),
            'raw_sha256': cache.get('raw_sha256'),
            'processed_sha256': cache.get('processed_sha256'),
            'relation_sample_seed': cache.get('relation_sample_seed'),
            'graph_relation_sample_seed': cache.get(
                'graph_relation_sample_seed'
            ),
        })
    return sorted(refs, key=lambda value: str(value['graph_name']))


def _validate_sage_checkpoint(runtime, context):
    cache_key = runtime.get('sgrl_cache_key')
    if cache_key != EXPECTED_SAGE_CACHE_KEY:
        raise ValueError(
            f'{context} SAGE cache key mismatch: {cache_key!r} != '
            f'{EXPECTED_SAGE_CACHE_KEY!r}.'
        )
    if any(key in runtime for key in EMBEDDING_RUNTIME_KEYS):
        raise ValueError(f'{context} unexpectedly used an embedding cache.')
    processed_caches = runtime.get('processed_caches')
    if not isinstance(processed_caches, list) or len(processed_caches) != 4:
        raise ValueError(f'{context} processed-cache provenance is invalid.')

    checkpoint_path = Path(runtime['sgrl_checkpoint_path'])
    if checkpoint_path.name != f'best_online_{cache_key}.pkl':
        raise ValueError(f'{context} online checkpoint path is noncanonical.')
    metadata_path = Path(runtime['sgrl_checkpoint_metadata_path'])
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    identity = metadata.get('identity')
    _expect(metadata, {
        'cache_schema_version': 1,
        'cache_key': cache_key,
        'online_checkpoint_sha256': runtime['sgrl_checkpoint_sha256'],
    }, f'{context} SGRL metadata')
    _expect(identity, {
        'cache_schema_version': 1,
        'train_graph_names': ['ssram'],
        'processed_caches': _canonical_sgrl_processed_refs(processed_caches),
        'graph_scope': 'source',
        'pretraining_seed': 0,
        'target_update': 'sgrl_dual_rsm_ema',
        'model': 'sage',
        'layers': 2,
        'hidden_dim': 64,
        'activation': 'tanh',
        'dropout': 0.3,
        'batch_size': 32768,
        'num_neighbors': 8,
        'num_hops': 2,
        'num_workers': 0,
        'use_bn': 0,
        'epochs': 5,
        'online_lr': 1e-6,
        'target_lr': 2e-7,
        'momentum': 0.99,
        'weight_decay': 0.0,
    }, f'{context} SGRL identity')

    target_sha = metadata.get('target_checkpoint_sha256')
    if not isinstance(target_sha, str) or len(target_sha) != 64:
        raise ValueError(f'{context} target checkpoint hash is invalid.')
    target_path = (
        checkpoint_path.parent.parent
        / 'pkl_target'
        / checkpoint_path.name.replace('best_online_', 'best_target_', 1)
    )
    if not target_path.is_file() or file_sha256(target_path) != target_sha:
        raise ValueError(f'{context} target checkpoint provenance is invalid.')


def _control_expectations(method):
    expectations = {
        **COMMON_EXPECTATIONS,
        **METHOD_EXPECTATIONS[method],
        **STAGE_SEED_EXPECTATIONS,
    }
    if method.startswith('joint_'):
        expectations.update(JOINT_COMMON_EXPECTATIONS)
    return expectations


def _load_cluster_control(root, run_name, method):
    context = f'{Path(root).name}/{run_name}'
    config_path = _single_config(root, run_name)
    config, metrics, metrics_path, checkpoint = _load_completed(
        config_path, context
    )
    args = config.get('args')
    _expect(args, _control_expectations(method), f'{context} args')
    if 'sp8192w' in str(args.get('dataset', '')).lower():
        raise ValueError(f'{context} includes the blind circuit.')
    runtime = config.get('runtime_metadata')
    _validate_runtime(runtime, context)
    return {
        'id': method,
        'config': config,
        'metrics': metrics,
        'runtime': runtime,
        'row': _metric_row(metrics, context),
        'config_path': config_path,
        'metrics_path': metrics_path,
        'checkpoint': checkpoint,
    }


def _repo_path(path, repo_root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(repo_root).resolve()))
    except ValueError:
        return str(path)


def _artifact_source(run, repo_root):
    source = {
        'git_commit': run['config'].get('git_commit'),
        'config_path': _repo_path(run['config_path'], repo_root),
        'config_sha256': file_sha256(run['config_path']),
        'metrics_path': _repo_path(run['metrics_path'], repo_root),
        'metrics_sha256': file_sha256(run['metrics_path']),
        'checkpoint_path': _repo_path(run['checkpoint'], repo_root),
        'checkpoint_sha256': file_sha256(run['checkpoint']),
    }
    audit = run.get('representation_audit')
    if audit is not None:
        source.update({
            'representation_audit_path': _repo_path(
                audit['path'], repo_root
            ),
            'representation_audit_sha256': audit['sha256'],
            'representation_audit_records': audit['record_count'],
        })
    return source


def _load_controls(static_root, fixed_root):
    return {
        'canonical_static_dual': _load_cluster_control(
            static_root, 'static_dual', 'static_dual'
        ),
        'cluster_rank0_lambda0': _load_cluster_control(
            static_root, 'joint_lambda0', 'joint_rank0_lambda0'
        ),
        'cluster_rank0_lambda005': _load_cluster_control(
            fixed_root,
            'joint_rank0_lambda005_seed0',
            'joint_rank0_lambda005',
        ),
    }


def validate_control_artifacts(repo_root, static_root, fixed_root):
    """Validate immutable incumbent controls before queuing a long screen."""
    controls = _load_controls(static_root, fixed_root)
    return {
        name: _artifact_source(run, repo_root)
        for name, run in controls.items()
    }


def _load_sage_run(root, run_name, manifest):
    context = f'{Path(root).name}/{run_name}'
    config_path = _single_config(root, run_name)
    config, metrics, metrics_path, checkpoint = _load_completed(
        config_path, context
    )
    if config.get('git_commit') != manifest.get('git_commit'):
        raise ValueError(f'{context} commit differs from the manifest.')
    expected_command = manifest.get('commands', {}).get(run_name)
    if config.get('command') != expected_command:
        raise ValueError(f'{context} command differs from the manifest.')
    args = config.get('args')
    _expect(args, SAGE_COMMON_EXPECTATIONS, f'{context} args')
    _expect(args, SAGE_METHOD_EXPECTATIONS[run_name], f'{context} args')
    if 'sp8192w' in str(args.get('dataset', '')).lower():
        raise ValueError(f'{context} includes the blind circuit.')
    runtime = config.get('runtime_metadata')
    _validate_runtime(runtime, context)
    _validate_sage_checkpoint(runtime, context)
    expected_lambda = SAGE_METHOD_EXPECTATIONS[run_name]['joint_gcl_lambda']
    _expect(metrics, {
        'joint_gcl_lambda': expected_lambda,
        'joint_gcl_lambda_schedule': 'constant',
        'joint_gcl_lambda_final': None,
        'joint_gcl_lambda_effective_last': expected_lambda,
        'joint_gcl_lambda_at_best_epoch': expected_lambda,
        'joint_gradient_strategy': 'none',
    }, f'{context} metrics')
    run = {
        'id': run_name,
        'config': config,
        'metrics': metrics,
        'runtime': runtime,
        'row': _metric_row(metrics, context),
        'config_path': config_path,
        'metrics_path': metrics_path,
        'checkpoint': checkpoint,
    }
    if metrics.get('joint_gradient_audit_summary') is not None:
        raise ValueError(f'{context} unexpectedly contains a gradient audit.')
    run['representation_audit'] = _validate_representation_audit(
        config_path.parent / 'joint_shared_representation_audit.jsonl',
        metrics.get('best_epoch'),
        context,
    )
    return run


def _relative(candidate, reference):
    return {
        metric: candidate[metric] / reference[metric] - 1.0
        for metric in METRICS
    }


def _comparison(candidate_name, candidate, reference_name, reference):
    relative = _relative(candidate, reference)
    return {
        'candidate': candidate_name,
        'reference': reference_name,
        'candidate_metrics': candidate,
        'reference_metrics': reference,
        'relative_delta': relative,
    }


def _promotion_gate(candidate, reference):
    relative = _relative(candidate, reference)
    checks = {
        metric: {
            'observed_degradation': relative[metric],
            'maximum_degradation': limit,
            'passes': relative[metric] <= limit,
        }
        for metric, limit in GATES.items()
    }
    return {
        'checks': checks,
        'passes': all(check['passes'] for check in checks.values()),
    }


def _public_run(run, repo_root):
    return {
        'method': run['id'],
        **run['row'],
        'best_epoch': run['metrics'].get('best_epoch'),
        'artifact': _artifact_source(run, repo_root),
    }


def build_report(root, static_root, fixed_root, repo_root=None):
    root = Path(root).resolve()
    static_root = Path(static_root).resolve()
    fixed_root = Path(fixed_root).resolve()
    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    manifest_path = root / 'selection_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing selection_manifest.json.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _expect(manifest, {
        'schema_version': 1,
        'protocol': 'strict_inductive',
        'method': SCREEN_NAME,
        'methods': list(RUN_NAMES),
        'seed': 0,
        'epochs': 160,
        'blind_circuits': [],
    }, 'manifest')
    _expect(manifest.get('execution'), {
        'ordering': list(RUN_NAMES),
        'checkpoint_producer': LAMBDA0_NAME,
        'checkpoint_readers': [LAMBDA005_NAME],
        'concurrent_checkpoint_writers': 1,
    }, 'manifest execution')
    _expect(manifest.get('selection'), {
        'candidates': list(RUN_NAMES),
        'primary_metric': 'source validation raw MSE',
        'source_val_tie_tolerance_prefer_lambda005': (
            SOURCE_VAL_TIE_TOLERANCE
        ),
        'minimum_source_val_improvement_over_cluster_lambda005': (
            MINIMUM_SOURCE_IMPROVEMENT
        ),
        'registered_references': {
            'canonical_static': REGISTERED_STATIC_ROW,
            'cluster_rank0_lambda005_val_mse': (
                REGISTERED_CLUSTER_LAMBDA005_VAL
            ),
        },
        'registered_upper_bounds': REGISTERED_UPPER_BOUNDS,
        'required_val_mse_strictly_below': REGISTERED_REQUIRED_VAL,
        'automatic_multiseed_expansion': False,
    }, 'manifest selection')
    discovered = {
        path.relative_to(root).parts[0]
        for path in root.rglob('run_config.json')
    }
    if discovered != set(RUN_NAMES):
        raise ValueError(
            f'SAGE screen run matrix mismatch: {sorted(discovered)} != '
            f'{sorted(RUN_NAMES)}.'
        )

    controls = _load_controls(static_root, fixed_root)
    control_artifacts = {
        name: _artifact_source(run, repo_root)
        for name, run in controls.items()
    }
    if manifest.get('control_artifacts') != control_artifacts:
        raise ValueError('Control artifacts changed after preregistration.')

    sage_runs = {
        name: _load_sage_run(root, name, manifest)
        for name in RUN_NAMES
    }
    lambda0 = sage_runs[LAMBDA0_NAME]
    lambda005 = sage_runs[LAMBDA005_NAME]
    mismatches = [
        field for field in PAIR_FIELDS
        if lambda0['runtime'].get(field) is None
        or lambda0['runtime'].get(field) != lambda005['runtime'].get(field)
    ]
    if mismatches:
        raise ValueError(f'SAGE lambda-pair provenance mismatch: {mismatches}.')

    canonical_runtime = controls['cluster_rank0_lambda005']['runtime']
    cross_arch_mismatches = [
        field for field in CROSS_ARCH_PROVENANCE_FIELDS
        if lambda0['runtime'].get(field) is None
        or lambda0['runtime'].get(field) != canonical_runtime.get(field)
    ]
    if cross_arch_mismatches:
        raise ValueError(
            'SAGE/canonical provenance mismatch: '
            f'{cross_arch_mismatches}.'
        )

    static_row = controls['canonical_static_dual']['row']
    cluster0_row = controls['cluster_rank0_lambda0']['row']
    cluster005_row = controls['cluster_rank0_lambda005']['row']
    sage0_row = lambda0['row']
    sage005_row = lambda005['row']
    for metric, registered in REGISTERED_STATIC_ROW.items():
        if not math.isclose(
                static_row[metric], registered, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f'Canonical static {metric} changed.')
    if not math.isclose(
            cluster005_row['val_mse'], REGISTERED_CLUSTER_LAMBDA005_VAL,
            rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('Cluster lambda=.05 source validation changed.')
    gates = {
        LAMBDA0_NAME: _promotion_gate(sage0_row, static_row),
        LAMBDA005_NAME: _promotion_gate(sage005_row, static_row),
    }
    gate_eligible = [name for name in RUN_NAMES if gates[name]['passes']]
    candidate_rows = {
        LAMBDA0_NAME: sage0_row,
        LAMBDA005_NAME: sage005_row,
    }
    advancement_eligible = [
        name for name in gate_eligible
        if (
            cluster005_row['val_mse']
            - candidate_rows[name]['val_mse']
        ) > MINIMUM_SOURCE_IMPROVEMENT
    ]
    selected = None
    selection_reason = (
        'no_candidate_passed_promotion_gates'
        if not gate_eligible
        else 'no_candidate_met_source_improvement'
    )
    if len(advancement_eligible) == 1:
        selected = advancement_eligible[0]
        selection_reason = 'only_candidate_meeting_full_advancement_rule'
    elif len(advancement_eligible) == 2:
        val_delta = abs(sage0_row['val_mse'] - sage005_row['val_mse'])
        if val_delta <= SOURCE_VAL_TIE_TOLERANCE:
            selected = LAMBDA005_NAME
            selection_reason = 'source_val_tie_prefer_positive_lambda'
        elif sage0_row['val_mse'] < sage005_row['val_mse']:
            selected = LAMBDA0_NAME
            selection_reason = 'lower_source_validation_mse'
        else:
            selected = LAMBDA005_NAME
            selection_reason = 'lower_source_validation_mse'
    selected_row = sage_runs[selected]['row'] if selected is not None else None
    source_improvement = (
        cluster005_row['val_mse'] - selected_row['val_mse']
        if selected_row is not None else None
    )
    advances = selected is not None
    comparisons = {
        'sage_lambda0_vs_canonical_static': _comparison(
            LAMBDA0_NAME, sage0_row,
            'canonical_static_dual', static_row,
        ),
        'sage_lambda005_vs_canonical_static': _comparison(
            LAMBDA005_NAME, sage005_row,
            'canonical_static_dual', static_row,
        ),
        'sage_lambda0_vs_cluster_lambda0': _comparison(
            LAMBDA0_NAME, sage0_row,
            'cluster_rank0_lambda0', cluster0_row,
        ),
        'sage_lambda005_vs_cluster_lambda005': _comparison(
            LAMBDA005_NAME, sage005_row,
            'cluster_rank0_lambda005', cluster005_row,
        ),
        'sage_lambda005_vs_sage_lambda0': _comparison(
            LAMBDA005_NAME, sage005_row, LAMBDA0_NAME, sage0_row
        ),
    }
    return {
        'schema_version': 1,
        'report_kind': 'compact_shared_sage_seed0_screen',
        'git_commit': manifest.get('git_commit'),
        'root': _repo_path(root, repo_root),
        'protocol': {
            'source': 'ssram',
            'transfer': list(TRANSFER_NAMES),
            'blind_circuit_present': False,
            'epochs': 160,
            'early_stopping': False,
            'fixed_view_seed': 20260711,
            'promotion_thresholds': GATES,
        },
        'runs': {
            name: _public_run(run, repo_root)
            for name, run in sage_runs.items()
        },
        'controls': {
            name: {
                **run['row'],
                'artifact': control_artifacts[name],
            }
            for name, run in controls.items()
        },
        'comparisons': comparisons,
        'promotion_gates_vs_canonical_static': gates,
        'selection': {
            'gate_eligible_candidates': gate_eligible,
            'advancement_eligible_candidates': advancement_eligible,
            'selected_candidate': selected,
            'selection_reason': selection_reason,
            'source_val_improvement_over_cluster_lambda005': source_improvement,
            'minimum_improvement_to_expand': MINIMUM_SOURCE_IMPROVEMENT,
            'source_val_tie_tolerance_prefer_lambda005': (
                SOURCE_VAL_TIE_TOLERANCE
            ),
            'advances_to_seeds12': advances,
            'automatic_expansion_permitted': False,
        },
        'paired_provenance': {
            'passes': True,
            'matched_fields': list(PAIR_FIELDS),
            'checkpoint_producer': LAMBDA0_NAME,
            'checkpoint_reader': LAMBDA005_NAME,
        },
        'manifest': {
            'path': _repo_path(manifest_path, repo_root),
            'sha256': file_sha256(manifest_path),
        },
    }


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / 'sage_seed0_summary.json'
    tsv_path = output_dir / 'sage_seed0_summary.tsv'
    atomic_write_text(
        json_path, json.dumps(report, indent=2, sort_keys=True) + '\n'
    )
    columns = [
        'record_type', 'id', 'method', 'reference', *METRICS,
        'passes', 'advances_to_seeds12', 'best_epoch',
    ]
    rows = []
    for name, run in report['runs'].items():
        rows.append({
            'record_type': 'run', 'id': name, 'method': name,
            **{metric: run[metric] for metric in METRICS},
            'best_epoch': run['best_epoch'],
        })
    for name, control in report['controls'].items():
        rows.append({
            'record_type': 'control', 'id': name, 'method': name,
            **{metric: control[metric] for metric in METRICS},
        })
    for name, comparison in report['comparisons'].items():
        rows.append({
            'record_type': 'relative_delta', 'id': name,
            'method': comparison['candidate'],
            'reference': comparison['reference'],
            **comparison['relative_delta'],
        })
    for name, gate in report['promotion_gates_vs_canonical_static'].items():
        rows.append({
            'record_type': 'promotion_gate', 'id': name, 'method': name,
            'reference': 'canonical_static_dual',
            **{
                metric: gate['checks'][metric]['observed_degradation']
                for metric in GATES
            },
            'passes': gate['passes'],
            'advances_to_seeds12': (
                report['selection']['advances_to_seeds12']
                if name == report['selection']['selected_candidate']
                else False
            ),
        })
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(tsv_path, buffer.getvalue())
    return json_path, tsv_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument(
        '--static_root', type=Path,
        default=Path('logs/strict_seed0_seven_20260712_v2'),
    )
    parser.add_argument(
        '--fixed_root', type=Path,
        default=Path('logs/strict_joint_rank0_lambda005_seed0_20260712'),
    )
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.root, args.static_root, args.fixed_root)
    json_path, tsv_path = write_report(
        report, args.output_dir or args.root
    )
    print(json.dumps({
        'selected_candidate': report['selection']['selected_candidate'],
        'candidate_gate_passes': {
            name: gate['passes']
            for name, gate in report[
                'promotion_gates_vs_canonical_static'
            ].items()
        },
        'advances_to_seeds12': report['selection']['advances_to_seeds12'],
        'json': str(json_path),
        'tsv': str(tsv_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
