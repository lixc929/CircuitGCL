#!/usr/bin/env python3
"""Validate and summarize the preregistered linear-lambda seed-0 control."""

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
    summary_stats,
)


METHOD_NAME = 'joint_rank0_lambda005_to0005_linear_seed0'
TRANSFER_NAMES = ('digtime', 'timing_ctrl', 'array_128_32_8t')
EXPECTED_AUDIT_EPOCHS = list(range(0, 160, 10))
PAIR_FIELDS = (
    'sgrl_cache_key',
    'sgrl_checkpoint_sha256',
    'split_fingerprint',
    'normalization_state_sha256',
    'downstream_initial_model_fingerprint',
    'train_sampler_fingerprint',
    'eval_sampler_fingerprints',
)
STATIC_PAIR_FIELDS = (
    'sgrl_cache_key',
    'sgrl_checkpoint_sha256',
    'split_fingerprint',
    'normalization_state_sha256',
)
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
CONTROL_EXPECTATIONS = {
    'joint_rank0_lambda005': {
        **COMMON_EXPECTATIONS,
        **JOINT_COMMON_EXPECTATIONS,
        **METHOD_EXPECTATIONS['joint_rank0_lambda005'],
        **STAGE_SEED_EXPECTATIONS,
    },
    'static_dual': {
        **COMMON_EXPECTATIONS,
        **METHOD_EXPECTATIONS['static_dual'],
        **STAGE_SEED_EXPECTATIONS,
    },
}
CORE_EXPECTATIONS = {
    **COMMON_EXPECTATIONS,
    **JOINT_COMMON_EXPECTATIONS,
    'hid_dim': 64,
    'sgrl': 1,
    'sgrl_mode': 'joint_shared',
    'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    'joint_lora_rank': 0,
    'joint_gcl_lambda': 0.05,
    'joint_gcl_lambda_schedule': 'linear',
    'joint_gcl_lambda_final': 0.005,
    'joint_gradient_audit': 1,
    'joint_gradient_audit_interval': 10,
    **STAGE_SEED_EXPECTATIONS,
    'gpu': 0,
}


def expected_lambda(epoch):
    if epoch < 0 or epoch >= 160:
        raise ValueError(f'Unexpected schedule epoch: {epoch}.')
    if epoch == 0:
        return 0.05
    if epoch == 159:
        return 0.005
    return 0.05 + (0.005 - 0.05) * epoch / 159


def _expect(payload, expectations, context):
    for key, expected in expectations.items():
        if key not in payload:
            raise ValueError(f'{context} is missing {key}.')
        actual = payload[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f'{context} mismatched {key}: {actual!r} != {expected!r}.'
            )


def _single_config(root, run_name, *, require_root_exclusive=False):
    paths = sorted((root / run_name).rglob('run_config.json'))
    if len(paths) != 1:
        raise ValueError(
            f'{root} must contain exactly one {run_name} artifact; '
            f'found {len(paths)}.'
        )
    if require_root_exclusive:
        all_paths = sorted(root.rglob('run_config.json'))
        if all_paths != paths:
            raise ValueError(
                f'{root} must contain only the single candidate artifact; '
                f'found total={len(all_paths)}.'
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
    return config, metrics, checkpoint


def _metric_row(metrics):
    val_mse = float(metrics.get('best_val_mse'))
    if not math.isfinite(val_mse):
        raise ValueError('Best validation MSE must be finite.')
    restored = result_mse(metrics.get('validation_results'))
    if not math.isclose(val_mse, restored, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError('Restored validation MSE does not reproduce selection.')
    tests = metrics.get('test_results')
    if not isinstance(tests, dict) or set(tests) != set(TRANSFER_NAMES):
        raise ValueError('Expected exactly the three development transfer results.')
    row = {
        'val_mse': val_mse,
        'digtime': result_mse(tests['digtime']),
        'timing_ctrl': result_mse(tests['timing_ctrl']),
        'array': result_mse(tests['array_128_32_8t']),
    }
    row['transfer_mean'] = sum(
        row[name] for name in ('digtime', 'timing_ctrl', 'array')
    ) / 3
    return row


def validate_audit_records(records):
    if [record.get('epoch') for record in records] != EXPECTED_AUDIT_EPOCHS:
        raise ValueError('Gradient audit epochs do not match 0..150 step 10.')
    for record in records:
        epoch = record['epoch']
        if record.get('batch_index') != 0:
            raise ValueError('Gradient audit must use batch zero.')
        if record.get('joint_gcl_lambda_schedule') != 'linear':
            raise ValueError('Gradient audit schedule must be linear.')
        if not math.isclose(
            float(record.get('joint_gcl_lambda')),
            expected_lambda(epoch),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f'Gradient audit lambda mismatch at epoch {epoch}.')
        cosine = record.get('global', {}).get('cosine')
        if cosine is None or not math.isfinite(float(cosine)):
            raise ValueError(f'Gradient audit cosine is invalid at epoch {epoch}.')
    return {
        'record_count': len(records),
        'global_cosine': summary_stats([
            float(record['global']['cosine']) for record in records
        ]),
    }


def _control(root, run_name, method):
    path = _single_config(root, run_name)
    config, metrics, _ = _load_completed(path, run_name)
    args = config.get('args')
    if not isinstance(args, dict):
        raise ValueError(f'{run_name} control is missing args.')
    _expect(args, CONTROL_EXPECTATIONS[method], f'{run_name} control args')
    runtime = config.get('runtime_metadata')
    if not isinstance(runtime, dict):
        raise ValueError(f'{run_name} control is missing runtime provenance.')
    _expect(runtime, {
        'source_graph_names': ['ssram'],
        'transfer_graph_names': list(TRANSFER_NAMES),
        'sgrl_train_graph_names': ['ssram'],
    }, f'{run_name} control runtime')
    return config, metrics, _metric_row(metrics)


def build_report(root, fixed_root, static_root):
    root = Path(root).resolve()
    fixed_root = Path(fixed_root).resolve()
    static_root = Path(static_root).resolve()
    manifest_path = root / 'selection_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing selection_manifest.json.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _expect(manifest, {
        'method': METHOD_NAME,
        'seeds': [0],
        'epochs': 160,
        'blind_circuits': [],
    }, 'manifest')
    schedule = manifest.get('schedule')
    if not isinstance(schedule, dict):
        raise ValueError('Manifest is missing its schedule.')
    _expect(schedule, {
        'name': 'linear', 'initial': 0.05, 'final': 0.005,
        'formula': '0.05 + (0.005 - 0.05) * epoch / 159',
    }, 'manifest schedule')

    config_path = _single_config(
        root, METHOD_NAME, require_root_exclusive=True
    )
    config, metrics, checkpoint = _load_completed(config_path, METHOD_NAME)
    if config.get('git_commit') != manifest.get('git_commit'):
        raise ValueError('Manifest and run commit do not match.')
    if config.get('command') != manifest.get('command'):
        raise ValueError('Manifest and executed command do not match.')
    args = config.get('args')
    if not isinstance(args, dict):
        raise ValueError('Run config is missing args.')
    _expect(args, CORE_EXPECTATIONS, 'candidate args')
    if 'sp8192w' in args['dataset'].lower():
        raise ValueError('Blind circuit appeared in candidate args.')
    runtime = config.get('runtime_metadata')
    if not isinstance(runtime, dict):
        raise ValueError('Candidate is missing runtime provenance.')
    if runtime.get('source_graph_names') != ['ssram']:
        raise ValueError('Candidate source graph provenance is invalid.')
    if runtime.get('transfer_graph_names') != list(TRANSFER_NAMES):
        raise ValueError('Candidate transfer graph provenance is invalid.')
    if runtime.get('sgrl_train_graph_names') != ['ssram']:
        raise ValueError('Candidate SGRL was not fit on source only.')

    best_epoch = metrics.get('best_epoch')
    if type(best_epoch) is not int or best_epoch < 0 or best_epoch >= 160:
        raise ValueError('Candidate has an invalid best epoch.')
    _expect(metrics, {
        'joint_gcl_lambda': 0.05,
        'joint_gcl_lambda_schedule': 'linear',
        'joint_gcl_lambda_final': 0.005,
        'joint_gcl_lambda_effective_last': 0.005,
    }, 'candidate metrics')
    if not math.isclose(
        float(metrics.get('joint_gcl_lambda_at_best_epoch')),
        expected_lambda(best_epoch),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError('Candidate best-epoch effective lambda is invalid.')
    candidate_row = _metric_row(metrics)

    audit_path = Path(runtime.get('joint_gradient_audit_path', ''))
    expected_audit_sha = runtime.get('joint_gradient_audit_sha256')
    if not audit_path.is_file() or file_sha256(audit_path) != expected_audit_sha:
        raise ValueError('Gradient audit path/SHA provenance is invalid.')
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    audit = validate_audit_records(records)
    recorded_audit = metrics.get('joint_gradient_audit_summary')
    if not isinstance(recorded_audit, dict):
        raise ValueError('Metrics are missing the gradient-audit summary.')
    if recorded_audit.get('schema_version') != 2:
        raise ValueError('Gradient-audit summary schema must be version 2.')
    if recorded_audit.get('record_count') != 16:
        raise ValueError('Gradient-audit summary count must be 16.')
    _expect(recorded_audit.get('lambda_schedule', {}), {
        'initial': 0.05, 'schedule': 'linear', 'final': 0.005, 'epochs': 160,
    }, 'gradient-audit summary')
    audit_summary_path = Path(runtime.get('joint_gradient_summary_path', ''))
    expected_summary_sha = runtime.get('joint_gradient_summary_sha256')
    if (
        not audit_summary_path.is_file()
        or file_sha256(audit_summary_path) != expected_summary_sha
    ):
        raise ValueError('Gradient summary path/SHA provenance is invalid.')
    if json.loads(audit_summary_path.read_text(encoding='utf-8')) != recorded_audit:
        raise ValueError('Gradient summary file and metrics do not match.')

    fixed_config, fixed_metrics, fixed_row = _control(
        fixed_root, 'joint_rank0_lambda005_seed0',
        'joint_rank0_lambda005',
    )
    static_config, _, static_row = _control(
        static_root, 'static_dual', 'static_dual'
    )
    fixed_runtime = fixed_config.get('runtime_metadata', {})
    mismatches = [
        field for field in PAIR_FIELDS
        if runtime.get(field) is None
        or runtime.get(field) != fixed_runtime.get(field)
    ]
    if mismatches:
        raise ValueError(f'Candidate/fixed provenance mismatch: {mismatches}.')
    static_runtime = static_config.get('runtime_metadata', {})
    static_mismatches = [
        field for field in STATIC_PAIR_FIELDS
        if runtime.get(field) is None
        or runtime.get(field) != static_runtime.get(field)
    ]
    if static_mismatches:
        raise ValueError(
            'Candidate/static provenance mismatch: '
            f'{static_mismatches}.'
        )

    vs_static = {
        metric: candidate_row[metric] / static_row[metric] - 1.0
        for metric in candidate_row
    }
    vs_fixed = {
        metric: candidate_row[metric] / fixed_row[metric] - 1.0
        for metric in candidate_row
    }
    gate_checks = {
        'val_mse': vs_static['val_mse'] <= 0.05,
        'transfer_mean': vs_static['transfer_mean'] <= 0.10,
        'digtime': vs_static['digtime'] <= 0.25,
        'timing_ctrl': vs_static['timing_ctrl'] <= 0.25,
        'array': vs_static['array'] <= 0.25,
    }
    validation_improvement = fixed_row['val_mse'] - candidate_row['val_mse']
    return {
        'schema_version': 1,
        'method': METHOD_NAME,
        'git_commit': config.get('git_commit'),
        'artifact_dir': str(config_path.parent),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': file_sha256(checkpoint),
        'candidate': candidate_row,
        'fixed_lambda005': fixed_row,
        'static_dual': static_row,
        'relative_to_static': vs_static,
        'relative_to_fixed_lambda005': vs_fixed,
        'promotion_gate': {
            'checks': gate_checks,
            'passes': all(gate_checks.values()),
        },
        'selection': {
            'source_val_improvement_over_fixed': validation_improvement,
            'minimum_improvement_to_expand': 2e-5,
            'advances_to_seeds12': (
                all(gate_checks.values()) and validation_improvement > 2e-5
            ),
        },
        'paired_provenance': {
            'passes': True,
            'fixed_lambda005_matched_fields': list(PAIR_FIELDS),
            'static_dual_matched_fields': list(STATIC_PAIR_FIELDS),
        },
        'gradient_audit': audit,
        'sources': {
            'candidate_config': str(config_path),
            'candidate_metrics': str(config_path.with_name('metrics.json')),
            'fixed_root': str(fixed_root),
            'static_root': str(static_root),
        },
    }


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / 'schedule_seed0_summary.json'
    tsv_path = output_dir / 'schedule_seed0_summary.tsv'
    atomic_write_text(
        json_path, json.dumps(report, indent=2, sort_keys=True) + '\n'
    )
    buffer = io.StringIO()
    columns = [
        'method', 'val_mse', 'digtime', 'timing_ctrl', 'array',
        'transfer_mean', 'promotion_gate_passes', 'advances_to_seeds12',
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter='\t')
    writer.writeheader()
    writer.writerow({
        'method': METHOD_NAME,
        **report['candidate'],
        'promotion_gate_passes': report['promotion_gate']['passes'],
        'advances_to_seeds12': report['selection']['advances_to_seeds12'],
    })
    atomic_write_text(tsv_path, buffer.getvalue())
    return json_path, tsv_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument(
        '--fixed_root', type=Path,
        default=Path('logs/strict_joint_rank0_lambda005_seed0_20260712'),
    )
    parser.add_argument(
        '--static_root', type=Path,
        default=Path('logs/strict_seed0_seven_20260712_v2'),
    )
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.root, args.fixed_root, args.static_root)
    json_path, tsv_path = write_report(
        report, args.output_dir or args.root
    )
    print(json.dumps({
        'promotion_gate_passes': report['promotion_gate']['passes'],
        'advances_to_seeds12': report['selection']['advances_to_seeds12'],
        'json': str(json_path),
        'tsv': str(tsv_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
