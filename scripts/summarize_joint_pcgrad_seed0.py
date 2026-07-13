#!/usr/bin/env python3
"""Validate and summarize the preregistered rank-0 PCGrad seed-0 control."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

from summarize_joint_lambda_schedule_seed0 import (
    PAIR_FIELDS,
    STATIC_PAIR_FIELDS,
    STAGE_SEED_EXPECTATIONS,
    TRANSFER_NAMES,
    _control,
    _expect,
    _load_completed,
    _metric_row,
    _single_config,
)
from summarize_strict_reuse import (
    COMMON_EXPECTATIONS,
    JOINT_COMMON_EXPECTATIONS,
    _verify_hashed_paths,
    atomic_write_text,
    file_sha256,
    summary_stats,
)


METHOD_NAME = 'joint_rank0_lambda005_pcgrad_seed0'
EXPECTED_COMPARISON_ROOTS = [
    'logs/strict_joint_rank0_lambda005_seed0_20260712',
    'logs/strict_seed0_seven_20260712_v2',
    'logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712',
]
EXPECTED_GRADIENT_EPOCHS = list(range(0, 160, 10))
EXPECTED_PCGRAD_EPOCHS = list(range(160))
EXPECTED_GRADIENT_GROUPS = {
    'embeddings', 'layers_0', 'layers_1', 'normalization_activation'
}
EXPECTED_SCOPE_PARAMETER_NAMES = [
    'node_type_embed.weight',
    'edge_type_embed.weight',
    'layers.0.lin_out.weight',
    'layers.0.lin_out.bias',
    'layers.0.lin_root.weight',
    'layers.1.lin_out.weight',
    'layers.1.lin_out.bias',
    'layers.1.lin_root.weight',
    'bn_node_x.weight',
    'bn_node_x.bias',
]
CORE_EXPECTATIONS = {
    **COMMON_EXPECTATIONS,
    **JOINT_COMMON_EXPECTATIONS,
    'hid_dim': 64,
    'sgrl': 1,
    'sgrl_mode': 'joint_shared',
    'sgrl_pretrain_target_update': 'sgrl_dual_rsm_ema',
    'joint_lora_rank': 0,
    'joint_gcl_lambda': 0.05,
    'joint_gcl_lambda_schedule': 'constant',
    'joint_gcl_lambda_final': None,
    'joint_gradient_strategy': 'pcgrad',
    'joint_gradient_audit': 1,
    'joint_gradient_audit_interval': 10,
    **STAGE_SEED_EXPECTATIONS,
    'gpu': 0,
}


def _finite(value, context):
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f'{context} must be finite.')
    return float(value)


def _validated_metric_row(row, context):
    expected = {'val_mse', 'digtime', 'timing_ctrl', 'array', 'transfer_mean'}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f'{context} must contain exactly {sorted(expected)}.')
    return {
        metric: _finite(value, f'{context} {metric}')
        for metric, value in row.items()
    }


def _require_matching_rows(first, second, context):
    for metric in first:
        if not math.isclose(
            float(first[metric]),
            float(second[metric]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f'{context} mismatched {metric}.')


def _validate_cosine_distribution(payload, expected_count, context):
    keys = (
        'mean', 'std', 'min', 'q10', 'q25', 'median',
        'q75', 'q90', 'max', 'negative_fraction',
    )
    if not isinstance(payload, dict) or payload.get('count') != expected_count:
        raise ValueError(f'{context} has an invalid count.')
    values = {key: _finite(payload.get(key), f'{context} {key}') for key in keys}
    ordered = [
        values['min'], values['q10'], values['q25'], values['median'],
        values['q75'], values['q90'], values['max'],
    ]
    if ordered != sorted(ordered) or ordered[0] < -1.000001 or ordered[-1] > 1.000001:
        raise ValueError(f'{context} has invalid cosine quantiles.')
    if not -1.000001 <= values['mean'] <= 1.000001 or values['std'] < 0.0:
        raise ValueError(f'{context} has invalid cosine moments.')
    if not 0.0 <= values['negative_fraction'] <= 1.0:
        raise ValueError(f'{context} has an invalid negative fraction.')
    return values


def _require_distribution_matches(recomputed, recorded, context):
    if not isinstance(recorded, dict) or recorded.get('count') != recomputed['count']:
        raise ValueError(f'{context} count does not reproduce.')
    if recomputed['count'] == 0:
        return
    keys = (
        'mean', 'min', 'q10', 'q25', 'median',
        'q75', 'q90', 'max', 'negative_fraction',
    )
    for key in keys:
        if not math.isclose(
            _finite(recorded.get(key), f'{context} {key}'),
            float(recomputed[key]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f'{context} {key} does not reproduce.')
    if not math.isclose(
        _finite(recorded.get('std'), f'{context} std'),
        float(recomputed['pstdev']),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(f'{context} std does not reproduce.')


def validate_gradient_audit(records):
    if [record.get('epoch') for record in records] != EXPECTED_GRADIENT_EPOCHS:
        raise ValueError('Gradient audit epochs do not match 0..150 step 10.')
    group_values = {}
    global_values = []
    supervised_norms = []
    gcl_norms = []
    for record in records:
        if record.get('batch_index') != 0:
            raise ValueError('Gradient audit must use batch zero.')
        if record.get('joint_gcl_lambda_schedule') != 'constant':
            raise ValueError('PCGrad raw-gradient audit must use constant lambda.')
        if not math.isclose(
            _finite(record.get('joint_gcl_lambda'), 'audit lambda'),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('PCGrad raw-gradient audit lambda must be 0.05.')
        global_stats = record.get('global', {})
        global_values.append(_finite(
            global_stats.get('cosine'),
            'global gradient cosine',
        ))
        supervised_norms.append(_finite(
            global_stats.get('supervised_norm'),
            'global supervised gradient norm',
        ))
        gcl_norms.append(_finite(
            global_stats.get('gcl_norm'),
            'global GCL gradient norm',
        ))
        groups = record.get('groups')
        if not isinstance(groups, dict) or set(groups) != EXPECTED_GRADIENT_GROUPS:
            raise ValueError('Gradient audit record has unexpected groups.')
        for group, stats in groups.items():
            group_values.setdefault(group, [])
            cosine = stats.get('cosine')
            if cosine is not None:
                group_values[group].append(
                    _finite(cosine, f'{group} gradient cosine')
                )
    return {
        'record_count': len(records),
        'global_cosine': summary_stats(global_values),
        'mean_supervised_norm': sum(supervised_norms) / len(supervised_norms),
        'mean_gcl_norm': sum(gcl_norms) / len(gcl_norms),
        'groups': {
            group: summary_stats(values) if values else {'count': 0}
            for group, values in sorted(group_values.items())
        },
    }


def validate_pcgrad_records(records):
    if [record.get('epoch') for record in records] != EXPECTED_PCGRAD_EPOCHS:
        raise ValueError('PCGrad audit epochs do not match exactly 0..159.')
    total_batches = 0
    total_conflicts = 0
    total_cosines = 0
    raw_cosine_sum = 0.0
    standard_norm_sum = 0.0
    projected_norm_sum = 0.0
    total_ratios = 0
    ratio_sum = 0.0
    epoch_conflict_fractions = []
    for record in records:
        epoch = record['epoch']
        if record.get('strategy') != 'pcgrad':
            raise ValueError(f'PCGrad record {epoch} has the wrong strategy.')
        if not math.isclose(
            _finite(record.get('joint_gcl_lambda'), 'PCGrad lambda'),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f'PCGrad lambda mismatch at epoch {epoch}.')
        batch_count = record.get('batch_count')
        conflict_count = record.get('conflict_count')
        if type(batch_count) is not int or batch_count <= 0:
            raise ValueError(f'PCGrad batch count is invalid at epoch {epoch}.')
        if (
            type(conflict_count) is not int
            or conflict_count < 0
            or conflict_count > batch_count
        ):
            raise ValueError(f'PCGrad conflict count is invalid at epoch {epoch}.')
        expected_fraction = conflict_count / batch_count
        if not math.isclose(
            _finite(record.get('conflict_fraction'), 'conflict fraction'),
            expected_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f'PCGrad conflict fraction mismatch at epoch {epoch}.')
        raw_cosine = record.get('raw_cosine')
        cosine_distribution = _validate_cosine_distribution(
            raw_cosine,
            batch_count,
            f'PCGrad epoch {epoch} raw cosine',
        )
        cosine_mean = cosine_distribution['mean']
        if not math.isclose(
            cosine_distribution['negative_fraction'],
            expected_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f'PCGrad conflict/cosine fraction mismatch at epoch {epoch}.'
            )
        standard_mean = _finite(
            record.get('mean_standard_combined_norm'),
            'standard combined norm',
        )
        projected_mean = _finite(
            record.get('mean_projected_combined_norm'),
            'projected combined norm',
        )
        ratio_mean = _finite(
            record.get('mean_projected_to_standard_norm'),
            'projected-to-standard norm ratio',
        )
        ratio_count = record.get('projected_to_standard_norm_count')
        if type(ratio_count) is not int or ratio_count != batch_count:
            raise ValueError(
                f'PCGrad norm-ratio count is invalid at epoch {epoch}.'
            )
        total_batches += batch_count
        total_conflicts += conflict_count
        total_cosines += batch_count
        raw_cosine_sum += cosine_mean * batch_count
        standard_norm_sum += standard_mean * batch_count
        projected_norm_sum += projected_mean * batch_count
        total_ratios += ratio_count
        ratio_sum += ratio_mean * ratio_count
        epoch_conflict_fractions.append(expected_fraction)
    if total_conflicts <= 0:
        raise ValueError('PCGrad never observed or projected a conflicting batch.')
    return {
        'epoch_count': len(records),
        'batch_count': total_batches,
        'conflict_count': total_conflicts,
        'conflict_fraction': total_conflicts / total_batches,
        'raw_cosine_mean': raw_cosine_sum / total_cosines,
        'mean_standard_combined_norm': standard_norm_sum / total_batches,
        'mean_projected_combined_norm': projected_norm_sum / total_batches,
        'projected_to_standard_norm_count': total_ratios,
        'mean_projected_to_standard_norm': ratio_sum / total_ratios,
        'epoch_conflict_fraction': summary_stats(epoch_conflict_fractions),
    }


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _validated_hashed_json(runtime, path_key, sha_key, context):
    path = Path(runtime.get(path_key, ''))
    if not path.is_file() or file_sha256(path) != runtime.get(sha_key):
        raise ValueError(f'{context} path/SHA provenance is invalid.')
    return path, json.loads(path.read_text(encoding='utf-8'))


def build_report(root, fixed_root, static_root, schedule_root):
    root = Path(root).resolve()
    fixed_root = Path(fixed_root).resolve()
    static_root = Path(static_root).resolve()
    schedule_root = Path(schedule_root).resolve()
    manifest_path = root / 'selection_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing selection_manifest.json.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _expect(manifest, {
        'schema_version': 1,
        'protocol': 'strict_inductive',
        'method': METHOD_NAME,
        'seeds': [0],
        'epochs': 160,
        'blind_circuits': [],
        'automatic_multiseed_expansion': False,
        'child_logical_gpu': 0,
    }, 'manifest')
    _expect(manifest.get('gradient_strategy', {}), {
        'name': 'pcgrad',
        'tasks': ['supervised', 'weighted_gcl'],
        'joint_gcl_lambda': 0.05,
        'projection_scope': 'shared_backbone.gnn',
        'scope_parameter_tensors': 10,
        'scope_parameter_values': 17536,
        'scope_parameter_names': EXPECTED_SCOPE_PARAMETER_NAMES,
        'projection': 'symmetric_if_global_dot_negative',
        'random_task_order': False,
        'nonshared_gradients': 'standard_combined_loss',
    }, 'manifest gradient strategy')
    physical_gpu = manifest.get('physical_gpu')
    if physical_gpu not in (3, 4):
        raise ValueError('Manifest physical GPU must be GPU3 or GPU4.')
    _expect(manifest, {
        'child_cuda_visible_devices': str(physical_gpu),
        'comparison_roots': EXPECTED_COMPARISON_ROOTS,
    }, 'manifest execution mapping')
    _expect(manifest.get('gpu_gate', {}), {
        'minimum_free_mb': 6500,
        'utilization': 'record_only',
        'compute_processes': 'record_only',
        'required_consecutive_capacity_samples': 1,
        'poll_seconds': 60,
    }, 'manifest GPU gate')
    _expect(manifest.get('gradient_audit', {}), {
        'enabled': True,
        'interval': 10,
        'batch_index': 0,
        'observational_only': True,
    }, 'manifest gradient audit')
    _expect(manifest.get('trigger_evidence', {}), {
        'linear_lambda_advances_to_seeds12': False,
        'linear_lambda_global_negative_fraction': 0.5,
        'linear_lambda_layer1_negative_fraction': 0.6875,
        'linear_lambda_layer1_mean_cosine': -0.025898593819076714,
    }, 'manifest trigger evidence')
    _expect(manifest.get('selection', {}), {
        'primary': 'source validation raw MSE',
        'transfer_role': 'promotion gates and reporting only',
        'baseline': 'fixed joint rank0 lambda=0.05 seed0',
        'minimum_source_val_improvement_to_expand': 2e-5,
        'promotion_gates': {
            'source_validation_degradation_max': 0.05,
            'transfer_mean_degradation_max': 0.10,
            'any_transfer_circuit_degradation_max': 0.25,
        },
    }, 'manifest selection')

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
    _expect(runtime, {
        'source_graph_names': ['ssram'],
        'transfer_graph_names': list(TRANSFER_NAMES),
        'sgrl_train_graph_names': ['ssram'],
    }, 'candidate runtime')
    hashed_paths = set()
    _verify_hashed_paths(runtime, hashed_paths, 'PCGrad candidate runtime')

    best_epoch = metrics.get('best_epoch')
    if type(best_epoch) is not int or best_epoch < 0 or best_epoch >= 160:
        raise ValueError('Candidate has an invalid best epoch.')
    _expect(metrics, {
        'joint_gcl_lambda': 0.05,
        'joint_gcl_lambda_schedule': 'constant',
        'joint_gcl_lambda_final': None,
        'joint_gcl_lambda_at_best_epoch': 0.05,
        'joint_gcl_lambda_effective_last': 0.05,
        'joint_gradient_strategy': 'pcgrad',
    }, 'candidate metrics')
    candidate_row = _metric_row(metrics)

    gradient_path = Path(runtime.get('joint_gradient_audit_path', ''))
    if (
        not gradient_path.is_file()
        or file_sha256(gradient_path)
        != runtime.get('joint_gradient_audit_sha256')
    ):
        raise ValueError('Gradient audit path/SHA provenance is invalid.')
    gradient_audit = validate_gradient_audit(_load_jsonl(gradient_path))
    gradient_summary_path, gradient_summary = _validated_hashed_json(
        runtime,
        'joint_gradient_summary_path',
        'joint_gradient_summary_sha256',
        'gradient summary',
    )
    if gradient_summary != metrics.get('joint_gradient_audit_summary'):
        raise ValueError('Gradient summary file and metrics do not match.')
    _expect(gradient_summary, {
        'schema_version': 2,
        'record_count': 16,
        'lambda_schedule': {
            'initial': 0.05,
            'schedule': 'constant',
            'final': None,
            'epochs': 160,
        },
    }, 'gradient summary')
    _require_distribution_matches(
        gradient_audit['global_cosine'],
        gradient_summary.get('global_cosine'),
        'global gradient summary',
    )
    if set(gradient_summary.get('groups', {})) != EXPECTED_GRADIENT_GROUPS:
        raise ValueError('Gradient summary has unexpected groups.')
    for group in sorted(EXPECTED_GRADIENT_GROUPS):
        _require_distribution_matches(
            gradient_audit['groups'][group],
            gradient_summary['groups'][group],
            f'{group} gradient summary',
        )
    for key in ('mean_supervised_norm', 'mean_gcl_norm'):
        if not math.isclose(
            _finite(gradient_summary.get(key), f'gradient summary {key}'),
            gradient_audit[key],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f'Gradient summary {key} does not reproduce.')

    pcgrad_path = Path(runtime.get('joint_pcgrad_audit_path', ''))
    if (
        not pcgrad_path.is_file()
        or file_sha256(pcgrad_path) != runtime.get('joint_pcgrad_audit_sha256')
    ):
        raise ValueError('PCGrad audit path/SHA provenance is invalid.')
    pcgrad_audit = validate_pcgrad_records(_load_jsonl(pcgrad_path))
    pcgrad_summary_path, pcgrad_summary = _validated_hashed_json(
        runtime,
        'joint_pcgrad_summary_path',
        'joint_pcgrad_summary_sha256',
        'PCGrad summary',
    )
    if pcgrad_summary != metrics.get('joint_pcgrad_audit_summary'):
        raise ValueError('PCGrad summary file and metrics do not match.')
    _expect(pcgrad_summary, {
        'schema_version': 1,
        'strategy': 'pcgrad',
        'scope': {
            'module': 'shared_backbone.gnn',
            'parameter_tensors': 10,
            'parameter_values': 17536,
            'parameter_names': EXPECTED_SCOPE_PARAMETER_NAMES,
        },
        'epoch_count': 160,
        'batch_count': pcgrad_audit['batch_count'],
        'conflict_count': pcgrad_audit['conflict_count'],
        'projected_to_standard_norm_count': pcgrad_audit[
            'projected_to_standard_norm_count'
        ],
    }, 'PCGrad summary')
    summary_cosine = _validate_cosine_distribution(
        pcgrad_summary.get('raw_cosine'),
        pcgrad_audit['batch_count'],
        'PCGrad overall raw cosine',
    )
    if not math.isclose(
        summary_cosine['negative_fraction'],
        pcgrad_audit['conflict_fraction'],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError('PCGrad overall conflict/cosine fraction mismatch.')
    if not math.isclose(
        _finite(pcgrad_summary.get('conflict_fraction'), 'PCGrad summary fraction'),
        pcgrad_audit['conflict_fraction'],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError('PCGrad summary conflict fraction does not reproduce.')
    summary_numeric_pairs = (
        ('raw_cosine', 'mean', 'raw_cosine_mean'),
        (None, 'mean_standard_combined_norm', 'mean_standard_combined_norm'),
        (None, 'mean_projected_combined_norm', 'mean_projected_combined_norm'),
        (None, 'mean_projected_to_standard_norm', 'mean_projected_to_standard_norm'),
    )
    for nested, summary_key, audit_key in summary_numeric_pairs:
        payload = pcgrad_summary.get(nested, {}) if nested else pcgrad_summary
        if not math.isclose(
            _finite(payload.get(summary_key), f'PCGrad summary {summary_key}'),
            pcgrad_audit[audit_key],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f'PCGrad summary {summary_key} does not reproduce epoch records.'
            )

    fixed_config, _, fixed_row = _control(
        fixed_root,
        'joint_rank0_lambda005_seed0',
        'joint_rank0_lambda005',
    )
    static_config, _, static_row = _control(
        static_root, 'static_dual', 'static_dual'
    )
    fixed_runtime = fixed_config.get('runtime_metadata', {})
    _verify_hashed_paths(fixed_runtime, hashed_paths, 'fixed control runtime')
    fixed_mismatches = [
        field for field in PAIR_FIELDS
        if runtime.get(field) is None
        or runtime.get(field) != fixed_runtime.get(field)
    ]
    if fixed_mismatches:
        raise ValueError(
            f'Candidate/fixed provenance mismatch: {fixed_mismatches}.'
        )
    static_runtime = static_config.get('runtime_metadata', {})
    _verify_hashed_paths(static_runtime, hashed_paths, 'static control runtime')
    static_mismatches = [
        field for field in STATIC_PAIR_FIELDS
        if runtime.get(field) is None
        or runtime.get(field) != static_runtime.get(field)
    ]
    if static_mismatches:
        raise ValueError(
            f'Candidate/static provenance mismatch: {static_mismatches}.'
        )

    schedule_summary_path = schedule_root / 'schedule_seed0_summary.json'
    if not schedule_summary_path.is_file():
        raise ValueError('The completed linear-lambda control summary is missing.')
    comparison_artifact = manifest.get('comparison_artifacts', {}).get(
        'linear_lambda_summary', {}
    )
    if Path(comparison_artifact.get('path', '')).resolve() != schedule_summary_path:
        raise ValueError('Manifest linear-lambda summary path is invalid.')
    if comparison_artifact.get('sha256') != file_sha256(schedule_summary_path):
        raise ValueError('Manifest linear-lambda summary SHA is invalid.')
    schedule_summary = json.loads(
        schedule_summary_path.read_text(encoding='utf-8')
    )
    if schedule_summary.get('method') != 'joint_rank0_lambda005_to0005_linear_seed0':
        raise ValueError('Unexpected linear-lambda comparison summary.')
    if schedule_summary.get('promotion_gate', {}).get('passes') is not True:
        raise ValueError('Linear-lambda prerequisite did not pass promotion gates.')
    if schedule_summary.get('selection', {}).get('advances_to_seeds12') is not False:
        raise ValueError('Linear-lambda prerequisite must have failed expansion.')
    schedule_improvement = _finite(
        schedule_summary.get('selection', {}).get(
            'source_val_improvement_over_fixed'
        ),
        'linear-lambda source validation improvement',
    )
    if schedule_improvement > 2e-5:
        raise ValueError('Linear-lambda prerequisite unexpectedly qualified.')
    schedule_row = _validated_metric_row(
        schedule_summary.get('candidate'), 'linear-lambda candidate'
    )
    schedule_fixed = _validated_metric_row(
        schedule_summary.get('fixed_lambda005'),
        'linear-lambda fixed control',
    )
    schedule_static = _validated_metric_row(
        schedule_summary.get('static_dual'),
        'linear-lambda static control',
    )
    _require_matching_rows(
        schedule_fixed, fixed_row, 'linear-lambda/fixed control'
    )
    _require_matching_rows(
        schedule_static, static_row, 'linear-lambda/static control'
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
        'linear_lambda': schedule_row,
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
        'gradient_audit': gradient_audit,
        'pcgrad_audit': pcgrad_audit,
        'sources': {
            'candidate_config': str(config_path),
            'candidate_metrics': str(config_path.with_name('metrics.json')),
            'gradient_summary': str(gradient_summary_path),
            'pcgrad_summary': str(pcgrad_summary_path),
            'fixed_root': str(fixed_root),
            'static_root': str(static_root),
            'linear_lambda_summary': str(schedule_summary_path),
            'verified_hashed_provenance_files': len(hashed_paths),
        },
    }


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / 'pcgrad_seed0_summary.json'
    tsv_path = output_dir / 'pcgrad_seed0_summary.tsv'
    atomic_write_text(
        json_path, json.dumps(report, indent=2, sort_keys=True) + '\n'
    )
    buffer = io.StringIO()
    columns = [
        'method', 'val_mse', 'digtime', 'timing_ctrl', 'array',
        'transfer_mean', 'promotion_gate_passes', 'advances_to_seeds12',
        'pcgrad_conflict_fraction',
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter='\t')
    writer.writeheader()
    writer.writerow({
        'method': METHOD_NAME,
        **report['candidate'],
        'promotion_gate_passes': report['promotion_gate']['passes'],
        'advances_to_seeds12': report['selection']['advances_to_seeds12'],
        'pcgrad_conflict_fraction': report['pcgrad_audit']['conflict_fraction'],
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
    parser.add_argument(
        '--schedule_root', type=Path,
        default=Path(
            'logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712'
        ),
    )
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        args.root,
        args.fixed_root,
        args.static_root,
        args.schedule_root,
    )
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
