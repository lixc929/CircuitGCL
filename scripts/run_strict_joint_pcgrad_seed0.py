#!/usr/bin/env python3
"""Queue the preregistered strict rank-0 PCGrad seed-0 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from summarize_joint_lambda_schedule_seed0 import (
    build_report as build_linear_schedule_report,
)
from summarize_strict_reuse import file_sha256
from run_strict_joint_lambda_schedule_seed0 import (
    COMMON_ARGS as LINEAR_SCHEDULE_ARGS,
    append_status,
    atomic_write_json,
    build_child_environment,
    current_git_state,
    gpu_snapshot,
)


METHOD_NAME = 'joint_rank0_lambda005_pcgrad_seed0'
DEFAULT_LOG_ROOT = Path(
    'logs/strict_joint_rank0_lambda005_pcgrad_seed0_audit_20260713'
)
ALLOWED_PHYSICAL_GPUS = frozenset({3, 4})
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


def _pcgrad_args():
    args = []
    excluded = {
        '--joint_gcl_lambda_schedule',
        '--joint_gcl_lambda_final',
    }
    for index in range(0, len(LINEAR_SCHEDULE_ARGS), 2):
        key = LINEAR_SCHEDULE_ARGS[index]
        value = LINEAR_SCHEDULE_ARGS[index + 1]
        if key not in excluded:
            args.extend([key, value])
    args.extend([
        '--joint_gcl_lambda_schedule', 'constant',
        '--joint_gradient_strategy', 'pcgrad',
    ])
    return args


COMMON_ARGS = _pcgrad_args()


def validate_physical_gpu(gpu):
    if gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError('Formal PCGrad runs permit only physical GPU3 or GPU4.')


def build_command(repo_root, log_root, python_bin):
    try:
        log_dir = log_root.relative_to(repo_root)
    except ValueError:
        log_dir = log_root
    return [
        python_bin,
        'main.py',
        *COMMON_ARGS,
        '--gpu', '0',
        '--log_dir', str(log_dir / METHOD_NAME),
    ]


def wait_for_gpu_capacity(
        gpu, status_path, expected_commit, repo_root, min_free_mb,
        poll_seconds, required_capacity_samples):
    ready_samples = 0
    while ready_samples < required_capacity_samples:
        commit, tracked_status = current_git_state(repo_root)
        if commit != expected_commit or tracked_status:
            raise RuntimeError(
                'Repository changed while the formal run was queued: '
                f'commit={commit} tracked_status={tracked_status!r}'
            )
        free_mb, utilization, processes = gpu_snapshot(gpu)
        ready = free_mb >= min_free_mb
        ready_samples = ready_samples + 1 if ready else 0
        append_status(
            status_path,
            'gpu_gate',
            f'waiting gpu={gpu} free_mb={free_mb} util={utilization} '
            f'processes={len(processes)} ready_samples={ready_samples}/'
            f'{required_capacity_samples}',
        )
        if ready_samples < required_capacity_samples:
            time.sleep(poll_seconds)
    return free_mb, utilization


def write_manifest(
        path, commit, command, physical_gpu, gate,
        schedule_summary_path):
    atomic_write_json(path, {
        'schema_version': 1,
        'git_commit': commit,
        'protocol': 'strict_inductive',
        'method': METHOD_NAME,
        'seeds': [0],
        'epochs': 160,
        'blind_circuits': [],
        'automatic_multiseed_expansion': False,
        'gradient_strategy': {
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
        },
        'gradient_audit': {
            'enabled': True,
            'interval': 10,
            'batch_index': 0,
            'observational_only': True,
        },
        'trigger_evidence': {
            'linear_lambda_advances_to_seeds12': False,
            'linear_lambda_global_negative_fraction': 0.5,
            'linear_lambda_layer1_negative_fraction': 0.6875,
            'linear_lambda_layer1_mean_cosine': -0.025898593819076714,
        },
        'selection': {
            'primary': 'source validation raw MSE',
            'transfer_role': 'promotion gates and reporting only',
            'baseline': 'fixed joint rank0 lambda=0.05 seed0',
            'minimum_source_val_improvement_to_expand': 2e-5,
            'promotion_gates': {
                'source_validation_degradation_max': 0.05,
                'transfer_mean_degradation_max': 0.10,
                'any_transfer_circuit_degradation_max': 0.25,
            },
        },
        'comparison_roots': [
            'logs/strict_joint_rank0_lambda005_seed0_20260712',
            'logs/strict_seed0_seven_20260712_v2',
            'logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712',
        ],
        'comparison_artifacts': {
            'linear_lambda_summary': {
                'path': str(schedule_summary_path.resolve()),
                'sha256': file_sha256(schedule_summary_path),
            },
        },
        'physical_gpu': physical_gpu,
        'child_cuda_visible_devices': str(physical_gpu),
        'child_logical_gpu': 0,
        'gpu_gate': gate,
        'command': command,
    })


def validate_linear_schedule_prerequisite(repo_root):
    schedule_root = repo_root / (
        'logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712'
    )
    summary_path = schedule_root / 'schedule_seed0_summary.json'
    if not summary_path.is_file():
        raise ValueError(
            'The validated linear-lambda seed0 prerequisite summary is missing: '
            f'{summary_path}'
        )
    fixed_root = repo_root / 'logs/strict_joint_rank0_lambda005_seed0_20260712'
    static_root = repo_root / 'logs/strict_seed0_seven_20260712_v2'
    recomputed = build_linear_schedule_report(
        schedule_root,
        fixed_root,
        static_root,
    )
    recorded = json.loads(summary_path.read_text(encoding='utf-8'))
    if recorded != recomputed:
        raise ValueError(
            'The linear-lambda prerequisite summary does not reproduce its '
            'validated artifacts.'
        )
    if not recomputed.get('promotion_gate', {}).get('passes'):
        raise ValueError('The linear-lambda prerequisite failed promotion gates.')
    selection = recomputed.get('selection', {})
    if selection.get('advances_to_seeds12') is not False:
        raise ValueError(
            'PCGrad is permitted only after linear lambda fails expansion.'
        )
    improvement = selection.get('source_val_improvement_over_fixed')
    if improvement is None or float(improvement) > 2e-5:
        raise ValueError('The linear-lambda prerequisite unexpectedly qualified.')
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument(
        '--python', default='/home/lixc/.conda/envs/RCG/bin/python'
    )
    parser.add_argument('--log_root', type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument('--min_free_mb', type=int, default=6500)
    parser.add_argument('--poll_seconds', type=int, default=60)
    parser.add_argument('--required_capacity_samples', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        validate_physical_gpu(args.gpu)
    except ValueError as error:
        parser.error(str(error))
    if args.min_free_mb <= 0:
        parser.error('GPU free-memory threshold must be positive.')
    if args.poll_seconds <= 0 or args.required_capacity_samples <= 0:
        parser.error('Polling controls must be positive.')
    if args.execute and (
        args.min_free_mb != 6500
        or args.poll_seconds != 60
        or args.required_capacity_samples != 1
    ):
        parser.error(
            'Formal PCGrad execution locks --min_free_mb 6500, '
            '--poll_seconds 60, and --required_capacity_samples 1.'
        )

    repo_root = Path(__file__).resolve().parents[1]
    log_root = (
        args.log_root.resolve()
        if args.log_root.is_absolute()
        else (repo_root / args.log_root).resolve()
    )
    command = build_command(repo_root, log_root, args.python)
    environment = build_child_environment(args.gpu)
    if not args.execute:
        print(json.dumps({
            'mode': 'dry_run',
            'writes_performed': False,
            'physical_gpu': args.gpu,
            'child_cuda_visible_devices': str(args.gpu),
            'child_logical_gpu': 0,
            'log_root': str(log_root),
            'command': command,
        }, indent=2, sort_keys=True))
        return 0

    if log_root.exists():
        parser.error(f'Refusing to overwrite existing log root: {log_root}')
    commit, tracked_status = current_git_state(repo_root)
    if tracked_status:
        parser.error(
            'Refusing formal execution with tracked worktree changes: '
            f'{tracked_status}'
        )
    try:
        schedule_summary_path = validate_linear_schedule_prerequisite(repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    log_root.mkdir(parents=True)
    status_path = log_root / 'queue_status.tsv'
    gate = {
        'minimum_free_mb': args.min_free_mb,
        'utilization': 'record_only',
        'compute_processes': 'record_only',
        'required_consecutive_capacity_samples': (
            args.required_capacity_samples
        ),
        'poll_seconds': args.poll_seconds,
    }
    write_manifest(
        log_root / 'selection_manifest.json',
        commit,
        command,
        args.gpu,
        gate,
        schedule_summary_path,
    )
    append_status(
        status_path, 'orchestrator',
        f'queue_started commit={commit} physical_gpu={args.gpu}',
    )
    queue_rc = 1
    try:
        free_mb, utilization = wait_for_gpu_capacity(
            args.gpu,
            status_path,
            commit,
            repo_root,
            args.min_free_mb,
            args.poll_seconds,
            args.required_capacity_samples,
        )
        append_status(
            status_path, METHOD_NAME,
            f'start physical_gpu={args.gpu} logical_gpu=0 '
            f'free_mb={free_mb} util={utilization}',
        )
        result = subprocess.run(command, cwd=repo_root, env=environment)
        append_status(
            status_path, METHOD_NAME, f'finish rc={result.returncode}'
        )
        if result.returncode != 0:
            queue_rc = result.returncode
            append_status(
                status_path, 'summary', 'not_started reason=method_failure'
            )
            return queue_rc

        post_commit, post_tracked_status = current_git_state(repo_root)
        if post_commit != commit or post_tracked_status:
            raise RuntimeError(
                'Repository changed during the formal run; refusing to run '
                'the post-run validator from a different worktree state: '
                f'commit={post_commit} '
                f'tracked_status={post_tracked_status!r}'
            )
        append_status(status_path, 'summary', 'start')
        summary = subprocess.run([
            args.python,
            'scripts/summarize_joint_pcgrad_seed0.py',
            str(log_root),
            '--output_dir', str(log_root),
        ], cwd=repo_root, env=environment)
        queue_rc = summary.returncode
        append_status(status_path, 'summary', f'finish rc={queue_rc}')
        return queue_rc
    except Exception as error:
        append_status(
            status_path, 'orchestrator',
            f'exception type={type(error).__name__}',
        )
        raise
    finally:
        append_status(
            status_path, 'orchestrator', f'queue_finished rc={queue_rc}'
        )


if __name__ == '__main__':
    sys.exit(main())
