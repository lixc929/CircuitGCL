#!/usr/bin/env python3
"""Queue the preregistered strict rank-0 linear-lambda seed-0 experiment."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


METHOD_NAME = 'joint_rank0_lambda005_to0005_linear_seed0'
DEFAULT_LOG_ROOT = Path(
    'logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712'
)
FORBIDDEN_PHYSICAL_GPUS = frozenset({0, 1})
CPU_THREAD_ENV = {
    'OPENBLAS_NUM_THREADS': '4',
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'NUMEXPR_NUM_THREADS': '4',
}
COMMON_ARGS = [
    '--task_level', 'edge',
    '--task', 'regression',
    '--dataset', 'ssram+digtime+timing_ctrl+array_128_32_8t',
    '--neg_edge_ratio', '0.0',
    '--net_only', 'True',
    '--protocol', 'strict_inductive',
    '--sgrl_graph_scope', 'source',
    '--normalization_scope', 'source',
    '--small_dataset_sample_rates', '1.0',
    '--large_dataset_sample_rates', '0.1',
    '--num_hops', '2',
    '--num_neighbors', '8',
    '--num_workers', '0',
    '--epochs', '160',
    '--early_stopping_patience', '0',
    '--early_stopping_min_delta', '0.0',
    '--batch_size', '512',
    '--lr', '0.0001',
    '--momentum', '0.99',
    '--weight_decay', '0.0',
    '--cl_model', 'clustergcn',
    '--cl_act_fn', 'tanh',
    '--cl_epochs', '5',
    '--cl_gnn_layers', '2',
    '--cl_hid_dim', '64',
    '--cl_batch_size', '32768',
    '--cl_num_neighbors', '8',
    '--cl_dropout', '0.3',
    '--model', 'clustergcn',
    '--num_gnn_layers', '2',
    '--num_head_layers', '2',
    '--hid_dim', '64',
    '--dropout', '0.1',
    '--use_bn', '0',
    '--act_fn', 'prelu',
    '--use_stats', '1',
    '--src_dst_agg', 'concat',
    '--num_classes', '5',
    '--noise_sigma', '0.001',
    '--regress_loss', 'mse',
    '--sgrl', '1',
    '--sgrl_mode', 'joint_shared',
    '--sgrl_reuse_stats', '1',
    '--sgrl_reuse_stats_fusion', 'concat',
    '--joint_shared_gnn_layers', '2',
    '--joint_lora_rank', '0',
    '--joint_lora_layer', '-1',
    '--joint_gcl_lambda', '0.05',
    '--joint_gcl_lambda_schedule', 'linear',
    '--joint_gcl_lambda_final', '0.005',
    '--joint_shared_audit', '1',
    '--joint_shared_audit_interval', '10',
    '--joint_gradient_audit', '1',
    '--joint_gradient_audit_interval', '10',
    '--sgrl_pretrain_target_update', 'sgrl_dual_rsm_ema',
    '--seed', '0',
    '--pretraining_seed', '0',
    '--embedding_inference_seed', '20260711',
    '--downstream_seed', '0',
    '--train_sampler_seed', '0',
    '--relation_sample_seed', '20260711',
    '--split_seed', '0',
    '--eval_seed', '20260711',
]


def validate_physical_gpu(gpu):
    if gpu < 0 or gpu in FORBIDDEN_PHYSICAL_GPUS:
        raise ValueError('Physical GPU0 and GPU1 are forbidden; choose GPU2 or higher.')


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


def build_child_environment(physical_gpu, base_environment=None):
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(CPU_THREAD_ENV)
    environment['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
    return environment


def gpu_snapshot(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    free_mb, utilization = (int(value.strip()) for value in output.split(','))
    process_output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    processes = [
        line.strip()
        for line in process_output.splitlines()
        if line.strip() and 'No running processes found' not in line
    ]
    return free_mb, utilization, processes


def append_status(path, stage, status):
    timestamp = datetime.datetime.now().astimezone().isoformat()
    with path.open('a', encoding='utf-8') as output:
        output.write(f'{timestamp}\t{stage}\t{status}\n')


def current_git_state(repo_root):
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=repo_root,
        text=True,
    ).strip()
    return commit, tracked_status


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
        # The operator explicitly permits sharing a busy GPU. Utilization and
        # process count are recorded for provenance but only free memory gates
        # launch; no existing process is stopped or modified.
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


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent, delete=False
    ) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write('\n')
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def write_manifest(path, commit, command, physical_gpu, gate):
    atomic_write_json(path, {
        'schema_version': 1,
        'git_commit': commit,
        'protocol': 'strict_inductive',
        'method': METHOD_NAME,
        'seeds': [0],
        'epochs': 160,
        'blind_circuits': [],
        'automatic_multiseed_expansion': False,
        'schedule': {
            'name': 'linear',
            'initial': 0.05,
            'final': 0.005,
            'formula': '0.05 + (0.005 - 0.05) * epoch / 159',
        },
        'gradient_audit': {
            'enabled': True,
            'interval': 10,
            'batch_index': 0,
            'observational_only': True,
        },
        'selection': {
            'primary': 'source validation raw MSE',
            'transfer_role': 'promotion gates and reporting only',
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
        ],
        'physical_gpu': physical_gpu,
        'child_cuda_visible_devices': str(physical_gpu),
        'child_logical_gpu': 0,
        'gpu_gate': gate,
        'command': command,
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, default=3)
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
            'scripts/summarize_joint_lambda_schedule_seed0.py',
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
