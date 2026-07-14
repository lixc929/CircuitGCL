#!/usr/bin/env python3
"""Run the preregistered compact shared-GraphSAGE seed-0 screen.

The lambda=0 arm is the only process allowed to create the source-only SAGE
SGRL checkpoint and metadata.  The lambda=.05 arm starts only after that
producer has completed successfully, so the second arm is a cache reader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_strict_joint_lambda_schedule_seed0 import (
    append_status,
    atomic_write_json,
    build_child_environment,
    current_git_state,
    wait_for_gpu_capacity,
)
from summarize_joint_sage_seed0 import (
    EXPECTED_SAGE_CACHE_KEY,
    validate_control_artifacts,
)


SCREEN_NAME = 'compact_shared_sage_seed0_screen'
LAMBDA0_NAME = 'joint_sage_rank0_lambda0_seed0'
LAMBDA005_NAME = 'joint_sage_rank0_lambda005_seed0'
RUN_NAMES = (LAMBDA0_NAME, LAMBDA005_NAME)
DEFAULT_LOG_ROOT = Path(
    'logs/strict_joint_sage_rank0_seed0_screen_20260714'
)
DEFAULT_STATIC_ROOT = Path('logs/strict_seed0_seven_20260712_v2')
DEFAULT_FIXED_ROOT = Path('logs/strict_joint_rank0_lambda005_seed0_20260712')
ALLOWED_PHYSICAL_GPUS = frozenset({3, 4})


# Every protocol-relevant value is explicit.  The two arms differ only in
# lambda; representation auditing remains enabled for both arms.
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
    '--e1_lr', '0.000001',
    '--e2_lr', '0.0000002',
    '--sgrl_online_lr', '0.000001',
    '--cl_model', 'sage',
    '--cl_act_fn', 'tanh',
    '--cl_epochs', '5',
    '--cl_gnn_layers', '2',
    '--cl_hid_dim', '64',
    '--cl_batch_size', '32768',
    '--cl_num_neighbors', '8',
    '--cl_dropout', '0.3',
    '--model', 'sage',
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
    '--sgrl_pretrain_target_update', 'sgrl_dual_rsm_ema',
    '--joint_shared_gnn_layers', '2',
    '--joint_lora_rank', '0',
    '--joint_lora_layer', '-1',
    '--joint_gcl_lambda_schedule', 'constant',
    '--joint_gradient_strategy', 'none',
    '--joint_shared_audit', '1',
    '--joint_shared_audit_interval', '10',
    '--joint_gradient_audit_interval', '10',
    '--seed', '0',
    '--pretraining_seed', '0',
    '--embedding_inference_seed', '20260711',
    '--downstream_seed', '0',
    '--train_sampler_seed', '0',
    '--relation_sample_seed', '20260711',
    '--split_seed', '0',
    '--eval_seed', '20260711',
]

METHOD_ARGS = {
    LAMBDA0_NAME: [
        '--joint_gcl_lambda', '0.0',
        '--joint_gradient_audit', '0',
    ],
    LAMBDA005_NAME: [
        '--joint_gcl_lambda', '0.05',
        '--joint_gradient_audit', '0',
    ],
}


def validate_physical_gpu(gpu):
    if type(gpu) is not int or gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError('Formal SAGE screening permits only physical GPU3/GPU4.')


def expected_sage_checkpoint_paths(repo_root):
    online = (
        Path(repo_root) / 'pkl' / 'pkl_online'
        / f'best_online_{EXPECTED_SAGE_CACHE_KEY}.pkl'
    )
    return {
        'online': online,
        'metadata': Path(f'{online}.metadata.json'),
        'target': (
            Path(repo_root) / 'pkl' / 'pkl_target'
            / f'best_target_{EXPECTED_SAGE_CACHE_KEY}.pkl'
        ),
    }


def validate_no_preexisting_sage_checkpoint(repo_root):
    paths = expected_sage_checkpoint_paths(repo_root)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            'Formal SAGE screen requires a new checkpoint produced by the '
            f'lambda=0 arm; pre-existing files: {existing}'
        )
    return {name: str(path) for name, path in paths.items()}


def _resolved_root(repo_root, path):
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _log_dir(repo_root, log_root, run_name):
    try:
        root = log_root.relative_to(repo_root)
    except ValueError:
        root = log_root
    return root / run_name


def build_command(repo_root, log_root, python_bin, run_name):
    if run_name not in METHOD_ARGS:
        raise ValueError(f'Unknown SAGE screen method: {run_name}')
    return [
        python_bin,
        'main.py',
        *COMMON_ARGS,
        *METHOD_ARGS[run_name],
        '--gpu', '0',
        '--log_dir', str(_log_dir(repo_root, log_root, run_name)),
    ]


def write_manifest(
        path, commit, commands, physical_gpus, gpu_gate, control_artifacts,
        checkpoint_paths):
    atomic_write_json(path, {
        'schema_version': 1,
        'git_commit': commit,
        'protocol': 'strict_inductive',
        'method': SCREEN_NAME,
        'methods': list(RUN_NAMES),
        'seed': 0,
        'epochs': 160,
        'blind_circuits': [],
        'architecture': {
            'online_target_and_deployment_backbone': 'sage',
            'layers': 2,
            'hidden_dim': 64,
            'rank': 0,
            'deployment_gnns': 1,
        },
        'execution': {
            'ordering': list(RUN_NAMES),
            'checkpoint_producer': LAMBDA0_NAME,
            'checkpoint_readers': [LAMBDA005_NAME],
            'concurrent_checkpoint_writers': 1,
            'checkpoint_paths_absent_before_launch': checkpoint_paths,
            'physical_gpus': physical_gpus,
            'child_logical_gpu': 0,
            'gpu_gate': gpu_gate,
        },
        'selection': {
            'candidates': list(RUN_NAMES),
            'primary_metric': 'source validation raw MSE',
            'source_val_tie_tolerance_prefer_lambda005': 2e-5,
            'minimum_source_val_improvement_over_cluster_lambda005': 2e-5,
            'promotion_gates_vs_canonical_static': {
                'source_validation_degradation_max': 0.05,
                'transfer_mean_degradation_max': 0.10,
                'any_transfer_circuit_degradation_max': 0.25,
            },
            'registered_references': {
                'canonical_static': {
                    'val_mse': 0.007809748407453299,
                    'digtime': 0.016486799344420433,
                    'timing_ctrl': 0.010408373549580574,
                    'array': 0.01004203874617815,
                    'transfer_mean': 0.012312403880059719,
                },
                'cluster_rank0_lambda005_val_mse': 0.008035913109779358,
            },
            'registered_upper_bounds': {
                'val_mse': 0.008200235827825964,
                'digtime': 0.02060849918052554,
                'timing_ctrl': 0.013010466936975718,
                'array': 0.012552548432722688,
                'transfer_mean': 0.013543644268065692,
            },
            'required_val_mse_strictly_below': 0.008015913109779358,
            'automatic_multiseed_expansion': False,
        },
        'comparison_roots': {
            'canonical_static_and_cluster_lambda0': (
                'logs/strict_seed0_seven_20260712_v2'
            ),
            'cluster_lambda005': (
                'logs/strict_joint_rank0_lambda005_seed0_20260712'
            ),
        },
        'control_artifacts': control_artifacts,
        'commands': commands,
    })


def _run_gpu_method(
        run_name, physical_gpu, command, environment, status_path,
        expected_commit, repo_root, min_free_mb, poll_seconds,
        required_capacity_samples):
    free_mb, utilization = wait_for_gpu_capacity(
        physical_gpu,
        status_path,
        expected_commit,
        repo_root,
        min_free_mb,
        poll_seconds,
        required_capacity_samples,
    )
    append_status(
        status_path,
        run_name,
        f'start physical_gpu={physical_gpu} logical_gpu=0 '
        f'free_mb={free_mb} util={utilization}',
    )
    result = subprocess.run(command, cwd=repo_root, env=environment)
    append_status(status_path, run_name, f'finish rc={result.returncode}')
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu_lambda0', type=int, default=3)
    parser.add_argument('--gpu_lambda005', type=int, default=4)
    parser.add_argument(
        '--python', default='/home/lixc/.conda/envs/RCG/bin/python'
    )
    parser.add_argument('--log_root', type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument('--static_root', type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument('--fixed_root', type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument('--min_free_mb', type=int, default=6500)
    parser.add_argument('--poll_seconds', type=int, default=60)
    parser.add_argument('--required_capacity_samples', type=int, default=1)
    args = parser.parse_args(argv)

    for gpu in (args.gpu_lambda0, args.gpu_lambda005):
        try:
            validate_physical_gpu(gpu)
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
            'Formal SAGE execution locks --min_free_mb 6500, '
            '--poll_seconds 60, and --required_capacity_samples 1.'
        )

    repo_root = Path(__file__).resolve().parents[1]
    log_root = _resolved_root(repo_root, args.log_root)
    static_root = _resolved_root(repo_root, args.static_root)
    fixed_root = _resolved_root(repo_root, args.fixed_root)
    commands = {
        name: build_command(repo_root, log_root, args.python, name)
        for name in RUN_NAMES
    }
    physical_gpus = {
        LAMBDA0_NAME: args.gpu_lambda0,
        LAMBDA005_NAME: args.gpu_lambda005,
    }
    environments = {
        name: build_child_environment(gpu)
        for name, gpu in physical_gpus.items()
    }

    if not args.execute:
        print(json.dumps({
            'mode': 'dry_run',
            'writes_performed': False,
            'screen': SCREEN_NAME,
            'log_root': str(log_root),
            'static_root': str(static_root),
            'fixed_root': str(fixed_root),
            'ordering': list(RUN_NAMES),
            'physical_gpus': physical_gpus,
            'child_logical_gpu': 0,
            'commands': commands,
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
        checkpoint_paths = validate_no_preexisting_sage_checkpoint(repo_root)
    except ValueError as error:
        parser.error(str(error))
    try:
        control_artifacts = validate_control_artifacts(
            repo_root, static_root, fixed_root
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f'Control-artifact validation failed: {error}')

    log_root.mkdir(parents=True)
    status_path = log_root / 'queue_status.tsv'
    gpu_gate = {
        'minimum_free_mb': args.min_free_mb,
        'utilization': 'record_only',
        'compute_processes': 'record_only',
        'required_consecutive_capacity_samples': args.required_capacity_samples,
        'poll_seconds': args.poll_seconds,
    }
    write_manifest(
        log_root / 'selection_manifest.json',
        commit,
        commands,
        physical_gpus,
        gpu_gate,
        control_artifacts,
        checkpoint_paths,
    )
    append_status(
        status_path,
        'orchestrator',
        f'queue_started commit={commit} screen={SCREEN_NAME}',
    )

    queue_rc = 1
    try:
        append_status(status_path, 'processed_cache_prewarm', 'start')
        with (log_root / 'processed_cache_prewarm.log').open(
                'w', encoding='utf-8') as output:
            prewarm = subprocess.run([
                args.python,
                'scripts/prewarm_strict_processed_caches.py',
                '--relation_sample_seed', '20260711',
            ], cwd=repo_root, env=environments[LAMBDA0_NAME],
                stdout=output, stderr=subprocess.STDOUT)
        append_status(
            status_path,
            'processed_cache_prewarm',
            f'finish rc={prewarm.returncode}',
        )
        if prewarm.returncode != 0:
            queue_rc = prewarm.returncode
            for name in RUN_NAMES:
                append_status(
                    status_path, name,
                    'not_started reason=processed_cache_prewarm_failure',
                )
            append_status(
                status_path, 'summary',
                'not_started reason=processed_cache_prewarm_failure',
            )
            return queue_rc

        producer_rc = _run_gpu_method(
            LAMBDA0_NAME,
            physical_gpus[LAMBDA0_NAME],
            commands[LAMBDA0_NAME],
            environments[LAMBDA0_NAME],
            status_path,
            commit,
            repo_root,
            args.min_free_mb,
            args.poll_seconds,
            args.required_capacity_samples,
        )
        if producer_rc != 0:
            queue_rc = producer_rc
            append_status(
                status_path,
                LAMBDA005_NAME,
                'not_started reason=checkpoint_producer_failure',
            )
            append_status(
                status_path, 'summary',
                'not_started reason=checkpoint_producer_failure',
            )
            return queue_rc

        post_producer_commit, post_producer_status = current_git_state(repo_root)
        if post_producer_commit != commit or post_producer_status:
            raise RuntimeError(
                'Repository changed after the SAGE checkpoint producer: '
                f'commit={post_producer_commit} '
                f'status={post_producer_status!r}'
            )
        append_status(
            status_path,
            'checkpoint_handoff',
            f'producer_completed name={LAMBDA0_NAME} reader={LAMBDA005_NAME}',
        )

        reader_rc = _run_gpu_method(
            LAMBDA005_NAME,
            physical_gpus[LAMBDA005_NAME],
            commands[LAMBDA005_NAME],
            environments[LAMBDA005_NAME],
            status_path,
            commit,
            repo_root,
            args.min_free_mb,
            args.poll_seconds,
            args.required_capacity_samples,
        )
        if reader_rc != 0:
            queue_rc = reader_rc
            append_status(
                status_path, 'summary',
                'not_started reason=method_failure '
                f'method={LAMBDA005_NAME}',
            )
            return queue_rc

        post_commit, post_tracked_status = current_git_state(repo_root)
        if post_commit != commit or post_tracked_status:
            raise RuntimeError(
                'Repository changed during the formal SAGE screen: '
                f'commit={post_commit} status={post_tracked_status!r}'
            )

        append_status(status_path, 'summary', 'start')
        summary = subprocess.run([
            args.python,
            'scripts/summarize_joint_sage_seed0.py',
            str(log_root),
            '--static_root', str(static_root),
            '--fixed_root', str(fixed_root),
            '--output_dir', str(log_root),
        ], cwd=repo_root, env=environments[LAMBDA005_NAME])
        queue_rc = summary.returncode
        append_status(status_path, 'summary', f'finish rc={queue_rc}')
        return queue_rc
    except Exception as error:
        append_status(
            status_path,
            'orchestrator',
            f'exception type={type(error).__name__}',
        )
        raise
    finally:
        append_status(
            status_path, 'orchestrator', f'queue_finished rc={queue_rc}'
        )


if __name__ == '__main__':
    sys.exit(main())
