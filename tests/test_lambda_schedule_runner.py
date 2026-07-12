import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_strict_joint_lambda_schedule_seed0 import (
    COMMON_ARGS,
    METHOD_NAME,
    build_child_environment,
    main as runner_main,
    validate_physical_gpu,
    wait_for_gpu_capacity,
)


class LambdaScheduleRunnerTest(unittest.TestCase):
    def test_dry_run_is_exact_nonblind_schedule(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner_main(['--gpu', '3']), 0)
        payload = json.loads(output.getvalue())
        command = payload['command']
        self.assertFalse(payload['writes_performed'])
        self.assertEqual(payload['child_logical_gpu'], 0)
        self.assertEqual(command[command.index('--gpu') + 1], '0')
        self.assertEqual(
            command[command.index('--joint_gcl_lambda_schedule') + 1],
            'linear',
        )
        self.assertEqual(
            command[command.index('--joint_gcl_lambda_final') + 1],
            '0.005',
        )
        self.assertNotIn('sp8192w', ' '.join(command).lower())
        self.assertEqual(len(COMMON_ARGS) % 2, 0)

    def test_gpu_validation_and_child_isolation(self):
        for gpu in (-1, 0, 1):
            with self.subTest(gpu=gpu):
                with self.assertRaises(ValueError):
                    validate_physical_gpu(gpu)
        validate_physical_gpu(2)
        validate_physical_gpu(3)
        environment = build_child_environment(3, {'CUDA_VISIBLE_DEVICES': 'all'})
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '3')

    def test_capacity_gate_records_but_allows_busy_gpu(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            status_path = Path(temporary_dir) / 'queue.tsv'
            with (
                patch(
                    'run_strict_joint_lambda_schedule_seed0.current_git_state',
                    return_value=('abc', ''),
                ),
                patch(
                    'run_strict_joint_lambda_schedule_seed0.gpu_snapshot',
                    side_effect=[
                        (6000, 100, ['foreign']),
                        (7000, 100, ['foreign']),
                    ],
                ),
                patch('run_strict_joint_lambda_schedule_seed0.time.sleep'),
            ):
                free_mb, utilization = wait_for_gpu_capacity(
                    3, status_path, 'abc', Path(temporary_dir),
                    6500, 1, 1,
                )
            self.assertEqual((free_mb, utilization), (7000, 100))
            status = status_path.read_text()
            self.assertIn('processes=1', status)
            self.assertIn('free_mb=6000', status)
            self.assertIn('free_mb=7000', status)

    def test_capacity_gate_rejects_commit_change(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch(
                'run_strict_joint_lambda_schedule_seed0.current_git_state',
                return_value=('changed', ''),
            ):
                with self.assertRaisesRegex(RuntimeError, 'Repository changed'):
                    wait_for_gpu_capacity(
                        3, Path(temporary_dir) / 'queue.tsv', 'expected',
                        Path(temporary_dir), 6500, 1, 1,
                    )

    def test_execute_propagates_summary_rc_and_records_queue(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            dir=repo_root / 'logs'
        ) as temporary_dir:
            log_root = Path(temporary_dir) / 'formal'
            results = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)]
            with (
                patch(
                    'run_strict_joint_lambda_schedule_seed0.current_git_state',
                    return_value=('abc', ''),
                ),
                patch(
                    'run_strict_joint_lambda_schedule_seed0.wait_for_gpu_capacity',
                    return_value=(39000, 0),
                ),
                patch(
                    'run_strict_joint_lambda_schedule_seed0.subprocess.run',
                    side_effect=results,
                ) as run,
            ):
                rc = runner_main([
                    '--execute', '--gpu', '3',
                    '--log_root', str(log_root),
                ])
            self.assertEqual(rc, 7)
            self.assertEqual(run.call_count, 2)
            status = (log_root / 'queue_status.tsv').read_text()
            self.assertIn(f'\t{METHOD_NAME}\tstart ', status)
            self.assertIn('\tsummary\tfinish rc=7\n', status)
            self.assertIn('\torchestrator\tqueue_finished rc=7\n', status)
            manifest = json.loads(
                (log_root / 'selection_manifest.json').read_text()
            )
            self.assertEqual(manifest['schedule']['initial'], 0.05)
            self.assertEqual(manifest['schedule']['final'], 0.005)
            self.assertEqual(manifest['blind_circuits'], [])
            self.assertEqual(
                manifest['gpu_gate']['compute_processes'], 'record_only'
            )
            summary_command = run.call_args_list[1].args[0]
            self.assertIn(
                'scripts/summarize_joint_lambda_schedule_seed0.py',
                summary_command,
            )

    def test_execute_refuses_changed_worktree_before_summary(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            dir=repo_root / 'logs'
        ) as temporary_dir:
            log_root = Path(temporary_dir) / 'formal'
            with (
                patch(
                    'run_strict_joint_lambda_schedule_seed0.current_git_state',
                    side_effect=[('abc', ''), ('changed', '')],
                ),
                patch(
                    'run_strict_joint_lambda_schedule_seed0.wait_for_gpu_capacity',
                    return_value=(39000, 0),
                ),
                patch(
                    'run_strict_joint_lambda_schedule_seed0.subprocess.run',
                    return_value=SimpleNamespace(returncode=0),
                ) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, 'changed during'):
                    runner_main([
                        '--execute', '--gpu', '3',
                        '--log_root', str(log_root),
                    ])
            self.assertEqual(run.call_count, 1)
            status = (log_root / 'queue_status.tsv').read_text()
            self.assertIn('\torchestrator\texception type=RuntimeError\n', status)
            self.assertIn('\torchestrator\tqueue_finished rc=1\n', status)
            self.assertNotIn('\tsummary\tstart\n', status)


if __name__ == '__main__':
    unittest.main()
