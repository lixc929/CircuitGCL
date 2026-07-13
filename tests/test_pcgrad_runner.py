import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_strict_joint_pcgrad_seed0 import (
    METHOD_NAME,
    main as runner_main,
    validate_physical_gpu,
    wait_for_gpu_capacity,
)


class PCGradRunnerTest(unittest.TestCase):
    def test_script_entrypoint_runs_from_repository_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / 'run_strict_joint_pcgrad_seed0.py')],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['mode'], 'dry_run')
        self.assertFalse(payload['writes_performed'])

    def test_dry_run_is_exact_nonblind_pcgrad_control(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner_main(['--gpu', '4']), 0)
        payload = json.loads(output.getvalue())
        command = payload['command']
        self.assertFalse(payload['writes_performed'])
        self.assertEqual(payload['physical_gpu'], 4)
        self.assertEqual(payload['child_logical_gpu'], 0)
        self.assertEqual(command[command.index('--gpu') + 1], '0')
        self.assertEqual(
            command[command.index('--joint_gradient_strategy') + 1],
            'pcgrad',
        )
        self.assertEqual(
            command[command.index('--joint_gcl_lambda_schedule') + 1],
            'constant',
        )
        self.assertNotIn('--joint_gcl_lambda_final', command)
        self.assertNotIn('sp8192w', ' '.join(command).lower())

    def test_gpu_validation_is_limited_to_gpu3_or_gpu4(self):
        for gpu in (-1, 0, 1, 2, 5):
            with self.subTest(gpu=gpu):
                with self.assertRaises(ValueError):
                    validate_physical_gpu(gpu)
        validate_physical_gpu(3)
        validate_physical_gpu(4)

    def test_capacity_gate_uses_memory_only_and_locks_commit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            status_path = Path(temporary_dir) / 'queue.tsv'
            with (
                patch(
                    'run_strict_joint_pcgrad_seed0.current_git_state',
                    return_value=('abc', ''),
                ),
                patch(
                    'run_strict_joint_pcgrad_seed0.gpu_snapshot',
                    side_effect=[
                        (6000, 100, ['foreign']),
                        (7000, 100, ['foreign']),
                    ],
                ),
                patch('run_strict_joint_pcgrad_seed0.time.sleep'),
            ):
                free_mb, utilization = wait_for_gpu_capacity(
                    4, status_path, 'abc', Path(temporary_dir),
                    6500, 1, 1,
                )
            self.assertEqual((free_mb, utilization), (7000, 100))
            status = status_path.read_text()
            self.assertIn('processes=1', status)
            self.assertIn('free_mb=6000', status)

            with patch(
                'run_strict_joint_pcgrad_seed0.current_git_state',
                return_value=('changed', ''),
            ):
                with self.assertRaisesRegex(RuntimeError, 'Repository changed'):
                    wait_for_gpu_capacity(
                        4, status_path, 'abc', Path(temporary_dir),
                        6500, 1, 1,
                    )

    def test_execute_propagates_validator_rc(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_root = Path(temporary_dir) / 'formal'
            prerequisite = Path(temporary_dir) / 'schedule_summary.json'
            prerequisite.write_text('{}')
            results = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=9)]
            with (
                patch(
                    'run_strict_joint_pcgrad_seed0.current_git_state',
                    return_value=('abc', ''),
                ),
                patch(
                    'run_strict_joint_pcgrad_seed0.wait_for_gpu_capacity',
                    return_value=(39000, 100),
                ),
                patch(
                    'run_strict_joint_pcgrad_seed0.validate_linear_schedule_prerequisite',
                    return_value=prerequisite,
                ),
                patch(
                    'run_strict_joint_pcgrad_seed0.subprocess.run',
                    side_effect=results,
                ) as run,
            ):
                rc = runner_main([
                    '--execute', '--gpu', '4',
                    '--log_root', str(log_root),
                ])
            self.assertEqual(rc, 9)
            self.assertEqual(run.call_count, 2)
            status = (log_root / 'queue_status.tsv').read_text()
            self.assertIn(f'\t{METHOD_NAME}\tstart ', status)
            self.assertIn('\tsummary\tfinish rc=9\n', status)
            self.assertIn('\torchestrator\tqueue_finished rc=9\n', status)
            manifest = json.loads(
                (log_root / 'selection_manifest.json').read_text()
            )
            self.assertEqual(
                manifest['gradient_strategy']['name'], 'pcgrad'
            )
            self.assertEqual(manifest['blind_circuits'], [])
            self.assertEqual(manifest['physical_gpu'], 4)
            self.assertEqual(
                manifest['gpu_gate']['utilization'], 'record_only'
            )
            self.assertIn(
                'scripts/summarize_joint_pcgrad_seed0.py',
                run.call_args_list[1].args[0],
            )


if __name__ == '__main__':
    unittest.main()
