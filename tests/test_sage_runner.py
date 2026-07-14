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

from run_strict_joint_sage_seed0 import (
    LAMBDA0_NAME,
    LAMBDA005_NAME,
    RUN_NAMES,
    main as runner_main,
    validate_no_preexisting_sage_checkpoint,
    validate_physical_gpu,
)


def command_value(command, name):
    return command[command.index(name) + 1]


class SageRunnerTest(unittest.TestCase):
    def test_dry_run_locks_two_nonblind_compact_sage_arms(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner_main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload['writes_performed'])
        self.assertEqual(payload['ordering'], list(RUN_NAMES))
        self.assertEqual(payload['child_logical_gpu'], 0)
        self.assertEqual(payload['physical_gpus'][LAMBDA0_NAME], 3)
        self.assertEqual(payload['physical_gpus'][LAMBDA005_NAME], 4)

        for name, expected_lambda in (
                (LAMBDA0_NAME, '0.0'), (LAMBDA005_NAME, '0.05')):
            command = payload['commands'][name]
            self.assertEqual(command_value(command, '--cl_model'), 'sage')
            self.assertEqual(command_value(command, '--model'), 'sage')
            self.assertEqual(command_value(command, '--cl_gnn_layers'), '2')
            self.assertEqual(command_value(command, '--cl_hid_dim'), '64')
            self.assertEqual(command_value(command, '--joint_lora_rank'), '0')
            self.assertEqual(
                command_value(command, '--joint_gcl_lambda'), expected_lambda
            )
            self.assertEqual(
                command_value(command, '--joint_gradient_audit'), '0'
            )
            self.assertEqual(
                command_value(command, '--joint_shared_audit'), '1'
            )
            self.assertEqual(command_value(command, '--gpu'), '0')
            self.assertNotIn('sp8192w', ' '.join(command).lower())

    def test_gpu_validation_allows_only_three_and_four(self):
        for gpu in (-1, 0, 1, 2, 5):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                validate_physical_gpu(gpu)
        validate_physical_gpu(3)
        validate_physical_gpu(4)

    def test_formal_writer_requires_all_sage_checkpoint_paths_absent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = validate_no_preexisting_sage_checkpoint(root)
            self.assertEqual(set(paths), {'online', 'metadata', 'target'})
            online = Path(paths['online'])
            online.parent.mkdir(parents=True)
            online.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError, 'pre-existing files'):
                validate_no_preexisting_sage_checkpoint(root)

    def _execute_with_results(self, results):
        repo_root = Path(__file__).resolve().parents[1]
        temporary = tempfile.TemporaryDirectory(dir=repo_root / 'logs')
        self.addCleanup(temporary.cleanup)
        log_root = Path(temporary.name) / 'formal'
        with (
            patch(
                'run_strict_joint_sage_seed0.current_git_state',
                return_value=('abc', ''),
            ),
            patch(
                'run_strict_joint_sage_seed0.validate_control_artifacts',
                return_value={'controls': 'locked'},
            ),
            patch(
                'run_strict_joint_sage_seed0.'
                'validate_no_preexisting_sage_checkpoint',
                return_value={
                    'online': '/tmp/online',
                    'metadata': '/tmp/metadata',
                    'target': '/tmp/target',
                },
            ),
            patch(
                'run_strict_joint_sage_seed0.wait_for_gpu_capacity',
                return_value=(30000, 100),
            ),
            patch(
                'run_strict_joint_sage_seed0.subprocess.run',
                side_effect=[SimpleNamespace(returncode=value) for value in results],
            ) as run,
        ):
            rc = runner_main(['--execute', '--log_root', str(log_root)])
        return rc, log_root, run

    def test_execute_orders_single_writer_reader_and_summary(self):
        rc, log_root, run = self._execute_with_results([0, 0, 0, 0])
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 4)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn('prewarm_strict_processed_caches.py', commands[0][1])
        self.assertIn(LAMBDA0_NAME, command_value(commands[1], '--log_dir'))
        self.assertIn(LAMBDA005_NAME, command_value(commands[2], '--log_dir'))
        self.assertIn('summarize_joint_sage_seed0.py', commands[3][1])

        status = (log_root / 'queue_status.tsv').read_text()
        self.assertLess(
            status.index(f'\t{LAMBDA0_NAME}\tfinish rc=0'),
            status.index('\tcheckpoint_handoff\tproducer_completed'),
        )
        self.assertLess(
            status.index('\tcheckpoint_handoff\tproducer_completed'),
            status.index(f'\t{LAMBDA005_NAME}\tstart '),
        )
        self.assertIn('\tsummary\tfinish rc=0\n', status)
        self.assertIn('\torchestrator\tqueue_finished rc=0\n', status)
        manifest = json.loads(
            (log_root / 'selection_manifest.json').read_text()
        )
        self.assertEqual(
            manifest['execution']['checkpoint_producer'], LAMBDA0_NAME
        )
        self.assertEqual(
            manifest['execution']['checkpoint_readers'], [LAMBDA005_NAME]
        )
        self.assertEqual(manifest['blind_circuits'], [])
        self.assertFalse(
            manifest['selection']['automatic_multiseed_expansion']
        )

    def test_producer_failure_stops_reader_and_summary(self):
        rc, log_root, run = self._execute_with_results([0, 7])
        self.assertEqual(rc, 7)
        self.assertEqual(run.call_count, 2)
        status = (log_root / 'queue_status.tsv').read_text()
        self.assertIn(
            f'\t{LAMBDA005_NAME}\tnot_started '
            'reason=checkpoint_producer_failure\n',
            status,
        )
        self.assertIn(
            '\tsummary\tnot_started reason=checkpoint_producer_failure\n',
            status,
        )
        self.assertIn('\torchestrator\tqueue_finished rc=7\n', status)

    def test_summary_failure_propagates_to_queue(self):
        rc, log_root, _ = self._execute_with_results([0, 0, 0, 9])
        self.assertEqual(rc, 9)
        status = (log_root / 'queue_status.tsv').read_text()
        self.assertIn('\tsummary\tfinish rc=9\n', status)
        self.assertIn('\torchestrator\tqueue_finished rc=9\n', status)


if __name__ == '__main__':
    unittest.main()
