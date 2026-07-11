import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_experiments import (
    discover_runs,
    paired_differences,
    summarize_groups,
)


def write_run(
        root, name, seed, val, test_offset=0.0, started='01:00:00',
        joint_lambda=0.0):
    artifact = root / name / f'{seed}_{started}_artifacts'
    artifact.mkdir(parents=True)
    checkpoint = artifact / 'best_model.pt'
    checkpoint.write_bytes(b'checkpoint')
    args = {
        'seed': seed,
        'gpu': 3,
        'log_dir': str(root / name),
        'run_artifact_dir': str(artifact),
        'run_log_path': str(artifact.with_suffix('.txt')),
        'sgrl': 1,
        'sgrl_mode': 'joint_shared',
        'regress_loss': 'mse',
        'joint_gcl_lambda': joint_lambda,
        'joint_lora_rank': 8,
    }
    config = {
        'status': 'completed',
        'started_at': f'2026-01-01T{started}+08:00',
        'ended_at': '2026-01-01T02:00:00+08:00',
        'git_commit': 'abc123',
        'args': args,
    }
    tests = {
        'digtime': {'mse_raw': 0.02 + test_offset},
        'timing_ctrl': {'mse_raw': 0.03 + test_offset},
        'array_128_32_8t': {'mse_raw': 0.04 + test_offset},
    }
    metrics = {
        'status': 'completed',
        'best_epoch': 4,
        'best_val_mse': val,
        'best_checkpoint': str(checkpoint),
        'test_results': tests,
    }
    (artifact / 'run_config.json').write_text(json.dumps(config))
    (artifact / 'metrics.json').write_text(json.dumps(metrics))


class SummarizeExperimentsTest(unittest.TestCase):
    def test_groups_duplicates_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_run(root, 'baseline', 0, 0.10)
            write_run(root, 'baseline', 1, 0.20)
            write_run(
                root, 'candidate', 0, 0.08,
                test_offset=-0.01, joint_lambda=0.05,
            )
            write_run(
                root, 'candidate', 1, 0.18,
                test_offset=-0.01, joint_lambda=0.05,
            )
            write_run(
                root,
                'candidate',
                1,
                0.19,
                test_offset=-0.005,
                started='01:30:00',
                joint_lambda=0.05,
            )

            rows = discover_runs([root])
            groups = summarize_groups(rows)

            candidate_name = next(
                name for name in groups if 'lambda=0.05' in name
            )
            baseline_name = next(
                name for name in groups if 'lambda=0.0' in name
            )
            self.assertEqual(groups[candidate_name]['num_runs'], 2)
            self.assertAlmostEqual(
                groups[candidate_name]['metrics']['val_mse']['mean'],
                0.13,
            )
            paired = paired_differences(rows, baseline_name)
            self.assertAlmostEqual(
                paired[candidate_name]['metrics']['val_mse']['mean'],
                -0.02,
            )
            duplicate_rows = [
                row for row in rows
                if row['run_name'] == 'candidate' and row['seed'] == 1
            ]
            self.assertEqual(sum(row['canonical'] for row in duplicate_rows), 1)
            self.assertTrue(all(
                any(item.startswith('duplicate_config:') for item in row['anomalies'])
                for row in duplicate_rows
            ))


if __name__ == '__main__':
    unittest.main()
