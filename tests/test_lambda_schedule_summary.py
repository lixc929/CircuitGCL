import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_joint_lambda_schedule_seed0 import (
    CONTROL_EXPECTATIONS,
    CORE_EXPECTATIONS,
    EXPECTED_AUDIT_EPOCHS,
    METHOD_NAME,
    STATIC_PAIR_FIELDS,
    build_report,
    expected_lambda,
    validate_audit_records,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result(value):
    return {'mse_raw': value}


def metrics(checkpoint, val, transfers, schedule=False):
    payload = {
        'status': 'completed',
        'best_epoch': 100,
        'best_val_mse': val,
        'best_checkpoint': str(checkpoint),
        'validation_results': result(val),
        'test_results': {
            name: result(value) for name, value in transfers.items()
        },
    }
    if schedule:
        payload.update({
            'joint_gcl_lambda': 0.05,
            'joint_gcl_lambda_schedule': 'linear',
            'joint_gcl_lambda_final': 0.005,
            'joint_gcl_lambda_at_best_epoch': expected_lambda(100),
            'joint_gcl_lambda_effective_last': 0.005,
        })
    return payload


class LambdaScheduleSummaryTest(unittest.TestCase):
    def _write_fixture(self, base):
        candidate_root = base / 'candidate'
        fixed_root = base / 'fixed'
        static_root = base / 'static'
        candidate_artifact = candidate_root / METHOD_NAME / 'artifact'
        fixed_artifact = fixed_root / 'joint_rank0_lambda005_seed0' / 'artifact'
        static_artifact = static_root / 'static_dual' / 'artifact'
        for artifact in (candidate_artifact, fixed_artifact, static_artifact):
            artifact.mkdir(parents=True)
            (artifact / 'best_model.pt').write_bytes(b'model')

        pair = {
            'sgrl_cache_key': 'cache',
            'sgrl_checkpoint_sha256': 'sgrl',
            'split_fingerprint': 'split',
            'normalization_state_sha256': 'normalization',
            'downstream_initial_model_fingerprint': 'initial',
            'train_sampler_fingerprint': 'train',
            'eval_sampler_fingerprints': {'val': 'eval'},
        }
        graph_runtime = {
            'source_graph_names': ['ssram'],
            'transfer_graph_names': [
                'digtime', 'timing_ctrl', 'array_128_32_8t'
            ],
            'sgrl_train_graph_names': ['ssram'],
        }
        audit_records = []
        for epoch in EXPECTED_AUDIT_EPOCHS:
            audit_records.append({
                'epoch': epoch,
                'batch_index': 0,
                'joint_gcl_lambda': expected_lambda(epoch),
                'joint_gcl_lambda_schedule': 'linear',
                'global': {'cosine': 0.1},
            })
        audit_path = candidate_artifact / 'joint_shared_gradient_audit.jsonl'
        audit_path.write_text(
            ''.join(json.dumps(record) + '\n' for record in audit_records)
        )
        audit_summary = {
            'schema_version': 2,
            'record_count': 16,
            'lambda_schedule': {
                'initial': 0.05,
                'schedule': 'linear',
                'final': 0.005,
                'epochs': 160,
            },
        }
        audit_summary_path = (
            candidate_artifact / 'joint_shared_gradient_summary.json'
        )
        audit_summary_path.write_text(json.dumps(audit_summary))
        runtime = {
            **pair,
            **graph_runtime,
            'joint_gradient_audit_path': str(audit_path),
            'joint_gradient_audit_sha256': sha256(audit_path),
            'joint_gradient_summary_path': str(audit_summary_path),
            'joint_gradient_summary_sha256': sha256(audit_summary_path),
        }
        command = ['python', 'main.py', '--joint_gcl_lambda_schedule', 'linear']
        candidate_config = {
            'status': 'completed',
            'git_commit': 'abc',
            'command': command,
            'args': dict(CORE_EXPECTATIONS),
            'runtime_metadata': runtime,
        }
        transfers = {
            'digtime': 0.014,
            'timing_ctrl': 0.011,
            'array_128_32_8t': 0.011,
        }
        candidate_metrics = metrics(
            candidate_artifact / 'best_model.pt', 0.0079, transfers, schedule=True
        )
        candidate_metrics['joint_gradient_audit_summary'] = audit_summary
        (candidate_artifact / 'run_config.json').write_text(
            json.dumps(candidate_config)
        )
        (candidate_artifact / 'metrics.json').write_text(
            json.dumps(candidate_metrics)
        )
        manifest = {
            'method': METHOD_NAME,
            'seeds': [0],
            'epochs': 160,
            'blind_circuits': [],
            'git_commit': 'abc',
            'command': command,
            'schedule': {
                'name': 'linear',
                'initial': 0.05,
                'final': 0.005,
                'formula': '0.05 + (0.005 - 0.05) * epoch / 159',
            },
        }
        (candidate_root / 'selection_manifest.json').write_text(
            json.dumps(manifest)
        )

        fixed_config = {
            'status': 'completed',
            'args': copy.deepcopy(
                CONTROL_EXPECTATIONS['joint_rank0_lambda005']
            ),
            'runtime_metadata': {
                **copy.deepcopy(pair),
                **copy.deepcopy(graph_runtime),
            },
        }
        (fixed_artifact / 'run_config.json').write_text(json.dumps(fixed_config))
        (fixed_artifact / 'metrics.json').write_text(json.dumps(metrics(
            fixed_artifact / 'best_model.pt', 0.0080, transfers
        )))
        (static_artifact / 'run_config.json').write_text(json.dumps({
            'status': 'completed',
            'args': copy.deepcopy(CONTROL_EXPECTATIONS['static_dual']),
            'runtime_metadata': {
                **{
                    field: copy.deepcopy(pair[field])
                    for field in STATIC_PAIR_FIELDS
                },
                **copy.deepcopy(graph_runtime),
            },
        }))
        (static_artifact / 'metrics.json').write_text(json.dumps(metrics(
            static_artifact / 'best_model.pt', 0.0078,
            {
                'digtime': 0.015,
                'timing_ctrl': 0.011,
                'array_128_32_8t': 0.010,
            },
        )))
        decoy_artifact = static_root / 'another_method' / 'artifact'
        decoy_artifact.mkdir(parents=True)
        (decoy_artifact / 'run_config.json').write_text(json.dumps({
            'status': 'completed',
        }))
        return candidate_root, fixed_root, static_root, candidate_artifact

    def test_build_report_validates_pair_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = self._write_fixture(Path(temporary_dir))
            report = build_report(*roots[:3])
            self.assertTrue(report['promotion_gate']['passes'])
            self.assertTrue(report['selection']['advances_to_seeds12'])
            self.assertTrue(report['paired_provenance']['passes'])
            self.assertEqual(report['gradient_audit']['record_count'], 16)

            config_path = roots[3] / 'run_config.json'
            config = json.loads(config_path.read_text())
            config['runtime_metadata']['split_fingerprint'] = 'wrong'
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                build_report(*roots[:3])

    def test_audit_rejects_wrong_effective_lambda(self):
        records = [
            {
                'epoch': epoch,
                'batch_index': 0,
                'joint_gcl_lambda': expected_lambda(epoch),
                'joint_gcl_lambda_schedule': 'linear',
                'global': {'cosine': 0.1},
            }
            for epoch in EXPECTED_AUDIT_EPOCHS
        ]
        self.assertEqual(validate_audit_records(records)['record_count'], 16)
        records[2]['joint_gcl_lambda'] = 0.05
        with self.assertRaisesRegex(ValueError, 'lambda mismatch'):
            validate_audit_records(records)

    def test_control_rejects_wrong_seed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = self._write_fixture(Path(temporary_dir))
            static_config_path = next(
                (roots[2] / 'static_dual').rglob('run_config.json')
            )
            config = json.loads(static_config_path.read_text())
            config['args']['seed'] = 1
            static_config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'mismatched seed'):
                build_report(*roots[:3])


if __name__ == '__main__':
    unittest.main()
