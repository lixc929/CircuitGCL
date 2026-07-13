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
    STATIC_PAIR_FIELDS,
)
from summarize_joint_pcgrad_seed0 import (
    CORE_EXPECTATIONS,
    EXPECTED_GRADIENT_EPOCHS,
    EXPECTED_PCGRAD_EPOCHS,
    EXPECTED_SCOPE_PARAMETER_NAMES,
    METHOD_NAME,
    build_report,
    validate_pcgrad_records,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result(value):
    return {'mse_raw': value}


def cosine_distribution(count=2):
    return {
        'count': count,
        'mean': 0.0,
        'std': 0.5,
        'min': -0.5,
        'q10': -0.4,
        'q25': -0.25,
        'median': 0.0,
        'q75': 0.25,
        'q90': 0.4,
        'max': 0.5,
        'negative_fraction': 0.5,
    }


def constant_distribution(value, count=16):
    return {
        'count': count,
        'mean': value,
        'std': 0.0,
        'min': value,
        'q10': value,
        'q25': value,
        'median': value,
        'q75': value,
        'q90': value,
        'max': value,
        'negative_fraction': 1.0 if value < 0.0 else 0.0,
    }


def metrics(checkpoint, val, transfers, pcgrad=False):
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
    if pcgrad:
        payload.update({
            'joint_gcl_lambda': 0.05,
            'joint_gcl_lambda_schedule': 'constant',
            'joint_gcl_lambda_final': None,
            'joint_gcl_lambda_at_best_epoch': 0.05,
            'joint_gcl_lambda_effective_last': 0.05,
            'joint_gradient_strategy': 'pcgrad',
        })
    return payload


class PCGradSummaryTest(unittest.TestCase):
    def _write_fixture(self, base):
        candidate_root = base / 'candidate'
        fixed_root = base / 'fixed'
        static_root = base / 'static'
        schedule_root = base / 'schedule'
        candidate_artifact = candidate_root / METHOD_NAME / 'artifact'
        fixed_artifact = fixed_root / 'joint_rank0_lambda005_seed0' / 'artifact'
        static_artifact = static_root / 'static_dual' / 'artifact'
        for artifact in (candidate_artifact, fixed_artifact, static_artifact):
            artifact.mkdir(parents=True)
            (artifact / 'best_model.pt').write_bytes(b'model')
        schedule_root.mkdir(parents=True)

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
        gradient_records = [
            {
                'epoch': epoch,
                'batch_index': 0,
                'joint_gcl_lambda': 0.05,
                'joint_gcl_lambda_schedule': 'constant',
                'global': {
                    'cosine': -0.1,
                    'supervised_norm': 1.0,
                    'gcl_norm': 2.0,
                },
                'groups': {
                    'embeddings': {'cosine': -0.2},
                    'layers_0': {'cosine': 0.1},
                    'layers_1': {'cosine': -0.1},
                    'normalization_activation': {'cosine': None},
                },
            }
            for epoch in EXPECTED_GRADIENT_EPOCHS
        ]
        gradient_path = candidate_artifact / 'joint_shared_gradient_audit.jsonl'
        gradient_path.write_text(''.join(
            json.dumps(record) + '\n' for record in gradient_records
        ))
        gradient_summary = {
            'schema_version': 2,
            'record_count': 16,
            'lambda_schedule': {
                'initial': 0.05,
                'schedule': 'constant',
                'final': None,
                'epochs': 160,
            },
            'global_cosine': constant_distribution(-0.1),
            'mean_supervised_norm': 1.0,
            'mean_gcl_norm': 2.0,
            'groups': {
                'embeddings': constant_distribution(-0.2),
                'layers_0': constant_distribution(0.1),
                'layers_1': constant_distribution(-0.1),
                'normalization_activation': {'count': 0},
            },
        }
        gradient_summary_path = (
            candidate_artifact / 'joint_shared_gradient_summary.json'
        )
        gradient_summary_path.write_text(json.dumps(gradient_summary))

        pcgrad_records = [
            {
                'epoch': epoch,
                'joint_gcl_lambda': 0.05,
                'strategy': 'pcgrad',
                'batch_count': 2,
                'conflict_count': 1,
                'conflict_fraction': 0.5,
                'raw_cosine': cosine_distribution(),
                'mean_standard_combined_norm': 1.0,
                'mean_projected_combined_norm': 1.1,
                'mean_projected_to_standard_norm': 1.1,
                'projected_to_standard_norm_count': 2,
            }
            for epoch in EXPECTED_PCGRAD_EPOCHS
        ]
        pcgrad_path = candidate_artifact / 'joint_shared_pcgrad_audit.jsonl'
        pcgrad_path.write_text(''.join(
            json.dumps(record) + '\n' for record in pcgrad_records
        ))
        pcgrad_summary = {
            'schema_version': 1,
            'strategy': 'pcgrad',
            'epoch_count': 160,
            'batch_count': 320,
            'conflict_count': 160,
            'conflict_fraction': 0.5,
            'scope': {
                'module': 'shared_backbone.gnn',
                'parameter_tensors': 10,
                'parameter_values': 17536,
                'parameter_names': EXPECTED_SCOPE_PARAMETER_NAMES,
            },
            'raw_cosine': cosine_distribution(320),
            'mean_standard_combined_norm': 1.0,
            'mean_projected_combined_norm': 1.1,
            'mean_projected_to_standard_norm': 1.1,
            'projected_to_standard_norm_count': 320,
        }
        pcgrad_summary_path = (
            candidate_artifact / 'joint_shared_pcgrad_summary.json'
        )
        pcgrad_summary_path.write_text(json.dumps(pcgrad_summary))

        runtime = {
            **pair,
            **graph_runtime,
            'joint_gradient_audit_path': str(gradient_path),
            'joint_gradient_audit_sha256': sha256(gradient_path),
            'joint_gradient_summary_path': str(gradient_summary_path),
            'joint_gradient_summary_sha256': sha256(gradient_summary_path),
            'joint_pcgrad_audit_path': str(pcgrad_path),
            'joint_pcgrad_audit_sha256': sha256(pcgrad_path),
            'joint_pcgrad_summary_path': str(pcgrad_summary_path),
            'joint_pcgrad_summary_sha256': sha256(pcgrad_summary_path),
        }
        command = ['python', 'main.py', '--joint_gradient_strategy', 'pcgrad']
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
            candidate_artifact / 'best_model.pt',
            0.0079,
            transfers,
            pcgrad=True,
        )
        candidate_metrics['joint_gradient_audit_summary'] = gradient_summary
        candidate_metrics['joint_pcgrad_audit_summary'] = pcgrad_summary
        (candidate_artifact / 'run_config.json').write_text(
            json.dumps(candidate_config)
        )
        (candidate_artifact / 'metrics.json').write_text(
            json.dumps(candidate_metrics)
        )
        fixed_row = {
            'val_mse': 0.0080,
            'digtime': 0.014,
            'timing_ctrl': 0.011,
            'array': 0.011,
            'transfer_mean': 0.012,
        }
        static_row = {
            'val_mse': 0.0078,
            'digtime': 0.015,
            'timing_ctrl': 0.011,
            'array': 0.010,
            'transfer_mean': 0.012,
        }
        schedule_summary_path = schedule_root / 'schedule_seed0_summary.json'
        schedule_summary_path.write_text(json.dumps({
            'method': 'joint_rank0_lambda005_to0005_linear_seed0',
            'candidate': {
                'val_mse': 0.0081,
                'digtime': 0.0145,
                'timing_ctrl': 0.011,
                'array': 0.011,
                'transfer_mean': 0.012166666666666666,
            },
            'fixed_lambda005': fixed_row,
            'static_dual': static_row,
            'promotion_gate': {'passes': True},
            'selection': {
                'advances_to_seeds12': False,
                'source_val_improvement_over_fixed': -0.0001,
            },
        }))
        manifest = {
            'schema_version': 1,
            'protocol': 'strict_inductive',
            'method': METHOD_NAME,
            'seeds': [0],
            'epochs': 160,
            'blind_circuits': [],
            'automatic_multiseed_expansion': False,
            'child_logical_gpu': 0,
            'physical_gpu': 4,
            'child_cuda_visible_devices': '4',
            'git_commit': 'abc',
            'command': command,
            'gpu_gate': {
                'minimum_free_mb': 6500,
                'utilization': 'record_only',
                'compute_processes': 'record_only',
                'required_consecutive_capacity_samples': 1,
                'poll_seconds': 60,
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
                    'sha256': sha256(schedule_summary_path),
                },
            },
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
            static_artifact / 'best_model.pt',
            0.0078,
            {
                'digtime': 0.015,
                'timing_ctrl': 0.011,
                'array_128_32_8t': 0.010,
            },
        )))
        decoy = static_root / 'another_method' / 'artifact'
        decoy.mkdir(parents=True)
        (decoy / 'run_config.json').write_text(json.dumps({'status': 'completed'}))
        return (
            candidate_root,
            fixed_root,
            static_root,
            schedule_root,
            candidate_artifact,
        )

    def test_build_report_validates_pcgrad_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = self._write_fixture(Path(temporary_dir))
            report = build_report(*roots[:4])
            self.assertTrue(report['promotion_gate']['passes'])
            self.assertTrue(report['selection']['advances_to_seeds12'])
            self.assertEqual(report['pcgrad_audit']['epoch_count'], 160)
            self.assertEqual(report['pcgrad_audit']['batch_count'], 320)
            self.assertEqual(report['pcgrad_audit']['conflict_count'], 160)
            self.assertTrue(report['paired_provenance']['passes'])

    def test_pcgrad_records_reject_mismatched_fraction(self):
        records = [
            {
                'epoch': epoch,
                'joint_gcl_lambda': 0.05,
                'strategy': 'pcgrad',
                'batch_count': 2,
                'conflict_count': 1,
                'conflict_fraction': 0.5,
                'raw_cosine': cosine_distribution(),
                'mean_standard_combined_norm': 1.0,
                'mean_projected_combined_norm': 1.1,
                'mean_projected_to_standard_norm': 1.1,
                'projected_to_standard_norm_count': 2,
            }
            for epoch in EXPECTED_PCGRAD_EPOCHS
        ]
        self.assertEqual(validate_pcgrad_records(records)['batch_count'], 320)
        records[10]['conflict_fraction'] = 0.25
        with self.assertRaisesRegex(ValueError, 'fraction mismatch'):
            validate_pcgrad_records(records)

    def test_build_report_rejects_tampered_pcgrad_summary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = self._write_fixture(Path(temporary_dir))
            metrics_path = roots[4] / 'metrics.json'
            payload = json.loads(metrics_path.read_text())
            payload['joint_pcgrad_audit_summary']['conflict_count'] = 159
            metrics_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                build_report(*roots[:4])


if __name__ == '__main__':
    unittest.main()
