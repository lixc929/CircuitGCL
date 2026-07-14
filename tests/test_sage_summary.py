import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_joint_sage_seed0 import (
    EXPECTED_SAGE_CACHE_KEY,
    LAMBDA0_NAME,
    LAMBDA005_NAME,
    REGISTERED_CLUSTER_LAMBDA005_VAL,
    REGISTERED_REQUIRED_VAL,
    REGISTERED_STATIC_ROW,
    REGISTERED_UPPER_BOUNDS,
    RESOLVED_SEED_EXPECTATIONS,
    RUN_NAMES,
    SAGE_COMMON_EXPECTATIONS,
    SAGE_METHOD_EXPECTATIONS,
    SCREEN_NAME,
    _control_expectations,
    build_report,
    file_sha256,
    validate_control_artifacts,
)


TRANSFER_NAMES = ('digtime', 'timing_ctrl', 'array_128_32_8t')


def result(value):
    return {'mse_raw': value}


class SageSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'candidate'
        self.static_root = self.base / 'static'
        self.fixed_root = self.base / 'fixed'
        self.shared = self.base / 'shared'
        self.shared.mkdir()
        self.processed_caches = [
            {
                'graph_name': name,
                'cache_key': f'cache-{name}',
                'raw_sha256': f'raw-{name}',
                'processed_sha256': f'processed-{name}',
                'relation_sample_seed': 20260711,
                'graph_relation_sample_seed': index + 10,
            }
            for index, name in enumerate(('ssram', *TRANSFER_NAMES))
        ]
        self.normalization = self.shared / 'normalization.pt'
        self.normalization.write_bytes(b'normalization')
        self.split = self.shared / 'split.pt'
        self.split.write_bytes(b'split')
        self.eval_fingerprints = {
            'val': 'eval-val',
            'val:best': 'eval-val',
            'test:digtime': 'eval-digtime',
            'test:timing_ctrl': 'eval-timing',
            'test:array_128_32_8t': 'eval-array',
        }
        self._write_controls()
        self.sage_online, self.sage_metadata, self.sage_target = (
            self._write_sage_checkpoint()
        )
        self.candidate_configs = {}
        self.candidate_metrics = {}
        self._write_candidate(LAMBDA0_NAME, 0.008000)
        self._write_candidate(LAMBDA005_NAME, 0.008010)
        self._write_manifest()

    def _runtime(self, checkpoint, metadata, *, cache_key, initial='initial'):
        return {
            'source_graph_names': ['ssram'],
            'transfer_graph_names': list(TRANSFER_NAMES),
            'sgrl_train_graph_names': ['ssram'],
            'resolved_seeds': copy.deepcopy(RESOLVED_SEED_EXPECTATIONS),
            'normalization_scope': 'source',
            'processed_caches': copy.deepcopy(self.processed_caches),
            'sgrl_cache_key': cache_key,
            'sgrl_checkpoint_path': str(checkpoint),
            'sgrl_checkpoint_sha256': file_sha256(checkpoint),
            'sgrl_checkpoint_metadata_path': str(metadata),
            'sgrl_checkpoint_metadata_sha256': file_sha256(metadata),
            'normalization_state_path': str(self.normalization),
            'normalization_state_sha256': file_sha256(self.normalization),
            'split_indices_path': str(self.split),
            'split_indices_sha256': file_sha256(self.split),
            'split_fingerprint': 'split-fingerprint',
            'downstream_initial_model_fingerprint': initial,
            'train_sampler_fingerprint': 'train-sampler',
            'eval_sampler_fingerprints': copy.deepcopy(
                self.eval_fingerprints
            ),
        }

    def _metrics(self, checkpoint, val, transfers, *, joint=False, lam=0.0):
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
        if joint:
            payload.update({
                'joint_gcl_lambda': lam,
                'joint_gcl_lambda_schedule': 'constant',
                'joint_gcl_lambda_final': None,
                'joint_gcl_lambda_effective_last': lam,
                'joint_gcl_lambda_at_best_epoch': lam,
                'joint_gradient_strategy': 'none',
                'joint_gradient_audit_summary': None,
            })
        return payload

    def _write_control(self, root, run_name, method, val, transfers):
        artifact = root / run_name / 'artifact'
        artifact.mkdir(parents=True)
        checkpoint = artifact / 'best_model.pt'
        checkpoint.write_bytes(f'{method}-model'.encode())
        sgrl_checkpoint = artifact / 'sgrl.pkl'
        sgrl_checkpoint.write_bytes(b'cluster-sgrl')
        sgrl_metadata = artifact / 'sgrl.pkl.metadata.json'
        sgrl_metadata.write_text('{}')
        config = {
            'status': 'completed',
            'git_commit': f'commit-{method}',
            'args': _control_expectations(method),
            'runtime_metadata': self._runtime(
                sgrl_checkpoint,
                sgrl_metadata,
                cache_key=f'cluster-{method}',
            ),
        }
        metrics = self._metrics(checkpoint, val, transfers)
        (artifact / 'run_config.json').write_text(json.dumps(config))
        (artifact / 'metrics.json').write_text(json.dumps(metrics))

    def _write_controls(self):
        static_transfers = {
            'digtime': REGISTERED_STATIC_ROW['digtime'],
            'timing_ctrl': REGISTERED_STATIC_ROW['timing_ctrl'],
            'array_128_32_8t': REGISTERED_STATIC_ROW['array'],
        }
        self._write_control(
            self.static_root,
            'static_dual',
            'static_dual',
            REGISTERED_STATIC_ROW['val_mse'],
            static_transfers,
        )
        self._write_control(
            self.static_root,
            'joint_lambda0',
            'joint_rank0_lambda0',
            0.008090950,
            static_transfers,
        )
        self._write_control(
            self.fixed_root,
            'joint_rank0_lambda005_seed0',
            'joint_rank0_lambda005',
            REGISTERED_CLUSTER_LAMBDA005_VAL,
            static_transfers,
        )

    def _write_sage_checkpoint(self):
        online_dir = self.shared / 'pkl' / 'pkl_online'
        target_dir = self.shared / 'pkl' / 'pkl_target'
        online_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        online = online_dir / f'best_online_{EXPECTED_SAGE_CACHE_KEY}.pkl'
        target = target_dir / f'best_target_{EXPECTED_SAGE_CACHE_KEY}.pkl'
        online.write_bytes(b'sage-online')
        target.write_bytes(b'sage-target')
        metadata = Path(f'{online}.metadata.json')
        ssram = self.processed_caches[0]
        identity = {
            'cache_schema_version': 1,
            'train_graph_names': ['ssram'],
            'processed_caches': [{
                'graph_name': ssram['graph_name'],
                'cache_key': ssram['cache_key'],
                'raw_sha256': ssram['raw_sha256'],
                'processed_sha256': ssram['processed_sha256'],
                'relation_sample_seed': ssram['relation_sample_seed'],
                'graph_relation_sample_seed': (
                    ssram['graph_relation_sample_seed']
                ),
            }],
            'graph_scope': 'source',
            'pretraining_seed': 0,
            'target_update': 'sgrl_dual_rsm_ema',
            'model': 'sage',
            'layers': 2,
            'hidden_dim': 64,
            'activation': 'tanh',
            'dropout': 0.3,
            'batch_size': 32768,
            'num_neighbors': 8,
            'num_hops': 2,
            'num_workers': 0,
            'use_bn': 0,
            'epochs': 5,
            'online_lr': 1e-6,
            'target_lr': 2e-7,
            'momentum': 0.99,
            'weight_decay': 0.0,
        }
        metadata.write_text(json.dumps({
            'cache_schema_version': 1,
            'cache_key': EXPECTED_SAGE_CACHE_KEY,
            'identity': identity,
            'online_checkpoint_sha256': file_sha256(online),
            'target_checkpoint_sha256': file_sha256(target),
        }))
        return online, metadata, target

    def _write_candidate(self, name, val):
        artifact = self.root / name / 'artifact'
        artifact.mkdir(parents=True)
        checkpoint = artifact / 'best_model.pt'
        checkpoint.write_bytes(name.encode())
        lam = SAGE_METHOD_EXPECTATIONS[name]['joint_gcl_lambda']
        command = ['python', 'main.py', '--method', name]
        config = {
            'status': 'completed',
            'git_commit': 'abc',
            'command': command,
            'args': {
                **copy.deepcopy(SAGE_COMMON_EXPECTATIONS),
                **copy.deepcopy(SAGE_METHOD_EXPECTATIONS[name]),
            },
            'runtime_metadata': self._runtime(
                self.sage_online,
                self.sage_metadata,
                cache_key=EXPECTED_SAGE_CACHE_KEY,
            ),
        }
        transfers = {
            'digtime': REGISTERED_STATIC_ROW['digtime'],
            'timing_ctrl': REGISTERED_STATIC_ROW['timing_ctrl'],
            'array_128_32_8t': REGISTERED_STATIC_ROW['array'],
        }
        metrics = self._metrics(
            checkpoint, val, transfers, joint=True, lam=lam
        )
        config_path = artifact / 'run_config.json'
        metrics_path = artifact / 'metrics.json'
        config_path.write_text(json.dumps(config))
        metrics_path.write_text(json.dumps(metrics))
        audit_records = []
        for epoch in (-1, *range(0, 160, 10)):
            audit_records.append(self._representation_record(
                'train', 'source_val', epoch
            ))
        for split_name in ('source_val', *TRANSFER_NAMES):
            audit_records.append(self._representation_record(
                'best', split_name, 100
            ))
        (artifact / 'joint_shared_representation_audit.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in audit_records)
        )
        self.candidate_configs[name] = config_path
        self.candidate_metrics[name] = metrics_path

    @staticmethod
    def _representation_record(phase, split, epoch):
        return {
            'phase': phase,
            'split': split,
            'epoch': epoch,
            'base_cosine_to_initial': 0.9,
            'base_norm_mean': 1.0,
            'base_relative_l2_to_initial': 0.1,
            'base_weight_relative_l2': 0.1,
            'task_cosine_to_base': 0.95,
            'task_norm_mean': 1.0,
            'task_relative_l2_to_base': 0.05,
            'stats_residual_scale': 0.1,
            'lora_modules': 0,
        }

    def _write_manifest(self):
        commands = {
            name: json.loads(self.candidate_configs[name].read_text())['command']
            for name in RUN_NAMES
        }
        manifest = {
            'schema_version': 1,
            'git_commit': 'abc',
            'protocol': 'strict_inductive',
            'method': SCREEN_NAME,
            'methods': list(RUN_NAMES),
            'seed': 0,
            'epochs': 160,
            'blind_circuits': [],
            'execution': {
                'ordering': list(RUN_NAMES),
                'checkpoint_producer': LAMBDA0_NAME,
                'checkpoint_readers': [LAMBDA005_NAME],
                'concurrent_checkpoint_writers': 1,
            },
            'selection': {
                'candidates': list(RUN_NAMES),
                'primary_metric': 'source validation raw MSE',
                'source_val_tie_tolerance_prefer_lambda005': 2e-5,
                'minimum_source_val_improvement_over_cluster_lambda005': 2e-5,
                'registered_references': {
                    'canonical_static': REGISTERED_STATIC_ROW,
                    'cluster_rank0_lambda005_val_mse': (
                        REGISTERED_CLUSTER_LAMBDA005_VAL
                    ),
                },
                'registered_upper_bounds': REGISTERED_UPPER_BOUNDS,
                'required_val_mse_strictly_below': REGISTERED_REQUIRED_VAL,
                'automatic_multiseed_expansion': False,
            },
            'control_artifacts': validate_control_artifacts(
                self.base, self.static_root, self.fixed_root
            ),
            'commands': commands,
        }
        self.root.mkdir(exist_ok=True)
        (self.root / 'selection_manifest.json').write_text(
            json.dumps(manifest)
        )

    def _rewrite_config(self, name, mutate):
        path = self.candidate_configs[name]
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    def _rewrite_metrics(self, name, mutate):
        path = self.candidate_metrics[name]
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    def report(self):
        return build_report(
            self.root,
            self.static_root,
            self.fixed_root,
            repo_root=self.base,
        )

    def test_tie_prefers_positive_lambda_and_advances(self):
        report = self.report()
        selection = report['selection']
        self.assertEqual(selection['selected_candidate'], LAMBDA005_NAME)
        self.assertEqual(
            selection['selection_reason'],
            'source_val_tie_prefer_positive_lambda',
        )
        self.assertTrue(selection['advances_to_seeds12'])

    def test_lower_source_val_selects_lambda_zero_outside_tolerance(self):
        def lower(metrics):
            metrics['best_val_mse'] = 0.00795
            metrics['validation_results'] = result(0.00795)

        self._rewrite_metrics(LAMBDA0_NAME, lower)
        report = self.report()
        self.assertEqual(
            report['selection']['selected_candidate'], LAMBDA0_NAME
        )
        self.assertTrue(report['selection']['advances_to_seeds12'])

    def test_ineligible_positive_lambda_cannot_mask_eligible_lambda_zero(self):
        def set_val(name, value):
            self._rewrite_metrics(name, lambda metrics: metrics.update({
                'best_val_mse': value,
                'validation_results': result(value),
            }))

        set_val(LAMBDA0_NAME, 0.008010)
        set_val(LAMBDA005_NAME, 0.008025)
        report = self.report()
        selection = report['selection']
        self.assertEqual(
            selection['advancement_eligible_candidates'], [LAMBDA0_NAME]
        )
        self.assertEqual(selection['selected_candidate'], LAMBDA0_NAME)
        self.assertTrue(selection['advances_to_seeds12'])

    def test_pair_provenance_mismatch_is_rejected(self):
        self._rewrite_config(
            LAMBDA005_NAME,
            lambda config: config['runtime_metadata'].__setitem__(
                'downstream_initial_model_fingerprint', 'different'
            ),
        )
        with self.assertRaisesRegex(ValueError, 'lambda-pair provenance'):
            self.report()

    def test_nonfinite_metric_is_rejected(self):
        self._rewrite_metrics(
            LAMBDA005_NAME,
            lambda metrics: metrics.__setitem__('best_val_mse', float('nan')),
        )
        with self.assertRaisesRegex(ValueError, 'invalid best_val_mse'):
            self.report()

    def test_embedding_cache_is_rejected(self):
        self._rewrite_config(
            LAMBDA005_NAME,
            lambda config: config['runtime_metadata'].__setitem__(
                'embedding_cache_key', 'forbidden'
            ),
        )
        with self.assertRaisesRegex(ValueError, 'embedding cache'):
            self.report()

    def test_target_checkpoint_hash_is_enforced(self):
        self.sage_target.write_bytes(b'tampered-target')
        with self.assertRaisesRegex(ValueError, 'target checkpoint provenance'):
            self.report()

    def test_representation_audit_is_required(self):
        audit = (
            self.candidate_configs[LAMBDA005_NAME].parent
            / 'joint_shared_representation_audit.jsonl'
        )
        audit.unlink()
        with self.assertRaisesRegex(ValueError, 'representation audit'):
            self.report()


if __name__ == '__main__':
    unittest.main()
