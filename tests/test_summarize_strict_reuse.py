import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_strict_reuse import (
    COMMON_EXPECTATIONS,
    EXPECTED_AUDIT_EPOCHS,
    EXPECTED_MATRIX,
    EXPECTED_METHOD_SEEDS,
    compare_methods,
    group_runs,
    load_run,
    promotion_gate,
    summary_stats,
    summarize_gradient_audit,
    validate_lambda_pairing,
    validate_matrix,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_run(method, seed, scale=1.0):
    provenance = {
        'sgrl_cache_key': f'cache-{seed}',
        'sgrl_checkpoint_sha256': f'sgrl-{seed}',
        'split_fingerprint': f'split-{seed}',
        'normalization_state_sha256': f'norm-{seed}',
        'downstream_initial_model_fingerprint': f'model-{seed}',
        'train_sampler_fingerprint': f'train-{seed}',
        'eval_sampler_fingerprints': {'val': f'eval-{seed}'},
    }
    return {
        'method': method,
        'seed': seed,
        'val_mse': 0.01 * scale,
        'digtime': 0.02 * scale,
        'timing_ctrl': 0.03 * scale,
        'array': 0.04 * scale,
        'transfer_mean': 0.03 * scale,
        'provenance': provenance,
    }


class StrictReuseSummaryTest(unittest.TestCase):
    def test_three_seed_group_comparison_gate_and_pairing(self):
        runs = []
        for method, seeds in EXPECTED_METHOD_SEEDS.items():
            scale = 1.02 if method == 'joint_rank0_lambda005' else 1.0
            for seed in seeds:
                runs.append(synthetic_run(method, seed, scale))

        summaries, by_method = group_runs(runs)
        self.assertEqual(
            summaries['joint_rank0_lambda005']['seeds'],
            [0, 1, 2],
        )
        comparison = compare_methods(
            'joint_rank0_lambda005', 'static_dual', by_method
        )
        self.assertAlmostEqual(
            comparison['metrics']['transfer_mean']['delta_of_means'],
            0.02,
        )
        gate = promotion_gate(comparison)
        self.assertTrue(gate['passes'])
        self.assertTrue(all(
            seed_gate['passes'] for seed_gate in gate['per_seed'].values()
        ))
        self.assertTrue(validate_lambda_pairing(by_method)['passes'])

        by_method['joint_rank0_lambda005'][1]['provenance'][
            'split_fingerprint'
        ] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'split_fingerprint'):
            validate_lambda_pairing(by_method)

    def test_gradient_audit_recomputed_from_32_raw_records(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runs = []
            for seed in (1, 2):
                records = []
                for index, epoch in enumerate(EXPECTED_AUDIT_EPOCHS):
                    cosine = -0.2 if index % 2 else 0.2
                    records.append({
                        'epoch': epoch,
                        'batch_index': 0,
                        'joint_gcl_lambda': 0.05,
                        'global': {'cosine': cosine},
                        'groups': {
                            'embeddings': {'cosine': cosine / 2},
                            'layers_0': {'cosine': cosine},
                            'layers_1': {'cosine': -cosine},
                        },
                    })
                path = root / f'seed{seed}.jsonl'
                path.write_text(
                    ''.join(json.dumps(record) + '\n' for record in records)
                )
                runs.append({
                    'seed': seed,
                    '_runtime_metadata': {
                        'joint_gradient_audit_path': str(path),
                    },
                    '_metrics': {
                        'joint_gradient_audit_summary': {
                            'record_count': 16,
                            'global_cosine': {
                                'mean': summary_stats([
                                    record['global']['cosine']
                                    for record in records
                                ])['mean'],
                            },
                        },
                    },
                })

            summary = summarize_gradient_audit(
                {'joint_rank0_lambda005': runs}, root
            )
            self.assertEqual(summary['record_count'], 32)
            self.assertEqual(summary['epochs'], EXPECTED_AUDIT_EPOCHS)
            self.assertAlmostEqual(
                summary['pooled']['global']['negative_fraction'], 0.5
            )
            self.assertAlmostEqual(summary['pooled']['global']['mean'], 0.0)

    def test_matrix_requires_exact_four_roots_and_run_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir)
            roots = []
            for root_name, runs in EXPECTED_MATRIX.items():
                root = base / root_name
                roots.append(root)
                for run_name in runs:
                    artifact = root / run_name / 'artifact'
                    artifact.mkdir(parents=True)
                    (artifact / 'run_config.json').write_text('{}')
            result = validate_matrix(roots)
            self.assertEqual(set(result), set(EXPECTED_MATRIX))

            extra = roots[0] / 'unexpected' / 'artifact'
            extra.mkdir(parents=True)
            (extra / 'run_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'run matrix mismatch'):
                validate_matrix(roots)

    def test_load_run_checks_protocol_metrics_and_hashed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            repo = Path(temporary_dir)
            root = repo / 'strict_root'
            artifact = root / 'no_gcl' / 'artifact'
            artifact.mkdir(parents=True)
            checkpoint = artifact / 'best_model.pt'
            checkpoint.write_bytes(b'model')
            normalization = artifact / 'normalization.pt'
            normalization.write_bytes(b'normalization')
            args = {
                **COMMON_EXPECTATIONS,
                'seed': 0,
                'pretraining_seed': 0,
                'embedding_inference_seed': 20260711,
                'downstream_seed': 0,
                'train_sampler_seed': 0,
                'relation_sample_seed': 20260711,
                'split_seed': 0,
                'eval_seed': 20260711,
                'sgrl': 0,
                'sgrl_mode': 'static',
                'hid_dim': 64,
            }
            config = {
                'status': 'completed',
                'git_commit': 'abc',
                'args': args,
                'runtime_metadata': {
                    'source_graph_names': ['ssram'],
                    'transfer_graph_names': [
                        'digtime', 'timing_ctrl', 'array_128_32_8t'
                    ],
                    'normalization_state_path': str(normalization),
                    'normalization_state_sha256': sha256(normalization),
                    'resolved_seeds': {
                        'pretraining_seed': 0,
                        'embedding_inference_seed': 20260711,
                        'downstream_seed': 0,
                        'train_sampler_seed': 0,
                        'relation_sample_seed': 20260711,
                        'split_seed': 0,
                        'eval_seed': 20260711,
                    },
                },
            }
            result = lambda value: {'mse_raw': value}
            metrics = {
                'status': 'completed',
                'best_epoch': 4,
                'best_val_mse': 0.01,
                'best_checkpoint': str(checkpoint),
                'validation_results': result(0.01 + 1e-9),
                'test_results': {
                    'digtime': result(0.02),
                    'timing_ctrl': result(0.03),
                    'array_128_32_8t': result(0.04),
                },
            }
            config_path = artifact / 'run_config.json'
            metrics_path = artifact / 'metrics.json'
            config_path.write_text(json.dumps(config))
            metrics_path.write_text(json.dumps(metrics))

            run = load_run(
                root, 'no_gcl', 'no_gcl', 0, repo, set()
            )
            self.assertAlmostEqual(run['transfer_mean'], 0.03)
            self.assertEqual(run['checkpoint_sha256'], sha256(checkpoint))

            original_config = json.loads(json.dumps(config))
            mutations = {
                'epochs': 1,
                'net_only': 1,
                'cl_model': 'sage',
                'num_head_layers': 1,
                'dropout': 0.9,
                'use_bn': 1,
                'act_fn': 'relu',
                'noise_sigma': 0.1,
            }
            for argument, wrong_value in mutations.items():
                with self.subTest(argument=argument):
                    config = json.loads(json.dumps(original_config))
                    config['args'][argument] = wrong_value
                    config_path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(
                        ValueError, f'mismatched {argument}'
                    ):
                        load_run(root, 'no_gcl', 'no_gcl', 0, repo, set())

            config = json.loads(json.dumps(original_config))
            config_path.write_text(json.dumps(config))
            metrics['best_epoch'] = 160
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'invalid best epoch'):
                load_run(root, 'no_gcl', 'no_gcl', 0, repo, set())

            metrics['best_epoch'] = 4
            metrics_path.write_text(json.dumps(metrics))
            normalization.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'provenance SHA mismatch'):
                load_run(root, 'no_gcl', 'no_gcl', 0, repo, set())


if __name__ == '__main__':
    unittest.main()
