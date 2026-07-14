import copy
import hashlib
import json
from pathlib import Path
import statistics
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
    EXPECTED_SEED0_SCREEN_COMMITS,
    SEED0_SCREEN_ROOT_NAMES,
    _tsv_rows,
    compare_methods,
    extend_report_with_seed0_screens,
    group_runs,
    load_run,
    normalize_seed0_screen_records,
    promotion_gate,
    summary_stats,
    summarize_gradient_audit,
    validate_canonical_core_roots,
    validate_lambda_pairing,
    validate_matrix,
    validate_seed0_screen_roots,
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
        'best_epoch': 1,
        'git_commit': 'core-commit',
        'artifact_dir': f'artifact/{method}/seed{seed}',
        'checkpoint_path': f'artifact/{method}/seed{seed}/best_model.pt',
        'checkpoint_sha256': f'{method}-{seed}-checkpoint',
    }


def synthetic_core_report():
    runs = []
    for method, seeds in EXPECTED_METHOD_SEEDS.items():
        if method == 'joint_rank0_lambda005':
            scale = 1.02
        elif method == 'joint_lora_r8_lambda005':
            scale = 1.30
        else:
            scale = 1.0
        for seed in seeds:
            runs.append(synthetic_run(method, seed, scale))
    summaries, by_method = group_runs(runs)
    gates = {
        method: promotion_gate(compare_methods(method, 'static_dual', by_method))
        for method, seeds in EXPECTED_METHOD_SEEDS.items()
        if seeds == {0, 1, 2}
    }
    return {
        'schema_version': 1,
        'report_kind': 'strict_reuse_teacher_report',
        'roots': ['core-a', 'core-b', 'core-c', 'core-d'],
        'protocol': {},
        'inventory': {
            'expected_artifacts': 20,
            'completed_artifacts': 20,
            'commits': ['core-commit'],
            'verified_hashed_provenance_files': 0,
            'anomalies': [],
        },
        'runs': runs,
        'method_summaries': summaries,
        'comparisons': {},
        'promotion_gates_vs_static_dual': gates,
        'rank0_lambda_pairing': {'passes': True},
        'gradient_audit': {'sources': [], 'pooled': {}},
    }


def metric_values(row):
    return {
        key: row[key]
        for key in ('val_mse', 'digtime', 'timing_ctrl', 'array', 'transfer_mean')
    }


def synthetic_screen_reports(core):
    static = metric_values(next(
        run for run in core['runs']
        if run['method'] == 'static_dual' and run['seed'] == 0
    ))
    cluster0 = metric_values(next(
        run for run in core['runs']
        if run['method'] == 'joint_rank0_lambda0' and run['seed'] == 0
    ))
    fixed = metric_values(next(
        run for run in core['runs']
        if run['method'] == 'joint_rank0_lambda005' and run['seed'] == 0
    ))

    def candidate(val_delta, transfer_scale):
        row = {
            'val_mse': fixed['val_mse'] + val_delta,
            'digtime': fixed['digtime'] * transfer_scale,
            'timing_ctrl': fixed['timing_ctrl'] * transfer_scale,
            'array': fixed['array'] * transfer_scale,
        }
        row['transfer_mean'] = statistics.mean(
            row[key] for key in ('digtime', 'timing_ctrl', 'array')
        )
        return row

    linear_row = candidate(1e-5, 0.99)
    pcgrad_row = candidate(1.5e-5, 0.98)
    sage0_row = candidate(3e-5, 0.97)
    sage005_row = candidate(2e-5, 0.96)

    def simple_report(screen, method, row):
        improvement = fixed['val_mse'] - row['val_mse']
        return {
            'schema_version': 1,
            'method': method,
            'git_commit': EXPECTED_SEED0_SCREEN_COMMITS[screen],
            'artifact_dir': f'/artifact/{screen}',
            'checkpoint': f'/artifact/{screen}/best_model.pt',
            'checkpoint_sha256': f'{screen}-checkpoint',
            'best_epoch': 100,
            'candidate': row,
            'fixed_lambda005': fixed,
            'static_dual': static,
            'promotion_gate': {'passes': True},
            'selection': {
                'source_val_improvement_over_fixed': improvement,
                'minimum_improvement_to_expand': 2e-5,
                'advances_to_seeds12': False,
            },
            'paired_provenance': {'passes': True},
        }

    linear = simple_report(
        'linear_lambda',
        'joint_rank0_lambda005_to0005_linear_seed0',
        linear_row,
    )
    pcgrad = simple_report(
        'pcgrad', 'joint_rank0_lambda005_pcgrad_seed0', pcgrad_row
    )
    pcgrad['linear_lambda'] = dict(linear_row)
    sage_runs = {}
    for method, row in (
        ('joint_sage_rank0_lambda0_seed0', sage0_row),
        ('joint_sage_rank0_lambda005_seed0', sage005_row),
    ):
        sage_runs[method] = {
            **row,
            'best_epoch': 100,
            'artifact': {
                'metrics_path': f'artifact/{method}/metrics.json',
                'checkpoint_path': f'artifact/{method}/best_model.pt',
                'checkpoint_sha256': f'{method}-checkpoint',
            },
        }
    sage = {
        'schema_version': 1,
        'report_kind': 'compact_shared_sage_seed0_screen',
        'git_commit': EXPECTED_SEED0_SCREEN_COMMITS['graphsage'],
        'runs': sage_runs,
        'controls': {
            'canonical_static_dual': static,
            'cluster_rank0_lambda0': cluster0,
            'cluster_rank0_lambda005': fixed,
        },
        'promotion_gates_vs_canonical_static': {
            method: {'passes': True} for method in sage_runs
        },
        'selection': {
            'advancement_eligible_candidates': [],
            'selected_candidate': None,
            'selection_reason': 'no_candidate_met_source_improvement',
            'minimum_improvement_to_expand': 2e-5,
            'advances_to_seeds12': False,
        },
        'paired_provenance': {'passes': True},
    }
    reports = {
        'linear_lambda': linear,
        'pcgrad': pcgrad,
        'graphsage': sage,
    }
    sources = {
        screen: {
            'json_path': f'{screen}.json',
            'json_sha256': f'{screen}-json',
            'tsv_path': f'{screen}.tsv',
            'tsv_sha256': f'{screen}-tsv',
            'manifest_path': f'{screen}-manifest.json',
            'manifest_sha256': f'{screen}-manifest',
            'blind_scanned_text_files': 1,
        }
        for screen in reports
    }
    return reports, sources


def relocate_screen_paths(reports, screen_roots):
    reports['linear_lambda']['artifact_dir'] = str(
        screen_roots['linear_lambda'] / 'artifact'
    )
    reports['linear_lambda']['checkpoint'] = str(
        screen_roots['linear_lambda'] / 'artifact' / 'best_model.pt'
    )
    reports['pcgrad']['artifact_dir'] = str(
        screen_roots['pcgrad'] / 'artifact'
    )
    reports['pcgrad']['checkpoint'] = str(
        screen_roots['pcgrad'] / 'artifact' / 'best_model.pt'
    )
    for method, run in reports['graphsage']['runs'].items():
        artifact = screen_roots['graphsage'] / method / 'artifact'
        run['artifact']['metrics_path'] = str(artifact / 'metrics.json')
        run['artifact']['checkpoint_path'] = str(artifact / 'best_model.pt')


class StrictReuseSummaryTest(unittest.TestCase):
    def test_seed0_screens_are_separate_from_three_seed_summaries(self):
        core = synthetic_core_report()
        reports, sources = synthetic_screen_reports(core)
        with tempfile.TemporaryDirectory() as temporary_dir:
            repo = Path(temporary_dir)
            screen_roots = {
                screen: repo / 'logs' / root_name
                for screen, root_name in SEED0_SCREEN_ROOT_NAMES.items()
            }
            relocate_screen_paths(reports, screen_roots)
            extended = extend_report_with_seed0_screens(
                core, reports, sources, screen_roots, repo, 4
            )

        self.assertEqual(extended['schema_version'], 2)
        self.assertFalse(extended['protocol']['blind_evaluation_completed'])
        self.assertEqual(
            extended['protocol']['blind_claim_scope'],
            'included_formal_text_artifacts_only',
        )
        self.assertEqual(len(extended['runs']), 20)
        self.assertEqual(len(extended['method_summaries']), 8)
        self.assertEqual(extended['inventory']['completed_artifacts'], 20)
        self.assertEqual(extended['inventory']['seed0_screen_artifacts'], 4)
        self.assertEqual(extended['inventory']['total_completed_artifacts'], 24)
        screen_records = extended['seed0_screens']['records']
        self.assertEqual(len(screen_records), 4)
        self.assertTrue(all(
            record['evidence_scope'] == 'seed0_screen'
            and record['advances_to_seeds12'] is False
            and 'pstdev' not in record
            for record in screen_records
        ))
        teacher = extended['teacher_tables']
        self.assertEqual(len(teacher['three_seed_formal_comparison']), 6)
        self.assertEqual(len(teacher['seed0_controlled_screens']), 5)
        self.assertEqual(
            extended['final_shared_selection']['method'],
            'joint_rank0_lambda005',
        )
        rows = _tsv_rows(extended)
        self.assertEqual(
            sum(row['record_type'] == 'seed0_screen_run' for row in rows), 4
        )
        self.assertEqual(
            sum(row['record_type'] == 'seed0_screen_control' for row in rows), 1
        )
        self.assertEqual(
            sum(row['record_type'] == 'final_selection' for row in rows), 1
        )

    def test_seed0_screen_cross_checks_fail_closed(self):
        core = synthetic_core_report()
        reports, sources = synthetic_screen_reports(core)
        broken = copy.deepcopy(reports)
        broken['pcgrad']['linear_lambda']['val_mse'] += 1e-4
        with self.assertRaisesRegex(ValueError, 'PCGrad/linear prerequisite'):
            normalize_seed0_screen_records(core, broken, sources)

        broken = copy.deepcopy(reports)
        broken['linear_lambda']['promotion_gate']['passes'] = False
        with self.assertRaisesRegex(ValueError, 'selection fields'):
            normalize_seed0_screen_records(core, broken, sources)

        broken = copy.deepcopy(reports)
        broken['graphsage']['extra'] = {'dataset': 'sram_sp_8192w'}
        with self.assertRaisesRegex(ValueError, 'blind identifier'):
            normalize_seed0_screen_records(core, broken, sources)

    def test_seed0_screen_roots_require_exact_repo_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            repo = Path(temporary_dir) / 'repo'
            roots = {}
            for screen, root_name in SEED0_SCREEN_ROOT_NAMES.items():
                root = repo / 'logs' / root_name
                root.mkdir(parents=True)
                roots[screen] = root
            validated = validate_seed0_screen_roots(roots, repo)
            self.assertEqual(set(validated), set(SEED0_SCREEN_ROOT_NAMES))

            external = Path(temporary_dir) / 'external' / (
                SEED0_SCREEN_ROOT_NAMES['linear_lambda']
            )
            external.mkdir(parents=True)
            roots['linear_lambda'] = external
            with self.assertRaisesRegex(ValueError, 'root must be'):
                validate_seed0_screen_roots(roots, repo)

    def test_canonical_core_roots_require_exact_repo_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            repo = Path(temporary_dir) / 'repo'
            roots = []
            for root_name, expected_runs in EXPECTED_MATRIX.items():
                root = repo / 'logs' / root_name
                roots.append(root)
                for run_name in expected_runs:
                    artifact = root / run_name / 'artifact'
                    artifact.mkdir(parents=True)
                    (artifact / 'run_config.json').write_text('{}')
            validated = validate_canonical_core_roots(roots, repo)
            self.assertEqual(set(validated), set(EXPECTED_MATRIX))

            external = Path(temporary_dir) / 'external' / roots[0].name
            for run_name in EXPECTED_MATRIX[roots[0].name]:
                artifact = external / run_name / 'artifact'
                artifact.mkdir(parents=True)
                (artifact / 'run_config.json').write_text('{}')
            roots[0] = external
            with self.assertRaisesRegex(ValueError, 'Canonical core root'):
                validate_canonical_core_roots(roots, repo)

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
