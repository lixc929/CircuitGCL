import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from processed_cache import (
    PROCESSED_CACHE_SCHEMA_VERSION,
    build_processed_cache_inputs,
    load_validated_processed_cache,
    processed_cache_filename,
    processed_cache_key,
    processed_cache_manifest_path,
    publish_processed_cache_manifest,
    validate_processed_cache,
)
from rng_utils import resolve_stage_seeds
from run_artifacts import file_sha256, write_json_atomic
from sram_dataset import SealSramDataset


def make_inputs(**overrides):
    values = dict(
        graph_name='tiny',
        raw_sha256='a' * 64,
        sample_rate=0.5,
        negative_edge_ratio=0.0,
        to_undirected=True,
        task_level='edge',
        net_only=True,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
        relation_sample_seed=17,
        graph_relation_sample_seed=23,
    )
    values.update(overrides)
    return build_processed_cache_inputs(**values)


def make_relation_graph(edges_per_relation=12):
    edge_ids = torch.arange(edges_per_relation * 3)
    return Data(
        tar_edge_index=torch.stack((edge_ids, edge_ids + 1)),
        tar_edge_type=torch.repeat_interleave(
            torch.tensor([2, 3, 4]), edges_per_relation
        ),
        tar_edge_y=torch.linspace(1e-20, 9e-16, edge_ids.numel()),
        tar_edge_dist=torch.full((3,), edges_per_relation, dtype=torch.long),
        num_nodes=int(edge_ids[-1]) + 2,
    )


class TinySealSramDataset(SealSramDataset):
    process_calls = 0

    def single_g_process(self, idx):
        type(self).process_calls += 1
        graph = Data(
            name=self.names[idx],
            x=torch.zeros((3, 1)),
            node_type=torch.tensor([0, 1, 2]),
            node_attr=torch.ones((3, 2)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_type=torch.tensor([0, 1]),
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label=torch.tensor([0.25, 0.75]),
        )
        torch.save((graph, None), self.processed_paths[idx])
        publish_processed_cache_manifest(
            self.processed_paths[idx],
            self._processed_cache_inputs(idx),
        )
        return graph.edge_label.numel()


class ProcessedCacheIdentityTest(unittest.TestCase):
    def assert_numpy_state_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2:], second[2:])

    def test_relation_seed_resolves_from_main_seed(self):
        args = SimpleNamespace(seed=31)
        resolved = resolve_stage_seeds(args)
        self.assertEqual(args.relation_sample_seed, 31)
        self.assertEqual(resolved['relation_sample_seed'], 31)

    def test_manifest_fields_have_collision_resistant_identity(self):
        base = make_inputs()
        self.assertEqual(base['cache_schema_version'], PROCESSED_CACHE_SCHEMA_VERSION)
        self.assertEqual(processed_cache_key(base), processed_cache_key(copy.deepcopy(base)))

        changed_values = {
            'cache_schema_version': 2,
            'graph_name': 'other',
            'raw_sha256': 'b' * 64,
            'sample_rate': 0.25,
            'negative_edge_ratio': 0.04,
            'to_undirected': False,
            'task_level': 'node',
            'net_only': False,
            'class_boundaries': [0.1, 0.3, 0.7, 0.9],
            'relation_sample_seed': 18,
            'graph_relation_sample_seed': 24,
        }
        for field, value in changed_values.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(base)
                changed[field] = value
                self.assertNotEqual(
                    processed_cache_key(base), processed_cache_key(changed)
                )

        changed_distribution = copy.deepcopy(base)
        changed_distribution['sampling_distribution']['version'] = 2
        self.assertNotEqual(
            processed_cache_key(base),
            processed_cache_key(changed_distribution),
        )

        self.assertNotEqual(
            processed_cache_filename('tiny', make_inputs(negative_edge_ratio=0.0)),
            processed_cache_filename('tiny', make_inputs(negative_edge_ratio=0.04)),
        )
        self.assertNotEqual(
            processed_cache_filename('tiny', make_inputs(to_undirected=True)),
            processed_cache_filename('tiny', make_inputs(to_undirected=False)),
        )
        self.assertNotEqual(
            processed_cache_filename('tiny', make_inputs(relation_sample_seed=17)),
            processed_cache_filename('tiny', make_inputs(relation_sample_seed=18)),
        )

    def test_raw_content_changes_identity_and_is_hashed_once(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_dir = root / 'sram' / 'raw'
            raw_dir.mkdir(parents=True)
            raw_path = raw_dir / 'tiny.pt'
            raw_path.write_bytes(b'first raw graph')
            first_hash = file_sha256(raw_path)
            raw_path.write_bytes(b'second raw graph')
            second_hash = file_sha256(raw_path)
            self.assertNotEqual(
                processed_cache_key(make_inputs(raw_sha256=first_hash)),
                processed_cache_key(make_inputs(raw_sha256=second_hash)),
            )

            dataset = object.__new__(SealSramDataset)
            dataset.folder = str(root / 'sram')
            dataset.names = ['tiny']
            dataset.sample_rates = [0.5]
            dataset.neg_edge_ratio = 0.0
            dataset.to_undirected = True
            dataset.task_level = 'edge'
            dataset.net_only = True
            dataset.class_boundaries_config = [0.2, 0.4, 0.6, 0.8]
            dataset.relation_sample_seed = 17
            dataset._raw_sha256_cache = {}
            dataset._processed_inputs_cache = {}
            with patch(
                    'sram_dataset.file_sha256', wraps=file_sha256
                ) as hash_file:
                dataset._processed_cache_inputs(0)
                dataset._processed_cache_inputs(0)
            self.assertEqual(hash_file.call_count, 1)

    def test_validated_hit_and_provenance(self):
        inputs = make_inputs()
        with tempfile.TemporaryDirectory() as temporary_dir:
            processed_path = Path(temporary_dir) / processed_cache_filename(
                'tiny', inputs
            )
            torch.save((torch.tensor([1, 2, 3]), None), processed_path)
            provenance = publish_processed_cache_manifest(
                processed_path, inputs
            )
            (payload, slices), loaded_provenance = \
                load_validated_processed_cache(processed_path, inputs)

        self.assertTrue(torch.equal(payload, torch.tensor([1, 2, 3])))
        self.assertIsNone(slices)
        self.assertEqual(provenance, loaded_provenance)
        for field in (
                'processed_path', 'cache_key', 'cache_schema_version',
                'manifest_path', 'manifest_sha256', 'raw_sha256',
                'processed_sha256', 'sampling_distribution',
                'relation_sample_seed', 'graph_relation_sample_seed'):
            self.assertIn(field, provenance)

    def test_invalid_cache_is_rejected_before_torch_load(self):
        inputs = make_inputs()
        with tempfile.TemporaryDirectory() as temporary_dir:
            processed_path = Path(temporary_dir) / processed_cache_filename(
                'tiny', inputs
            )
            processed_path.write_bytes(b'processed')
            with patch('processed_cache.torch.load') as trusted_load:
                with self.assertRaisesRegex(RuntimeError, 'manifest is missing'):
                    load_validated_processed_cache(processed_path, inputs)
            trusted_load.assert_not_called()

            manifest_path = Path(
                processed_cache_manifest_path(processed_path)
            )
            manifest_path.write_text('{bad json', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'malformed sidecar'):
                validate_processed_cache(processed_path, inputs)

            mismatched = make_inputs(raw_sha256='b' * 64)
            write_json_atomic(manifest_path, {
                'cache_key': processed_cache_key(mismatched),
                'inputs': mismatched,
                'processed_sha256': file_sha256(processed_path),
            })
            with self.assertRaisesRegex(RuntimeError, 'cache key mismatch'):
                validate_processed_cache(processed_path, inputs)

            write_json_atomic(manifest_path, {
                'cache_key': processed_cache_key(inputs),
                'inputs': inputs,
                'processed_sha256': file_sha256(processed_path),
            })
            processed_path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'processed SHA256 mismatch'):
                validate_processed_cache(processed_path, inputs)

    def test_legacy_name_is_not_an_authoritative_hit(self):
        inputs = make_inputs(negative_edge_ratio=0.04)
        legacy_name = 'tiny_nr0.0_processed.pt'
        fingerprinted_name = processed_cache_filename('tiny', inputs)
        self.assertNotEqual(legacy_name, fingerprinted_name)
        with tempfile.TemporaryDirectory() as temporary_dir:
            legacy_path = Path(temporary_dir) / legacy_name
            legacy_path.write_bytes(b'legacy cache')
            with self.assertRaisesRegex(RuntimeError, 'manifest is missing'):
                validate_processed_cache(legacy_path, inputs)
            self.assertEqual(legacy_path.read_bytes(), b'legacy cache')

    def test_dataset_lifecycle_reuses_only_validated_fingerprinted_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / 'datasets'
            raw_dir = root / 'sram' / 'raw'
            processed_dir = root / 'sram' / 'processed_for_edges'
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            (raw_dir / 'tiny.pt').write_bytes(b'tiny raw fixture')
            legacy_path = processed_dir / 'tiny_nr0.0_processed.pt'
            legacy_path.write_bytes(b'legacy remains immutable')

            TinySealSramDataset.process_calls = 0
            first = TinySealSramDataset(
                name='tiny',
                root=str(root),
                neg_edge_ratio=0.04,
                to_undirected=True,
                sample_rates=[0.5],
                task_level='edge',
                net_only=True,
                class_boundaries=[0.2, 0.4, 0.6, 0.8],
                relation_sample_seed=17,
            )
            first_path = Path(first.processed_cache_provenance[0][
                'processed_path'
            ])
            self.assertEqual(TinySealSramDataset.process_calls, 1)
            self.assertTrue(first_path.is_file())
            self.assertTrue(Path(
                processed_cache_manifest_path(first_path)
            ).is_file())
            self.assertNotEqual(first_path.name, legacy_path.name)

            second = TinySealSramDataset(
                name='tiny',
                root=str(root),
                neg_edge_ratio=0.04,
                to_undirected=True,
                sample_rates=[0.5],
                task_level='edge',
                net_only=True,
                class_boundaries=[0.2, 0.4, 0.6, 0.8],
                relation_sample_seed=17,
            )
            self.assertEqual(TinySealSramDataset.process_calls, 1)
            self.assertEqual(
                first.processed_cache_provenance,
                second.processed_cache_provenance,
            )
            self.assertEqual(
                legacy_path.read_bytes(), b'legacy remains immutable'
            )

    def test_relation_sampling_is_deterministic_and_rng_neutral(self):
        dataset = object.__new__(SealSramDataset)
        dataset.names = ['tiny']
        dataset.sample_rates = [0.5]
        dataset.neg_edge_ratio = 0.0
        dataset.relation_sample_seed = 17
        graph = make_relation_graph()

        random.seed(101)
        np.random.seed(101)
        torch.random.default_generator.manual_seed(101)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        first = dataset._sample_relation_edges(graph, 0)
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_state_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

        second = dataset._sample_relation_edges(graph, 0)
        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))

        dataset.relation_sample_seed = 1017
        different = dataset._sample_relation_edges(graph, 0)
        self.assertFalse(torch.equal(first[0], different[0]))


if __name__ == '__main__':
    unittest.main()
