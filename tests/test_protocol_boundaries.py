from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from balanced_mse import get_lds_statistics
from downstream_train import eval_epoch, fixed_evaluation_rng
from rng_utils import (
    batch_fingerprint,
    resolve_stage_seeds,
    seed_all,
    split_fingerprint,
    state_dict_fingerprint,
)
from sampling import build_split_indices, dataset_sampling
from sgrl_train import (
    build_sgrl_training_state,
    embedding_cache_fingerprint,
    get_all_contrastive_embed,
    sgrl_cache_fingerprint,
)
from sram_dataset import SealSramDataset, adaption_for_sgrl


class ForbiddenTransferDataset:
    names = ['source', 'forbidden_transfer']

    def __init__(self):
        self.source = Data(
            node_type=torch.tensor([0, 1, 2]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_type=torch.tensor([0, 1]),
            node_attr=torch.ones(3, 2),
        )

    def __getitem__(self, index):
        if index != 0:
            raise AssertionError('Strict SGRL accessed a transfer graph.')
        return self.source


class NormalizerHarness:
    _ensure_collated_data = SealSramDataset._ensure_collated_data
    fit_node_feature_normalizer = SealSramDataset.fit_node_feature_normalizer
    apply_node_feature_normalizer = SealSramDataset.apply_node_feature_normalizer

    def __init__(self):
        self.source = Data(
            node_type=torch.tensor([0, 0, 1]),
            node_attr=torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]),
        )
        self.transfer = Data(
            node_type=torch.tensor([0, 1]),
            node_attr=torch.tensor([[100.0, 100.0], [200.0, 200.0]]),
        )
        self._data = Batch.from_data_list([self.source, self.transfer])
        self._data_list = None

    def __getitem__(self, index):
        return [self.source, self.transfer][index]


class TinyEdgeDataset:
    names = ['source', 'transfer']

    def __init__(self, source, transfer):
        self.graphs = [source, transfer]

    def __getitem__(self, index):
        return self.graphs[index]


def make_sgrl_args(**overrides):
    values = dict(
        protocol='strict_inductive',
        sgrl_graph_scope='source',
        seed=0,
        pretraining_seed=0,
        embedding_inference_seed=0,
        sgrl_pretrain_target_update='sgrl_dual_rsm_ema',
        cl_model='clustergcn',
        cl_gnn_layers=2,
        cl_hid_dim=4,
        cl_act_fn='tanh',
        cl_dropout=0.0,
        cl_batch_size=8,
        cl_num_neighbors=2,
        cl_epochs=1,
        use_bn=0,
        num_hops=2,
        momentum=0.99,
        e1_lr=1e-4,
        e2_lr=2e-5,
        weight_decay=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_edge_graph(name, num_nodes):
    nodes = torch.arange(num_nodes, dtype=torch.long)
    forward_edges = torch.stack((nodes, (nodes + 1) % num_nodes))
    edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)
    edge_label_index = forward_edges
    edge_label = torch.stack((
        torch.linspace(0.05, 0.95, num_nodes),
        (nodes % 5).float(),
    ), dim=1)
    return Data(
        name=name,
        x=torch.arange(num_nodes * 2, dtype=torch.float32).view(num_nodes, 2),
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )


class DeterministicEdgeRegressor(torch.nn.Module):
    def forward(self, batch):
        labels = batch.edge_label
        prediction = labels[:, 0] * 0.5 + 0.1
        return prediction, labels[:, 1].long(), labels


class IdentityEmbeddingModel(torch.nn.Module):
    def embed(self, batch, unused_num_hop):
        return batch.x[:, :2]


class ProtocolBoundaryTest(unittest.TestCase):
    def assert_numpy_state_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2:], second[2:])

    def downstream_signature(self, dataset, simulate_cache_miss):
        args = SimpleNamespace(
            seed=7,
            split_seed=19,
            downstream_seed=23,
            train_sampler_seed=29,
            eval_seed=31,
            task_level='edge',
            num_hops=1,
            num_neighbors=2,
            batch_size=4,
            num_workers=0,
        )
        seed_all(args.seed, include_cuda=False)
        if simulate_cache_miss:
            random.random()
            np.random.rand(17)
            torch.rand(dataset[1].num_nodes * 3)

        # The same boundary is used immediately before downstream_train.
        seed_all(args.downstream_seed, include_cuda=False)
        split_indices = build_split_indices(args, dataset)
        train_loader, _, _, _, _ = dataset_sampling(
            args, dataset, split_indices=split_indices
        )

        # downstream_train resets immediately before model construction too.
        seed_all(args.downstream_seed, include_cuda=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )
        model_fingerprint = state_dict_fingerprint(model.state_dict())
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        first_batch_fingerprint = batch_fingerprint(next(iter(train_loader)))
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_state_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        return (
            model_fingerprint,
            split_fingerprint(split_indices),
            first_batch_fingerprint,
        )

    def test_stage_seed_resolution_and_cache_identity(self):
        args = SimpleNamespace(
            seed=13,
            pretraining_seed=None,
            embedding_inference_seed=None,
            downstream_seed=None,
            train_sampler_seed=None,
            split_seed=None,
            eval_seed=0,
        )
        resolved = resolve_stage_seeds(args)
        for field in (
            'pretraining_seed',
            'embedding_inference_seed',
            'downstream_seed',
            'train_sampler_seed',
            'split_seed',
        ):
            self.assertEqual(resolved[field], 13)
        self.assertEqual(resolved['eval_seed'], 0)

        first = make_sgrl_args(seed=13, pretraining_seed=17)
        second = make_sgrl_args(seed=13, pretraining_seed=18)
        first_checkpoint = sgrl_cache_fingerprint(first, ['source'])
        second_checkpoint = sgrl_cache_fingerprint(second, ['source'])
        self.assertNotEqual(first_checkpoint, second_checkpoint)
        first.embedding_inference_seed = 23
        second_view = make_sgrl_args(
            seed=13,
            pretraining_seed=17,
            embedding_inference_seed=24,
        )
        self.assertNotEqual(
            embedding_cache_fingerprint(first, first_checkpoint),
            embedding_cache_fingerprint(second_view, first_checkpoint),
        )

    def test_cache_hit_and_miss_have_identical_downstream_signature(self):
        dataset = TinyEdgeDataset(
            make_edge_graph('source', 12),
            make_edge_graph('transfer', 8),
        )
        cache_hit = self.downstream_signature(
            dataset, simulate_cache_miss=False
        )
        cache_miss = self.downstream_signature(
            dataset, simulate_cache_miss=True
        )
        self.assertEqual(cache_hit, cache_miss)

    def test_transfer_graph_change_does_not_change_source_signature(self):
        source = make_edge_graph('source', 12)
        small_transfer = TinyEdgeDataset(
            source, make_edge_graph('transfer', 6)
        )
        large_transfer = TinyEdgeDataset(
            source, make_edge_graph('transfer', 18)
        )
        self.assertEqual(
            self.downstream_signature(
                small_transfer, simulate_cache_miss=True
            ),
            self.downstream_signature(
                large_transfer, simulate_cache_miss=True
            ),
        )

    def test_fixed_embedding_inference_repeats_and_restores_rng(self):
        graph = make_edge_graph('embedding_graph', 10)
        loader = NeighborLoader(
            graph,
            num_neighbors=[2],
            batch_size=3,
            shuffle=False,
            num_workers=0,
        )
        model = IdentityEmbeddingModel()
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = f'{temporary_dir}/identity.pt'
            torch.save(model.state_dict(), checkpoint_path)
            random.seed(101)
            np.random.seed(101)
            torch.manual_seed(101)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            results = [
                get_all_contrastive_embed(
                    model,
                    checkpoint_path,
                    graph,
                    loader,
                    hidden_dim=2,
                    num_hop=1,
                    device=torch.device('cpu'),
                    inference_seed=37,
                    return_sampler_fingerprint=True,
                )
                for _ in range(5)
            ]
        self.assertEqual(len({fingerprint for _, fingerprint in results}), 1)
        for embeddings, _ in results[1:]:
            self.assertTrue(torch.equal(embeddings, results[0][0]))
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_state_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_fixed_evaluation_and_embedding_contexts_are_cpu_only(self):
        graph = make_edge_graph('cpu_scope_graph', 6)
        loader = NeighborLoader(
            graph,
            num_neighbors=[2],
            batch_size=3,
            shuffle=False,
            num_workers=0,
        )
        model = IdentityEmbeddingModel()
        args = SimpleNamespace(eval_seed=41)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = f'{temporary_dir}/identity.pt'
            torch.save(model.state_dict(), checkpoint_path)
            original_fork_rng = torch.random.fork_rng
            with patch(
                    'rng_utils.torch.random.fork_rng',
                    wraps=original_fork_rng,
                ) as fork_rng, patch(
                    'rng_utils.torch.cuda.is_available'
                ) as cuda_available, patch(
                    'rng_utils.torch.cuda.device_count'
                ) as device_count, patch(
                    'rng_utils.torch.manual_seed'
                ) as torch_manual_seed, patch(
                    'rng_utils.torch.cuda.manual_seed_all'
                ) as manual_seed_all:
                with fixed_evaluation_rng(args, 'val'):
                    torch.rand(1)
                get_all_contrastive_embed(
                    model,
                    checkpoint_path,
                    graph,
                    loader,
                    hidden_dim=2,
                    num_hop=1,
                    device=torch.device('cpu'),
                    inference_seed=37,
                )

        self.assertEqual(fork_rng.call_count, 2)
        for fork_call in fork_rng.call_args_list:
            self.assertEqual(fork_call.kwargs, {'devices': []})
        cuda_available.assert_not_called()
        device_count.assert_not_called()
        torch_manual_seed.assert_not_called()
        manual_seed_all.assert_not_called()

    def test_real_link_neighbor_evaluation_is_fixed_for_five_runs(self):
        graph = make_edge_graph('evaluation_graph', 12)
        loader = LinkNeighborLoader(
            graph,
            num_neighbors=[2],
            edge_label_index=graph.edge_label_index,
            edge_label=graph.edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        args = SimpleNamespace(
            task='regression',
            regress_loss='mse',
            eval_seed=41,
        )
        model = DeterministicEdgeRegressor()
        criterion = torch.nn.MSELoss(reduction='mean')
        results = [
            eval_epoch(
                args,
                loader,
                model,
                torch.device('cpu'),
                split='val',
                criterion=criterion,
            )
            for _ in range(5)
        ]
        self.assertEqual(
            len({result['sampler_fingerprint'] for result in results}), 1
        )
        reference_mse = results[0]['mse_raw']
        for result in results[1:]:
            self.assertEqual(result['mse_raw'], reference_mse)

    def test_source_only_sgrl_graph_never_accesses_transfer(self):
        graph = adaption_for_sgrl(ForbiddenTransferDataset(), [0])
        self.assertEqual(graph.num_graphs, 1)
        self.assertEqual(graph.num_nodes, 3)

    def test_source_normalizer_ignores_transfer_extremes(self):
        dataset = NormalizerHarness()
        source_state = dataset.fit_node_feature_normalizer([0, 1], [0])
        all_state = dataset.fit_node_feature_normalizer([0, 1], None)

        self.assertTrue(torch.equal(
            source_state[0], torch.tensor([[2.0, 2.0]]),
        ))
        self.assertTrue(torch.equal(
            source_state[1], torch.tensor([[3.0, 4.0]]),
        ))
        self.assertGreater(all_state[0].max(), source_state[0].max())
        dataset.apply_node_feature_normalizer(source_state)
        self.assertTrue(torch.equal(source_state[0], torch.tensor([[2.0, 2.0]])))

    def test_split_is_seeded_and_lds_weights_reach_loader_labels(self):
        source = Data(
            edge_label_index=torch.tensor([
                list(range(10)), list(range(1, 11))
            ]),
            edge_label=torch.stack((
                torch.linspace(0.05, 0.95, 10),
                torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            ), dim=1),
        )
        transfer = Data(
            name='transfer',
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label=torch.tensor([[0.2, 1.0], [0.8, 4.0]]),
        )
        dataset = TinyEdgeDataset(source, transfer)
        args = SimpleNamespace(
            seed=7,
            split_seed=19,
            task_level='edge',
            num_hops=1,
            num_neighbors=2,
            batch_size=4,
            num_workers=0,
        )
        split_a = build_split_indices(args, dataset)
        split_b = build_split_indices(args, dataset)
        self.assertTrue(torch.equal(split_a['train'], split_b['train']))
        train_classes = source.edge_label[split_a['train'], 1]
        _, _, _, class_weights = get_lds_statistics(
            train_classes, 'gaussian', 5, 1.0, num_classes=5
        )

        def capture_loader(unused_graph, **kwargs):
            return SimpleNamespace(**kwargs)

        with patch('sampling.LinkNeighborLoader', side_effect=capture_loader):
            train_loader, val_loader, test_loaders, _, _ = dataset_sampling(
                args,
                dataset,
                split_indices=split_a,
                class_weights=class_weights,
            )
        self.assertEqual(train_loader.edge_label.size(1), 3)
        self.assertEqual(val_loader.edge_label.size(1), 3)
        self.assertEqual(test_loaders['transfer'].edge_label.size(1), 3)
        expected = class_weights[train_loader.edge_label[:, 1].long()]
        self.assertTrue(torch.allclose(train_loader.edge_label[:, 2], expected))

    def test_fixed_eval_rng_is_repeatable_and_rng_neutral(self):
        args = SimpleNamespace(eval_seed=123)
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        with fixed_evaluation_rng(args, 'val'):
            first = (random.random(), np.random.rand(), torch.rand(3))
        with fixed_evaluation_rng(args, 'val:best'):
            second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(random.getstate(), python_state)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_state[0])
        self.assertTrue(np.array_equal(current_numpy[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_target_update_modes_have_auditable_optimizer_sets(self):
        dual_args = make_sgrl_args()
        dual = build_sgrl_training_state(dual_args, torch.device('cpu'))
        dual_online, _, dual_online_optimizer, dual_target_optimizer = dual
        self.assertIsNotNone(dual_target_optimizer)
        online_ids = {
            id(parameter)
            for group in dual_online_optimizer.param_groups
            for parameter in group['params']
        }
        target_ids = {
            id(parameter)
            for group in dual_target_optimizer.param_groups
            for parameter in group['params']
        }
        self.assertTrue(online_ids.isdisjoint(target_ids))

        ema_args = make_sgrl_args(
            sgrl_pretrain_target_update='circuitgcl_text_ema_only'
        )
        ema_online, ema_target, _, ema_target_optimizer = \
            build_sgrl_training_state(ema_args, torch.device('cpu'))
        self.assertIsNone(ema_target_optimizer)
        for online_parameter, target_parameter in zip(
                ema_online.online_encoder.parameters(),
                ema_target.target_encoder.parameters()):
            self.assertTrue(torch.equal(online_parameter, target_parameter))
        self.assertNotEqual(
            sgrl_cache_fingerprint(dual_args, ['source']),
            sgrl_cache_fingerprint(ema_args, ['source']),
        )


if __name__ == '__main__':
    unittest.main()
