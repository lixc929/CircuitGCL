from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from balanced_mse import get_lds_statistics
from downstream_train import fixed_evaluation_rng
from sampling import build_split_indices, dataset_sampling
from sgrl_train import build_sgrl_training_state, sgrl_cache_fingerprint
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


class ProtocolBoundaryTest(unittest.TestCase):
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
