import copy
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Data

from model import JointSharedGraphHead


def make_args(**overrides):
    values = {
        'cl_hid_dim': 8,
        'joint_shared_gnn_layers': 2,
        'shared_gnn_layers': 1,
        'cl_gnn_layers': 2,
        'cl_model': 'clustergcn',
        'cl_act_fn': 'tanh',
        'cl_dropout': 0.0,
        'use_bn': 0,
        'use_stats': 1,
        'momentum': 0.9,
        'task': 'regression',
        'task_level': 'edge',
        'net_only': True,
        'src_dst_agg': 'concat',
        'num_classes': 5,
        'class_boundaries': [0.2, 0.4, 0.6, 0.8],
        'num_head_layers': 2,
        'act_fn': 'prelu',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_batch(node_attr=None):
    if node_attr is None:
        node_attr = torch.randn(6, 17)
    node_attr = node_attr.clone()
    node_attr[torch.tensor([2, 5]), 0] = torch.tensor([1.0, 2.0])
    return Data(
        node_type=torch.tensor([0, 1, 2, 0, 1, 2]),
        node_attr=node_attr,
        edge_index=torch.tensor([
            [0, 1, 2, 3, 4, 5, 0, 3],
            [1, 2, 0, 4, 5, 3, 3, 0],
        ]),
        edge_type=torch.zeros(8, dtype=torch.long),
        edge_label=torch.tensor([[0.2, 1.0], [0.7, 3.0]]),
    )


class JointSharedTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = JointSharedGraphHead(make_args())
        encoder_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in self.model.shared_backbone.gnn.state_dict().items()
        }
        self.model.load_online_encoder_state(encoder_state)
        self.model.eval()

    def test_zero_init_stats_residual_preserves_pretrained_path(self):
        first = make_batch(torch.randn(6, 17))
        second = make_batch(torch.randn(6, 17) * 100.0)
        with torch.no_grad():
            first_x = self.model.shared_backbone(first)
            second_x = self.model.shared_backbone(second)
        self.assertTrue(torch.equal(
            self.model.shared_backbone.stats_residual_scale,
            torch.zeros(1),
        ))
        self.assertTrue(torch.allclose(first_x, second_x, atol=1e-7))

    def test_joint_loss_updates_online_but_not_target_gradients(self):
        batch = make_batch()
        target_before = copy.deepcopy(self.model.target_backbone.state_dict())
        prediction, _, labels, online_x = self.model(
            batch,
            return_backbone=True,
        )
        supervised = torch.nn.functional.mse_loss(
            prediction.view(-1),
            labels[:, 0],
        )
        gcl = self.model.gcl_loss(batch, online_x=online_x)
        (supervised + 0.1 * gcl).backward()

        self.assertTrue(any(
            parameter.grad is not None
            for parameter in self.model.shared_backbone.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in self.model.target_backbone.parameters()
        ))
        self.assertLess(
            self.model.deployment_parameter_count(),
            sum(parameter.numel() for parameter in self.model.parameters()),
        )

        with torch.no_grad():
            next(self.model.shared_backbone.parameters()).add_(1.0)
        self.model.update_target_backbone()
        target_after = self.model.target_backbone.state_dict()
        self.assertTrue(any(
            not torch.equal(target_before[key], target_after[key])
            for key in target_before
            if target_before[key].is_floating_point()
        ))


if __name__ == '__main__':
    unittest.main()
