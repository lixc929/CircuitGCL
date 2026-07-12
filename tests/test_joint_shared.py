import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Data

from model import JointSharedGraphHead, MergeableLoRALinear
from downstream_train import (
    build_joint_gradient_audit_context,
    build_joint_shared_audit_context,
    finalize_joint_gradient_audit,
    record_joint_gradient_audit,
    record_joint_shared_representation_audit,
)


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
        'joint_lora_rank': 0,
        'joint_lora_alpha': None,
        'joint_lora_layer': -1,
        'task': 'regression',
        'task_level': 'edge',
        'net_only': True,
        'src_dst_agg': 'concat',
        'num_classes': 5,
        'class_boundaries': [0.2, 0.4, 0.6, 0.8],
        'num_head_layers': 2,
        'act_fn': 'prelu',
        'use_sgrl_joint_shared': 1,
        'joint_shared_audit': 0,
        'joint_shared_audit_interval': 5,
        'joint_gradient_audit': 0,
        'joint_gradient_audit_interval': 10,
        'joint_gcl_lambda': 0.0,
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

    def make_lora_model(self, rank=2):
        model = JointSharedGraphHead(make_args(joint_lora_rank=rank))
        encoder_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in model.shared_backbone.gnn.state_dict().items()
        }
        model.load_online_encoder_state(encoder_state)
        model.eval()
        return model

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

    def test_lora_zero_init_and_merge_preserve_task_output(self):
        model = self.make_lora_model(rank=2)
        batch = make_batch()
        with torch.no_grad():
            base_x = model.shared_backbone(batch, task_path=False)
            initial_task_x = model.shared_backbone(batch, task_path=True)
        self.assertTrue(torch.allclose(base_x, initial_task_x, atol=1e-7))

        lora_modules = [
            module
            for module in model.shared_backbone.modules()
            if isinstance(module, MergeableLoRALinear)
        ]
        self.assertEqual(len(lora_modules), 2)
        with torch.no_grad():
            for module in lora_modules:
                module.lora_b.weight.fill_(0.05)
            task_before_merge = model.shared_backbone(
                batch,
                task_path=True,
            )
            base_after_update = model.shared_backbone(
                batch,
                task_path=False,
            )
        self.assertFalse(torch.allclose(task_before_merge, base_after_update))

        unmerged = model.deployment_parameter_count()
        expected_lora_values = model.shared_backbone.lora_parameter_count()
        self.assertEqual(
            unmerged - model.merged_deployment_parameter_count(),
            expected_lora_values,
        )
        merged_values = model.merge_task_lora_for_deployment()
        with torch.no_grad():
            task_after_merge = model.shared_backbone(
                batch,
                task_path=True,
            )
        self.assertEqual(merged_values, expected_lora_values)
        self.assertTrue(torch.allclose(
            task_before_merge,
            task_after_merge,
            atol=1e-6,
        ))
        self.assertFalse(any(
            isinstance(module, MergeableLoRALinear)
            for module in model.shared_backbone.modules()
        ))

    def test_lora_gcl_base_path_excludes_stats_and_task_delta(self):
        model = self.make_lora_model(rank=2)
        first = make_batch(torch.randn(6, 17))
        second = make_batch(torch.randn(6, 17) * 20.0)
        with torch.no_grad():
            model.shared_backbone.stats_residual_scale.fill_(1.0)
            for module in model.shared_backbone._lora_modules():
                module.lora_b.weight.fill_(0.1)
            first_base = model.shared_backbone(first, task_path=False)
            second_base = model.shared_backbone(second, task_path=False)
            first_task = model.shared_backbone(first, task_path=True)
            second_task = model.shared_backbone(second, task_path=True)

        self.assertTrue(torch.allclose(first_base, second_base, atol=1e-7))
        self.assertFalse(torch.allclose(first_task, second_task))
        self.assertFalse(torch.allclose(first_task, first_base))

    def test_joint_audit_defers_transfer_until_best_checkpoint(self):
        model = self.make_lora_model(rank=2)
        args = make_args(
            joint_lora_rank=2,
            joint_shared_audit=1,
            joint_shared_audit_interval=2,
        )
        source = make_batch()
        transfer = make_batch()
        with tempfile.TemporaryDirectory() as temporary_dir:
            args.run_artifact_dir = temporary_dir
            context = build_joint_shared_audit_context(
                args,
                model,
                [source],
                {'transfer': [transfer]},
                torch.device('cpu'),
            )
            initial = record_joint_shared_representation_audit(
                args, model, context, epoch=-1
            )
            with torch.no_grad():
                next(model.shared_backbone.gnn.parameters()).add_(0.01)
                for module in model.shared_backbone._lora_modules():
                    module.lora_b.weight.fill_(0.02)
            skipped = record_joint_shared_representation_audit(
                args, model, context, epoch=1
            )
            trained = record_joint_shared_representation_audit(
                args, model, context, epoch=2
            )
            final = record_joint_shared_representation_audit(
                args,
                model,
                context,
                epoch=2,
                phase='best',
                force=True,
            )

            self.assertEqual([row['split'] for row in initial], ['source_val'])
            self.assertEqual(skipped, [])
            self.assertEqual([row['split'] for row in trained], ['source_val'])
            self.assertEqual(
                {row['split'] for row in final},
                {'source_val', 'transfer'},
            )
            self.assertGreater(trained[0]['base_weight_relative_l2'], 0.0)
            self.assertGreater(trained[0]['lora_relative_delta'], 0.0)
            audit_path = Path(temporary_dir) / (
                'joint_shared_representation_audit.jsonl'
            )
            self.assertEqual(len(audit_path.read_text().splitlines()), 4)

    def test_gradient_audit_does_not_change_backward_gradients(self):
        audited_model = copy.deepcopy(self.model)
        reference_model = copy.deepcopy(self.model)
        args = make_args(
            joint_gradient_audit=1,
            joint_gradient_audit_interval=2,
            joint_gcl_lambda=0.05,
        )
        batch = make_batch()

        def losses(model):
            prediction, _, labels, online_x = model(
                batch,
                return_backbone=True,
            )
            supervised = torch.nn.functional.mse_loss(
                prediction.view(-1),
                labels[:, 0],
            )
            return supervised, model.gcl_loss(batch, online_x=online_x)

        with tempfile.TemporaryDirectory() as temporary_dir:
            args.run_artifact_dir = temporary_dir
            context = build_joint_gradient_audit_context(
                args,
                audited_model,
            )
            supervised, gcl = losses(audited_model)
            skipped = record_joint_gradient_audit(
                args,
                context,
                supervised,
                gcl,
                epoch=1,
                batch_index=0,
            )
            record = record_joint_gradient_audit(
                args,
                context,
                supervised,
                gcl,
                epoch=2,
                batch_index=0,
            )

            self.assertIsNone(skipped)
            self.assertIsNotNone(record)
            self.assertIsNotNone(record['global']['cosine'])
            self.assertTrue(all(
                parameter.grad is None
                for parameter in audited_model.parameters()
            ))

            (supervised + args.joint_gcl_lambda * gcl).backward()
            reference_supervised, reference_gcl = losses(reference_model)
            (
                reference_supervised
                + args.joint_gcl_lambda * reference_gcl
            ).backward()
            for audited_parameter, reference_parameter in zip(
                audited_model.parameters(),
                reference_model.parameters(),
            ):
                if audited_parameter.grad is None:
                    self.assertIsNone(reference_parameter.grad)
                else:
                    self.assertTrue(torch.allclose(
                        audited_parameter.grad,
                        reference_parameter.grad,
                        atol=1e-7,
                    ))

            summary = finalize_joint_gradient_audit(context)
            self.assertEqual(summary['record_count'], 1)
            self.assertEqual(summary['global_cosine']['count'], 1)
            self.assertEqual(
                len((Path(temporary_dir) / (
                    'joint_shared_gradient_audit.jsonl'
                )).read_text().splitlines()),
                1,
            )
            self.assertTrue((Path(temporary_dir) / (
                'joint_shared_gradient_summary.json'
            )).exists())

        self.assertIsNone(build_joint_gradient_audit_context(
            make_args(),
            self.model,
        ))


if __name__ == '__main__':
    unittest.main()
