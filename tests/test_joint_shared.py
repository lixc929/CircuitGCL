import copy
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from model import JointSharedGraphHead, MergeableLoRALinear
from sgrl_models import CustomConv
from downstream_train import (
    build_joint_pcgrad_context,
    build_joint_gradient_audit_context,
    build_joint_shared_audit_context,
    finalize_joint_gradient_audit,
    finalize_joint_pcgrad,
    joint_pcgrad_backward,
    joint_gcl_lambda_at_epoch,
    record_joint_gradient_audit,
    record_joint_pcgrad_epoch,
    record_joint_shared_representation_audit,
    validate_joint_gcl_schedule,
    validate_joint_gradient_strategy,
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
        'joint_gradient_strategy': 'none',
        'joint_gcl_lambda': 0.0,
        'joint_gcl_lambda_schedule': 'constant',
        'joint_gcl_lambda_final': None,
        'epochs': 160,
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

    def test_rank0_sage_constructs_and_runs_joint_forward(self):
        model = JointSharedGraphHead(make_args(cl_model='sage'))
        model.eval()
        batch = make_batch()

        self.assertEqual(model.shared_backbone.lora_rank, 0)
        self.assertTrue(all(
            isinstance(layer, SAGEConv)
            for layer in model.shared_backbone.gnn.layers
        ))
        with torch.no_grad():
            prediction, true_class, labels, online_x = model(
                batch,
                return_backbone=True,
            )
            gcl_loss = model.gcl_loss(batch, online_x=online_x)

        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertEqual(tuple(true_class.shape), (2,))
        self.assertEqual(tuple(labels.shape), (2, 2))
        self.assertEqual(tuple(online_x.shape), (6, 8))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(gcl_loss))

    def test_sage_pretraining_checkpoint_loads_online_and_target_state(self):
        args = make_args(cl_model='sage')
        pretrained_encoder = CustomConv(args)
        pretrained_state = pretrained_encoder.state_dict()
        with torch.no_grad():
            for index, tensor in enumerate(pretrained_state.values(), start=1):
                tensor.fill_(index / 100.0 if tensor.is_floating_point() else index)
        checkpoint_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in pretrained_state.items()
        }
        model = JointSharedGraphHead(args)

        model.load_online_encoder_state(checkpoint_state)

        online_state = model.shared_backbone.gnn.state_dict()
        target_state = model.target_backbone.gnn.state_dict()
        for key, value in online_state.items():
            self.assertTrue(torch.equal(value, pretrained_state[key]), key)
            self.assertTrue(torch.equal(target_state[key], value), key)

    def test_rank0_sage_and_clustergcn_have_equal_parameter_counts(self):
        cluster_model = JointSharedGraphHead(make_args(cl_model='clustergcn'))
        sage_model = JointSharedGraphHead(make_args(cl_model='sage'))

        cluster_gnn_parameters = sum(
            parameter.numel()
            for parameter in cluster_model.shared_backbone.gnn.parameters()
        )
        sage_gnn_parameters = sum(
            parameter.numel()
            for parameter in sage_model.shared_backbone.gnn.parameters()
        )
        self.assertEqual(sage_gnn_parameters, cluster_gnn_parameters)
        self.assertEqual(
            sage_model.deployment_parameter_count(),
            cluster_model.deployment_parameter_count(),
        )

    def test_nonzero_lora_rejects_sage_checkpoint_load(self):
        args = make_args(cl_model='sage', joint_lora_rank=2)
        pretrained_encoder = CustomConv(args)
        checkpoint_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in pretrained_encoder.state_dict().items()
        }
        model = JointSharedGraphHead(args)

        with self.assertRaisesRegex(ValueError, 'supports clustergcn only'):
            model.load_online_encoder_state(checkpoint_state)

    def make_lora_model(self, rank=2):
        model = JointSharedGraphHead(make_args(joint_lora_rank=rank))
        encoder_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in model.shared_backbone.gnn.state_dict().items()
        }
        model.load_online_encoder_state(encoder_state)
        model.eval()
        return model

    def test_joint_gcl_lambda_schedule_endpoints_and_validation(self):
        constant = make_args(joint_gcl_lambda=0.05)
        validate_joint_gcl_schedule(constant)
        self.assertEqual(joint_gcl_lambda_at_epoch(constant, 0), 0.05)
        self.assertEqual(joint_gcl_lambda_at_epoch(constant, 159), 0.05)

        linear = make_args(
            joint_gcl_lambda=0.05,
            joint_gcl_lambda_schedule='linear',
            joint_gcl_lambda_final=0.005,
        )
        validate_joint_gcl_schedule(linear)
        self.assertEqual(joint_gcl_lambda_at_epoch(linear, 0), 0.05)
        self.assertEqual(joint_gcl_lambda_at_epoch(linear, 159), 0.005)
        self.assertAlmostEqual(
            joint_gcl_lambda_at_epoch(linear, 80),
            0.05 + (80 / 159) * (0.005 - 0.05),
        )
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        joint_gcl_lambda_at_epoch(linear, 80)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        with self.assertRaisesRegex(ValueError, 'outside'):
            joint_gcl_lambda_at_epoch(linear, 160)

        invalid = (
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='constant',
                joint_gcl_lambda_final=0.005,
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=None,
            ),
            make_args(
                joint_gcl_lambda=float('nan'),
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=float('inf'),
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=-0.001,
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=0.005,
                epochs=1,
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=0.06,
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=0.005,
                use_sgrl_joint_shared=0,
            ),
        )
        for args in invalid:
            with self.assertRaises(ValueError):
                validate_joint_gcl_schedule(args)

    def test_pcgrad_validation_and_deterministic_projection(self):
        valid = make_args(
            joint_gcl_lambda=0.05,
            joint_gradient_strategy='pcgrad',
        )
        validate_joint_gradient_strategy(valid)
        invalid = (
            make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                use_sgrl_joint_shared=0,
            ),
            make_args(joint_gradient_strategy='pcgrad'),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                joint_gcl_lambda_schedule='linear',
                joint_gcl_lambda_final=0.005,
            ),
            make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                joint_lora_rank=2,
            ),
        )
        for args in invalid:
            with self.assertRaises(ValueError):
                validate_joint_gradient_strategy(args)

        shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        supervised_only = torch.nn.Parameter(torch.tensor(1.0))
        gcl_only = torch.nn.Parameter(torch.tensor(1.0))
        context = {'named_parameters': (('shared', shared),)}
        supervised = shared[0] + 2.0 * supervised_only
        gcl = -shared[0] + shared[1] + 3.0 * gcl_only
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        stats = joint_pcgrad_backward(context, supervised, gcl, 0.5)
        self.assertTrue(stats['projection_applied'])
        self.assertAlmostEqual(stats['raw_cosine'], -(0.5 ** 0.5))
        self.assertTrue(torch.allclose(
            shared.grad,
            torch.tensor([0.5, 1.0]),
            atol=1e-7,
        ))
        self.assertAlmostEqual(supervised_only.grad.item(), 2.0)
        self.assertAlmostEqual(gcl_only.grad.item(), 1.5)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

        shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        supervised_only = torch.nn.Parameter(torch.tensor(1.0))
        gcl_only = torch.nn.Parameter(torch.tensor(1.0))
        context = {'named_parameters': (('shared', shared),)}
        supervised = shared[0] + 2.0 * supervised_only
        gcl = shared[0] + shared[1] + 3.0 * gcl_only
        stats = joint_pcgrad_backward(context, supervised, gcl, 0.5)
        self.assertFalse(stats['projection_applied'])
        self.assertTrue(torch.allclose(
            shared.grad,
            torch.tensor([1.5, 0.5]),
            atol=1e-7,
        ))
        self.assertAlmostEqual(supervised_only.grad.item(), 2.0)
        self.assertAlmostEqual(gcl_only.grad.item(), 1.5)

        shared = torch.nn.Parameter(torch.tensor(1.0))
        supervised_exclusive = torch.nn.Parameter(torch.tensor(1.0))
        context = {
            'named_parameters': (
                ('shared', shared),
                ('supervised_exclusive', supervised_exclusive),
            )
        }
        supervised = shared + supervised_exclusive
        gcl = -shared
        with self.assertRaisesRegex(RuntimeError, 'task-exclusive'):
            joint_pcgrad_backward(context, supervised, gcl, 0.5)

        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(1.0))
        unused = torch.nn.Parameter(torch.tensor(1.0))
        context = {
            'named_parameters': (
                ('first', first),
                ('second', second),
                ('unused', unused),
            )
        }
        stats = joint_pcgrad_backward(
            context,
            first + second,
            first - 2.0 * second,
            1.0,
        )
        self.assertTrue(stats['projection_applied'])
        self.assertAlmostEqual(first.grad.item(), 2.7, places=6)
        self.assertAlmostEqual(second.grad.item(), -0.9, places=6)
        self.assertIsNone(unused.grad)

        shared = torch.nn.Parameter(torch.tensor(1.0))
        context = {'named_parameters': (('shared', shared),)}
        stats = joint_pcgrad_backward(
            context,
            shared * 0.0,
            shared,
            1.0,
        )
        self.assertFalse(stats['projection_applied'])
        self.assertIsNone(stats['raw_cosine'])
        self.assertAlmostEqual(shared.grad.item(), 1.0)

    def test_pcgrad_epoch_audit_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                run_artifact_dir=temporary_dir,
                epochs=2,
            )
            model = SimpleNamespace(
                shared_backbone=SimpleNamespace(
                    gnn=torch.nn.Linear(2, 2, bias=False)
                )
            )
            context = build_joint_pcgrad_context(args, model)
            projected = {
                'raw_cosine': -0.5,
                'projection_applied': True,
                'standard_combined_norm': 1.0,
                'projected_combined_norm': 1.2,
                'projected_to_standard_norm': 1.2,
            }
            aligned = {
                'raw_cosine': 0.25,
                'projection_applied': False,
                'standard_combined_norm': 2.0,
                'projected_combined_norm': 2.0,
                'projected_to_standard_norm': 1.0,
            }
            record_joint_pcgrad_epoch(
                args, context, 0, [projected, aligned]
            )
            record_joint_pcgrad_epoch(args, context, 1, [projected])
            summary = finalize_joint_pcgrad(context)
            self.assertEqual(summary['epoch_count'], 2)
            self.assertEqual(summary['batch_count'], 3)
            self.assertEqual(summary['conflict_count'], 2)
            self.assertAlmostEqual(summary['conflict_fraction'], 2 / 3)
            self.assertEqual(
                len(Path(context['path']).read_text().splitlines()),
                2,
            )
            self.assertTrue(Path(context['summary_path']).is_file())

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

    def test_pcgrad_accepts_real_rank0_joint_scope(self):
        batch = make_batch()
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                run_artifact_dir=temporary_dir,
            )
            context = build_joint_pcgrad_context(args, self.model)
            prediction, _, labels, online_x = self.model(
                batch,
                return_backbone=True,
            )
            supervised = torch.nn.functional.mse_loss(
                prediction.view(-1),
                labels[:, 0],
            )
            gcl = self.model.gcl_loss(batch, online_x=online_x)
            stats = joint_pcgrad_backward(
                context,
                supervised,
                gcl,
                0.05,
            )
            self.assertTrue(math.isfinite(stats['raw_cosine']))
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in self.model.shared_backbone.gnn.parameters()
            ))
            self.assertTrue(all(
                parameter.grad is None
                for parameter in self.model.target_backbone.parameters()
            ))

    def test_pcgrad_real_model_forced_conflict_step_and_ema(self):
        batch = make_batch()
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = make_args(
                joint_gcl_lambda=0.05,
                joint_gradient_strategy='pcgrad',
                run_artifact_dir=temporary_dir,
            )
            context = build_joint_pcgrad_context(args, self.model)
            online_x = self.model.shared_backbone(batch, task_path=False)
            base_loss = online_x.square().mean()
            perturbation = 1e-3 * online_x.mean()
            head_parameter = next(self.model.head_layers.parameters())
            predictor_parameter = next(self.model.predictor.parameters())
            supervised = base_loss + 2.0 * head_parameter.view(-1)[0]
            gcl = (
                -base_loss
                + perturbation
                + 3.0 * predictor_parameter.view(-1)[0]
            )
            target_before = copy.deepcopy(
                self.model.target_backbone.state_dict()
            )
            shared_before = copy.deepcopy(
                self.model.shared_backbone.gnn.state_dict()
            )
            optimizer = torch.optim.SGD(
                [
                    parameter for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                lr=0.01,
            )
            optimizer.zero_grad()
            stats = joint_pcgrad_backward(
                context,
                supervised,
                gcl,
                0.05,
            )
            self.assertTrue(stats['projection_applied'])
            self.assertAlmostEqual(
                head_parameter.grad.view(-1)[0].item(),
                2.0,
                places=6,
            )
            self.assertAlmostEqual(
                predictor_parameter.grad.view(-1)[0].item(),
                0.15,
                places=6,
            )
            self.assertTrue(all(
                parameter.grad is None
                for parameter in self.model.target_backbone.parameters()
            ))
            optimizer.step()
            shared_after = self.model.shared_backbone.gnn.state_dict()
            self.assertTrue(any(
                not torch.equal(shared_before[key], shared_after[key])
                for key in shared_before
                if shared_before[key].is_floating_point()
            ))
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
            joint_gcl_lambda_schedule='linear',
            joint_gcl_lambda_final=0.005,
            epochs=4,
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
            effective_lambda = joint_gcl_lambda_at_epoch(args, 2)
            self.assertAlmostEqual(
                record['joint_gcl_lambda'], effective_lambda
            )
            self.assertTrue(all(
                parameter.grad is None
                for parameter in audited_model.parameters()
            ))

            (supervised + effective_lambda * gcl).backward()
            reference_supervised, reference_gcl = losses(reference_model)
            (
                reference_supervised
                + effective_lambda * reference_gcl
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
            self.assertEqual(summary['schema_version'], 2)
            self.assertEqual(
                summary['lambda_schedule']['schedule'], 'linear'
            )
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

    def test_gradient_audit_does_not_change_pcgrad_backward(self):
        audited_model = copy.deepcopy(self.model)
        reference_model = copy.deepcopy(self.model)
        args = make_args(
            joint_gradient_audit=1,
            joint_gradient_audit_interval=10,
            joint_gcl_lambda=0.05,
            joint_gradient_strategy='pcgrad',
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
            gradient_context = build_joint_gradient_audit_context(
                args, audited_model
            )
            audited_pcgrad = build_joint_pcgrad_context(args, audited_model)
            reference_pcgrad = build_joint_pcgrad_context(args, reference_model)
            audited_supervised, audited_gcl = losses(audited_model)
            reference_supervised, reference_gcl = losses(reference_model)
            record_joint_gradient_audit(
                args,
                gradient_context,
                audited_supervised,
                audited_gcl,
                epoch=0,
                batch_index=0,
            )
            audited_stats = joint_pcgrad_backward(
                audited_pcgrad,
                audited_supervised,
                audited_gcl,
                0.05,
            )
            reference_stats = joint_pcgrad_backward(
                reference_pcgrad,
                reference_supervised,
                reference_gcl,
                0.05,
            )
            self.assertEqual(
                audited_stats['projection_applied'],
                reference_stats['projection_applied'],
            )
            self.assertAlmostEqual(
                audited_stats['raw_cosine'],
                reference_stats['raw_cosine'],
            )
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


if __name__ == '__main__':
    unittest.main()
