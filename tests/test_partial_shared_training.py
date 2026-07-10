import unittest
import copy
import json
import random
import tempfile
from types import SimpleNamespace

import torch
import numpy as np
from torch_geometric.data import Data

from downstream_train import (
    apply_partial_shared_backbone_eval_policy,
    build_partial_shared_audit_context,
    build_optimizer,
    maybe_unfreeze_partial_shared_backbone,
    record_partial_shared_representation_audit,
    set_partial_shared_backbone_trainable,
)


class DummyPartialSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Dropout(0.5),
        )
        self.head = torch.nn.Linear(2, 1)


class DummyAuditBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, batch):
        return self.linear(batch.audit_x)


class DummyAuditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = DummyAuditBackbone()
        self.use_stats = True
        self.stats_fusion = 'gate'
        self.stats_gate = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.Sigmoid(),
        )

    def _encode_stats(self, batch):
        return batch.audit_stats

    def _fuse_stats(self, x, batch):
        stats_x = self._encode_stats(batch)
        gate = self.stats_gate(torch.cat((x, stats_x), dim=1))
        return gate * x + (1.0 - gate) * stats_x


class RngConsumingLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        random.random()
        np.random.rand()
        torch.rand(1)
        yield self.batch


def make_args(**overrides):
    values = {
        'use_sgrl_partial_shared': 1,
        'partial_shared_freeze_epochs': 2,
        'partial_shared_backbone_lr': 1e-5,
        'partial_shared_backbone_eval_policy': 'frozen_only',
        'lr': 1e-3,
        'finetune_sgrl_online': 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PartialSharedTrainingTest(unittest.TestCase):
    def test_unfreeze_preserves_optimizer_and_downstream_state(self):
        args = make_args()
        model = DummyPartialSharedModel()
        set_partial_shared_backbone_trainable(args, model, False)
        optimizer = build_optimizer(args, model)

        loss = model.head(torch.ones(2, 2)).sum()
        loss.backward()
        optimizer.step()
        head_weight = model.head.weight
        step_before = optimizer.state[head_weight]['step'].clone()
        optimizer_id = id(optimizer)

        returned = maybe_unfreeze_partial_shared_backbone(
            args,
            model,
            optimizer,
            epoch=2,
        )

        self.assertEqual(id(returned), optimizer_id)
        self.assertTrue(all(
            param.requires_grad for param in model.shared_backbone.parameters()
        ))
        self.assertTrue(torch.equal(
            optimizer.state[head_weight]['step'],
            step_before,
        ))
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[-1]['lr'], 1e-5)

    def test_eval_policy_is_independent_from_trainability(self):
        model = DummyPartialSharedModel()
        args = make_args(
            partial_shared_freeze_epochs=0,
            partial_shared_backbone_eval_policy='always',
        )
        model.train()
        applied = apply_partial_shared_backbone_eval_policy(args, model, epoch=0)

        self.assertTrue(applied)
        self.assertFalse(model.shared_backbone.training)
        self.assertTrue(all(
            param.requires_grad for param in model.shared_backbone.parameters()
        ))
        self.assertTrue(model.head.training)

    def test_frozen_only_policy_returns_to_train_mode_after_warmup(self):
        model = DummyPartialSharedModel()
        args = make_args(partial_shared_backbone_eval_policy='frozen_only')

        model.train()
        self.assertTrue(
            apply_partial_shared_backbone_eval_policy(args, model, epoch=1)
        )
        self.assertFalse(model.shared_backbone.training)

        model.train()
        self.assertFalse(
            apply_partial_shared_backbone_eval_policy(args, model, epoch=2)
        )
        self.assertTrue(model.shared_backbone.training)

    def test_representation_audit_records_drift_and_gate_statistics(self):
        model = DummyAuditModel()
        reference = copy.deepcopy(model.shared_backbone)
        batch = Data(
            audit_x=torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            ]),
            audit_stats=torch.ones(4, 2),
            edge_label=torch.tensor([[0.2, 1.0], [0.6, 3.0]]),
        )
        args = make_args(task_level='edge')

        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = f'{temporary_dir}/representation_audit.jsonl'
            context = {
                'reference_backbone': reference,
                'fixed_batches': {'source_val': batch},
                'device': torch.device('cpu'),
                'path': audit_path,
            }
            initial = record_partial_shared_representation_audit(
                args, model, context, epoch=-1
            )[0]
            self.assertAlmostEqual(initial['root_cosine_mean'], 1.0, places=6)
            self.assertAlmostEqual(initial['weight_relative_l2'], 0.0, places=6)
            self.assertIn('gate_mean', initial)

            with torch.no_grad():
                model.shared_backbone.linear.weight.add_(0.5)
            changed = record_partial_shared_representation_audit(
                args, model, context, epoch=0
            )[0]
            self.assertGreater(changed['weight_relative_l2'], 0.0)

            with open(audit_path, encoding='utf-8') as audit_file:
                lines = [json.loads(line) for line in audit_file]
            self.assertEqual(len(lines), 2)

    def test_audit_batch_capture_preserves_rng_state(self):
        model = DummyAuditModel()
        batch = Data(
            audit_x=torch.ones(4, 2),
            audit_stats=torch.ones(4, 2),
            edge_label=torch.tensor([[0.2, 1.0], [0.6, 3.0]]),
        )
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        with tempfile.TemporaryDirectory() as temporary_dir:
            args = make_args(
                partial_shared_audit=1,
                run_artifact_dir=temporary_dir,
            )
            build_partial_shared_audit_context(
                args,
                model,
                RngConsumingLoader(batch),
                {'transfer': RngConsumingLoader(batch)},
                torch.device('cpu'),
            )

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        self.assertTrue(np.array_equal(
            restored_numpy_state[1], numpy_state[1]
        ))
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == '__main__':
    unittest.main()
