import unittest
from types import SimpleNamespace

import torch

from downstream_train import (
    apply_partial_shared_backbone_eval_policy,
    build_optimizer,
    maybe_unfreeze_partial_shared_backbone,
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


if __name__ == '__main__':
    unittest.main()
