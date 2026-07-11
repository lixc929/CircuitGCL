from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from downstream_train import (
    Logger,
    build_optimizer,
    regress_train,
    validation_improved,
)
from model import JointSharedGraphHead
from run_artifacts import prepare_run_artifacts
from test_joint_shared import make_args


class TinyBatch:
    def to(self, device):
        return self


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        prediction = self.weight.expand(2)
        labels = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return prediction, None, labels


class RegressionProtocolTest(unittest.TestCase):
    def test_checkpoint_selection_uses_unrounded_mse(self):
        logger = Logger(task='regression')
        true = torch.tensor([[0.0], [1.0]])
        pred = torch.tensor([[0.0], [1.01]])
        logger.update_stats(true, pred, batch_size=2, loss=0.00005)
        result = logger.write_epoch('val')

        self.assertEqual(result['mse'], 0.0)
        self.assertGreater(result['mse_raw'], 0.0)
        self.assertTrue(validation_improved(0.00006, result))
        self.assertFalse(validation_improved(0.00004, result))
        self.assertTrue(validation_improved(
            0.00990960,
            {'mse': 0.0099, 'mse_raw': 0.00986081},
        ))

    def test_joint_optimizer_keeps_lora_at_task_learning_rate(self):
        args = make_args(joint_lora_rank=2)
        args.use_sgrl_joint_shared = 1
        args.joint_backbone_lr = 1e-5
        args.lr = 1e-4
        model = JointSharedGraphHead(args)
        encoder_state = {
            f'online_encoder.{key}': value.clone()
            for key, value in model.shared_backbone.gnn.state_dict().items()
        }
        model.load_online_encoder_state(encoder_state)

        optimizer = build_optimizer(args, model)
        groups_by_lr = {
            group['lr']: {id(parameter) for parameter in group['params']}
            for group in optimizer.param_groups
        }
        lora_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and ('.lora_a.' in name or '.lora_b.' in name)
        }
        base_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.startswith('shared_backbone.gnn.')
            and '.lora_a.' not in name
            and '.lora_b.' not in name
        }

        self.assertTrue(lora_ids <= groups_by_lr[1e-4])
        self.assertTrue(base_ids <= groups_by_lr[1e-5])
        self.assertTrue(lora_ids.isdisjoint(groups_by_lr[1e-5]))

    def test_transfer_sets_are_evaluated_once_after_best_reload(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = SimpleNamespace(
                seed=0,
                log_dir=temporary_dir,
                task='regression',
                task_level='edge',
                regress_loss='mse',
                epochs=2,
                lr=0.1,
                use_sgrl_joint_shared=0,
                use_sgrl_partial_shared=0,
                partial_shared_audit=0,
                partial_shared_freeze_epochs=0,
                early_stopping_patience=0,
                early_stopping_min_delta=0.0,
                joint_gcl_lambda=0.0,
            )
            prepare_run_artifacts(args, Path(temporary_dir) / 'run.txt')
            model = TinyRegressor()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            splits = []
            validation_results = iter([
                {'loss': 0.1, 'mse': 0.1, 'mse_raw': 0.1},
                {'loss': 0.2, 'mse': 0.2, 'mse_raw': 0.2},
            ])

            def fake_eval(*unused_args, split, **unused_kwargs):
                splits.append(split)
                if split == 'val':
                    return next(validation_results)
                return {'loss': 0.3, 'mse': 0.3, 'mse_raw': 0.3}

            with patch('downstream_train.eval_epoch', side_effect=fake_eval):
                results = regress_train(
                    args,
                    model,
                    optimizer,
                    torch.nn.MSELoss(),
                    [TinyBatch()],
                    [],
                    {'target': []},
                    max_label=None,
                    device=torch.device('cpu'),
                )

            self.assertEqual(splits, ['val', 'val', 'test:target'])
            self.assertEqual(results['best_epoch'], 0)
            checkpoint = torch.load(
                Path(args.run_artifact_dir) / 'best_model.pt',
                map_location='cpu',
                weights_only=True,
            )
            self.assertTrue(torch.equal(
                model.state_dict()['weight'],
                checkpoint['model_state_dict']['weight'],
            ))
            self.assertEqual(
                checkpoint['metrics']['test_results']['target']['mse_raw'],
                0.3,
            )


if __name__ == '__main__':
    unittest.main()
