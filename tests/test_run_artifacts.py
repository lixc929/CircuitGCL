import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from run_artifacts import (
    prepare_run_artifacts,
    save_best_checkpoint,
    write_run_metrics,
)


class RunArtifactsTest(unittest.TestCase):
    def test_run_artifacts_are_isolated_and_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = SimpleNamespace(seed=2, log_dir=str(root))
            artifact_dir = prepare_run_artifacts(args, root / 'run_a.txt')

            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()

            checkpoint_path = save_best_checkpoint(
                args,
                model,
                optimizer,
                epoch=3,
                metrics={'best_val_mse': 0.1},
            )
            write_run_metrics(args, {'status': 'completed'})

            self.assertEqual(checkpoint_path.parent, artifact_dir)
            self.assertTrue((artifact_dir / 'run_config.json').is_file())
            self.assertTrue((artifact_dir / 'metrics.json').is_file())
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(checkpoint['epoch'], 3)
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('optimizer_state_dict', checkpoint)

            config = json.loads(
                (artifact_dir / 'run_config.json').read_text(encoding='utf-8')
            )
            self.assertEqual(config['args']['seed'], 2)
            self.assertEqual(config['artifact_dir'], str(artifact_dir))

            other_args = SimpleNamespace(seed=3, log_dir=str(root))
            other_dir = prepare_run_artifacts(other_args, root / 'run_b.txt')
            self.assertNotEqual(artifact_dir, other_dir)


if __name__ == '__main__':
    unittest.main()
