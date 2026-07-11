from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import joblib
import torch

from balanced_mse import train_gmm


class SourceOnlyDataset:
    names = ['source', 'forbidden_transfer']

    def __init__(self):
        values = torch.linspace(0.01, 0.99, 80)
        classes = torch.bucketize(values, torch.tensor([0.2, 0.4, 0.6, 0.8]))
        self.source = SimpleNamespace(
            edge_label=torch.stack((values, classes), dim=1)
        )

    def __getitem__(self, index):
        if index != 0:
            raise AssertionError('Training GMM accessed a transfer label.')
        return self.source


class RebalancingProtocolTest(unittest.TestCase):
    def test_gmm_uses_source_only_and_is_run_isolated(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / 'artifact' / 'gmm.pkl'
            returned = train_gmm(SourceOnlyDataset(), str(output))
            payload = joblib.load(output)

            self.assertEqual(returned, str(output))
            self.assertEqual(
                set(payload),
                {'means', 'weights', 'variances'},
            )
            self.assertFalse(Path(f'{output}.tmp').exists())


if __name__ == '__main__':
    unittest.main()
