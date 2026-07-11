from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / 'scripts'
for path in (REPO_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_label_distribution import compare_to_source, distribution
from downstream_train import regression_bin_metrics


class LabelDistributionTest(unittest.TestCase):
    def test_distribution_shift_and_binned_errors(self):
        bins = np.linspace(0.0, 1.0, 11)
        source = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
        target = np.array([0.65, 0.75, 0.85, 0.95])
        summary = distribution(source, bins, [0.2, 0.4, 0.6, 0.8])
        comparison = compare_to_source(source, target, bins)
        errors = regression_bin_metrics(
            np.array([0.05, 0.15, 0.95]),
            np.array([0.10, 0.10, 0.85]),
            bins,
        )

        self.assertEqual(summary['count'], 6)
        self.assertGreater(comparison['jensen_shannon_10bin'], 0.0)
        self.assertGreater(comparison['target_fraction_above_source_q90'], 0.5)
        self.assertEqual(errors[0]['count'], 1)
        self.assertAlmostEqual(errors[0]['mse'], 0.0025)
        self.assertEqual(errors[-1]['count'], 1)
        self.assertAlmostEqual(errors[-1]['bias'], -0.1)


if __name__ == '__main__':
    unittest.main()
