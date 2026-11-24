"""CPU gate for calibrate_alpha: the ratio, the median, and the step-0 detection.

What can go wrong here is arithmetic and bookkeeping, not modelling: a mean where the README asks
for a median, a ratio taken per pose instead of over the accumulation window, a window with no
mixed labels silently reported as a converged alpha, or a step-0 calibration mistaken for a real
one. Each of those has a check.
"""

import unittest

import torch
from torch import nn

from extra_training.joint_reranker.calibrate_alpha import (
    measure_window, summarise, format_report)
from extra_training.joint_reranker.integration import PosePrediction
from extra_training.joint_reranker.model import PAEPoseReranker, antibody_antigen_pair_mask


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 64)


class CalibrateAlphaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = _Head()
        self.rr = PAEPoseReranker(attention_dropout=0).eval()
        self.asym = torch.tensor([0, 0, 1, 1, 2, 2])
        self.mask = antibody_antigen_pair_mask(self.asym, (0, 1), (2,))

    def predict(self, pose):
        x = self.head.w(pose)
        ce = x.square().mean()
        iptm = x.softmax(dim=-1)[self.mask][:, 0].mean()
        return PosePrediction(x, self.mask, iptm, ce)

    def _group(self, dockq):
        poses = [torch.randn(6, 6, 3) for _ in range(len(dockq))]
        return poses, torch.tensor(dockq)

    def test_window_measures_both_norms_and_the_cosine(self):
        w = measure_window(self.rr, [self._group([0.0, 0.5, 0.1, 0.9])], self.predict,
                           list(self.head.parameters()), temperature=0.5)
        self.assertGreater(w.pae_norm, 0)
        self.assertGreater(w.rank_norm, 0)
        self.assertTrue(-1.0001 <= w.cosine <= 1.0001)
        self.assertEqual(w.ranking_active, 1)
        # alpha at parity is exactly the norm ratio
        self.assertAlmostEqual(w.alpha_parity, w.pae_norm / w.rank_norm, places=5)

    def test_step_zero_residual_is_detected(self):
        """A freshly built reranker has a zero-init final layer, so the residual must read ~0."""
        w = measure_window(self.rr, [self._group([0.0, 0.9])], self.predict,
                           list(self.head.parameters()), temperature=0.5)
        self.assertLess(w.residual_rms, 1e-6)
        # and once the final layer is non-zero it must read non-zero
        with torch.no_grad():
            self.rr.score_head[-1].weight.add_(0.5)
            if self.rr.score_head[-1].bias is not None:
                self.rr.score_head[-1].bias.add_(0.1)
        w2 = measure_window(self.rr, [self._group([0.0, 0.9])], self.predict,
                            list(self.head.parameters()), temperature=0.5)
        self.assertGreater(w2.residual_rms, 1e-6)

    def test_all_good_or_all_bad_window_is_refused(self):
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            measure_window(self.rr, [self._group([0.9, 0.8])], self.predict,
                           list(self.head.parameters()))
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            measure_window(self.rr, [self._group([0.0, 0.1])], self.predict,
                           list(self.head.parameters()))

    def test_summary_is_a_median_and_reports_the_spread(self):
        ws = [measure_window(self.rr, [self._group([0.0, 0.5, 0.9])], self.predict,
                             list(self.head.parameters()), temperature=0.5) for _ in range(5)]
        s = summarise(ws)
        vals = sorted(w.alpha_parity for w in ws)
        self.assertAlmostEqual(s["median_alpha_parity"], vals[2], places=6)
        self.assertNotAlmostEqual(s["median_alpha_parity"],
                                  sum(vals) / len(vals), places=12)  # not a mean
        self.assertAlmostEqual(s["alpha_by_ratio"]["0.3"], 0.3 * vals[2], places=6)
        self.assertAlmostEqual(s["recommended_alpha"], s["alpha_by_ratio"]["0.3"], places=9)
        self.assertGreaterEqual(s["spread_factor"], 1.0)
        self.assertNotIn("p10_alpha_parity", s)   # min/max, not percentiles
        self.assertIn("Recommended start", format_report(s))

    def test_window_accumulates_over_groups_not_per_pose(self):
        """Two groups in one window must not give the same norms as one group alone."""
        g = self._group([0.0, 0.9])
        one = measure_window(self.rr, [g], self.predict, list(self.head.parameters()),
                             temperature=0.5)
        two = measure_window(self.rr, [g, self._group([0.1, 0.8])], self.predict,
                             list(self.head.parameters()), temperature=0.5)
        self.assertEqual(two.n_groups, 2)
        self.assertNotAlmostEqual(one.pae_norm, two.pae_norm, places=6)

    def test_empty_window_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one group"):
            measure_window(self.rr, [], self.predict, list(self.head.parameters()))
        with self.assertRaisesRegex(ValueError, "no calibration windows"):
            summarise([])


if __name__ == "__main__":
    unittest.main()
