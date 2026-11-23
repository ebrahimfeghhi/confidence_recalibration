"""CPU-only correctness gates. Run: python -m unittest discover -s extra_training/joint_reranker -v"""

import unittest
import torch
import torch.nn.functional as F

from extra_training.joint_reranker import (
    PAEPoseReranker, antibody_antigen_pair_mask, acceptable_mass_loss,
    joint_objective, shared_gradient_stats, alpha_for_shared_gradient_ratio,
)


class JointRerankerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.roles = torch.tensor([0, 0, 1, 1, 2, 2])
        self.mask = antibody_antigen_pair_mask(self.roles, (0, 1), (2,))
        self.model = PAEPoseReranker(pair_chunk_size=5, attention_dropout=0.0).eval()

    def _batch(self, k=3, t=6):
        return (torch.randn(k, t, t, 64, requires_grad=True),
                self.mask.expand(k, -1, -1).clone(),
                torch.linspace(0.3, 0.7, k, requires_grad=True))

    def test_mask_direction_and_frame(self):
        self.assertEqual(int(self.mask.sum()), 16)
        self.assertTrue(self.mask[0, 4].item())
        self.assertTrue(self.mask[4, 0].item())
        self.assertFalse(self.mask[0, 1].item())
        frame = torch.tensor([True, False, True, True, True, True])
        restricted = antibody_antigen_pair_mask(self.roles, (0, 1), (2,), frame)
        self.assertEqual(int(restricted.sum()), 14)
        with self.assertRaisesRegex(ValueError, "no scored"):
            antibody_antigen_pair_mask(self.roles, (0,), (8,))
        with self.assertRaisesRegex(ValueError, "overlap"):
            antibody_antigen_pair_mask(self.roles, (0,), (0,))

    def test_shapes_and_exact_initialization(self):
        logits, mask, iptm = self._batch()
        scores = self.model(logits, mask, iptm)
        self.assertEqual(scores.shape, (3,))
        torch.testing.assert_close(scores, iptm, rtol=0, atol=0)
        for k in (1, 2, 5):
            logits = torch.randn(k, 8, 8, 64)
            role = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
            mask = antibody_antigen_pair_mask(role, (0,), (2,))
            scores = self.model(logits, mask.expand(k, -1, -1), torch.zeros(k))
            self.assertEqual(scores.shape, (k,))

    def test_rectangular_pae_mask(self):
        asym = torch.tensor([0, 0, 2, 2, 3])
        frame = torch.tensor([True, False, True, True, False])
        mask = antibody_antigen_pair_mask(asym, (0,), (2,), frame)
        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(int(mask.sum()), 6)
        self.assertEqual(self.model([torch.randn(3, 5, 64)], [mask], torch.tensor([0.4])).shape, (1,))

    def test_streaming_inference_matches_full_forward(self):
        logits, mask, iptm = self._batch()
        with torch.no_grad():
            torch.nn.init.normal_(self.model.score_head[-1].weight, std=0.025)
            full = self.model(logits, mask, iptm)
            emb = torch.stack([self.model.encode_pose(l, m) for l, m in zip(logits, mask)])
            streaming = self.model.score_embeddings(emb, iptm)
        torch.testing.assert_close(full, streaming, rtol=0, atol=0)

    def test_variable_crop_size_in_same_group(self):
        a = torch.randn(6, 6, 64)
        b = torch.randn(8, 8, 64)
        roles = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        mb = antibody_antigen_pair_mask(roles, (0,), (2,))
        self.assertEqual(self.model([a, b], [self.mask, mb], torch.tensor([0.4, 0.6])).shape, (2,))

    def test_permutation_equivariance(self):
        logits, mask, iptm = self._batch(k=4)
        with torch.no_grad():
            torch.nn.init.normal_(self.model.score_head[-1].weight, std=0.05)
            first = self.model(logits, mask, iptm)
            order = torch.tensor([2, 0, 3, 1])
            second = self.model(logits[order], mask[order], iptm[order])
        torch.testing.assert_close(second, first[order], atol=2e-6, rtol=2e-6)

    def test_attention_chunk_parity(self):
        logits, mask, iptm = self._batch(k=2)
        big = PAEPoseReranker(pair_chunk_size=10000, attention_dropout=0).eval()
        torch.nn.init.normal_(self.model.score_head[-1].weight, std=0.02)
        big.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(logits, mask, iptm),
                                   big(logits, mask, iptm), atol=2e-6, rtol=2e-6)

    def test_listwise_loss_good_bad_and_skipped(self):
        scores = torch.tensor([2., 1., 0.], requires_grad=True)
        dockq = torch.tensor([0.01, 0.80, 0.50])
        loss, active = acceptable_mass_loss(scores, dockq, temperature=1.0)
        self.assertTrue(active)
        target = torch.logsumexp(scores, 0) - torch.logsumexp(scores[1:], 0)
        torch.testing.assert_close(loss, target)
        loss.backward()
        self.assertGreater(scores.grad[0].item(), 0)
        self.assertLess(scores.grad[1].item(), 0)
        for labels in (torch.zeros(3), torch.ones(3)):
            empty, active = acceptable_mass_loss(scores, labels)
            self.assertFalse(active)
            self.assertEqual(float(empty.detach()), 0.)
            self.assertTrue(empty.requires_grad)

    def test_end_to_end_gradients(self):
        head = torch.nn.Linear(3, 64)
        coords = torch.randn(3, 6, 6, 3)
        logits = head(coords)
        iptm = logits.softmax(-1)[..., 0].mean(dim=(1, 2))
        mask = self.mask.expand(3, -1, -1)
        labels = torch.randint(0, 64, (3, 6, 6))
        ce = torch.stack([F.cross_entropy(logits[i][self.mask], labels[i][self.mask])
                          for i in range(3)])
        dockq = torch.tensor([0.05, 0.74, 0.10])
        result = joint_objective(self.model, logits, mask, iptm, ce, dockq,
                                 alpha=1.0, temperature=0.5)
        self.assertTrue(result.ranking_active)
        self.assertGreater(torch.autograd.grad(result.ranking, head.weight,
                                               retain_graph=True)[0].norm().item(), 0)
        result.total.backward()
        self.assertGreater(head.weight.grad.norm().item(), 0)
        self.assertGreater(self.model.score_head[-1].weight.grad.norm().item(), 0)
        # After first layer update, gradients also flow along the learned
        # PAE-pair encoder path rather than only through live ipTM.
        with torch.no_grad():
            self.model.score_head[-1].weight.add_(0.005)
        head.zero_grad()
        self.model.zero_grad()
        logits = head(coords)
        result = joint_objective(self.model, logits, mask, torch.zeros(3),
                                 ce.detach(), dockq, temperature=0.5)
        self.assertGreater(torch.autograd.grad(result.ranking, head.weight,
                                               retain_graph=True)[0].norm().item(), 0)
        result.ranking.backward()
        self.assertGreater(self.model.phi[0].weight.grad.norm().item(), 0)

    def test_gradient_calibration_and_non_consuming_probe(self):
        p = torch.nn.Parameter(torch.tensor([1., 3.]))
        pae = p.square().sum()
        rank = 3 * p.sum()
        stats = shared_gradient_stats(pae, rank, [p])
        self.assertAlmostEqual(stats["pae_norm"], 40**0.5, places=5)
        self.assertAlmostEqual(stats["rank_norm"], 18**0.5, places=5)
        self.assertAlmostEqual(alpha_for_shared_gradient_ratio(stats, ratio=.3),
                               .3 * (40 / 18)**.5, places=5)
        (pae + rank).backward()
        torch.testing.assert_close(p.grad, torch.tensor([5., 9.]))

    def test_invalid_shapes_and_masks(self):
        logits, mask, iptm = self._batch()
        with self.assertRaisesRegex(ValueError, "no valid"):
            self.model(logits, torch.zeros_like(mask), iptm)
        with self.assertRaisesRegex(ValueError, "number"):
            self.model(logits, mask, iptm[:2])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(logits, mask[:, :5, :5], iptm)


if __name__ == "__main__":
    unittest.main()
