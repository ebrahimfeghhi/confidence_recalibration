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

    # iptm_pooled's row set on a 7-token fixture: token 3 has no frame, and
    # token 6 is a glycan/hetero token INSIDE antigen chain 2 -- the chain-ID
    # filter cannot see it, only is_ligand can, which is the case the new
    # parameter exists for.
    LIG_ASYM = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    LIG_FRAME = torch.tensor([True, True, True, False, True, True, True])
    LIG_FLAG = torch.tensor([False, False, False, False, False, False, True])

    def test_ligand_exclusion_drops_rows_and_keeps_columns(self):
        kept = antibody_antigen_pair_mask(self.LIG_ASYM, (0, 1), (2,), self.LIG_FRAME)
        live = antibody_antigen_pair_mask(self.LIG_ASYM, (0, 1), (2,), self.LIG_FRAME,
                                          self.LIG_FLAG)
        self.assertEqual(kept.shape, (6, 7))        # has_frame alone: token 3 gone
        self.assertEqual(live.shape, (5, 7))        # ligand token 6 gone as a ROW
        self.assertEqual(int(kept.sum()), 21)
        self.assertEqual(int(live.sum()), 17)
        # THE LIGAND TOKEN IS STILL A COLUMN. iptm_pooled's target axis is
        # `tok` plus cross-chain-ness and neither has_frame nor the ligand flag
        # filters it, so dropping column 6 would pool over fewer pairs than the
        # ipTM being corrected -- the same class of mismatch, in the other axis.
        self.assertTrue(live[0, 6].item())
        self.assertTrue(live[2, 6].item())
        # The kept rows must be shape-aligned with a PAE grid sliced by the SAME
        # boolean; slicing by has_frame alone is the silent-misalignment bug.
        rows = torch.randn(7, 7, 64)[self.LIG_FRAME & ~self.LIG_FLAG]
        self.assertEqual(self.model([rows], [live], torch.tensor([0.5])).shape, (1,))

    def test_ligand_exclusion_is_noop_without_a_ligand(self):
        none = torch.zeros(7, dtype=torch.bool)
        for frame in (None, self.LIG_FRAME):
            base = antibody_antigen_pair_mask(self.LIG_ASYM, (0, 1), (2,), frame)
            same = antibody_antigen_pair_mask(self.LIG_ASYM, (0, 1), (2,), frame, none)
            self.assertTrue(torch.equal(same, base))
            # iptm_pooled's own `lig` is a `> 0` on a long scatter_add, so a
            # caller handing through the integer form must get the same mask.
            self.assertTrue(torch.equal(
                antibody_antigen_pair_mask(self.LIG_ASYM, (0, 1), (2,), frame, none.long()),
                base))

    def test_ligand_row_set_matches_has_frame_and_not_ligand(self):
        asym, frame, lig = self.LIG_ASYM, self.LIG_FRAME, self.LIG_FLAG
        ab, ag = (asym == 0) | (asym == 1), asym == 2
        full = (ab[:, None] & ag[None, :]) | (ag[:, None] & ab[None, :])
        got = antibody_antigen_pair_mask(asym, (0, 1), (2,), frame, lig)
        self.assertTrue(torch.equal(got, full[frame & ~lig]))
        self.assertTrue(torch.equal(got, antibody_antigen_pair_mask(
            asym, (0, 1), (2,), frame, lig.long())))
        # Rows come back in increasing token index -- the order row_t documents
        # and the order `logits[live]` produces. A permuted row set would pair
        # one row's PAE with another row's pooled feature and still typecheck.
        self.assertTrue(torch.equal(got[0], full[0]))
        self.assertTrue(torch.equal(got[-1], full[5]))
        with self.assertRaisesRegex(ValueError, "is_ligand_token must have shape"):
            antibody_antigen_pair_mask(asym, (0, 1), (2,), frame, lig[:5])
        # An all-ligand crop leaves no anchor row at all, which is iptm_pooled
        # returning None rather than something scorable.
        with self.assertRaisesRegex(ValueError, "no scored"):
            antibody_antigen_pair_mask(asym, (0, 1), (2,), frame,
                                       torch.ones(7, dtype=torch.bool))

    def test_existing_mask_behaviour_unchanged_without_the_parameter(self):
        self.assertTrue(torch.equal(
            self.mask, antibody_antigen_pair_mask(self.roles, (0, 1), (2,))))
        self.assertTrue(torch.equal(
            self.mask, antibody_antigen_pair_mask(self.roles, (0, 1), (2,), None, None)))
        frame = torch.tensor([True, False, True, True, True, True])
        no_lig = antibody_antigen_pair_mask(self.roles, (0, 1), (2,), frame)
        self.assertEqual((tuple(no_lig.shape), int(no_lig.sum())), ((5, 6), 14))

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
