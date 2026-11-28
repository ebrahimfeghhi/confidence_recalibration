"""CPU gate for protenix_head_bridge: the plumbing, with the REAL ipTM kernel.

The confidence head and ProtenixLoss are stubbed -- they need CUDA and a trunk -- but
`rank_loss.iptm_pooled` is the real one, so what this pins is exactly the part that can silently
go wrong between the trainer and the reranker: which rows are live, whether the mask matches the
logit grid it is applied to, whether the ipTM is the fixed-d0 max, and whether a detached scalar
is caught instead of training the reranker alone.

WHY A STUB IS ENOUGH HERE. Every quantity this adapter is responsible for is a SHAPE or a MASK
decision; the numbers come from functions pinned by their own gates (test_iptm_pooled_max_d0.py,
test_pool_mean.py, test_pae_row_weight.py). A stub that returns the right shapes exercises every
line of the adapter except the vendor forward itself.
"""

import os
import sys
import types
import unittest

import torch
from torch import nn

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")

from extra_training.joint_reranker.protenix_head_bridge import (   # noqa: E402
    PoseInput, live_row_mask, make_predict_one, scored_pair_mask)

NB = 64


def _feat(n_ab=12, n_ag=10, n_lig=0, no_frame=0):
    """A feat dict with the five fields the adapter and iptm_pooled read."""
    n = n_ab + n_ag + n_lig
    asym = torch.cat([torch.zeros(n_ab), torch.ones(n_ag), torch.full((n_lig,), 2.0)]).long()
    hf = torch.ones(n, dtype=torch.float32)
    if no_frame:
        hf[:no_frame] = 0.0
    lig = torch.zeros(n, dtype=torch.long)
    if n_lig:
        lig[n_ab + n_ag:] = 1
    return {"asym_id": asym, "has_frame": hf, "is_ligand": lig,
            "atom_to_token_idx": torch.arange(n, dtype=torch.long)}


def _configs():
    pae = types.SimpleNamespace(min_bin=0.0, max_bin=32.0, no_bins=NB)
    return types.SimpleNamespace(loss=types.SimpleNamespace(pae=pae),
                                 triangle_multiplicative="torch",
                                 triangle_attention="torch")


class _StubHead(nn.Module):
    """Stands in for run_confidence_head: one trainable weight, [1,T,T,64] logits."""
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, n, NB) * 0.1)

    def run_confidence_head(self, **kw):
        x = kw["x_pred_coords"]
        lg = (self.w * (1.0 + x.reshape(-1)[0])).unsqueeze(0)
        z = torch.zeros(())
        return z, lg, z, z


class _Zero:
    """Stands in for the --pae-only stub, matched BY CLASS NAME as the bridge checks it."""


class _StubLoss:
    """calculate_losses returning a LIVE scalar built from the prediction, as the real one does."""
    def __init__(self, detach=False, pae_only=True):
        self.detach = detach
        if pae_only:
            self.plddt_loss = self.pde_loss = self.resolved_loss = _Zero()

    def calculate_losses(self, feat, pred, labels, mode="train"):
        v = pred["pae"].square().mean()
        return (v.detach() if self.detach else v), {"pae_loss": float(v)}


class HeadBridgeTests(unittest.TestCase):
    def _pose(self, feat, n):
        return PoseInput(feat_head=feat, feat_cache=feat, labels={},
                         s_inputs=torch.zeros(1), s_trunk=torch.zeros(1),
                         z_trunk=torch.zeros(1), coords=torch.full((1,), 0.3),
                         pose_id=("1abc", 500000, 0))

    def test_live_rows_match_iptm_pooled(self):
        """The adapter's `live` must be has_frame & ~ligand, the row set iptm_pooled scores."""
        feat = _feat(n_lig=4, no_frame=3)
        live = live_row_mask(feat)
        hf = feat["has_frame"].bool()
        lig = torch.zeros_like(feat["asym_id"]).scatter_add(
            0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0
        self.assertTrue(torch.equal(live, hf & ~lig))
        self.assertEqual(int(live.sum()), 12 + 10 - 3)   # ligands and frameless rows both gone

    def test_mask_shape_and_no_reference_input(self):
        feat = _feat(n_lig=4)
        mask = scored_pair_mask(feat, (0, 1))
        self.assertEqual(tuple(mask.shape), (int(live_row_mask(feat).sum()),
                                             feat["asym_id"].shape[0]))
        # ligand columns are never AB or AG, so no ligand pair can be scored
        asym = feat["asym_id"]
        self.assertFalse(bool(mask[:, asym == 2].any()))

    def test_overlapping_or_empty_chain_ids_refused(self):
        feat = _feat()
        n = feat["asym_id"].shape[0]
        npt = int(((feat["asym_id"] == 0) | (feat["asym_id"] == 1)).sum())
        mk = lambda ab, ag, pair=(0, 1): make_predict_one(  # noqa: E731
            _StubHead(n), _configs(), _StubLoss(), iptm_pair=pair, n_pair_tokens=npt,
            antibody_asym_ids=ab, antigen_asym_ids=ag, autocast=False,
            set_pae_mask=lambda _p: None)
        with self.assertRaisesRegex(ValueError, "overlap"):
            mk((0, 1), (1,))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            mk((0,), ())
        with self.assertRaisesRegex(ValueError, "does not straddle"):
            mk((0,), (1,), pair=(0, 0))

    def test_mask_rows_are_the_SCORED_PAIR_only_not_the_chain_union(self):
        """An H+L antibody must not pool L-AG rows against a baseline scored on H-AG alone.

        iptm_pooled restricts its anchor rows to (arow == i) | (arow == j) for the ONE pair it
        scores (rank_loss.py:404) and pins d0 to that pair. A chain-list union would add the other
        antibody chain's rows on nearly every structure in this set.
        """
        n_h, n_l, n_ag = 8, 7, 9
        asym = torch.cat([torch.zeros(n_h), torch.ones(n_l),
                          torch.full((n_ag,), 2.0)]).long()
        feat = {"asym_id": asym, "has_frame": torch.ones(asym.shape[0]),
                "is_ligand": torch.zeros(asym.shape[0], dtype=torch.long),
                "atom_to_token_idx": torch.arange(asym.shape[0], dtype=torch.long)}
        mask = scored_pair_mask(feat, (0, 2))            # heavy chain against the antigen
        live = live_row_mask(feat)
        arow = asym[live]
        self.assertFalse(bool(mask[arow == 1].any()))     # no light-chain rows
        self.assertFalse(bool(mask[:, asym == 1].any()))  # and no light-chain columns
        self.assertTrue(bool(mask[arow == 0].any()))
        # exactly the cross-chain pairs of (0, 2)
        self.assertEqual(int(mask.sum()), n_h * n_ag + n_ag * n_h)

    def test_predict_one_is_live_and_uses_fixed_d0_max(self):
        feat = _feat()
        n = feat["asym_id"].shape[0]
        head, loss = _StubHead(n), _StubLoss()
        npt = int(((feat["asym_id"] == 0) | (feat["asym_id"] == 1)).sum())
        fn = make_predict_one(head, _configs(), loss, iptm_pair=(0, 1), n_pair_tokens=npt,
                              antibody_asym_ids=(0,), antigen_asym_ids=(1,), pool="max",
                              autocast=False, set_pae_mask=lambda _p: None)
        pred = fn(self._pose(feat, n))
        self.assertEqual(tuple(pred.pae_logits.shape), (int(live_row_mask(feat).sum()), n, NB))
        self.assertEqual(tuple(pred.pair_mask.shape), pred.pae_logits.shape[:2])
        for t in (pred.pae_logits, pred.iptm, pred.pae_ce):
            self.assertTrue(t.requires_grad)
        # the ipTM must be the FIXED-d0 max, i.e. what iptm_pooled returns with the pin
        import rank_loss as RKL
        with torch.no_grad():
            want = RKL.iptm_pooled((head.w * 1.3).unsqueeze(0), feat, _configs(), (0, 1),
                                   pool="max", n_pair_tokens=npt)
        self.assertAlmostEqual(float(pred.iptm), float(want), places=5)
        # and both objectives reach the stub head
        (pred.pae_ce + pred.iptm).backward()
        self.assertGreater(float(head.w.grad.abs().sum()), 0)

    def test_detached_pae_ce_is_refused(self):
        feat = _feat()
        n = feat["asym_id"].shape[0]
        npt = int(((feat["asym_id"] == 0) | (feat["asym_id"] == 1)).sum())
        fn = make_predict_one(_StubHead(n), _configs(), _StubLoss(detach=True),
                              iptm_pair=(0, 1), n_pair_tokens=npt,
                              antibody_asym_ids=(0,), antigen_asym_ids=(1,), autocast=False,
                              set_pae_mask=lambda _p: None)
        with self.assertRaisesRegex(ValueError, "detached"):
            fn(self._pose(feat, n))

    def test_missing_d0_pin_is_refused(self):
        """An unpinned n_pair_tokens is the crop-dependent d0 bug; refuse rather than default."""
        with self.assertRaisesRegex(ValueError, "UNCROPPED"):
            make_predict_one(_StubHead(4), _configs(), _StubLoss(), iptm_pair=(0, 1),
                             n_pair_tokens=0, antibody_asym_ids=(0,), antigen_asym_ids=(1,), set_pae_mask=lambda _p: None)

    def test_batched_sample_axis_is_refused(self):
        """A leading axis > 1 means several poses were batched; attributing them is the bug."""
        feat = _feat()
        n = feat["asym_id"].shape[0]

        class _Batched(_StubHead):
            def run_confidence_head(self, **kw):
                z = torch.zeros(())
                return z, self.w.unsqueeze(0).repeat(3, 1, 1, 1), z, z

        npt = int(((feat["asym_id"] == 0) | (feat["asym_id"] == 1)).sum())
        fn = make_predict_one(_Batched(n), _configs(), _StubLoss(), iptm_pair=(0, 1),
                              n_pair_tokens=npt, antibody_asym_ids=(0,), antigen_asym_ids=(1,),
                              autocast=False, set_pae_mask=lambda _p: None)
        with self.assertRaisesRegex(ValueError, "ONE pose per forward"):
            fn(self._pose(feat, n))


    def test_pae_only_and_the_mask_hook_are_required(self):
        feat = _feat()
        n = feat["asym_id"].shape[0]
        npt = int(((feat["asym_id"] == 0) | (feat["asym_id"] == 1)).sum())
        base = dict(iptm_pair=(0, 1), n_pair_tokens=npt, antibody_asym_ids=(0,),
                    antigen_asym_ids=(1,), autocast=False)
        # without --pae-only, `pae_ce` would carry pLDDT + PDE + resolved too
        with self.assertRaisesRegex(ValueError, "--pae-only"):
            make_predict_one(_StubHead(n), _configs(), _StubLoss(pae_only=False),
                             set_pae_mask=lambda _p: None, **base)
        # and a missing mask hook would reuse the previous pose's scored-pair mask
        with self.assertRaisesRegex(ValueError, "set_pae_mask is required"):
            make_predict_one(_StubHead(n), _configs(), _StubLoss(), **base)
        # the hook is called ONCE PER POSE, before the loss
        seen = []
        fn = make_predict_one(_StubHead(n), _configs(), _StubLoss(),
                              set_pae_mask=lambda p: seen.append(p.pose_id), **base)
        fn(self._pose(feat, n))
        fn(self._pose(feat, n))
        self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()
