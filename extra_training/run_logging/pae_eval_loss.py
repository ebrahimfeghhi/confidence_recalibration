#!/usr/bin/env python
"""The TRAINING PAE loss, computed inside the evaluator's existing forward pass.

WHY THIS EXISTS. The DockQ metric and the held-out PAE loss were two separate passes over the same
poses, costing ~12 min of three cards per epoch on top of the ~11 min eval. They see the same
inputs once --val-loss-uncropped is on, so the second forward was pure duplication: the confidence
head already produces the 64-bin PAE logits the loss needs, and phase_j_rescore_cache.pair_iptm_only
has them in scope for one matmul before discarding them.

WHAT IS COMPUTED, precisely: the same quantity the training curve plots and the same quantity
shell_scripts/val_pae_once.sh measured for shipped (good 3.0991 / bad 2.4336), Arm E, Arm G and
alpha 250 -- the PLAIN per-pose mean of the labelled pae_loss, masked to the scored chain pairs,
with NO row weighting and NO class balance. Those two are training choices that reweight which
poses the gradient listens to; applying either here would make a balanced arm's number
incomparable with every arm measured before it.

THE JOIN IS THE RISK, NOT THE ARITHMETIC -- AND IT IS SIDESTEPPED, NOT GUESSED. A pose's label row
is normally addressed by its SAMPLE index, not by its position in the pose dimension, and this
project has paid for that confusion three times (the rel target regressed pose 1 onto pose 4's true
iTM; --emit-iptm-rows wrote the pose index while 73.8% of its rows carried another sample's DockQ,
moving the aggregate by 0.0009 -- less than the noise it was read against).

Here the LOSS VALUE does not depend on which row is used, because for PAE every row is the same
array: the per-pose permutation in the label cache only ever swaps symmetric SIDE-CHAIN atoms, and
PAE is built from the per-token REPRESENTATIVE atoms, which it never touches.
run_logging/check_pae_label_shared.py measured this over all 8,981 training labels (0 moved) and it
was re-measured over all 2,490 validation labels before this module was written: 2,490 identical,
worst displacement 0.000000 A. So row 0 serves every pose.

That is what lets the loss be computed INSIDE the forward, where the logits are still alive, even
though the evaluator resolves its pose-to-sample join (BC.resolve_seed_dir, by coordinate identity)
only AFTER the forward returns. This module therefore keys its result by POSE INDEX and the caller
attributes pose i to sample ks[i] once the join exists -- the attribution still goes through the
canonical join, never through an assumed ordering.

NOTHING HERE TOUCHES THE TRAINER. The trainer's own loss path is unchanged, so its numbers cannot
move; this module is verified against it numerically instead, by reproducing a standalone
--loss-only measurement of the same bundle (run_logging/check_eval_pae_matches.py).
"""
import csv
import os

import numpy as np

import label_index as LI


class PaeLossCtx:
    """Built once per eval process; `unit_fn` returns a per-unit closure over the live logits."""

    def __init__(self, configs, pool_csv, roles_csv, labels_dir, torch_mod,
                 split_accept=0.23, device=None):
        self.torch = torch_mod
        self.labels_dir = labels_dir
        self.split_accept = float(split_accept)
        self.device = device
        self.n_unmasked = 0
        self.n_scored = 0
        self.n_no_label = 0
        self.n_no_row = 0

        from protenix.model.loss import ProtenixLoss
        self.loss_fn = ProtenixLoss(configs)
        # ONLY THE PAE TERM. ProtenixLoss computes pLDDT, PDE and resolved unconditionally, and
        # each needs label arrays this path has no reason to carry. Stubbing them mirrors what the
        # arms themselves run under --pae-only, so the pae_loss value is produced by the same
        # configuration that produced the training curve.
        class _Zero(torch_mod.nn.Module):
            def forward(self, *_a, **_k):
                return torch_mod.zeros((), dtype=torch_mod.float32)

            def calculate_label(self, *_a, **_k):
                raise AssertionError("stubbed loss term asked for a label")

        for _name in ("plddt_loss", "pde_loss", "resolved_loss"):
            if getattr(self.loss_fn, _name, None) is None:
                raise SystemExit(f"STOP: ProtenixLoss has no {_name} to stub; the eval PAE loss "
                                 f"would silently include a term the arms do not train.")
            setattr(self.loss_fn, _name, _Zero())

        # THE MASK, BUILT EXACTLY AS phase_k_train BUILDS IT -- chain_roles maps ref_chain to
        # model_chain, model chains are lettered in asym_id order by write_job_cif, so the
        # letter's position IS the asym index, and the pair is normalised (min, max) because the
        # pool's order is the REFERENCE order and comes out descending for most interfaces.
        roles = {}
        with open(roles_csv) as fh:
            for r in csv.DictReader(fh):
                roles.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]
        self.scored_asym, miss = {}, []
        with open(pool_csv) as fh:
            for r in csv.DictReader(fh):
                m1 = roles.get(r["pdb"], {}).get(r["chain1"])
                m2 = roles.get(r["pdb"], {}).get(r["chain2"])
                if m1 is None or m2 is None:
                    miss.append(f"{r['pdb']}:{r['chain1']}/{r['chain2']}")
                    continue
                a1, a2 = ord(m1) - ord("A"), ord(m2) - ord("A")
                self.scored_asym.setdefault(r["pdb"], set()).add((min(a1, a2), max(a1, a2)))
        if miss:
            raise SystemExit(f"STOP: {len(miss)} pool interface(s) have no role mapping, e.g. "
                             f"{miss[:4]}. The PAE mask would silently drop their blocks.")

        # THE MASK IS INSTALLED ONCE, and the per-unit value is swapped through this dict. The
        # trainer does the same thing for the same reason: calculate_label is called from inside
        # Protenix, so there is no argument to pass it through.
        self._keep = {"m": None}
        _orig = self.loss_fn.pae_loss.calculate_label

        def _masked_calc(*args, **kw):
            sq, bins, pm = _orig(*args, **kw)
            m = self._keep["m"]
            if m is not None:
                # BOOL AND, NOT A FLOAT MULTIPLY. The float path exists in the trainer only for
                # --pae-row-frac, which is off here; keeping the mask boolean means PAELoss's
                # sum(loss * pair_mask) / sum(pair_mask) is the plain masked mean.
                pm = pm & m.to(pm.device) if pm.dtype == self.torch.bool else \
                    pm * m.to(pm.dtype).to(pm.device)
            return sq, bins, pm

        self.loss_fn.pae_loss.calculate_label = _masked_calc

    def _mask(self, pdb, asym, has_frame):
        """[N_frame, N_token] bool over this pose's tokens; None when nothing is scored.

        NEVER AN ALL-ZERO MASK. PAELoss reduces by sum(pair_mask), so an empty mask divides by eps
        and returns a meaningless finite number instead of failing -- the worst outcome for a
        metric. Falling back to None (unmasked) is wrong too, so the caller counts these and the
        pose is EXCLUDED rather than reported.
        """
        t = self.torch
        pairs = self.scored_asym.get(pdb)
        if not pairs:
            return None
        aa = asym.reshape(-1).long()
        fa = aa[has_frame.reshape(-1).bool()]
        keep = t.zeros((fa.numel(), aa.numel()), dtype=t.bool, device=aa.device)
        for x, y in pairs:
            keep |= (((fa[:, None] == x) & (aa[None, :] == y))
                     | ((fa[:, None] == y) & (aa[None, :] == x)))
        return keep if bool(keep.any()) else None

    def unit_fn(self, pdb, seed, build_cache_mod):
        """A closure the confidence path calls with the LIVE logits: {pose_index: pae_loss}.

        KEYED BY POSE INDEX because the evaluator's pose-to-sample join is resolved after the
        forward; the caller turns pose i into sample ks[i]. See the module docstring for why
        label row 0 is correct for every pose.

        Returns None when this unit has no label on disk, so a fold without labels evaluates
        exactly as it does today instead of failing.
        """
        t = self.torch
        lab_p = os.path.join(self.labels_dir, pdb, f"free_{seed}.npz")
        if not os.path.exists(lab_p):
            self.n_no_label += 1
            return None
        lab = np.load(lab_p, allow_pickle=True)

        def compute(pae_logits, feat, x_pred):
            out = {}
            lab_hf = t.as_tensor(np.asarray(lab["lf_has_frame"])).to(feat["asym_id"].device)
            m = self._mask(pdb, feat["asym_id"], lab_hf)
            if m is None:
                # EXCLUDED AND COUNTED, never reported as a number. PAELoss reduces by
                # sum(pair_mask), so an all-zero mask divides by eps and returns a meaningless
                # finite value -- the worst outcome for a metric.
                self.n_unmasked += int(pae_logits.shape[0])
                return out
            self._keep["m"] = m
            try:
                # ONE LABEL DICT FOR THE UNIT, built once. minimal_loss_inputs moves arrays to the
                # device, so rebuilding it per pose would repeat that transfer five times for an
                # array that is identical across poses.
                # BOTH RETURN VALUES. The first is the NINE feature keys the loss actually
                # reads, taken from the label npz's own lf_* arrays -- frame_atom_index,
                # pae_rep_atom_mask and has_frame among them. The evaluator's `feat` comes from
                # the TRUNK and carries asym_id, has_frame, atom_to_token_idx and is_ligand only,
                # so reading frame_atom_index off it raised KeyError. The mask below is still
                # built from the evaluator's asym_id, because that is the vector whose chain
                # numbering the logits are in.
                lfeat, ld = build_cache_mod.minimal_loss_inputs(
                    lab, LI.pae_target_row(lab), pae_logits.device, t)
                for i in range(int(pae_logits.shape[0])):
                    # THE PAE TERM DIRECTLY, NOT calculate_losses. The aggregate entry point
                    # computes the diffusion weighting from pred_dict["noise_level"] -- a
                    # quantity that exists during training and that this path has no business
                    # inventing (it raised KeyError: 'noise_level' on the first unit). The term's
                    # own signature is exactly the seven arguments below, taken verbatim from
                    # protenix/model/loss.py:1787, so this is the same call the trainer makes
                    # with the same masked calculate_label hooked underneath it.
                    with t.no_grad():
                        v = self.loss_fn.pae_loss(
                            logits=pae_logits[i:i + 1],
                            pred_coordinate=x_pred[i:i + 1].detach(),
                            true_coordinate=ld["coordinate"],
                            coordinate_mask=ld["coordinate_mask"],
                            frame_atom_index=lfeat["frame_atom_index"],
                            rep_atom_mask=lfeat["pae_rep_atom_mask"],
                            has_frame=lfeat["has_frame"],
                        )
                    out[i] = float(v)
                    self.n_scored += 1
            finally:
                self._keep["m"] = None
            return out

        return compute

    def report(self):
        return (f"eval PAE loss: {self.n_scored} pose(s) scored, {self.n_unmasked} with no scored "
                f"chain pair (excluded), {self.n_no_label} unit(s) with no label, "
                f"{self.n_no_row} pose(s) unattributable to a sample")
