#!/usr/bin/env python
"""Does the union crop over all nsample poses equal the pose-0 crop, and what does it cost?

THE EVIDENCE BEHIND a16911de, which changed the training crop's DEFAULT from the union over all
poses to pose 0 alone. That commit is only safe while the two masks agree, so the claim is
checkable rather than asserted: this times both and compares the masks token by token.

Usage: python -I run_logging/crop_pose_equivalence.py [n_units]   (CPU only, no GPU needed)
       n_units <= 0 audits EVERY (structure, seed) unit in the selection, which is what
       "validated" requires; a positive n_units is a fast sample of that same population.

WHY THEY CAN AGREE AT ALL. The crop is cut from the LABEL coordinates -- ground truth -- and all
nsample poses share one ground truth, so they share one interface. The label is nonetheless stored
per pose because Protenix permutes it to match each prediction, and that permutation is real: over
60 structures the label coordinates move up to 2.572819 A between poses, on 468 of 480 pose pairs.

It does not reach the crop because it is a WITHIN-RESIDUE permutation -- symmetric atoms swapped
inside one token (a Phe ring flip, an Asp carboxyl, an Arg guanidinium). The token's SET of atom
positions is unchanged, so its distance to every other chain is unchanged, so its interface status
is unchanged. Measured: 0 tokens out of all of those had their atom-position set change, and the
representative atoms -- which seed the margin -- move 0.000000 A.

THE PERMUTATION THAT WOULD BITE is a whole-chain swap between identical copies, which would move
the interface to different token INDICES and let a pose-0 crop cut away what pose 1 is scored on.
Only possible where a structure has two or more chains of equal token count: 24 of 120 sampled,
77 of 491 overall. Never observed. That is why the union was there, and why this script exists:
assert_pose_invariant_crop() in phase_j_build_cache is the in-run gate, this is the offline audit.
"""
import csv
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC

# j583i8's OWN settings and OWN selection, so the numbers are about the run this decides for.
CONTACT, MARGIN = 5.0, 10.0
SELECTION = "training/full_train/structures_no9dzj.csv"
SKIP = set("9mx1 9q5m 9dzf 9b9j 9e7x 9q5v 9q7b 8vba 9bbn 9dc3".split())


def units(limit):
    """EVERY (structure, seed) UNIT, not one per structure.

    The first version took the first seed of each structure, which is the wrong population for
    this question: the permutation being audited is chosen per POSE within a unit, and nothing
    says seed 500000's permutation is seed 500007's. 160 structures at one seed each certified
    1/10th of the units it appeared to. Review of 9e838321 raised it.

    limit <= 0 means every unit in the selection.
    """
    sel = [r["pdb"].strip() for r in csv.DictReader(open(SELECTION))]
    sel = [p for p in sel if p and p not in SKIP]
    out = []
    for pdb in sel:
        ld = os.path.join("training/fold/labels", pdb)
        td = os.path.join("training/fold/trunks", pdb)
        if not (os.path.isdir(ld) and os.path.isdir(td)):
            continue
        for f in sorted(os.listdir(ld)):
            if f.endswith(".npz") and os.path.exists(os.path.join(td, f)):
                out.append((pdb, os.path.join(ld, f), os.path.join(td, f)))
    # SHUFFLED ACROSS THE WHOLE UNIT LIST, so a partial audit is a sample of structures AND seeds
    # rather than the alphabetical head of either.
    random.seed(0)
    random.shuffle(out)
    if limit > 0:
        if len(out) < limit:
            # FEWER THAN ASKED IS A FAILURE, NOT A SMALLER PASS. "0 of 40 differ" reads the same
            # whether 40 units were checked or 4, and a missing cache would quietly shrink it.
            raise SystemExit(f"STOP: asked for {limit} unit(s) but only {len(out)} exist under "
                             f"training/fold for this selection; a short audit certifies less "
                             f"than it appears to.")
        out = out[:limit]
    return out


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    us = units(n_want)
    print(f"{len(us)} unit(s) from {SELECTION}, contact {CONTACT} margin {MARGIN}")
    t_all = t_one = 0.0
    differ = nocrop = 0
    sz_all = sz_one = ntok = 0
    for pdb, lp, tp in us:
        lab = np.load(lp, allow_pickle=True)
        z = np.load(tp)
        # LOADED OUTSIDE THE TIMED REGION. The flag changes the crop, not the disk read, and a
        # read folded in would swamp the difference it is meant to show.
        t0 = time.perf_counter()
        ka, _ = BC.interface_crop_tokens(z, lab, CONTACT, MARGIN, all_poses=True)
        t_all += time.perf_counter() - t0
        t0 = time.perf_counter()
        ko, _ = BC.interface_crop_tokens(z, lab, CONTACT, MARGIN, all_poses=False)
        t_one += time.perf_counter() - t0
        # ASYMMETRIC None IS A DIFFERENCE, NOT A SKIP. "union crops, pose 0 cannot" is precisely
        # the pose-dependent interface this audit exists to catch, and the first version skipped
        # it and could still exit 0. Only both-None is a legitimate skip: neither definition finds
        # an interface, so the trainer falls back to the full structure either way. Review of
        # 9e838321 found it.
        if (ka is None) != (ko is None):
            differ += 1
            print(f"  {pdb} {os.path.basename(lp)}: CROP DEFINED FOR ONE AND NOT THE OTHER "
                  f"(union={'None' if ka is None else int(ka.sum())}, "
                  f"pose0={'None' if ko is None else int(ko.sum())})")
            continue
        if ka is None:
            nocrop += 1
            print(f"  {pdb} {os.path.basename(lp)}: nocrop under both definitions")
            continue
        if not np.array_equal(ka, ko):
            differ += 1
            print(f"  {pdb} {os.path.basename(lp)}: MASKS DIFFER  union {int(ka.sum())} vs pose0 {int(ko.sum())}, "
                  f"{int((ka & ~ko).sum())} token(s) the pose-0 crop would drop")
        sz_all += int(ka.sum())
        sz_one += int(ko.sum())
        ntok += int(ka.shape[0])

    n = len(us)
    print(f"\nunion over all poses : {t_all:8.3f} s   {t_all / n * 1000:7.1f} ms/unit")
    print(f"pose 0 only          : {t_one:8.3f} s   {t_one / n * 1000:7.1f} ms/unit")
    print(f"saving               : {t_all - t_one:8.3f} s   "
          f"{(t_all - t_one) / n * 1000:7.1f} ms/unit   ({100 * (t_all - t_one) / t_all:.1f}%)")
    print(f"kept tokens          : union {sz_all}  pose0 {sz_one}  of {ntok}")
    print(f"masks differ on      : {differ} of {n}   (nocrop {nocrop})")
    # A DIFFERENCE IS A STOP, NOT A NOTE. The commit this backs is only safe while the count is
    # zero, so a non-zero count exits non-zero and a CI caller sees it.
    if differ:
        raise SystemExit(f"STOP: the pose-0 crop differs from the union on {differ} unit(s); "
                         f"a16911de's premise does not hold on this data.")


if __name__ == "__main__":
    main()
