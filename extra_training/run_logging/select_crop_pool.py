#!/usr/bin/env python
"""Pick each structure's final crop set from the pre-built pools, by escalating crop size.

THE RULE, at the owner's direction:

    collect every crop at 256 whose DockQ regret is within --regret of the uncropped selection
    if fewer than --min-crops:  add 384's passing crops, then 512, then 768, then 1250
    if still fewer than --min-crops:  mark the structure UNCROPPED

APPLIED UNIFORMLY, including to the 412 single-seed structures whose regret comes from 5 poses and
is therefore noisy. Special-casing them was considered and rejected: the noise costs a few crops
from a pool that is usually over a hundred, and any structure it pushes under the floor is caught
by the escalation anyway.

ALL PASSING CROPS ARE KEPT, not just --min-crops of them. An arm may run past 40 epochs and a
larger pool costs nothing; --min-crops is the floor that triggers escalation, not a cap.

ESCALATION ADDS CROPS BECAUSE IT ADDS (ANCHOR, SIZE) PAIRS, which is worth stating because the
intuition is usually wrong: the eligible-anchor set is fixed by the 15 A pair-contact rule and does
NOT grow with crop size, so a structure with 74 anchors has 74 crops at every size. What a bigger
size gives is a DIFFERENT mask from each of those same anchors.

Writes <out>/<pdb>.npz with the selected masks and, for each, which size it came from, plus
<out>/selection.csv summarising every structure.

    python -I run_logging/select_crop_pool.py --min-crops 40 --regret 0.05 \
        --out training/crop_final
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = (256, 384, 512, 768, 1250)


def passing(d, thr):
    """Boolean over this pool's crops: valid AND no interface worse than uncropped by > thr.

    An interface whose regret could not be computed (too few poses with DockQ) cannot veto a crop
    -- it contributes no evidence either way, and treating missing evidence as failure would
    silently empty the pools of exactly the structures we know least about.
    """
    reg, rfull, val = d["regret"], d["regret_full"], d["valid"]
    ok = ~np.isnan(reg) & ~np.isnan(rfull)[:, None]
    if not ok.any():
        return val.copy(), False
    return val & ~((reg - rfull[:, None] > thr) & ok).any(0), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="training/crop_pool_{}")
    ap.add_argument("--min-crops", type=int, default=40)
    ap.add_argument("--regret", type=float, default=0.05)
    ap.add_argument("--out", default="training/crop_final")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)

    have = [cs for cs in LADDER if os.path.isdir(a.pools.format(cs))]
    print(f"pools available: {have}")
    pdbs = sorted({f[:-4] for cs in have for f in os.listdir(a.pools.format(cs))
                   if f.endswith(".npz")})

    rows, rung_hist, uncropped = [], collections.Counter(), []
    for pdb in pdbs:
        picked_idx, picked_size, scored = [], [], None
        used = []
        for cs in have:
            p = os.path.join(a.pools.format(cs), f"{pdb}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            ok, was_scored = passing(d, a.regret)
            scored = was_scored if scored is None else (scored or was_scored)
            if ok.any():
                picked_idx.append(d["crops"][ok])
                picked_size.append(np.full(int(ok.sum()), cs, np.int16))
                used.append(cs)
            if sum(len(x) for x in picked_idx) >= a.min_crops:
                break
        n = sum(len(x) for x in picked_idx)
        rung_hist[tuple(used)] += 1
        if n < a.min_crops:
            uncropped.append((pdb, n))
        else:
            # masks from different sizes have different lengths; store ragged as an object array
            np.savez_compressed(
                os.path.join(a.out, f"{pdb}.npz"),
                sizes=np.concatenate(picked_size),
                n_crop=np.int32(n),
                **{f"c{i}": arr for i, arr in enumerate(picked_idx)},
                __meta=json.dumps(dict(min_crops=a.min_crops, regret=a.regret,
                                       ladder=used, scored=bool(scored))))
        rows.append(dict(pdb=pdb, n_selected=n, sizes="+".join(map(str, used)),
                         escalated=int(len(used) > 1), scored=int(bool(scored)),
                         uncropped=int(n < a.min_crops)))

    with open(os.path.join(a.out, "selection.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    nsel = np.array([r["n_selected"] for r in rows])
    esc = sum(r["escalated"] for r in rows)
    print(f"\n{len(rows)} structures")
    print(f"  256 alone sufficed      : {len(rows)-esc-len(uncropped)}  "
          f"({100*(len(rows)-esc-len(uncropped))/len(rows):.1f}%)")
    print(f"  needed escalation       : {esc}")
    print(f"  fell through to UNCROPPED: {len(uncropped)}")
    print(f"  selected crops: min {nsel.min()}  median {np.median(nsel):.0f}  "
          f"mean {nsel.mean():.0f}  max {nsel.max()}")
    print("\n  rungs used:")
    for k, v in sorted(rung_hist.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {'+'.join(map(str, k)) or '(none)':24s} {v:4d}")
    if uncropped:
        print(f"\n  UNCROPPED structures ({len(uncropped)}):")
        for p, n in uncropped[:12]:
            print(f"    {p}  only {n} passing crop(s) across every rung")


if __name__ == "__main__":
    main()
