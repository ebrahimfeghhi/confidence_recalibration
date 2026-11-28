#!/usr/bin/env python
"""Held-out PAE loss, split into good and bad poses -- the same split the training plot uses.

THE SPLIT IS THE TRAINER'S, NOT A NEW ONE. A pose is GOOD when every interface of its structure
is acceptable at the DockQ threshold, BAD when none is, and is counted in NEITHER group when it
straddles -- some interfaces acceptable and some not. That is ~6% of poses and they are excluded
rather than assigned, because one pose carries one PAE loss while its interfaces disagree about
whether it is a success.

WHY IT JOINS ON (pdb, seed, sample) -- AND WHY THE TRAJECTORY'S `pose` COLUMN IS NOT THAT SAMPLE.
results.csv names DockQ by interface, seed and sample. The loss trajectory's `pose` column is the
LABEL ROW index, and label row k is sample_index[k], which is NOT k: measured over the 2,490
validation labels, 2,469 of them (99.2%) are permuted. Reading `pose` as the sample therefore
looked up a different pose's DockQ for essentially every row -- the pose-index-for-sample-index
confusion this project has now paid for four times. The fix is one lookup: the label npz's own
sample_index maps row -> sample, and that is what the join uses. The ALL-poses mean is unaffected
(the same set of losses is averaged either way); the GOOD/BAD split is what was wrong.

So the join is per (pdb, seed, sample) and gathers every interface
of that structure.
"""
import argparse, collections, csv, glob, os, statistics as st
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import label_index as LI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma-separated run dirs holding trajectory.csv")
    ap.add_argument("--results", default="validation/fold/results.csv")
    ap.add_argument("--accept", type=float, default=0.23)
    ap.add_argument("--label", default="")
    ap.add_argument("--labels", default="validation/fold/labels",
                    help="label root whose sample_index maps a trajectory row to its sample")
    a = ap.parse_args()

    by = collections.defaultdict(list)
    for r in csv.DictReader(open(a.results)):
        if r.get("arm") not in (None, "", "free"):
            continue
        pdb = r["iface"].split("_")[0]
        try:
            by[(pdb, int(r["seed"]), int(r["sample"]))].append(float(r["dockq"]))
        except (ValueError, KeyError):
            continue

    good, bad, straddle, nolab = [], [], 0, 0
    _smap = {}
    n_nolabel = n_norow = 0
    for d in [x.strip() for x in a.runs.split(",") if x.strip()]:
        p = os.path.join(d, "trajectory.csv")
        if not os.path.exists(p):
            raise SystemExit(f"STOP: no trajectory.csv in {d}")
        for r in csv.DictReader(open(p)):
            v = r.get("pae_loss")
            if v in (None, "", "nan"):
                continue
            try:
                seed = int(str(r["unit"]).split("_")[-1]); pose = int(r["pose"])
                # ROW -> SAMPLE, THROUGH THE LABEL. Cached per unit: 2,490 npz opens would
                # otherwise become 12,450.
                _k = (r["pdb"], seed)
                if _k not in _smap:
                    _lp = os.path.join(a.labels, r["pdb"], f"free_{seed}.npz")
                    try:
                        _smap[_k] = LI.sample_index_array(
                            np.load(_lp, allow_pickle=True))
                    except Exception:
                        _smap[_k] = None
                _sl = _smap[_k]
                if _sl is None:
                    n_nolabel += 1
                    continue
                if pose >= len(_sl):
                    n_norow += 1
                    continue
                pose = _sl[pose]
            except (ValueError, KeyError):
                continue
            dq = by.get((r["pdb"], seed, pose))
            if not dq:
                nolab += 1; continue
            if all(x >= a.accept for x in dq):   good.append(float(v))
            elif all(x < a.accept for x in dq):  bad.append(float(v))
            else:                                straddle += 1

    n = len(good) + len(bad) + straddle
    if n_nolabel or n_norow:
        print(f"    NOTE {n_nolabel} pose(s) had no label file and {n_norow} "
              f"had a row beyond their label's sample_index; both excluded.")
    print(f"  {a.label or 'val PAE'}: {n} labelled pose(s)"
          f"{f', {nolab} with no DockQ row' if nolab else ''}")
    print(f"    good poses {len(good):>6}   mean pae_loss {st.mean(good):.4f}" if good else "    good: none")
    print(f"    bad  poses {len(bad):>6}   mean pae_loss {st.mean(bad):.4f}" if bad else "    bad: none")
    print(f"    straddling {straddle:>6}   excluded, as in training")
    if good and bad:
        print(f"    ALL        {len(good)+len(bad):>6}   mean pae_loss "
              f"{st.mean(good + bad):.4f}")

if __name__ == "__main__":
    main()
