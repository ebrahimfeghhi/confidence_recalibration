"""Do the PAE labels differ across poses of a seed? Whole training set, not a sample.

THE QUESTION THIS DECIDES. phase_j_build_cache permutes the ground truth PER POSE, so a unit holds
five labels. Protenix instead permutes ONCE and scores every sample against that one label, which
is why its loss takes a single label and why batching poses is possible there and not here.

If the per-pose permutation only ever swaps SYMMETRIC SIDE-CHAIN atoms, the PAE target -- built
from the per-token representative atoms -- is untouched, and sharing pose 0's label costs exactly
nothing for a PAE-only arm. A sample of 60 said that. This checks every unit, because "nothing
moved in 60 files" and "nothing ever moves" are different claims and only the second one licenses
dropping four fifths of the labels.
"""
import glob
import os
import sys

import numpy as np

TOL = 1e-6


def one(f):
    """(status, max_rep_displacement). status: ok | moved | skip."""
    try:
        z = np.load(f)
    except Exception:                                               # noqa: BLE001
        return "skip", 0.0
    if "label_coordinate" not in z.files or "lf_pae_rep_atom_mask" not in z.files:
        return "skip", 0.0
    c = np.asarray(z["label_coordinate"])
    m = np.asarray(z["label_coordinate_mask"]).astype(bool)
    rep = np.asarray(z["lf_pae_rep_atom_mask"]).reshape(-1).astype(bool)
    if c.ndim < 3 or c.shape[0] < 2:
        return "skip", 0.0
    worst = 0.0
    for j in range(1, c.shape[0]):
        sh = m[0] & m[j] & rep
        if not sh.any():
            continue
        d = np.linalg.norm(c[0][sh] - c[j][sh], axis=-1)
        worst = max(worst, float(d.max()))
        # A REP ATOM RESOLVED IN ONE POSE AND NOT ANOTHER also changes the target, so it counts.
        if (rep & (m[0] ^ m[j])).any():
            worst = max(worst, float("inf"))
    return ("ok" if worst <= TOL else "moved"), worst


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fs = sorted(glob.glob("training/fold/labels/*/free_*.npz"))[shard::nsh]
    ok = moved = skip = 0
    worst, worst_f = 0.0, ""
    for f in fs:
        st, w = one(f)
        if st == "ok":
            ok += 1
        elif st == "moved":
            moved += 1
            if w > worst:
                worst, worst_f = w, f
        else:
            skip += 1
    print(f"shard {shard}: files {len(fs)}  identical_PAE_target {ok}  MOVED {moved}  skip {skip}"
          f"  worst {worst:.6f} {os.path.basename(os.path.dirname(worst_f))}/"
          f"{os.path.basename(worst_f) if worst_f else ''}", flush=True)


if __name__ == "__main__":
    main()
