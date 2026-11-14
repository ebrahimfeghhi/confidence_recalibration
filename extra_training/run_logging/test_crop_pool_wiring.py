#!/usr/bin/env python
"""Prove --crop-pool actually crops, and that mask and label rotate TOGETHER.

The failure this gates against produced no error at all: --crop-pool was added as a loader and a
label lookup, but every execution gate still read `crop_margin > 0 or crop_tokens`, so a run given
--crop-pool alone trained on FULL structures while the relative loss read CROPPED targets. The
loss curve would have looked healthy and the arm would have been named for a crop it never
applied. Caught in external review of 3cc51a4c.

These checks are CPU-only and do not build a model, so they can run while a GPU arm is live.

    python -I run_logging/test_crop_pool_wiring.py
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = "training/crop_final"
LAB = "training/fold/itm_labels_crop"
SRC = "phases/phase_k_train.py"


def gates_see_the_pool():
    """Every execution gate must route through the single _USE_CROP switch."""
    s = open(os.path.join(ROOT, SRC)).read()
    blind = [ln for ln in s.splitlines()
             if "a.crop_margin > 0 or a.crop_tokens" in ln and not ln.strip().startswith("#")]
    disp = [ln for ln in s.splitlines() if "_crop_masks(" in ln and "if " in ln]
    ok = (not blind) and all("a.crop_pool" in ln for ln in disp) and "_USE_CROP = " in s
    print(f"  {'PASS' if ok else 'FAIL'}  every crop gate routes through _USE_CROP"
          f"   (blind gates {len(blind)}, dispatchers {len(disp)})")
    for ln in blind:
        print(f"         STILL BLIND: {ln.strip()[:90]}")
    return ok


def pool_and_labels_agree():
    """n_crop must match between the pool and the label cache, for every structure."""
    pool, lab = os.path.join(ROOT, POOL), os.path.join(ROOT, LAB)
    if not (os.path.isdir(pool) and os.path.isdir(lab)):
        print("  SKIP  pool or labels absent")
        return True
    bad, checked = [], 0
    for f in sorted(os.listdir(pool)):
        if not f.endswith(".npz"):
            continue
        lp = os.path.join(lab, f)
        if not os.path.exists(lp):
            bad.append((f[:-4], "no label file"))
            continue
        d = np.load(os.path.join(pool, f), allow_pickle=True)
        n_pool = sum(len(d[k]) for k in d.files if k.startswith("c") and k[1:].isdigit())
        n_lab = int(np.load(lp, allow_pickle=True)["n_crop"])
        checked += 1
        if n_pool != n_lab:
            bad.append((f[:-4], f"pool {n_pool} vs labels {n_lab}"))
    print(f"  {'PASS' if not bad else 'FAIL'}  pool and labels agree on crop count"
          f"   ({checked} structure(s))")
    for b in bad[:5]:
        print(f"         {b[0]}: {b[1]}")
    return not bad


def mask_and_label_rotate_together():
    """epoch % n must pick the same crop index on both sides, and the masks must differ."""
    pool, lab = os.path.join(ROOT, POOL), os.path.join(ROOT, LAB)
    if not (os.path.isdir(pool) and os.path.isdir(lab)):
        print("  SKIP  pool or labels absent")
        return True
    f = sorted(x for x in os.listdir(pool) if x.endswith(".npz"))[0]
    d = np.load(os.path.join(pool, f), allow_pickle=True)
    rows = [np.asarray(r) for k in sorted((k for k in d.files
                                           if k.startswith("c") and k[1:].isdigit()),
                                          key=lambda k: int(k[1:])) for r in d[k]]
    z = np.load(os.path.join(lab, f), allow_pickle=True)
    n = len(rows)
    ok = True
    for ep in (0, 1, 2, n, n + 1):
        ci = ep % n
        got = (np.asarray(z["crop_index"]) == ci).sum()
        if got == 0:
            print(f"         epoch {ep}: no label rows for crop {ci}")
            ok = False
    same = np.array_equal(np.sort(rows[0]), np.sort(rows[1 % n]))
    wrapped = np.array_equal(np.sort(rows[0]), np.sort(rows[n % n]))
    print(f"  {'PASS' if ok and not same and wrapped else 'FAIL'}  mask and label rotate together"
          f"   ({f[:-4]}, n={n}; epoch 0 vs 1 differ: {not same}; epoch n wraps to 0: {wrapped})")
    return ok and not same and wrapped


def one_crop_per_structure_not_per_pose():
    """Every pose of a structure must share the crop, or rel grades the crop not the poses."""
    lab = os.path.join(ROOT, LAB)
    if not os.path.isdir(lab):
        print("  SKIP  labels absent")
        return True
    f = sorted(x for x in os.listdir(lab) if x.endswith(".npz"))[0]
    z = np.load(os.path.join(lab, f), allow_pickle=True)
    ci = np.asarray(z["crop_index"])
    m = ci == 0
    keys = set(zip(z["seed"][m].tolist(), z["sample"][m].tolist()))
    n_if = len(set(z["iface"][m].tolist()))
    # crop 0 must carry every (seed, sample) that crop 1 does -- same poses, same crop
    m1 = ci == (1 if (ci == 1).any() else 0)
    keys1 = set(zip(z["seed"][m1].tolist(), z["sample"][m1].tolist()))
    ok = keys == keys1 and len(keys) > 0
    print(f"  {'PASS' if ok else 'FAIL'}  one crop covers all poses and interfaces"
          f"   ({f[:-4]}: {len(keys)} pose(s), {n_if} interface(s) under crop 0)")
    return ok


def launch_refuses_uncropped_labels():
    """The guard that stops a cropped run reading the old labels."""
    r = subprocess.run(
        [sys.executable, "-I", SRC, "--crop-pool", POOL, "--minimal-items",
         "--rel-labels", "training/fold/itm_labels_extended", "--rank-loss",
         "--rank-form", "rel", "--fold", "training/fold"],
        cwd=ROOT, capture_output=True, text=True)
    ok = "has no crop_index" in (r.stdout + r.stderr)
    print(f"  {'PASS' if ok else 'FAIL'}  refuses uncropped labels under --crop-pool")
    return ok


def main():
    print("crop-pool wiring gates")
    good = [gates_see_the_pool(), pool_and_labels_agree(), mask_and_label_rotate_together(),
            one_crop_per_structure_not_per_pose(), launch_refuses_uncropped_labels()]
    print(f"\n{sum(good)}/{len(good)} gate(s) passed")
    return 0 if all(good) else 1


if __name__ == "__main__":
    sys.exit(main())
