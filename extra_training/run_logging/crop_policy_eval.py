#!/usr/bin/env python
"""Score a per-structure crop-size POLICY by composing the measured 256/384/512 runs.

NO RESAMPLING. Sections 21 and 22 already measured every interface at all three crop sizes with
the pair-restricted anchor, one shared crop per structure, d0 pinned at the uncropped scored
chain-pair count. A policy that assigns ONE size per structure is therefore a row selection over
those runs -- every number below is a measurement, not a projection, and the denominators are the
same 886 paired interfaces throughout.

WHY ROW SELECTION IS EXACT HERE. The policy keys on the FULL uncropped assembly token count, which
is a property of the structure and identical across the three runs, and the crop is shared by
every interface of a structure -- so "this structure used 384" selects a consistent set of rows,
not a blend of regimes. Composition would NOT be exact for a policy that varied size per interface
within a structure, and this script refuses such a policy rather than quietly averaging.

    python -I run_logging/crop_policy_eval.py
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = {256: "training/runs/crop_ab_pair.csv",
       384: "training/runs/crop_cs384/rank.csv",
       512: "training/runs/crop_cs512/rank.csv"}
MANIFEST = "train_val_splits/cache_train_structures_extended.csv"
SKIP = set("4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m".split(","))

# The per-pose cost fitted on 87ja6a epoch 12 (trajectory_rank0 wall deltas, 10,717 poses):
# t(N) = C0 + K2*N^2 + K3*N^3. C0 is the part that does NOT shrink with the crop -- trunk H2D, the
# optimiser step, the collective, unit setup -- so it is already inside these speedups rather than
# missing from them. At 11,500 poses an epoch, C0 alone is 15.5 min of the measured 94.3.
C0, K2, K3 = 0.081, 4.072e-07, 3.394e-10
EPOCH_MIN, POSES_PER_EPOCH = 94.3, 11500
EVAL_MIN, EVAL_EVERY = 12.5, 5          # the held-out pass does NOT shrink with a training crop


def tcost(n):
    n = np.asarray(n, dtype=float)
    return C0 + K2 * n ** 2 + K3 * n ** 3


def load():
    data = {}
    for cs, p in SRC.items():
        rows = list(csv.DictReader(open(os.path.join(ROOT, p))))
        data[cs] = {(r["pdb"], r["iface"]): r for r in rows}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    tok = np.array([int(float(data[256][k]["n_tokens"])) for k in keys])
    return data, keys, tok


def col(data, cs_per_key, keys, name):
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        v = data[cs_per_key[i]][k][name]
        out[i] = float(v) if v not in ("", "nan") else np.nan
    return out


def speedup(rule):
    """Epoch speedup over uncropped, on the FULL 846-structure training distribution."""
    tk = np.array([int(r["tokens"]) for r in csv.DictReader(open(os.path.join(ROOT, MANIFEST)))
                   if r["pdb"] not in SKIP], dtype=float)
    sizes = np.array([min(rule(n), n) for n in tk], dtype=float)
    return tcost(tk).sum() / tcost(sizes).sum(), sizes.mean(), tk


POLICIES = [
    ("fixed 256", lambda n: 256),
    ("fixed 384", lambda n: 384),
    ("fixed 512", lambda n: 512),
    ("256; 512 if N>800", lambda n: 256 if n <= 800 else 512),
    ("THREE-TIER 256/384/512 at 800,1100", lambda n: 256 if n <= 800 else (384 if n <= 1100 else 512)),
    ("256; 384 if N>600; 512 if N>1000", lambda n: 256 if n <= 600 else (384 if n <= 1000 else 512)),
    ("256; 384 if N>600; 512 if N>800", lambda n: 256 if n <= 600 else (384 if n <= 800 else 512)),
]


def main():
    data, keys, tok = load()
    n = len(keys)
    print(f"{n} paired interfaces over {len({k[0] for k in keys})} structures; "
          f"1000 crops each at every size; pair-restricted anchor; d0 pinned uncropped\n")

    rows = []
    for name, rule in POLICIES:
        cs = [rule(t) for t in tok]
        bad = [c for c in cs if c not in SRC]
        if bad:
            raise SystemExit(f"STOP: policy {name} wants unmeasured crop size(s) {sorted(set(bad))}")
        ret = col(data, cs, keys, "iface_ret_15a_mean")
        usable = col(data, cs, keys, "usable_rate")
        joint = col(data, cs, keys, "joint_usable_rate")
        rho = col(data, cs, keys, "delta")
        reg = col(data, cs, keys, "regret_delta")
        regsd = col(data, cs, keys, "regret_crop_std")
        regp90 = col(data, cs, keys, "regret_crop_p90")
        worse = col(data, cs, keys, "regret_crop_worse_frac")
        sp, meansz, _ = speedup(rule)
        rows.append(dict(name=name, cs=cs, ret=ret, usable=usable, joint=joint, rho=rho,
                         reg=reg, regsd=regsd, regp90=regp90, worse=worse, sp=sp, sz=meansz))

    print("POLICY                                 ret15A  usable   joint  rho d    reg d   "
          "reg p90  worse%  speedup  mean crop")
    for r in rows:
        print(f"  {r['name']:36s} {np.nanmean(r['ret']):6.3f} {np.nanmean(r['usable']):7.4f} "
              f"{np.nanmean(r['joint']):7.4f} {np.nanmean(r['rho']):+7.4f} "
              f"{np.nanmean(r['reg']):+8.4f} {np.nanmean(r['regp90']):8.4f} "
              f"{100*np.nanmean(r['worse']):6.1f} {r['sp']:8.2f}x {r['sz']:9.1f}")

    three = next(r for r in rows if r["name"].startswith("THREE-TIER"))
    two = next(r for r in rows if r["name"] == "256; 512 if N>800")
    base = next(r for r in rows if r["name"] == "fixed 256")

    print("\n--- per size bin, THREE-TIER vs 256->512@800 vs fixed 256 ---")
    print(f"{'bin':14s} {'n':>5s} {'3t ret':>8s} {'2t ret':>8s} {'3t joint':>9s} {'2t joint':>9s} "
          f"{'3t reg d':>9s} {'2t reg d':>9s}")
    for lo, hi, lab in ((0, 450, "<450"), (450, 600, "450-600"), (600, 800, "600-800"),
                        (800, 1100, "800-1100"), (1100, 10 ** 9, ">1100")):
        s = (tok >= lo) & (tok < hi)
        if not s.sum():
            continue
        print(f"  {lab:12s} {s.sum():5d} {np.nanmean(three['ret'][s]):8.3f} "
              f"{np.nanmean(two['ret'][s]):8.3f} {np.nanmean(three['joint'][s]):9.4f} "
              f"{np.nanmean(two['joint'][s]):9.4f} {np.nanmean(three['reg'][s]):+9.4f} "
              f"{np.nanmean(two['reg'][s]):+9.4f}")

    print("\n--- worst cases under the THREE-TIER policy ---")
    order = np.argsort(three["joint"])
    for i in order[:6]:
        print(f"  {keys[i][1]:14s} N={tok[i]:5d} crop={three['cs'][i]:4d}  "
              f"joint {three['joint'][i]:.3f}  ret15A {three['ret'][i]:.3f}  "
              f"reg d {three['reg'][i]:+.4f}")
    print(f"  interfaces with joint < 0.90: {int((three['joint'] < .9).sum())} of {n} "
          f"({100*np.nanmean(three['joint'] < .9):.1f}%)   "
          f"vs 256->512@800: {int((two['joint'] < .9).sum())}   "
          f"vs fixed 256: {int((base['joint'] < .9).sum())}")

    print("\n--- THREE-TIER minus 256->512@800 ---")
    for k, lab in (("joint", "joint usable"), ("ret", "interface kept 15A"),
                   ("reg", "regret delta"), ("rho", "rho delta")):
        d = np.nanmean(three[k]) - np.nanmean(two[k])
        print(f"  {lab:22s} {d:+.4f}")
    print(f"  {'speedup':22s} {three['sp']-two['sp']:+.2f}x   "
          f"({two['sp']:.2f}x -> {three['sp']:.2f}x)")

    print("\n--- end-to-end epoch time, and what the fit does and does not cover ---")
    _, _, tk = speedup(lambda n: 256)
    print(f"  measured uncropped epoch: {EPOCH_MIN:.1f} min over {POSES_PER_EPOCH} poses/rank")
    print(f"  C0 = {C0:.3f} s/pose is the NON-scaling part (trunk H2D, step, collective, setup):")
    print(f"    {C0*POSES_PER_EPOCH/60:.1f} min of the {EPOCH_MIN:.1f} min epoch, already inside "
          f"every speedup above")
    for r in (base, two, three):
        ep = EPOCH_MIN / r["sp"]
        per5 = 5 * ep + EVAL_MIN
        print(f"  {r['name']:36s} epoch {ep:5.1f} min   5 epochs + one held-out eval "
              f"{per5:6.1f} min  (uncropped {5*EPOCH_MIN+EVAL_MIN:.1f})")
    print(f"  CAVEAT: the fit is per-POSE wall deltas on epoch 12. It does not model the held-out "
          f"eval ({EVAL_MIN} min every {EVAL_EVERY} epochs, which does NOT shrink), nor any "
          f"per-epoch cost outside poses. Epoch numbers are therefore an upper bound on the "
          f"saving, though less optimistic than if C0 were omitted.")


if __name__ == "__main__":
    main()
