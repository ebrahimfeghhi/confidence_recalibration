#!/usr/bin/env python
"""Gate: the evaluator's crop IS the trainer's crop, and the flag cannot silently do nothing.

WHY. phase_k_eval grew a --crop-margin so a head can be scored on the inputs it trained on. If
that crop differs from phase_k_train's by any parameter, the "cropped" number measures a third
thing that neither the training curve nor the uncropped eval can be compared against -- and it
would look perfectly reasonable. So this re-derives the crop from the SAME library functions the
evaluator calls and checks it against the TRAINER'S OWN recorded crop_sizes_rank*.csv, which is
ground truth written by the run itself.

CPU ONLY. The crop is a function of the trunk, the label and (contact, margin, all_poses) -- no
model, no device, no RNG -- which is exactly why it can be gated for free.
"""
import csv
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import numpy as np                                                           # noqa: E402
import phase_j_build_cache as BC                                             # noqa: E402

RUN = os.path.join(ROOT, "training", "runs", "o2qczq")
FOLD = os.path.join(ROOT, "training", "fold")
CONTACT, MARGIN, ALL_POSES = 5.0, 10.0, False      # o2qczq's own --crop-contact / --crop-margin
N = int(os.environ.get("CROP_GATE_N", "25"))
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


rec = []
for f in sorted(os.listdir(RUN)) if os.path.isdir(RUN) else []:
    if f.startswith("crop_sizes_rank") and f.endswith(".csv"):
        rec += list(csv.DictReader(open(os.path.join(RUN, f))))
if not rec:
    print("  SKIP: no crop_sizes_rank*.csv to check against")
    sys.exit(0)
check("the trainer recorded crops to check against", len(rec) > 1000, f"{len(rec)} record(s)")
# EVERY RECORD IS A CROP. If the trainer ever scored a whole structure this gate's premise -- that
# the training distribution is cropped -- would be wrong, so it is asserted rather than assumed.
check("the trainer cropped EVERY pose",
      all(r.get("status", "").strip() == "cropped" for r in rec),
      f"{sum(r.get('status','').strip() != 'cropped' for r in rec)} not cropped")

random.seed(0)
ok = bad = 0
first_bad = []
for r in random.sample(rec, min(N, len(rec))):
    tp = os.path.join(FOLD, "trunks", r["pdb"], r["unit"] + ".npz")
    lp = os.path.join(FOLD, "labels", r["pdb"], r["unit"] + ".npz")
    if not (os.path.exists(tp) and os.path.exists(lp)):
        continue
    z, lab = np.load(tp), np.load(lp, allow_pickle=True)
    kt, _ka = BC.interface_crop_tokens(z, lab, CONTACT, MARGIN, all_poses=ALL_POSES)
    got = int(kt.sum()) if kt is not None else int(
        np.asarray(z["ci_feat_asym_id"]).reshape(-1).shape[0])
    if got == int(r["kept_tok"]):
        ok += 1
    else:
        bad += 1
        if len(first_bad) < 3:
            first_bad.append(f"{r['pdb']}/{r['unit']} eval {got} vs trainer {r['kept_tok']}")
check(f"the eval's crop reproduces the trainer's on {ok + bad} sampled unit(s)",
      bad == 0 and ok > 0, f"{ok} exact, {bad} mismatched " + "; ".join(first_bad))

# THE PARAMETERS MATTER, which is the other half: a gate that passes whatever margin is used would
# not be protecting anything. A different margin must give a DIFFERENT crop.
diffs = 0
for r in random.sample(rec, min(8, len(rec))):
    tp = os.path.join(FOLD, "trunks", r["pdb"], r["unit"] + ".npz")
    lp = os.path.join(FOLD, "labels", r["pdb"], r["unit"] + ".npz")
    if not (os.path.exists(tp) and os.path.exists(lp)):
        continue
    z, lab = np.load(tp), np.load(lp, allow_pickle=True)
    k10, _ = BC.interface_crop_tokens(z, lab, CONTACT, MARGIN, all_poses=ALL_POSES)
    k25, _ = BC.interface_crop_tokens(z, lab, CONTACT, 25.0, all_poses=ALL_POSES)
    if k10 is not None and k25 is not None and int(k10.sum()) != int(k25.sum()):
        diffs += 1
check("margin 10 and margin 25 give different crops", diffs > 0,
      f"{diffs} of the sampled unit(s) differ")

# AND THE CROP MUST ACTUALLY REMOVE SOMETHING, or "cropped" and "uncropped" are the same number.
kept_frac = [int(r["kept_tok"]) / int(r["n_tok"]) for r in rec[:5000] if int(r["n_tok"]) > 0]
med = sorted(kept_frac)[len(kept_frac) // 2]
check("the crop removes a large fraction of tokens", med < 0.9, f"median kept {med:.3f}")

print(f"\n  {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks pass'}")
sys.exit(1 if FAIL else 0)
