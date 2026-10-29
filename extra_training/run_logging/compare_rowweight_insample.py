#!/usr/bin/env python
"""In-sample selection, row-weighted arm against the unweighted 60-epoch lineage, SAME population.

THE POPULATION CONTROL IS THE POINT. The lineage trained on 487 structures; nskyne trains on the
368 that have at least one pose above DockQ 0.23. The 119 excluded ones score zero on frac
acceptable BY DEFINITION -- no pose is acceptable, so no model can pick one -- so scoring each arm
on its own manifest would hand nskyne a ~24-point head start that is arithmetic, not learning.
Both arms are therefore scored on pool_winnable.csv: the same 738 interfaces, the same budget, the
same oracle.

WHAT IS COMPARED. Epoch 60 of each arm: 60 epochs of compute each, both after ten epochs at a
1e-4 floor, both on the same 738 interfaces at the same budget against the same oracle.

  lineage  ep60  45q96x/resume_ep60.pt
  nskyne   ep60  nskyne/resume_ep60.pt

BUDGET 10 IS FORCED: the training fold was folded at seeds 500000-500009, so ten poses per
interface is all there is. These numbers are NOT comparable with any valfull figure, which runs at
30 against its own oracle.

Usage:
  python -I run_logging/compare_rowweight_insample.py --dry-run     # show the plan, touch nothing
  python -I run_logging/compare_rowweight_insample.py               # extract, evaluate, tabulate
  python -I run_logging/compare_rowweight_insample.py --report-only  # tabulate what already exists
"""
import argparse
import shutil
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/ebrahim/Protenix/.venv/bin/python"
POOL = "training/full_train/pool_winnable.csv"
FOLD = "training/fold"
BUDGET = 10
SEED_BASE = 500000
# ONE CACHE PER POOL. The join is a function of the pool and the trunks, so it is keyed to this
# pool rather than reusing pool_insample's -- a mismatched cache would silently join the wrong
# interfaces.
JOIN_CACHE = "training/full_train/join_cache_winnable.json"

# EPOCH 55 FOR BOTH, at the user's direction. nskyne was stopped at 55 complete epochs rather than
# run to 60, so 55 is the last checkpoint it has; the lineage has a resume_ep55.pt as well, so the
# pair is epoch-matched rather than 55-against-60. Both are past their lr floor by then -- the
# lineage sat at 1e-4 from epoch 51, nskyne's cosine reached 1e-4 at epoch 50.
#
# THE ROW-WEIGHTED ARM IS EVALUATED FIRST, also at the user's direction, so its number lands
# before the control's rather than after a second eval pass.
#
# WHAT STILL DIFFERS BESIDES THE ROW WEIGHTING: the lr schedule (the lineage reached 1e-4 by
# hand-stepped drops at 1e-2 to epoch 40, 1e-3 to 50; nskyne by a cosine over epochs 20-50) and
# the training manifest (487 structures against 368). Three changes, not one. Stated here because
# the table cannot show it.
ARMS = [("rowweight", 55, "training/runs/nskyne/resume_ep55.pt"),
        ("lineage", 55, "training/runs/45q96x/resume_ep55.pt")]

ap = argparse.ArgumentParser()
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--pool-mean", action="store_true",
                help="also write the MEAN-over-rows pooled ipTM as an iptm_mean column, so both "
                     "poolings come out of ONE pass. Selection and the self-check stay on the "
                     "max; the mean is compared offline.")
ap.add_argument("--arm", default="", help="label:epoch:bundle -- score just this one checkpoint")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--report-only", action="store_true")
ap.add_argument("--allow-busy", action="store_true",
                help="evaluate even if cards_busy.py reports occupancy. OFF by default: landing "
                     "this on top of a live training arm slows it and risks an OOM.")
a = ap.parse_args()
if a.arm:
    _l, _e, _b = a.arm.split(":", 2)
    ARMS = [(_l, int(_e), _b)]
OUT_D = os.path.join(ROOT, "training", "runs", "rowweight_compare")


def sh(cmd, **kw):
    print("    $ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def pose_flags(bundle):
    """The architecture flags this bundle was trained with, read from its own guard.

    NOT hardcoded: the lineage gained --frame-direction at epoch 48, so an epoch-50 bundle and an
    epoch-35 one need different flags, and guessing would load the wrong module shape.
    """
    import torch
    g = (torch.load(bundle, map_location="cpu", weights_only=False).get("guard") or {})
    out = []
    if g.get("pose_perblock"):
        out.append("--pose-perblock")
        for k, f in (("pose_perblock_tied", "--pose-perblock-tied"),
                     ("pose_perblock_share", "--pose-perblock-share"),
                     ("pose_perblock_copy", "--pose-perblock-copy")):
            if g.get(k):
                out.append(f)
    if g.get("frame_direction"):
        out.append("--frame-direction")
    return out, g


def evaluate(label, ep, bundle):
    tag = f"{label}_ep{ep}"
    out = os.path.join(OUT_D, f"{tag}_b{BUDGET}.json")
    if os.path.exists(out):
        print(f"  {tag}: exists, skipping")
        return 0
    if not os.path.exists(os.path.join(ROOT, bundle)):
        print(f"  {tag}: SKIP -- {bundle} does not exist yet")
        return 0
    # HEAD EXTRACTION USES extract_heads.py'S REAL INTERFACE: a run id plus --epochs, writing
    # head_ep<N>_step<S>.pt INTO THE RUN DIRECTORY. This used to pass --bundle/--out, which that
    # script has never accepted; the call failed with "unrecognized arguments" the first time this
    # script was actually executed, because every prior invocation had been --dry-run and the dry
    # path never reaches here. Reusing the script rather than unwrapping the bundle inline keeps
    # its "no 'head' key" guard and its atomic write.
    head = os.path.join(OUT_D, f"{tag}_head.pt")
    if not os.path.exists(head):
        rid = os.path.basename(os.path.dirname(bundle))
        if sh([PY, "-I", os.path.join(ROOT, "run_logging", "extract_heads.py"),
               rid, "--epochs", str(ep)]):
            print(f"  {tag}: head extraction FAILED")
            return 1
        cand = sorted(glob.glob(os.path.join(ROOT, "training", "runs", rid,
                                             f"head_ep{ep}_step*.pt")))
        if not cand:
            cand = sorted(glob.glob(os.path.join(ROOT, "training", "runs", rid,
                                                 f"head_ep{ep}.pt")))
        if len(cand) != 1:
            print(f"  {tag}: expected exactly one head_ep{ep} file for {rid}, found {cand}")
            return 1
        # COPIED, not symlinked, so the evaluated object cannot change underneath a later re-read.
        shutil.copyfile(cand[0], head + ".tmp")
        os.replace(head + ".tmp", head)
        print(f"  {tag}: head {os.path.basename(cand[0])} -> {os.path.basename(head)}")
    pf, g = pose_flags(os.path.join(ROOT, bundle))
    print(f"  {tag}: pose flags {pf or ['none']}  "
          f"(row-weighted: frac {g.get('pae_row_frac', 'absent')})")
    extra = ["--fold", FOLD, "--results", os.path.join(FOLD, "results.csv"),
             "--pool", POOL, "--roles", os.path.join(FOLD, "chain_roles.csv"),
             "--seed-base", str(SEED_BASE)] + pf + [
             "--join-cache", JOIN_CACHE, "--pair-iptm-only", "--prefetch"] + (
             ["--pool-mean"] if a.pool_mean else [])
    return sh([PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
               "--cards", a.cards, "--ckpt", head, "--budget", str(BUDGET),
               "--work", os.path.join(OUT_D, tag), "--out", out,
               "--extra", " ".join(extra)])


def report():
    rows = []
    for label, ep, _b in ARMS:
        p = os.path.join(OUT_D, f"{label}_ep{ep}_b{BUDGET}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            d["label"], d["epoch"] = label, ep
            rows.append(d)
    if not rows:
        print("  nothing evaluated yet")
        return
    n = {r["n_interfaces"] for r in rows}
    print(f"\n  IN-SAMPLE SELECTION on {sorted(n)} interface(s) at budget {BUDGET}")
    if len(n) > 1:
        print("  WARNING: the arms were scored on DIFFERENT interface counts; the comparison "
              "below is NOT paired and should not be read.")
    orc = {round(r["oracle_mean_dockq"], 4) for r in rows}
    base = {round(r["baseline_mean_dockq"], 4) for r in rows}
    print(f"  oracle mean DockQ {sorted(orc)}   shipped-ipTM baseline {sorted(base)}")
    print(f"\n  {'arm':>10} {'ep':>3}  {'mean DockQ':>11} {'frac acc':>9}  "
          f"{'vs shipped':>11} {'% headroom':>11}")
    for r in sorted(rows, key=lambda x: (x["epoch"], x["label"])):
        d = r["mean_dockq"] - r["baseline_mean_dockq"]
        hr = r["oracle_mean_dockq"] - r["baseline_mean_dockq"]
        print(f"  {r['label']:>10} {r['epoch']:>3}  {r['mean_dockq']:>11.4f} "
              f"{r['frac_acc']:>9.4f}  {d:>+11.4f} {100*d/hr if hr else 0:>10.1f}%")
    for ep in (60,):
        pair = {r["label"]: r for r in rows if r["epoch"] == ep}
        if len(pair) == 2:
            dd = pair["rowweight"]["mean_dockq"] - pair["lineage"]["mean_dockq"]
            df = pair["rowweight"]["frac_acc"] - pair["lineage"]["frac_acc"]
            print(f"\n  EPOCH {ep}, row-weighted minus lineage: "
                  f"mean DockQ {dd:+.4f}, frac acceptable {df:+.4f}")
    print("\n  Both arms scored on the SAME 738 winnable interfaces, so the 119 unwinnable "
          "structures the\n  row-weighted arm drops cannot inflate its frac acceptable.")


def main():
    os.makedirs(OUT_D, exist_ok=True)
    print(f"  plan: {len(ARMS)} checkpoint(s), pool {POOL}, budget {BUDGET}, cards {a.cards}")
    for label, ep, b in ARMS:
        here = "present" if os.path.exists(os.path.join(ROOT, b)) else "NOT YET"
        print(f"    {label:>10} ep{ep}: {b}  [{here}]")
    if a.dry_run:
        return 0
    if a.report_only:
        report()
        return 0
    busy = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "cards_busy.py")],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    if busy and busy[-1].strip() != "0" and not a.allow_busy:
        print(f"\n  REFUSING: cards_busy reports {busy[-1].strip()}. Pass --allow-busy to score "
              f"anyway, which will slow the live arm and risks an OOM on a shared card.")
        return 1
    rc = 0
    for label, ep, b in ARMS:
        rc |= evaluate(label, ep, b)
    report()
    return rc


if __name__ == "__main__":
    sys.exit(main())
