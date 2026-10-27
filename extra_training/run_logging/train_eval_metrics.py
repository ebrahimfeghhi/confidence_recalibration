#!/usr/bin/env python
"""Score a finished arm on the TRAINING pool and report it beside its validation numbers.

THE QUESTION THIS ANSWERS: does an arm fail because it cannot fit the training set, or because it
fits it and does not generalise? Every eval in this project so far has run on valfull, so nothing
on disk distinguished those two until now.

BUDGET 10, AND IT IS FORCED BY THE DATA, NOT A CHOICE. Training structures were folded at seeds
500000-500009 -- 10 seeds x 5 samples = 50 poses each -- while validation has 30 seeds x 5 = 150.
So a train-pool number is NOT comparable to the budget-30 valfull numbers in EXPERIMENTS.md. This
script therefore also re-aggregates the arm's EXISTING valfull rows at 10 seeds, which costs no
GPU, so the pair printed side by side is measured at the same budget on both sides.

THE ARCHITECTURE FLAGS ARE DERIVED FROM args.json, NEVER PASSED BY HAND. phase_k_eval builds the
confidence head from them; a share arm scored without --pose-perblock-share either trips the
evaluator's state-dict guard or, if the shapes happen to line up, silently scores a different
model. The shell version of this took them as arguments, which is one typo away from a wrong
number that looks fine. The mapping below is the same one phase_k_train.eval_checkpoint uses.

Examples
    # score three finished arms, running the eval where rows are missing
    python -I run_logging/train_eval_metrics.py ip8ga4 dntxmn tv6hue
    # just re-print the table from rows already on disk
    python -I run_logging/train_eval_metrics.py ip8ga4 dntxmn tv6hue --no-run
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 9dzj IS EXCLUDED. It is the one structure in full_train/pool.csv that the arms did NOT
# train on (their selection is structures_no9dzj.csv), and its fold is missing the sample
# JSONs for seed 500001 -- 5 poses with no usable OST chain mapping, which made the
# aggregate refuse 2 of 3 shards and report nothing. Dropping it is right on both counts:
# this is an IN-SAMPLE measurement, and 9dzj is out of sample.
TRAIN_POOL = os.path.join(ROOT, "training", "full_train", "pool_insample.csv")
TRAIN_FOLD = os.path.join(ROOT, "training", "fold")
TRAIN_SEED_BASE = 500000
TRAIN_BUDGET = 10          # 10 seeds is all the training fold has


def arch_flags(a):
    """Exactly phase_k_train.eval_checkpoint's mapping, so the evaluator builds what we trained."""
    f = []
    if a.get("frame_direction"):
        f.append("--frame-direction")
    if a.get("pose_perblock"):
        f.append("--pose-perblock")
        for key, flag in (("pose_perblock_tied", "--pose-perblock-tied"),
                          ("pose_perblock_share", "--pose-perblock-share"),
                          ("pose_perblock_copy", "--pose-perblock-copy")):
            if a.get(key):
                f.append(flag)
    return f


def final_ckpt(rid):
    """The arm's last head_ep<N>_step<S>.pt, by epoch then step -- not by mtime.

    mtime ordering breaks the moment a directory is copied or a file is touched, and picking the
    wrong checkpoint would be invisible in the output.
    """
    best = None
    for p in glob.glob(os.path.join(ROOT, "training", "runs", rid, "head_ep*_step*.pt")):
        b = os.path.basename(p)
        try:
            ep = int(b.split("head_ep")[1].split("_step")[0])
            st = int(b.split("_step")[1].split(".pt")[0])
        except (IndexError, ValueError):
            continue
        if best is None or (ep, st) > best[0]:
            best = ((ep, st), p)
    return best[1] if best else None


def run_eval(rid, ckpt, flags, cards, work, out):
    extra = ["--fold", TRAIN_FOLD,
             "--results", os.path.join(TRAIN_FOLD, "results.csv"),
             "--pool", TRAIN_POOL,
             "--roles", os.path.join(TRAIN_FOLD, "chain_roles.csv"),
             "--seed-base", str(TRAIN_SEED_BASE)] + flags + [
             # The join is a function of the pool and the trunks, so it is cached BESIDE THE POOL:
             # the first arm pays for it and every later arm is free.
             "--join-cache", os.path.join(os.path.dirname(TRAIN_POOL), "join_cache.json"),
             "--pair-iptm-only", "--prefetch"]
    cmd = [sys.executable, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
           "--cards", cards, "--ckpt", ckpt, "--budget", str(TRAIN_BUDGET),
           "--work", work, "--out", out, "--extra", " ".join(extra)]
    print(f"  running: {' '.join(cmd[:10])} ... --extra '{' '.join(extra)}'", flush=True)
    r = subprocess.run(cmd, env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    return r.returncode


def rows_in(d, budget):
    """iface -> [(seed, sample, trained_iptm, shipped_iptm, dockq)] from a shard directory.

    THE SAMPLE INDEX IS KEPT because the canonical tie-break needs it. Without it this could not
    reproduce phase_k_eval's selection at all.
    """
    by = collections.defaultdict(list)
    pat = os.path.join(d, f"rows_b{budget}_*.csv")
    for f in glob.glob(pat):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                by[r["iface"]].append((int(r["seed"]), int(r["sample"]), float(r["iptm"]),
                                       float(r["stored_iptm"]), float(r["dockq"])))
    return by


def summarise(by, seed_base, budget, thresh):
    """Trained vs shipped selection on the SAME interfaces, paired.

    SELECTION AND THRESHOLD ARE phase_k_eval's, EXACTLY. This used `max(..., key=iptm)` and
    `dockq > thresh`; the evaluator uses `max(v, key=lambda t: (iptm, -seed, -sample))` and
    `>= ACCEPTABLE`. Its own comment says the tie-break exists because "row order differs between
    a sharded run and a single-process one, so two poses with equal ipTM selected different DockQ
    depending on how the work was split" -- which is precisely what this script was doing. The two
    disagreed on ocqtyt's training pool: shipped DockQ 0.4205 here against 0.4202 from the
    evaluator, and frac 0.5426 against 0.5436. Caught in review.
    """
    cut = seed_base + budget
    tr, sh, orc = [], [], []
    for v in by.values():
        s = [x for x in v if x[0] < cut]
        if not s:
            continue
        # x = (seed, sample, trained_iptm, shipped_iptm, dockq)
        tr.append(max(s, key=lambda x: (x[2], -x[0], -x[1]))[4])
        sh.append(max(s, key=lambda x: (x[3], -x[0], -x[1]))[4])
        orc.append(max(x[4] for x in s))
    if not tr:
        return None
    d = [a - b for a, b in zip(tr, sh)]
    ft = sum(1 for x in tr if x >= thresh) / len(tr)
    fs = sum(1 for x in sh if x >= thresh) / len(sh)
    return {"n": len(tr), "dockq": statistics.mean(tr), "shipped": statistics.mean(sh),
            "paired": statistics.mean(d),
            "se": (statistics.stdev(d) / len(d) ** 0.5) if len(d) > 1 else float("nan"),
            "frac": ft, "frac_shipped": fs, "oracle": statistics.mean(orc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rids", nargs="+")
    ap.add_argument("--cards", default="4,5,7",
                    help="the eval is sharded one shard per card; three cards is the layout every "
                         "other eval in this project uses and what the timings assume.")
    ap.add_argument("--thresh", type=float, default=0.23)
    ap.add_argument("--no-run", action="store_true",
                    help="aggregate rows already on disk and run nothing")
    a = ap.parse_args()

    out_rows = []
    for rid in a.rids:
        run = os.path.join(ROOT, "training", "runs", rid)
        try:
            args = json.load(open(os.path.join(run, "args.json")))
        except OSError as e:
            print(f"  {rid}: no args.json ({e}); skipping")
            continue
        work = os.path.join(run, "train_pool_eval")
        mj = os.path.join(run, "metric_trainpool.json")
        have = bool(glob.glob(os.path.join(work, f"rows_b{TRAIN_BUDGET}_*.csv")))
        if not have and not a.no_run:
            ck = final_ckpt(rid)
            if not ck:
                print(f"  {rid}: no head_ep*_step*.pt to score; skipping")
                continue
            fl = arch_flags(args)
            print(f"  {rid}: scoring {os.path.basename(ck)} on the train pool, arch {fl}")
            os.makedirs(work, exist_ok=True)
            rc = run_eval(rid, ck, fl, a.cards, work, mj)
            if rc != 0:
                print(f"  {rid}: eval returned {rc}; skipping")
                continue
        tp = summarise(rows_in(work, TRAIN_BUDGET), TRAIN_SEED_BASE, TRAIN_BUDGET, a.thresh)
        # The arm's own validation rows, re-aggregated at the SAME budget. Free: already on disk.
        # ONLY IF THEY ARE valfull. ocqtyt monitored val10's 18 interfaces, and printing those in
        # a column headed "val" beside everyone else's 92 would invite exactly the comparison the
        # project's own notes forbid -- val10 and valfull are never comparable. A different pool
        # is named and skipped rather than silently averaged.
        pool = os.path.basename(os.path.dirname(str(args.get("train_eval_pool") or "")))
        vdirs = sorted(glob.glob(os.path.join(run, "train_eval_shards_ep*")),
                       key=lambda q: int(q.split("_ep")[-1]))
        if pool != "valfull":
            vp, vnote = None, f"val skipped: this arm evaluated on {pool or 'an unknown pool'}"
        elif not vdirs:
            vp, vnote = None, "val skipped: no eval shard rows on disk"
        else:
            vp, vnote = summarise(rows_in(vdirs[-1], 30), 400000, TRAIN_BUDGET, a.thresh), None
        out_rows.append((rid, tp, vp, vnote))

    print()
    print(f"  BUDGET {TRAIN_BUDGET} ON BOTH SIDES (50 poses). Train = in-sample; val = valfull, "
          f"its own rows re-aggregated at 10 seeds.")
    print(f"  {'arm':9s} {'pool':6s} {'n':>4s} {'DockQ':>7s} {'shipped':>8s} {'paired':>9s} "
          f"{'se':>7s} {'frac':>7s} {'fr.ship':>8s} {'oracle':>7s}")
    for rid, tp, vp, vnote in out_rows:
        for label, s in (("train", tp), ("val", vp)):
            if not s:
                why = vnote if label == "val" and vnote else "no rows"
                print(f"  {rid:9s} {label:6s}   ({why})")
                continue
            print(f"  {rid:9s} {label:6s} {s['n']:4d} {s['dockq']:7.4f} {s['shipped']:8.4f} "
                  f"{s['paired']:+9.4f} {s['se']:7.4f} {s['frac']:7.4f} {s['frac_shipped']:8.4f} "
                  f"{s['oracle']:7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
