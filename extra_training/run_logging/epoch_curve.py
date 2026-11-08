#!/usr/bin/env python
"""ONE definition of "this arm's per-epoch training curve", for every plotter to share.

WHY THIS MODULE EXISTS. trajectory.csv is written by RANK 0 ONLY, and its bare `pae_loss` column is
one pose's value. Averaging those rows equally therefore gives a rank-local per-pose mean -- over
the third of the structures rank 0 happened to walk, unbiased but far noisier than the real thing,
and NOT the run's global PAE. The trainer already reduces `batch_<term>` across every rank for each
optimiser step and records how many poses contributed in `batch_n` (phase_k_train.py:3939), so the
batch_n-weighted mean over `step_taken == 1` rows is the global per-pose mean.

This defect has now been diagnosed THREE times in this repository -- in plot_train_insample.py,
then again in plot_uncropped_progress.py, then again in plot_rel_arm.py, whose figure was sent to
the user before anyone noticed. Each time it was fixed in one file and the fix was not propagated.
Hence a shared helper plus run_logging/test_epoch_curve.py, which fails if any plot_*.py builds an
epoch mean out of a per-pose column again.

Rows for steps that were NOT taken are excluded rather than averaged in: a step that failed or was
skipped has no gradient behind it, and its batch value describes work that never updated the model.
The rank-0 fallback is kept only for runs predating the batch columns and every return value says
which source it came from, so a noisier curve can never be silently read as the global one.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: columns that exist per POSE in trajectory.csv and must never be epoch-averaged directly
PER_POSE = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss",
            "pae_loss_unweighted")


def run_chain(rid):
    """Every run in the arm, oldest first, resolved from args.json's recorded --resume."""
    import plot_three
    return list(plot_three.run_chain(rid))


def epoch_means(rid, term="pae_loss", chain=True):
    """{epoch: (mean, n_poses, source)} for one arm. source is "all-ranks" or "rank0-fallback".

    A LATER RUN WINS on any epoch both wrote: a resume re-runs the epoch it was interrupted in, so
    the parent's partial rows for that epoch must not be averaged in with the full ones.
    """
    rids = run_chain(rid) if chain else [rid]
    out = {}
    for r_ in rids:
        p = os.path.join(ROOT, "training", "runs", r_, "trajectory.csv")
        if not os.path.exists(p):
            continue
        agg, fb = collections.defaultdict(list), collections.defaultdict(list)
        with open(p) as fh:
            for row in csv.DictReader(fh):
                try:
                    ep = int(row["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    fb[ep].append(float(row[term]))
                except (KeyError, TypeError, ValueError):
                    pass
                if (row.get("step_taken") or "").strip() != "1":
                    continue
                try:
                    v, n = float(row["batch_" + term]), float(row["batch_n"])
                except (KeyError, TypeError, ValueError):
                    continue
                if n > 0:
                    agg[ep].append((v, n))
        for ep in set(agg) | set(fb):
            b = agg.get(ep) or []
            if b:
                tot = sum(n for _v, n in b)
                out[ep] = (sum(v * n for v, n in b) / tot, int(tot), "all-ranks")
            elif fb.get(ep):
                out[ep] = (sum(fb[ep]) / len(fb[ep]), len(fb[ep]), "rank0-fallback")
    return out


def partial_epochs(means, frac=0.98):
    """Epochs whose pose count is short of the arm's full epoch -- draw these hollow."""
    if not means:
        return set()
    full = max(n for _m, n, _s in means.values())
    return {e for e, (_m, n, _s) in means.items() if n < frac * full}


if __name__ == "__main__":
    _rid = sys.argv[1] if len(sys.argv) > 1 else "5dgrsu"
    _term = sys.argv[2] if len(sys.argv) > 2 else "pae_loss"
    m = epoch_means(_rid, _term)
    _part = partial_epochs(m)
    print(f"{_rid} chain {run_chain(_rid)}  term {_term}")
    print(f"{'ep':>3} {'mean':>10} {'poses':>8}  source")
    for e in sorted(m):
        mu, n, src = m[e]
        print(f"{e:>3} {mu:>10.4f} {n:>8} {src}" + ("  PARTIAL" if e in _part else ""))
