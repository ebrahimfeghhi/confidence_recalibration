#!/usr/bin/env python
"""Choose the softmax temperature for --rank-form expected, by measuring what it does.

    L = - sum_i softmax(ipTM_i / tau) * DockQ_i          over every pose of one interface
    dL/ds_i = -(p_i / tau) * (DockQ_i - E[DockQ])        p = softmax(ipTM/tau)

THE DIRECTION OF THE PUSH DOES NOT DEPEND ON tau. Every pose above the group's expected DockQ is
pushed up and every pose below it is pushed down, at any temperature. What tau changes is the
MAGNITUDE p_i, and p_i is large exactly where ipTM is already large. So a cold tau concentrates
the gradient on the poses the model already ranks top, and starves the poses whose ipTM is low --
which are precisely the poses that have to be LIFTED for selection to improve.

That tension is what this script measures, and it is why an earlier recommendation here was
wrong. Choosing tau to make E[DockQ] numerically match the argmax DockQ (columns `E[DockQ]` and
`gap` below) selects a COLD temperature, because a cold softmax is a better numerical stand-in
for argmax. But matching the number is bookkeeping: it rewards a loss that only looks at poses
already on top. The quantity that matters for improving selection is `underrated%` -- the share
of gradient reaching high-DockQ, low-ipTM poses -- and it moves the opposite way.

THE SCORES ARE THE SHIPPED HEAD's, which is the right starting point: all three temperature arms
begin from the shipped weights, so this is the gradient allocation at step 0. The `iptm` column of
training/fold/results.csv and validation/fold/results.csv was verified EXACTLY equal to the
`stored_iptm` the evaluator writes (2,700 overlapping poses, max |diff| 0.0), so the label files
are a valid source and no eval shards are needed.

COLUMNS
  n_if          interfaces contributing (>= 2 poses and a non-constant DockQ)
  poses/if      median poses per interface -- the population argmax selects from
  E[DockQ]      mean over interfaces of -L, the loss's own view of selection quality
  argmax        mean over interfaces of the DockQ of the highest-ipTM pose: the REAL metric
  gap           E[DockQ] - argmax. Near zero means the surrogate tracks the metric numerically.
                NOT the criterion -- see above.
  eff#          median 1/sum(p^2), the effective number of poses receiving gradient, out of
                poses/if. Small means the step is driven by a handful of poses.
  top5%         share of total |gradient| on the five poses that carry most of it
  underrated%   SHARE OF |GRADIENT| REACHING POSES WITH DockQ ABOVE AND ipTM BELOW THE
                INTERFACE MEDIAN. These are the poses that must be lifted. THIS IS THE COLUMN
                TO CHOOSE ON.
  oracle%       share of |gradient| on the single highest-DockQ pose of the interface
  oracle_rank   median rank of that pose when poses are ordered by |gradient|, out of poses/if.
                1 means the loss is pushing hardest on the best available pose.

Run:
  /home/ebrahim/Protenix/.venv/bin/python -I run_logging/measure_softmax_tau.py
  ... --restrict-to training/overfit30/structures.csv     # the 30 structures the arms train on
  ... --taus 0.02,0.05,0.10                               # a specific set
"""
import argparse
import collections
import csv
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import rank_loss as R  # noqa: E402

TRAIN = "training/fold/results.csv"
VAL = "validation/fold/results.csv"
TAUS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50)


def load(paths, restrict=None):
    """{iface: [(iptm, dockq), ...]} pooled over EVERY seed and sample of that interface.

    POOLED ACROSS SEEDS DELIBERATELY: deployment picks argmax over all poses of an interface
    regardless of which seed produced them, so that is the group the softmax must cover.
    """
    by = collections.defaultdict(list)
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    it, dq = float(r["iptm"]), float(r["dockq"])
                except (TypeError, ValueError):
                    continue
                if it != it or dq != dq:          # NaN rows exist where a pose failed to score
                    continue
                iface = r["iface"]
                if restrict is not None and iface.split("_")[0] not in restrict:
                    continue
                by[iface].append((it, dq))
    # An interface with one pose has nothing to select between, and one with a constant DockQ
    # has no gradient at any temperature -- both would dilute every column with zeros.
    return {k: v for k, v in by.items() if len(v) >= 2 and len(set(d for _, d in v)) > 1}


def stats_at(by, tau):
    out = collections.defaultdict(list)
    for v in by.values():
        s = [x[0] for x in v]
        d = [x[1] for x in v]
        loss, c, n = R.expected_dockq(s, d, tau=tau)
        ac = [abs(x) for x in c]
        tot = sum(ac) or 1e-30
        z = [x / tau for x in s]
        m = max(z)
        ex = [math.exp(q - m) for q in z]
        sm = sum(ex)
        p = [q / sm for q in ex]
        md, mi = st.median(d), st.median(s)
        io = max(range(len(d)), key=lambda i: d[i])
        order = sorted(range(len(ac)), key=lambda i: -ac[i])
        out["n"].append(n)
        out["e"].append(-loss)
        out["am"].append(max(v, key=lambda x: x[0])[1])
        out["eff"].append(1.0 / sum(q * q for q in p))
        out["top5"].append(sum(sorted(ac, reverse=True)[:5]) / tot)
        out["und"].append(sum(a for a, dd, ss in zip(ac, d, s) if dd > md and ss < mi) / tot)
        out["orc"].append(ac[io] / tot)
        out["orank"].append(1 + order.index(io))
    return out


def table(name, by, taus):
    print(f"\n{name}: {len(by)} interfaces, {st.median([len(v) for v in by.values()]):.0f} "
          f"poses/interface (median), {sum(len(v) for v in by.values())} poses")
    print("  tau     E[DockQ]  argmax    gap       eff#    top5%   underrated%  oracle%  oracle_rank")
    for t in taus:
        o = stats_at(by, t)
        print(f"  {t:<7.3f} {st.mean(o['e']):.4f}    {st.mean(o['am']):.4f}    "
              f"{st.mean(o['e']) - st.mean(o['am']):+.4f}   {st.median(o['eff']):7.1f} "
              f"{100 * st.mean(o['top5']):6.1f}% {100 * st.mean(o['und']):10.1f}% "
              f"{100 * st.mean(o['orc']):7.2f}% {st.median(o['orank']):10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--taus", default=",".join(str(t) for t in TAUS))
    ap.add_argument("--restrict-to", default=None,
                    help="a structures.csv; keeps only interfaces whose pdb is listed")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",") if x.strip()]

    restrict = None
    if a.restrict_to:
        with open(a.restrict_to) as fh:
            rd = csv.DictReader(fh)
            key = "pdb" if "pdb" in (rd.fieldnames or []) else (rd.fieldnames or ["pdb"])[0]
            restrict = {r[key].strip() for r in rd if r.get(key)}
        print(f"restricted to {len(restrict)} structure(s) from {a.restrict_to}")

    tr = load([a.train], restrict)
    va = load([a.val], restrict)
    bo = load([a.train, a.val], restrict)
    for nm, by in (("TRAINING SET", tr), ("VALIDATION SET", va), ("BOTH COMBINED", bo)):
        if by:
            table(nm, by, taus)
        else:
            print(f"\n{nm}: no interfaces after filtering")
    print("\nCHOOSE ON underrated% -- the gradient share reaching poses that must be LIFTED "
          "(DockQ above, ipTM below the interface median).\n`gap` is numerical fidelity to "
          "argmax and moves the OPPOSITE way; it is not the criterion.")


if __name__ == "__main__":
    main()
