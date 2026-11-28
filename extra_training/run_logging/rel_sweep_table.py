#!/usr/bin/env python
"""HELD-OUT VAL ACROSS THE ALPHA SWEEP, one row per arm, with Arm E and shipped as the controls.

WHY A SEPARATE SCRIPT. arm_table.py prints ONE arm against shipped, which is the right shape while
an arm is live and the wrong shape for a sweep: the question here is not "did this arm beat
shipped" but "which alpha, if any, beat the arm that has no ranking term at all". That comparison
only means anything side by side, and reprinting six tables and eyeballing them is how a 0.0003
difference gets called a win.

EVERY NUMBER IS READ FROM metric_valfull83_ep<N>.json, never retyped. mean_dockq is max-pooled and
mean_dockq_select_mean is mean-pooled; shipped is baseline_mean_dockq, which the evaluator writes
from the SAME rows, so it is a matched baseline rather than a remembered constant.

AN ARM WITH NO EVAL YET PRINTS AS "--". A sweep is read while it is still running, and a missing
file means "not yet", not zero.
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ALPHAS = (25, 50, 100, 200, 250, 300)


def read(rid, epoch, pool):
    p = os.path.join(ROOT, f"training/runs/{rid}/metric_{pool}_ep{epoch}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def rid_for(alpha):
    p = os.path.join(ROOT, f"training/runs/rel_alpha{alpha}_rid")
    return open(p).read().strip() if os.path.exists(p) else None


def cell(d, key):
    v = d.get(key) if d else None
    return "--" if v is None else f"{v:.4f}"


def pct(d, key):
    v = d.get(key) if d else None
    return "--" if v is None else f"{v * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()

    rows, shipped = [], None
    base = read("73rf4v", a.epoch, "valfull83")
    if base:
        shipped = base
        rows.append(("Arm E (no ranking term)", "73rf4v", base))
    for al in ALPHAS:
        rid = rid_for(al)
        if not rid:
            rows.append((f"rel alpha {al}", "not started", None))
            continue
        rows.append((f"rel alpha {al}", rid, read(rid, a.epoch, "valfull83")))

    print(f"HELD-OUT val, epoch {a.epoch} -- mean-pooled ipTM in parentheses; shipped is max-pooled\n")
    hdr = ("| arm | run | mean DockQ | frac acceptable |",
           "|---|---|---|---|")
    if shipped:
        print(f"shipped on the same rows: {shipped.get('baseline_mean_dockq')} / "
              f"{(shipped.get('baseline_frac_acc') or 0) * 100:.1f}%  "
              f"(n={shipped.get('n_interfaces')})\n")
    for h in hdr:
        print(h)
    best = None
    for name, rid, d in rows:
        q = f"{cell(d, 'mean_dockq')} ({cell(d, 'mean_dockq_select_mean')})"
        f = f"{pct(d, 'frac_acc')} ({pct(d, 'frac_acc_select_mean')})"
        print(f"| {name} | `{rid}` | {q} | {f} |")
        if d:
            vals = [x for x in (d.get("mean_dockq"), d.get("mean_dockq_select_mean"))
                    if x is not None]
            if vals and (best is None or max(vals) > best[0]):
                best = (max(vals), name)
    if best:
        print(f"\nbest val mean DockQ so far: {best[0]:.4f}  ({best[1]})")


if __name__ == "__main__":
    main()
