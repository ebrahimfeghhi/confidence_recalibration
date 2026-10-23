#!/usr/bin/env python
"""Derive an L2-SP lambda for a new trainable set from a --grad-probe, instead of by hand.

    anchor gradient norm = 2 * lambda * ||W - W0||          (see phase_k_train, the L2-SP block)

So the anchor's strength RELATIVE TO THE TASK is  2*lambda*||dW|| / |g_task|, and carrying a
lambda across a change of trainable set only preserves that ratio if neither ||dW|| nor |g_task|
moved. Both move, usually in the same direction, which is how a lambda that worked becomes a
lambda that silently pins the weights.

TWO FACTORS, BOTH MEASURED:

  task gradient    |g_rank| is read from the probe of each configuration. If the new set's task
                   gradient is smaller, a fixed lambda counts for MORE against it.

  parameter count  ||dW||^2 sums over all trainable parameters, so at the same PER-PARAMETER
                   drift a set with k times as many parameters has ||dW|| sqrt(k) times larger,
                   and the anchor gradient scales with it.

                       lambda_new = lambda_old * (|g_new| / |g_old|) / sqrt(n_new / n_old)

WHY PER-PARAMETER DRIFT IS THE RIGHT INVARIANT, and its limit. It keeps each individual weight
about as free to move as it was, which is what "the same anchor strength" ought to mean when the
only change is how many copies of a matrix exist. It is NOT a claim about the FUNCTION: four
injection sites each moved by d change the head's output more than one site moved by d, so if the
goal were to hold the functional change fixed, the right lambda would be tighter than this. That
is a different experiment and the choice is recorded rather than hidden.

WORKED, for the arm this was written for (zfahpb against h7msge):
  h7msge  10,240 params, |g_rank| 7.96e-02 (probe ev3d6u), lambda 1e-3
  zfahpb  40,960 params, |g_rank| 4.02e-02 (probe ygpn6t)
  ratio 0.505 / sqrt(4) = 0.253  ->  lambda 2.53e-4, used as 2.5e-4

LAMBDA IS STILL NOT TUNED. This only transfers a chosen lambda to a new parameter set on equal
terms; it does not say the original was right. docs/softmax_tau_choice.md makes the same point
about tau.

Run:
  ... run_logging/l2sp_lambda.py --from-run h7msge --from-probe ev3d6u \\
                                 --to-run   zfahpb --to-probe   ygpn6t
  ... run_logging/l2sp_lambda.py --old-lambda 1e-3 --old-grad 7.96e-02 --old-params 10240 \\
                                 --new-grad 4.02e-02 --new-params 40960
"""
import argparse
import json
import math
import os
import re
import sys

RUNS = os.path.join("training", "runs")
# "    |g_rank|     4.0207e-02   ratio 0.4727"  -- the ACCUMULATED block, which is what the
# optimiser sees. The per-pose lines in the same report are a different (larger) number and
# taking one of those would overstate the task gradient; the probe says so itself.
G_RANK = re.compile(r"\|g_rank\|\s+([0-9.]+e?[-+]?[0-9]*)")


def probe_grad(rid):
    """|g_rank| from a probe run's log: the accumulated value, which the report prints repeatedly
    (once per rank). All ranks print the same reduced figure, so any of them will do, but they
    must all agree -- a disagreement would mean the gradient was not reduced before logging."""
    p = os.path.join(RUNS, rid, "run.log")
    if not os.path.exists(p):
        raise SystemExit(f"STOP: no run.log at {p}")
    vals = []
    with open(p, errors="replace") as fh:
        for line in fh:
            m = G_RANK.search(line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        raise SystemExit(f"STOP: {p} has no |g_rank| line -- was it run with --grad-probe?")
    lo, hi = min(vals), max(vals)
    if hi - lo > 1e-9 * max(1.0, hi):
        raise SystemExit(f"STOP: {rid} reports disagreeing |g_rank| across ranks: {lo} .. {hi}. "
                         f"That would mean the gradient was not reduced before logging.")
    return vals[0]


def run_params(rid):
    """Trainable parameter count from the run's own log, not recomputed and not assumed."""
    p = os.path.join(RUNS, rid, "run.log")
    if not os.path.exists(p):
        raise SystemExit(f"STOP: no run.log at {p}")
    pat = re.compile(r"--train-only \[[^\]]*\]: \d+ of \d+ head tensor\(s\), ([\d,]+) of")
    with open(p, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                return int(m.group(1).replace(",", ""))
    raise SystemExit(f"STOP: {p} has no --train-only line to read a parameter count from")


def run_lambda(rid):
    a = json.load(open(os.path.join(RUNS, rid, "args.json")))
    v = a.get("l2_sp")
    if not v:
        raise SystemExit(f"STOP: {rid} has l2_sp = {v!r}; it was not an anchored run")
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run"), ap.add_argument("--from-probe")
    ap.add_argument("--to-run"), ap.add_argument("--to-probe")
    ap.add_argument("--old-lambda", type=float), ap.add_argument("--old-grad", type=float)
    ap.add_argument("--old-params", type=int), ap.add_argument("--new-grad", type=float)
    ap.add_argument("--new-params", type=int)
    a = ap.parse_args()

    lam = a.old_lambda if a.old_lambda else run_lambda(a.from_run)
    g_o = a.old_grad if a.old_grad else probe_grad(a.from_probe)
    n_o = a.old_params if a.old_params else run_params(a.from_run)
    g_n = a.new_grad if a.new_grad else probe_grad(a.to_probe)
    n_n = a.new_params if a.new_params else run_params(a.to_run)

    r_g = g_n / g_o
    r_n = math.sqrt(n_n / n_o)
    out = lam * r_g / r_n
    print(f"  old: lambda {lam:g}   |g_task| {g_o:.4e}   {n_o:,} params")
    print(f"  new:                  |g_task| {g_n:.4e}   {n_n:,} params")
    print(f"  task-gradient factor  {r_g:.4f}      (smaller task gradient -> fixed lambda bites harder)")
    print(f"  parameter-count factor {r_n:.4f}     (sqrt of the ratio; ||dW|| grows with it)")
    print(f"\n  lambda_new = {lam:g} * {r_g:.4f} / {r_n:.4f} = {out:.4g}")
    # Rounded, because carrying four significant figures implies a precision the probe does not
    # have -- it is one accumulation window at one parameter state.
    for cand in (1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2):
        if abs(math.log10(cand / out)) < math.log10(1.6):
            print(f"  nearest value worth using: {cand:g}")
            break


if __name__ == "__main__":
    main()
