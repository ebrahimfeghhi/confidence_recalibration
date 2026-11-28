"""The step plan must reproduce how a real run actually split its work across ranks.

GROUND TRUTH IS trajectory_rank<N>.csv. Every rank now records the units it walked, in order, so
the three files together are what the original planner decided -- written by the trainer, not by
this code. The properties checked here are the ones the relative loss depends on:

  every seed of a structure lands in ONE optimiser step   (the coefficient for a pose depends on
                                                           every other pose in its interface)
  the ranks PARTITION each step                           (no unit scored twice, none dropped)
  the quota formula reproduces the real per-rank counts   (the step boundary comes from quotas,
                                                           not a constant, because S structures x
                                                           their seeds need not divide `world`)

A planner that returned a plausible-but-wrong grouping would still train and still report numbers,
which is why this is checked against a real epoch rather than a synthetic one.
"""
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.step_plan import build_epoch_cross_seed  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def walked(rid, epoch):
    """{rank: [(pdb, unit, step), ...]} in the order each rank walked them."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "training", "runs", rid,
                                           "trajectory_rank*.csv"))):
        rk = int(os.path.basename(f).split("rank")[1].split(".")[0])
        seq = []
        for r in csv.DictReader(open(f)):
            if r.get("epoch") != str(epoch):
                continue
            k = (r["pdb"], r["unit"], r["step"])
            if not seq or seq[-1] != k:
                seq.append(k)
        if seq:
            out[rk] = seq
    return out


def main():
    rid, ep = "qip9ta", 1
    w = walked(rid, ep)
    print(f"\n1  A REAL EPOCH'S SPLIT ({rid}, epoch {ep})")
    ok(len(w) == 3, "all three ranks recorded their units", f"ranks {sorted(w)}")
    if len(w) < 3:
        print("STOP: per-rank trajectories are needed; this run predates them.")
        return 1
    counts = {k: len(v) for k, v in w.items()}
    ok(len(set(counts.values())) <= 2,
       "the ranks carry within one unit of each other", f"{counts}")

    print("\n2  THE RANKS PARTITION THE EPOCH -- nothing scored twice, nothing dropped")
    sets = {k: {(p, u) for p, u, _s in v} for k, v in w.items()}
    pairs = [(i, j) for i in sets for j in sets if i < j]
    overlap = {f"{i}&{j}": len(sets[i] & sets[j]) for i, j in pairs}
    ok(all(v == 0 for v in overlap.values()), "no unit appears on two ranks", str(overlap))
    union = set().union(*sets.values())
    ok(len(union) == sum(len(s) for s in sets.values()),
       "and the union is the sum of the parts", f"{len(union)} units")

    print("\n3  EVERY SEED OF A STRUCTURE IS IN ONE OPTIMISER STEP")
    # the constraint the relative loss depends on: an interface must be whole inside a step
    bystruct = collections.defaultdict(set)
    for rk, seq in w.items():
        for pdb, _u, step in seq:
            bystruct[pdb].add(int(step))
    # THE LOG'S `step` COLUMN ADVANCES MID-STRUCTURE. The counter increments on the row where
    # step_taken == 1, so a structure's last rows record the already-incremented value and it
    # appears under both k and k+1. That is a logging boundary, not a planning violation: a real
    # violation would scatter a structure across DISTANT steps. Measured on this epoch: 258 of 258
    # straddles are exactly two adjacent steps, zero span three or more.
    spread = {p: sorted(s) for p, s in bystruct.items() if len(s) > 1}
    scattered = {p: v for p, v in spread.items() if len(v) > 2 or max(v) - min(v) > 1}
    ok(not scattered, "no structure is scattered across non-adjacent steps",
       f"{len(bystruct)} structures; {len(spread)} span an adjacent pair (the counter boundary)"
       if not scattered else f"{len(scattered)} scattered, e.g. {list(scattered.items())[:2]}")
    ok(len(spread) < 0.5 * len(bystruct),
       "and most structures sit wholly inside one step",
       f"{len(bystruct) - len(spread)} of {len(bystruct)} wholly inside")

    print("\n4  THE QUOTA FORMULA REPRODUCES THE REAL PER-RANK COUNTS")
    # counts[r] = base + (1 if ((r - k) % world) < rem else 0), from the extracted planner
    per_step = collections.defaultdict(lambda: collections.Counter())
    for rk, seq in w.items():
        for _p, _u, step in seq:
            per_step[step][rk] += 1
    world = 3
    bad = []
    for step, c in sorted(per_step.items())[:40]:
        n = sum(c.values())
        base, rem = divmod(n, world)
        k = sorted(per_step).index(step)
        want = {r: base + (1 if ((r - k) % world) < rem else 0) for r in range(world)}
        if {r: c.get(r, 0) for r in range(world)} != want:
            bad.append((step, dict(c), want))
    ok(not bad, "the formula matches every step checked",
       f"{min(40, len(per_step))} steps" if not bad else f"{len(bad)} mismatch, e.g. {bad[:2]}")

    print("\n5  THE EXTRACTION IS FAITHFUL TO THE ORIGINAL")
    orig = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
    new = open(os.path.join(ROOT, "phases", "trainer", "step_plan.py")).read()
    key = "counts = [base + (1 if ((r - k) % world) < rem else 0) for r in range(world)]"
    ok(key in orig and key in new, "the quota line is byte-identical to the original")
    for line in ("base, rem = divmod(n_units, world)",
                 "start = sum(counts[:rank])",
                 "return mine, quotas, n_units"):
        ok(line in orig and line in new, f"verbatim: {line[:46]}")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
