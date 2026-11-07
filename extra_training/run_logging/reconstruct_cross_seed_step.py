#!/usr/bin/env python
"""WHICH UNITS EACH RANK WALKS AT A GIVEN CROSS-SEED STEP, without spending a GPU hour to find out.

WHY THIS EXISTS. Both uncropped attempts (tztukd, kfthfe) died of CUDA OOM at step 158 of 231, and
trajectory.csv is written by RANK 0 ONLY while the OOM was on RANK 1. Rank 0's step-158 structures
are 276-731 tokens, which cannot need 93 GiB, so attributing the failure to them was wrong -- the
reviewer's first priority was to establish what the FAILING rank actually held. Reaching step 158
again costs 3.2 h of three cards; the assignment is deterministic, so it can be computed instead.

IT CALLS THE REAL PACKER. phases/lr_schedule.pack_cross_seed_steps is imported, not reimplemented:
the 426-vs-232 lr-horizon bug was exactly a second implementation of this packing drifting from the
first, and a reconstruction that disagreed with the trainer would be worse than none.

THE RNG STREAM MUST MATCH CALL FOR CALL, because every shuffle draws from the same Random. The
trainer's order is: shuffle(order), then per packed chunk shuffle(each structure's unit list) and
shuffle(the chunk's units). Any extra or missing draw desynchronises everything after it, which is
why --verify-against is not optional in spirit: it checks the reconstruction against rank 0's
recorded trajectory at the same step, and a mismatch means this file is wrong, not the trainer.
"""
import argparse, collections, csv, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import lr_schedule as _LRS


def expand(txt):
    out = []
    for part in str(txt).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--indices", default="training/prep/all857_indices.csv")
    ap.add_argument("--skip", default="9mx1,4zto,9q5m")
    ap.add_argument("--seed", type=int, default=0, help="--seed as the arm passed it")
    ap.add_argument("--epoch", type=int, default=0, help="the `ep` the builder was called with")
    ap.add_argument("--structs-per-step", type=int, default=2)
    ap.add_argument("--world", type=int, default=3)
    ap.add_argument("--step", type=int, default=158)
    ap.add_argument("--verify-against", default="",
                    help="a trajectory.csv; its rows at --step must equal rank 0's reconstruction")
    a = ap.parse_args()

    skip = {x.strip().lower() for x in a.skip.split(",") if x.strip()}
    tok = {}
    for r in csv.DictReader(open(a.indices)):
        p = (r.get("pdb_id") or "").strip().lower()
        try:
            n = int(float(r.get("num_tokens") or 0))
        except ValueError:
            continue
        if p and n > tok.get(p, 0):
            tok[p] = n

    # `by` AS THE TRAINER BUILDS IT: structure -> its units in manifest/seed order.
    by = {}
    for r in csv.DictReader(open(a.manifest)):
        p = r["pdb"].strip().lower()
        if p in skip:
            continue
        by[p] = expand(r.get("seeds") or "")

    rng_e = random.Random(a.seed + a.epoch)
    order = sorted(by)
    rng_e.shuffle(order)
    S = max(1, a.structs_per_step)
    depth = {p: len(by[p]) for p in order}
    chunks, dropped, U = _LRS.pack_cross_seed_steps(depth, order, S, preserve_order=False)

    steps = []
    for chunk in chunks:
        units = []
        for p in chunk:
            u = by[p][:]
            rng_e.shuffle(u)
            units += [(p, s) for s in u]
        rng_e.shuffle(units)
        steps.append(units)

    print(f"packed {len(steps)} step(s) of {U} unit(s); {len(dropped)} dropped")
    if a.step >= len(steps):
        raise SystemExit(f"STOP: --step {a.step} but only {len(steps)} step(s) packed")

    n_units = len(steps[0])
    base, rem = divmod(n_units, a.world)
    k = a.step
    counts = [base + (1 if ((r - k) % a.world) < rem else 0) for r in range(a.world)]
    units = steps[k]
    per = {}
    for r in range(a.world):
        start = sum(counts[:r])
        per[r] = units[start:start + counts[r]]

    print(f"\n=== step {k}: {len(units)} unit(s), counts {counts} ===")
    worst = {}
    for r in range(a.world):
        mx = max((tok.get(p, 0) for p, _ in per[r]), default=0)
        worst[r] = mx
        print(f"  rank {r}: largest {mx} tokens")
        for p, s in sorted(per[r], key=lambda x: -tok.get(x[0], 0)):
            print(f"      {p:6s} free_{s}  {tok.get(p, 0):5d} tok")
    hi = max(worst, key=lambda r: worst[r])
    print(f"\n  LARGEST UNIT IN THE STEP is {worst[hi]} tokens, on rank {hi}")

    if a.verify_against:
        got = {(r["pdb"].strip().lower(), int(r["unit"].replace("free_", "")))
               for r in csv.DictReader(open(a.verify_against)) if int(r["step"]) == k}
        want = set(per[0])
        # THE RECORDED STEP MAY BE PARTIAL -- it is the step the run died in -- so the check is
        # that every row rank 0 actually wrote is one this reconstruction puts on rank 0.
        extra = got - want
        print(f"\n  VERIFY against {a.verify_against}: rank 0 recorded {len(got)} unit(s) at "
              f"step {k}, reconstruction puts {len(want)} there")
        if extra:
            print(f"  MISMATCH: {len(extra)} recorded unit(s) are NOT in rank 0's reconstruction:")
            for p, s in sorted(extra):
                print(f"      {p} free_{s}")
            return 1
        print(f"  MATCH: every recorded unit is in rank 0's reconstruction "
              f"({len(got)} of {len(want)} walked before the run died)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
