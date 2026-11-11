"""CPU gate: the cover-all epoch plan covers every structure, on the REAL manifest.

Asserts the conditions external review named for 216b0630, against phases/epoch_plan.cover_all --
the same function phase_k_train calls, not a copy of it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phases"))
import csv                                                          # noqa: E402

import epoch_plan as EP                                             # noqa: E402

MANIFEST = "train_val_splits/cache_train_structures_extended.csv"
SKIP = set("4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m".split(","))
WORLD = 3


def build_flat():
    """The epoch's units exactly as the trainer enumerates them: (pdb, seed) per declared seed."""
    flat, depth = [], {}
    for r in csv.DictReader(open(MANIFEST)):
        if r["pdb"] in SKIP:
            continue
        sp = (r.get("seeds") or "").strip()
        seeds = (list(range(int(sp.split("-")[0]), int(sp.split("-")[1]) + 1))
                 if "-" in sp else [int(sp)] if sp else [500000])
        depth[r["pdb"]] = len(seeds)
        for sd in seeds:
            flat.append((r["pdb"], f"free_{sd}.npz", None))
    return flat, depth


def main():
    flat, depth = build_flat()
    allp = {p for p, _l, _c in flat}
    singles = {p for p, d in depth.items() if d < WORLD}
    print(f"manifest: {len(allp)} structure(s), {len(flat)} unit(s); "
          f"{len(singles)} with fewer than world={WORLD} seeds")
    fails = []

    # 1. EVERY STRUCTURE IN EVERY EPOCH, singletons above all. This is the defect.
    for ep in range(6):
        cov = EP.coverage(flat, WORLD, 0, ep)
        miss = allp - set(cov)
        if miss:
            fails.append(f"epoch {ep}: {len(miss)} structure(s) never appear, e.g. {sorted(miss)[:5]}")
        ms = singles - set(cov)
        if ms:
            fails.append(f"epoch {ep}: {len(ms)} SINGLE-SEED structure(s) dropped")
    print(f"  {'PASS' if not fails else 'FAIL'}  every structure appears in epochs 0-5 "
          f"({len(singles)}/{len(singles)} singletons)")

    # 2. EVERY UNIT EXACTLY ONCE, minus the truncated tail.
    ep = 0
    per_rank = [EP.cover_all(flat, WORLD, r, 0, ep) for r in range(WORLD)]
    seen = [u for mine in per_rank for u in mine]
    pad = (-len(flat)) % WORLD
    ok_once = len(set(seen)) == len(flat) and len(seen) == len(flat) + pad
    if not ok_once:
        fails.append(f"units: {len(seen)} dealt, {len(set(seen))} distinct, "
                     f"expected {len(flat)} distinct and {len(flat) + pad} dealt")
    print(f"  {'PASS' if ok_once else 'FAIL'}  every unit dealt at least once "
          f"({len(set(seen))} distinct of {len(flat)}; {pad} padded for an equal split)")

    # 3. EQUAL COUNTS PER RANK, or the first all-reduce hangs instead of failing.
    sizes = [len(m) for m in per_rank]
    ok_eq = len(set(sizes)) == 1
    if not ok_eq:
        fails.append(f"per-rank unit counts differ: {sizes}")
    print(f"  {'PASS' if ok_eq else 'FAIL'}  every rank walks the same number of units {sizes}")

    # 4. RANKS HOLD DISJOINT UNITS WITHIN A STEP, and a step spans the intended structures.
    K = 16
    bad_overlap, uniq = 0, []
    for i in range(0, min(len(per_rank[0]), 40 * K), K):
        step = [m[i:i + K] for m in per_rank]
        if len({u for m in step for u in m}) != sum(len(m) for m in step):
            bad_overlap += 1
        uniq.append(len({u[0] for m in step for u in m}))
    ok_dis = bad_overlap == 0
    if not ok_dis:
        fails.append(f"{bad_overlap} step(s) had the same unit on two ranks")
    print(f"  {'PASS' if ok_dis else 'FAIL'}  ranks disjoint within a step; distinct structures "
          f"per step min {min(uniq)} median {sorted(uniq)[len(uniq)//2]} max {max(uniq)} "
          f"(ideal {K * WORLD})")

    # 5. THE OLD BUILDER REALLY DID DROP THEM -- the gate must catch the bug it was written for.
    by = {}
    for p, lp, c in flat:
        by.setdefault(p, []).append((lp, c))
    old_cov = {p for p, u in by.items() if len(u) // WORLD > 0}
    caught = len(allp - old_cov) == len(singles) and len(singles) > 0
    if not caught:
        fails.append("control: the old group-of-world builder did NOT drop the singletons, so "
                     "this gate is not testing what it claims")
    print(f"  {'PASS' if caught else 'FAIL'}  control: the old builder drops "
          f"{len(allp - old_cov)} structure(s), which is the bug")

    print()
    if fails:
        print(f"{len(fails)} FAILURE(S):")
        for f in fails:
            print("   " + f)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
