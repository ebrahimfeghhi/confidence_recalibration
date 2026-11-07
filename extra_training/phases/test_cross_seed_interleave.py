#!/usr/bin/env python
"""THE PER-RANK SPLIT OF A CROSS-SEED STEP IS BALANCED, asserted rather than simulated once.

WHY THIS TEST EXISTS. build_epoch_cross_seed used to shuffle a step's units globally and then hand
each rank a CONTIGUOUS slice, so the shuffle was the rank assignment. For a step of two 15-seed
structures that meant a random split of 15+15 -- one rank could walk 8 units of the larger
structure and another 2. Per-unit memory and time go as tokens^2 (measured on this population:
peak_GB = 2.02 + 2.896e-05 * tokens^2), so the ranks finished at different times and every one of
them waited at the step's all-reduce.

IT TESTS THE PROPERTY, NOT A NUMBER. The claim is "each rank gets the same count of every
structure whenever that structure's depth divides world", which is exactly what makes the balance
structural instead of lucky. A test that only checked an imbalance percentage would pass on a
dealer that happened to look even on one seed.

THE NEGATIVE CONTROL IS THE POINT. Asserting the new dealer is balanced proves nothing unless the
old global shuffle is shown to be unbalanced on the same input -- the "assert the feature CHANGES
something" rule. Without it this file would pass against the code it was written to replace.
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_schedule as _LRS


def split(units, world, k=0):
    """phase_k_train's own per-rank split, as it appears there."""
    base, rem = divmod(len(units), world)
    counts = [base + (1 if ((r - k) % world) < rem else 0) for r in range(world)]
    out = []
    for r in range(world):
        start = sum(counts[:r])
        out.append(units[start:start + counts[r]])
    return out


def counts_per_structure(slice_):
    d = {}
    for p, _s in slice_:
        d[p] = d.get(p, 0) + 1
    return d


fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("=== 1. two 15-seed structures over 3 ranks: exactly 5 of each, per rank ===")
lists = [[("A", s) for s in range(15)], [("B", s) for s in range(15)]]
units = _LRS.deal_round_robin(lists)
check("step keeps all 30 units", len(units) == 30, f"got {len(units)}")
check("it is a permutation, nothing invented or lost",
      sorted(units) == sorted(x for l in lists for x in l))
for r, sl in enumerate(split(units, 3)):
    c = counts_per_structure(sl)
    check(f"rank {r} holds 5 A and 5 B", c == {"A": 5, "B": 5}, str(c))

print("\n=== 2. the negative control: a global shuffle is NOT balanced ===")
worst = 0
for seed in range(200):
    rng = random.Random(seed)
    u = [x for l in lists for x in l]
    rng.shuffle(u)
    for sl in split(u, 3):
        c = counts_per_structure(sl)
        worst = max(worst, abs(c.get("A", 0) - c.get("B", 0)))
check("over 200 seeds the old shuffle produces a lopsided rank", worst >= 4,
      f"worst A-B gap was only {worst}")
print(f"        (worst A-vs-B gap on a rank under the old shuffle: {worst} of 10 units)")

print("\n=== 3. the rotation's leader is randomised, the balance is not ===")
leaders = set()
for seed in range(50):
    rng = random.Random(seed)
    ls = [list(x) for x in lists]
    rng.shuffle(ls)
    u = _LRS.deal_round_robin(ls)
    leaders.add(u[0][0])
    for sl in split(u, 3):
        if counts_per_structure(sl) != {"A": 5, "B": 5}:
            fails.append("balance held under a shuffled leader")
check("both structures lead at least once", leaders == {"A", "B"}, str(leaders))
check("balance holds for every shuffled leader",
      "balance held under a shuffled leader" not in fails)

print("\n=== 4. depths that do NOT divide world: still the best possible, never worse ===")
# 3 structures of depth 10 over 3 ranks: 30 units, each rank should get 10 with 3/3/4-ish mixes
lists3 = [[(p, s) for s in range(10)] for p in "ABC"]
u = _LRS.deal_round_robin(lists3)
check("30 units preserved", len(u) == 30)
for r, sl in enumerate(split(u, 3)):
    c = counts_per_structure(sl)
    check(f"rank {r} sees all 3 structures", len(c) == 3, str(c))
    check(f"rank {r} is within one unit of even", max(c.values()) - min(c.values()) <= 1, str(c))

print("\n=== 5. a step of 1-seed structures: unchanged, and that is expected ===")
# 30 distinct depth-1 structures. No assignment can give three ranks the same structures.
lists1 = [[(f"S{i}", 0)] for i in range(30)]
u = _LRS.deal_round_robin(lists1)
check("30 units preserved", len(u) == 30)
check("each rank gets 10 distinct structures",
      all(len(counts_per_structure(sl)) == 10 for sl in split(u, 3)))
check("no structure is split across ranks (depth 1 cannot be)",
      all(set(counts_per_structure(sl).values()) == {1} for sl in split(u, 3)))

print("\n=== 6. the dealer does not mutate its input ===")
src = [[("A", 1), ("A", 2)], [("B", 1)]]
snap = [list(x) for x in src]
_LRS.deal_round_robin(src)
check("caller's lists are intact", src == snap)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {sorted(set(fails))}")
    sys.exit(1)
print("ALL CHECKS PASSED")
