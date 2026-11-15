#!/usr/bin/env python
"""The FIXED in-sample slice: one file of item ids, used by every scoring run.

WHY FIXED. A slice re-derived per scoring run moves with the thing it measures -- an item whose
bin changes because the selector changed would silently enter or leave the denominator. The ids and
their bins are written once and read thereafter.

ONLY 15-SEED ITEMS. An item whose structure has one loadable trunk offers 5 candidates, not 75, so
its bin means something different and it is not comparable with the rest. Those are excluded and
counted.

THE THIN BINS ARE TAKEN WHOLE because they are the regime of interest and there are few of them;
the easy bin is sampled, stratified by how many of its seeds are entirely good.
"""
import argparse
import collections
import csv
import glob
import random

ACC = 0.23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--trunks", default="training/fold/trunks")
    ap.add_argument("--out", default="training/slice280/items.csv")
    ap.add_argument("--total", type=int, default=280)
    ap.add_argument("--quota", default="1:25,2-3:20,4+:20")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    have = {}
    cand = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(a.results)):
        if r.get("arm") not in (None, "free"):
            continue
        p = r["iface"].split("_")[0]
        if p not in have:
            have[p] = len(glob.glob(f"{a.trunks}/{p}/free_*.npz"))
        if have[p] < 15:
            continue
        try:
            cand[r["iface"]][r["seed"]].append(float(r["dockq"]))
        except (TypeError, ValueError):
            continue

    n_all = len({i for i in csv.DictReader(open(a.results))} ) if False else None
    thin, easy = [], collections.defaultdict(list)
    for iface, seeds in sorted(cand.items()):
        flat = [d for v in seeds.values() for d in v]
        if not any(d >= ACC for d in flat):
            continue
        k = max(sum(1 for d in v if d >= ACC) for v in seeds.values())
        if k < 5:
            thin.append((iface, f"{k}/5"))
        else:
            n_all_good = sum(1 for v in seeds.values() if sum(1 for d in v if d >= ACC) == 5)
            g = "1" if n_all_good == 1 else ("2-3" if n_all_good <= 3 else "4+")
            easy[g].append((iface, f"5/5 ({g} all-good seeds)"))

    rng = random.Random(a.seed)
    quota = {}
    for part in a.quota.split(","):
        g, n = part.split(":")
        quota[g] = int(n)
    picked = list(thin)
    print(f"items with 15 loadable seeds and >=1 good candidate: "
          f"{len(thin) + sum(len(v) for v in easy.values())}")
    print(f"  thin bins (1/5..4/5) kept whole: {len(thin)}")
    for g in ("1", "2-3", "4+"):
        print(f"  easy, {g} all-good seed(s): {len(easy[g])} available, quota {quota[g]}")

    # FILL TO --total IN THE SAME PROPORTION when the thin bins come up short.
    short = max(0, a.total - len(thin) - sum(quota.values()))
    tot_q = sum(quota.values())
    take = {g: quota[g] + (short * quota[g] // tot_q if short else 0) for g in quota}
    rem = a.total - len(thin) - sum(take.values())
    if rem > 0:
        big = max(quota, key=lambda g: len(easy[g]))
        take[big] += rem
    for g in ("1", "2-3", "4+"):
        pool = sorted(easy[g])
        n = min(take[g], len(pool))
        picked += rng.sample(pool, n)
        if n < take[g]:
            print(f"  WARNING: wanted {take[g]} from easy/{g}, only {len(pool)} exist")
    picked.sort()

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "bin"])
        for iface, b in picked:
            w.writerow([iface, b])
    # and the pool file the evaluator takes
    pool_out = a.out.replace("items.csv", "pool.csv")
    with open(pool_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2"])
        for iface, _ in picked:
            pdb, pair = iface.split("_")
            w.writerow([pdb, pair[0], pair[1]])
    print(f"wrote {a.out}: {len(picked)} item(s)")
    print(f"wrote {pool_out}")
    print("  by bin: " + ", ".join(f"{b}={n}" for b, n in
                                   sorted(collections.Counter(b for _, b in picked).items())))


if __name__ == "__main__":
    main()
