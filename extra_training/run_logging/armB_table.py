#!/usr/bin/env python
"""Arm B epoch 10 vs the shipped head: rho, mean DockQ, frac acceptable.

ONE TABLE FROM TWO IN-SAMPLE PASSES. The 280 slice interfaces were scored by the training run's
own epoch-10 pass and the other 667 by a standalone pass with the SAME checkpoint, flags and
budget, so their rows merge -- disjoint interfaces, one measurement. Merging is checked, not
assumed: an interface appearing in both would mean two different scorings of one item and is
refused.

SHIPPED IS MAX-POOLED AND HAS NO MEAN-POOLED TWIN. Every row carries `stored_iptm`, one number,
because mean pooling needs the per-anchor-row PAE distribution and that is computed only for the
head under test. So the parenthesised mean-pooled column is OURS only, and an ours-mean vs
shipped comparison is mean-against-max. Said in the output rather than left for the reader.
"""
import csv, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_fullset as P

ACC = P.ACC
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
SD = ("/tmp/claude-1004/-home-ebrahim-confidence-recalibration/"
      "d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad")

SIDES = {
    "HELD-OUT val (141 ifaces)": ["training/runs/w57hjo/valfull83_eval_shards_ep10"],
    "IN-SAMPLE (947 ifaces)": ["training/runs/w57hjo/slice280_eval_shards_ep10",
                               f"{SD}/fulltrain15rest_armB_ep10"],
}


def load(dirs):
    by, owner = collections.defaultdict(list), {}
    for d in dirs:
        if not glob.glob(os.path.join(d, "rows_*.csv")):
            return None, f"no rows in {d}"
        if len(glob.glob(os.path.join(d, "rows_*.csv.meta.json"))) != 3:
            return None, f"{os.path.basename(d)} is incomplete (needs 3 shard markers)"
        for f in glob.glob(os.path.join(d, "rows_*.csv")):
            for r in csv.DictReader(open(f)):
                i = r["iface"]
                if i in owner and owner[i] != d:
                    return None, f"{i} scored in two passes ({owner[i]} and {d})"
                owner[i] = d
                m = r.get("iptm_mean")
                by[i].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"]),
                              float(m) if m not in (None, "") else None))
    return by, None


def stats(by, key):
    pick = {i: max(c, key=key)[2] for i, c in by.items()}
    rh = [P._spearman([key(t) for t in c], [t[2] for t in c])
          for c in by.values() if len({key(t) for t in c}) > 1]
    n = len(by)
    return (sum(rh) / len(rh) if rh else float("nan"),
            sum(pick.values()) / n,
            sum(1 for v in pick.values() if v >= ACC) / n)


print("ARM B epoch 10 vs SHIPPED  --  (mean-pooled ipTM in parentheses, OURS only;")
print("                               shipped has no mean-pooled column, it is max-pooled)\n")
for lab, dirs in SIDES.items():
    by, err = load(dirs)
    if err:
        print(f"  {lab}: not ready -- {err}\n")
        continue
    sh = stats(by, lambda t: t[1])
    mx = stats(by, lambda t: t[0])
    mn = stats(by, lambda t: t[3] if t[3] is not None else t[0])
    print(f"  {lab}  n={len(by)}")
    print(f"    {'':20s} {'rho':>18s} {'mean DockQ':>18s} {'frac acceptable':>18s}")
    print(f"    {'shipped':20s} {sh[0]:>18.4f} {sh[1]:>18.4f} {sh[2]*100:>17.1f}%")
    print(f"    {'Arm B ep10':20s} "
          f"{f'{mx[0]:.4f} ({mn[0]:.4f})':>18s} "
          f"{f'{mx[1]:.4f} ({mn[1]:.4f})':>18s} "
          f"{f'{mx[2]*100:.1f}% ({mn[2]*100:.1f}%)':>18s}")
    print()
