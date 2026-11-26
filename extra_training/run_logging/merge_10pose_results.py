#!/usr/bin/env python
"""ONE 10-POSE TABLE in the schema the analysis scripts already read.

The original five poses live in training/fold/results_with_trunks.csv; the five regenerated ones
live in training/fold/poses_extra/{results_extra,dockq_extra}.csv as samples 100-104. This writes
a single csv with the columns gap_matched_within_vs_across.py and friends expect -- iface, seed,
sample, dockq, iptm, iptm_mean, has_trunk, arm, source -- so the 5-pose and 10-pose worlds can be
compared through the SAME code rather than through a second implementation of the same joins.

NOTHING IS OVERWRITTEN. This only ever writes to --out, and the two source files are opened read
only. The 5-pose store stays the store.

--which selects what lands in the output:
  old   the original five only            (reproduces the 5-pose analyses)
  new   the regenerated five only         (an INDEPENDENT replication: same trunks, fresh noise)
  both  all ten, renumbered 0-9           (the depth question)

A unit is emitted only if it carries the full complement for the chosen mode, because a partial
seed silently changes what "within-seed" means -- 4 poses and 10 poses are different experiments.
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["iface", "seed", "sample", "dockq", "iptm", "iptm_mean", "has_trunk", "arm", "source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--extra-iptm", default=f"{ROOT}/training/fold/poses_extra/results_extra.csv")
    ap.add_argument("--extra-dockq", default=f"{ROOT}/training/fold/poses_extra/dockq_extra.csv")
    ap.add_argument("--which", choices=("old", "new", "both"), default="both")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    old = collections.defaultdict(dict)
    with open(a.old) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") != "free":
                continue
            try:
                s = int(r["sample"])
                if not 0 <= s <= 4:
                    continue
                old[(r["iface"], int(r["seed"]))][s] = (
                    float(r["dockq"]), float(r["iptm"]),
                    float(r["iptm_mean"]) if r.get("iptm_mean") not in (None, "") else "")
            except (ValueError, KeyError, TypeError):
                continue

    ip = {}
    with open(a.extra_iptm) as fh:
        for r in csv.DictReader(fh):
            try:
                ip[(r["iface"], int(r["seed"]), int(r["sample"]))] = (
                    float(r["iptm"]),
                    float(r["iptm_mean"]) if r.get("iptm_mean") not in (None, "") else "")
            except (ValueError, KeyError, TypeError):
                continue
    new = collections.defaultdict(dict)
    with open(a.extra_dockq) as fh:
        for r in csv.DictReader(fh):
            try:
                k = (r["iface"], int(r["seed"]), int(r["sample"]))
                if k not in ip:
                    continue
                new[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), ip[k][0], ip[k][1])
            except (ValueError, KeyError, TypeError):
                continue

    want_old = {"old", "both"}
    want_new = {"new", "both"}
    rows, units, part = [], 0, 0
    keys = set(old) | set(new)
    for key in sorted(keys):
        o = old.get(key, {})
        n = new.get(key, {})
        if a.which in want_old and len(o) != 5:
            part += 1
            continue
        if a.which in want_new and len(n) != 5:
            part += 1
            continue
        seq = []
        if a.which in want_old:
            seq += [(s, o[s], "old") for s in sorted(o)]
        if a.which in want_new:
            seq += [(s, n[s], "new") for s in sorted(n)]
        units += 1
        for i, (_s, v, src) in enumerate(seq):
            rows.append([key[0], key[1], i, f"{v[0]:.6f}", f"{v[1]:.6f}",
                         (f"{v[2]:.6f}" if v[2] != "" else ""), 1, "free", src])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"which={a.which}  units {units}  rows {len(rows)}  "
          f"skipped for an incomplete complement {part}  -> {a.out}")


if __name__ == "__main__":
    main()
