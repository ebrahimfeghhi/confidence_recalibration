#!/usr/bin/env python
"""THE SAME GAP-MATCHED COMPARISON, but pairs are formed WITHIN A STRUCTURE, not within an
interface. run_logging/gap_matched_within_vs_across.py pools only poses of one interface, so it
cannot say whether the between-seed advantage at large DockQ gaps survives when the comparison is
the one a structure-level selector actually faces.

FOUR POPULATIONS, because two things vary at once inside a structure and they are different
questions:
  same seed, same iface   the original "within" -- the 10 sample pairs of one seed
  diff seed, same iface   the original "across"
  same seed, diff iface   two interfaces of the SAME complex scored off the SAME trunk
  diff seed, diff iface   both differ
A cross-interface pair asks whether ipTM is comparable BETWEEN interfaces, which is a calibration
question, not a pose-ranking one: the two numbers come from different chain pairs with different
token counts and therefore different d0. Reported because a structure-level selector has to make
exactly that comparison, and because if it is bad it bounds anything built on per-structure
argmax.

WHY THE SEED AXIS IS THE SAME TRUNK ONLY WITHIN A SEED. These are shipped-weights inference
folds, so there is NO CROP anywhere in them -- cropping is a training-time thing and does not
enter here. What a seed does fix is the trunk: one trunk per structure per seed, shared by all of
that structure's interfaces and all five poses. So "same seed, diff iface" holds the trunk fixed
and varies the chain pair, while "diff seed, same iface" does the reverse. That is why the four
cells are worth separating rather than pooling into "within" and "across".

Everything else matches the interface-level script: DockQ ties dropped, an ipTM tie counts as half
a hit, bins on |dDockQ|, and the median |dipTM| beside each accuracy so a movement-with-no-
ordering result is visible rather than inferred.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
CELLS = [("same seed, same iface", True, True), ("diff seed, same iface", False, True),
         ("same seed, diff iface", True, False), ("diff seed, diff iface", False, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=15,
                    help="keep only structures whose every interface carries this many seeds")
    ap.add_argument("--iptm-col", default="iptm")
    a = ap.parse_args()

    import numpy as np
    import rank_loss as RKL
    roles_join = RKL.load_roles(a.roles)

    cell = collections.defaultdict(dict)
    with open(a.results) as fh:
        rd = csv.DictReader(fh)
        col = a.iptm_col if a.iptm_col in (rd.fieldnames or []) else "pair_iptm"
        for r in rd:
            if r.get("has_trunk") not in (None, "", "1"):
                continue
            if "arm" in r and r.get("arm") not in (None, "", "free"):
                continue
            try:
                cell[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r[col]))
            except (ValueError, KeyError, TypeError):
                continue

    # structure -> list of (iface_id, seed, dockq, iptm) with complete 5-sample seeds only
    by_struct = collections.defaultdict(list)
    iface_ids = {}
    for (iface, seed), d in cell.items():
        if len(d) != int(a.poses):
            continue
        p = RKL.parse_iface(iface, roles_join)
        if p is None:
            continue
        pdb = p[0]
        iid = iface_ids.setdefault(iface, len(iface_ids))
        for k in sorted(d):
            by_struct[pdb].append((iid, seed, d[k][0], d[k][1]))

    # KEEP ONLY STRUCTURES WHOSE EVERY INTERFACE HAS THE FULL SEED SET, so a cross-interface pair
    # is never an artefact of one interface being folded at more seeds than another.
    keep = {}
    for pdb, rows in by_struct.items():
        per = collections.Counter(iid for iid, _s, _d, _i in rows)
        want = int(a.seeds) * int(a.poses)
        if per and all(v == want for v in per.values()):
            keep[pdb] = rows
    n_if = len({iid for rows in keep.values() for iid, _s, _d, _i in rows})
    multi = sum(1 for rows in keep.values()
                if len({iid for iid, _s, _d, _i in rows}) > 1)
    print(f"structures {len(keep)} ({multi} with >1 interface)   interfaces {n_if}   "
          f"poses {sum(len(v) for v in keep.values())}")

    acc = {c[0]: collections.defaultdict(lambda: [0.0, 0, []]) for c in CELLS}
    for pdb, rows in keep.items():
        iid = np.array([r[0] for r in rows])
        sd = np.array([r[1] for r in rows])
        dq = np.array([r[2] for r in rows], dtype=float)
        ip = np.array([r[3] for r in rows], dtype=float)
        ii, jj = np.triu_indices(len(rows), k=1)
        gap = np.abs(dq[ii] - dq[jj])
        live = gap > 0
        ii, jj, gap = ii[live], jj[live], gap[live]
        same_s = sd[ii] == sd[jj]
        same_i = iid[ii] == iid[jj]
        corr = np.where(ip[ii] == ip[jj], 0.5,
                        ((ip[ii] > ip[jj]) == (dq[ii] > dq[jj])).astype(float))
        dip = np.abs(ip[ii] - ip[jj])
        b = np.digitize(gap, EDGES) - 1
        for name, ws, wi in CELLS:
            m = (same_s == ws) & (same_i == wi)
            if not m.any():
                continue
            for k in range(len(EDGES) - 1):
                mm = m & (b == k)
                if not mm.any():
                    continue
                t = acc[name][k]
                t[0] += float(corr[mm].sum())
                t[1] += int(mm.sum())
                t[2].append(dip[mm])

    for name, _ws, _wi in CELLS:
        print()
        print(f"=== {name}")
        print(f"{'|dDockQ| bin':>16s} {'acc':>8s} {'n':>10s} {'med|dipTM|':>11s}")
        tot_h = tot_n = 0
        allp = []
        for k, (lo_, hi_) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            t = acc[name].get(k)
            if not t or not t[1]:
                continue
            md = float(np.median(np.concatenate(t[2])))
            print(f"{f'[{lo_:.2f},{hi_:.2f})':>16s} {t[0] / t[1]:8.4f} {t[1]:10d} {md:11.4f}")
            tot_h += t[0]
            tot_n += t[1]
            allp.append(np.concatenate(t[2]))
        if tot_n:
            print(f"{'ALL':>16s} {tot_h / tot_n:8.4f} {tot_n:10d} "
                  f"{float(np.median(np.concatenate(allp))):11.4f}")


if __name__ == "__main__":
    main()
