#!/usr/bin/env python
"""One CSV for every per-pose result we hold, across every dataset, in one schema.

WHY. The results are spread over five stores in two different schemas under two different
roots, and nothing joins them. Any question of the form "how many poses do we have for X, and
do we have the pair rep for them" currently needs a bespoke script. This writes the index.

ONE ROW PER POSE. Columns:
    dataset     pxm2025 | foldbench | foldbench_extra | sabdab2 | trainfold | val83
    structure   the PDB id
    interface   <pdb>_<chain1><chain2>, as the source stores it
    seed        the fold seed; this is what selects the trunk
    pose        the diffusion sample index within the seed, 0..4. ARBITRARY BUT STABLE --
                it is the source store's own `sample` column, which is just the order the
                sampler emitted them in. Pose 0 of seed 500000 means nothing in common with
                pose 0 of seed 500001.
    dockq       DockQ of that pose at that interface
    iptm        chain-pair ipTM, max-pooled, for that interface and pose
    has_trunk   1 if a cached pair rep exists for (structure, seed), else 0
    source      the file the row came from, so any number here is traceable

has_trunk IS KEYED ON (structure, seed), NOT ON THE POSE OR THE INTERFACE. One trunk per
structure per seed, shared by all of that structure's interfaces and all of its poses -- which
is the whole reason within-seed pose ranking is hard. The index is built by scanning every
trunk root for `trunks/<pdb>/free_<seed>.npz`, so it reports what is actually on disk rather
than trusting a `has_trunk` column that a source store may have computed at a different time.

TWO SOURCE SCHEMAS, both read, neither mutated:
    pdb,chain1,chain2,iface,seed,sample,dockq,pair_iptm,metadata   (pxm2025, foldbench)
    iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm[,has_trunk]   (everything else)
In the second schema only `arm == free` rows are kept: the other arms are pose-injection
experiments, not independent folds, so mixing them in would double-count a seed.

DUPLICATES. A (dataset, interface, seed, pose) key should appear once. If a store repeats one,
the first occurrence wins and the collision is counted and reported -- silently dropping them
would hide a store that was appended to twice.
"""
import argparse
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # extra_training/
REPO = os.path.dirname(HERE)                                          # repo root

# (dataset label, path, schema) -- schema "baseline" or "arm"
SOURCES = [
    ("pxm2025",         f"{REPO}/experimental_data/protenix/pxm2025/pxm2025_baseline_results.csv",        "baseline"),
    ("foldbench",       f"{REPO}/experimental_data/protenix/foldbench/foldbench_baseline_results.csv",    "baseline"),
    ("foldbench_extra", f"{REPO}/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds/results.csv", "arm"),
    ("sabdab2",         f"{HERE}/validation/fold_sab98/results.csv",      "arm"),
    ("trainfold",       f"{HERE}/training/fold/results_with_trunks.csv",  "arm"),
    ("val83",           f"{HERE}/validation/fold/results.csv",            "arm"),
]

TRUNK_ROOTS = [
    f"{HERE}/training/fold/trunks",
    f"{HERE}/training/fold_regen/trunks",
    f"{HERE}/training/fold_deferred/trunks",
    f"{HERE}/validation/fold/trunks",
    f"{HERE}/validation/fold_sab98/trunks",
    f"{HERE}/validation/fold_trunkfix/trunks",
    f"{HERE}/validation/fold_deferred/trunks",
]

COLS = ["dataset", "structure", "interface", "seed", "pose",
        "dockq", "iptm", "has_trunk", "source"]


def trunk_index(roots):
    """{(pdb, seed)} for every cached pair rep on disk, over every root."""
    have = set()
    per_root = {}
    for root in roots:
        n = 0
        if not os.path.isdir(root):
            per_root[root] = None
            continue
        for pdb in os.listdir(root):
            d = os.path.join(root, pdb)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.endswith(".npz"):
                    continue
                stem = fn[:-4]
                # free_500000.npz -> seed 500000. Anything else is not a free-arm trunk.
                if not stem.startswith("free_"):
                    continue
                try:
                    have.add((pdb, int(stem[5:])))
                except ValueError:
                    continue
                n += 1
        per_root[root] = n
    return have, per_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/training/fold/all_results.csv")
    ap.add_argument("--datasets", default="",
                    help="comma-separated subset of dataset labels; default all")
    args = ap.parse_args()

    want = {s.strip() for s in args.datasets.split(",") if s.strip()}

    have, per_root = trunk_index(TRUNK_ROOTS)
    print(f"cached trunks on disk: {len(have)} (structure, seed) pair(s)")
    for root, n in per_root.items():
        tag = "MISSING" if n is None else f"{n} npz"
        print(f"    {os.path.relpath(root, REPO)}: {tag}")

    rows = []
    seen = set()
    report = []
    for label, path, schema in SOURCES:
        if want and label not in want:
            continue
        if not os.path.exists(path):
            report.append((label, "FILE MISSING", 0, 0, 0, 0))
            continue
        kept = dupes = skipped_arm = bad = 0
        with open(path) as fh:
            rd = csv.DictReader(fh)
            fields = rd.fieldnames or []
            ipcol = "pair_iptm" if "pair_iptm" in fields else "iptm"
            for r in rd:
                if schema == "arm" and "arm" in fields:
                    if (r.get("arm") or "").strip() not in ("", "free"):
                        skipped_arm += 1
                        continue
                iface = (r.get("iface") or "").strip()
                if not iface:
                    bad += 1
                    continue
                pdb = r.get("pdb") or iface.split("_")[0]
                try:
                    seed = int(r["seed"])
                    pose = int(r["sample"])
                    dockq = float(r["dockq"])
                    iptm = float(r[ipcol])
                except (KeyError, TypeError, ValueError):
                    bad += 1
                    continue
                key = (label, iface, seed, pose)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                rows.append({
                    "dataset": label,
                    "structure": pdb,
                    "interface": iface,
                    "seed": seed,
                    "pose": pose,
                    "dockq": f"{dockq:.6f}",
                    "iptm": f"{iptm:.6f}",
                    "has_trunk": 1 if (pdb, seed) in have else 0,
                    "source": os.path.relpath(path, REPO),
                })
                kept += 1
        report.append((label, os.path.relpath(path, REPO), kept, dupes, skipped_arm, bad))

    rows.sort(key=lambda d: (d["dataset"], d["interface"], d["seed"], d["pose"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.out)

    print(f"\n{'dataset':<18} {'rows':>8} {'dup':>6} {'non-free':>9} {'bad':>5}   source")
    for label, path, kept, dupes, skipped_arm, bad in report:
        print(f"{label:<18} {kept:>8} {dupes:>6} {skipped_arm:>9} {bad:>5}   {path}")

    print(f"\n{'dataset':<18} {'poses':>8} {'ifaces':>7} {'structs':>8} {'seeds':>6} "
          f"{'w/ trunk':>9} {'%':>6}")
    agg = collections.defaultdict(lambda: {"n": 0, "if": set(), "st": set(),
                                           "sd": set(), "tr": 0})
    for d in rows:
        a = agg[d["dataset"]]
        a["n"] += 1
        a["if"].add(d["interface"])
        a["st"].add(d["structure"])
        a["sd"].add((d["structure"], d["seed"]))
        a["tr"] += int(d["has_trunk"])
    for label in [s[0] for s in SOURCES]:
        if label not in agg:
            continue
        a = agg[label]
        pct = 100.0 * a["tr"] / a["n"] if a["n"] else 0.0
        print(f"{label:<18} {a['n']:>8} {len(a['if']):>7} {len(a['st']):>8} "
              f"{len(a['sd']):>6} {a['tr']:>9} {pct:>5.1f}%")
    tot = sum(a["n"] for a in agg.values())
    tot_tr = sum(a["tr"] for a in agg.values())
    print(f"{'TOTAL':<18} {tot:>8} {'':>7} {'':>8} {'':>6} {tot_tr:>9} "
          f"{100.0*tot_tr/tot if tot else 0:>5.1f}%")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
