#!/usr/bin/env python
"""One arm's val and in-sample numbers as a TABLE: mean DockQ, frac acceptable, rho, PAE loss.

THE FOURTH METRIC IS THE PAE LOSS, split by pose quality, read from the sidecar that
phase_k_eval --pae-loss writes beside each shard's rows. It is the SAME quantity the
training curve plots and the same one the standalone --loss-only pass measured (verified
bit-identical on 45 poses, max |diff| 0.00000000), computed in this pass's own forward at
no extra cost. A pose is GOOD when every interface of its structure clears DockQ 0.23 and
BAD when none does; the ~5% that straddle are excluded rather than assigned, exactly as in
training. Absent sidecars simply omit the columns, so an older pass still tabulates.

TABLES, NOT PLOTS, for the metrics. A four-panel figure per epoch was how these were reported;
the numbers are four rows and reading them off an axis loses precision the decimals carry. The
only figure worth keeping is the PAE training loss, where the SHAPE is the finding -- the
good-pose turn-up at epoch 12 -- and a table of per-epoch losses would hide it.

SHIPPED IS MAX-POOLED AND HAS NO MEAN-POOLED TWIN. Every row carries `stored_iptm`, one number,
because mean pooling needs the per-anchor-row PAE distribution and that is computed only for the
head under test. So the parenthesised mean-pooled column is OURS only, and ours-mean against
shipped is mean-against-max. Said in the output rather than left for the reader.
"""
import argparse, csv, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_index as LI
import numpy as np
import plot_fullset as P

ACC = P.ACC
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)


def load(dirs):
    by, owner = collections.defaultdict(list), {}
    for d in dirs:
        if not glob.glob(os.path.join(d, "rows_*.csv")):
            return None, f"no rows in {os.path.basename(d)}"
        if len(glob.glob(os.path.join(d, "rows_*.csv.meta.json"))) != 3:
            return None, f"{os.path.basename(d)} incomplete (needs 3 shard markers)"
        for f in glob.glob(os.path.join(d, "rows_*.csv")):
            for r in csv.DictReader(open(f)):
                i = r["iface"]
                # AN INTERFACE IN TWO PASSES IS TWO SCORINGS OF ONE ITEM, not more data.
                if i in owner and owner[i] != d:
                    return None, f"{i} scored in two passes"
                owner[i] = d
                m = r.get("iptm_mean")
                by[i].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"]),
                              float(m) if m not in (None, "") else None))
    return by, None


def pae_split(dirs, results="validation/fold/results.csv", accept=ACC):
    """(good, bad, all, n_good, n_bad) mean PAE loss from the sidecars, or None if absent.

    THE ROW -> SAMPLE MAP GOES THROUGH label_index. The sidecar already stores a SAMPLE id (the
    evaluator attributes it through its coordinate-identity join), so no mapping is needed here --
    which is the point of writing a sample rather than a row.
    """
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "paeloss_*.csv"))
    if not files:
        return None
    dq = collections.defaultdict(list)
    if os.path.exists(results):
        for r in csv.DictReader(open(results)):
            try:
                dq[(r["iface"].split("_")[0], int(r["seed"]), int(r["sample"]))].append(
                    float(r["dockq"]))
            except (ValueError, KeyError):
                continue
    good, bad, allv = [], [], []
    for f in files:
        for r in csv.DictReader(open(f)):
            try:
                v = float(r["pae_loss"])
                k = (r["pdb"], int(r["seed"]), int(r["sample"]))
            except (ValueError, KeyError):
                continue
            allv.append(v)
            d = dq.get(k)
            if not d:
                continue
            if all(x >= accept for x in d):
                good.append(v)
            elif all(x < accept for x in d):
                bad.append(v)
    if not allv:
        return None
    return (sum(good) / len(good) if good else float("nan"),
            sum(bad) / len(bad) if bad else float("nan"),
            sum(allv) / len(allv), len(good), len(bad))


def stats(by, key):
    pick = {i: max(c, key=key)[2] for i, c in by.items()}
    rh = [P._spearman([key(t) for t in c], [t[2] for t in c])
          for c in by.values() if len({key(t) for t in c}) > 1]
    n = len(by)
    return (sum(rh) / len(rh) if rh else float("nan"),
            sum(pick.values()) / n,
            sum(1 for v in pick.values() if v >= ACC) / n)


def table(label, dirs, md=False,
          results="validation/fold/results.csv"):
    by, err = load(dirs)
    if err:
        print(f"  {label}: not ready -- {err}\n")
        return
    pl = pae_split(dirs, results)
    sh = stats(by, lambda t: t[1])
    mx = stats(by, lambda t: t[0])
    mn = stats(by, lambda t: t[3] if t[3] is not None else t[0])
    if md:
        print(f"**{label}** (n={len(by)})\n")
        print("| | mean DockQ | frac acceptable | rho |")
        print("|---|---|---|---|")
        print(f"| shipped | {sh[1]:.4f} | {sh[2]*100:.1f}% | {sh[0]:.4f} |")
        print(f"| ours | {mx[1]:.4f} ({mn[1]:.4f}) | {mx[2]*100:.1f}% ({mn[2]*100:.1f}%) | "
              f"{mx[0]:.4f} ({mn[0]:.4f}) |")
        if pl:
            print(f"\n| PAE loss | good ({pl[3]}) | bad ({pl[4]}) | all |")
            print("|---|---|---|---|")
            print(f"| ours | {pl[0]:.4f} | {pl[1]:.4f} | {pl[2]:.4f} |")
        print()
    else:
        print(f"  {label}  n={len(by)}")
        print(f"    {'':12s} {'mean DockQ':>18s} {'frac acceptable':>18s} {'rho':>18s}")
        print(f"    {'shipped':12s} {sh[1]:>18.4f} {sh[2]*100:>17.1f}% {sh[0]:>18.4f}")
        print(f"    {'ours':12s} {f'{mx[1]:.4f} ({mn[1]:.4f})':>18s} "
              f"{f'{mx[2]*100:.1f}% ({mn[2]*100:.1f}%)':>18s} "
              f"{f'{mx[0]:.4f} ({mn[0]:.4f})':>18s}")
        if pl:
            print(f"    {'PAE loss':12s} good {pl[0]:.4f} ({pl[3]})   "
                  f"bad {pl[1]:.4f} ({pl[4]})   all {pl[2]:.4f}")
        print()


ap = argparse.ArgumentParser()
ap.add_argument("--rid", required=True, help="run id, or a rid FILE under training/runs")
ap.add_argument("--epochs", default="10,20")
ap.add_argument("--extra-insample", default="",
                help="comma-separated extra shard dirs to merge into the in-sample side")
ap.add_argument("--md", action="store_true", help="markdown tables")
a = ap.parse_args()

rid = a.rid
if rid.startswith("training/runs/"):
    rid = rid[len("training/runs/"):]
p = os.path.join("training/runs", rid)
if os.path.isfile(p):
    rid = open(p).read().strip()

print(f"{rid}  --  mean-pooled ipTM in parentheses (OURS only; shipped is max-pooled)\n")
for e in [x.strip() for x in a.epochs.split(",") if x.strip()]:
    v = f"training/runs/{rid}/valfull83_eval_shards_ep{e}"
    s = [f"training/runs/{rid}/slice280_eval_shards_ep{e}"]
    s += [x for x in a.extra_insample.split(",") if x]
    if os.path.isdir(v):
        table(f"HELD-OUT val, epoch {e}", [v], a.md,
              results="validation/fold/results.csv")
    if all(os.path.isdir(x) for x in s):
        table(f"IN-SAMPLE, epoch {e}", s, a.md,
              results="training/fold/results_with_trunks.csv")
