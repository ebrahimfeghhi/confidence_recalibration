#!/usr/bin/env python
"""Per-epoch RANK CORRELATION between the head's ipTM and the truth it is selected against.

WHY THIS PANEL EXISTS. Selection is an argmax of ipTM over ~150 poses per interface, and DockQ is
whatever that one pose scores. With 18 interfaces, frac_acc moves in steps of 1/18 and mean DockQ
is an average of 18 argmax outcomes -- mdv9x7 swung 0.2072 -> 0.2018 -> 0.2904 across three
ADJACENT epochs with no meaningful change to the model. Panels 3 and 4 therefore cannot resolve a
real change in the ORDERING, which is the only thing selection depends on.

Spearman(ipTM, DockQ) within an interface uses every pose instead of just the winner, so it reads
the ordering directly. Measured once: the SHIPPED head sits at 0.196 -- the statistic we select by
barely tracks pose quality, and that is the whole 0.2909-vs-0.5968 oracle gap. The capacity arms
then split: the pose-share arm climbed 0.222 -> 0.271 over 50 epochs while the full head FELL to
0.078, far below untrained. Panel 3 showed none of that.

THE SHIPPED BASELINE IS DERIVED, NOT PASSED. Every shard row carries `stored_iptm`, the fold's own
ipTM for that pose, beside the recomputed `iptm`. So rho_shipped is computed from the same rows as
rho_trained over the same interfaces -- a hardcoded constant here would be the same mistake the
0.5775 baseline was.

CACHED PER RUN, because the per-epoch plot refreshes every 9 minutes and a full recompute is
250 epochs x 2,700 rows. An epoch is only cached once its metric_train_ep*.json exists: that file
is written after the eval completes, so a mid-write shard set is skipped rather than frozen at a
partial value.
"""
import csv
import collections
import glob
import os
import re

CACHE = "rank_rho_b{b}.csv"


def _rows(d, ep, budget):
    out = []
    for f in glob.glob(os.path.join(d, f"train_eval_shards_ep{ep}", f"rows_b{budget}_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if not r.get("dockq") or not r.get("stored_iptm"):
                    continue
                try:
                    out.append((r["iface"], float(r["iptm"]), float(r["dockq"]),
                                float(r["stored_iptm"])))
                except (ValueError, KeyError):
                    continue
    return out


def _spear(x, y):
    from scipy.stats import spearmanr
    v = spearmanr(x, y).statistic
    return None if v != v else v


def _one(d, ep, budget):
    """(n_ifaces, rho_trained, rho_shipped) for one epoch, or None."""
    rows = _rows(d, ep, budget)
    if not rows:
        return None
    byif = collections.defaultdict(list)
    for ifc, it, dq, sit in rows:
        byif[ifc].append((it, dq, sit))
    tr, sh = [], []
    for ifc, v in byif.items():
        dq = [x[1] for x in v]
        # A CONSTANT TRUTH COLUMN HAS NO ORDERING to recover, so the interface is dropped rather
        # than contributing a nan that would poison the mean.
        if len(set(dq)) < 2:
            continue
        a = _spear([x[0] for x in v], dq)
        b = _spear([x[2] for x in v], dq)
        if a is not None and b is not None:
            tr.append(a); sh.append(b)
    if not tr:
        return None
    return len(tr), sum(tr) / len(tr), sum(sh) / len(sh)


def load(root, runs, budget):
    """epoch -> (rho_trained, rho_shipped, n_ifaces) across a '+'-joined lineage."""
    got = {}
    for part in runs.split("+"):
        d = os.path.join(root, part)
        if not os.path.isdir(d):
            continue
        cp = os.path.join(d, CACHE.format(b=budget))
        cache = {}
        if os.path.exists(cp):
            with open(cp) as fh:
                for r in csv.DictReader(fh):
                    try:
                        cache[int(r["epoch"])] = (float(r["rho_trained"]),
                                                  float(r["rho_shipped"]), int(r["n_ifaces"]))
                    except (ValueError, KeyError):
                        continue
        eps = sorted(int(m.group(1)) for m in
                     (re.search(r"train_eval_shards_ep(\d+)$", p)
                      for p in glob.glob(os.path.join(d, "train_eval_shards_ep*"))) if m)
        dirty = False
        for ep in eps:
            if ep in cache:
                continue
            # SETTLED ONLY. metric_train_ep*.json is written after the eval finishes, so an epoch
            # still being scored is skipped this refresh instead of cached half-done.
            if not os.path.exists(os.path.join(d, f"metric_train_ep{ep}.json")):
                continue
            r = _one(d, ep, budget)
            if r is None:
                continue
            cache[ep] = (r[1], r[2], r[0]); dirty = True
        if dirty:
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                fh.write("epoch,rho_trained,rho_shipped,n_ifaces\n")
                for ep in sorted(cache):
                    t, s, n = cache[ep]
                    fh.write(f"{ep},{t:.6f},{s:.6f},{n}\n")
            os.replace(tmp, cp)
        got.update(cache)
    return got
