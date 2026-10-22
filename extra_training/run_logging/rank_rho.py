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
COLS = ("rho_trained", "rho_shipped", "n_ifaces", "inv_trained", "inv_shipped", "spread")


def _rows(d, ep, budget):
    out = []
    for f in glob.glob(os.path.join(d, f"train_eval_shards_ep{ep}", f"rows_b{budget}_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if not r.get("dockq") or not r.get("stored_iptm"):
                    continue
                try:
                    out.append((r["iface"], float(r["iptm"]), float(r["dockq"]),
                                float(r["stored_iptm"]), int(r["seed"])))
                except (ValueError, KeyError):
                    continue
    return out


def _spear(x, y):
    from scipy.stats import spearmanr
    v = spearmanr(x, y).statistic
    return None if v != v else v


def _one(d, ep, budget):
    """Per-epoch readouts for one run, or None.

    FOUR QUANTITIES, NOT ONE, because they can move independently and the loss only supervises one
    of them:

      rho        mean within-interface Spearman(ipTM, DockQ) over all ~150 poses -- what selection
                 needs, and the only one that was tracked at first.
      inv        WITHIN-SEED pairwise inversion rate, the fraction of same-seed pose pairs the head
                 orders backwards. This is the quantity --rank-loss actually supervises, and it is
                 NOT what the loss reports: the linear loss is a covariance, so it can fall by
                 widening gaps that were already correct while the inverted fraction stands still.
                 Shipped head: 0.448 (a coin flip is 0.5).
      spread     mean within-seed ipTM standard deviation. The linear loss has no stopping point,
                 so this is the thing that can inflate without bound; shared-parameter updates can
                 also move seed means, which is how a within-seed loss reaches cross-seed ordering.
      n_ifaces   how many interfaces contributed.
    """
    rows = _rows(d, ep, budget)
    if not rows:
        return None
    byif = collections.defaultdict(list)
    for ifc, it, dq, sit, sd in rows:
        byif[ifc].append((it, dq, sit, sd))
    tr, sh = [], []
    inv_t = inv_s = tot = 0
    spreads = []
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
        # WITHIN-SEED, which needs the seed back. byif threw it away, so regroup here.
        g = collections.defaultdict(list)
        for _it, _dq, _sit, _sd in v:
            g[_sd].append((_it, _dq, _sit))
        for q in g.values():
            if len(q) > 1:
                _m = sum(x[0] for x in q) / len(q)
                spreads.append((sum((x[0] - _m) ** 2 for x in q) / len(q)) ** 0.5)
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    if q[i][1] == q[j][1]:
                        continue
                    tot += 1
                    if (q[i][0] - q[j][0]) * (q[i][1] - q[j][1]) < 0:
                        inv_t += 1
                    if (q[i][2] - q[j][2]) * (q[i][1] - q[j][1]) < 0:
                        inv_s += 1
    if not tr or not tot:
        return None
    return (len(tr), sum(tr) / len(tr), sum(sh) / len(sh),
            inv_t / tot, inv_s / tot, sum(spreads) / max(1, len(spreads)))


def load(root, runs, budget):
    """epoch -> (rho_trained, rho_shipped, n_ifaces, inv_trained, inv_shipped, spread)."""
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
                        if "inv_trained" not in r:      # pre-expansion cache: recompute
                            raise KeyError("inv_trained")
                        cache[int(r["epoch"])] = tuple(
                            [float(r["rho_trained"]), float(r["rho_shipped"]),
                             int(r["n_ifaces"]), float(r["inv_trained"]),
                             float(r["inv_shipped"]), float(r["spread"])])
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
            cache[ep] = (r[1], r[2], r[0], r[3], r[4], r[5]); dirty = True
        if dirty:
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                fh.write("epoch," + ",".join(COLS) + "\n")
                for ep in sorted(cache):
                    t, sh_, n, it_, is_, sp_ = cache[ep]
                    fh.write(f"{ep},{t:.6f},{sh_:.6f},{n},{it_:.6f},{is_:.6f},{sp_:.6f}\n")
            os.replace(tmp, cp)
        got.update(cache)
    return got
