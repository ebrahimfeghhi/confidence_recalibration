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
COLS = ("rho_trained", "rho_shipped", "n_ifaces", "inv_trained", "inv_shipped",
        "spread", "tie_trained", "tie_shipped", "rankloss_trained", "rankloss_shipped",
        "top1_trained", "top1_shipped", "dockq_mean", "dockq_sem", "frac_mean", "frac_sem",
        "rho_sem", "top1_sem")


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
                 orders backwards, with a PREDICTED TIE SCORED AS HALF AN ERROR. This is the
                 quantity --rank-loss actually supervises, and it is NOT what the loss reports: the
                 linear loss is a covariance, so it can fall by widening gaps that were already
                 correct while the inverted fraction stands still. 0.0 is perfect, 0.5 is random or
                 no ordering at all, 1.0 is fully reversed. Shipped head: 0.4502.

                 THE HALF-CREDIT IS NOT COSMETIC. Counting only strict inversions (product < 0)
                 scores a predicted tie as CORRECT, so a head that gave all five poses the same
                 ipTM would report an inversion rate of 0.0 -- a perfect score for having learned
                 no ordering whatsoever. That is a live failure mode for a loss whose whole job is
                 to separate scores. Measured tie frequency is small (0.43% shipped, 0.00% in both
                 trained arms, so the correction moves 0.4481 to 0.4502) but it is reported in its
                 own column rather than assumed away. Found in review of 04503a9.
      tie        fraction of informative pairs the head scores exactly equal.
      top1       WITHIN-SEED TOP-1 ACCURACY: for each seed group of 5, does the highest-ipTM pose
                 also hold the highest DockQ? Fraction of that interface's seed groups where it
                 does, then averaged over interfaces. Random choice among 5 is 0.20, so that is
                 the floor, and 1.0 means the head picks the best of every five. This replaced the
                 pairwise inversion rate as the headline because it is the quantity a reader can
                 state in a sentence: "it picks the best of five X% of the time." A DockQ tie at
                 the top counts as correct if the chosen pose is among the tied best.
      dockq/frac PER-INTERFACE selected-pose DockQ and acceptability, reduced to mean and STANDARD
                 ERROR over the 18 interfaces. metric_train_ep*.json already carries the means;
                 these are recomputed from the shards so the SPREAD is available, and the mean is
                 cross-checked against the json rather than trusted.
      rankloss   THE TRAINING LOSS, EVALUATED HELD-OUT. Exactly the objective --rank-loss
                 minimises -- the DockQ-weighted linear pairwise loss over the poses of one seed --
                 but computed on the val10 shards instead of the training poses. The trainer's own
                 per-epoch `rank-loss:` line is a TRAIN number; this is its held-out twin, so the
                 generalisation gap on the new objective is readable directly. 0 means no
                 alignment, negative is better.
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
    import sys as _sys, os as _os
    _ph = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "phases")
    if _ph not in _sys.path:
        _sys.path.insert(0, _ph)
    import rank_loss as _RKL
    tr, sh = [], []
    inv_t = inv_s = tie_t = tie_s = 0.0
    rl_t, rl_s, rl_n = 0.0, 0.0, 0
    top_t, top_s = [], []
    sel_dq, sel_ok = [], []
    tot = 0
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
        # THE SELECTED POSE for this interface: argmax ipTM over ALL its poses, which is the
        # selection rule phase_k_eval implements. Kept per interface so the spread over the 18 is
        # reportable instead of only their mean.
        _best = max(v, key=lambda x: x[0])
        sel_dq.append(_best[1])
        sel_ok.append(1.0 if _best[1] >= 0.23 else 0.0)
        # WITHIN-SEED, which needs the seed back. byif threw it away, so regroup here.
        g = collections.defaultdict(list)
        for _it, _dq, _sit, _sd in v:
            g[_sd].append((_it, _dq, _sit))
        _t_hit = _s_hit = _n_grp = 0
        for q in g.values():
            if len(q) < 2:
                continue
            _n_grp += 1
            _dqmax = max(x[1] for x in q)
            # A DockQ TIE AT THE TOP counts as correct if the chosen pose is among the tied best.
            if max(q, key=lambda x: x[0])[1] >= _dqmax:
                _t_hit += 1
            if max(q, key=lambda x: x[2])[1] >= _dqmax:
                _s_hit += 1
        if _n_grp:
            top_t.append(_t_hit / _n_grp)
            top_s.append(_s_hit / _n_grp)
        for q in g.values():
            # THE HELD-OUT RANK LOSS, same function the trainer differentiates.
            if len(q) > 1:
                _d = [x[1] for x in q]
                if len(set(_d)) > 1:
                    _lt, _, _n1 = _RKL.pair_terms([x[0] for x in q], _d)
                    _ls, _, _n2 = _RKL.pair_terms([x[2] for x in q], _d)
                    if _n1:
                        rl_t += _lt; rl_s += _ls; rl_n += 1
            if len(q) > 1:
                _m = sum(x[0] for x in q) / len(q)
                spreads.append((sum((x[0] - _m) ** 2 for x in q) / len(q)) ** 0.5)
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    if q[i][1] == q[j][1]:
                        continue          # no ordering information in the TRUTH
                    tot += 1
                    _pt = (q[i][0] - q[j][0]) * (q[i][1] - q[j][1])
                    _ps = (q[i][2] - q[j][2]) * (q[i][1] - q[j][1])
                    if _pt < 0:
                        inv_t += 1.0
                    elif _pt == 0:
                        inv_t += 0.5; tie_t += 1.0
                    if _ps < 0:
                        inv_s += 1.0
                    elif _ps == 0:
                        inv_s += 0.5; tie_s += 1.0
    if not tr or not tot:
        return None
    def _ms(xs):
        """mean and STANDARD ERROR of the mean -- the question is whether an epoch-to-epoch move
        is real, and sd/sqrt(n) answers that. The raw sd over 18 interfaces is ~0.25 for DockQ,
        which says only that interfaces differ, which is not in doubt."""
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, (var / len(xs)) ** 0.5
    dq_m, dq_e = _ms(sel_dq)
    fr_m, fr_e = _ms(sel_ok)
    # rho and top-1 are means over interfaces too, so they carry the same kind of error as
    # DockQ and frac_acc. Reporting a band on two panels and not the other two would imply the
    # others are exact.
    _, rho_e = _ms(tr)
    _, top_e = _ms(top_t)
    return (len(tr), sum(tr) / len(tr), sum(sh) / len(sh),
            inv_t / tot, inv_s / tot, sum(spreads) / max(1, len(spreads)),
            tie_t / tot, tie_s / tot,
            rl_t / max(1, rl_n), rl_s / max(1, rl_n),
            sum(top_t) / max(1, len(top_t)), sum(top_s) / max(1, len(top_s)),
            dq_m, dq_e, fr_m, fr_e, rho_e, top_e)


def load(root, runs, budget):
    """epoch -> the tuple named by COLS, across a '+'-joined lineage."""
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
                        if "top1_sem" not in r:           # pre-expansion cache: recompute
                            raise KeyError("top1_sem")
                        cache[int(r["epoch"])] = tuple(
                            [float(r["rho_trained"]), float(r["rho_shipped"]),
                             int(r["n_ifaces"]), float(r["inv_trained"]),
                             float(r["inv_shipped"]), float(r["spread"]),
                             float(r["tie_trained"]), float(r["tie_shipped"]),
                             float(r["rankloss_trained"]), float(r["rankloss_shipped"]),
                             float(r["top1_trained"]), float(r["top1_shipped"]),
                             float(r["dockq_mean"]), float(r["dockq_sem"]),
                             float(r["frac_mean"]), float(r["frac_sem"]),
                             float(r["rho_sem"]), float(r["top1_sem"])])
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
            cache[ep] = (r[1], r[2], r[0], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                         r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17])
            dirty = True
        if dirty:
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                fh.write("epoch," + ",".join(COLS) + "\n")
                for ep in sorted(cache):
                    (t, sh_, n, it_, is_, sp_, tt_, ts_, rt_, rs_,
                     o1_, o0_, dm_, de_, fm_, fe_, re_, oe_) = cache[ep]
                    fh.write(f"{ep},{t:.6f},{sh_:.6f},{n},{it_:.6f},{is_:.6f},{sp_:.6f},"
                             f"{tt_:.6f},{ts_:.6f},{rt_:.8f},{rs_:.8f},"
                             f"{o1_:.6f},{o0_:.6f},{dm_:.6f},{de_:.6f},{fm_:.6f},{fe_:.6f},"
                             f"{re_:.6f},{oe_:.6f}\n")
            os.replace(tmp, cp)
        got.update(cache)
    return got
