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

# BUMPED TO _v2 WHEN xsloss WAS ADDED. A cache written before it holds only the within-seed
# LINEAR rankloss, and silently reading those rows back under the new column layout would mix two
# different objectives in one column. Old caches stay on disk under the old name, distinguishable.
# BUMPED TO _v3 WHEN exloss WAS ADDED. A _v2 cache has 21 columns and no expected-DockQ metric;
# the reader refuses any header that is not exactly ["epoch"] + COLS, so a stale cache is
# recomputed from the shards rather than read back one column short.
# v4: xsloss_* is computed with the run's OWN --rank-pairs / --rank-within-weight /
# --rank-topk-decay instead of hardcoded cross-seed defaults. Pre-knob runs reproduce their v3
# numbers exactly, since the defaults are what they trained; the version bump exists so arms that
# DID set those flags are recomputed rather than read from a cache of the wrong quantity.
# v5 adds xscanon_*, the canonical cross-seed loss every arm can be compared on, alongside the
# own-objective xsloss_* that v4 introduced. v4 made the held-out loss the twin of the training
# curve and in doing so made it incomparable between arms training different functions; both
# columns exist so neither question has to borrow the other's number.
CACHE = "rank_rho_b{b}_v5.csv"


def run_rank_form(d):
    """(form, tau) the run at directory `d` actually TRAINED, from its own args.json.

    THE HELD-OUT METRIC HAS TO MATCH THE OBJECTIVE. This module hardcoded form="softplus" at
    tau=DEFAULT_TAU, which was right while every arm was pairwise and silently wrong the moment
    --rank-form expected existed: the arm would train -expected DockQ at tau 0.05 and be graded
    on a pairwise softplus at tau 0.0305, two quantities of opposite sign. Caught in review of
    92338a36 before any expected-form arm had run.

    Falls back to the pairwise default when args.json is missing or carries no ranking fields,
    which is what every pre-92338a36 run looks like.

    SEE ALSO run_rank_objective, which returns the FULL objective. This function kept returning
    only (form, tau) while --rank-pairs, --rank-within-weight and --rank-topk-decay were added,
    so an arm training pairs=all at L=8 was again being graded on a cross-seed loss it was not
    minimising -- the exact failure this docstring was written about, repeated one knob later.
    """
    import json
    p = os.path.join(d, "args.json")
    try:
        with open(p) as fh:
            a = json.load(fh)
    except (OSError, ValueError):
        return "softplus", None
    form = a.get("rank_form") or "softplus"
    tau = a.get("rank_tau")
    if tau is None or float(tau) <= 0:
        tau = None
    return form, (float(tau) if tau is not None else None)
def run_rank_objective(d):
    """Every setting that defines WHAT the run minimises, as kwargs for rank_loss.group_loss.

    THE HELD-OUT LOSS IS COMPUTED WITH THESE, not with hardcoded cross-seed defaults, so
    xsloss_trained is always the twin of the training curve and the gap between them is
    generalisation rather than a change of function.

    BACKWARD COMPATIBLE BY CONSTRUCTION: the defaults here are exactly what every pre-knob run
    trained (cross-seed pairs, within-seed weight 1, no rank decay), so those runs' numbers are
    bit-identical to what the hardcoded version produced. Only arms that actually set the new
    flags move, and those were being measured against the wrong function.
    """
    import json
    import rank_loss as _RKL   # imported here, as elsewhere in this module
    form, tau = run_rank_form(d)
    try:
        with open(os.path.join(d, "args.json")) as fh:
            a = json.load(fh)
    except (OSError, ValueError):
        a = {}
    return dict(form=form,
                tau=tau if tau is not None else
                    (_RKL.DEFAULT_SOFTMAX_TAU if form == "expected" else _RKL.DEFAULT_TAU),
                pairs=a.get("rank_pairs") or "cross",
                within_weight=float(a.get("rank_within_weight") or 1.0),
                rank_decay_k=float(a.get("rank_topk_decay") or 0.0),
                weight_by_gap=bool(a.get("rank_weight_by_gap", True)))


COLS = ("rho_trained", "rho_shipped", "n_ifaces", "inv_trained", "inv_shipped",
        "spread", "tie_trained", "tie_shipped", "rankloss_trained", "rankloss_shipped",
        "top1_trained", "top1_shipped", "dockq_mean", "dockq_sem", "frac_mean", "frac_sem",
        "rho_sem", "top1_sem",
        # THE OBJECTIVE 4y3ve6 ACTUALLY TRAINS. Distinct from rankloss_* above, which is the OLD
        # within-seed LINEAR loss and is kept only so historical arms stay comparable. See the
        # module docstring.
        "xsloss_trained", "xsloss_shipped", "xsloss_sem",
        # THE SAME LOSS AT THE CANONICAL SETTINGS -- cross-seed softplus at DEFAULT_TAU, no
        # within-seed weight, no rank decay -- FOR EVERY ARM REGARDLESS OF WHAT IT TRAINED.
        # xsloss_* above is each run's OWN objective, which is what makes it the twin of the
        # training curve; that same property makes it INCOMPARABLE ACROSS ARMS that train
        # different functions. Both are needed and they answer different questions:
        #   xsloss_*       vs the training curve  -> generalisation
        #   xscanon_*      vs another arm         -> which arm orders held-out poses better
        # For a cross-seed arm at DEFAULT_TAU the two are identical by construction.
        "xscanon_trained", "xscanon_shipped",
        # THE EXPECTED-DockQ OBJECTIVE, held out, at THIS RUN's OWN softmax temperature.
        # NEGATIVE: it is -E[DockQ of the selected pose], floor -max(DockQ) per group, so it is
        # NOT on the same scale as xsloss_* above and the two must never share an axis. Computed
        # for every run, including pairwise ones, so any two arms can be compared on it -- but
        # only at EQUAL tau: the metric itself depends on tau, so exloss from a tau 0.02 arm and
        # a tau 0.10 arm are different quantities. Cross-arm selection comparisons belong to
        # dockq_mean and frac_mean, which depend on no hyperparameter at all.
        "exloss_trained", "exloss_shipped", "exloss_sem", "exloss_tau")


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
      rankloss   THE OLD WITHIN-SEED LINEAR LOSS, held out. The DockQ-weighted linear pairwise
                 loss over the poses of ONE seed. It is what arms up to and including ntwjfp
                 trained, and it is KEPT ONLY so those arms stay comparable. It is NOT the
                 objective of any cross-seed softplus arm: do not read it as that arm's
                 generalisation. It is a covariance, so it can be negative; 0 means no alignment.
      xsloss     THE CROSS-SEED SOFTPLUS, held out -- the objective --rank-form softplus
                 --rank-cross-seed actually minimises, evaluated on val10 with the SAME tau
                 (rank_loss.DEFAULT_TAU), the same per-interface normalisation and the same
                 cross-seed-only pair selection as training. NON-NEGATIVE, floor 0 = every
                 cross-seed pair correctly ordered, so LOWER IS BETTER and it is not on the same
                 scale as rankloss -- the two must never share an axis. Computed with
                 rank_loss.group_loss, the float-only path, which test_rank_loss.py asserts agrees
                 with the autograd group_coeffs the trainer differentiates to within 1e-9.
                 Reference: shipped 0.0803; 8eiek8, which trains no ranking term at all, reaches
                 0.052-0.055. So an arm must beat ~0.05 to have done anything the PAE objective
                 did not already do.
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
    # PER-INTERFACE held-out loss under the RUN'S OWN objective, trained and shipped.
    xs_t, xs_s = [], []
    xc_t, xc_s = [], []
    ex_t, ex_s = [], []
    # THE TEMPERATURE THE HELD-OUT EXPECTED METRIC USES. This run's own --rank-tau when it
    # trained the expected form; otherwise DEFAULT_SOFTMAX_TAU, so a pairwise arm still gets a
    # comparable number at the canonical 0.05.
    _rf, _rt = run_rank_form(d)
    _ex_tau = _rt if (_rf == "expected" and _rt is not None) else _RKL.DEFAULT_SOFTMAX_TAU
    # THE RUN'S OWN OBJECTIVE for the held-out pairwise loss. The expected form is handled
    # separately below and is pair-free, so it keeps its own path.
    _obj = run_rank_objective(d)
    if _obj["form"] == "expected":
        _obj = dict(form="softplus", tau=_RKL.DEFAULT_TAU, pairs="cross",
                    within_weight=1.0, rank_decay_k=0.0, weight_by_gap=True)
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
        # THE RUN'S OWN OBJECTIVE, held out -- same form, same tau, same pair family, same
        # within-seed weight and same rank decay as training, so this IS the training curve's
        # twin and the gap between them is generalisation. Previously hardcoded to cross-seed
        # softplus at DEFAULT_TAU, which was right for every arm until --rank-pairs existed.
        # Keyed (seed, index-within-seed) because group_coeffs reads the seed field to exclude
        # within-seed pairs; the second element only has to be unique inside a seed.
        _sb, _ss, _db, _n_in_sd = {}, {}, {}, collections.defaultdict(int)
        for _it, _dq, _sit, _sd in v:
            _k = (_sd, _n_in_sd[_sd]); _n_in_sd[_sd] += 1
            _sb[_k] = _it; _ss[_k] = _sit; _db[_k] = _dq
        # group_loss, NOT group_coeffs: the value is all this needs, and the autograd path made a
        # 50-epoch recompute take minutes for a gradient immediately discarded.
        # test_rank_loss.py asserts the two agree to 1e-9 so the diagnostic cannot drift from the
        # objective.
        _lv_t, _np_t = _RKL.group_loss(_sb, _db, **_obj)
        _lv_s, _np_s = _RKL.group_loss(_ss, _db, **_obj)
        # The canonical yardstick, identical settings for every arm.
        _cv_t, _cn_t = _RKL.group_loss(_sb, _db, form="softplus", tau=_RKL.DEFAULT_TAU,
                                       pairs="cross")
        _cv_s, _cn_s = _RKL.group_loss(_ss, _db, form="softplus", tau=_RKL.DEFAULT_TAU,
                                       pairs="cross")
        if _cn_t and _cn_s:
            xc_t.append(_cv_t); xc_s.append(_cv_s)
        if _np_t and _np_s:
            xs_t.append(_lv_t); xs_s.append(_lv_s)
        # THE EXPECTED-DockQ OBJECTIVE, held out, at this run's own temperature. Pair-free, so no
        # pairs="cross" and no gap weighting -- the softmax runs over every pose of the
        # interface, which is the same population argmax selects from.
        _ev_t, _en_t = _RKL.group_loss(_sb, _db, form="expected", tau=_ex_tau)
        _ev_s, _en_s = _RKL.group_loss(_ss, _db, form="expected", tau=_ex_tau)
        if _en_t and _en_s:
            ex_t.append(_ev_t); ex_s.append(_ev_s)
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
    # MEAN OVER INTERFACES, matching how training normalises per interface-group and then means
    # over groups. The softplus loss is NON-NEGATIVE with 0 as its floor (every pair ordered),
    # unlike rankloss_* which is a covariance and can be negative -- so the two columns are not
    # on the same scale and must never be plotted on one axis.
    xs_m, xs_e = _ms(xs_t)
    xs_sm, _ = _ms(xs_s)
    ex_m, ex_e = _ms(ex_t)
    ex_sm, _ = _ms(ex_s)
    # IN COLS ORDER, DELIBERATELY. This used to return its own order and `load` permuted it with
    # positional literals -- (r[1], r[2], r[0], r[3], ...) -- truncated at r[17], with a writer
    # that unpacked exactly eighteen names. Adding xsloss_* then produced a cache whose header had
    # 22 fields and whose rows had 19, which parsed without error and surfaced later as an
    # IndexError at the plot. Returning in COLS order removes the permutation, and the writer now
    # iterates COLS, so a new column is added in ONE place.
    assert len(COLS) == 27, f"COLS changed to {len(COLS)}; update this return"
    return (sum(tr) / len(tr), sum(sh) / len(sh), len(tr),
            inv_t / tot, inv_s / tot, sum(spreads) / max(1, len(spreads)),
            tie_t / tot, tie_s / tot,
            rl_t / max(1, rl_n), rl_s / max(1, rl_n),
            sum(top_t) / max(1, len(top_t)), sum(top_s) / max(1, len(top_s)),
            dq_m, dq_e, fr_m, fr_e, rho_e, top_e,
            xs_m, xs_sm, xs_e,
            (sum(xc_t) / len(xc_t)) if xc_t else float("nan"),
            (sum(xc_s) / len(xc_s)) if xc_s else float("nan"),
            ex_m, ex_sm, ex_e, _ex_tau)


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
            # VALIDATED AGAINST COLS, AND BUILT FROM IT. The previous reader named all eighteen
            # columns by hand with an ad-hoc `if "top1_sem" not in r` guard for one earlier
            # expansion. When xsloss_* was added, a cache whose HEADER had the new 22 fields but
            # whose ROWS still had 19 (written mid-edit) parsed without error -- DictReader fills
            # the missing keys with None, every hand-named key happened to fall inside the first
            # 19, and the result was a silently SHORT tuple that only failed later as an
            # IndexError at the plot. So: refuse any cache whose header is not exactly COLS, and
            # build the tuple by iterating COLS so a new column cannot be forgotten here.
            with open(cp) as fh:
                rdr = csv.reader(fh)
                try:
                    head = next(rdr)
                except StopIteration:
                    head = []
                if head == ["epoch"] + list(COLS):
                    for row in rdr:
                        if len(row) != len(head):
                            continue
                        r = dict(zip(head, row))
                        try:
                            cache[int(r["epoch"])] = tuple(
                                int(r[c]) if c == "n_ifaces" else float(r[c]) for c in COLS)
                        except (ValueError, KeyError, TypeError):
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
            if len(r) != len(COLS):
                raise SystemExit(f"STOP: _one returned {len(r)} value(s) for {len(COLS)} "
                                 f"column(s); the cache would be written short.")
            cache[ep] = r
            dirty = True
        if dirty:
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                fh.write("epoch," + ",".join(COLS) + "\n")
                for ep in sorted(cache):
                    vals = []
                    for c, v in zip(COLS, cache[ep]):
                        if c == "n_ifaces":
                            vals.append(str(int(v)))
                        elif c.startswith(("rankloss", "xsloss")):
                            vals.append(f"{float(v):.8f}")      # the small ones need the digits
                        else:
                            vals.append(f"{float(v):.6f}")
                    fh.write(f"{ep}," + ",".join(vals) + "\n")
            os.replace(tmp, cp)
        got.update(cache)
    return got
