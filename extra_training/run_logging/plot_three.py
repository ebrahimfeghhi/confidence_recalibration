#!/usr/bin/env python
"""Three held-out panels: mean DockQ, frac acceptable, and Spearman(ipTM, DockQ) -- each with s.e.

Usage: python -I run_logging/plot_three.py <rid> [--out path]

ALL THREE ARE HELD OUT, on val10's 18 interfaces at budget 30, and all three carry the standard
error ACROSS INTERFACES rather than across poses. That matters: the paired s.e. on mean DockQ over
these 18 interfaces is 0.026, so the set resolves ~0.05 DockQ and nothing smaller. A band that
overlaps the dashed line is not a result.

The dashed line is the SHIPPED head, derived from this run's own stored_iptm column -- a baseline,
not another arm.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_rho as RHO

# DERIVED FROM THE RUN'S OWN SHARDS, NOT HARDCODED.
#
# The constants that used to live here -- DockQ 0.2909, frac 0.5556, rho 0.1958, oracle 0.5968 --
# are all EIGHTEEN-INTERFACE figures from val10. Drawing them against a 92-interface run would put
# the reference line in the wrong place and every comparison on the plot would be wrong by an
# unknown amount. The shipped head's selection is recoverable exactly: each shard row carries
# stored_iptm, so argmax(stored_iptm) per interface IS what the shipped head would have picked.
def _shards_dir(rid, epoch, budget):
    """Which run in the chain actually holds this epoch's shards."""
    import glob
    for r in run_chain(rid):
        d = os.path.join("training", "runs", r, f"train_eval_shards_ep{epoch}")
        if glob.glob(os.path.join(d, f"rows_b{budget}_*.csv")):
            return d
    return os.path.join("training", "runs", rid, f"train_eval_shards_ep{epoch}")


def shipped_and_oracle(rid, epoch, budget):
    """(shipped_dockq, shipped_frac, shipped_rho, oracle_dockq, oracle_frac, n_ifaces)."""
    import glob
    from collections import defaultdict
    by = defaultdict(list)
    for f in glob.glob(os.path.join(_shards_dir(rid, epoch, budget),
                                    f"rows_b{budget}_*.csv")):
        for r in csv.DictReader(open(f)):
            by[r["iface"]].append((float(r["iptm"]), float(r["dockq"]), float(r["stored_iptm"])))
    s_dq, s_ok, o_dq, o_ok, rhos = [], [], [], [], []
    for v in by.values():
        best_sh = max(v, key=lambda x: x[2])[1]
        s_dq.append(best_sh); s_ok.append(1.0 if best_sh >= 0.23 else 0.0)
        best = max(x[1] for x in v)
        o_dq.append(best); o_ok.append(1.0 if best >= 0.23 else 0.0)
        r = RHO._spear([x[2] for x in v], [x[1] for x in v])
        if r is not None:
            rhos.append(r)
    n = max(1, len(by))
    return (sum(s_dq)/n, sum(s_ok)/n, (sum(rhos)/len(rhos) if rhos else float("nan")),
            sum(o_dq)/n, sum(o_ok)/n, len(by))


def train_rank_loss(rid):
    """{epoch: training ranking loss} from the run's own log, rank 0 only.

    THE SAME LOSS AS THE HELD-OUT ONE, which is why they may share an axis: same form
    (softplus), same tau, same cross-seed grouping, same gap weighting, same floor of 0. What
    differs is the population -- the training groups over overfit30 against the val10
    interfaces -- and the moment of measurement: the training value accumulates DURING the epoch
    while the weights move, the held-out value is computed at that epoch's checkpoint. So a small
    train-side lead is expected even with no overfitting at all.

    Rank 0 only. All three ranks print an identical figure (the loss is reduced across ranks
    before it is logged), so taking every line would triple-count each epoch.
    """
    p = os.path.join("training", "runs", rid, "run.log")
    if not os.path.exists(p):
        return {}
    pat = re.compile(r"\[rank 0\] epoch (\d+) rank-loss \([^)]*\): ([+-][0-9.]+)")
    out = {}
    with open(p, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def first_acceptable_rank(rid, budget, thresh=0.23, _chain=True):
    """{epoch: dict} -- rank of the FIRST acceptable pose when poses are ordered by ipTM.

    WHY THIS AND NOT frac. `frac acceptable` only asks whether rank 1 happens to be acceptable, so
    it is blind to the whole ordering below it: an arm that lifts the best pose from rank 90 to
    rank 2 scores identically to one that does nothing. This metric sees that, and it is still
    on-criterion -- rank 1 for every interface IS frac 1.0.

    Rank is 1-indexed, so 1 is perfect and lower is better. The threshold is `>=`, matching
    shipped_and_oracle so this panel and the frac panel agree about what "acceptable" means.

    CENSORING, AND WHY THE COUNT IS REPORTED. An interface with NO acceptable pose anywhere in its
    budget has no first acceptable rank at all. Dropping those silently would bias the mean
    optimistically -- the hardest interfaces would be the ones excluded. On val10 the oracle frac
    is 1.0000, so nothing is censored in practice, but the count travels with the number so that a
    future pool where it is not 1.0 cannot read as an improvement.
    """
    import glob
    from collections import defaultdict
    out = {}
    if _chain:
        for _r in run_chain(rid):
            out.update(first_acceptable_rank(_r, budget, thresh, _chain=False))
        return out
    pat = os.path.join("training", "runs", rid, f"train_eval_shards_ep*")
    for d in sorted(glob.glob(pat)):
        m = re.search(r"train_eval_shards_ep(\d+)$", d)
        if not m:
            continue
        epoch = int(m.group(1))
        by = defaultdict(list)
        for f in glob.glob(os.path.join(d, f"rows_b{budget}_*.csv")):
            for r in csv.DictReader(open(f)):
                by[r["iface"]].append((float(r["iptm"]), float(r["dockq"]), float(r["stored_iptm"])))
        if not by:
            continue
        res = {}
        for tag, key in (("trained", 0), ("shipped", 2)):
            ranks, cens, npose = [], 0, []
            for v in by.values():
                npose.append(len(v))
                ordered = sorted(v, key=lambda x: -x[key])
                hit = next((i + 1 for i, x in enumerate(ordered) if x[1] >= thresh), None)
                if hit is None:
                    cens += 1
                else:
                    ranks.append(float(hit))
            n = len(ranks)
            if not n:
                continue
            mu = sum(ranks) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in ranks) / (n - 1)) if n > 1 else 0.0
            sr = sorted(ranks)
            med = sr[n // 2] if n % 2 else 0.5 * (sr[n // 2 - 1] + sr[n // 2])
            res[tag] = {"mean": mu, "sem": (sd / math.sqrt(n) if n else float("nan")),
                        "median": med, "n": n, "censored": cens,
                        "poses": (sum(npose) / len(npose) if npose else 0),
                        "at1": sum(1 for x in ranks if x == 1.0)}
        # PAIRED, trained minus shipped on the SAME interface. The unpaired means above have an
        # s.e. of ~13 on a mean of ~37 because a handful of interfaces first become acceptable in
        # the 90s while most are acceptable at rank 1 -- between-interface variance swamps the
        # effect. Pairing removes it entirely, and every interface is scored by both orderings on
        # the same poses, so the pairing is exact rather than approximate.
        if "trained" in res and "shipped" in res:
            deltas = []
            for v in by.values():
                ot = sorted(v, key=lambda x: -x[0])
                os_ = sorted(v, key=lambda x: -x[2])
                ht = next((i + 1 for i, x in enumerate(ot) if x[1] >= thresh), None)
                hs = next((i + 1 for i, x in enumerate(os_) if x[1] >= thresh), None)
                if ht is not None and hs is not None:
                    deltas.append(float(ht - hs))
            if deltas:
                n = len(deltas)
                mu = sum(deltas) / n
                sd = math.sqrt(sum((x - mu) ** 2 for x in deltas) / (n - 1)) if n > 1 else 0.0
                res["paired"] = {
                    "mean": mu, "sem": (sd / math.sqrt(n) if n else float("nan")),
                    "n": n,
                    "better": sum(1 for x in deltas if x < 0),
                    "worse": sum(1 for x in deltas if x > 0),
                    "same": sum(1 for x in deltas if x == 0),
                }
        if res:
            out[epoch] = res
    return out


def run_chain(rid, runs=None):
    """[oldest_ancestor, ..., rid] -- a resumed run and everything it continues.

    A resume writes to a FRESH directory while continuing the parent's epoch numbering, so
    24u342's own records start at epoch 9 and plotting it alone gives a single point with no
    history. The link is not guessed from the name: args.json records the --resume checkpoint
    path, and the parent rid is the directory that path sits in. Name-based guessing would break
    the moment a resume is given an unrelated id, which is exactly when the plot matters most.

    Cycles are impossible through a checkpoint path but are guarded anyway, since a hand-edited
    args.json should not hang the plotter.
    """
    runs = runs or os.path.join("training", "runs")
    chain, seen = [rid], {rid}
    cur = rid
    for _ in range(32):
        try:
            args = json.load(open(os.path.join(runs, cur, "args.json")))
        except Exception:
            break
        ck = args.get("resume")
        if not ck:
            break
        parent = os.path.basename(os.path.dirname(os.path.abspath(ck)))
        if not parent or parent in seen or not os.path.isdir(os.path.join(runs, parent)):
            break
        chain.insert(0, parent)
        seen.add(parent)
        cur = parent
    return chain


def load_chain(rid, budget, runs=None):
    """rank_rho rows for the whole chain, child winning on any shared epoch."""
    runs = runs or os.path.join("training", "runs")
    merged = {}
    for r in run_chain(rid, runs):
        merged.update(RHO.load(runs, r, budget) or {})
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--no-rho", dest="with_rho", action="store_false",
                    help="drop the Spearman(ipTM, DockQ) panel. SHOWN BY DEFAULT, but it is NOT "
                         "the criterion: h7msge lifted rho +0.075 over shipped while selected "
                         "DockQ and frac acceptable both ended below shipped, which is the whole "
                         "reason the loss was changed. Kept visible because the two moving apart "
                         "is itself the finding worth watching.")
    a = ap.parse_args()

    # THROUGH rank_rho, NOT A HARDCODED FILENAME. This read "rank_rho_b{b}.csv" directly and so
    # silently found nothing once the cache identity became _v2 -- the plot was blocked on a stale
    # constant while the metrics themselves were fine. RHO.CACHE and RHO.load are the one source.
    got = load_chain(a.rid, a.budget)
    if not got:
        raise SystemExit(f"no cached epochs for {a.rid} at budget {a.budget} yet")
    IX = {n: i for i, n in enumerate(RHO.COLS)}
    ep = sorted(got)
    g = lambda k: [got[e][IX[k]] for e in ep]

    s_dq, s_fr, s_rho, o_dq, o_fr, n_if = shipped_and_oracle(a.rid, ep[-1], a.budget)
    # THE TWO METRICS THAT DECIDE IT, then the objective. Spearman is opt-in; see --with-rho.
    panels = [
        ("mean DockQ of the selected pose", g("dockq_mean"), g("dockq_sem"), s_dq, o_dq),
        ("fraction acceptable (DockQ > 0.23)", g("frac_mean"), g("frac_sem"), s_fr, o_fr),
    ]

    # RANK OF THE FIRST ACCEPTABLE POSE, ordering by ipTM. Read from the shards rather than the
    # rank_rho cache, which is at 25 fixed columns with an assert; adding two more would force a
    # cache identity bump and a recomputation of every epoch of every arm for a quantity that is
    # cheap to derive from files already on disk.
    #
    # LOWER IS BETTER AND 1 IS PERFECT, so this panel is inverted relative to the others -- the
    # y-axis is flipped so that "up" still means "improving", and the title says so, because a
    # panel where up means worse sitting beside three where up means better is a misreading waiting
    # to happen.
    far = first_acceptable_rank(a.rid, a.budget)
    far_ep = [e for e in ep if e in far and "trained" in far[e]]
    far_panel = None
    if far_ep:
        far_panel = (far_ep,
                     [far[e]["trained"]["mean"] for e in far_ep],
                     [far[e]["trained"]["sem"] for e in far_ep],
                     far[far_ep[-1]].get("shipped", {}).get("mean"),
                     [far[e].get("paired") for e in far_ep])
    if a.with_rho:
        panels.append(("Spearman(ipTM, DockQ) within interface",
                       g("rho_trained"), g("rho_sem"), s_rho, None))
    n_ax = len(panels) + 1 + (1 if far_panel else 0)   # + objective panel, + rank panel
    fig, axes = plt.subplots(1, n_ax, figsize=(5.25 * n_ax, 4.6))
    for ax, (title, y, se, ship, orac) in zip(axes, panels):
        lo = [v - s for v, s in zip(y, se)]
        hi = [v + s for v, s in zip(y, se)]
        ax.fill_between(ep, lo, hi, color="#1f77b4", alpha=0.18, lw=0)
        ax.plot(ep, y, color="#1f77b4", lw=2.0, marker="o", ms=3.5)
        ax.axhline(ship, color="#d62728", ls="--", lw=1.6)
        ax.text(ax.get_xlim()[1], ship, f" shipped {ship:.4f}", color="#d62728",
                fontsize=9, va="bottom", ha="right")
        if orac is not None:
            ax.axhline(orac, color="#2ca02c", ls=":", lw=1.4)
            ax.text(ax.get_xlim()[1], orac, f" oracle {orac:.4f}", color="#2ca02c",
                    fontsize=9, va="bottom", ha="right")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("epoch", fontsize=10.5)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel(f"held out, {n_if} interfaces, +- s.e. across interfaces", fontsize=10)

    if far_panel:
        fep, fy, fse, fship, fpair = far_panel
        axr = axes[len(panels)]
        lo = [v - s_ for v, s_ in zip(fy, fse)]
        hi = [v + s_ for v, s_ in zip(fy, fse)]
        axr.fill_between(fep, lo, hi, color="#9467bd", alpha=0.18, lw=0)
        axr.plot(fep, fy, color="#9467bd", lw=2.0, marker="o", ms=3.5)
        if fship is not None:
            axr.axhline(fship, color="#d62728", ls="--", lw=1.6)
            axr.text(axr.get_xlim()[0], fship, f" shipped {fship:.2f}", color="#d62728",
                     fontsize=9, va="bottom", ha="left")
        axr.axhline(1.0, color="#2ca02c", ls=":", lw=1.4)
        axr.text(axr.get_xlim()[0], 1.0, " oracle 1.00", color="#2ca02c",
                 fontsize=9, va="top", ha="left")
        axr.invert_yaxis()                      # see the comment where far_panel is built
        axr.set_title("rank of first acceptable pose (ipTM order)\nLOWER IS BETTER -- axis flipped",
                      fontsize=10.5)
        axr.set_xlabel("epoch", fontsize=10.5)
        axr.grid(alpha=0.15)
        # The unpaired s.e. is ~13 on a mean of ~37, so the paired delta is the number that can
        # actually resolve a change; it is annotated here rather than given a panel of its own.
        q = fpair[-1]
        if q:
            axr.text(0.02, 0.02,
                     f"paired vs shipped at ep{fep[-1]}: {q['mean']:+.2f} +- {q['sem']:.2f}"
                     f"  ({q['better']} better / {q['worse']} worse / {q['same']} same)",
                     transform=axr.transAxes, fontsize=8.5, va="bottom", color="#444444")

    # FOURTH PANEL: THE OBJECTIVE ITSELF, TRAIN AGAINST HELD OUT. Its own axes because this is a
    # loss near 0.04, not a DockQ-scale quantity, and sharing with the panels above would flatten
    # it to a line. The two curves are the SAME loss, which is the only thing that makes the
    # vertical gap between them a generalisation gap.
    #
    # WHICH HELD-OUT COLUMN IS THE MATCHING ONE DEPENDS ON THE RUN'S FORM. This read
    # xsloss_trained unconditionally, so an --rank-form expected arm would have had its NEGATIVE
    # training curve (-E[DockQ]) drawn against a POSITIVE pairwise softplus held-out curve, on
    # one axis, labelled as the same loss. Caught in review of 92338a36. rank_rho now computes
    # both metrics for every run and run_rank_form says which objective was trained.
    ax = axes[len(panels) + (1 if far_panel else 0)]
    _form, _tau = RHO.run_rank_form(os.path.join("training", "runs", a.rid))
    if _form == "expected":
        ho_col, ho_se_col = "exloss_trained", "exloss_sem"
        _taus = sorted({round(v, 6) for v in g("exloss_tau")})
        _lbl = f"-E[DockQ], softmax tau {_taus[0]:g}" if len(_taus) == 1 else "-E[DockQ]"
        _floor_txt = " floor -max(DockQ) per group"
    else:
        ho_col, ho_se_col = "xsloss_trained", "xsloss_sem"
        _lbl = "cross-seed softplus"
        _floor_txt = " floor 0 = every pair ordered"
    tr = {}
    for _r in run_chain(a.rid):
        tr.update(train_rank_loss(_r))
    tr_ep = [e for e in ep if e in tr]
    if tr_ep:
        ax.plot(tr_ep, [tr[e] for e in tr_ep], color="#ff7f0e", lw=2.0, marker="s", ms=3.5,
                label="training (overfit30)")
    ho, ho_se = g(ho_col), g(ho_se_col)
    ax.fill_between(ep, [v - s for v, s in zip(ho, ho_se)], [v + s for v, s in zip(ho, ho_se)],
                    color="#1f77b4", alpha=0.18, lw=0)
    ax.plot(ep, ho, color="#1f77b4", lw=2.0, marker="o", ms=3.5,
            label=f"held out (val10, {n_if} ifaces)")
    ax.axhline(0.0, color="#555555", ls="-", lw=0.9)
    ax.text(ax.get_xlim()[1], 0.0, _floor_txt, color="#555555",
            fontsize=8.5, va="bottom", ha="right")
    ax.set_title(f"{_lbl}: training vs held out", fontsize=11.5)
    ax.set_xlabel("epoch", fontsize=10.5)
    ax.set_ylabel(f"{_lbl} (lower is better)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.suptitle(f"{a.rid} -- cross-seed softplus  (bands are s.e. over {n_if} interfaces; "
                 f"dashed = SHIPPED head on these same {n_if}, derived from stored_iptm)",
                 fontsize=11.5, y=1.02)
    out = a.out or os.path.join("training", "runs", "three_panels.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({len(ep)} epoch(s): {ep[0]}..{ep[-1]})")


if __name__ == "__main__":
    main()
