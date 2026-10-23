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
def shipped_and_oracle(rid, epoch, budget):
    """(shipped_dockq, shipped_frac, shipped_rho, oracle_dockq, oracle_frac, n_ifaces)."""
    import glob
    from collections import defaultdict
    by = defaultdict(list)
    for f in glob.glob(os.path.join("training", "runs", rid,
                                    f"train_eval_shards_ep{epoch}", f"rows_b{budget}_*.csv")):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()

    # THROUGH rank_rho, NOT A HARDCODED FILENAME. This read "rank_rho_b{b}.csv" directly and so
    # silently found nothing once the cache identity became _v2 -- the plot was blocked on a stale
    # constant while the metrics themselves were fine. RHO.CACHE and RHO.load are the one source.
    got = RHO.load(os.path.join("training", "runs"), a.rid, a.budget)
    if not got:
        raise SystemExit(f"no cached epochs for {a.rid} at budget {a.budget} yet")
    IX = {n: i for i, n in enumerate(RHO.COLS)}
    ep = sorted(got)
    g = lambda k: [got[e][IX[k]] for e in ep]

    s_dq, s_fr, s_rho, o_dq, o_fr, n_if = shipped_and_oracle(a.rid, ep[-1], a.budget)
    panels = [
        ("mean DockQ of the selected pose", g("dockq_mean"), g("dockq_sem"), s_dq, o_dq),
        ("fraction acceptable (DockQ > 0.23)", g("frac_mean"), g("frac_sem"), s_fr, o_fr),
        ("Spearman(ipTM, DockQ) within interface", g("rho_trained"), g("rho_sem"), s_rho, None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
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
    ax = axes[3]
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
    tr = train_rank_loss(a.rid)
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
