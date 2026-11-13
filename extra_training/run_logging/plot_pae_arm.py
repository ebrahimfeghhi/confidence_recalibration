#!/usr/bin/env python
"""Three panels for a PAE-ONLY arm: training PAE loss, held-out mean DockQ, held-out frac-acc.

WHY A SEPARATE PLOTTER. plot_rel_arm.py draws four panels, three of which -- the relative-iTM
term, within-interface Spearman and score spread -- come from the RANKING objective. An arm with
--pae-only and no --rank-loss has none of those, so three of its four panels stay empty for the
whole run. That is not a degraded view of the arm, it is the wrong instrument.

WHAT IS ACTUALLY WORTH LOOKING AT HERE. The PAE loss is the only thing being optimised, and the
held-out selection metrics are the only things that decide whether the arm worked. Training loss
going down is what any working optimiser does; this project has recorded cropped training
improving a training metric WITHOUT improving held-out selection four times, so the two are drawn
side by side and the shipped baseline is drawn as a line rather than quoted in a caption.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_curve as EC                                            # noqa: E402
import plot_three                                                  # noqa: E402

RID = (sys.argv[1] if len(sys.argv) > 1 else "privsp").split(",")[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/{RID}_pae.png"
CHAIN = list(plot_three.run_chain(RID))

# training PAE per epoch, from the same helper plot_rel_arm uses
EP = plot_three.epoch_table(CHAIN) if hasattr(plot_three, "epoch_table") else None
if EP is None:
    import csv
    import collections
    # THE NEWEST RUN IN THE CHAIN WINS AN EPOCH; THE CHAIN IS NOT CONCATENATED.
    #
    # A resume at an epoch boundary ALWAYS leaves the abandoned run holding a partial next epoch:
    # we stop it after its checkpoint, but it has already written rows for the epoch it was part
    # way through. Summing the chain then averages a partial attempt together with the complete
    # redo of the SAME epoch number. Measured on this chain 2026-11-11: ea9vy2 holds 3,938 rows
    # for epoch 3 and 7roo28 holds the full 11,541, and the concatenating version reported epoch 3
    # over 15,479 rows -- a mean of two different attempts, one of them a third of an epoch, drawn
    # as one point with nothing saying so.
    #
    # eval_curve.series already resolves the held-out metrics this way ("newest run winning"), so
    # this makes the training panel agree with the panels beside it. CHAIN is oldest-first, so the
    # last run that has rows for an epoch is the one that owns it.
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid in CHAIN:
        p = f"training/runs/{rid}/trajectory.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            try:
                e = int(float(r["epoch"]))
                v = float(r["pae_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if v == v:
                per[rid][e].append(v)
    acc = collections.defaultdict(list)
    for rid in CHAIN:
        for e, vs in per.get(rid, {}).items():
            acc[e] = vs
    # A PARTIAL EPOCH IS DRAWN HOLLOW, not hidden and not drawn solid: the last epoch of a live
    # run is always partial and its mean moves as it fills.
    # PARTIAL EPOCHS ARE DROPPED, NOT DRAWN HOLLOW. At the user's direction. The last epoch of a
    # live run is always incomplete and its mean moves as it fills, so plotting it puts a number
    # on the page that will be different in ten minutes -- a hollow marker says so but does not
    # stop anyone reading it. A COMPLETED EPOCH IS ONE THAT WROTE ITS CHECKPOINT.
    import glob as _glob
    import re as _re
    done = set()
    for rid in CHAIN:
        for f in _glob.glob(f"training/runs/{rid}/head_ep*.pt"):
            m = _re.search(r"head_ep(\d+)_", f)
            if m:
                done.add(int(m.group(1)))
    EP = sorted((e, sum(v) / len(v), True) for e, v in acc.items() if e in done)

S, SER = EC.shipped(), EC.series(CHAIN)
RHO, RHO_SHIP = EC.rho_series(CHAIN)
# AT THE USER'S DIRECTION: blue is the MAX pooling, orange is the MEAN pooling, for both the
# arm's markers and the shipped baselines, so a reader never has to match a line to a series.
C_MAX, C_MEAN = "tab:blue", "tab:orange"
C_MAX_B, C_MEAN_B = "tab:blue", "tab:orange"
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
fig.suptitle(f"PAE-only arm -- {RID}    (held-out: 83 structures / 141 interfaces, budget 30)")

# 1. the only thing being optimised
xs = [e for e, _v, _d in EP]
ys = [v for _e, v, _d in EP]
if xs:
    ax[0].plot(xs, ys, "-o", color="tab:blue", ms=5)
else:
    ax[0].text(.5, .5, "no completed epoch yet", ha="center", va="center",
               transform=ax[0].transAxes, color="#666")
ax[0].set_title("training PAE loss")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("PAE loss (all ranks)")
ax[0].grid(alpha=.3)

# 2 and 3. the only things that decide whether it worked
eps = sorted(SER)
for k, (col, lab, ship, orc) in enumerate((
        ("mean_dockq", "mean DockQ of the selected pose", S.get("mean_dockq"), S.get("oracle_mean_dockq")),
        ("frac_acc", "fraction acceptable (DockQ >= 0.23)", S.get("frac_acc"), S.get("oracle_frac_acc")))):
    # THE SHIPPED LINE MUST MATCH THE POOLING THE ARM SELECTS BY, or the plot invites the same
    # apples-to-oranges reading. eval_curve.shipped() returns the MEAN-pooled baseline; the metric
    # json carries the MAX-pooled one as baseline_*, written by the same pass. Reading this arm's
    # max-pooled 0.5850 against the mean-pooled 0.5735 is the mistake this prevents.
    _sb = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
    if _sb == "max":
        _bv = next((SER[e].get("baseline_" + col) for e in eps
                    if SER[e].get("baseline_" + col) is not None), None)
        if _bv is not None:
            ship = _bv
    a = ax[k + 1]
    if eps:
        a.plot(eps, [SER[e].get(col) for e in eps], "-s",
               color=(C_MAX if _sb == "max" else C_MEAN), ms=6,
               label=f"arm, {_sb}-pooled ipTM (what it selects on)")
        # THE OTHER POOLING, WHICHEVER IT IS. A run that SELECTS by the mean writes its max
        # column as `_select_max`, and one that selects by the max writes its mean as
        # `_select_mean`; the json's own `select_by` says which it is. Drawing whichever is
        # present, labelled by what it actually is, stops the two being confused -- which is
        # exactly the mistake that made me read this arm's max-pooled 0.5850 against the shipped
        # MEAN baseline of 0.5735 instead of the max baseline of 0.5813.
        _selby = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
        _other = "mean" if _selby == "max" else "max"
        mx = f"{col}_select_{_other}"
        if any(SER[e].get(mx) is not None for e in eps):
            a.plot(eps, [SER[e].get(mx) for e in eps], "-s",
                   color=(C_MAX if _other == "max" else C_MEAN), ms=6,
                   label=f"arm, {_other}-pooled ipTM")
    else:
        a.text(.5, .5, "no held-out eval yet\n(first at epoch 5)", ha="center", va="center",
               transform=a.transAxes, color="#666")
    # TWO BASELINES, EACH IN ITS SERIES' COLOUR. One grey line invited reading the arm's
    # max-pooled point against the mean-pooled baseline, which is the mistake that turned
    # +0.0038 into a reported +0.0115. Blue is max-pooled, orange is mean-pooled, matching the
    # markers above them.
    #
    # BOTH COME FROM eval_curve.shipped(), NOT FROM THE ARM'S OWN METRIC JSON. The max line used
    # to be read from SER[e]["baseline_<col>"], which is written by the arm's eval pass -- so it
    # could not be drawn until that arm reached epoch 5, and every plot before then showed the
    # mean-pooled baseline alone with nothing saying the other was missing. A reader comparing a
    # training curve against "the shipped line" would have been comparing against one pooling
    # without being told which.
    #
    # The two shipped passes carry BOTH poolings already: metric_shipped_kept30.json (92
    # interfaces) and metric_shipped_val30.json (49) each hold mean_dockq / frac_acc under
    # select_by=mean plus mean_dockq_select_max / frac_acc_select_max, and eval_curve.KEYS
    # combines all four over the 141. Verified 2026-11-11 to reproduce the project's quoted
    # numbers exactly: max-pooled 0.5813 / 0.7234, mean-pooled 0.5735 / 0.7163. So nothing is
    # hardcoded and nothing waits on the arm.
    _bmax = S.get(col + "_select_max") if S else None
    _bmean = S.get(col) if S else None
    if _bmax is not None:
        a.axhline(_bmax, color=C_MAX_B, lw=1.4, ls="-",
                  label=f"shipped, max-pooled ({_bmax:.4f})")
    if _bmean is not None:
        a.axhline(_bmean, color=C_MEAN_B, lw=1.4, ls="-",
                  label=f"shipped, mean-pooled ({_bmean:.4f})")
    a.set_title(lab + (f"\noracle {orc:.4f}" if orc else ""))
    a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[1].set_ylabel("mean DockQ"); ax[2].set_ylabel("frac acceptable")

# 4. THE CORRELATION. rho is ipTM against DockQ WITHIN an interface -- the thing the confidence
# head is for, independent of which pose a selection rule happens to pick. It is kept next to the
# selection panels deliberately: this project has recorded five arms whose rho ROSE while their
# selected DockQ FELL, so reading one without the other is how that keeps being missed.
_re = sorted(RHO)
if _re:
    ax[3].plot(_re, [RHO[e]["max"] for e in _re], "-s", color=C_MAX, ms=6,
               label="arm, max-pooled ipTM")
    ax[3].plot(_re, [RHO[e]["mean"] for e in _re], "-s", color=C_MEAN, ms=6,
               label="arm, mean-pooled ipTM")
else:
    ax[3].text(.5, .5, "no rho yet\n(needs the iptm_mean column,\nwritten under --pool-mean)",
               ha="center", va="center", transform=ax[3].transAxes, color="#666")
if RHO_SHIP == RHO_SHIP:
    # BLUE, AND LABELLED MAX-POOLED, because that is what it is. eval_curve.rho_series derives it
    # from results.csv's stored_iptm, which holds ONE pooling -- the max-pooled shipped ipTM. A
    # grey line with no pooling in its label sat under BOTH arm curves and invited reading the
    # arm's mean-pooled rho against it, which is not a pooling-matched comparison; Codex raised
    # exactly this on issue 15. There is no mean-pooled shipped rho anywhere in the repo and it
    # is not derivable from stored_iptm -- it needs the shipped head forwarded with --pool-mean --
    # so the honest thing is to draw the one that exists, name its pooling, and say the other is
    # absent rather than let a reader assume the line covers both.
    ax[3].axhline(RHO_SHIP, color=C_MAX_B, lw=1.4,
                  label=f"shipped, max-pooled ({RHO_SHIP:+.4f})")
    ax[3].plot([], [], ' ', label="shipped mean-pooled rho: not computed")
ax[3].set_title("within-interface Spearman, ipTM vs DockQ")
ax[3].set_xlabel("epoch"); ax[3].set_ylabel("median rho"); ax[3].grid(alpha=.3)
h, l = ax[1].get_legend_handles_labels()
_h3, _l3 = ax[3].get_legend_handles_labels()
for _hh, _ll in zip(_h3, _l3):
    if _ll not in l:
        h.append(_hh); l.append(_ll)
if h:
    # BELOW THE AXES, NOT ON TOP OF THEM. At ncol=3 and -.04 the legend sat over the "epoch"
    # labels of the middle panels. tight_layout runs FIRST so it measures the axes without the
    # legend, then the legend is placed in the margin that rect reserves for it.
    fig.tight_layout(rect=(0, .14, 1, .95))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .005),
               fontsize=9)
else:
    fig.tight_layout(rect=(0, .04, 1, .95))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"-> {OUT}")
print(f"\n{RID}  (chain {' -> '.join(CHAIN)})")
print(f"{'ep':>5} {'train PAE':>11}")
for e, v, d in EP:
    print(f"{e:5d} {v:11.4f}")
if eps:
    # THE TABLE MUST NAME THE POOLING IT PRINTS. The primary column is whatever `select_by` says;
    # the alternate is written as _select_<the other one>. Hardcoding _select_max printed nan for
    # a max-selected arm, whose alternate is _select_mean -- the same confusion that made me read
    # 0.5850 against the wrong baseline.
    _sb2 = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
    _alt = "mean" if _sb2 == "max" else "max"
    _bm = next((SER[e].get("baseline_mean_dockq") for e in eps
                if SER[e].get("baseline_mean_dockq") is not None), S.get("mean_dockq"))
    _bf = next((SER[e].get("baseline_frac_acc") for e in eps
                if SER[e].get("baseline_frac_acc") is not None), S.get("frac_acc"))
    print(f"\n{'ep':>5} {_sb2 + '-pool':>11} {_alt + '-pool':>11} {'frac':>9} {'frac(' + _alt + ')':>12}"
          f"   shipped {_sb2}-pooled {_bm:.4f} / {_bf:.4f}")
    for e in eps:
        _a = SER[e].get(f"mean_dockq_select_{_alt}")
        _af = SER[e].get(f"frac_acc_select_{_alt}")
        print(f"{e:5d} {SER[e].get('mean_dockq'):11.4f} "
              f"{(_a if _a is not None else float('nan')):11.4f} "
              f"{SER[e].get('frac_acc'):9.4f} "
              f"{(_af if _af is not None else float('nan')):12.4f}")
else:
    print("\nno held-out eval yet -- first at epoch 5")
if _re:
    print(f"\n{'ep':>5} {'rho max':>9} {'rho mean':>10}   shipped {RHO_SHIP:+.4f}")
    for e in _re:
        print(f"{e:5d} {RHO[e]['max']:+9.4f} {RHO[e]['mean']:+10.4f}")
