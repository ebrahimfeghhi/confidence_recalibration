#!/usr/bin/env python
"""The uncropped hard-list eval across epochs, with the SHIPPED head as epoch 0.

    python -I run_logging/plot_hard_eval.py <rid> [out.png]

EPOCH 0 IS THE SHIPPED HEAD ON THE SAME POSES. Every metric json carries baseline_mean_dockq and
baseline_frac_acc -- the shipped head's own selection over exactly the interfaces and candidates
that pass scored, reproduced from the stored ipTM. Drawing a shipped number from any other source
would compare two different candidate sets, which is how a 0.006 gap was once misattributed to
stored-vs-forwarded ipTM when it was cropped-vs-uncropped.

THE TWO PASSES ARE POOLED OVER INTERFACES, not averaged as two numbers: hard15 holds 130
interfaces and hard1 holds 21, so a plain mean of the two would give the 21 six times the weight
they have. Each pass is also drawn on its own, because a change in only one of them is worth
seeing rather than smoothing away.
"""
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RIDS = (sys.argv[1] if len(sys.argv) > 1
        else sys.exit("usage: plot_hard_eval.py <rid>[,<rid2>...] [out.png]")).split(",")
RID = RIDS[-1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/hard_eval_{RID}.png"
PASSES = ("hardall15", "hard15", "hard1")

got, src = {}, {}
for rid in RIDS:
    for nm in PASSES:
        for f in glob.glob(f"training/runs/{rid}/metric_{nm}_ep*.json"):
            ep = int(re.search(r"_ep(\d+)\.json", f).group(1))
            try:
                got.setdefault(ep, {})[nm] = json.load(open(f))
            except (ValueError, OSError):
                continue
            src[ep] = rid
if not got:
    sys.exit(f"STOP: no metric_hard*_ep*.json under {RIDS} yet.")


def pooled(dd, key, bkey=None):
    """Interface-weighted pool over whichever passes are present."""
    num = den = 0.0
    for nm, d in dd.items():
        n = float(d.get("n_interfaces") or 0)
        v = d.get(bkey if bkey else key)
        if n <= 0 or v is None:
            continue
        num += float(v) * n
        den += n
    return (num / den) if den else float("nan")


eps = sorted(got)
# THE BASELINE IS THE SAME AT EVERY EPOCH -- it is the shipped head -- so take it from the first
# epoch that reports one, and say so if they ever disagree.
base_d, base_f = float("nan"), float("nan")
for ep in eps:
    b1 = pooled(got[ep], "", "baseline_mean_dockq")
    b2 = pooled(got[ep], "", "baseline_frac_acc")
    if b1 == b1:
        base_d, base_f = b1, b2
        break

# ---- THE MARGIN TO CORRECT SELECTION, from the passes' own per-pose rows -----------------------
#
# gap = score(the top-ranked pose) - score(the best acceptable pose), averaged over the winnable
# lists. It is >= 0 by construction and exactly 0 on a list the head already gets right, so the
# curve is "how far from selecting correctly", and 0 is success rather than an arbitrary level.
#
# WHY IT IS WORTH ITS OWN PANEL. top-1 is a step function and mean DockQ mixes "picked a different
# pose" with "picked a slightly worse one". This says how much the good pose would have to gain,
# in score, for the pick to flip -- measured on shipped it is +0.1628 at the median, which is a
# wide margin and not a near miss.
#
# BACKFILLED FROM ROWS THAT ALREADY EXIST. Every pass keeps rows_*.csv carrying iptm, stored_iptm
# and dockq per pose, so every past epoch can be computed without re-running anything.
ACCEPT = 0.23


def margin(_unused, ep, col):
    """Mean over winnable lists of score(top pose) - score(best acceptable pose).

    THE RID COMES FROM `src`, not from the caller: with a chained arm, epoch 50's rows live under
    one run id and epoch 55's under another, so a single rid would silently find nothing for half
    the epochs and the curve would have holes where the resume was.
    """
    by = {}
    for nm in PASSES:
        for f in glob.glob(f"training/runs/{src.get(ep, RID)}/{nm}_eval_shards_ep{ep}/rows_*.csv"):
            for r in csv.DictReader(open(f)):
                try:
                    by.setdefault(r["iface"], []).append((float(r[col]), float(r["dockq"])))
                except (TypeError, ValueError, KeyError):
                    pass
    gaps = []
    for cand in by.values():
        acc = [c for c in cand if c[1] >= ACCEPT]
        if not acc:
            continue                      # unwinnable: no margin is defined
        gaps.append(max(c[0] for c in cand) - max(c[0] for c in acc))
    return (sum(gaps) / len(gaps)) if gaps else float("nan"), len(gaps)


plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
# ONE SERIES AND ONE REFERENCE PER PANEL. The per-pass curves were dropped: hard15 carries 130 of
# the 151 interfaces, so the pooled line is hard15 to within a few percent and the third curve
# only crowded the axes. The passes remain separate in the printed numbers below, where reading
# one against the other is actually possible.
#
# SHIPPED IS A LINE, NOT A MARKER. A diamond at x=0 put the baseline inside the data region and
# made the curve look like it started there; a horizontal rule says "this is the level to beat"
# across the whole axis, which is what it is.
# WHERE THE TRAINING SET CHANGED. A resume marker says the run id changed; it does not say the
# arm started seeing different data. The pool extension is a different kind of break and gets its
# own rule, or a step at that epoch reads as learning.
_PX = "training/overfit_hard/extended_at_epoch"
_px = None
try:
    _px = float(open(_PX).read().strip())
except (OSError, ValueError):
    pass
# epoch:label, one line per REAL change. A resume that changed nothing gets no rule -- four
# vertical lines on a 20-point axis is unreadable and three of them meant nothing.
_EVENTS = []
try:
    for _l in open("training/runs/arm_events.txt"):
        _l = _l.strip()
        if _l and _l[0] != "#":
            _parts = _l.split("|")
            if len(_parts) == 3:
                _e, _rid, _lab = _parts
                if _rid.strip() not in RIDS:   # another chain's history, not this one's
                    continue
            else:                               # old two-field form
                _e, _lab = _l.split(":", 1)
            _EVENTS.append((float(_e), _lab.strip()))
except OSError:
    pass



def _rules(a_, first):
    for _e, _lab in _EVENTS:
        _c = "tab:green" if "data" in _lab.lower() else "tab:orange"
        a_.axvline(_e, color=_c, ls="-." if _c == "tab:green" else "--", lw=1.8,
                   label=_lab if first else None)

for a_, key, lab in ((ax[0], "mean_dockq", "mean DockQ of the selected pose"),
                     (ax[1], "frac_acc", "fraction acceptable at top-1")):
    b = base_d if key.startswith("mean") else base_f
    ys = [pooled(got[e], key) for e in eps]
    if b == b:
        a_.axhline(b, color="#555", ls="--", lw=1.6, label="shipped head, same poses")
    a_.plot(eps, ys, "-o", color="tab:blue", lw=2, ms=6, label="this arm")
    _rules(a_, a_ is ax[0])
    a_.set_xlabel("epoch"); a_.set_ylabel(lab)
    a_.set_title(lab, fontsize=11.5)
    a_.grid(alpha=.3)
    a_.set_xticks(eps[:: max(1, len(eps) // 6)])
    _v = [y for y in ys if y == y] + ([b] if b == b else [])
    if len(_v) > 1:
        _lo, _hi = min(_v), max(_v)
        _m = (_hi - _lo) * .30 or abs(_hi) * .05 or 1e-3
        a_.set_ylim(_lo - _m, _hi + _m)

# ONE LEGEND, OUTSIDE BOTH PANELS, for the same reason plot_rel_arm puts it there: a per-panel box
# sits in the data region and starts covering points as soon as a curve climbs into that corner.
# PANEL 3: the margin, with shipped's own margin as the reference.
_gm = [margin(RID, e, "iptm") for e in eps]
_gb = [margin(RID, e, "stored_iptm") for e in eps]
_bse = next((g for g, n in _gb if g == g), float("nan"))
ax[2].axhline(0.0, color="tab:green", ls=":", lw=1.6, label="0 = top pick is acceptable")
if _bse == _bse:
    ax[2].axhline(_bse, color="#555", ls="--", lw=1.6)
ax[2].plot(eps, [g for g, _n in _gm], "-o", color="tab:blue", lw=2, ms=6)
ax[2].set_xlabel("epoch")
ax[2].set_ylabel("top score - best acceptable score")
ax[2].set_title("margin to correct selection\n(lower is better, 0 = correct)", fontsize=11.5)
_rules(ax[2], False)
ax[2].grid(alpha=.3); ax[2].set_xticks(eps[:: max(1, len(eps) // 6)])

_h, _l = ax[0].get_legend_handles_labels()
_h2, _l2 = ax[2].get_legend_handles_labels()
_h, _l = _h + _h2, _l + _l2
# DEDUPED. Two panels contribute the same shipped-head handle, and a repeated entry is what
# pushed the row past the figure width.
_seen, _hh, _ll = set(), [], []
for _x, _y in zip(_h, _l):
    if _y and _y not in _seen:
        _seen.add(_y); _hh.append(_x); _ll.append(_y)
fig.legend(_hh, _ll, loc="lower center", ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.004), handlelength=2.4, columnspacing=1.8)
fig.suptitle(f"UNCROPPED hard lists -- {' -> '.join(RIDS)}   "
             f"({len(eps)} eval epoch(s), 151 interfaces)", fontsize=13)
fig.tight_layout(rect=[0, .13, 1, .92])
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"-> {OUT}")
print(f"  epoch 0 SHIPPED      dockq {base_d:.4f}   frac {base_f:.4f}")
print(f"  shipped margin to correct selection: {_bse:+.4f}")
for e, (g, n) in zip(eps, _gm):
    print(f"  epoch {e:<3d} margin {g:+.4f} over {n} winnable list(s)")
for e in eps:
    print(f"  epoch {e:<3d} this arm   dockq {pooled(got[e], 'mean_dockq'):.4f}   "
          f"frac {pooled(got[e], 'frac_acc'):.4f}   "
          f"({src.get(e, RID)}; passes: {', '.join(sorted(got[e]))})")
