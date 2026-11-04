#!/usr/bin/env python
"""Training side of one run, 2 x 3: what the optimiser sees, and what it buys ON ITS OWN DATA.

    python -I run_logging/plot_train_insample.py <rid> [out.png]

ROW 1 IS THE OPTIMISER'S VIEW, per epoch out of trajectory.csv and run.log: mean PAE loss, the
HELD-OUT RANKING LOSS's training-side rel term, and mean gradient norm. None of the three is a
result -- they say whether the run is descending. Row 1 falling is the null expectation for any
amount of data, which is exactly why it is drawn next to row 2 rather than on its own.

BOTH LOSS TERMS ARE DRAWN, which takes explaining because only one of them is in `loss`. The
objective is PAE + 75 * the per-interface rel term, but the rel term enters as GRADIENT
COEFFICIENTS computed in pass A and applied in pass B -- it is never accumulated as a loss value,
so trajectory.csv's `loss` column equals `pae_loss` on every row (checked: 0 mismatches in
156,640). The rel term's own magnitude is instead printed once per epoch per rank as
`epoch N rank-loss (rel, cross-seed pairs)`, and that line is what panel 2 reads and averages over
ranks. Plotting only `loss` hides half the objective.

The kept-token fraction used to occupy panel 3 and was dropped: it is a property of the DRAW, not
of the run's progress, and it sat at 0.412 +- 0.002 for 31 straight epochs without ever saying
anything. It is still in every epoch's report line if it is ever wanted.

ROW 2 IS THE IN-SAMPLE UNCROPPED EVAL, on the run's OWN training interfaces -- the same three
readouts, computed by the same code path, as the held-out plot's top row (plot_val_2x3.py). It is
the overfitting measurement: in-sample is the ceiling a given head and loss can reach when
generalisation is not asked for, so

    in-sample above shipped, held-out flat   ->  it fits, it does not transfer  (the usual arm)
    in-sample AT or BELOW shipped            ->  it does not even fit; more regularisation is the
                                                 wrong lever and the capacity or the augmentation
                                                 is the thing to change

The second case is not hypothetical -- egw5wq reads 0.5451 in-sample against shipped 0.5474 at
epoch 20, after the train loss has fallen 7%. A plot of row 1 alone is consistent with that run
working.

MEAN-POOLED ONLY. The MAX readout nothing optimises is in the held-out plot; repeating its row here
would double the panel count to say the same thing about a pool that is not the target.

THE POOL IS NAMED, NOT GLOBBED. A run with --extra-eval writes several pools per epoch under the
same epoch number, and an unqualified read drew egw5wq's in-sample numbers as held-out on
2026-11-03. This asks eval_dirs for `insample` and plots nothing rather than something wrong.
"""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import eval_dirs                                                             # noqa: E402
import plot_three                                                            # noqa: E402
import select_tied                                                           # noqa: E402

RID = sys.argv[1] if len(sys.argv) > 1 else "egw5wq"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/train_insample_{RID}.png"
POOL = os.environ.get("INSAMPLE_POOL_TAG", "insample")
ACC = 0.23
RUNS = os.path.join("training", "runs")
RIDS = list(plot_three.run_chain(RID))

# ---------------------------------------------------------------- row 1: trajectory.csv
# Column names, not indices: `epoch` has moved column twice as fields were added, and an index
# that silently reads `step_taken` instead plots a flat line of 1.0 that looks like a converged run.
# PER (rid, epoch), NOT POOLED ACROSS THE CHAIN. A resume continues the parent's epoch
# NUMBERING, so an epoch interrupted mid-flight appears TWICE: egw5wq was stopped 840 poses into
# epoch 28 and 57zxgz redid all 6100 of it. Pooling gave epoch 28 6940 rows, which then set the
# "full epoch" bar at 0.98 * 6940 = 6801 and disqualified every ordinary 6100-row epoch -- the
# plot came back with ONE point and looked like a run with no history. For each epoch the chain
# member with the MOST rows is the one that actually ran it to completion, so that one wins and
# the abandoned prefix is dropped rather than averaged into it.
_by = defaultdict(dict)                     # ep -> rid -> {"loss": [...], "grad": [...], "lr": v}
for rid in RIDS:
    pth = os.path.join(RUNS, rid, "trajectory.csv")
    if not os.path.exists(pth):
        continue
    for r in csv.DictReader(open(pth)):
        try:
            ep = int(r["epoch"])
        except (KeyError, ValueError, TypeError):
            continue
        slot = _by[ep].setdefault(rid, {"loss": [], "grad": [], "lr": None, "bpae": []})
        # THE EPOCH MEAN COMES FROM batch_pae_loss, NOT pae_loss. See the note at the top of
        # this block: pae_loss is ONE pose on THIS rank, and trajectory.csv is rank 0 only.
        if (r.get("step_taken") or "").strip() == "1":
            try:
                _v, _n = float(r["batch_pae_loss"]), float(r["batch_n"])
                if _v == _v and _v > 0 and _n > 0:
                    slot["bpae"].append((_v, _n))
            except (KeyError, ValueError, TypeError):
                pass
        for key, dest in (("loss", "loss"), ("grad_norm", "grad")):
            try:
                v = float(r[key])
            except (KeyError, ValueError, TypeError):
                continue
            if v == v and v > 0:            # drop the nan rows a skipped step writes
                slot[dest].append(v)
        try:
            slot["lr"] = float(r["lr"])
        except (KeyError, ValueError, TypeError):
            pass

loss, grad, lr, bpae = {}, {}, {}, {}
for ep, per_rid in _by.items():
    rid = max(per_rid, key=lambda k: len(per_rid[k]["loss"]))
    loss[ep] = per_rid[rid]["loss"]
    grad[ep] = per_rid[rid]["grad"]
    bpae[ep] = per_rid[rid]["bpae"]
    if per_rid[rid]["lr"] is not None:
        lr[ep] = per_rid[rid]["lr"]

# The last epoch is PARTIAL while the run is live -- its mean is over a prefix of the shuffle and
# is not comparable with a full epoch. The bar is the MODE of the per-epoch counts, not the max:
# one long epoch must not disqualify the rest, which is the same failure in a different dress.
_counts = [len(v) for v in loss.values() if v]
full = st.mode(_counts) if _counts else 0
t_eps = sorted(e for e in loss if len(loss[e]) >= 0.98 * full)
def _epoch_pae(e):
    """All-ranks, all-poses epoch mean, weighted by batch_n.

    Falls back to the rank-0 per-pose mean only if batch_pae_loss is absent, which is the
    case for runs predating that column. The fallback is unbiased but ~6.6x noisier.
    """
    b = bpae.get(e) or []
    if b:
        return sum(v * n for v, n in b) / sum(n for _, n in b)
    return st.mean(loss[e])

t_loss = [_epoch_pae(e) for e in t_eps]
t_grad = [st.mean(grad[e]) if grad.get(e) else float("nan") for e in t_eps]

# LR steps, for the verticals -- read off the schedule the run actually ran, not the flag.
steps = [e for e in t_eps[1:] if lr.get(e, 0) < lr.get(e - 1, 0)]

# THE REL TERM, one line per rank per epoch, averaged over ranks. Each rank reports the mean over
# the interface-groups IT held, and --rank-cross-seed splits a structure's seeds across ranks, so
# the three numbers are means over disjoint equal-sized partitions and their unweighted mean is
# the epoch's mean.
RANKL = re.compile(r"epoch (\d+) rank-loss \([^)]*\): ([+-]?[0-9.]+) over (\d+) interface-group")
rankl = defaultdict(list)
for rid in RIDS:
    p = os.path.join(RUNS, rid, "run.log")
    if os.path.exists(p):
        for m in RANKL.finditer(open(p, errors="replace").read()):
            rankl[int(m.group(1))].append(float(m.group(2)))
k_eps = sorted(e for e in rankl if e in set(t_eps))
k_val = [st.mean(rankl[e]) for e in k_eps]

# ---------------------------------------------------------------- row 2: in-sample eval
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")

# idx 0 = MAX-pooled, 1 = MEAN-pooled, 2 = shipped, 3 = true DockQ -- same order as plot_val_2x3
e_eps, ins, ship = [], defaultdict(list), defaultdict(list)
n_iface = 0
for ep, d in sorted(eval_dirs.chain_shard_dirs(RIDS, RUNS, tag=POOL).items()):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(d, "rows_b*_s*.csv")):
        for r in csv.DictReader(open(f)):
            vm, vs = r.get("iptm_mean"), r.get("stored_iptm")
            if not vm or not vs or not r.get("dockq"):
                continue
            by[r["iface"]].append((float(r["iptm"]), float(vm), float(vs), float(r["dockq"])))
    if not by:
        continue
    e_eps.append(ep)
    n_iface = max(n_iface, len(by))
    dq, fa, rh, sdq, sfa, srh = [], [], [], [], [], []
    for v in by.values():
        a, b = select_tied.expected_pick(v, 1, 3, ACC);  dq.append(a);  fa.append(b)
        a, b = select_tied.expected_pick(v, 2, 3, ACC); sdq.append(a); sfa.append(b)
        truth = [p[3] for p in v]
        rh.append(spearman([p[1] for p in v], truth))
        srh.append(spearman([p[2] for p in v], truth))
    ins["dq"].append(sum(dq) / len(dq));   ship["dq"].append(sum(sdq) / len(sdq))
    ins["fa"].append(sum(fa) / len(fa));   ship["fa"].append(sum(sfa) / len(sfa))
    ins["rho"].append(st.median(rh));      ship["rho"].append(st.median(srh))

# ---------------------------------------------------------------- draw
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))

def finish(ax, title, xs):
    ax.set_title(title, fontsize=12.5, weight="bold", pad=8)
    ax.set_xlabel("epoch", fontsize=11)
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for e in steps:
        ax.axvline(e, ls=":", color="#b03a3a", lw=1.4, zorder=1)
    if xs:
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

TOP = [(t_loss, "train mean PAE loss (all ranks, all poses)", "#1f77b4", t_eps),
       (k_val, "rel term, per interface-group (the OTHER half of the objective)", "#8e44ad", k_eps),
       (t_grad, "mean gradient norm", "#e67e22", t_eps)]
for ci, (y, lab, col, xs) in enumerate(TOP):
    ax = axes[0][ci]
    if y:
        ax.plot(xs, y, "o-", color=col, lw=2.2, ms=4.5, zorder=4)
    else:
        ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#888")
    finish(ax, lab, xs)
if steps:
    axes[0][0].annotate("dotted = LR drop (" + ", ".join(f"ep {e}" for e in steps) + ")",
                        xy=(.02, .06), xycoords="axes fraction", fontsize=9.5, color="#b03a3a")
if k_val:
    _lo, _hi = min(k_val), max(k_val)
    _pad = max((_hi - _lo) * .12, 1e-5)
    axes[0][1].set_ylim(_lo - _pad, _hi + _pad)
    axes[0][1].annotate(f"{k_val[0]:.5f} -> {k_val[-1]:.5f}  ({100*(k_val[-1]/k_val[0]-1):+.1f}%)",
                        xy=(.03, .06), xycoords="axes fraction", fontsize=9.5, color="#8e44ad")

BOT = [("dq", "mean selected DockQ", "#1b6ca8"),
       ("fa", f"frac acceptable (>= {ACC})", "#2e8b57"),
       ("rho", "median Spearman vs DockQ", "#8e44ad")]
for ci, (key, lab, col) in enumerate(BOT):
    ax = axes[1][ci]
    if e_eps:
        y, s = ins[key], ship[key]
        ax.plot(e_eps, y, "o-", color=col, lw=2.5, ms=7.5, zorder=4, label="ours")
        ax.axhline(s[-1], ls="--", color="#555", lw=1.8, zorder=3, label=f"shipped {s[-1]:.4f}")
        ax.set_xticks(e_eps)
        lo = min(min(y), s[-1]); hi = max(max(y), s[-1]); pad = max((hi - lo) * .45, 2e-4)
        ax.set_ylim(lo - pad, hi + pad)
        ax.legend(fontsize=10, framealpha=.95)
    else:
        ax.text(.5, .5, f"no `{POOL}` eval shards", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
    finish(ax, f"IN-SAMPLE  --  {lab}", e_eps)

fig.suptitle(f"{'+'.join(RIDS)}  --  training signal (row 1) and in-sample uncropped eval on "
             f"{n_iface} TRAINING interfaces (row 2)", fontsize=13.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.955))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=135)
print(f"  {len(t_eps)} train epoch(s), {len(e_eps)} in-sample eval epoch(s) -> {OUT}")
print(f"  {'ep':>4} {'PAE loss':>10} {'rel term':>10} {'grad':>8}")
for e, a, b in zip(t_eps, t_loss, t_grad):
    _r = st.mean(rankl[e]) if rankl.get(e) else float("nan")
    print(f"  {e:>4} {a:>10.4f} {_r:>10.5f} {b:>8.3f}")
if e_eps:
    print(f"  {'ep':>4} {'IS dq':>10} {'IS fa':>10} {'IS rho':>10}")
    for i, e in enumerate(e_eps):
        print(f"  {e:>4} {ins['dq'][i]:>10.4f} {ins['fa'][i]:>10.4f} {ins['rho'][i]:>10.4f}")
    print(f"  {'ship':>4} {ship['dq'][-1]:>10.4f} {ship['fa'][-1]:>10.4f} {ship['rho'][-1]:>10.4f}")
