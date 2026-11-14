#!/usr/bin/env python
"""The distribution of |g_rank|/|g_pae| over a run's steps, and the alpha it implies.

    python -I run_logging/grad_split_report.py <rid> [<rid> ...]

WHY A DISTRIBUTION AND NOT A NUMBER. --grad-probe measures one accumulation window and exits, so
every alpha_rank this project has used came from a single draw over 2-3 structures. The five such
measurements on record span ratio 0.0041 to 0.4728. Most of that is not noise -- they froze
different fractions of the head, which changes the denominator by ~50x -- but even the three
comparable ones span 2x. --grad-norm-every writes one row per measured step, which is what makes
a spread readable rather than inferred.

READ alpha_for_parity, NOT ratio, ACROSS RUNS. ratio has the run's own alpha_rank folded in;
alpha_for_parity divides it out and answers "what alpha would equalise the two gradients at this
step", so two runs at different alphas are comparable in that column and not in the other.

THE MEDIAN IS THE ESTIMATOR, matching how section 17 reports gradient share. A mean over a ratio
whose denominator can approach zero is dominated by whichever step had the smallest PAE gradient,
and that step is the least informative one, not the most.

PARITY IS A REFERENCE POINT, NOT A RECOMMENDATION. The rel coefficients sum to zero per group, so
the accumulated norm is a residual of much larger per-pose terms -- 7.6x on 3hzrs8. Setting alpha
to accumulated parity therefore drives each individual pose far harder than the headline suggests,
which is the leading suspicion for the +218% gradient-norm climb in paper_results section 24. The
number this prints is where parity IS; whether to sit at it is a separate question.
"""
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _q(v, p):
    if not v:
        return float("nan")
    w = sorted(v)
    i = min(len(w) - 1, max(0, int(round(p * (len(w) - 1)))))
    return w[i]


def rows_for(rid):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "training", "runs", rid,
                                           "grad_split.r*.csv"))):
        # RANK 0 ONLY. Every rank writes the SAME all-reduced vector norms -- the collective is
        # what makes them the step's gradient rather than one rank's share -- so reading all
        # three would triple-count identical rows and narrow the apparent spread by sqrt(3).
        if not f.endswith(".r0.csv"):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    out.append({k: (float(v) if k != "epoch" and k != "step" and k != "poses"
                                    else int(float(v)))
                                for k, v in r.items()})
                except (TypeError, ValueError):
                    continue
    return out


if len(sys.argv) < 2:
    sys.exit(__doc__)

for rid in sys.argv[1:]:
    rs = rows_for(rid)
    if not rs:
        print(f"{rid}: no grad_split.r0.csv rows "
              f"(was --grad-norm-every passed, and did a measured step complete?)")
        continue
    gp = [r["g_pae"] for r in rs]
    gr = [r["g_rank"] for r in rs]
    ra = [r["ratio"] for r in rs if r["ratio"] == r["ratio"] and r["ratio"] > 0]
    pa = [r["alpha_for_parity"] for r in rs
          if r["alpha_for_parity"] == r["alpha_for_parity"] and r["alpha_for_parity"] > 0]
    co = [r["cos"] for r in rs if r["cos"] == r["cos"]]
    al = sorted({r["alpha_rank"] for r in rs})
    print(f"\n{rid}: {len(rs)} measured step(s), epoch(s) "
          f"{min(r['epoch'] for r in rs)}..{max(r['epoch'] for r in rs)}, "
          f"alpha_rank {al if len(al) > 1 else al[0]:g}" if len(al) == 1 else
          f"\n{rid}: {len(rs)} measured step(s), alpha_rank {al}")
    print(f"  |g_pae|            median {_q(gp, 0.5):.4e}   p10 {_q(gp, 0.1):.4e}   "
          f"p90 {_q(gp, 0.9):.4e}")
    print(f"  |g_rank|           median {_q(gr, 0.5):.4e}   p10 {_q(gr, 0.1):.4e}   "
          f"p90 {_q(gr, 0.9):.4e}")
    print(f"  ratio (with alpha) median {_q(ra, 0.5):.4e}   p10 {_q(ra, 0.1):.4e}   "
          f"p90 {_q(ra, 0.9):.4e}")
    print(f"  cos(PAE, rank)     median {_q(co, 0.5):+.4f}   p10 {_q(co, 0.1):+.4f}   "
          f"p90 {_q(co, 0.9):+.4f}")
    print(f"  ALPHA FOR PARITY   median {_q(pa, 0.5):.4g}   p10 {_q(pa, 0.1):.4g}   "
          f"p90 {_q(pa, 0.9):.4g}   spread {(_q(pa, 0.9) / _q(pa, 0.1)):.1f}x")
    if al and len(al) == 1:
        _cur = al[0]
        _med = _q(pa, 0.5)
        if _med == _med and _med > 0:
            print(f"  -> this run used alpha {_cur:g}, which is {_cur / _med:.2f}x the median "
                  f"parity alpha {_med:.4g}")
    if abs(_q(co, 0.5)) < 0.1:
        print("  the two gradients are near-orthogonal at the median, so rescaling one barely "
              "turns the other")
