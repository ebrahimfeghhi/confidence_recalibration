"""Measure tau, the softplus temperature for the pairwise ranking loss.

WHY THIS EXISTS. The ranking loss weights a pose pair by sigmoid(-m/tau), where m is the signed
margin sign(dockq_i - dockq_j) * (iptm_i - iptm_j). tau is the margin scale over which the loss
goes from "push hard" to "satisfied", so it has to match the ipTM gaps the data actually contains.
Get it wrong in either direction and the loss sees nothing:

  tau >> typical gap  -> every pair sits at m/tau ~ 0, weight ~ 0.5, correct and inverted treated
                         identically. This is what the FIRST softplus attempt did: tau = 1 against
                         gaps of 0.008 gave 0.495 for a correct pair vs 0.505 for an inverted one.
  tau << typical gap  -> nearly every correct pair is already at 5 tau, weight 0.007, so the loss
                         ignores them and only ever sees outright inversions. Very sparse signal.

MEASURED ANSWER, and THE POPULATION IS PART OF THE ANSWER. Over training/fold/results.csv, taking
rows deduplicated by (seed, sample), forming only CROSS-SEED pairs, dropping pairs whose DockQ
differ by less than DQ_TOL, and keeping interfaces with at least --min-pairs such pairs:

    497 structures, 1,027 interfaces, 1,056,035 pairs  ->  median |gap| = 0.0305

That is rank_loss.DEFAULT_TAU. Two earlier figures in the notes came from looser populations and
should not be quoted: 0.0315 over 1,154,400 pairs was computed BEFORE dropping DockQ ties, and
0.0313 was val10 only. For comparison the 20-structure overfit30 subset gives 0.0429 and val10
gives 0.0313. USE THE TRAINING NUMBER -- tau shapes the training gradient, so setting it from val
would let the eval set influence training.

CROSS-SEED, NOT WITHIN-SEED. The earlier 0.0081 figure was within-seed pairs only. Cross-seed
pairs are 3.9x wider, and the loss compares across seeds, so within-seed numbers do not transfer.
Pairs are formed only between DIFFERENT seeds for exactly this reason.

ONE GLOBAL TAU IS DEFENSIBLE, and this script prints the evidence. Per-interface median gap spans
~1,700x, which sounds fatal but is not, because the two failure modes are not symmetric:
  wide interfaces  -- their CORRECT pairs saturate and are ignored (right, they are already
                      ordered) while their INVERTED pairs still get full weight. Handled.
  tight interfaces -- everything looks like a tie, so no discrimination. But they are tight
                      BECAUSE the head has no opinion there, and a per-interface tau would
                      amplify that noise ~80x. Contributing little is the safer behaviour.
The number that matters is the last block: the share of gradient reaching inverted pairs.

Usage:
    python -I run_logging/measure_tau.py                         # full training set
    python -I run_logging/measure_tau.py --results validation/fold/results.csv
    python -I run_logging/measure_tau.py --restrict-to training/overfit30/structures.csv
    python -I run_logging/measure_tau.py --tau 0.05              # evaluate a candidate instead
"""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict

DQ_TOL = 1e-6          # pairs closer than this in DockQ carry no ordering information
BAND = (0.2, 3.0)      # multiples of tau counted as "discriminating"


def load(paths, keep=None):
    """iface -> [(seed, iptm, dockq)]. Later files override earlier ones on (seed, sample)."""
    seen = defaultdict(dict)
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if keep is not None and r["iface"].split("_")[0].lower() not in keep:
                    continue
                try:
                    seen[r["iface"]][(r["seed"], r["sample"])] = (float(r["iptm"]), float(r["dockq"]))
                except (KeyError, ValueError):
                    continue
    return {k: [(s, ip, dq) for (s, _), (ip, dq) in v.items()] for k, v in seen.items()}


def cross_seed_pairs(poses):
    """(abs_gap, signed_margin) for every CROSS-SEED pair with a usable DockQ difference."""
    out = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            si, ipi, dqi = poses[i]
            sj, ipj, dqj = poses[j]
            if si == sj:
                continue
            dd = dqi - dqj
            if abs(dd) < DQ_TOL:
                continue
            out.append((abs(ipi - ipj), (1.0 if dd > 0 else -1.0) * (ipi - ipj)))
    return out


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def pct(sorted_vals, q):
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["training/fold/results.csv"])
    ap.add_argument("--restrict-to", default=None,
                    help="structures.csv; limits to those structures (e.g. the overfit30 subset)")
    ap.add_argument("--tau", type=float, default=None,
                    help="evaluate this tau instead of the measured median")
    ap.add_argument("--min-pairs", type=int, default=10,
                    help="skip interfaces with fewer cross-seed pairs than this")
    a = ap.parse_args()

    keep = None
    if a.restrict_to:
        keep = set()
        with open(a.restrict_to) as fh:
            for r in csv.DictReader(fh):
                keep.add((r.get("pdb") or r.get("pdb_id") or next(iter(r.values()))).lower())

    data = load(a.results, keep)
    if not data:
        raise SystemExit("no usable rows -- check --results and --restrict-to")

    gaps, per_iface = [], []
    for ifc, poses in data.items():
        pr = cross_seed_pairs(poses)
        if len(pr) < a.min_pairs:
            continue
        g = [x for x, _ in pr]
        gaps.extend(g)
        per_iface.append((st.median(g), ifc, len(poses)))
    if not gaps:
        raise SystemExit("no cross-seed pairs survived the DockQ-tie filter")

    gaps.sort()
    structs = {i.split("_")[0] for i in data}
    tau = a.tau if a.tau is not None else st.median(gaps)
    src = "GIVEN" if a.tau is not None else "MEASURED as the median"

    print(f"source: {' '.join(a.results)}" + (f"  restricted to {a.restrict_to}" if keep else ""))
    print(f"  {len(structs)} structure(s), {len(data)} interface(s), "
          f"{len(per_iface)} usable, {len(gaps):,} cross-seed pair(s)")
    print(f"\ngap distribution: median {st.median(gaps):.4f}  p75 {pct(gaps,.75):.4f}  "
          f"p90 {pct(gaps,.90):.4f}  p99 {pct(gaps,.99):.4f}  max {gaps[-1]:.4f}")
    print(f"\nTAU = {tau:.4f}   ({src})")

    pm = sorted(m for m, _, _ in per_iface)
    print(f"\nper-interface median gap: min {pm[0]:.4f}  p5 {pct(pm,.05):.4f}  "
          f"med {st.median(pm):.4f}  p95 {pct(pm,.95):.4f}  max {pm[-1]:.4f}")
    if pm[0] > 0:
        print(f"  spans {pm[-1]/pm[0]:.0f}x overall, {pct(pm,.95)/pct(pm,.05):.1f}x p5-to-p95")

    tight = sum(1 for m, _, _ in per_iface if m < BAND[0] * tau)
    wide = sum(1 for m, _, _ in per_iface if m > BAND[1] * tau)
    print(f"\nat this tau, of {len(per_iface)} interface(s):")
    print(f"  {tight:>5} ({100*tight/len(per_iface):>4.1f}%) too TIGHT  -- pairs look like ties, no discrimination")
    print(f"  {wide:>5} ({100*wide/len(per_iface):>4.1f}%) too WIDE   -- correct pairs saturate (fine); inversions still push")
    print(f"  {len(per_iface)-tight-wide:>5} "
          f"({100*(len(per_iface)-tight-wide)/len(per_iface):>4.1f}%) responsive")

    # WHERE THE GRADIENT MASS ACTUALLY GOES.
    #
    # THIS MEASURES THE PAIRWISE SCORE-SPACE GRADIENT MASS, not the parameter gradient. A pair's
    # contribution to dL/d(s_i - s_j) is, for the two forms,
    #
    #     linear:    |d_i - d_j| / W
    #     softplus:  2 |d_i - d_j| sigmoid(-m_ij / tau) / W        W = sum of the weights
    #
    # so at a tie the ratio is exactly 1 (sigmoid(0) = 1/2), which is the 2*tau property. Parameter
    # Jacobians and cross-pose cancellation are NOT included and can change the picture: an earlier
    # probe measured a 65x cancellation between per-pose and accumulated gradient norms. So this
    # says how the OBJECTIVE allocates its attention over pairs, nothing more.
    #
    # THE BASELINE IS THE DockQ-WEIGHTED LINEAR LOSS, because that is what every arm actually ran.
    # An earlier version of this script compared against the raw inversion COUNT fraction, which
    # is the right baseline only for --no-rank-weight-by-gap: with the gap as the weight the linear
    # loss does NOT give every pair equal gradient magnitude, it gives each |d_i - d_j|/W.
    #
    # NORMALISED PER INTERFACE by that interface's own W, matching group_coeffs, then summed over
    # interfaces -- so every interface contributes the same total mass under the linear form (1.0)
    # and interfaces are weighted equally rather than by pair count.
    acc = {k: {"cor": 0.0, "inv": 0.0} for k in ("lin", "sp", "lin_u", "sp_u")}
    n_cor = n_inv = n_used = 0
    for poses in data.values():
        pr = cross_seed_pairs(poses)
        if len(pr) < a.min_pairs:
            continue
        n_used += 1
        w_sum = sum(g for g, _ in pr)
        u_sum = float(len(pr))
        for gap, m in pr:
            k = "cor" if m > 0 else "inv"
            if k == "cor":
                n_cor += 1
            else:
                n_inv += 1
            s = sigmoid(-m / tau)
            acc["lin"][k] += gap / w_sum
            acc["sp"][k] += 2.0 * gap * s / w_sum
            acc["lin_u"][k] += 1.0 / u_sum
            acc["sp_u"][k] += 2.0 * s / u_sum

    n_tot = n_cor + n_inv
    print(f"\npopulation for the allocation below: {n_used} interface(s), {n_tot:,} pair(s)")
    print(f"  inverted pairs are {100*n_inv/n_tot:.1f}% BY COUNT")
    print(f"\nshare of PAIRWISE SCORE-SPACE GRADIENT MASS reaching INVERTED pairs")
    print(f"  (per-interface normalisation by that interface's W; NOT parameter gradients)")
    rows = (("lin", "DockQ-weighted LINEAR   <- the baseline that ran"),
            ("sp", f"DockQ-weighted SOFTPLUS tau={tau:.4f}"),
            ("lin_u", "unweighted LINEAR"),
            ("sp_u", f"unweighted SOFTPLUS tau={tau:.4f}"))
    share = {}
    for k, lab in rows:
        tm = acc[k]["cor"] + acc[k]["inv"]
        share[k] = 100.0 * acc[k]["inv"] / tm if tm else float("nan")
        print(f"    {lab:44s} {share[k]:5.1f}%   (mass/interface {tm/max(n_used,1):.3f})")
    print(f"\n  REDISTRIBUTION toward inverted pairs")
    print(f"    weighted   forms: {share['sp'] - share['lin']:+.1f} points")
    print(f"    unweighted forms: {share['sp_u'] - share['lin_u']:+.1f} points")


if __name__ == "__main__":
    main()
