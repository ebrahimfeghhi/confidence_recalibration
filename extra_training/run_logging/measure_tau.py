"""Measure tau, the softplus temperature for the pairwise ranking loss.

WHY THIS EXISTS. The ranking loss weights a pose pair by sigmoid(-m/tau), where m is the signed
margin sign(dockq_i - dockq_j) * (iptm_i - iptm_j). tau is the margin scale over which the loss
goes from "push hard" to "satisfied", so it has to match the ipTM gaps the data actually contains.
Get it wrong and the softplus stops doing the job it was chosen for:

  tau >> typical gap  -> every pair sits at m/tau ~ 0, weight ~ 0.5, so correct and inverted pairs
                         are pushed almost equally hard. THE GRADIENT DOES NOT VANISH and it does
                         not cancel -- both push toward the better-DockQ pose, so they add. What
                         is lost is SATURATION: the loss never stops pushing a pair it has already
                         ordered, and the term degenerates into the linear loss. That is the sense
                         in which the FIRST softplus attempt was inert at tau = 1 against gaps of
                         0.008 (0.495 correct vs 0.505 inverted): the NONLINEARITY was inert, not
                         the gradient. Quoting it as "no gradient" is a misreading -- at tau 0.0305
                         a 0.0137 within-seed gap still gives 0.376 against 0.624, a 1.66x ratio.
  tau << typical gap  -> nearly every correct pair is already at 5 tau, weight 0.007, so the loss
                         ignores them and only ever sees outright inversions. Very sparse signal.

MEASURED ANSWER, and THE POPULATION IS PART OF THE ANSWER. Over training/fold/results.csv, taking
rows deduplicated by (seed, sample), forming only CROSS-SEED pairs, dropping pairs whose DockQ
differ by less than DQ_TOL, and keeping interfaces with at least --min-pairs such pairs:

    497 structures, 1,027 interfaces, 1,056,035 pairs  ->  median |gap| = 0.0305

That is rank_loss.DEFAULT_TAU. Two earlier figures in the notes came from looser populations and
should not be quoted: 0.0315 over 1,154,400 pairs was computed BEFORE dropping DockQ ties, and
0.0313 was val10 only. For comparison the 30-structure overfit30 subset gives 0.0426 and val10
gives 0.0313. USE THE TRAINING NUMBER -- tau shapes the training gradient, so setting it from val
would let the eval set influence training.

CROSS-SEED, NOT WITHIN-SEED, unless --pairs says otherwise. Over the same 497 structures the
within-seed median is 0.0036, so cross-seed pairs are 8.5x wider and the two cannot share a tau.

THE 0.0081 FIGURE QUOTED IN THE NOTES AND IN rank_loss.py IS STALE -- reproduce with
`--pairs within` and it comes back 0.0036. It dates from the same looser population as the
retracted 0.0315 cross-seed figure, i.e. before DockQ ties were dropped, and the ties it kept are
exactly the zero-gap pairs that pull a within-seed median down. Both numbers here are measured
after the DQ_TOL filter.

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


def cross_seed_pairs(poses, mode="cross", straddle=0.0):
    """(abs_ipTM_gap, signed_margin, same_seed, abs_DockQ_gap) per pair of family `mode`.

    The DockQ gap travels with the pair because it is the loss's weight w_ij, and the weighted
    median needs it -- see weighted_median.

    mode="cross"  pairs from DIFFERENT seeds -- the deployed objective, and what DEFAULT_TAU
                  was measured over.
    mode="within" pairs from the SAME seed. A SEPARATE FAMILY WITH A SEPARATE TAU: these gaps are
                  several times tighter, so the cross-seed tau puts every one of them at m/tau ~ 0
                  where sigmoid cannot tell a correct pair from an inverted one. Measuring the two
                  with one script is the point -- the ratio is the thing to read.
    mode="all"    both, which is only meaningful if the loss gives each family its own tau.
    """
    if mode not in ("cross", "within", "all"):
        raise ValueError(f"unknown mode {mode!r}")
    out = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            si, ipi, dqi = poses[i]
            sj, ipj, dqj = poses[j]
            same = (si == sj)
            if (mode == "cross" and same) or (mode == "within" and not same):
                continue
            if straddle > 0 and ((dqi > straddle) == (dqj > straddle)):
                continue
            dd = dqi - dqj
            if abs(dd) < DQ_TOL:
                continue
            out.append((abs(ipi - ipj), (1.0 if dd > 0 else -1.0) * (ipi - ipj),
                        same, abs(dd)))
    return out


def weighted_median(pairs_wv):
    """Median of v under weights w, over [(w, v), ...]. The weight is the loss's OWN w_ij.

    WHY NOT A PLAIN MEDIAN. The loss weights a pair by w_ij = |dDockQ| (x L for within-seed under
    --rank-within-weight), so the gap a plain median reports is the median of the pairs COUNTED,
    not of the pairs WEIGHTED. Those differ whenever |dDockQ| correlates with |dipTM| -- which it
    does, since both grow with how different two poses are. DEFAULT_TAU = 0.0305 is an unweighted
    median and so inherits this; the weighted figure is printed beside it rather than silently
    replacing it, because every arm so far was tuned against the unweighted one.
    """
    tot = sum(w for w, _ in pairs_wv)
    if tot <= 0:
        return float("nan")
    ordered = sorted(pairs_wv, key=lambda t: t[1])
    acc = 0.0
    for w, v in ordered:
        acc += w
        if acc >= tot / 2.0:
            return v
    # ONLY REACHABLE WITH A NON-FINITE WEIGHT, since acc sums to tot otherwise. Fall back to the
    # LARGEST value, not to the last element of the unsorted input -- that returned an arbitrary
    # element. This is the lower weighted median, while the figure printed beside it is
    # statistics.median, which interpolates; immaterial at a million pairs, but it means part of
    # any small discrepancy between them is definitional rather than from the weighting.
    return ordered[-1][1]


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
                    help="skip interfaces with fewer pairs than this")
    ap.add_argument("--within-weight", type=float, default=1.0, metavar="L",
                    help="with --pairs all, mirror --rank-within-weight in the DIAGNOSTICS: the "
                         "weighted median and the gradient-mass allocation. IT DOES NOT MOVE THE "
                         "HEADLINE TAU, which is the unweighted median over pairs as counted, and "
                         "that is deliberate -- tau is set from a property of the data rather "
                         "than of the knob, so arms at different L can share one tau and differ "
                         "in one thing. Use it to see where the gradient goes at a given L, not "
                         "to pick tau for that L.")
    ap.add_argument("--pairs", choices=("cross", "within", "all"), default="cross",
                    help="which pair family to measure. 'within' is a DIFFERENT population with "
                         "a different scale, not a variant reading of the same one -- see "
                         "cross_seed_pairs. Default 'cross', which is what DEFAULT_TAU is.")
    ap.add_argument("--straddle", type=float, default=0.0, metavar="DOCKQ",
                    help="keep only pairs with exactly one pose above this DockQ. Matches "
                         "phase_k_train.py --rank-straddle; the tau for such an arm must be "
                         "measured over the pairs it actually trains, not over all of them.")
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

    gaps, wgaps, per_iface = [], [], []
    for ifc, poses in data.items():
        pr = cross_seed_pairs(poses, a.pairs, a.straddle)
        if len(pr) < a.min_pairs:
            continue
        # THE HEADLINE MEDIAN IS DELIBERATELY UNWEIGHTED, over pairs as counted. An earlier
        # comment here claimed --within-weight repeated within-seed gaps into this list; it never
        # did, and the flag's help claimed the pooled median "moves with L" when it does not.
        # Keeping it unweighted is the right behaviour -- tau is set from a property of the DATA,
        # not of the knob, which is what lets two arms at different L share one tau and differ in
        # one thing. L reaches only the weighted median printed beside it, as a diagnostic.
        g = [_gap for _gap, _m, _same, _dq in pr]
        gaps.extend(g)
        # The loss's OWN weight w_ij = |dDockQ| x (L for within-seed), so the weighted median
        # below is the gap the objective actually sees rather than the one it merely counts.
        wgaps.extend((_dq * (a.within_weight if (_same and a.pairs == "all") else 1.0), _gap)
                     for _gap, _m, _same, _dq in pr)
        per_iface.append((st.median(g), ifc, len(poses)))
    if not gaps:
        raise SystemExit(f"no {a.pairs}-seed pairs survived the DockQ-tie filter")

    gaps.sort()
    structs = {i.split("_")[0] for i in data}
    tau = a.tau if a.tau is not None else st.median(gaps)
    src = "GIVEN" if a.tau is not None else "MEASURED as the median"

    print(f"source: {' '.join(a.results)}" + (f"  restricted to {a.restrict_to}" if keep else ""))
    print(f"  {len(structs)} structure(s), {len(data)} interface(s), "
          f"{len(per_iface)} usable, {len(gaps):,} {a.pairs}-seed pair(s)")
    print(f"\n[{a.pairs.upper()}-SEED PAIRS]  gap distribution: median {st.median(gaps):.4f}  p75 {pct(gaps,.75):.4f}  "
          f"p90 {pct(gaps,.90):.4f}  p99 {pct(gaps,.99):.4f}  max {gaps[-1]:.4f}")
    _wmed = weighted_median(wgaps)
    print(f"  weighted by the loss's own w_ij = |dDockQ|"
          + (f" x {a.within_weight:g} on within-seed pairs" if a.pairs == "all"
             and a.within_weight != 1.0 else "")
          + f":  median {_wmed:.4f}")
    print(f"\nTAU = {tau:.4f}   ({src}, UNWEIGHTED median -- the statistic DEFAULT_TAU uses)")
    if _wmed == _wmed and tau > 0:
        print(f"  the weighted median is {_wmed/tau:.2f}x this; tau above the weighted median "
              "means less saturation than the unweighted figure implies.")

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
        pr = cross_seed_pairs(poses, a.pairs)
        if len(pr) < a.min_pairs:
            continue
        n_used += 1
        # |dDockQ| x L, THE LOSS'S OWN w_ij -- not the |ipTM gap|. This block read `gap`, the
        # ipTM gap, while the comment above it and group_coeffs both define the per-pair mass as
        # |d_i - d_j| / W over DockQ. The two correlate, so the result looked plausible: it put
        # the softplus share at 81.6% against a true 74.5%, and the headline "redistribution"
        # figure at +41.3 points against +32.8. Those wrong numbers were quoted in rank_loss.py,
        # this docstring and the --rank-tau help, and are corrected there.
        def _w(_g, _m, _sm, _d):
            return _d * (a.within_weight if (_sm and a.pairs == "all") else 1.0)
        w_sum = sum(_w(*t) for t in pr)
        u_sum = float(len(pr))
        if w_sum <= 0:
            continue
        for gap, m, _same, _dq in pr:
            k = "cor" if m > 0 else "inv"
            if k == "cor":
                n_cor += 1
            else:
                n_inv += 1
            s = sigmoid(-m / tau)
            wij = _w(gap, m, _same, _dq)
            acc["lin"][k] += wij / w_sum
            acc["sp"][k] += 2.0 * wij * s / w_sum
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
