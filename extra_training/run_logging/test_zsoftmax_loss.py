#!/usr/bin/env python
"""The z-scored softmax ranking loss, and that it is wired into the trainer. CPU, no model.

    python -I run_logging/test_zsoftmax_loss.py

WHAT IT PINS. The loss is specified as

    z_i = (s_i - m) / d        m, d = mean and std of the item's own scores, AS CONSTANTS
    p   = softmax(z)
    L   = -log( sum of p over the candidates whose quality is acceptable )

so the gradient is dL/ds_j = (1/d) * (p_j - [j acceptable] * p_j / P_good) and nothing else. The
coefficients are computed by autograd on a 1-D tensor and compared here against that closed form,
in the opposite direction from rel_coeffs: there, a hand-derived closed form was found to be wrong
by ~7x with a sign flip because it omitted a d(std)/ds term. Here the term is absent by
specification, so the closed form is exact and is the right independent check.

IT ALSO ASSERTS THE TERM CHANGES SOMETHING, which a loss test can otherwise pass while the wiring
does nothing: acceptable candidates must receive a NEGATIVE coefficient (their score is pushed up)
and unacceptable ones a positive one, and an item the head already ranks correctly must produce a
near-zero loss while one it ranks backwards produces a large one.
"""
import ast
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phases"))
import rank_loss as RKL                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def closed_form(s, acc):
    """({key: dL/ds}, L) from the specification, with m and d held constant."""
    ks = sorted(s)
    v = [float(s[k]) for k in ks]
    n = len(v)
    m = sum(v) / n
    d = (sum((x - m) ** 2 for x in v) / n) ** 0.5
    z = [(x - m) / d for x in v]
    hi = max(z)
    ez = [math.exp(x - hi) for x in z]
    Z = sum(ez)
    p = [e / Z for e in ez]
    pg = sum(p[i] for i, k in enumerate(ks) if acc[k])
    c = {k: (p[i] - (p[i] / pg if acc[k] else 0.0)) / d for i, k in enumerate(ks)}
    return c, -math.log(pg)


print("z-scored softmax loss")
rng = random.Random(11)
worst_c = worst_l = 0.0
for _ in range(40):
    n = rng.choice([2, 3, 5, 25, 75, 150])
    s = {("sd%d" % (i // 5), i % 5): rng.gauss(0.5, rng.choice([0.01, 0.12, 0.4]))
         for i in range(n)}
    ng = rng.randint(1, n - 1)
    good = set(rng.sample(sorted(s), ng))
    acc = {k: (k in good) for k in s}
    c, nn, L = RKL.zsoftmax_coeffs(s, acc)
    cf, Lf = closed_form(s, acc)
    worst_c = max(worst_c, max(abs(c[k] - cf[k]) for k in s))
    worst_l = max(worst_l, abs(L - Lf))
check("coefficients match the closed form over 40 random items", worst_c < 1e-8,
      f"max |autograd - closed form| = {worst_c:.2e}")
check("the loss matches -log(acceptable softmax mass)", worst_l < 1e-10,
      f"max |diff| = {worst_l:.2e}")

# ---- the gradient direction, which is what makes this a ranking term ------------------------
s = {i: 0.1 + 0.01 * i for i in range(20)}
acc = {i: (i < 3) for i in range(20)}                 # the three WORST-scored are the good ones
c, nn, L = RKL.zsoftmax_coeffs(s, acc)
check("acceptable candidates get a negative coefficient (their score is pushed UP)",
      all(c[k] < 0 for k in range(3)), f"c[good] = {[round(c[k], 4) for k in range(3)]}")
check("unacceptable candidates get a positive coefficient (pushed DOWN)",
      all(c[k] > 0 for k in range(3, 20)))
check("the coefficients sum to zero, so the term cannot move the item's level",
      abs(sum(c.values())) < 1e-6, f"sum = {sum(c.values()):+.2e}")

# ---- it is a selection loss: being right costs nothing, being wrong costs a lot --------------
right = {i: (0.9 if i < 3 else 0.1 + 0.001 * i) for i in range(50)}
wrong = {i: (0.1 if i < 3 else 0.9 - 0.001 * i) for i in range(50)}
lab = {i: (i < 3) for i in range(50)}
l_r = RKL.zsoftmax_loss_value(right, lab)
l_w = RKL.zsoftmax_loss_value(wrong, lab)
check("good candidates on top -> small loss; at the back -> large", l_r < 0.5 < l_w,
      f"{l_r:.4f} vs {l_w:.4f}")
# WHAT THIS FILE WILL AND WILL NOT CLAIM ABOUT THE LOSS SURFACE.
#
# Two earlier versions of this block overclaimed, each caught in external review, and the second
# correction is the reason the claims are now this narrow:
#
#   1. It first asserted a FLOOR of 0.209 at k=3 of n=50 and 0.262 at k=37 of 150, read off the
#      configuration where every acceptable candidate is tied at the top. Wrong: the numerator
#      needs only ONE acceptable candidate to carry the softmax mass, and one outlier with the
#      rest left in the pack reaches 0.0364 and 0.00052.
#   2. It then called those outlier values the minimum, and the tied value a STATIONARY POINT.
#      Both unsupported. Nudging a second acceptable candidate up from 0.100 to 0.101 lowers the
#      3/50 value 0.03642296 -> 0.03641640, so it is not a minimum; and sum|dL/ds| at the tied
#      configuration is 1.588, not zero, so it is not stationary under the implemented update.
#
# A TEST THAT ONLY EVALUATES THE LOSS CANNOT ESTABLISH EITHER PROPERTY, which is the lesson. What
# follows is confined to what evaluating the function does establish: comparisons between named
# configurations, and the norm the term would actually hand over at them. No minimum is claimed,
# and no statement is made about how the floor depends on k or n.
def ev(sc, k):
    """(loss, sum|dL/ds|) at an explicit score vector."""
    s = {i: sc[i] for i in range(len(sc))}
    c, _n, L = RKL.zsoftmax_coeffs(s, {i: (i < k) for i in range(len(sc))})
    return L, (sum(abs(v) for v in c.values()) if c else 0.0)


_tied = {(3, 50): 0.20898, (37, 150): 0.26251}
_one = {(3, 50): 0.036423, (37, 150): 0.000520}
check("one acceptable candidate carrying the mass beats the whole acceptable class tied",
      all(ev([0.9] + [0.1] * (n - 1), k)[0] < _tied[(k, n)] / 4 for k, n in _tied),
      ", ".join(f"{k}/{n}: {ev([0.9] + [0.1] * (n - 1), k)[0]:.6f} vs tied {_tied[(k, n)]:.5f}"
                for k, n in _tied))
# AND THAT VALUE IS NOT THE MINIMUM EITHER, asserted so no future reading of this file can take it
# for one. A single candidate moved by 0.001 improves it.
_b = [0.9] + [0.1] * 49
_n2 = list(_b)
_n2[1] = 0.101
check("and even THAT is not a minimum -- a 0.001 nudge on a second acceptable score improves it",
      ev(_n2, 3)[0] < ev(_b, 3)[0],
      f"{ev(_b, 3)[0]:.8f} -> {ev(_n2, 3)[0]:.8f}")
# THE PLATEAU IS REAL BUT IT IS NOT STATIONARITY. The standardised loss is invariant to how far
# the tied classes are separated, while the gradient norm falls as 1/separation. So an optimiser
# that preserves the within-class ties walks along a flat loss with a LIVE gradient, which is why
# the epoch line reports the spread and the norm and not the loss alone.
_L1, _g1 = ev([1.0] * 3 + [0.0] * 47, 3)
_L2, _g2 = ev([1000.0] * 3 + [0.0] * 47, 3)
check("the tied configuration is a symmetry-preserving PLATEAU: the loss is invariant to "
      "separation", abs(_L1 - _L2) < 1e-12, f"{_L1:.8f} at 1x, {_L2:.8f} at 1000x")
check("but the gradient there is NOT zero, so it is no stationary point of this update",
      _g1 > 1.0 and abs(_g2 - _g1 / 1000.0) < 1e-6,
      f"sum|dL/ds| {_g1:.5f} at 1x, {_g2:.5f} at 1000x -- falls as 1/separation")

# ---- invariances the z-score buys ------------------------------------------------------------
base = {i: rng.gauss(0.5, 0.1) for i in range(30)}
lab30 = {i: (i % 7 == 0) for i in range(30)}
L0 = RKL.zsoftmax_loss_value(base, lab30)
L_shift = RKL.zsoftmax_loss_value({k: v + 10.0 for k, v in base.items()}, lab30)
L_scale = RKL.zsoftmax_loss_value({k: v * 4.0 for k, v in base.items()}, lab30)
check("shifting every score leaves the loss unchanged", abs(L0 - L_shift) < 1e-9,
      f"{L0:.6f} vs {L_shift:.6f}")
check("scaling every score leaves the loss unchanged (the z-score removes it)",
      abs(L0 - L_scale) < 1e-9, f"{L0:.6f} vs {L_scale:.6f}")
c0 = RKL.zsoftmax_coeffs(base, lab30)[0]
c4 = RKL.zsoftmax_coeffs({k: v * 4.0 for k, v in base.items()}, lab30)[0]
check("but the coefficients scale as 1/d, as dz/ds demands",
      max(abs(c4[k] - c0[k] / 4.0) for k in base) < 1e-9)

# ---- the skips, each for its own stated reason ------------------------------------------------
st = {}
check("an all-acceptable item is skipped", RKL.zsoftmax_coeffs({1: .1, 2: .2},
                                                               {1: True, 2: True}, st) == ({}, 0, 0.0))
check("an all-unacceptable item is skipped", RKL.zsoftmax_coeffs({1: .1, 2: .2},
                                                                 {1: False, 2: False}, st) == ({}, 0, 0.0))
check("both are counted as one-sided, not as faults", st.get("onesided") == 2, f"{st}")
st2 = {}
check("an item with zero score spread is skipped, not floored",
      RKL.zsoftmax_coeffs({1: .3, 2: .3, 3: .3}, {1: True, 2: False, 3: False}, st2)
      == ({}, 0, 0.0))
check("and counted as flat, which is score collapse", st2.get("flat") == 1, f"{st2}")
check("a single-candidate item is skipped", RKL.zsoftmax_coeffs({1: .1}, {1: True}) == ({}, 0, 0.0))
check("a candidate with no label is dropped, not imputed",
      RKL.zsoftmax_coeffs({1: .1, 2: .9, 3: .5}, {1: True, 2: False})[1] == 2)
st3 = {}
RKL.zsoftmax_coeffs({i: 0.1 * i for i in range(10)}, {i: (i < 2) for i in range(10)}, st3)
check("a scored item reports mass, count and saturation", st3.get("items") == 1
      and 0.0 < st3["pgood"][0] < 1.0 and st3["ngood"][0] == 2 and st3["zmax"][0] > 0,
      f"mass {st3['pgood'][0]:.4f}, max|z| {st3['zmax'][0]:.2f}")

# ---- THE WIRING. A correct loss reached by no code path is the failure this half of the file
# exists for, and the repo's own history has three instances of it.
print("trainer wiring")
src = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
tree = ast.parse(src)
check("--rank-form accepts zsoftmax",
      'choices=("rel", "zsoftmax")' in src)
check("--rank-accept exists and defaults to the repo's acceptable threshold",
      '"--rank-accept", type=float, default=0.23' in src)
check("the interface-form constant names both forms",
      '_RANK_IFACE_FORMS = ("rel", "zsoftmax")' in src)
check("the coefficient dispatch calls the new loss", "RKL.zsoftmax_coeffs(s_by, _acc_by" in src)
check("rank_accept is compared, saved and has a migration default",
      src.count('"rank_accept": a.rank_accept') == 2 and '"rank_accept": 0.23,' in src,
      f'compared+saved {src.count(chr(34) + "rank_accept" + chr(34) + ": a.rank_accept")}x')
# THE PLUMBING SITES MUST GO THROUGH THE CONSTANT. Every remaining `rank_form == "rel"` has to be
# rel-specific on purpose; this pins the count so a new one cannot appear unexamined.
rel_only = src.count('a.rank_form == "rel"')
check("every shared plumbing site goes through _RANK_IFACE_FORMS",
      src.count("a.rank_form in _RANK_IFACE_FORMS") >= 10,
      f'{src.count("a.rank_form in _RANK_IFACE_FORMS")} shared, {rel_only} rel-only left')
# rel's iTM cache must stay rel-only: under an interface form _pair is the INTERFACE NAME, and the
# asym-keyed DockQ branch would look up a key that cannot exist, leave dq_by empty and skip every
# item in silence -- a ranking arm that trains nothing and says it ran.
check("zsoftmax takes its own label branch rather than rel's iTM cache",
      'elif a.rank_form == "zsoftmax":' in src and "no regression target at all" in src.lower())
check("zsoftmax refuses to run within one seed",
      "needs --rank-cross-seed" in src and "would ask the head to rank 5 candidates" in src)
check("an unwired future form still stops instead of borrowing rel's coefficients",
      "has no coefficient path" in src)
check("the epoch line names the objective that ran",
      "z-scored softmax, -log(mass on the acceptable candidates) per interface" in src)
assert isinstance(tree, ast.Module)

print()
if FAIL:
    print(f"{len(FAIL)} failure(s): {FAIL}")
    sys.exit(1)
print("all checks pass")
