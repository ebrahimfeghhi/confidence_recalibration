#!/usr/bin/env python
"""Properties of both ranking-loss forms. Run: python -I phases/test_rank_loss.py

These are the claims the design rests on, so they are asserted rather than argued.
"""
import math
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))

import rank_loss as R

DQ = [0.62, 0.41, 0.33, 0.28, 0.11]
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("\n1. THE 2*tau PREFACTOR: at a TIED pair the softplus gradient equals the linear one")
lin, _ = R.coeffs(DQ)
for tau in (0.0305, 0.001, 1.0, 50.0):
    sp, _ = R.coeffs_softplus([0.5] * len(DQ), DQ, tau=tau)
    worst = max(abs(a - b) for a, b in zip(lin, sp))
    check(f"tau={tau:<8g} max|softplus - linear|", worst < 1e-12, f"{worst:.2e}")
print(f"     linear   {[round(x,6) for x in lin]}")

print("\n2. SPREAD: THE LINEAR GRADIENT NEVER DECAYS, THE SOFTPLUS ONE DOES")
# The loss VALUE of both forms keeps improving as a correct ordering spreads -- softplus is bounded
# below by 0 and approaches it. What matters is the GRADIENT: linear keeps pushing with undiminished
# force forever, softplus stops. That is the stopping point the linear form does not have.
first_lin = first_sp = None
for sp_width in (0.008, 0.032, 0.128, 0.512):
    s = [0.5 + sp_width * (0.5 - i / (len(DQ) - 1)) for i in range(len(DQ))]
    c_lin, _ = R.coeffs(DQ)
    c_sp, _ = R.coeffs_softplus(s, DQ, tau=0.0305)
    g_lin = max(abs(x) for x in c_lin)
    g_sp = max(abs(x) for x in c_sp)
    first_lin = first_lin if first_lin is not None else g_lin
    first_sp = first_sp if first_sp is not None else g_sp
    print(f"     spread {sp_width:<6.3f}  max|dL/ds| linear {g_lin:.6f}   softplus {g_sp:.6f}")
check("linear gradient is unchanged by a 64x spread", abs(g_lin - first_lin) < 1e-12,
      f"{g_lin:.6f} vs {first_lin:.6f}")
check("softplus gradient decays >20x over the same range", g_sp < first_sp / 20.0,
      f"{first_sp:.6f} -> {g_sp:.6f}")

print("\n3. GRADIENT GOES TO INVERTED PAIRS, NOT ALREADY-CORRECT ONES")
tau = 0.0305
# two poses, DockQ 0.6 vs 0.2; vary only the ipTM ordering
for gap, label in ((-0.02, "INVERTED by 0.02"), (0.0, "tied"), (0.02, "correct by 0.02"),
                   (0.16, "correct by 0.16")):
    c, _ = R.coeffs_softplus([0.5 + gap, 0.5], [0.6, 0.2], tau=tau)
    w = abs(c[0]) / 2.0          # single pair: dL/ds = -(2/W)*w*sign*sigmoid = -2*sigmoid
    print(f"     {label:<18} sigmoid factor {w:.4f}")
c_inv, _ = R.coeffs_softplus([0.48, 0.5], [0.6, 0.2], tau=tau)
c_cor, _ = R.coeffs_softplus([0.66, 0.5], [0.6, 0.2], tau=tau)
ratio = abs(c_inv[0]) / abs(c_cor[0])
check("inverted pair outweighs a comfortably-correct one by >50x", ratio > 50, f"{ratio:.0f}x")

print("\n4. SIGN CONVENTION: the better pose is pushed UP (negative coefficient)")
c, _ = R.coeffs_softplus([0.5, 0.5], [0.9, 0.1], tau=tau)
check("c[better] < 0 and c[worse] > 0", c[0] < 0 < c[1], f"{c[0]:+.4f} / {c[1]:+.4f}")

print("\n5. DockQ TIES ARE DROPPED FROM NUMERATOR AND DENOMINATOR")
_, c_tie, n_tie = R.pair_terms_softplus([0.1, 0.2, 0.3], [0.5, 0.5, 0.5], tau=tau)
check("all-tied group yields 0 pairs", n_tie == 0, f"n_pairs={n_tie}")
_, _, n3 = R.pair_terms_softplus([0.1, 0.2, 0.3], [0.5, 0.5, 0.9], tau=tau)
check("one distinct label leaves 2 pairs, not 3", n3 == 2, f"n_pairs={n3}")

print("\n6. group_coeffs USES CROSS-SEED PAIRS ONLY")
s = {(1, 0): 0.50, (1, 1): 0.52, (2, 0): 0.48, (2, 1): 0.55}
d = {(1, 0): 0.60, (1, 1): 0.20, (2, 0): 0.70, (2, 1): 0.10}
_, n_cross, _ = R.group_coeffs(s, d, cross_seed_only=True, tau=tau)
_, n_all, _ = R.group_coeffs(s, d, cross_seed_only=False, tau=tau)
check("2 seeds x 2 samples -> 4 cross-seed pairs", n_cross == 4, f"{n_cross}")
check("including within-seed -> 6 pairs", n_all == 6, f"{n_all}")

print("\n7. group_coeffs AGREES WITH pair_terms_softplus when every pose is its own seed")
sv = {(i, 0): v for i, v in enumerate([0.50, 0.52, 0.48, 0.55, 0.51])}
dv = {(i, 0): v for i, v in enumerate(DQ)}
gc, n_g, l_g = R.group_coeffs(sv, dv, cross_seed_only=True, tau=tau)
keys = sorted(sv)
ref_l, ref_c, n_r = R.pair_terms_softplus([sv[k] for k in keys], [dv[k] for k in keys], tau=tau)
worst = max(abs(gc[k] - ref_c[i]) for i, k in enumerate(keys))
check("same n_pairs", n_g == n_r, f"{n_g} vs {n_r}")
check("same loss", abs(l_g - ref_l) < 1e-12, f"{abs(l_g-ref_l):.2e}")
check("same coefficients", worst < 1e-12, f"{worst:.2e}")

print("\n8. group_coeffs form='linear' reproduces the linear coefficients")
gl, _, _ = R.group_coeffs(sv, dv, form="linear", cross_seed_only=True)
lin5, _ = R.coeffs(DQ)
worst = max(abs(gl[k] - lin5[i]) for i, k in enumerate(keys))
check("linear path matches coeffs()", worst < 1e-12, f"{worst:.2e}")

print("\n9. A BADLY INVERTED PAIR CANNOT DOMINATE -- the weight caps at 1")
for gap in (-0.1, -1.0, -100.0):
    c, _ = R.coeffs_softplus([0.5 + gap, 0.5], [0.6, 0.2], tau=tau)
    print(f"     margin {gap:>8.1f}  sigmoid factor {abs(c[0])/2.0:.6f}")
c, _ = R.coeffs_softplus([0.5 - 100.0, 0.5], [0.6, 0.2], tau=tau)
check("factor <= 1 even at margin -100", abs(c[0]) / 2.0 <= 1.0 + 1e-9, f"{abs(c[0])/2.0:.6f}")

print("\n10. tau <= 0 is refused rather than silently dividing")
try:
    R.coeffs_softplus([0.5, 0.5], [0.6, 0.2], tau=0.0)
    check("tau=0 raises", False)
except ValueError:
    check("tau=0 raises ValueError", True)

print("\n11. group_loss AGREES WITH group_coeffs, so the fast diagnostic path cannot drift")
import random as _rnd
_r = _rnd.Random(0)
_worst = 0.0
for _trial in range(25):
    _nsd, _nsm = _r.randint(2, 5), _r.randint(2, 5)
    _s = {(sd, sm): _r.uniform(0.2, 0.9) for sd in range(_nsd) for sm in range(_nsm)}
    _d = {k: round(_r.uniform(0.0, 0.8), 3) for k in _s}
    for _form in ("softplus", "linear"):
        for _wbg in (True, False):
            _, _n1, _l1 = R.group_coeffs(_s, _d, form=_form, tau=tau,
                                         weight_by_gap=_wbg, cross_seed_only=True)
            _l2, _n2 = R.group_loss(_s, _d, form=_form, tau=tau,
                                    weight_by_gap=_wbg, cross_seed_only=True)
            assert _n1 == _n2, (_form, _wbg, _n1, _n2)
            _worst = max(_worst, abs(_l1 - _l2))
check("25 random groups x 2 forms x 2 weightings agree", _worst < 1e-9, f"max|diff| {_worst:.2e}")
# AND THE OVERFLOW GUARD: a badly inverted pair sends x to +inf in log1p(exp(x)).
_l_big, _ = R.group_loss({(0,0): 0.0, (1,0): 100.0}, {(0,0): 0.9, (1,0): 0.1},
                         form="softplus", tau=tau)
check("a margin of -100 does not overflow", _l_big == _l_big and _l_big < float("inf"),
      f"{_l_big:.4f}")


# ---------------------------------------------------------------------------------------------
# EXPECTED-DockQ FORM.  L = -sum_i softmax(s/tau)_i * dq_i
# ---------------------------------------------------------------------------------------------
print("\n--- expected-DockQ form ---")

# THE CLOSED-FORM GRADIENT AGAINST AUTOGRAD. expected_dockq computes dL/ds_i analytically while
# group_coeffs differentiates the same expression; if they disagree the trainer is stepping on a
# gradient that does not belong to the loss it reports.
_worst_g = 0.0
_r = _rnd.Random(11)
for _ in range(40):
    _nsd, _nsm = _r.randint(2, 5), _r.randint(2, 5)
    _s = {(sd, sm): _r.uniform(0.2, 0.9) for sd in range(_nsd) for sm in range(_nsm)}
    _d = {k: round(_r.uniform(0.0, 0.9), 3) for k in _s}
    for _t in (0.02, 0.05, 0.10):
        _keys = sorted(_s)
        _l_cf, _c_cf, _n_cf = R.expected_dockq([_s[k] for k in _keys], [_d[k] for k in _keys], tau=_t)
        _c_ag, _n_ag, _l_ag = R.group_coeffs(_s, _d, form="expected", tau=_t)
        assert _n_cf == _n_ag == len(_keys), (_n_cf, _n_ag, len(_keys))
        _worst_g = max(_worst_g, abs(_l_cf - _l_ag))
        for k in _keys:
            _worst_g = max(_worst_g, abs(_c_cf[_keys.index(k)] - _c_ag[k]))
check("closed form == autograd over 40 groups x 3 temperatures", _worst_g < 1e-9,
      f"max|diff| {_worst_g:.2e}")

# THE VALUE PATH AGREES WITH THE GRADIENT PATH, as for the pairwise forms.
_worst_v = 0.0
for _ in range(25):
    _s = {(sd, sm): _r.uniform(0.2, 0.9) for sd in range(3) for sm in range(4)}
    _d = {k: round(_r.uniform(0.0, 0.8), 3) for k in _s}
    for _t in (0.02, 0.05, 0.10):
        _, _n1, _l1 = R.group_coeffs(_s, _d, form="expected", tau=_t)
        _l2, _n2 = R.group_loss(_s, _d, form="expected", tau=_t)
        assert _n1 == _n2, (_n1, _n2)
        _worst_v = max(_worst_v, abs(_l1 - _l2))
check("group_loss == group_coeffs for expected", _worst_v < 1e-12, f"max|diff| {_worst_v:.2e}")

# IT IS -EXPECTED DockQ, so the loss is bounded by the group's own labels and a model that puts
# all mass on the best pose scores exactly -max(dq).
_s_win = {(0, 0): 10.0, (0, 1): 0.0, (1, 0): 0.0}
_d_win = {(0, 0): 0.80, (0, 1): 0.10, (1, 0): 0.20}
_l_win, _ = R.group_loss(_s_win, _d_win, form="expected", tau=0.05)
check("mass on the best pose gives -max(dq)", abs(_l_win - (-0.80)) < 1e-6, f"{_l_win:.6f}")
_s_lose = {(0, 0): 0.0, (0, 1): 10.0, (1, 0): 0.0}
_l_lose, _ = R.group_loss(_s_lose, _d_win, form="expected", tau=0.05)
check("mass on the worst pose gives -min(dq)", abs(_l_lose - (-0.10)) < 1e-6, f"{_l_lose:.6f}")

# LOWER TAU IS SHARPER. With the leader held fixed, a colder softmax must put more mass on it,
# so the loss moves toward -max(dq) monotonically as tau falls.
_s_t = {(0, 0): 0.60, (0, 1): 0.50, (1, 0): 0.40}
_d_t = {(0, 0): 0.70, (0, 1): 0.30, (1, 0): 0.10}
_ls = [R.group_loss(_s_t, _d_t, form="expected", tau=t)[0] for t in (0.10, 0.05, 0.02)]
check("colder tau -> closer to -max(dq)", _ls[0] > _ls[1] > _ls[2],
      "  ".join(f"tau {t:g}: {l:+.5f}" for t, l in zip((0.10, 0.05, 0.02), _ls)))

# A POSE WITH NO CHANCE OF SELECTION GETS NO GRADIENT -- the property the pairwise forms lack and
# the entire reason this form exists. The pairwise softplus gives the same hopeless pose a
# gradient of the same order as a contender's.
_s_h = {(0, 0): 0.90, (0, 1): 0.89, (1, 0): 0.10}
_d_h = {(0, 0): 0.20, (0, 1): 0.70, (1, 0): 0.60}
_c_e, _, _ = R.group_coeffs(_s_h, _d_h, form="expected", tau=0.05)
_c_p, _, _ = R.group_coeffs(_s_h, _d_h, form="softplus", tau=0.0305)
_ratio_e = abs(_c_e[(1, 0)]) / max(abs(_c_e[(0, 1)]), 1e-30)
_ratio_p = abs(_c_p[(1, 0)]) / max(abs(_c_p[(0, 1)]), 1e-30)
check("expected starves the hopeless pose where softplus does not",
      _ratio_e < 1e-6 and _ratio_p > 0.1,
      f"hopeless/contender gradient: expected {_ratio_e:.2e}, softplus {_ratio_p:.3f}")

# NO OVERFLOW AT A COLD TEMPERATURE. z = 0.9/0.02 = 45 and exp(45) is finite, but a wider score
# range at tau 0.001 would not be -- the shift by max(z) is what makes this safe.
_l_cold, _c_cold, _ = R.expected_dockq([0.0, 1.0], [0.1, 0.9], tau=0.001)
check("tau 0.001 does not overflow", _l_cold == _l_cold and abs(_l_cold + 0.9) < 1e-9,
      f"{_l_cold:.6f}")

# TAU MUST BE POSITIVE for this form too.
try:
    R.expected_dockq([0.1, 0.2], [0.3, 0.4], tau=0.0)
    check("tau <= 0 raises", False, "no exception")
except ValueError:
    check("tau <= 0 raises", True, "ValueError")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURE(S): {FAILS}"))
sys.exit(1 if FAILS else 0)
