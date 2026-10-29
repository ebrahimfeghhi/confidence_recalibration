#!/usr/bin/env python
"""Gate for phases/itm_quality_loss.py. Asserts the PROPERTIES the design rests on, not values.

WHY THESE TESTS AND NOT OTHERS. Every assertion here corresponds to a failure that has actually
happened in this project:

  - MARGIN COLLAPSE. The ranking arms lowered their loss by flattening the scores (x5frz6: within
    -seed std 0.00593 -> 0.00147, top1-top2 gap 0.00205 -> 0.00015, loss falling throughout). The
    central claim of this loss is that it CANNOT be minimised that way. test_collapse_is_penalised
    and test_collapse_is_worse_than_truth pin it.
  - INDIFFERENCE / DILUTION. A loss averaged over all poses gives a lone acceptable pose 1/P of
    the gradient, so "call everything bad" is nearly free. test_lone_good_pose_keeps_its_share
    pins the separate normalisation that fixes it, at the measured 43% figure.
  - A SILENT NO-OP. The row-weight feature once cast its weights to bool and became an exact
    no-op that still logged itself as working, so a gate that only checks the no-op direction is
    not a gate. test_asymmetry_is_real asserts the two harmless directions are STRICTLY cheaper
    than the two dangerous ones on the same magnitude of error.
  - A CLOSED-FORM GRADIENT THAT DISAGREES WITH AUTOGRAD. dL_ds exists so the group loss can be
    applied one pose at a time through the existing pass-A path; if it drifts from the loss it is
    worse than useless. test_dL_ds_matches_autograd compares them on random data.

Usage:  python -I run_logging/test_itm_quality_loss.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phases"))
import itm_quality_loss as Q  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_zero_at_truth():
    g = t([0.9, 0.5, 0.1, 0.05])
    dq = t([0.80, 0.30, 0.05, 0.02])
    L = Q.itm_quality_loss(g.clone(), g, dockq=dq)
    check("exact prediction gives exactly zero loss", float(L) == 0.0, f"L={float(L):.3e}")


def test_asymmetry_is_real_at_P_above_1():
    """The asymmetry must survive the NORMALISATION, not just the bare weight factor.

    WHY THIS EXISTS SEPARATELY. The P=1 version below floors BOTH masses at 1.0, so it never
    exercises the normalisation at all -- it measures only w vs (1-w). An adversarial review
    found 5 mutations that pass the P=1 test while changing real behaviour (dropping the down
    relu, m_dn := sum(w), floors of 0.1 and 3.0, flooring only m_dn). This group is the training
    shape: a realistic mix, so both masses are above the floor.
    """
    P = 20
    dq = torch.cat([torch.full((5,), 0.80), torch.full((P - 5,), 0.05)])
    g = torch.cat([torch.full((5,), 0.80), torch.full((P - 5,), 0.10)])
    d = 0.4
    lo = g.clone(); lo[0] -= d                     # good pose underrated  (dangerous)
    hi = g.clone(); hi[0] += d                     # good pose overrated   (harmless)
    bad_up = g.clone(); bad_up[5] += d             # bad pose overrated    (dangerous)
    bad_dn = g.clone(); bad_dn[5] -= d             # bad pose underrated   (harmless)
    v = {k: float(Q.itm_quality_loss(x, g, dockq=dq))
         for k, x in (("good underrated", lo), ("good overrated", hi),
                      ("bad overrated", bad_up), ("bad underrated", bad_dn))}
    for k, x in v.items():
        print(f"        {k:>18}: {x:.6f}")
    check("P>1: underrating a good pose beats overrating it",
          v["good underrated"] > 50 * max(v["good overrated"], 1e-12),
          f"{v['good underrated']:.6f} vs {v['good overrated']:.6f}")
    # 13.5x, NOT the 50x the first version of this assertion guessed. A bad pose at DockQ 0.05
    # still carries w = 0.0266, so underrating it is not free -- it enters the UP term at that
    # weight. The asymmetry is real but an order of magnitude weaker in this direction than in
    # the other, which is the same imbalance the down-dilution test pins. Asserting the measured
    # value rather than a round number I preferred.
    check("P>1: overrating a bad pose beats underrating it by >10x",
          v["bad overrated"] > 10 * max(v["bad underrated"], 1e-12),
          f"{v['bad overrated']:.6f} vs {v['bad underrated']:.6f} "
          f"({v['bad overrated']/max(v['bad underrated'],1e-12):.1f}x)")


def test_asymmetry_is_real():
    """The two dangerous mistakes must cost strictly more than the two harmless ones."""
    d = 0.4
    # one acceptable pose (DockQ 0.8) and one unacceptable (DockQ 0.02), scored in isolation so
    # the comparison is not confounded by the other pose's weight mass.
    good_dq, bad_dq = t([0.80]), t([0.02])
    g = t([0.9])
    low, high = t([0.9 - d]), t([0.9 - d])     # placeholders, set per case below
    cases = {}
    cases["good underrated"] = float(Q.itm_quality_loss(t([0.9 - d]), g, dockq=good_dq))
    cases["good overrated"] = float(Q.itm_quality_loss(t([0.9 + d]), g, dockq=good_dq))
    g2 = t([0.1])
    cases["bad overrated"] = float(Q.itm_quality_loss(t([0.1 + d]), g2, dockq=bad_dq))
    cases["bad underrated"] = float(Q.itm_quality_loss(t([max(0.0, 0.1 - d)]), g2, dockq=bad_dq))
    for k, v in cases.items():
        print(f"        {k:>18}: {v:.5f}")
    check("underrating a GOOD pose costs more than overrating it",
          cases["good underrated"] > 10 * max(cases["good overrated"], 1e-12),
          f"{cases['good underrated']:.5f} vs {cases['good overrated']:.5f}")
    check("overrating a BAD pose costs more than underrating it",
          cases["bad overrated"] > 10 * max(cases["bad underrated"], 1e-12),
          f"{cases['bad overrated']:.5f} vs {cases['bad underrated']:.5f}")
    del low, high


def test_collapse_is_penalised():
    """Flattening every score to one constant must NOT be a cheap escape."""
    g = t([0.90, 0.85, 0.20, 0.12, 0.08])
    dq = t([0.85, 0.60, 0.10, 0.04, 0.02])
    flat = Q.itm_quality_loss(torch.full_like(g, 0.3), g, dockq=dq)
    truth = Q.itm_quality_loss(g.clone(), g, dockq=dq)
    check("a collapsed (constant) prediction has strictly positive loss", float(flat) > 1e-4,
          f"L_flat={float(flat):.5f}")
    check("the truthful prediction beats the collapsed one", float(truth) < float(flat),
          f"{float(truth):.5f} < {float(flat):.5f}")


def test_collapse_is_worse_than_truth_even_when_mostly_bad():
    """The x5frz6 shape: one good pose among many bad ones, and the model says 'all bad'."""
    P = 50
    g = torch.cat([t([0.90]), torch.full((P - 1,), 0.08)])
    dq = torch.cat([t([0.80]), torch.full((P - 1,), 0.05)])
    all_bad = g.clone()
    all_bad[0] = 0.08                      # the only lie: the good pose called bad
    L_lie = float(Q.itm_quality_loss(all_bad, g, dockq=dq))
    L_true = float(Q.itm_quality_loss(g.clone(), g, dockq=dq))
    check("calling the lone good pose bad is expensive", L_lie > 0.05,
          f"L={L_lie:.4f} (truth {L_true:.4f})")


def test_lone_good_pose_keeps_its_share():
    """Separate normalisation: 1 acceptable of 50 must hold a large share of the UP term.

    ASSERTED THROUGH THE LOSS, not through quality_weight. The first version computed the share
    from the weight vector alone, so the 43% figure -- the only evidence for the separate
    normalisation -- was never actually a claim about itm_quality_loss, and any change to the
    normalisation passed.
    """
    P = 50
    dq = torch.cat([t([0.80]), torch.full((P - 1,), 0.05)])
    g = torch.cat([t([0.90]), torch.full((P - 1,), 0.10)])
    _, parts = Q.itm_quality_loss(g.clone(), g, dockq=dq, return_parts=True)
    share = parts["up_share_of_best"]
    check("lone good pose holds >35% of the up-term weight at width 0.05", share > 0.35,
          f"share={100*share:.0f}%  (1/P would be {100/P:.0f}%)")
    # THE KNOWN DILUTION, labelled as what it is. The earlier assertion was named "the other 49
    # hold a MINORITY of it" while printing that they hold 57% -- it asserted the opposite of the
    # number beside it, and its condition reduced to share > 1/3, already implied by the line
    # above. This records the real figure instead.
    check("the 49 unacceptable poses hold the MAJORITY of the up-term (known dilution)",
          0.5 < (1 - share) < 0.65,
          f"good {100*share:.0f}% vs 49 bad {100*(1-share):.0f}% -- the documented limit")
    # AND THE GRADIENT SHARE, which is what actually matters
    s_ = torch.zeros(P, requires_grad=True)
    Q.itm_quality_loss(s_, g, dockq=dq).backward()
    gshare = float(s_.grad[0].abs() / s_.grad.abs().sum())
    check("lone good pose holds >35% of the up-direction GRADIENT", gshare > 0.35,
          f"gradient share={100*gshare:.0f}%")


def test_down_direction_dilution_is_pinned():
    """The single-pose argmax steal, which the up-term fix does NOT address.

    A bad pose's coefficient is (1-w_p)/sum(1-w) ~ 1/P, so ONE bad pose lifting itself above the
    good one is cheap. Measured at P=50 it costs 4.7% of what calling the good pose bad costs.
    This test does not assert that is acceptable -- it PINS the ratio so a change is noticed,
    because nothing else in the gate looks at the down direction's scale.
    """
    P = 50
    dq = torch.cat([t([0.80]), torch.full((P - 1,), 0.05)])
    g = torch.cat([t([0.90]), torch.full((P - 1,), 0.10)])
    steal = g.clone(); steal[1] = 0.901            # one bad pose takes the argmax
    lie = g.clone(); lie[0] = 0.10                 # the good pose called bad
    L_steal, L_lie = (float(Q.itm_quality_loss(x, g, dockq=dq)) for x in (steal, lie))
    ratio = L_steal / L_lie
    check("the single-pose argmax steal is pinned at its measured cost",
          0.02 < ratio < 0.12,
          f"steal {L_steal:.4f} vs lie {L_lie:.4f} = {100*ratio:.1f}% (documented 4.7%)")
    # the wholesale lift IS defended against, which is the nuance worth keeping
    allup = g.clone(); allup[1:] = 0.901
    L_all = float(Q.itm_quality_loss(allup, g, dockq=dq))
    check("lifting ALL bad poses is expensive (only the single steal is cheap)", L_all > L_lie,
          f"all-lifted {L_all:.4f} > lie {L_lie:.4f}")


def test_zero_loss_optimum_can_have_the_wrong_argmax():
    """KNOWN, MEASURED LIMITATION -- pinned so a reviewer does not have to discover it.

    g_p and DockQ are different quantities and only agree at within-group Kendall tau +0.48. In 40
    of 415 real mixed groups an UNACCEPTABLE pose has higher true iTM than every acceptable one.
    Since s == g gives L == 0 exactly, the loss is perfectly satisfied by the wrong selection
    there, and strictly PREFERS it to the selection-correct answer. This asserts that the
    pathology is real, not that it is acceptable.
    """
    g = t([0.40, 0.85])            # pose 0 acceptable, pose 1 not
    dq = t([0.75, 0.05])
    L_truth = float(Q.itm_quality_loss(g.clone(), g, dockq=dq))
    L_right = float(Q.itm_quality_loss(t([0.90, 0.10]), g, dockq=dq))
    check("s == g gives zero loss even when its argmax is the UNACCEPTABLE pose",
          L_truth == 0.0 and int(g.argmax()) == 1, f"L={L_truth:.3e}, argmax={int(g.argmax())}")
    check("and the loss PREFERS that to the selection-correct prediction",
          L_truth < L_right, f"wrong-argmax {L_truth:.4f} < correct-argmax {L_right:.4f}")


def test_no_acceptable_pose_is_finite():
    g = t([0.10, 0.08, 0.05])
    dq = t([0.02, 0.01, 0.00])
    L, parts = Q.itm_quality_loss(t([0.5, 0.5, 0.5]), g, dockq=dq, return_parts=True)
    check("an interface with no acceptable pose gives a finite loss", torch.isfinite(L).item(),
          f"L={float(L):.5f}, up={float(parts['up']):.3e}, down={float(parts['down']):.5f}")
    check("  and that loss is all DOWNWARD pressure", float(parts["down"]) > 0,
          f"down={float(parts['down']):.5f}")
    # all-acceptable is the mirror case
    dq2 = t([0.90, 0.85, 0.80])
    L2, p2 = Q.itm_quality_loss(t([0.0, 0.0, 0.0]), t([0.9, 0.9, 0.9]), dockq=dq2,
                                return_parts=True)
    check("an interface where every pose is acceptable is also finite",
          torch.isfinite(L2).item() and float(p2["up"]) > 0,
          f"L={float(L2):.5f}, up={float(p2['up']):.5f}")


def test_gradient_stops_at_the_target():
    """An acceptable pose at or above g_p gets no meaningful gradient.

    EXACTLY ZERO IS THE WRONG ASSERTION and the first version of this test used it. With a soft
    weight, a pose at DockQ 0.80 still carries 1-w = 1.1e-5, so overrating it costs 6.7e-6 rather
    than 0. Demanding exact zero would demand a hard threshold at DockQ 0.23, which is what the
    sigmoid was chosen over. The property that matters is that the harmless direction is
    NEGLIGIBLE beside the dangerous one, so that is what is asserted -- with the dangerous case
    measured in the same call rather than assumed.
    """
    g = t([0.6, 0.6, 0.6])
    dq = t([0.80, 0.80, 0.80])
    s = torch.tensor([0.6, 0.9, 0.3], requires_grad=True)   # at target, above, below
    Q.itm_quality_loss(s, g, dockq=dq).backward()
    at, above, below = (abs(float(s.grad[i])) for i in range(3))
    check("no gradient on a good pose exactly at its target", at < 1e-9, f"grad={at:.3e}")
    check("overrating a good pose is negligible beside underrating one",
          above < below / 1000.0, f"above={above:.3e} vs below={below:.3e} "
                                  f"(ratio {above/max(below, 1e-30):.1e})")


def test_width_is_a_dial():
    dq = t([0.05, 0.23, 0.50])
    narrow, _ = Q.quality_weight(dq, width=0.01)
    wide, _ = Q.quality_weight(dq, width=10.0)
    check("narrow width approaches a hard step at 0.23",
          float(narrow[0]) < 0.01 and abs(float(narrow[1]) - 0.5) < 1e-6
          and float(narrow[2]) > 0.99, f"{[round(float(x), 4) for x in narrow]}")
    check("wide width sends every weight toward 0.5 (plain regression)",
          all(abs(float(x) - 0.5) < 0.03 for x in wide),
          f"{[round(float(x), 4) for x in wide]}")


def test_dL_ds_matches_autograd():
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(25):
        P = int(torch.randint(2, 40, (1,)))
        g = torch.rand(P)
        dq = torch.rand(P)
        s = torch.rand(P, requires_grad=True)
        Q.itm_quality_loss(s, g, dockq=dq).backward()
        c = Q.dL_ds(s.detach(), g, dockq=dq)
        worst = max(worst, float((s.grad - c).abs().max()))
    check("closed-form dL_ds matches autograd over 25 random groups", worst < 1e-6,
          f"worst |diff| = {worst:.3e}")


def test_label_and_weight_are_detached():
    g = torch.tensor([0.9, 0.1], requires_grad=True)
    dq = torch.tensor([0.8, 0.02], requires_grad=True)
    s = torch.tensor([0.4, 0.4], requires_grad=True)
    Q.itm_quality_loss(s, g, dockq=dq).backward()
    gs = 0.0 if g.grad is None else float(g.grad.abs().sum())
    ds = 0.0 if dq.grad is None else float(dq.grad.abs().sum())
    check("no gradient leaks into the target g", gs == 0.0, f"sum|g.grad|={gs:.3e}")
    check("no gradient leaks into the weight source dockq", ds == 0.0,
          f"sum|dq.grad|={ds:.3e}")


def test_rejects_bad_width_and_weights():
    """width <= 0 gave NaN, and w outside [0,1] gave a NEGATIVE loss. Both now raise."""
    ok = []
    for kw in ({"dockq": t([0.23, 0.5]), "width": 0.0},
               {"dockq": t([0.23, 0.5]), "width": -0.05},
               {"w": t([1.5, 0.5])},
               {"w": t([-0.5, 0.5])}):
        try:
            Q.itm_quality_loss(t([0.1, 0.2]), t([0.1, 0.2]), **kw)
            ok.append(False)
        except ValueError:
            ok.append(True)
    check("width <= 0 and w outside [0,1] are rejected", all(ok), f"{sum(ok)}/4")
    # the underflow this replaced: 1 - sigmoid(x) hits EXACTLY zero at width 0.03, DockQ 0.8
    _, w_dn = Q.quality_weight(t([0.80]), width=0.03)
    naive = float(1.0 - torch.sigmoid(t([(0.80 - 0.23) / 0.03])))
    check("the down weight does not underflow to exactly zero (regression guard)",
          float(w_dn[0]) > 0.0 and naive == 0.0,
          f"sigmoid(-z)={float(w_dn[0]):.3e}, 1-sigmoid(z)={naive:.1f}")


def test_no_share_reported_when_there_is_no_acceptable_pose():
    """up_share_of_best must be None when there is no WEIGHT, not when there is no ERROR.

    The first version of this test fed DockQ 0.01/0.00 and expected None, but at width 0.05 those
    still carry w = 0.012 and 0.010 -- a real, if small, weight -- so a share of 0.549 is the
    correct answer and `up` was 0.0 only because nothing was underrated. Zero upward ERROR and
    zero upward WEIGHT are different conditions and only the second makes the share meaningless.

    Worth recording: with real DockQ values at width 0.05 the weight is never genuinely ~0, so
    this guard is close to unreachable in production and is exercised here with w passed directly.
    """
    g = t([0.10, 0.08])
    _, parts = Q.itm_quality_loss(t([0.5, 0.5]), g, w=torch.zeros(2), return_parts=True)
    check("up_share_of_best is None when no pose carries any acceptable weight",
          parts["up_share_of_best"] is None and float(parts["up"]) == 0.0,
          f"share={parts['up_share_of_best']}, up={float(parts['up']):.3e}")
    # and with a small but real weight the share IS defined, which is the case that misled me
    _, p2 = Q.itm_quality_loss(t([0.5, 0.5]), g, dockq=t([0.01, 0.00]), return_parts=True)
    check("  but a small real weight still reports a share", p2["up_share_of_best"] is not None,
          f"share={p2['up_share_of_best']:.3f} with up={float(p2['up']):.3e}")


def test_rejects_bad_shapes():
    ok = 0
    for args, kw in (((t([0.1, 0.2]), t([0.1])), {"dockq": t([0.5, 0.5])}),
                     ((t([[0.1, 0.2]]), t([[0.1, 0.2]])), {"dockq": t([[0.5, 0.5]])}),
                     ((t([0.1]), t([0.1])), {}),
                     ((t([0.1]), t([0.1])), {"dockq": t([0.5]), "w": t([0.5])})):
        try:
            Q.itm_quality_loss(*args, **kw)
        except ValueError:
            ok += 1
    check("mismatched shapes and ambiguous weight arguments are rejected", ok == 4, f"{ok}/4")


if __name__ == "__main__":
    for fn in (test_zero_at_truth, test_asymmetry_is_real_at_P_above_1,
               test_asymmetry_is_real, test_collapse_is_penalised,
               test_down_direction_dilution_is_pinned,
               test_zero_loss_optimum_can_have_the_wrong_argmax,
               test_rejects_bad_width_and_weights,
               test_no_share_reported_when_there_is_no_acceptable_pose,
               test_collapse_is_worse_than_truth_even_when_mostly_bad,
               test_lone_good_pose_keeps_its_share, test_no_acceptable_pose_is_finite,
               test_gradient_stops_at_the_target, test_width_is_a_dial,
               test_dL_ds_matches_autograd, test_label_and_weight_are_detached,
               test_rejects_bad_shapes):
        print(f"\n  -- {fn.__name__} --")
        fn()
    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
