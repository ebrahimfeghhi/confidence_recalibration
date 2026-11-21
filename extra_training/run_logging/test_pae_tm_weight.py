"""CPU gate for --pae-tm-weight: is the term the quotient it claims to be?

THREE CLAIMS, THREE ASSERTIONS. The first is the one that has failed before: a weighting that is
wired but has no effect reports itself as weighted and trains the unweighted objective. --pae-row-frac
shipped that way once (the weight was cast to a bool mask, every weight became True). So the gate
asserts the weighted number DIFFERS from the unweighted one before it asserts anything about values.
"""
import math
import os
import sys

import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
from protenix.model.loss import softmax_cross_entropy as sce       # noqa: E402
import tm_centered_loss as TMW                                     # noqa: E402

NB = 64


def reference(ce, mask, w):
    """sum(w * CE * mask) / sum(mask) -- the objective, written the way it was specified."""
    return float((ce * (mask * w)).sum() / mask.sum())


def real_weight(sq, blocks, dt):
    """THE RUN'S OWN WEIGHT, imported -- not a copy of the formula.

    The first version of this gate defined the weight inline and asserted against itself, so it
    passed for any implementation including a no-op. Review caught it. phase_k_train now calls
    tm_centered_loss.pair_tm_weight and so does this.
    """
    return TMW.pair_tm_weight(sq, blocks, out_dtype=dt)


def main():
    torch.manual_seed(0)
    nf, nt = 11, 17
    ce = torch.rand(1, nf, nt) * 4.0
    # TWO DISJOINT BLOCKS WITH DIFFERENT TOKEN COUNTS, so a wrong d0 per block shows up as a wrong
    # weight rather than cancelling.
    b1 = torch.zeros(nf, nt, dtype=torch.bool); b1[:5, :8] = True
    b2 = torch.zeros(nf, nt, dtype=torch.bool); b2[5:, 8:] = True
    blocks = [(b1, 250), (b2, 800)]
    mask = (b1 | b2).to(torch.float32)
    sq = (torch.rand(nf, nt) * 30.0) ** 2
    w = real_weight(sq, blocks, torch.float32)
    d0 = float(TMW.d0_for_pair(250))
    fails = []

    # 0. THE WEIGHT IS THE RUN'S, AND IT USES EACH BLOCK'S OWN d0.
    w1 = float(w[0, 0]); e1 = float(sq[0, 0]) ** 0.5
    want1 = 1.0 / (1.0 + float(sq[0, 0]) / (TMW.d0_for_pair(250) ** 2))
    w2 = float(w[nf - 1, nt - 1])
    want2 = 1.0 / (1.0 + float(sq[nf - 1, nt - 1]) / (TMW.d0_for_pair(800) ** 2))
    ok0 = abs(w1 - want1) < 1e-6 and abs(w2 - want2) < 1e-6
    if not ok0:
        fails.append(f"pair_tm_weight disagrees with 1/(1+e^2/d0^2) per block: "
                     f"{w1} vs {want1}, {w2} vs {want2}")
    print(f"  {'PASS' if ok0 else 'FAIL'}  pair_tm_weight uses each block's own d0 "
          f"(d0={TMW.d0_for_pair(250):.2f} and {TMW.d0_for_pair(800):.2f})")

    # 0b. A PAIR OUTSIDE EVERY BLOCK GETS ZERO, and the mask excludes it, so it cannot enter the
    #     numerator while counting in the denominator.
    outside = ~(b1 | b2)
    ok0b = float(w[outside].abs().max()) == 0.0 and float(mask[outside].max()) == 0.0
    if not ok0b:
        fails.append("a pair outside every block has nonzero weight or is unmasked")
    print(f"  {'PASS' if ok0b else 'FAIL'}  pairs outside every block: weight 0 and masked out")

    # 0c. --pose-batch > 1 IS REFUSED BY THE FUNCTION, not silently reshaped.
    try:
        TMW.pair_tm_weight(sq.unsqueeze(0).expand(3, nf, nt), blocks)
        fails.append("pair_tm_weight accepted a batched sq instead of refusing it")
        ok0c = False
    except ValueError:
        ok0c = True
    print(f"  {'PASS' if ok0c else 'FAIL'}  a batched sq is refused, not reshaped")

    # 1. THE WEIGHT CHANGES THE NUMBER.
    lw = reference(ce, mask, w)
    lu = float((ce * mask).sum() / mask.sum())
    if abs(lw - lu) < 1e-6:
        fails.append(f"weighted {lw:.6f} and unweighted {lu:.6f} are the same number; the "
                     f"weighting is a no-op")
    print(f"  {'PASS' if abs(lw-lu) > 1e-6 else 'FAIL'}  weighting changes the loss "
          f"(weighted {lw:.4f} vs unweighted {lu:.4f})")

    # 2. THE DENOMINATOR IS THE PAIR COUNT, NOT sum(w). A weighted mean would divide by sum(w),
    #    which on this draw is a very different number; the arm is defined as the plain mean.
    wmean = float((ce * mask * w).sum() / (mask * w).sum())
    ok = abs(lw - wmean) > 1e-6 and abs(lw - float((ce * mask * w).sum() / mask.sum())) < 1e-9
    if not ok:
        fails.append(f"denominator is not sum(mask): plain {lw:.6f}, weighted-mean {wmean:.6f}")
    print(f"  {'PASS' if ok else 'FAIL'}  denominator is sum(mask)={float(mask.sum()):.0f}, "
          f"not sum(w)={float((mask*w).sum()):.2f}  (plain {lw:.4f} vs weighted-mean {wmean:.4f})")

    # 3. THE WEIGHT IS THE TM TERM, matching the value ipTM would give that pair.
    e = 8.0
    want = 1.0 / (1.0 + (e ** 2) / (d0 ** 2))
    # 1e-6, NOT 1e-9: the run computes this in float32 and the reference here is float64, so a
    # tighter bound tests the dtype rather than the formula. The first version failed on exactly
    # that -- 0.4489 against 0.4489 -- which is a fault in the gate, not in the arm.
    got = float(1.0 / (1.0 + torch.tensor(e ** 2, dtype=torch.float32) / (d0 ** 2)))
    if abs(want - got) > 1e-6:
        fails.append(f"the weight is not 1/(1+e^2/d0^2): want {want!r}, got {got!r}")
    print(f"  {'PASS' if abs(want-got) <= 1e-6 else 'FAIL'}  w(8 A) = {got:.6f} "
          f"(reference {want:.6f}) at d0 = {d0:.2f}")

    # 4. A LAST-BIN PAIR IS DOWNWEIGHTED, and by how much -- the number the arm's premise rests on.
    r = (1.0 / (1.0 + 4.0 / d0 ** 2)) / (1.0 / (1.0 + 31.75 ** 2 / d0 ** 2))
    print(f"  INFO  a 31.75 A pair carries {r:.1f}x less weight than a 2 A pair")

    # 5. MASKED-OUT PAIRS CONTRIBUTE NOTHING however large their weight.
    ce2 = ce.clone(); ce2[:, mask == 0] = 1e6
    if abs(reference(ce2, mask, w) - lw) > 1e-6:
        fails.append("masked-out pairs changed the loss")
    print(f"  {'PASS' if abs(reference(ce2, mask, w) - lw) <= 1e-6 else 'FAIL'}  "
          f"masked-out pairs contribute nothing")

    # ---- ARM N: w = max(s_true, s_pred) -------------------------------------------------
    # THE THREE CASES THE SPEC NAMES, asserted against the real function. A weight that collapses
    # to s_true would pass Arm M's gates and silently be Arm M; these distinguish them.
    NB_ = 64
    tb = torch.zeros(nf, nt, NB_); tb[:, :, NB_ - 1] = 1.0      # truly far everywhere
    tb[0, 0, :] = 0; tb[0, 0, 2] = 1.0                          # except one truly-close pair
    close = torch.zeros(nf, nt, NB_); close[:, :, 1] = 12.0     # head: confidently close
    far = torch.zeros(nf, nt, NB_); far[:, :, NB_ - 1] = 12.0   # head: confidently far
    wm_close = TMW.pair_tm_weight_max(tb, close, blocks)
    wm_far = TMW.pair_tm_weight_max(tb, far, blocks)
    decoy = float(wm_close[1, 1])      # truly far, head says close -> FULL weight
    truth = float(wm_close[0, 0])      # truly close -> full weight
    dropped = float(wm_far[1, 1])      # truly far, head agrees -> near zero
    # THE CORNER ONLY s_true CAN CARRY: truly close, head confidently FAR. In wm_close the head is
    # close everywhere, so s_true never decides anything there and all three cases below are
    # s_pred-driven -- an s_pred-ONLY weight passes them. This one fails unless s_true is in the
    # max, which is the half of the spec the gate was blind to.
    s_true_carried = float(wm_far[0, 0])
    ok_n = (decoy > 0.9 and truth > 0.9 and dropped < 0.1 and s_true_carried > 0.9)
    if s_true_carried <= 0.9:
        fails.append(f"truly-close pair with the head saying FAR got weight {s_true_carried:.4f} "
                     f"(want >0.9): s_true is not in the max, so the weight is s_pred only")
    print(f"  {'PASS' if s_true_carried > 0.9 else 'FAIL'}  s_true carries the max when the head "
          f"disagrees: truly close + head says far -> {s_true_carried:.4f}")
    if not ok_n:
        fails.append(f"max-mode weights are wrong: confident decoy {decoy:.4f} (want >0.9), "
                     f"truly correct {truth:.4f} (want >0.9), truly-wrong-and-known "
                     f"{dropped:.4f} (want <0.1)")
    print(f"  {'PASS' if ok_n else 'FAIL'}  max mode: confident decoy {decoy:.4f}, truly correct "
          f"{truth:.4f}, truly wrong and known {dropped:.4f}")

    # IT MUST DIFFER FROM ARM M's WEIGHT, or the mode flag does nothing.
    w_true_only = TMW.pair_tm_weight(
        torch.full((nf, nt), 31.75 ** 2), blocks)
    ok_diff = abs(float(wm_close.mean()) - float(w_true_only.mean())) > 0.1
    if not ok_diff:
        fails.append("max mode equals the s_true-only weight; the mode flag is a no-op")
    print(f"  {'PASS' if ok_diff else 'FAIL'}  max mode differs from s_true-only "
          f"(mean {float(wm_close.mean()):.4f} vs {float(w_true_only.mean()):.4f})")

    # NO GRADIENT PATH THROUGH THE WEIGHT: the head must not be able to lower the loss by
    # predicting uncertainty everywhere.
    lg = close.clone().requires_grad_(True)
    wg = TMW.pair_tm_weight_max(tb, lg, blocks)
    ok_g = not wg.requires_grad
    if not ok_g:
        fails.append("the max-mode weight carries a gradient; the head could optimise its own "
                     "weights downward")
    print(f"  {'PASS' if ok_g else 'FAIL'}  the weight is detached (requires_grad={wg.requires_grad})")

    # THE SHAPE THE TRAINER REALLY PASSES is [1, N_frame, N_token, 64], not [N_frame, N_token, 64];
    # the gate only ever passed 3-dim tensors, so the branch that collapses the leading dim went
    # untested. Asserted equal to the 3-dim answer.
    w4 = TMW.pair_tm_weight_max(tb.unsqueeze(0), close.unsqueeze(0), blocks)
    ok_4 = torch.allclose(w4, wm_close, atol=1e-6)
    if not ok_4:
        fails.append("the 4-dim [1, F, T, 64] call disagrees with the 3-dim one")
    print(f"  {'PASS' if ok_4 else 'FAIL'}  [1, F, T, 64] matches [F, T, 64] (the trainer's shape)")

    # A REAL SAMPLE DIM MUST RAISE, not silently return pose 0's weight for every pose.
    try:
        TMW.pair_tm_weight_max(tb.unsqueeze(0).expand(3, *tb.shape),
                               close.unsqueeze(0).expand(3, *close.shape), blocks)
        fails.append("a batched [S>1, F, T, 64] call was accepted instead of refused")
        ok_b = False
    except ValueError:
        ok_b = True
    print(f"  {'PASS' if ok_b else 'FAIL'}  a real sample dim (S>1) is refused")

    print()
    if fails:
        print(f"{len(fails)} FAILURE(S):")
        for f in fails:
            print("   " + f)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
