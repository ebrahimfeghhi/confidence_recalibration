#!/usr/bin/env python
"""Gate for --rank-form rel: the mean-centred iTM regression.

Each assertion is a property the design depends on, or a bug class that has already happened here.

  ZERO-SUM COEFFICIENTS. dL/ds_i = (2/P) r_i with sum_j r_j == 0, so sum_i dL/ds_i == 0
  identically. That is the loss's defining property -- it is why the term cannot move an
  interface's level -- and a future edit that breaks it would silently turn this into a loss that
  competes with PAE over the absolute scale.

  SHIFT INVARIANCE. Adding a constant to every prediction of an interface must not change the
  loss. If it does, the across-complex signal has leaked back in and the whole point is gone.

  COLLAPSE IS NOT AN ESCAPE. Flattening every prediction must cost MORE than the honest one.
  Every ranking arm in this repo lowered its loss by margin collapse instead (x5frz6 drove
  within-seed std 0.00593 -> 0.00147 and the top1-top2 gap to 0.00015 while its loss fell), so
  this is the property that distinguishes the design from those.

  THE LABEL MATCHES THE POOLING. g_max and g_mean both exist in the cache; regressing a
  mean-pooled score onto a max-pooled label would fit a different quantity than the one produced.

  A MISSING LABEL FILE IS FATAL. An empty label dict makes every group "short", the loop skips
  them all, and the run trains on PAE alone while the log and the saved guard both say rel -- the
  row-weight bool-cast failure in another costume, which logged itself as working for a whole run.

CPU only, no GPU, no model.

Usage:  python -I run_logging/test_rel_itm_loss.py
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import rank_loss as RKL  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def grp(sv, gv):
    return ({(0, i): v for i, v in enumerate(sv)}, {(0, i): v for i, v in enumerate(gv)})


def main():
    s, g = grp([0.5, 0.3, 0.7, 0.4, 0.55], [0.9, 0.1, 0.8, 0.2, 0.6])

    c, P, L = RKL.rel_coeffs(s, g)
    check("coefficients sum to EXACTLY zero", abs(sum(c.values())) < 1e-12,
          f"sum = {sum(c.values()):.3e}")
    check("every pose gets a coefficient", len(c) == P == len(s), f"{len(c)} of {len(s)}")

    # direction: the best pose by g must be pushed UP (negative dL/ds)
    best = max(range(len(g)), key=lambda i: g[(0, i)])
    worst = min(range(len(g)), key=lambda i: g[(0, i)])
    check("the best pose by g is pushed UP", c[(0, best)] < 0, f"c={c[(0,best)]:+.4f}")
    check("the worst pose by g is pushed DOWN", c[(0, worst)] > 0, f"c={c[(0,worst)]:+.4f}")

    # shift invariance, both sides
    s_up = {k: v + 0.37 for k, v in s.items()}
    g_up = {k: v - 0.21 for k, v in g.items()}
    check("adding a constant to every PREDICTION changes nothing",
          abs(RKL.rel_loss_value(s_up, g) - L) < 1e-12,
          f"{RKL.rel_loss_value(s_up, g):.10f} vs {L:.10f}")
    check("adding a constant to every LABEL changes nothing",
          abs(RKL.rel_loss_value(s, g_up) - L) < 1e-12,
          f"{RKL.rel_loss_value(s, g_up):.10f} vs {L:.10f}")

    # zero at the truth, and at the truth plus any constant
    check("zero loss when s == g", RKL.rel_loss_value(g, g) < 1e-15,
          f"{RKL.rel_loss_value(g, g):.3e}")
    check("zero loss when s == g + c (level is NOT supervised, by design)",
          RKL.rel_loss_value({k: v + 0.4 for k, v in g.items()}, g) < 1e-15)

    # collapse must be strictly worse
    flat = RKL.rel_loss_value({k: 0.5 for k in s}, g)
    check("a collapsed prediction costs MORE than the honest one", flat > L,
          f"flat {flat:.6f} > honest {L:.6f}")
    # ... and worse than ANY prediction that preserves the ordering
    good = RKL.rel_loss_value({k: g[k] * 0.5 for k in s}, g)
    check("  and more than a shrunk-but-correctly-ordered prediction", flat > good,
          f"flat {flat:.6f} > shrunk {good:.6f}")

    # closed form against autograd
    sv = torch.tensor([0.5, 0.3, 0.7, 0.4, 0.55], dtype=torch.float64, requires_grad=True)
    gv = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.6], dtype=torch.float64)
    r = (sv - sv.mean()) - (gv - gv.mean())
    (r ** 2).mean().backward()
    auto = sv.grad.numpy()
    cf = np.array([c[(0, i)] for i in range(5)])
    check("closed-form dL/ds matches autograd", float(np.abs(auto - cf).max()) < 1e-12,
          f"worst |diff| = {float(np.abs(auto - cf).max()):.3e}")

    # P < 2 is skipped, not crashed
    c1, P1, L1 = RKL.rel_coeffs({(0, 0): 0.5}, {(0, 0): 0.9})
    check("a single-pose group is skipped rather than crashing", c1 == {} and L1 == 0.0,
          f"P={P1}")

    # STANDARDIZE: scale-invariant ABOVE the std floor, and still penalising below it. The first
    # version of this test asserted full invariance and failed, which is how the overstated claim
    # in rel_coeffs' docstring was found -- scaling by 0.1 pushes the std under the 0.02 floor,
    # the clamp engages, and the loss moves again (0.3718 -> 0.3557).
    Ls = RKL.rel_loss_value(s, g, standardize=True)
    mild = RKL.rel_loss_value({k: v * 0.9 for k, v in s.items()}, g, standardize=True)
    hard = RKL.rel_loss_value({k: v * 0.02 for k, v in s.items()}, g, standardize=True)
    check("standardize=True is scale-invariant while the std stays above the floor",
          abs(mild - Ls) < 1e-9, f"{mild:.10f} vs {Ls:.10f}")
    check("  and still penalises a collapse that goes below the floor", hard > Ls,
          f"hard-shrunk {hard:.6f} > {Ls:.6f}")

    # ---- the label cache ----
    lab = os.path.join(R, "training", "fold", "itm_labels")
    man = os.path.join(lab, "manifest.json")
    if os.path.exists(man):
        m = json.load(open(man))
        n_npz = len(glob.glob(os.path.join(lab, "*.npz")))
        check("label cache covers every selected structure",
              m["structures"] == n_npz == 368 and m["rows_unmapped"] == 0,
              f"{m['structures']} structures, {n_npz} files, {m['rows']} rows, "
              f"{m['rows_unmapped']} unmapped")
        z = np.load(glob.glob(os.path.join(lab, "*.npz"))[0])
        check("both poolings are stored, so the label can match --rank-pool",
              "g_mean" in z and "g_max" in z, f"{sorted(z.files)}")
        check("g_max >= g_mean for every row (the max of rows bounds their mean)",
              bool((z["g_max"] >= z["g_mean"] - 1e-6).all()),
              f"{int((z['g_max'] < z['g_mean'] - 1e-6).sum())} violation(s)")
    else:
        check("label cache exists", False, f"no {man}")

    # ---- the trainer refuses the misconfigurations ----
    py = "/home/ebrahim/Protenix/.venv/bin/python"
    tr = os.path.join(R, "phases", "phase_k_train.py")
    # THE REL GUARDS SIT AFTER SEVERAL EARLIER ONES, so the invocation has to get past those
    # first or the test passes/fails for the wrong reason. It already did that twice: once on
    # LAYERNORM_TYPE and once on --require-replay.
    base = ["--require-replay", "skip", "--allow-code-drift", "--rank-loss",
            "--rank-form", "rel"]
    for args, want in ((base + ["--rel-labels", lab], "needs --rank-cross-seed"),
                       (base + ["--rank-cross-seed"], "needs --rel-labels")):
        # LAYERNORM_TYPE IS REQUIRED BY AN EARLIER GUARD. Without it the trainer exits on that
        # one first and this test would pass or fail for the wrong reason -- which it did.
        env = dict(os.environ, LAYERNORM_TYPE="openfold")
        out = subprocess.run([py, "-I", tr, *args], capture_output=True, text=True,
                             timeout=300, env=env)
        txt = out.stdout + out.stderr
        check(f"trainer refuses: {want}", want in txt,
              (txt.strip().splitlines() or ["(no output)"])[-1][:100])


def test_standardized_gradient_matches_autograd():
    """BLOCKER 4. standardize=True divides by a std that DEPENDS on s, so the closed form
    (2/P) r_i omits the derivative through it. Measured before the fix on a 5-pose group:
        closed form [-0.44738, -0.03322, 0.26788, 0.13618, 0.07654]
        true        [-3.33852, +0.52290, 1.12625, 1.36759, 0.32179]
    wrong by ~7x AND sign-flipped on pose 1, so the term pushed that pose the wrong way while
    reporting a loss that disagreed with its own gradient.
    """
    torch.manual_seed(3)
    worst = 0.0
    for _ in range(20):
        P = int(torch.randint(3, 25, (1,)))
        sv, gv = torch.rand(P, dtype=torch.float64), torch.rand(P, dtype=torch.float64)
        s_ = {(0, i): float(sv[i]) for i in range(P)}
        g_ = {(0, i): float(gv[i]) for i in range(P)}
        c, _, _ = RKL.rel_coeffs(s_, g_, standardize=True)
        t = sv.clone().requires_grad_(True)
        sc = t - t.mean()
        sc = sc / sc.std(unbiased=False).clamp_min(RKL.DEFAULT_REL_STD_FLOOR)
        gc = gv - gv.mean()
        gc = gc / gc.std(unbiased=False).clamp_min(RKL.DEFAULT_REL_STD_FLOOR)
        ((sc - gc) ** 2).mean().backward()
        au = t.grad - t.grad.mean()
        worst = max(worst, max(abs(c[(0, i)] - float(au[i])) for i in range(P)))
    check("standardize=True gradient matches autograd over 20 random groups", worst < 1e-9,
          f"worst |diff| = {worst:.3e}")


def test_pose_index_is_not_the_sample_index():
    """BLOCKER 1. The label cache is keyed by the SAMPLE index; pass A keys by the POSE index.

    Measured on the TRAINING fold: sample_index is non-identity in 3971 of 4000 label files
    (99.3%) across 120 distinct patterns, e.g. 10fj/free_500000 -> [0, 4, 3, 2, 1]. So looking a
    label up by the pose index trains a pose against another pose's target. This asserts the
    permutation is real AND that it changes the answer, so the test cannot pass vacuously on a
    fold where the ordering happens to be the identity.
    """
    labs = sorted(glob.glob(os.path.join(R, "training", "fold", "labels", "*",
                                         "free_5000*.npz")))[:600]
    if not labs:
        check("training-fold label files are present", False, "none found")
        return
    nonid = 0
    for f in labs:
        si = np.asarray(np.load(f, allow_pickle=True)["sample_index"]).reshape(-1)
        if not np.array_equal(si, np.arange(len(si))):
            nonid += 1
    check("sample_index is a NON-identity permutation on this fold", nonid > 0.5 * len(labs),
          f"{nonid} of {len(labs)} ({100*nonid/len(labs):.1f}%) non-identity")

    # and the two lookups genuinely differ: same pose, two keys, two labels
    cache = os.path.join(R, "training", "fold", "itm_labels")
    differ = tot = 0
    for f in labs[:200]:
        pdb = os.path.basename(os.path.dirname(f))
        cp = os.path.join(cache, f"{pdb}.npz")
        if not os.path.exists(cp):
            continue
        z = np.load(f, allow_pickle=True)
        si = np.asarray(z["sample_index"]).reshape(-1)
        c = np.load(cp)
        sd = int(os.path.basename(f)[5:-4])
        by = {(int(a_), int(b_), int(x), int(y)): float(g)
              for a_, b_, x, y, g in zip(c["seed"], c["sample"], c["ax"], c["ay"], c["g_mean"])}
        for i in range(len(si)):
            for (s2, sm, x, y), g in list(by.items())[:1]:
                if s2 != sd:
                    continue
            ks = [(sd, int(si[i]), x, y) for (s3, sm3, x, y) in by if s3 == sd][:1]
            kp = [(sd, i, x, y) for (s3, sm3, x, y) in by if s3 == sd][:1]
            if ks and kp and ks[0] in by and kp[0] in by:
                tot += 1
                if abs(by[ks[0]] - by[kp[0]]) > 1e-9:
                    differ += 1
    check("keying by pose index gives a DIFFERENT label than keying by sample index",
          tot > 0 and differ > 0.5 * tot,
          f"{differ} of {tot} pose(s) would get the wrong label ({100*differ/max(tot,1):.0f}%)")


def test_trainer_uses_the_sample_index_for_rel_labels():
    """STATIC GUARD for the class, not the instance: this bug class is documented in
    phase_k_train.py for --emit-iptm-rows and was still repeated for the rel target."""
    src = open(os.path.join(R, "phases", "phase_k_train.py")).read()
    check("pass A emits the true sample index", '"sample": _samp_a' in src)
    check("the rel label lookup is keyed by the sample index, not the pose index",
          "_sx = allsamp.get(" in src and "_gl.get((int(_sd), int(_sx)" in src)
    check("the permutation is gathered across ranks", "all_gather_object(_bag_s" in src)
    check("the probe scores with the SAME pooling as the backward",
          "RKL_probe.iptm_pooled(" in src and "RKL_probe.iptm(pred" not in src)
    check("rel requires EXACT label coverage per group",
          'len(dq_by) != len(s_by)' in src)
    ev = open(os.path.join(R, "phases", "phase_k_eval.py")).read()
    check("the evaluator can select by the mean-pooled score",
          '"--select-by"' in ev and 'a.select_by == "mean"' in ev)


if __name__ == "__main__":
    main()
    for fn in (test_standardized_gradient_matches_autograd,
               test_pose_index_is_not_the_sample_index,
               test_trainer_uses_the_sample_index_for_rel_labels):
        print(f"\n  -- {fn.__name__} --")
        fn()
    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
