"""Compare two runs that differ ONLY in --pose-batch. They must train the same objective.

bf16 autocast means exact equality is not expected; what is expected is agreement to a tolerance
far tighter than any real change to the objective would produce. A batched reduction applied over
the wrong axis, or a label row misaligned with its pose, moves the loss by percent, not by 1e-4.
"""
import csv
import math
import sys


def rows(d):
    try:
        return list(csv.DictReader(open(f"training/runs/{d}/trajectory_rank0.csv")))
    except FileNotFoundError:
        return []


def per_unit(rs):
    out = {}
    for r in rs:
        k = (r["pdb"], r["unit"])
        try:
            out.setdefault(k, []).append(float(r["pae_loss"]))
        except (TypeError, ValueError):
            pass
    return {k: sum(v) / len(v) for k, v in out.items() if v}


def grads(rs):
    g = []
    for r in rs:
        v = r.get("grad_norm")
        if v in (None, "", "nan"):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            g.append(f)
    return g


def main():
    A, B = rows("pb_equiv_1"), rows("pb_equiv_5")
    if not A or not B:
        print(f"  REFUSING: no trajectory (A {len(A)} rows, B {len(B)} rows). See each run's log.")
        return 1
    pa, pb = per_unit(A), per_unit(B)
    common = sorted(set(pa) & set(pb))
    print(f"  rows: A {len(A)}  B {len(B)}")
    print(f"  units compared: {len(common)}  (A has {len(pa)}, B has {len(pb)})")
    if not common:
        print("  REFUSING: the two runs share no unit, so nothing was compared.")
        return 1
    d = [abs(pa[k] - pb[k]) for k in common]
    rel = [abs(pa[k] - pb[k]) / max(1e-9, abs(pa[k])) for k in common]
    worst = max(common, key=lambda k: abs(pa[k] - pb[k]))
    print(f"  per-unit mean PAE loss |diff|: max {max(d):.3e}  mean {sum(d)/len(d):.3e}")
    print(f"                     relative  : max {max(rel):.3e}")
    print(f"  worst unit {worst[0]}/{worst[1]}: A {pa[worst]:.6f}  B {pb[worst]:.6f}")
    ga, gb = grads(A), grads(B)
    print(f"  grad_norm A: {[round(x, 5) for x in ga[:5]]}")
    print(f"  grad_norm B: {[round(x, 5) for x in gb[:5]]}")

    # THE STRONGEST CHECK AVAILABLE, AND IT WAS SITTING UNUSED. Both runs save head_final.pt.
    # With a fixed --seed and identical data the two paths must reach the SAME weights, which is
    # a statement about the backward -- where the sample-axis scaling and the accumulator
    # counting live. A per-unit mean of one logged column cannot see either of them: run A's
    # mean-of-five equals run B's single meaned row by construction.
    wmax = None
    try:
        import torch
        wa = torch.load("training/runs/pb_equiv_1/head_final.pt", map_location="cpu")
        wb = torch.load("training/runs/pb_equiv_5/head_final.pt", map_location="cpu")
        wa = wa.get("model", wa) if isinstance(wa, dict) else wa
        wb = wb.get("model", wb) if isinstance(wb, dict) else wb
        keys = sorted(set(wa) & set(wb))
        if keys:
            wmax = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in keys)
            print(f"  final weights: {len(keys)} tensor(s) compared, max |diff| {wmax:.3e}")
        else:
            print("  final weights: no common tensor -- NOT compared")
    except Exception as e:                                          # noqa: BLE001
        print(f"  final weights: could not compare ({type(e).__name__}: {e})")

    ok_loss = max(rel) < 2e-3
    ok_grad = bool(ga) and len(ga) == len(gb) and all(
        abs(x - y) / max(1e-9, abs(x)) < 5e-3 for x, y in zip(ga, gb))
    # AGAINST max, NOT min. With min, a run that died after 3 of 27 units passes trivially:
    # min(3, 27) * 0.9 = 2.7 and 3 >= 2.7. Raised in adversarial review.
    ok_cov = len(common) >= max(len(pa), len(pb)) * 0.95
    print()
    print(f"  {'PASS' if ok_loss else 'FAIL'}  per-unit loss agrees "
          f"(tolerance 2e-3 relative; bf16 autocast makes exact equality impossible)")
    print(f"  {'PASS' if ok_grad else 'FAIL'}  gradient norms agree "
          f"({len(ga)} vs {len(gb)} optimiser step(s))")
    print(f"  {'PASS' if ok_cov else 'FAIL'}  the two runs walked the same units")
    ok_w = (wmax is not None) and wmax < 1e-4
    print(f"  {'PASS' if ok_w else 'FAIL'}  final weights agree "
          f"({'max |diff| %.3e' % wmax if wmax is not None else 'NOT COMPARED -- treated as a failure'})")
    return 0 if (ok_loss and ok_grad and ok_cov and ok_w) else 1


if __name__ == "__main__":
    raise SystemExit(main())
