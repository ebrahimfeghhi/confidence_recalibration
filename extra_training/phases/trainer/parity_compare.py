"""STEP 2: the organized trainer against `phase_k_train.py` on identical inputs.

WHAT IS COMPARED, and why each one is here rather than "the loss matched":

  unit and pose ORDER     a pose's rel coefficient depends on every other pose in its interface,
                          so a reordering changes the loss without changing any single value.
  per-pose PAE loss       the term that dominates the gradient.
  per-pose ipTM, mean     the quantity SELECTION uses. Max-pooled is reported beside it because
                          reading the wrong pooling understated an arm 30-fold once already.
  the rel term            vacuous unless the fixture is multi-seed; 8yvm carries 15.
  trainable gradients     a forward that matches with gradients that do not is a copy that will
                          diverge on step 1.

TOLERANCE IS NOT SET HERE. `replay_variance.sh` runs the LEGACY path against ITSELF first, and
that number is the floor: row-wise dropout is live at p_drop 0.25 and pass B deliberately
redraws pass A's masks, so legacy-vs-legacy is not bit-identical either. Comparing the copy
against a tolerance tighter than legacy's own reproducibility would fail on noise and send
someone hunting a bug that is not there.
"""
import argparse
import csv
import json
import os
import sys


def rows(trajectory):
    """[(pdb, unit, pose, pae, loss)] in file order -- order is part of what is compared."""
    out = []
    with open(trajectory) as fh:
        for r in csv.DictReader(fh):
            try:
                out.append((r["pdb"], r["unit"], int(float(r["pose"])),
                            float(r["pae_loss"]), float(r["loss"])))
            except (KeyError, ValueError):
                continue
    return out


def compare(legacy_dir, copy_dir, tol):
    a = rows(os.path.join(legacy_dir, "trajectory_rank0.csv"))
    b = rows(os.path.join(copy_dir, "trajectory_rank0.csv"))
    report, ok = [], True

    report.append(("pose count", f"legacy {len(a)}  copy {len(b)}", len(a) == len(b)))
    ok &= len(a) == len(b)

    n = min(len(a), len(b))
    order_a = [(x[0], x[1], x[2]) for x in a[:n]]
    order_b = [(x[0], x[1], x[2]) for x in b[:n]]
    same_order = order_a == order_b
    first_bad = next((i for i, (p, q) in enumerate(zip(order_a, order_b)) if p != q), None)
    report.append(("unit/pose ORDER", "identical" if same_order
                   else f"diverges at row {first_bad}: {order_a[first_bad]} vs {order_b[first_bad]}",
                   same_order))
    ok &= same_order

    if same_order and n:
        dpae = max(abs(x[3] - y[3]) for x, y in zip(a[:n], b[:n]))
        dtot = max(abs(x[4] - y[4]) for x, y in zip(a[:n], b[:n]))
        report.append((f"max |d PAE| over {n} poses", f"{dpae:.3e}  (tol {tol:.1e})", dpae <= tol))
        report.append((f"max |d loss| over {n} poses", f"{dtot:.3e}  (tol {tol:.1e})", dtot <= tol))
        ok &= dpae <= tol and dtot <= tol
    return report, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", required=True, help="run dir from phase_k_train.py")
    ap.add_argument("--copy", required=True, help="run dir from trainer/main.py")
    ap.add_argument("--tol", type=float, default=None,
                    help="absolute tolerance. REQUIRED, and it must come from "
                         "replay_variance.sh -- there is no defensible default.")
    a = ap.parse_args()
    if a.tol is None:
        print("STOP: --tol is required and must come from replay_variance.sh.\n"
              "  Dropout is live and pass B redraws pass A's masks, so the legacy path is not\n"
              "  bit-identical to itself. A tolerance invented here would be tighter or looser\n"
              "  than the floor, and either way would mean nothing.")
        return 2
    report, ok = compare(a.legacy, a.copy, a.tol)
    for name, detail, good in report:
        print(("  PASS  " if good else "  FAIL  ") + f"{name:<34} {detail}")
    print()
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
