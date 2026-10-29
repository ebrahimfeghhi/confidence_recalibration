#!/usr/bin/env python
"""Pin the lr at the epochs launch_rowweight_cosine.sh names. CPU only, no GPU, no model.

WHY THIS EXISTS. The schedule is one of the arm's three interventions, and until this test the
only way to learn what lr epoch 50 would see was to run fifty epochs. It failed before the fix and
passes after, which is the point: the horizon came from len(flat) // world while build_epoch walks
floor(n_seeds / world) WHOLE seed-groups per structure, so one of ten seeds sat out uncounted. That
claimed 205 steps per epoch against 184 actual -- an 11% stretch that moved "1e-4 by epoch 50" to
epoch 55.7 and left lr near 8e-4 at epoch 50, roughly eight times intended.

Usage:  python -I run_logging/test_lr_schedule.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import lr_schedule as LRS  # noqa: E402

# EXACTLY launch_rowweight_cosine.sh. Read from the script where it is cheap to, so a flag edited
# there without editing this test shows up as a failure rather than as a stale assertion.
LAUNCH = os.path.join(ROOT, "shell_scripts", "launch_rowweight_cosine.sh")
SEL = os.path.join(ROOT, "training", "full_train", "structures_winnable.csv")
LR, LR_FINAL, WARMUP = 0.01, 1e-4, 20
EPOCHS, START, END, SHAPE = 60, 20, 50, "cosine"
WORLD, NSAMPLE, SEEDS = 3, 5, 10
STRUCTS_PER_STEP = 6                      # accum 85 -> nearest whole number of structures
ACCUM_LOCAL = STRUCTS_PER_STEP * NSAMPLE  # 30

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    n_struct = sum(1 for _ in csv.DictReader(open(SEL)))
    units = {f"s{i}": SEEDS for i in range(n_struct)}
    spe = LRS.steps_per_epoch(units, WORLD, NSAMPLE, ACCUM_LOCAL)
    naive = (n_struct * SEEDS // WORLD) * NSAMPLE // ACCUM_LOCAL + 1   # the shipped formula
    print(f"  {n_struct} structures x {SEEDS} seeds, world {WORLD}, nsample {NSAMPLE}, "
          f"accum_local {ACCUM_LOCAL}")
    check("steps/epoch comes from the builder's grouping, not len(flat)//world",
          spe == 184, f"builder {spe}, naive {naive}")
    check("the naive count is the 205 the review flagged", naive == 205, f"{naive}")

    def lr_at_epoch(e, steps_per_ep=spe):
        return LRS.lr_at_step(int(round(e * steps_per_ep)), LR, LR_FINAL, WARMUP,
                              steps_per_ep, EPOCHS, START, END, SHAPE)

    print("\n  lr with the CORRECT horizon (184 steps/epoch):")
    for e in (0, 1, 19.9, 20, 35, 49.9, 50, 55, 60):
        print(f"    epoch {e:>5}: {lr_at_epoch(e):.6g}")

    check("lr is held at 1e-2 right up to epoch 20",
          abs(lr_at_epoch(19.9) - LR) < 1e-12, f"{lr_at_epoch(19.9):.6g}")
    check("decay starts at epoch 20", lr_at_epoch(20.5) < LR, f"{lr_at_epoch(20.5):.6g}")
    mid = lr_at_epoch(35)
    want_mid = LR_FINAL + 0.5 * (LR - LR_FINAL)          # cosine midpoint of a 20..50 window
    check("epoch 35 is the cosine midpoint of the 20-50 window",
          abs(mid - want_mid) / want_mid < 0.02, f"{mid:.6g} vs {want_mid:.6g}")
    check("lr REACHES 1e-4 by epoch 50",
          abs(lr_at_epoch(50) - LR_FINAL) < 1e-9, f"{lr_at_epoch(50):.6g}")
    check("lr HOLDS 1e-4 after epoch 50",
          all(abs(lr_at_epoch(e) - LR_FINAL) < 1e-9 for e in (50, 55, 60, 60.5)))

    # THE REGRESSION, MODELLED AS THE MISMATCH IT WAS. The run takes `spe` steps per epoch while
    # the scheduler believes `naive`, so the step counter at real epoch e is e*spe but the decay
    # window is built from naive. Passing naive to both would be self-consistent and would hide it.
    def lr_at_real_epoch_with_bad_horizon(e):
        return LRS.lr_at_step(int(round(e * spe)), LR, LR_FINAL, WARMUP,
                              naive, EPOCHS, START, END, SHAPE)

    print(f"\n  lr the run would ACTUALLY have seen: {spe} steps/epoch taken, "
          f"{naive} assumed by the scheduler")
    for e in (20, 22.3, 35, 50, 55.7, 60):
        print(f"    epoch {e:>5}: {lr_at_real_epoch_with_bad_horizon(e):.6g}")
    bad50 = lr_at_real_epoch_with_bad_horizon(50)
    check("the bug WOULD have left lr ~8x too high at epoch 50 (regression guard)",
          bad50 > 5e-4, f"{bad50:.6g} against the intended {LR_FINAL:.6g} "
                        f"({bad50/LR_FINAL:.1f}x)")
    check("the bug WOULD have held 1e-2 past epoch 20 (flat hold stretched to ~22.3)",
          abs(lr_at_real_epoch_with_bad_horizon(22.0) - LR) < 1e-12,
          f"epoch 22.0 lr {lr_at_real_epoch_with_bad_horizon(22.0):.6g}")
    check("the bug WOULD have reached 1e-4 only near epoch 55.7",
          abs(lr_at_real_epoch_with_bad_horizon(55.7) - LR_FINAL) < 1e-7,
          f"{lr_at_real_epoch_with_bad_horizon(55.7):.6g}")

    # defaults must reproduce the historical schedule, or every prior arm is reinterpreted
    flat = [LRS.lr_at_step(s, LR, 0.0, WARMUP, spe, EPOCHS) for s in (0, 19, 20, 5000)]
    check("lr_final 0 is flat after warmup (every arm before this one)",
          flat[-1] == LR and flat[2] == LR and abs(flat[0] - LR / WARMUP) < 1e-12,
          f"{[round(x, 6) for x in flat]}")
    whole = [LRS.lr_at_step(s, LR, LR_FINAL, WARMUP, spe, EPOCHS) for s in (WARMUP, spe * EPOCHS)]
    check("decay window defaults to the whole run", abs(whole[1] - LR_FINAL) < 1e-9
          and abs(whole[0] - LR) / LR < 0.01, f"{[f'{x:.6g}' for x in whole]}")

    # the launch script still says what this test assumes
    txt = open(LAUNCH).read()
    for frag in ("--lr 0.01", "--lr-final 1e-4", "--lr-decay-start-epoch 20",
                 "--lr-decay-end-epoch 50", "--epochs 60", "--nsample 5", "--accum 85"):
        check(f"launch script still passes {frag}", frag in txt)

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
