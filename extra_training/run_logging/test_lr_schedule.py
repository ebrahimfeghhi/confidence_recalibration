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

    # ---- CROSS-SEED WITH --rank-structs-per-step: the horizon bug, round two ----
    #
    # build_epoch_cross_seed chunks the shuffled STRUCTURE list into groups of S and makes each
    # chunk one optimiser step, because a pose's coefficient needs every other pose of its
    # interface in the same step. So accum_local does not govern the step count at all. The live
    # rel arm (gu3kh7) printed "COSINE 0.01 -> 0.0001 over steps 20..12300" -- 205 steps/epoch --
    # against 122 actual, which would have reached 1e-4 near epoch 84 instead of 50. Same class as
    # the 205-against-184 error this file was written for, in the branch that fix did not cover.
    ubp368 = {f"s{i}": 10 for i in range(368)}
    spe_cs = LRS.steps_per_epoch(ubp368, world=3, nsample=5, accum_local=30,
                               cross_seed=True, structs_per_step=3)
    check("cross-seed with S=3 gives floor(368/3) = 122 steps/epoch, not 205",
          spe_cs == 122, f"got {spe_cs}")
    check("  and the SHORT final chunk is dropped, so it is a floor not a ceiling",
          LRS.steps_per_epoch({f"s{i}": 10 for i in range(10)}, world=3, nsample=5,
                            accum_local=30, cross_seed=True, structs_per_step=3) == 3,
          "10 structures at S=3 -> 3 steps, the last one short and dropped")
    check("the non-cross-seed path is UNCHANGED at 184",
          LRS.steps_per_epoch(ubp368, world=3, nsample=5, accum_local=30) == 184)
    check("structs_per_step is ignored without cross_seed",
          LRS.steps_per_epoch(ubp368, world=3, nsample=5, accum_local=30,
                            structs_per_step=3) == 184)
    for ep, want in ((20, 1e-2), (50, 1e-4), (60, 1e-4)):
        got = LRS.lr_at_step(ep * spe_cs, lr=0.01, lr_final=1e-4, warmup_steps=20,
                           steps_per_epoch=spe_cs, epochs=60, decay_start_epoch=20,
                           decay_end_epoch=50, shape="cosine")
        check(f"cross-seed lr at epoch {ep} is {want:.0e}", abs(got - want) < 1e-9,
              f"got {got:.4e}")
    # REGRESSION GUARD reproducing the live run's wrong horizon
    bad = LRS.lr_at_step(50 * spe_cs, lr=0.01, lr_final=1e-4, warmup_steps=20,
                       steps_per_epoch=205, epochs=60, decay_start_epoch=20,
                       decay_end_epoch=50, shape="cosine")
    check("the bug WOULD have left lr far above 1e-4 at epoch 50", bad > 5e-4,
          f"{bad:.3e} = {bad / 1e-4:.0f}x the intended floor")

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
