"""Read a finished run's training loss and recommend the next learning rate.

THE RULE, as the user specified it: a loss that descends SMOOTHLY means the step size can go up a
little; a SPORADIC loss means it should come down a little. This file makes "smoothly" and
"sporadic" measurable rather than a judgement, prints the numbers it decided on, and bounds the
result so a sweep cannot walk off into a range nobody chose.

WHAT IS MEASURED, at OPTIMISER-STEP granularity rather than per pose. The trajectory logs one row
per pose, but the optimiser only acts once per --accum poses, so pose-level jitter is not evidence
about the step size -- it is the within-batch variance the accumulation exists to average over.
Rows are therefore grouped by `step` first.

    descent   = mean(first fifth of steps) - mean(last fifth)
                positive means the loss fell. Fifths rather than endpoints because the first and
                last single steps are noisy and warmup makes the very first steps unrepresentative.
    roughness = sd(step-to-step differences) / |descent|
                how large the wobble is RELATIVE to the progress made. Scale-free, so it does not
                depend on the absolute size of this loss, which is ~1e-4 x the sum of four terms.
    up_frac   = fraction of consecutive steps where the loss ROSE
                a clean descent sits near 0.3-0.4 even with noise; near 0.5 is a random walk.
    nonfinite = non-finite gradient events, which the trainer already counts and drops

DECISION, in order, first match wins:
    nonfinite > 0            -> DECREASE. A gradient that overflowed is direct evidence the step is
                                too big, and it outranks any smoothness statistic.
    descent <= 0             -> DECREASE. Nothing was learned; a larger step cannot fix that.
    roughness > ROUGH or
      up_frac > UP           -> DECREASE. Progress is being made but is dominated by wobble.
    otherwise                -> INCREASE.

THE TWO THRESHOLDS ARE THE WEAK POINT AND ARE NAMED HERE RATHER THAN BURIED. ROUGH = 1.0 says "the
step-to-step wobble is as large as the total progress"; UP = 0.45 says "nearly half the steps go the
wrong way, so this is close to a random walk". Neither is derived from this project's data -- there
was no finished run to derive them from when this was written -- so the report prints both
statistics next to their thresholds, and a human reading summary.tsv can see whether the call was
marginal. The step is 2x, not the 3.16x of a half-decade, because the user asked for a small amount.
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

ROUGH = 1.0
UP = 0.45
FACTOR = 2.0
LR_MIN, LR_MAX = 1e-5, 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--lr", type=float, required=True, help="the lr this run used")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not os.path.exists(a.trajectory):
        print(json.dumps({"next_lr": a.lr / FACTOR, "verdict": "no trajectory -- decrease",
                          "reason": f"{a.trajectory} absent"}))
        return 0
    rows = list(csv.DictReader(open(a.trajectory)))
    # GROUP BY OPTIMISER STEP. Pose-level jitter is within-batch variance, not evidence about the
    # step size, and with --accum 256 there are 256 poses behind every update.
    per_step = collections.defaultdict(list)
    for r in rows:
        try:
            per_step[int(r["step"])].append(float(r["loss"]))
        except (TypeError, ValueError, KeyError):
            continue
    # THE EXPLICIT FLAG, not nan. grad_norm is nan BOTH for a pose that was not an optimiser step
    # and for a step whose gradient was non-finite, so counting nan read 175 of 200 smoke-test poses
    # as gradient failures and would have made this rule say "decrease" for every run ever.
    if rows and "nonfinite_grad" in rows[0]:
        nonfinite = sum(1 for r in rows if str(r.get("nonfinite_grad", "0")).strip() == "1")
    else:
        nonfinite = 0
        print("  NOTE: this trajectory predates the nonfinite_grad column, so gradient failures "
              "cannot be counted and are assumed absent", file=sys.stderr)
    steps = sorted(per_step)
    loss = [statistics.mean(per_step[s]) for s in steps]
    out = {"lr": a.lr, "n_steps": len(steps), "n_poses": len(rows), "nonfinite_steps": nonfinite}
    if len(steps) < 10:
        out.update(next_lr=a.lr / FACTOR, verdict="decrease",
                   reason=f"only {len(steps)} optimiser step(s) -- too few to judge, so the "
                          f"conservative direction is taken")
    else:
        k = max(2, len(loss) // 5)
        descent = statistics.mean(loss[:k]) - statistics.mean(loss[-k:])
        diffs = [loss[i] - loss[i - 1] for i in range(1, len(loss))]
        rough = (statistics.stdev(diffs) / abs(descent)) if descent else float("inf")
        up = sum(1 for d in diffs if d > 0) / len(diffs)
        out.update(descent=round(descent, 8), roughness=round(rough, 3), up_frac=round(up, 3),
                   thresholds={"roughness": ROUGH, "up_frac": UP, "factor": FACTOR})
        if nonfinite:
            out.update(next_lr=a.lr / FACTOR, verdict="decrease",
                       reason=f"{nonfinite} non-finite gradient step(s) -- direct evidence the step "
                              f"is too large, which outranks smoothness")
        elif descent <= 0:
            out.update(next_lr=a.lr / FACTOR, verdict="decrease",
                       reason=f"descent {descent:.3g} <= 0: the loss did not fall, and a larger "
                              f"step cannot fix that")
        elif rough > ROUGH or up > UP:
            out.update(next_lr=a.lr / FACTOR, verdict="decrease",
                       reason=f"sporadic: roughness {rough:.2f} vs {ROUGH}, up_frac {up:.2f} vs "
                              f"{UP} -- progress is dominated by wobble")
        else:
            out.update(next_lr=a.lr * FACTOR, verdict="increase",
                       reason=f"smooth: roughness {rough:.2f} <= {ROUGH} and up_frac {up:.2f} <= "
                              f"{UP}, descent {descent:.3g} > 0")
    out["next_lr"] = min(LR_MAX, max(LR_MIN, out["next_lr"]))
    out["bounds"] = [LR_MIN, LR_MAX]
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
