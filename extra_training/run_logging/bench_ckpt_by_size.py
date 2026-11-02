#!/usr/bin/env python
"""Choose --ckpt-token-threshold by MEASURING peak memory and backward time, never by arithmetic.

THE QUESTION. The arm runs --blocks-per-ckpt 1 -- every one of the head's four pairformer blocks
recomputed in backward -- on EVERY unit, because the largest ones need it. gpu_backward is 70.3
min of a 143.9 min epoch (epoch 17, CUDA events, so this is device time and not a launch timer),
and roughly 30% of a checkpointed backward is recompute. The 400-token units are paying that for
nothing.

WHY A THRESHOLD IS SAFE AT ALL. The peak is a PER-POSE property: _one_pose is a function, its
frame is destroyed on return, and every large tensor is freed before the next pose allocates. So
an uncheckpointed small unit does not ADD to the peak a large unit sets -- it only has to stay
under it. The run's peak is 87.71 GB allocated on cards of 95.6 GB, and section 11 measured the
worst case at 96,116 MiB of 97,887 live, i.e. 1.8 GB of headroom. That headroom is why this is
measured and not guessed: a threshold one bucket too high is an OOM forty minutes into an epoch.

WHAT IT DOES. Two short runs over the SAME units, differing only in --blocks-per-ckpt:

  A  --blocks-per-ckpt 1   (the live setting)
  B  --blocks-per-ckpt 0   (checkpointing off for everything)

--mem-log already prints `MEM <pdb>/<seed> pose <i> tokens <n>: allocated .. peak .. reserved ..`
after each pose's backward, so B's curve of peak-vs-tokens is read directly. The recommended
threshold is the largest token count whose peak in B stays under A's observed maximum minus
--margin-gb.

AND IT CHECKS THE EQUIVALENCE RATHER THAN ASSUMING IT. Checkpointing is gradient-exact and
test_perblock_ckpt_equiv.py already gates blocks_per_ckpt=1 against off at 0.000e+00 on every
parameter -- but that test runs three poses in isolation. This compares the two runs' per-pose
loss columns over every pose they both walked. A difference here means the recompute is NOT
reproducing the forward on this workload, and no timing number would be worth having.

Usage:
  python -I run_logging/bench_ckpt_by_size.py --units 24 --cards 4 --dry-run
  python -I run_logging/bench_ckpt_by_size.py --units 24 --cards 4
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"
MEM = re.compile(r"MEM (?P<pdb>\S+)/(?P<seed>\S+) pose (?P<pose>\d+) tokens (?P<tok>-?\d+): "
                 r"allocated (?P<alloc>[\d.]+) GB\s+peak (?P<peak>[\d.]+)\s+"
                 r"reserved (?P<res>[\d.]+)")
EPOCH = re.compile(r"\[rank 0\] epoch \d+ took (?P<min>[\d.]+) min")
BUCKET = re.compile(r"(?P<k>gpu_backward|gpu_forward) (?P<m>[\d.]+)m/(?P<n>\d+)")

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="training/runs/jb5veo/resume_ep19.pt",
                help="the arm's bundle; every architecture setting is taken from its guard so "
                     "the benchmark cannot drift from the run it is for")
ap.add_argument("--units", type=int, default=24,
                help="structures to walk. They are the manifest's, so the token range is the "
                     "run's own, not a synthetic sweep.")
ap.add_argument("--cards", default="4",
                help="the card(s) made visible to the benchmark. ONE card is the default and the "
                     "right experiment: both numbers this measures -- per-pose peak memory and "
                     "per-pose device time in backward -- are rank-local, and the arm's own "
                     "gpu_backward/gpu_forward buckets are per-rank too, so a single rank is "
                     "directly comparable while adding no collective or barrier noise. Must name "
                     "cards nothing else holds; the standing limit is 3.")
ap.add_argument("--margin-gb", type=float, default=6.0,
                help="headroom demanded below the checkpointed peak. Section 11 measured 1.8 GB "
                     "of real headroom at the worst case, so the default is deliberately larger "
                     "than the slack: the threshold should not consume the margin that exists.")
ap.add_argument("--out", default="training/runs/ckpt_bench")
ap.add_argument("--master-port", type=int, default=29671,
                help="must not collide with the live arm's MASTER_PORT (29540 at the time of "
                     "writing; read it from /proc/<rank0>/environ before changing this)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

_CARDS = [c for c in a.cards.split(",") if c.strip() != ""]
if not _CARDS:
    sys.exit("STOP: --cards is empty, so the benchmark would inherit the shell's view of the box "
             "and could run on every card on it.")
if len(_CARDS) > 3:
    sys.exit(f"STOP: --cards names {len(_CARDS)} card(s). The standing limit on this box is 3.")
if len(_CARDS) > 1:
    print(f"  NOTE: {len(_CARDS)} cards requested. WORLD_SIZE is set to {len(_CARDS)} but this "
          f"launches ONE process, so ranks 1..{len(_CARDS)-1} will never join and a collective "
          f"will hang. Pass a single card unless you also launch the other ranks.")

OUT = os.path.join(ROOT, a.out)


ARGV = os.path.join(ROOT, "training/runs/ckpt_bench/arm_argv.txt")


def arm_argv():
    """The LIVE arm's own trainer argv, so the benchmark cannot drift from what it benchmarks.

    NOT RECONSTRUCTED FROM args.json. Every loss flag -- --rank-loss, --rank-form, --alpha-rank,
    --pae-only, the pose pathways -- has to be passed again on a resume, because the bundle's
    guard COMPARES them rather than supplying them, and an omitted one is a drift hard-stop, not
    a default. Rebuilding that line from parsed values means re-deriving every store_true and
    every default, and a single wrong one benchmarks a different model.
    """
    if not os.path.exists(ARGV):
        sys.exit(f"STOP: {os.path.relpath(ARGV, ROOT)} is missing. It is the arm's trainer argv, "
                 f"captured from /proc/<rank0>/cmdline while the run was up:\n"
                 f"  tr '\\0' '\\n' < /proc/<pid>/cmdline > {os.path.relpath(ARGV, ROOT)}")
    v = [x for x in open(ARGV).read().split("\n") if x != ""]
    if "--resume" not in v:
        sys.exit(f"STOP: {ARGV} has no --resume, so it is not a resume command line.")
    return v


def override(v, flag, value):
    """Replace a flag's value, or append the pair when the arm did not pass it."""
    if flag in v:
        i = v.index(flag)
        v[i + 1] = value
        return v
    return v + [flag, value]


def run(tag, bpc, lean):
    """One short run with --mem-log, the arm's argv otherwise.

    Three arms are compared, differing only in these two settings: see the ARMS table below.
    """
    work = os.path.join(a.out, tag)
    log = os.path.join(OUT, tag + ".log")
    if os.path.exists(log) and os.path.getsize(log) > 0:
        print(f"  {tag}: log exists, reusing")
        return log
    v = arm_argv()
    for flag, val in (("--resume", a.bundle), ("--blocks-per-ckpt", str(bpc)),
                      ("--limit-units", str(a.units)), ("--epochs", "1"),
                      ("--save-every", "0"), ("--out", work),
                      # NOT THE ARM'S EVAL. --train-eval-every would spend cards scoring 92
                      # held-out interfaces in the middle of a timing measurement.
                      ("--train-eval-every", "0")):
        v = override(v, flag, val)
    # A THRESHOLD IN THE ARM'S LINE WOULD MAKE BOTH ARMS OF THIS COMPARISON THE SAME RUN.
    if "--ckpt-token-threshold" in v:
        v = override(v, "--ckpt-token-threshold", "0")
    # --lean-readouts IS A store_true, so it is added or removed rather than given a value. The
    # arm's own line may or may not carry it; this makes the arm's state explicit either way.
    while "--lean-readouts" in v:
        v.remove("--lean-readouts")
    if lean:
        v.append("--lean-readouts")
    if "--mem-log" not in v:
        v.append("--mem-log")
    v = override(v, "--resume-note",
                 f"BENCHMARK ONLY (bench_ckpt_by_size.py), blocks_per_ckpt={bpc}. Writes no "
                 f"bundle the arm will resume from and takes no scheduled step.")
    if "--resume-override" not in v:
        v += ["--resume-override", "lr"]
    print(f"\n  === {tag} (blocks_per_ckpt={bpc}, lean_readouts={lean}) ===")
    print(f"    {len(v)} argv token(s) from the arm, --blocks-per-ckpt {bpc}, "
          f"--limit-units {a.units}")
    if a.dry_run:
        return None
    os.makedirs(OUT, exist_ok=True)
    # THE DEVICES AND THE RANK LAYOUT ARE SET HERE, NOT INHERITED. The arm's three ranks get
    # CUDA_VISIBLE_DEVICES/RANK/WORLD_SIZE from its launcher, never from the argv, so an argv
    # replayed out of a shell inherits NOTHING: the trainer would then see every card on the
    # box, which breaks the 3-card limit and can land on a card another user holds. A stale
    # WORLD_SIZE inherited from a parent is the same trap phase_k_train.py:1125 guards against.
    env = {**os.environ, "LAYERNORM_TYPE": "openfold",
           "CUDA_VISIBLE_DEVICES": a.cards,
           "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": str(len(a.cards.split(","))),
           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(a.master_port)}
    print(f"    CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
          f"WORLD_SIZE={env['WORLD_SIZE']} RANK=0")
    with open(log, "w") as fh:
        r = subprocess.run(v, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        sys.exit(f"STOP: {tag} exited {r.returncode}; see {os.path.relpath(log, ROOT)}")
    return log


def parse(log):
    rows, buckets, took = [], {}, None
    for line in open(log, errors="replace"):
        m = MEM.search(line)
        if m:
            rows.append((int(m["tok"]), float(m["peak"]), float(m["alloc"]), float(m["res"]),
                         m["pdb"], m["seed"], int(m["pose"])))
        e = EPOCH.search(line)
        if e:
            took = float(e["min"])
            for b in BUCKET.finditer(line):
                buckets[b["k"]] = (float(b["m"]), int(b["n"]))
    return rows, buckets, took


# THE THREE ARMS.
#
#   fat_ckpt_on    what the live arm runs today. Its peak is the CEILING the other arms must stay
#                  under, because it is the only configuration demonstrated to survive an epoch.
#   lean_ckpt_on   --lean-readouts added, nothing else. The difference against fat_ckpt_on is the
#                  headroom the strip buys, which is the number the threshold gets to spend.
#   lean_ckpt_off  the configuration a small unit would actually run in. Its peak-vs-tokens curve
#                  is what the threshold is read off.
#
# fat_ckpt_off is deliberately NOT run: nothing would be done with it. The question is how large
# a unit can skip recompute in the configuration we would ship, not in one we would not.
ARMS = (("fat_ckpt_on", 1, False), ("lean_ckpt_on", 1, True), ("lean_ckpt_off", 0, True))
logs = {tag: run(tag, bpc, lean) for tag, bpc, lean in ARMS}
if a.dry_run:
    print("\n  DRY RUN -- nothing executed")
    sys.exit(0)

F, bF, tF = parse(logs["fat_ckpt_on"])
A, bA, tA = parse(logs["lean_ckpt_on"])
B, bB, tB = parse(logs["lean_ckpt_off"])
for _t, _r in (("fat_ckpt_on", F), ("lean_ckpt_on", A), ("lean_ckpt_off", B)):
    if not _r:
        sys.exit(f"STOP: no MEM lines parsed for {_t}. --mem-log did not fire.")

peakF = max(r[1] for r in F)
peakA = max(r[1] for r in A)
print(f"\n  peak over {len(F)} pose(s), as the arm runs today: {peakF:.2f} GB")
print(f"  peak with --lean-readouts, checkpointing still on: {peakA:.2f} GB")
print(f"  HEADROOM THE STRIP BUYS: {peakF - peakA:+.2f} GB")
print(f"  budget for an uncheckpointed unit: {peakF - a.margin_gb:.2f} GB "
      f"(today's peak - {a.margin_gb:g} GB margin)\n")

# PEAK VS TOKENS, BINNED, so one outlier structure does not set the threshold on its own.
edges = [0, 400, 500, 600, 700, 800, 1000, 1250, 1600, 10 ** 9]
print(f"  {'tokens':<14}{'n':>5}{'peak OFF':>11}{'peak ON':>10}{'delta':>9}")
# AGAINST TODAY'S PEAK, not the lean one. The ceiling is the memory the run has been shown to
# survive; spending the strip's savings on a higher threshold is the entire point.
budget = peakF - a.margin_gb
ok_max = 0
for lo, hi in zip(edges, edges[1:]):
    gb = [r for r in B if lo <= r[0] < hi]
    ga = [r for r in A if lo <= r[0] < hi]
    if not gb:
        continue
    pb, pa = max(r[1] for r in gb), (max(r[1] for r in ga) if ga else float("nan"))
    fits = pb <= budget
    if fits:
        ok_max = max(ok_max, max(r[0] for r in gb))
    print(f"  {lo}-{hi if hi < 10**9 else '':<9}{len(gb):>5}{pb:>11.2f}{pa:>10.2f}"
          f"{pb - pa:>+9.2f}   {'fits' if fits else 'OVER BUDGET'}")

print(f"\n  RECOMMENDED --ckpt-token-threshold {ok_max}"
      if ok_max else "\n  RECOMMENDED: none -- no bucket fits the budget; leave checkpointing on")

if bA.get("gpu_backward") and bB.get("gpu_backward"):
    mA, nA = bA["gpu_backward"]
    mB, nB = bB["gpu_backward"]
    if nA != nB:
        print(f"\n  NOT COMPARABLE: {nA} backward(s) on, {nB} off -- the runs walked different "
              f"pose counts, so the totals are not a speedup.")
    else:
        print(f"\n  gpu_backward (CUDA events, {nA} pose(s)): on {mA:.1f}m  off {mB:.1f}m  "
              f"{(1 - mB / mA) * 100:+.1f}% when uncheckpointed")
        print(f"  gpu_forward: on {bA['gpu_forward'][0]:.1f}m  off {bB['gpu_forward'][0]:.1f}m")
        print(f"  NOTE: that is the saving on the units that QUALIFY, not on the epoch. Scale it "
              f"by the fraction of units at or below the threshold.")

# ---- THE EQUIVALENCE CHECK. Timing is worthless if the recompute is not reproducing the forward.
def losses(tag):
    p = os.path.join(ROOT, a.out, tag, "trajectory.csv")
    if not os.path.exists(p):
        return None
    out = {}
    for x in csv.DictReader(open(p)):
        try:
            out[(x["pdb"], x["unit"], x["pose"])] = float(x["loss"])
        except (ValueError, KeyError):
            pass
    return out


la, lb = losses("lean_ckpt_on"), losses("lean_ckpt_off")
if la and lb:
    keys = sorted(set(la) & set(lb))
    if not keys:
        print("\n  EQUIVALENCE: no shared poses -- NOT CHECKED")
    else:
        d = np.array([abs(la[k] - lb[k]) for k in keys])
        print(f"\n  EQUIVALENCE over {len(keys)} shared pose(s): max |loss_on - loss_off| "
              f"{d.max():.3e}, median {np.median(d):.3e}")
        if d.max() > 1e-4:
            sys.exit(f"STOP: checkpointing is NOT gradient-exact on this workload "
                     f"(max {d.max():.3e}). No timing number here is worth having.")
        print("  within tolerance: the recompute reproduces the forward.")
else:
    print("\n  EQUIVALENCE: a trajectory.csv is missing -- NOT CHECKED")
