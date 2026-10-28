#!/usr/bin/env python
"""Score fawgtu's saved bundles: TRAIN metrics, then VALFULL metrics, then VAL PAE.

ORDER IS THE USER'S: train first, then val, then val PAE. Each stage is independent and skippable,
and every stage is idempotent -- an epoch whose output already exists is left alone -- so this can
be stopped and restarted without redoing work.

WHY THIS RUN NEEDS A PURPOSE-BUILT SCRIPT. fawgtu trained with no evaluation of any kind, which is
what made its epochs fast, and that has two consequences the usual tooling does not expect:

  1. IT WROTE ONLY RESUME BUNDLES, no head_ep*.pt. phase_k_eval's --ckpt does torch.load and then
     compares keys against the model's confidence_head parameters, so a bundle -- {head, opt,
     epoch, step, guard} -- fails that guard instead of loading. run_logging/extract_heads.py
     unwraps them first; verified against tv6hue's known-good head checkpoint, identical 254-key
     set and matching shapes.

  2. recover_val_pae.py CANNOT BE USED for stage 3, which is worth stating because it is the
     obvious tool and it would fail confusingly. It rebuilds the held-out command from the run's
     own args.json, and fawgtu's val_loss_fold, val_loss_pool, val_loss_selection,
     val_loss_preflight, val_loss_seeds and val_loss_prep are all EMPTY STRINGS -- the flags were
     never passed. It would reconstruct a val pass with no val population. So stage 3 instead
     invokes phase_k_train --loss-only directly, with the held-out flags supplied explicitly,
     copying val_loss_checkpoint()'s own command so the number stays comparable to every other
     arm's val PAE (shipped 2.9756).

BUDGETS DIFFER BETWEEN STAGES AND ARE NOT COMPARABLE ACROSS THEM. Stage 1 reports train AND
valfull at budget 10, because the training fold has only 10 seeds and a paired comparison needs
the same depth on both sides. Stage 2 reports valfull at budget 30, which is what every figure in
paper_results.md uses. A budget-10 valfull number and a budget-30 valfull number are different
measurements of the same population; do not put them in one column.

Usage:
  python -I run_logging/eval_fawgtu_bundles.py --epochs 5,10,15,20,25,30,35,40
  python -I run_logging/eval_fawgtu_bundles.py --epochs 40 --stages train
  python -I run_logging/eval_fawgtu_bundles.py --epochs 40 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="fawgtu")
ap.add_argument("--epochs", required=True, help="comma list of epochs to score")
ap.add_argument("--stages", default="train,val,valpae",
                help="comma list from train,val,valpae -- run in THIS order regardless")
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--val-budget", type=int, default=30, help="stage 2 budget on valfull")
ap.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
ap.add_argument("--allow-busy", action="store_true",
                help="proceed even if cards_busy.py reports occupancy. Off by default: landing a "
                     "three-card evaluation on top of a training run is how both get slower and "
                     "one of them OOMs.")
a = ap.parse_args()

RUN_D = os.path.join(ROOT, "training", "runs", a.run)
EPOCHS = [int(x) for x in a.epochs.split(",") if x.strip()]
STAGES = [s.strip() for s in a.stages.split(",") if s.strip()]
for s in STAGES:
    if s not in ("train", "val", "valpae"):
        raise SystemExit(f"STOP: unknown stage {s!r}; expected train, val or valpae")

# THE ARCHITECTURE FLAGS COME FROM THE RUN'S OWN args.json, never retyped. A head checkpoint whose
# pose pathways the evaluator does not construct loads with those tensors left at zero -- the
# pathway silently absent -- and phase_k_eval's own guard is what catches it. Reading them here
# means the guard never has to.
ARGS = json.load(open(os.path.join(RUN_D, "args.json")))
POSE = []
if ARGS.get("pose_perblock"):
    POSE.append("--pose-perblock")
    for k, f in (("pose_perblock_tied", "--pose-perblock-tied"),
                 ("pose_perblock_share", "--pose-perblock-share"),
                 ("pose_perblock_copy", "--pose-perblock-copy")):
        if ARGS.get(k):
            POSE.append(f)
if ARGS.get("frame_direction"):
    POSE.append("--frame-direction")

VALFULL = {"fold": "validation/fold",
           "pool": "training/valfull/pool.csv",
           "selection": "training/valfull/structures.csv",
           "preflight": "training/valfull/alignment_preflight.json",
           "prep": "validation/prep",
           "seeds": "400000-400029",
           "seed_base": 400000}


def busy():
    r = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "cards_busy.py")],
                       capture_output=True, text=True, cwd=ROOT)
    return (r.stdout.strip().splitlines() or ["?"])[-1]


def head_for(ep):
    """The flat head checkpoint for an epoch, extracting it from the bundle if needed."""
    import glob
    hit = glob.glob(os.path.join(RUN_D, f"head_ep{ep}_step*.pt")) \
        + glob.glob(os.path.join(RUN_D, f"head_ep{ep}.pt"))
    if hit:
        return sorted(hit)[0]
    bundle = os.path.join(RUN_D, f"resume_ep{ep}.pt")
    if not os.path.exists(bundle):
        return None
    print(f"    extracting head from resume_ep{ep}.pt")
    if not a.dry_run:
        subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "extract_heads.py"),
                        a.run, "--epochs", str(ep)], cwd=ROOT, check=True)
    return os.path.join(RUN_D, f"head_ep{ep}.pt")


def show(cmd, cwd=ROOT):
    print("      " + " ".join(cmd[:6]) + " ...", flush=True)
    if a.dry_run:
        print("      FULL: " + " ".join(cmd), flush=True)
        return 0
    return subprocess.run(cmd, cwd=cwd,
                          env={**os.environ, "LAYERNORM_TYPE": "openfold"}).returncode


def stage_train(ep, head):
    """Train-pool selected DockQ and frac acceptable, with valfull at the SAME budget 10."""
    out = os.path.join(RUN_D, f"trainmetrics_ep{ep}.json")
    if os.path.exists(out):
        print(f"    train: {os.path.basename(out)} exists, skipping")
        return 0
    return show([PY, "-I", os.path.join(ROOT, "run_logging", "train_eval_metrics.py"),
                 a.run, "--ckpt", head, "--cards", a.cards, "--out", out] + POSE)


def stage_val(ep, head):
    """Valfull selected DockQ at budget 30 -- the number paper_results.md is written in."""
    work = os.path.join(RUN_D, f"valeval_ep{ep}_b{a.val_budget}")
    out = work + ".json"
    if os.path.exists(out):
        print(f"    val: {os.path.basename(out)} exists, skipping")
        return 0
    extra = ["--fold", VALFULL["fold"],
             "--results", os.path.join(VALFULL["fold"], "results.csv"),
             "--pool", VALFULL["pool"],
             "--roles", os.path.join(VALFULL["fold"], "chain_roles.csv"),
             "--seed-base", str(VALFULL["seed_base"])] + POSE + ["--pair-iptm-only", "--prefetch"]
    return show([PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
                 "--cards", a.cards, "--ckpt", head, "--budget", str(a.val_budget),
                 "--work", work, "--out", out, "--extra", " ".join(extra)])


def stage_valpae(ep):
    """Held-out PAE for the bundle, via --loss-only. See the module docstring for why not
    recover_val_pae.py. The command mirrors val_loss_checkpoint() so the number is comparable to
    every other arm's val PAE."""
    bundle = os.path.join(RUN_D, f"resume_ep{ep}.pt")
    out = os.path.join(RUN_D, f"valpae_ep{ep}")
    if os.path.exists(os.path.join(out, "trajectory.csv")):
        print(f"    valpae: {os.path.basename(out)}/trajectory.csv exists, skipping")
        return 0
    cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_train.py"),
           "--loss-only", "--resume", bundle, "--resume-allow-drift",
           "--resume-note", (f"held-out PAE measurement of resume_ep{ep}.pt; --loss-only reads "
                             f"the weights and writes no checkpoint, so the bundle's schedule is "
                             f"not continued. fawgtu ran with no val pass, so this is the first "
                             f"time its weights meet the held-out fold."),
           "--fold", VALFULL["fold"],
           "--labels", os.path.join(VALFULL["fold"], "labels"),
           "--manifest-selection", VALFULL["selection"],
           "--alignment-preflight", VALFULL["preflight"], "--minimal-items",
           "--pae-scored-pairs", VALFULL["pool"],
           "--seeds", VALFULL["seeds"],
           "--indices", os.path.join(VALFULL["prep"], "job_indices.csv"),
           # THE RUN'S OWN GEOMETRY, so the held-out number is comparable to its training curve.
           "--crop-margin", str(ARGS.get("crop_margin", 10.0)),
           "--crop-contact", str(ARGS.get("crop_contact", 5.0)),
           "--crop-size", str(ARGS.get("crop_size", 3400)),
           "--accum", str(ARGS.get("accum", 85)),
           "--nsample", str(ARGS.get("nsample", 5)),
           # alpha_pae 1 ALWAYS: the number this produces IS the PAE loss, and a ranking-only arm
           # passing 0 through would report exactly 0.0 while looking healthy.
           "--alpha-pae", "1.0",
           "--alpha-confidence", str(ARGS.get("alpha_confidence", 1.0)),
           "--clip-norm", str(ARGS.get("clip_norm", 25.0)),
           "--blocks-per-ckpt", str(ARGS.get("blocks_per_ckpt", 0)),
           "--epochs", "1", "--save-every", "0", "--warmup-steps", "0",
           "--shuffle", "structure", "--out", out] + POSE
    if ARGS.get("pae_only"):
        cmd.append("--pae-only")
    if ARGS.get("crop_all_poses"):
        cmd.append("--crop-all-poses")
    return show(cmd)


b = busy()
print(f"  cards {a.cards}: cards_busy reports {b}")
if b != "0" and not (a.dry_run or a.allow_busy):
    raise SystemExit("STOP: cards are occupied. Wait for the training run to finish, or pass "
                     "--allow-busy if you intend to share them.")
print(f"  run {a.run}, epochs {EPOCHS}, stages {STAGES}, arch flags {' '.join(POSE) or '(none)'}")

fails = []
for name in ("train", "val", "valpae"):
    if name not in STAGES:
        continue
    print(f"\n=== STAGE {name} ===", flush=True)
    for ep in EPOCHS:
        print(f"  epoch {ep}:", flush=True)
        if name == "valpae":
            rc = stage_valpae(ep)
        else:
            h = head_for(ep)
            if h is None:
                print(f"    no resume_ep{ep}.pt and no head checkpoint; skipping")
                continue
            rc = stage_train(ep, h) if name == "train" else stage_val(ep, h)
        if rc:
            fails.append((name, ep, rc))
            print(f"    FAILED rc={rc}", flush=True)

print(f"\n  {len(fails)} failure(s)" + (f": {fails}" if fails else ""))
sys.exit(1 if fails else 0)
