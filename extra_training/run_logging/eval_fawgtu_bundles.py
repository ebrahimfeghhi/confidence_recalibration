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

BUDGETS DIFFER BETWEEN STAGES AND ARE NOT COMPARABLE ACROSS THEM. Stage 1 is the TRAIN POOL
ONLY, at budget 10 -- forced, since the training fold holds only seeds 500000-500009. Stage 2 is
VALFULL ONLY, at budget 30, which is what every figure in paper_results.md uses. Valfull is
deliberately NOT also measured at budget 10: the user asked for it at 30 and nowhere else, and an
extra budget-10 valfull column invites exactly the comparison that should not be made.

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


def show(cmd, cwd=ROOT, cards=None):
    """Run a stage. `cards` sets CUDA_VISIBLE_DEVICES for stages that do NOT shard themselves.

    THE SHARDED STAGES MUST NOT GET IT. phase_k_eval_sharded assigns each shard its own card and
    sets the variable per child; handing it a pre-set CUDA_VISIBLE_DEVICES would renumber the
    devices under it. But a direct phase_k_train call inherits this process's environment, and
    this orchestrator is normally started with CUDA_VISIBLE_DEVICES="" so the driver itself never
    claims a card -- which would have left stage 3 with no GPU at all. Caught before it ran.
    """
    print("      " + " ".join(cmd[:6]) + " ...", flush=True)
    if a.dry_run:
        print("      FULL: " + " ".join(cmd), flush=True)
        return 0
    env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
    if cards is not None:
        env["CUDA_VISIBLE_DEVICES"] = cards
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


TRAIN = {"fold": "training/fold",
         "pool": "training/full_train/pool_insample.csv",
         "join_cache": "training/full_train/join_cache.json",
         "seed_base": 500000,
         "budget": 10}
"""BUDGET 10 IS FORCED, not chosen: the training fold was folded at seeds 500000-500009, so ten
is all there is. The valfull side runs at 30 and the two numbers are therefore NOT comparable to
each other -- they answer "how well does it pick on structures it trained on" and "how well does
it pick on held-out structures", each against its own oracle."""


def stage_train(ep, head):
    """Train-pool selected DockQ and frac acceptable. TRAIN POOL ONLY.

    NOT train_eval_metrics.py, for two independent reasons. It takes only [--cards] [--thresh]
    [--no-run] and positional rids -- no --ckpt, no --out, so it auto-discovers a checkpoint
    instead of scoring the one asked for, and the first version of this script passed it flags it
    does not have and failed with "unrecognized arguments". It also reports valfull at budget 10
    alongside the train pool, which is a measurement the user explicitly does not want: valfull is
    wanted at budget 30 and nowhere else.

    So this calls the same sharded evaluator stage 2 does, pointed at the training fold.
    """
    work = os.path.join(RUN_D, f"traineval_ep{ep}_b{TRAIN['budget']}")
    out = work + ".json"
    if os.path.exists(out):
        print(f"    train: {os.path.basename(out)} exists, skipping")
        return 0
    extra = ["--fold", TRAIN["fold"],
             "--results", os.path.join(TRAIN["fold"], "results.csv"),
             "--pool", TRAIN["pool"],
             "--roles", os.path.join(TRAIN["fold"], "chain_roles.csv"),
             "--seed-base", str(TRAIN["seed_base"])] + POSE + [
             # CACHED BESIDE THE POOL, as train_eval_metrics did: the join is a function of the
             # pool and the trunks, so the first arm pays for it and later ones are free.
             "--join-cache", TRAIN["join_cache"],
             "--pair-iptm-only", "--prefetch"]
    return show([PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
                 "--cards", a.cards, "--ckpt", head, "--budget", str(TRAIN["budget"]),
                 "--work", work, "--out", out, "--extra", " ".join(extra)])


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
    cards = [c.strip() for c in a.cards.split(",") if c.strip()]
    base = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_train.py"),
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
           "--shuffle", "structure"] + POSE
    if ARGS.get("pae_only"):
        base.append("--pae-only")
    if ARGS.get("crop_all_poses"):
        base.append("--crop-all-poses")

    # SHARDED ACROSS THE CARDS, one process per card, because --loss-only is single-process and
    # 1,590 held-out units on one GPU is three times the wall clock for no reason. phase_k_train
    # carries --manifest-shard/--manifest-nshards for exactly this, and the trainer's own val
    # monitor shards the same way.
    procs = []
    for i, c in enumerate(cards):
        sub = os.path.join(out, f"shard{i}")
        os.makedirs(sub, exist_ok=True)
        cmd = base + ["--manifest-shard", str(i), "--manifest-nshards", str(len(cards)),
                      "--out", sub]
        print(f"    shard {i} -> card {c}", flush=True)
        if a.dry_run:
            print("      FULL: " + " ".join(cmd), flush=True)
            continue
        procs.append(subprocess.Popen(
            cmd, cwd=ROOT, env={**os.environ, "LAYERNORM_TYPE": "openfold",
                                "CUDA_VISIBLE_DEVICES": c},
            stdout=open(os.path.join(sub, "shard.log"), "w"), stderr=subprocess.STDOUT))
    if a.dry_run:
        return 0
    rcs = [pr.wait() for pr in procs]
    if any(rcs):
        print(f"    shard return codes {rcs}", flush=True)
        return max(rcs)

    # POOLED POSE-WEIGHTED, not as a mean of shard means: the shards hold unequal pose counts, so
    # averaging their averages would silently weight a small shard like a large one. This is the
    # same formula the trainer uses on its own val trajectories.
    import csv as _csv
    tot, n = 0.0, 0
    for i in range(len(cards)):
        t = os.path.join(out, f"shard{i}", "trajectory.csv")
        if not os.path.exists(t):
            print(f"    shard {i} wrote no trajectory.csv -- NOT publishing a partial pool, "
                  f"because a mean over a biased subset of the held-out set looks comparable "
                  f"and is not", flush=True)
            return 1
        for r in _csv.DictReader(open(t)):
            v = r.get("pae_loss")
            if v not in (None, ""):
                tot += float(v)
                n += 1
    if not n:
        print("    pooled 0 poses", flush=True)
        return 1
    res = {"epoch": ep, "val_pae": tot / n, "n_poses": n, "shards": len(cards),
           "bundle": os.path.basename(bundle)}
    with open(os.path.join(out, "valpae.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    VAL PAE {res['val_pae']:.4f} over {n} pose(s)", flush=True)
    return 0


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
