#!/usr/bin/env python
"""Score a head and the SHIPPED head on the SAME CROPPED inputs, in-sample and held out.

THE QUESTION. phase_k_eval has always scored full structures while phase_k_train crops every
pose, so every comparison in this project has read a cropped-trained head off its training
distribution and the shipped head on its own (section 8c). This runs all four cells of
{in-sample, held out} x {this head, shipped} under one crop, so "did the loss teach the head
anything" is asked on the inputs the loss used.

FOUR PASSES, NOT TWO. The rows' stored_iptm column is the shipped head's RECORDED ipTM, produced
by the ordinary full-structure pipeline -- it is an uncropped number and using it as the cropped
baseline would reintroduce exactly the asymmetry this exists to remove. The shipped head gets its
own cropped pass, with no --ckpt and none of the architecture flags.

EVERY SETTING COMES FROM THE CHECKPOINT'S GUARD, never from this file: crop margin, crop contact,
crop all-poses, the pose pathways and the pooling. A hardcoded margin that drifted from the arm's
would produce a plausible number measuring a third thing.

Usage:
  python -I run_logging/cropped_eval.py --bundle training/runs/t4jtjr/resume_ep30.pt --dry-run
  python -I run_logging/cropped_eval.py --bundle training/runs/t4jtjr/resume_ep30.pt
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/ebrahim/Protenix/.venv/bin/python"

# (tag, fold, pool, seed_base, budget, join_cache)
POOLS = [
    ("insample", "training/fold", "training/full_train/pool_winnable.csv", 500000, 10,
     "training/full_train/join_cache_winnable.json"),
    ("valfull", "validation/fold", "training/valfull/pool.csv", 400000, 30,
     "training/valfull/join_cache.json"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True, help="a resume_ep<N>.pt whose guard supplies every setting")
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--out", default="training/runs/cropped_eval")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

import torch                                                                 # noqa: E402
_b = torch.load(os.path.join(ROOT, a.bundle), map_location="cpu", weights_only=False)
G = _b.get("guard") or {}
EP, STEP = int(_b["epoch"]), int(_b["step"])
POSE_FLAGS = []
if G.get("pose_perblock"):
    POSE_FLAGS.append("--pose-perblock")
    for k, f in (("pose_perblock_tied", "--pose-perblock-tied"),
                 ("pose_perblock_share", "--pose-perblock-share"),
                 ("pose_perblock_copy", "--pose-perblock-copy")):
        if G.get(k):
            POSE_FLAGS.append(f)
if G.get("frame_direction"):
    POSE_FLAGS.append("--frame-direction")
CROP = ["--crop-margin", str(G.get("crop_margin")), "--crop-contact", str(G.get("crop_contact"))]
if G.get("crop_all_poses"):
    CROP.append("--crop-all-poses")
# BOTH POOLINGS OUT OF EVERY PASS, including the shipped head's. There has never been a
# mean-pooled shipped number in this project, which is why the held-out MEAN comparison has always
# been shipped-MAX against ours-MEAN; one flag closes that.
POOL_FLAGS = ["--pool-mean", "--select-by", "mean"]
if float(G.get("crop_margin") or 0) <= 0:
    sys.exit("STOP: this bundle's guard has no crop_margin, so there is no crop to match.")
print(f"  bundle epoch {EP} step {STEP}")
print(f"  crop from the guard: {' '.join(CROP)}")
print(f"  architecture from the guard: {' '.join(POSE_FLAGS) or 'none'}")
print(f"  pooling: {' '.join(POOL_FLAGS)}   (rank_pool={G.get('rank_pool')})")

OUT = os.path.join(ROOT, a.out)
os.makedirs(OUT, exist_ok=True)
# the head, extracted once and COPIED so the scored object cannot change underneath a re-read
rid = os.path.basename(os.path.dirname(os.path.join(ROOT, a.bundle)))
HEAD = os.path.join(OUT, f"{rid}_ep{EP}_head.pt")
if not os.path.exists(HEAD) and not a.dry_run:
    cand = glob.glob(os.path.join(ROOT, "training", "runs", rid, f"head_ep{EP}_step*.pt"))
    if not cand:
        subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "extract_heads.py"),
                        rid, "--epochs", str(EP)], cwd=ROOT, check=True)
        cand = glob.glob(os.path.join(ROOT, "training", "runs", rid, f"head_ep{EP}_step*.pt"))
    if len(cand) != 1:
        sys.exit(f"STOP: expected one head_ep{EP}_step*.pt for {rid}, found {cand}")
    shutil.copyfile(cand[0], HEAD + ".tmp")
    os.replace(HEAD + ".tmp", HEAD)
    print(f"  head {os.path.basename(cand[0])} -> {os.path.basename(HEAD)}")

jobs = []
for tag, fold, pool, sb, budget, jc in POOLS:
    for who in ("ours", "shipped"):
        name = f"{tag}_{who}_ep{EP}" if who == "ours" else f"{tag}_shipped"
        extra = ["--fold", fold, "--results", os.path.join(fold, "results.csv"),
                 "--pool", pool, "--roles", os.path.join(fold, "chain_roles.csv"),
                 "--seed-base", str(sb), "--labels", os.path.join(fold, "labels"),
                 "--join-cache", jc, "--pair-iptm-only", "--prefetch"] + CROP + POOL_FLAGS
        if who == "ours":
            extra += POSE_FLAGS
        cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
               "--cards", a.cards, "--budget", str(budget),
               "--work", os.path.join(OUT, name), "--out", os.path.join(OUT, name + ".json"),
               "--extra", " ".join(extra)]
        if who == "ours":
            cmd[cmd.index("--cards"):cmd.index("--cards")] = ["--ckpt", HEAD]
        jobs.append((name, cmd))

print(f"\n  {len(jobs)} pass(es):")
for name, _c in jobs:
    print(f"    {name}")
if a.dry_run:
    print("\n  DRY RUN -- nothing executed")
    sys.exit(0)

env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
for name, cmd in jobs:
    out = os.path.join(OUT, name + ".json")
    if os.path.exists(out):
        print(f"\n  {name}: exists, skipping")
        continue
    print(f"\n  === {name} ===", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if rc:
        sys.exit(f"STOP: {name} failed with {rc}")
print("\n  all passes done")
