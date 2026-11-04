#!/usr/bin/env python
"""Score an UNCROPPED arm's head on the in-sample winnable pool, in ONE pass.

WHY ONE PASS AND NOT FOUR. cropped_eval.py needs a separate shipped pass because the rows'
stored_iptm is the shipped head's RECORDED ipTM from the ordinary FULL-STRUCTURE pipeline, and
comparing a cropped head against it reintroduces the very input asymmetry that script exists to
remove (section 8c). An UNCROPPED arm is already on that pipeline's inputs, so stored_iptm IS the
matched baseline and a second forward pass would reproduce it. This script therefore REFUSES a
bundle whose guard has a crop -- use cropped_eval.py for those.

EVERY ARCHITECTURE SETTING COMES FROM THE CHECKPOINT'S GUARD, never from this file.

Usage:
  python -I run_logging/insample_eval.py --bundle training/runs/jb5veo/resume_ep19.pt --dry-run
  python -I run_logging/insample_eval.py --bundle training/runs/egw5wq/resume_ep27.pt --pool-name valfull
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

# THE TWO UNCROPPED POOLS, as the trainer's own passes define them -- so an off-cadence reading
# lands on the SAME number the run would have written at a multiple of 5 or 10. Each entry is
# exactly what --eval-pool / --extra-eval pass, including the budget, because a budget change is a
# different quantity and not a sampling choice: `insample` is the in-sample pool cropped_eval.py
# uses, so its numbers land beside section 9's.
#
# WHY THIS SCRIPT AND NOT eval_fawgtu_bundles.py, which also scores a run's bundles: its train
# stage reads pool_insample.csv rather than the 738-interface pool_winnable.csv, and NEITHER of
# its stages passes --pool-mean, so both readings are a different pool and a different pooling
# from the ones every metric_*_ep<N>.json in this project carries. Checked 2026-11-03 before
# using it for egw5wq epoch 27.
#
# The join cache is a BASE path; phase_k_eval_sharded appends _shard<N>.
POOLS = {
    "insample": {"fold": "training/fold",
                 "pool": "training/full_train/pool_winnable.csv",
                 "seed_base": 500000,
                 "budget": 10,          # 10 seeds x 5 samples = all 50 training poses
                 "join": "training/full_train/join_cache_winnable.json"},
    "valfull":  {"fold": "validation/fold",
                 "pool": "training/valfull/pool.csv",
                 "seed_base": 400000,
                 "budget": 30,
                 "join": "training/valfull/join_cache.json"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True,
                help="a resume_ep<N>.pt whose guard supplies every architecture setting")
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--out", default="training/runs/insample_eval")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--pool-name", default="insample", choices=sorted(POOLS),
                help="which uncropped pool to score; see POOLS")
a = ap.parse_args()

_P = POOLS[a.pool_name]
FOLD, POOL = _P["fold"], _P["pool"]
SEED_BASE, BUDGET, JOIN = _P["seed_base"], _P["budget"], _P["join"]

for p in (POOL, os.path.join(FOLD, "results.csv"), os.path.join(FOLD, "chain_roles.csv")):
    if not os.path.exists(os.path.join(ROOT, p)):
        sys.exit(f"STOP: missing {p}")

import torch                                                                 # noqa: E402
_b = torch.load(os.path.join(ROOT, a.bundle), map_location="cpu", weights_only=False)
G = _b.get("guard") or {}
EP, STEP = int(_b["epoch"]), int(_b["step"])

_cm = float(G.get("crop_margin") or 0.0)
if _cm > 0:
    sys.exit(f"STOP: this bundle trains CROPPED (crop_margin {_cm}). stored_iptm is an uncropped "
             f"number, so it is not a matched baseline for it -- use cropped_eval.py, which runs "
             f"the shipped head's own cropped pass.")

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
# BOTH POOLINGS out of the one pass, as cropped_eval.py does, so MEAN and MAX are both reported.
POOL_FLAGS = ["--pool-mean", "--select-by", "mean"]

rid = os.path.basename(os.path.dirname(os.path.join(ROOT, a.bundle)))
print(f"  {rid} bundle epoch {EP} step {STEP}")
print(f"  crop: NONE (guard crop_margin {_cm:g}) -- stored_iptm is the matched shipped baseline")
print(f"  architecture from the guard: {' '.join(POSE_FLAGS) or 'none'}")
print(f"  pooling: {' '.join(POOL_FLAGS)}   (rank_pool={G.get('rank_pool')})")
print(f"  pool: {POOL}  budget {BUDGET}  seed-base {SEED_BASE}")

OUT = os.path.join(ROOT, a.out)
os.makedirs(OUT, exist_ok=True)
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

# WRITE WHERE THE PLOTS LOOK. An off-cadence pass used to land in --out
# (training/runs/insample_eval/<pool>_<rid>_ep<N>/), and every plot in this project finds its eval
# rows with eval_dirs, which globs training/runs/<rid>/<tag>_eval_shards_ep<N>. So egw5wq's
# epoch-27 valfull and in-sample passes were computed, reported and then INVISIBLE to the figures
# -- the plots simply skipped epoch 27 and nothing said why. Writing the canonical paths makes an
# off-cadence pass indistinguishable from one the trainer wrote on its own cadence.
#
# IT REFUSES TO CLOBBER rather than merging into an existing directory: a half-populated shard dir
# from a different checkpoint would be averaged in silently, and the result would be a number for
# no particular head.
RUN_DIR = os.path.join(ROOT, "training", "runs", rid)
SHARD_DIR = os.path.join(RUN_DIR, f"{a.pool_name}_eval_shards_ep{EP}")
METRIC = os.path.join(RUN_DIR, f"metric_{a.pool_name}_ep{EP}.json")
if os.path.isdir(SHARD_DIR) and glob.glob(os.path.join(SHARD_DIR, "rows_b*_s*.csv")):
    sys.exit(f"STOP: {SHARD_DIR} already holds shard rows. The trainer's own pass at this epoch "
             f"has already written them, or a previous run of this script did. Remove it "
             f"deliberately if you mean to recompute.")

name = f"{a.pool_name}_{rid}_ep{EP}"
extra = ["--fold", FOLD, "--results", os.path.join(FOLD, "results.csv"),
         "--pool", POOL, "--roles", os.path.join(FOLD, "chain_roles.csv"),
         "--seed-base", str(SEED_BASE), "--labels", os.path.join(FOLD, "labels"),
         "--join-cache", JOIN, "--pair-iptm-only", "--prefetch"] + POOL_FLAGS + POSE_FLAGS
cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
       "--ckpt", HEAD, "--cards", a.cards, "--budget", str(BUDGET),
       "--work", SHARD_DIR, "--out", METRIC,
       "--extra", " ".join(extra)]

print(f"\n  1 pass: {name}")
print(f"  shards -> {SHARD_DIR}\n  metric -> {METRIC}")
print("    " + " ".join(cmd[:8]) + " ...")
if a.dry_run:
    print("\n  DRY RUN -- nothing executed")
    print("  extra: " + " ".join(extra))
    sys.exit(0)

outj = METRIC
if os.path.exists(outj):
    print(f"\n  {name}: exists, skipping")
else:
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    if r.returncode != 0:
        sys.exit(f"STOP: {name} exited {r.returncode}")
print("\n  " + json.dumps(json.load(open(outj)), indent=1))
