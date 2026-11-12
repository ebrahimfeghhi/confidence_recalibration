#!/usr/bin/env python
"""IN-SAMPLE SELECTION METRICS OVER THE WHOLE CURRENT TRAINING SET, both heads, both poolings.

WHY THIS EXISTS AND WHY insample_eval.py DOES NOT COVER IT. insample_eval.py scores
training/full_train/pool_winnable.csv -- 738 interfaces over 368 structures, the "winnable"
subset, which is 43% of what the live arm actually trains on. The training set is
training/combined/pool.csv: 1688 interfaces over 853 structures. This script scores that.

THE MIXED SEED PROFILE IS PRESERVED, NOT FLATTENED. Measured 2026-11-11 over training/fold/trunks:
437 of the 853 structures hold 15 trunks (seeds 500000-500014) and 416 hold exactly 1 (seed
500000); every one of the 1688 interfaces has scored DockQ in training/fold/results.csv. A single
`--budget N` label cannot describe that, and phase_k_eval's coverage guard correctly refuses a
budget whose intended units do not all exist -- which is right, because "budget 15" over a
1-trunk structure is a claim about poses that were never generated.

So the pool is SPLIT BY TRUNK DEPTH into two sub-pools, each scored at the budget it genuinely
has, and the two are then aggregated over interfaces:

    deep     437 structures, budget 15  -> 15 seeds x 5 samples = 75 candidate poses each
    shallow  416 structures, budget  1  ->  1 seed  x 5 samples =  5 candidate poses each

Flattening everything to budget 1 would have thrown away 70 of the 75 candidates on the deep
half; running budget 15 over the whole pool is simply false for the shallow half. The split is
the only honest option that keeps every interface.

IDENTICAL CANDIDATES FOR BOTH HEADS. Both heads are scored over the same two sub-pools at the
same budgets with the same seed base, so the candidate set is identical by construction and the
only thing that varies is which pose each selector picks.

BOTH POOLINGS COME OUT OF ONE PASS PER HEAD-SUBPOOL. The shard rows carry `iptm` (max-pooled),
`iptm_mean` (mean-pooled) and `stored_iptm` per pose, so selection by each is a post-hoc argmax
over the same rows -- no extra forward pass, and no possibility of the two poolings disagreeing
about which candidates existed.

THE SHIPPED HEAD IS FORWARDED, NOT READ FROM CSV. results.csv stores one ipTM column and it is
the max-pooled one, so a mean-pooled shipped number is not derivable from it. The shipped passes
run the stock head (no --ckpt, no pose flags) to obtain it. `stored_iptm` then serves as an
independent control on the shipped max column.

Usage:
  python -I run_logging/insample_fullset.py --dry-run
  python -I run_logging/insample_fullset.py --bundle training/runs/rdi25k/resume_ep10.pt
  python -I run_logging/insample_fullset.py --report-only
"""
import argparse
import collections
import csv
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/ebrahim/Protenix/.venv/bin/python"
FOLD = "training/fold"
POOL = "training/combined/pool.csv"
ROLES = "training/fold/chain_roles_all.csv"   # covers 853 of 853; fold/chain_roles.csv has 501
SEED_BASE = 500000
NSAMPLE = 5
ACCEPTABLE = 0.23
WORK = "training/runs/insample_fullset"

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="training/runs/rdi25k/resume_ep10.pt")
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--dry-run", action="store_true", help="run every gate, launch nothing")
ap.add_argument("--report-only", action="store_true", help="aggregate shards already on disk")
a = ap.parse_args()

W = os.path.join(ROOT, WORK)
os.makedirs(W, exist_ok=True)


def say(m):
    print(m, flush=True)


# ---- build the two sub-pools, and GATE the partition ------------------------------------------
rows = list(csv.DictReader(open(os.path.join(ROOT, POOL))))
structs = sorted({r["pdb"] for r in rows})
depth = {}
for s in structs:
    d = os.path.join(ROOT, FOLD, "trunks", s)
    depth[s] = len(os.listdir(d)) if os.path.isdir(d) else 0

SUBS = {"deep": 15, "shallow": 1}
assign = {}
for s in structs:
    if depth[s] >= 15:
        assign[s] = "deep"
    elif depth[s] >= 1:
        assign[s] = "shallow"
    else:
        sys.exit(f"STOP: {s} has ZERO trunks; it cannot be scored and silently dropping it would "
                 f"shrink the dataset. Regenerate its trunk or remove it from {POOL} deliberately.")

sub_rows = collections.defaultdict(list)
for r in rows:
    sub_rows[assign[r["pdb"]]].append(r)

# EVERY INTERFACE EXACTLY ONCE. Codex asked for sharding without duplicate interfaces; this is
# the stronger statement -- the sub-pools PARTITION the pool, so aggregation over them is an
# aggregation over the pool with no double counting and no omission.
_all = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in rows]
_part = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for sub in SUBS for r in sub_rows[sub]]
if sorted(_all) != sorted(_part):
    sys.exit("STOP: the sub-pools do not partition the pool.")
if len(set(_all)) != len(_all):
    sys.exit(f"STOP: {len(_all) - len(set(_all))} duplicate interface(s) in {POOL}.")

# ROLES MUST COVER EVERY STRUCTURE. interface_chain_pairs SKIPS an interface whose roles it
# cannot resolve, so a short roles file shrinks the metric without saying so.
_roles = {r["pdb"] for r in csv.DictReader(open(os.path.join(ROOT, ROLES)))}
_norole = sorted(set(structs) - _roles)
if _norole:
    sys.exit(f"STOP: {len(_norole)} structure(s) absent from {ROLES}, e.g. {_norole[:5]}. "
             f"interface_chain_pairs would skip them and the metric would cover a subset.")

# EVERY INTENDED UNIT MUST HAVE A TRUNK *AND* A SCORED DOCKQ, checked here rather than discovered
# by a shard dying 40 minutes in.
scored = collections.defaultdict(set)
with open(os.path.join(ROOT, FOLD, "results.csv")) as fh:
    for r in csv.DictReader(fh):
        if r.get("dockq") not in (None, ""):
            scored[r["iface"]].add(int(r["seed"]))

say(f"  pool {POOL}: {len(_all)} interfaces over {len(structs)} structures")
for sub, budget in SUBS.items():
    ss = sorted({r["pdb"] for r in sub_rows[sub]})
    ifs = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in sub_rows[sub]]
    want = set(range(SEED_BASE, SEED_BASE + budget))
    bad_t = [s for s in ss if not want <= {int(f.split("_")[1].split(".")[0])
                                          for f in os.listdir(os.path.join(ROOT, FOLD, "trunks", s))}]
    bad_s = [i for i in ifs if not want <= scored[i]]
    say(f"  {sub:8s} budget {budget:2d}: {len(ifs):5d} interfaces, {len(ss):4d} structures, "
        f"{len(ifs) * budget * NSAMPLE:7d} candidate poses")
    if bad_t:
        sys.exit(f"STOP: {len(bad_t)} {sub} structure(s) lack a trunk in the budget window, "
                 f"e.g. {bad_t[:4]}.")
    if bad_s:
        sys.exit(f"STOP: {len(bad_s)} {sub} interface(s) lack a scored DockQ in the budget "
                 f"window, e.g. {bad_s[:4]}.")
    p = os.path.join(W, f"pool_{sub}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "chain1", "chain2"])
        w.writeheader()
        for r in sub_rows[sub]:
            w.writerow({k: r[k] for k in ("pdb", "chain1", "chain2")})
say("  GATES PASSED: partition exact, roles complete, every intended unit has a trunk and a score")

# ---- the arm's architecture, from its own guard ------------------------------------------------
import torch                                                                  # noqa: E402
_b = torch.load(os.path.join(ROOT, a.bundle), map_location="cpu", weights_only=False)
G = _b.get("guard") or {}
EP = int(_b["epoch"])
POSE = []
if G.get("pose_perblock"):
    POSE.append("--pose-perblock")
    for k, f in (("pose_perblock_tied", "--pose-perblock-tied"),
                 ("pose_perblock_share", "--pose-perblock-share"),
                 ("pose_perblock_copy", "--pose-perblock-copy")):
        if G.get(k):
            POSE.append(f)
if G.get("frame_direction"):
    POSE.append("--frame-direction")
RID = os.path.basename(os.path.dirname(os.path.join(ROOT, a.bundle)))
say(f"  arm {RID} bundle epoch {EP}; architecture from its guard: {' '.join(POSE) or 'none'}")

HEAD = os.path.join(W, f"{RID}_ep{EP}_head.pt")
if not a.dry_run and not a.report_only and not os.path.exists(HEAD):
    cand = glob.glob(os.path.join(ROOT, "training", "runs", RID, f"head_ep{EP}_step*.pt"))
    if len(cand) != 1:
        sys.exit(f"STOP: expected one head_ep{EP}_step*.pt for {RID}, found {cand}")
    import shutil
    shutil.copyfile(cand[0], HEAD + ".tmp")
    os.replace(HEAD + ".tmp", HEAD)
    say(f"  head {os.path.basename(cand[0])} -> {os.path.basename(HEAD)}")

# ---- the four passes, SHIPPED FIRST -------------------------------------------------------------
PASSES = [(f"shipped_{s}", s, SUBS[s], None) for s in ("deep", "shallow")] + \
         [(f"ep{EP}_{s}", s, SUBS[s], HEAD) for s in ("deep", "shallow")]

say(f"\n  {len(PASSES)} sharded pass(es) on cards {a.cards}, shipped first:")
for name, sub, budget, ckpt in PASSES:
    say(f"    {name:16s} pool_{sub}.csv  budget {budget:2d}  "
        f"{'SHIPPED (no --ckpt, no pose flags)' if ckpt is None else 'arm head + ' + ' '.join(POSE)}")

if a.dry_run:
    say("\n  DRY RUN -- gates only, nothing launched")
    sys.exit(0)

if not a.report_only:
    for name, sub, budget, ckpt in PASSES:
        out = os.path.join(W, f"metric_{name}.json")
        shards = os.path.join(W, f"shards_{name}")
        if os.path.exists(out):
            say(f"\n  {name}: metric exists, skipping")
            continue
        extra = ["--fold", os.path.join(ROOT, FOLD),
                 "--results", os.path.join(ROOT, FOLD, "results.csv"),
                 "--pool", os.path.join(W, f"pool_{sub}.csv"),
                 "--roles", os.path.join(ROOT, ROLES),
                 "--seed-base", str(SEED_BASE),
                 "--labels", os.path.join(ROOT, FOLD, "labels"),
                 "--pair-iptm-only", "--prefetch", "--pool-mean", "--select-by", "mean"]
        if ckpt is not None:
            extra += POSE
        cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
               "--cards", a.cards, "--budget", str(budget),
               "--work", shards, "--out", out, "--extra", " ".join(extra)]
        if ckpt is not None:
            cmd[3:3] = ["--ckpt", ckpt]
        say(f"\n  === {name} ===")
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "LAYERNORM_TYPE": "openfold"})
        if r.returncode != 0:
            sys.exit(f"STOP: {name} exited {r.returncode}; later passes not launched.")

# ---- aggregate over interfaces -----------------------------------------------------------------
def collect(tag):
    """iface -> list of (iptm, iptm_mean, stored_iptm, dockq) over its candidate poses."""
    by = collections.defaultdict(list)
    for sub in SUBS:
        for f in glob.glob(os.path.join(W, f"shards_{tag}_{sub}", "rows_b*_s*.csv")):
            for r in csv.DictReader(open(f)):
                if r["dockq"] in ("", None):
                    continue
                def _f(k):
                    v = r.get(k, "")
                    return float(v) if v not in ("", None) else None
                by[r["iface"]].append((_f("iptm"), _f("iptm_mean"), _f("stored_iptm"),
                                       float(r["dockq"])))
    return by


def summarise(by, col):
    sel, orc = [], []
    for ifc, cand in by.items():
        pick = [c for c in cand if c[col] is not None]
        if not pick:
            return None
        sel.append(max(pick, key=lambda c: c[col])[3])
        orc.append(max(c[3] for c in cand))
    n = len(sel)
    return {"n_interfaces": n,
            "mean_dockq": round(sum(sel) / n, 4),
            "frac_acc": round(sum(1 for d in sel if d >= ACCEPTABLE) / n, 4),
            "oracle_mean_dockq": round(sum(orc) / n, 4),
            "oracle_frac_acc": round(sum(1 for d in orc if d >= ACCEPTABLE) / n, 4)}


TABLE, ships = [], collect("shipped")
arms = collect(f"ep{EP}")
for label, by, col in (("shipped, max-pooled ipTM", ships, 0),
                       ("shipped, mean-pooled ipTM", ships, 1),
                       (f"epoch {EP}, max-pooled ipTM", arms, 0),
                       (f"epoch {EP}, mean-pooled ipTM", arms, 1)):
    s = summarise(by, col)
    if s:
        TABLE.append((label, s))
# INDEPENDENT CONTROL on the shipped max column: stored_iptm from results.csv, selected the same
# way. It is a different provenance for the same quantity, so a disagreement means one of the two
# is not scoring what it claims.
ctl = summarise(ships, 2)

say("")
say(f"  {'selector':30s} {'n_if':>6} {'mean DockQ':>11} {'frac acc':>9}")
for label, s in TABLE:
    say(f"  {label:30s} {s['n_interfaces']:6d} {s['mean_dockq']:11.4f} {s['frac_acc']:9.4f}")
if ctl:
    say(f"  {'(control) stored max-pooled':30s} {ctl['n_interfaces']:6d} "
        f"{ctl['mean_dockq']:11.4f} {ctl['frac_acc']:9.4f}")
if TABLE:
    say(f"  {'oracle over the candidates':30s} {TABLE[0][1]['n_interfaces']:6d} "
        f"{TABLE[0][1]['oracle_mean_dockq']:11.4f} {TABLE[0][1]['oracle_frac_acc']:9.4f}")

res = {"pool": POOL, "roles": ROLES, "seed_base": SEED_BASE, "nsample": NSAMPLE,
       "subpools": {s: {"budget": b, "n_interfaces": len(sub_rows[s]),
                        "n_structures": len({r["pdb"] for r in sub_rows[s]})}
                    for s, b in SUBS.items()},
       "bundle": a.bundle, "epoch": EP, "pose_flags": POSE,
       "rows": {label: s for label, s in TABLE},
       "control_stored_max_pooled": ctl}
with open(os.path.join(W, "fullset_report.json"), "w") as fh:
    json.dump(res, fh, indent=1)
say(f"\n  -> {WORK}/fullset_report.json")
