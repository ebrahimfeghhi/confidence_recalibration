#!/usr/bin/env python
"""IN-SAMPLE SELECTION METRICS OVER THE WHOLE CURRENT TRAINING SET, both heads, both poolings.

WHY THIS EXISTS AND WHY insample_eval.py DOES NOT COVER IT. insample_eval.py scores
training/full_train/pool_winnable.csv -- 738 interfaces over 368 structures, the "winnable"
subset, which is 43% of what the live arm actually trains on. The training set is
training/combined/pool.csv: 1688 interfaces over 853 structures. This script scores that.

THE MIXED SEED PROFILE IS PRESERVED, NOT FLATTENED. Measured 2026-11-11 over training/fold/trunks:
437 of the 853 structures hold 15 trunks (seeds 500000-500014) and 416 hold exactly 1 (seed
500000 for 415 of them, 500002 for the odd one -- see below); every one of the 1688 interfaces
has scored DockQ in training/fold/results.csv. A single
`--budget N` label cannot describe that, and phase_k_eval's coverage guard correctly refuses a
budget whose intended units do not all exist -- which is right, because "budget 15" over a
1-trunk structure is a claim about poses that were never generated.

So the pool is SPLIT BY TRUNK DEPTH into sub-pools, each scored at the budget it genuinely has,
and they are then aggregated over interfaces. Measured 2026-11-11 by walking every structure's
trunk directory:

    deep              437 structures, seed base 500000, budget 15  -> 75 candidate poses each
    shallow           415 structures, seed base 500000, budget  1  ->  5 candidate poses each
    shallow_seed500002  1 structure,  seed base 500002, budget  1  ->  5 candidate poses each

Flattening everything to budget 1 would have thrown away 70 of the 75 candidates on the deep
half; running budget 15 over the whole pool is simply false for the shallow half. The split is
the only honest option that keeps every interface.

THE SUB-POOLS ARE DISCOVERED, NOT DECLARED, BECAUSE ONE STRUCTURE IS OFF-BASE. 9dzj (interfaces
9dzj_CE and 9dzj_DE) keeps exactly one trunk and it is seed free_500002, not free_500000 -- the
2026-11-04 prune left a different survivor there than in the other 415 shallow structures. A
hardcoded `shallow = (500000, 1)` therefore hard-stopped the whole script on 9dzj's missing
500000 trunk, which is the coverage guard working correctly: budget 1 from base 500000 claims a
pose that does not exist. So each structure's (seed_base, budget) is read off its own trunk store
-- base = its lowest trunk seed, budget = the contiguous run from there -- and structures sharing
a pair share a sub-pool. 9dzj lands in a sub-pool of its own, 10 candidate poses wide, rather
than being dropped: dropping it would have shrunk the metric by 2 interfaces without saying so,
which is the failure mode this whole file exists to prevent.

IDENTICAL CANDIDATES FOR BOTH HEADS. Every head is scored over the same sub-pools at the same
budgets from the same per-sub-pool seed base, so the candidate set is identical by construction
and the only thing that varies is which pose each selector picks.

BOTH POOLINGS COME OUT OF ONE PASS PER HEAD-SUBPOOL. The shard rows carry `iptm` (max-pooled),
`iptm_mean` (mean-pooled) and `stored_iptm` per pose, so selection by each is a post-hoc argmax
over the same rows -- no extra forward pass, and no possibility of the two poolings disagreeing
about which candidates existed.

THE SHIPPED HEAD IS FORWARDED, NOT READ FROM CSV. results.csv stores one ipTM column and it is
the max-pooled one, so a mean-pooled shipped number is not derivable from it. The shipped passes
run the stock head (no --ckpt, no pose flags) to obtain it. `stored_iptm` then serves as an
independent control on the shipped max column.

ONE HEAD PER INVOCATION. --head names the single head this run scores, either the stock Protenix
confidence head or one epoch bundle, and the run writes its shards under a directory named after
that head. The 4-row shipped-vs-arm table is then a pure read of whatever head directories exist
(--report-only), so the two passes can be run hours apart, on different cards, or not at all,
without one invocation implying anything about the other. The previous shape hardcoded a fixed
4-pass sequence, which meant a shipped pass could not be run without also naming an arm bundle.

Usage:
  python -I run_logging/insample_fullset.py --head shipped --dry-run
  python -I run_logging/insample_fullset.py --head shipped
  python -I run_logging/insample_fullset.py --head training/runs/rdi25k/resume_ep10.pt
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
SEED_BASE = 500000       # the nominal base; each sub-pool's real base is read off its trunks
NSAMPLE = 5
ACCEPTABLE = 0.23
WORK = "training/runs/insample_fullset"

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="",
                help="'shipped' for the stock Protenix head (no --ckpt, no pose flags), or the "
                     "path to one epoch bundle. Required unless --report-only.")
ap.add_argument("--cards", default="4,5,7")
ap.add_argument("--allow-busy-cards", action="store_true",
                help="skip the cards_busy refusal. For the case where you have deliberately "
                     "freed only some cards and are passing exactly those in --cards.")
ap.add_argument("--dry-run", action="store_true", help="run every gate, launch nothing")
ap.add_argument("--report-only", action="store_true", help="aggregate shards already on disk")
a = ap.parse_args()
if not a.head and not a.report_only:
    sys.exit("STOP: pass --head shipped or --head <bundle.pt>. There is no default head: the old "
             "default silently scored training/runs/rdi25k/resume_ep10.pt, so a run meant for the "
             "shipped head could extract and forward an arm's head without the command saying so.")

W = os.path.join(ROOT, WORK)
PASSDIR = os.path.join(W, "passes")      # passes/<head tag>/<sub-pool>/ holds that pass's shards
os.makedirs(PASSDIR, exist_ok=True)


def say(m):
    print(m, flush=True)


# ---- build the sub-pools, and GATE the partition ------------------------------------------
rows = list(csv.DictReader(open(os.path.join(ROOT, POOL))))
structs = sorted({r["pdb"] for r in rows})
trunks = {}
for s in structs:
    d = os.path.join(ROOT, FOLD, "trunks", s)
    trunks[s] = sorted(int(f.split("_")[1].split(".")[0])
                       for f in os.listdir(d)) if os.path.isdir(d) else []

# (SEED BASE, BUDGET) PER STRUCTURE, READ OFF ITS OWN TRUNK STORE. base is its lowest surviving
# trunk seed and budget the contiguous run from there, which is exactly the window phase_k_eval's
# intended_seeds() claims for `--seed-base base --budget budget`. Structures sharing a pair share
# a pass. Measured 2026-11-11: 437 x (500000, 15), 415 x (500000, 1), 1 x (500002, 1) -- 9dzj,
# whose only survivor is seed 500002. Assuming (500000, 1) for every shallow structure made the
# coverage guard reject 9dzj outright, correctly, since budget 1 from 500000 names a pose it does
# not have.
SUBS, assign = {}, {}
for s in structs:
    t = trunks[s]
    if not t:
        sys.exit(f"STOP: {s} has ZERO trunks; it cannot be scored and silently dropping it would "
                 f"shrink the dataset. Regenerate its trunk or remove it from {POOL} deliberately.")
    base, budget = t[0], 1
    while base + budget in set(t):
        budget += 1
    name = {15: "deep", 1: "shallow"}.get(budget, f"budget{budget}")
    if base != SEED_BASE:
        name += f"_seed{base}"
    SUBS[name] = (base, budget)
    assign[s] = name

# DEEPEST FIRST, so the pass that carries 95% of the candidate poses is the one that runs while
# the operator is still watching, and so the printed order is stable across invocations.
SUB_ORDER = sorted(SUBS, key=lambda n: (-SUBS[n][1], SUBS[n][0]))

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
for sub in SUB_ORDER:
    base, budget = SUBS[sub]
    ss = sorted({r["pdb"] for r in sub_rows[sub]})
    ifs = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in sub_rows[sub]]
    want = set(range(base, base + budget))
    bad_t = [s for s in ss if not want <= {int(f.split("_")[1].split(".")[0])
                                          for f in os.listdir(os.path.join(ROOT, FOLD, "trunks", s))}]
    bad_s = [i for i in ifs if not want <= scored[i]]
    say(f"  {sub:20s} seed base {base} budget {budget:2d}: {len(ifs):5d} interfaces, "
        f"{len(ss):4d} structures, {len(ifs) * budget * NSAMPLE:7d} candidate poses")
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

# ---- THE ONE HEAD THIS INVOCATION SCORES --------------------------------------------------------
# THE SHIPPED HEAD IS DEFINED BY WHAT IS *NOT* PASSED: no --ckpt and no pose flags. It is stock
# Protenix, so our arms' --pose-perblock/--frame-direction architecture must not be handed to it;
# doing so would build a different model and call the result "shipped".
HEAD, POSE, HTAG, EP = None, [], "shipped", None
if a.head and a.head != "shipped":
    import torch                                                              # noqa: E402
    _b = torch.load(os.path.join(ROOT, a.head), map_location="cpu", weights_only=False)
    # IT MUST BE A RESUME BUNDLE, NOT AN EXTRACTED HEAD. training/runs/<rid>/head_ep<N>_step<S>.pt
    # is a bare state dict with no 'epoch' and no 'guard', so pointing --head at one used to die on
    # a KeyError after the gates had already run -- and worse, a guard-less load would have fallen
    # back to zero pose flags, i.e. scored a pose-architecture arm as if it were stock Protenix.
    if not isinstance(_b, dict) or "epoch" not in _b or "guard" not in _b:
        sys.exit(f"STOP: {a.head} carries no 'epoch'/'guard', so it is an extracted head, not a "
                 f"resume bundle. Pass the bundle (training/runs/<rid>/resume_ep<N>.pt); the head "
                 f"is extracted from it here. Without its guard the pose architecture is unknown.")
    # ARCHITECTURE FROM THE BUNDLE'S OWN guard, NEVER HARDCODED: the arms differ in pose wiring and
    # a mismatched flag set loads the checkpoint into the wrong module graph.
    G = _b.get("guard") or {}
    EP = int(_b["epoch"])
    if G.get("pose_perblock"):
        POSE.append("--pose-perblock")
        for k, f in (("pose_perblock_tied", "--pose-perblock-tied"),
                     ("pose_perblock_share", "--pose-perblock-share"),
                     ("pose_perblock_copy", "--pose-perblock-copy")):
            if G.get(k):
                POSE.append(f)
    if G.get("frame_direction"):
        POSE.append("--frame-direction")
    RID = os.path.basename(os.path.dirname(os.path.join(ROOT, a.head)))
    HTAG = f"{RID}_ep{EP}"
    say(f"  head: arm {RID} bundle epoch {EP}; architecture from its guard: "
        f"{' '.join(POSE) or 'none'}")

    HEAD = os.path.join(W, f"{HTAG}_head.pt")
    if not a.dry_run and not a.report_only and not os.path.exists(HEAD):
        cand = glob.glob(os.path.join(ROOT, "training", "runs", RID, f"head_ep{EP}_step*.pt"))
        if not cand:
            subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "extract_heads.py"),
                            RID, "--epochs", str(EP)], cwd=ROOT, check=True)
            cand = glob.glob(os.path.join(ROOT, "training", "runs", RID, f"head_ep{EP}_step*.pt"))
        if len(cand) != 1:
            sys.exit(f"STOP: expected one head_ep{EP}_step*.pt for {RID}, found {cand}")
        import shutil
        shutil.copyfile(cand[0], HEAD + ".tmp")
        os.replace(HEAD + ".tmp", HEAD)
        say(f"  head {os.path.basename(cand[0])} -> {os.path.basename(HEAD)}")
elif a.head:
    say("  head: SHIPPED stock Protenix confidence head (no --ckpt, no pose flags)")

# ---- this head's passes, ONE PER SUB-POOL -------------------------------------------------------
PASSES = [(sub, SUBS[sub][0], SUBS[sub][1]) for sub in SUB_ORDER]
HDIR = os.path.join(PASSDIR, HTAG)

if a.head:
    say(f"\n  {len(PASSES)} sharded pass(es) for head '{HTAG}' on cards {a.cards}:")
    for sub, base, budget in PASSES:
        say(f"    {sub:20s} pool_{sub}.csv  seed base {base}  budget {budget:2d}  "
            f"{'SHIPPED (no --ckpt, no pose flags)' if HEAD is None else 'arm head + ' + ' '.join(POSE)}")

# ---- CARDS GATE, LAST SO THE DATA GATES ABOVE ARE ALWAYS SEEN ----------------------------------
# THIS SCRIPT IS RUN BY SOMEONE WHO HAS JUST STOPPED A TRAINING RUN, which is the moment at which
# "the cards are free now" is an assumption rather than an observation. Without this gate the pass
# would pile its three shards onto a live arm. Measured 2026-11-11: cards_busy reported 5 while
# the regularised arm held 4/5/7, and an earlier pass today refused for the same reason because
# orphaned shards of mine were still resident -- that refusal was the gate working.
#
# IT ASKS ABOUT THE CARDS THIS PASS WILL ACTUALLY USE (--cards), not cards_busy's own 4,5,7
# default, so freeing a different card set is recognised. cards_busy reads GPU occupancy by UUID
# and FAILS CLOSED, printing the non-numeric "UNKNOWN" when it cannot establish occupancy, so
# anything other than an exact "0" here is a refusal -- an unknown state must read as busy.
#
# IT RUNS IN --dry-run TOO, because a dry run that skipped the one gate standing between this
# script and a live arm would certify a launch it had not checked.
def cards_gate():
    if a.allow_busy_cards:
        say(f"\n  --allow-busy-cards: NOT checking cards {a.cards}. Only correct when you have "
            f"deliberately freed exactly these cards and know what else is resident.")
        return
    r = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging", "cards_busy.py"), a.cards],
                       cwd=ROOT, capture_output=True, text=True)
    busy = (r.stdout or "").strip().splitlines()[-1].strip() if (r.stdout or "").strip() else ""
    if busy != "0":
        sys.exit(f"\nSTOP: cards_busy reports {busy or '(no output)'} for cards {a.cards} -- "
                 f"refusing to launch onto occupied cards.\n"
                 f"{(r.stderr or '').rstrip()}\n"
                 f"  Wait for the cards to free, pass --cards with the ones that are, or "
                 f"--allow-busy-cards to override deliberately.")
    say(f"\n  cards {a.cards}: cards_busy reports 0 foreign processes")


# NOT IN --report-only: it reads shards off disk and touches no card, so refusing it while the
# cards are busy would block reading a result that is already computed.
if not a.report_only:
    cards_gate()

if a.dry_run:
    say("\n  DRY RUN -- gates only, nothing launched")
    sys.exit(0)

if not a.report_only:
    os.makedirs(HDIR, exist_ok=True)
    # WHAT THIS HEAD IS, RECORDED BESIDE ITS SHARDS. --report-only reads these sidecars to find the
    # heads on disk, so a table row can never be labelled from a flag the aggregating invocation
    # happened to be given rather than from the pass that produced the rows.
    with open(os.path.join(HDIR, "head.json"), "w") as fh:
        json.dump({"tag": HTAG, "kind": "shipped" if HEAD is None else "ckpt",
                   "head": a.head, "epoch": EP, "pose_flags": POSE,
                   "subpools": {s: {"seed_base": b, "budget": g} for s, b, g in PASSES}}, fh,
                  indent=1)
    for sub, base, budget in PASSES:
        out = os.path.join(HDIR, f"metric_{sub}.json")
        shards = os.path.join(HDIR, sub)
        if os.path.exists(out):
            say(f"\n  {HTAG}/{sub}: metric exists, skipping")
            continue
        extra = ["--fold", os.path.join(ROOT, FOLD),
                 "--results", os.path.join(ROOT, FOLD, "results.csv"),
                 "--pool", os.path.join(W, f"pool_{sub}.csv"),
                 "--roles", os.path.join(ROOT, ROLES),
                 "--seed-base", str(base),
                 "--labels", os.path.join(ROOT, FOLD, "labels"),
                 "--pair-iptm-only", "--prefetch", "--pool-mean", "--select-by", "mean",
                 # MERGE-ONLY, AND THE REASON THE DEEP POOL DID NOT HAVE TO BE RE-RUN TWICE.
                 # One unresolvable pose used to refuse a merge holding 67,325 of 67,425 rows.
                 "--drop-incomplete-ifaces"]
        if HEAD is not None:
            extra += POSE
        cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
               "--cards", a.cards, "--budget", str(budget),
               "--work", shards, "--out", out, "--extra", " ".join(extra)]
        if HEAD is not None:
            cmd[3:3] = ["--ckpt", HEAD]
        say(f"\n  === {HTAG} / {sub} ===")
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "LAYERNORM_TYPE": "openfold"})
        if r.returncode != 0:
            sys.exit(f"STOP: {HTAG}/{sub} exited {r.returncode}; later passes not launched.")

# ---- aggregate over interfaces -----------------------------------------------------------------
def collect(tag):
    """iface -> list of (iptm, iptm_mean, stored_iptm, dockq) over its candidate poses.

    AGGREGATION ACROSS SUB-POOLS IS THE POINT: the sub-pools partition the 1688 interfaces, so
    unioning their rows under one head gives that head's metric over the whole training set.
    """
    by = collections.defaultdict(list)
    want_depth = {}
    for sub in SUBS:
        _base, _budget = SUBS[sub]
        for f in glob.glob(os.path.join(PASSDIR, tag, sub, "rows_b*_s*.csv")):
            for r in csv.DictReader(open(f)):
                if r["dockq"] in ("", None):
                    continue
                def _f(k):
                    v = r.get(k, "")
                    return float(v) if v not in ("", None) else None
                by[r["iface"]].append((_f("iptm"), _f("iptm_mean"), _f("stored_iptm"),
                                       float(r["dockq"])))
                want_depth[r["iface"]] = _budget * NSAMPLE
    # AN INTERFACE SHORT OF ITS SUB-POOL'S DEPTH IS DROPPED WHOLE, NOT AVERAGED IN SHORT.
    #
    # Selecting over 25 of 75 candidates is an easier problem than selecting over 75, so an
    # interface kept at partial depth would raise this head's number for a reason that has
    # nothing to do with the head. 4zto is the measured case: its OST chain mapping failed when
    # its poses were generated, 50 of its 75 pose jsons are failure records, and it cannot be
    # scored at depth by anything. Dropping it costs 2 of 1688 interfaces and is reported.
    short = sorted(i for i, c in by.items() if len(c) != want_depth.get(i))
    for i in short:
        del by[i]
    return by, short


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


# EVERY HEAD THAT HAS SHARDS ON DISK, SHIPPED FIRST. This is what makes the 4-row comparison a
# read rather than a re-run: two single-head invocations leave two sidecars, and this block finds
# both. A head whose rows do not cover all 1688 interfaces is NAMED AND EXCLUDED rather than
# averaged -- a partial pass reported as a whole-training-set number is the exact substitution the
# gates above exist to block, and it would look like a head that simply scored worse.
heads = []
for hj in sorted(glob.glob(os.path.join(PASSDIR, "*", "head.json"))):
    h = json.load(open(hj))
    heads.append((0 if h["kind"] == "shipped" else 1, h["tag"], h))
heads.sort()

TABLE, ships, partial = [], None, []
DROPPED = {}
for _, tag, h in heads:
    by, short = collect(tag)
    DROPPED[tag] = short
    if h["kind"] == "shipped":
        ships = by
    name = "shipped" if h["kind"] == "shipped" else f"epoch {h['epoch']} ({tag})"
    if short:
        say(f"  {name}: DROPPED {len(short)} interface(s) short of their sub-pool's depth: "
            f"{', '.join(short[:6])}{' ...' if len(short) > 6 else ''}")
    if len(by) + len(short) != len(_all):
        partial.append(f"{name}: {len(by)} scored + {len(short)} dropped of "
                       f"{len(_all)} interfaces")
        continue
    for pool_label, col in (("max-pooled ipTM", 0), ("mean-pooled ipTM", 1)):
        s = summarise(by, col)
        if s:
            TABLE.append((f"{name}, {pool_label}", s))

# INDEPENDENT CONTROL on the shipped MAX column ONLY: stored_iptm from results.csv, selected the
# same way. It is a different provenance for the same quantity, so a disagreement means one of the
# two is not scoring what it claims. It is NOT a baseline for any mean-pooled row -- results.csv
# holds one ipTM column and it is max-pooled, so no mean-pooled shipped number is derivable from
# it, which is why the shipped head has to be forwarded at all.
ctl = summarise(ships, 2) if ships is not None else None

say("")
say(f"  {'selector':42s} {'n_if':>6} {'mean DockQ':>11} {'frac acc':>9}")
for label, s in TABLE:
    say(f"  {label:42s} {s['n_interfaces']:6d} {s['mean_dockq']:11.4f} {s['frac_acc']:9.4f}")
if ctl:
    say(f"  {'(control) shipped stored MAX-pooled':42s} {ctl['n_interfaces']:6d} "
        f"{ctl['mean_dockq']:11.4f} {ctl['frac_acc']:9.4f}")
if TABLE:
    say(f"  {'oracle over the candidates':42s} {TABLE[0][1]['n_interfaces']:6d} "
        f"{TABLE[0][1]['oracle_mean_dockq']:11.4f} {TABLE[0][1]['oracle_frac_acc']:9.4f}")
say("  max-pooled and mean-pooled rows are DIFFERENT selectors; compare each only to its own "
    "pooling.")
for p in partial:
    say(f"  EXCLUDED, pass incomplete -- {p}")

res = {"pool": POOL, "roles": ROLES, "nsample": NSAMPLE,
       "subpools": {s: {"seed_base": SUBS[s][0], "budget": SUBS[s][1],
                        "n_interfaces": len(sub_rows[s]),
                        "n_structures": len({r["pdb"] for r in sub_rows[s]})}
                    for s in SUB_ORDER},
       "heads": [h for _, _, h in heads],
       "rows": {label: s for label, s in TABLE},
       "control_stored_max_pooled": ctl,
       "excluded_incomplete": partial}
with open(os.path.join(W, "fullset_report.json"), "w") as fh:
    json.dump(res, fh, indent=1)
say(f"\n  -> {WORK}/fullset_report.json")
