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
import hashlib
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
ap.add_argument("--insample-crop-tokens", default="",
                help="score the IN-SAMPLE pool on this mask instead of the one the guard's digest "
                     "names. THE ONLY REASON THIS EXISTS: the training masks in this project are "
                     "built with --force-gt, which unions the ground-truth interface contacts in "
                     "on top of the pose union, so the crop encodes where the true epitope is. "
                     "That is defensible as a TRAINING crop read out uncropped, and it is not a "
                     "number anyone can deploy. Passing a label-free mask here scores the head on "
                     "a crop it did NOT train under -- deliberately -- and the deviation is "
                     "printed and recorded rather than absorbed. The guard digest is still "
                     "checked against the mask args.json names, so the crop being deviated FROM "
                     "is known to be intact.")
ap.add_argument("--valfull-crop-tokens", default="",
                help="the valfull token mask matching the guard's TRAIN mask. REQUIRED when the "
                     "guard sets crop_tokens, and there is no default on purpose. The train mask "
                     "is built over training/fold's 368 structures at seed-base 500000; applying "
                     "it to validation/fold is not a thing that can be done, and deriving the "
                     "valfull path by string surgery on the train path would silently score a "
                     "third thing the first time a naming convention moved. Build it with "
                     "build_union_crop_masks.py at the SAME --margin, --contact, --force-gt and "
                     "--ost-pairs as the train mask; the metadata of both is checked below.")
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
# THE CROP IS PER POOL, because a precomputed token mask is per STRUCTURE and the two pools hold
# different structures. Under --crop-margin the rule is computed inline from labels and the flags
# are identical for both pools, so that branch still builds one list and shares it.
# THE GUARD'S crop_tokens IS A DIGEST, NOT A PATH. phase_k_train stores a 16-char sha256 over the
# per-structure masks plus the builder's provenance blob, precisely so that rebuilding a mask
# under the same filename cannot resume as "unchanged". The PATH lives in the run's args.json. So
# the path comes from args.json and is then checked AGAINST the guard's digest, which is a
# stronger guarantee than either alone: args.json says which file, the guard proves it is still
# byte-for-byte the file the head trained under.
RID = os.path.basename(os.path.dirname(os.path.join(ROOT, a.bundle)))
_ARGS_PATH = os.path.join(ROOT, "training", "runs", RID, "args.json")
if not os.path.exists(_ARGS_PATH):
    sys.exit(f"STOP: {_ARGS_PATH} is missing, so the train mask path cannot be recovered.")
_ARGS = json.load(open(_ARGS_PATH))
TRAIN_MASK = (_ARGS.get("crop_tokens") or "").strip()
GUARD_CROP_DIGEST = (G.get("crop_tokens") or "").strip()


def _crop_tok_digest(rel):
    """Reproduce phase_k_train's _CROP_TOK_DIGEST for a mask on disk.

    Kept byte-identical to phases/phase_k_train.py around line 1281 on purpose: __pairs holds
    chain INDICES and is hashed as int64, everything else is packed bits, and __meta goes in last.
    A reimplementation that drifted would reject a correct mask.
    """
    import numpy as np
    h = hashlib.sha256()
    z = np.load(os.path.join(ROOT, rel))
    for k in sorted(k for k in z.files if not (k.endswith("__n") or k == "__meta")):
        h.update(k.encode())
        if k.endswith("__pairs"):
            h.update(np.asarray(z[k], dtype=np.int64).tobytes())
        else:
            h.update(np.packbits(np.asarray(z[k]).astype(bool)).tobytes())
    if "__meta" in z.files:
        h.update(np.asarray(z["__meta"]).tobytes())
    return h.hexdigest()[:16]


def _mask_meta(rel):
    """The build settings a mask RECORDS, so a mismatch is a stop rather than a plausible number."""
    import numpy as np
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        sys.exit(f"STOP: mask {rel} does not exist.")
    z = np.load(full, allow_pickle=True)
    if "__meta" not in z:
        sys.exit(f"STOP: mask {rel} has no __meta, so its build settings cannot be checked. "
                 f"Rebuild it with the current build_union_crop_masks.py.")
    return json.loads(str(z["__meta"][0]))


# ---------------------------------------------------------------------------------------------
# WHICH CROP EVERY PASS RUNS UNDER. Three cases, and the third is why this block exists:
#
#   1. a token-mask arm on ITS OWN mask, the crop matching training bit for bit;
#   2. a margin arm on ITS OWN margin, the original behaviour, unchanged;
#   3. EITHER arm on a crop it did NOT train under, named explicitly on the command line.
#
# CASE 3 EXISTS BECAUSE EVERY TRAINING CROP IN THIS PROJECT LEAKS. --crop-margin cuts around
# label_coordinate, and the token masks are built with --force-gt, which unions the ground-truth
# contacts in on top of the pose union. Both hand the head the true interface, so nothing either
# produces is a deployment number -- and the leak is not small: the ground-truth-cropped arm
# (o2qczq/t4jtjr) reads +0.0163 held out on its label crop against +0.0058 uncropped. A LABEL-FREE
# pose-union mask, built from ci_coordinate -- what the fold predicted, available at inference --
# is the honest version, and running one head on both says how much of the gain was the leak.
#
# THE DEVIATION IS NEVER INFERRED. It must be asked for by naming BOTH masks, its rule is checked
# for cross-pool consistency, the before/after of every build setting is printed, and it lands in
# the crop fingerprint so its passes can never be confused with the training-crop ones.
EXPLICIT = bool(a.insample_crop_tokens or a.valfull_crop_tokens)
if EXPLICIT and not (a.insample_crop_tokens and a.valfull_crop_tokens):
    sys.exit("STOP: --insample-crop-tokens and --valfull-crop-tokens go together. One pool scored "
             "on a token mask and the other on the guard's crop would put two different crop "
             "rules in one table, which is the comparison this file exists to prevent.")

# THE GUARD DIGEST IS CHECKED WHENEVER THERE IS ONE, override or not: the crop being deviated
# FROM has to be known intact, or "off-training crop" names nothing.
if GUARD_CROP_DIGEST:
    if not TRAIN_MASK:
        sys.exit(f"STOP: the guard records a crop_tokens digest {GUARD_CROP_DIGEST} but "
                 f"{_ARGS_PATH} has no crop_tokens path, so the train mask cannot be located.")
    _got = _crop_tok_digest(TRAIN_MASK)
    if _got != GUARD_CROP_DIGEST:
        sys.exit(f"STOP: {TRAIN_MASK} hashes to {_got} but the head was trained under "
                 f"{GUARD_CROP_DIGEST}. The mask on disk has been rebuilt since; scoring on it "
                 f"would read the head off a crop it never saw.")
    print(f"  train mask {TRAIN_MASK} verified against the guard digest {GUARD_CROP_DIGEST}")

if EXPLICIT:
    MASKS = {"insample": a.insample_crop_tokens, "valfull": a.valfull_crop_tokens}
    METAS = {t: _mask_meta(m) for t, m in MASKS.items()}
    OFF_TRAIN = MASKS["insample"] != TRAIN_MASK
    # THE RULE MUST BE THE SAME ON BOTH POOLS. Two masks built at different margins would make the
    # in-sample and held-out cells incomparable while every number still looked reasonable.
    for key in ("contact", "margin", "force_gt", "ost_pairs"):
        got = {t: METAS[t].get(key) for t in MASKS}
        if len({repr(v) for v in got.values()}) != 1:
            sys.exit(f"STOP: the two masks disagree on {key}: {got}. Rebuild one of them so both "
                     f"carry the same rule.")
    # AND EACH MASK MUST BELONG TO THE POOL IT IS APPLIED TO, by pool file and evaluated seed
    # range. A train mask handed in as the valfull mask would miss every structure and the
    # evaluator's own fallback would quietly score full structures instead -- a silent uncropped
    # pass wearing a cropped label.
    for tag, fold, pool, sb, budget, jc in POOLS:
        m = METAS[tag]
        if m.get("pool") != pool or int(m.get("seed_base")) != sb or int(m.get("budget")) != budget:
            sys.exit(f"STOP: the {tag} mask was built for pool={m.get('pool')} "
                     f"seed_base={m.get('seed_base')} budget={m.get('budget')}, but this pass "
                     f"scores pool={pool} seed_base={sb} budget={budget}.")
    if OFF_TRAIN:
        # EVERY SETTING PRINTED BEFORE AND AFTER. The guard-geometry check below is skipped in
        # this branch on purpose -- deviating is the point -- so this print is the only thing
        # standing between a deliberate deviation and a typo, and it is kept exhaustive for that
        # reason.
        _trained = (f"{TRAIN_MASK} ({GUARD_CROP_DIGEST})" if GUARD_CROP_DIGEST else
                    f"--crop-margin {G.get('crop_margin')} --crop-contact {G.get('crop_contact')}"
                    + (" --crop-all-poses" if G.get("crop_all_poses") else ""))
        print("  *** OFF-TRAINING CROP, ON PURPOSE ***")
        print(f"      trained under  {_trained}")
        print(f"      scoring under  {MASKS['insample']} ({_crop_tok_digest(MASKS['insample'])})")
        print(f"                     {MASKS['valfull']} ({_crop_tok_digest(MASKS['valfull'])})")
        print("      scored mask rule: margin %s, contact %s, force_gt %s, ost_pairs %s"
              % (METAS["insample"]["margin"], METAS["insample"]["contact"],
                 METAS["insample"]["force_gt"], METAS["insample"]["ost_pairs"]))
        if TRAIN_MASK:
            _tm = _mask_meta(TRAIN_MASK)
            print("      train  mask rule: margin %s, contact %s, force_gt %s, ost_pairs %s"
                  % (_tm.get("margin"), _tm.get("contact"), _tm.get("force_gt"),
                     _tm.get("ost_pairs")))
            # ONE DIRECTION IS REFUSED OUTRIGHT. Dropping --force-gt removes a leak the head had
            # while training, which is the honest read. ADDING it hands the head ground-truth
            # contacts it never saw, which inflates the number and is not a thing anyone wants.
            if METAS["insample"].get("force_gt") and not _tm.get("force_gt"):
                sys.exit("STOP: the override mask uses --force-gt while the arm trained WITHOUT "
                         "it. That adds ground-truth contacts the head never had, which is the "
                         "one direction this flag must not be used in.")
    else:
        # SCORING A TOKEN ARM ON ITS OWN MASK: the geometry must match the guard. Checked only
        # here, because under an override the margin is MEANT to differ from the arm's.
        for key, gkey in (("contact", "crop_contact"), ("margin", "crop_margin")):
            if G.get(gkey) is not None and float(METAS["insample"][key]) != float(G.get(gkey)):
                sys.exit(f"STOP: mask {key}={METAS['insample'][key]} but the guard's {gkey}="
                         f"{G.get(gkey)}. The mask is not the arm's crop rule.")
    CROP_FOR = {t: ["--crop-tokens", MASKS[t]] for t in MASKS}
    # --crop-contact is read only by --crop-margin and --crop-predicted-margin, so it is NOT
    # passed here: the contact threshold is already baked into the mask, and repeating it on the
    # command line would read as though it still had an effect.
    CROP_DESC = ("crop-tokens  insample=%s  valfull=%s  (margin %s, contact %s, force_gt %s, "
                 "ost_pairs %s%s)" % (MASKS["insample"], MASKS["valfull"],
                                      METAS["insample"]["margin"], METAS["insample"]["contact"],
                                      METAS["insample"]["force_gt"], METAS["insample"]["ost_pairs"],
                                      ", OFF-TRAINING CROP" if OFF_TRAIN else ""))
elif GUARD_CROP_DIGEST:
    sys.exit("STOP: this bundle's guard trains under the token mask " + TRAIN_MASK + ", so the "
             "held-out pool needs its OWN mask built under the same rule -- the train mask covers "
             "training/fold's structures and cannot be applied to validation/fold. Pass "
             "--insample-crop-tokens " + TRAIN_MASK + " --valfull-crop-tokens <mask>. Build the "
             "held-out mask with:\n"
             "  " + PY + " -I run_logging/build_union_crop_masks.py \\\n"
             "    --pool training/valfull/pool.csv --roles validation/fold/chain_roles.csv \\\n"
             "    --trunks validation/fold/trunks --labels validation/fold/labels \\\n"
             "    --fold validation/fold --out training/valfull/union_crop_m0_gt_ost.npz \\\n"
             "    --contact 5.0 --margin 0.0 --force-gt --ost-pairs \\\n"
             "    --seed-base 400000 --budget 30 --nsample 5\n"
             "Or drop --force-gt --ost-pairs --roles --fold for the LABEL-FREE mask, which is the "
             "only one of the two that could ever be a deployment number.")
elif float(G.get("crop_margin") or 0) > 0:
    _c = ["--crop-margin", str(G.get("crop_margin")), "--crop-contact", str(G.get("crop_contact"))]
    if G.get("crop_all_poses"):
        _c.append("--crop-all-poses")
    CROP_FOR = {"insample": list(_c), "valfull": list(_c)}
    CROP_DESC = " ".join(_c)
    OFF_TRAIN = False
else:
    sys.exit("STOP: this bundle's guard sets neither crop_margin nor crop_tokens, so there is no "
             "crop to match.")
# BOTH POOLINGS OUT OF EVERY PASS, including the shipped head's. There has never been a
# mean-pooled shipped number in this project, which is why the held-out MEAN comparison has always
# been shipped-MAX against ours-MEAN; one flag closes that.
POOL_FLAGS = ["--pool-mean", "--select-by", "mean"]
print(f"  bundle epoch {EP} step {STEP}")
print(f"  crop from the guard: {CROP_DESC}")
print(f"  architecture from the guard: {' '.join(POSE_FLAGS) or 'none'}")
print(f"  pooling: {' '.join(POOL_FLAGS)}   (rank_pool={G.get('rank_pool')})")

OUT = os.path.join(ROOT, a.out)
os.makedirs(OUT, exist_ok=True)
# the head, extracted once and COPIED so the scored object cannot change underneath a re-read
rid = RID
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

# THE CROP GOES IN THE FILENAME. Every pass is skipped when its .json already exists, and the
# shipped pass carries no epoch, so a second crop variant run into the same --out would reuse the
# FIRST variant's shipped numbers and silently compare two different crops.
#
# THE FINGERPRINT HASHES MASK CONTENTS, NOT FILENAMES. Hashing the flags alone meant a mask
# rebuilt under the SAME filename -- at a different margin, contact or forcing rule -- produced
# the same fingerprint, so the evaluator reused result files computed on the OLD tokens. That is
# the identical failure phase_k_train's crop digest exists to prevent on the training side, and it
# is why the guard stores a digest rather than a path. Raised in review. The per-structure digests
# go in beside the flags, so changing a margin, all-poses, or one bit of one structure's mask all
# force a fresh pass, while a re-run of genuinely identical settings still reuses the cache.
_CROP_PARTS = []
for _t, _v in sorted(CROP_FOR.items()):
    _CROP_PARTS.append((_t, tuple(_v)))
    if _v[:1] == ["--crop-tokens"]:
        _CROP_PARTS.append((_t + ":digest", _crop_tok_digest(_v[1])))
CROP_ID = hashlib.sha1(repr(_CROP_PARTS).encode()).hexdigest()[:8]

jobs = []
for tag, fold, pool, sb, budget, jc in POOLS:
    for who in ("ours", "shipped"):
        name = (f"{tag}_{who}_ep{EP}_{CROP_ID}" if who == "ours"
                else f"{tag}_shipped_{CROP_ID}")
        extra = ["--fold", fold, "--results", os.path.join(fold, "results.csv"),
                 "--pool", pool, "--roles", os.path.join(fold, "chain_roles.csv"),
                 "--seed-base", str(sb), "--labels", os.path.join(fold, "labels"),
                 "--join-cache", jc, "--pair-iptm-only", "--prefetch"] \
                + CROP_FOR[tag] + POOL_FLAGS
        if who == "ours":
            extra += POSE_FLAGS
        cmd = [PY, "-I", os.path.join(ROOT, "phases", "phase_k_eval_sharded.py"),
               "--cards", a.cards, "--budget", str(budget),
               "--work", os.path.join(OUT, name), "--out", os.path.join(OUT, name + ".json"),
               "--extra", " ".join(extra)]
        if who == "ours":
            cmd[cmd.index("--cards"):cmd.index("--cards")] = ["--ckpt", HEAD]
        jobs.append((name, cmd))

print(f"  crop fingerprint: {CROP_ID}")
# MACHINE-READABLE, for the shell script to hand to the report. The report refuses to pick a file
# by modification time, so it needs this exact value; parsing it out of the human line would break
# the first time the wording changed.
print(f"CROP_ID={CROP_ID}")
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
