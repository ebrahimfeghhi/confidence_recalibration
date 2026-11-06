#!/bin/bash
# THE DEFAULT ARM: --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED on both
# sides, over the 851-structure mixed-depth population.
#
# rel IS THE DEFAULT BY THE OWNER'S STANDING INSTRUCTION, 2026-11-06, and no other --rank-form is
# substituted without asking. It regresses each pose's ipTM on the TRUE iTM after subtracting the
# interface's own mean:
#
#     L = (1/P) sum_i [ (s_i - mean_s) - (g_i - mean_g) ]^2
#
# so it reads every pose of an interface regardless of seed. There is NO pair family, NO tau, and
# --rank-pairs and --rank-within-weight do not apply -- they are absent from the argv below rather
# than set to something that looks operative.
#
# WHY rel RATHER THAN A PAIRWISE FORM, measured not assumed: the head's failure is POSITIONAL, not
# collapse. On d32kqp epoch 66 the correlation between interface-MEAN ipTM and interface-MEAN DockQ
# was +0.866 while the median WITHIN-interface Spearman was +0.317, against +0.690 for the true
# iTM. Subtracting each interface's own mean from both sides removes the across-complex signal the
# head already has, leaving only the within-complex ordering that selection uses.
#
# ITS COEFFICIENTS SUM TO EXACTLY ZERO, so it cannot move an interface's level and its accumulated
# gradient cancels by construction. --alpha-rank must come from gradient NORMS via --grad-probe and
# never from loss values; 75 is the owner's figure, carried from the earlier rel arms.
#
# WHY UNCROPPED ON BOTH SIDES. Every previous arm trained on a crop and evaluated uncropped, and the
# gap shows: o2qczq scored 0.6305 cropped and 0.5387 uncropped, below shipped 0.5474. A cropped
# reading carries forced ground-truth contacts and is never a deployment number. Uncropped costs
# ~57 s/step and ~3.6 h/epoch at world=3, so quote that before proposing a long run.
#
# --ckpt-token-threshold IS 0 HERE, AND 855 IS WHAT KILLED tztukd. The flag means units AT OR BELOW
# that many tokens run with checkpointing OFF. 855 was MEASURED by run_logging/bench_ckpt_by_size.py
# on a population whose largest unit was 1,543 tokens AND which trained under token dropout, so its
# pair work was about a quarter of full. This population's largest unit is 2,289 tokens and there is
# no dropout, so an unchecked sub-855 unit no longer fits: tztukd died of CUDA OOM 2.6 h into epoch 1
# on 8g30, a 731-token structure, in the backward through pairformer.pair_transition, with 93.9 of
# 94.97 GiB in use on two ranks at once.
#
# The flag's own help says "SET IT FROM run_logging/bench_ckpt_by_size.py, NOT BY GUESSING ... A
# threshold chosen by arithmetic would be an OOM 40 minutes into an epoch." I carried a measured
# number to a population it was not measured on, which is the same mistake as guessing. 0 disables
# the size rule so every unit keeps --blocks-per-ckpt 1.
#
# IT COSTS COMPUTE AND NO MATHEMATICS. Checkpointing recomputes the head's four pairformer blocks in
# backward, about 30% more compute there, and gpu_backward was 70.3 min of a 143.9 min epoch at the
# old scale. It is gradient-exact: test_perblock_ckpt_equiv.py gates blocks_per_ckpt=1 against off
# at 0.000e+00 on every parameter. So this slows the arm and changes nothing about what it learns.
# Re-benchmark on THIS population to recover the speed; do not reuse 855.
#
# --indices all857_indices.csv IS REQUIRED. phase_k_train and the fingerprint gate both DEFAULT to
# job_indices.csv, which names 501 structures; this population's labels were built against
# all857_indices.csv and the indices path is part of the label fingerprint. Left on the default the
# run rejects every label it loads -- the gate reported 2,010 such labels over 223 structures, with
# `code` and `loss` matching exactly, which is the signature of a wrong indices file and not of a
# stale label.
#
# --alignment-preflight IS MANDATORY AND UNCONDITIONAL. The guard is `if True:` because the full
# dataset-item path was removed, so the minimal path is the only path and check_alignment can no
# longer run per structure. Dropping --minimal-items does NOT avoid it. The marker is built by
# phases/phase_k_alignment_preflight.py --workers 9 over this exact manifest and its digest must
# equal the trainer's or the launch refuses.
#
# THREE STRUCTURES ARE SKIPPED, EACH FOR A MEASURED REASON:
#   9mx1  never folded -- 10 GPU units queued at training/fold/queue_9mx1.tsv, never run.
#   4zto  its ost_json does not associate a pose with a sample ("2.json describes , not sample 2"),
#         because it was scored by the full-complex route.
#   9q5m  its E-H interface has no rows at seeds 500010-500014 and no route can produce them: chain
#         E is the copy with only 81 of 532 residues resolved, so it ties at 1.00 against all four
#         model antigen copies and only geometry can say which. Recorded with its evidence in
#         docs/unrecoverable_interfaces.csv. Skipping the whole structure also drops its five
#         HEALTHY interfaces; a per-interface exclusion would be better and does not exist yet.
# So the trained population is 851 of 853 declared structures and 6,941 of 6,971 declared units,
# packing into 231 cross-seed steps of 30 units. Those are the numbers this script's banner quotes;
# an earlier version quoted the pre-exclusion 6,971/232 and was wrong by the three skips.
#
# FIVE GATES RUN BEFORE ANY CARD IS TOUCHED: the pool agrees with the declared depth; every declared
# unit has a trunk and a label; the alignment marker passed and covers this manifest; the rel labels
# exist, report zero unmapped rows and cover every trained structure AND every expected
# (seed, sample, iface) key; and the cards are free with
# MemAvailable above the 250 GB floor other users need.

set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-10}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
MARKER="${MARKER:-training/alignment_preflight_sharded.json}"
RELLAB="${RELLAB:-training/fold/itm_labels_extended}"
# 0 MEANS CHECKPOINT EVERY UNIT, which is the only safe default for an UNCROPPED population.
CKPT_THRESH="${CKPT_THRESH:-0}"
# NOEVAL=1 DROPS THE HELD-OUT EVALUATION. The owner asked for training only on 2026-11-06; the
# eval-free config existed only in resume_noeval_to_10.sh, which needs a bundle, so a fresh
# relaunch after an OOM silently brought eval back. A fresh launch must be able to honour it.
NOEVAL="${NOEVAL:-0}"
POOL=training/combined/pool.csv
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  DRY=$DRY cards=$CARDS epochs=$EPOCHS"

[ -s "$MANIFEST" ] || { say "REFUSING: no $MANIFEST"; exit 1; }
say "gate 1: the pool must agree with the declared depth"
$PY -I run_logging/build_combined_train_pool.py --expect-manifest "$MANIFEST" \
  --allow-short docs/unrecoverable_interfaces.csv >/dev/null \
  || { say "REFUSING: the pool gate fails"; exit 1; }
say "  OK"
say "gate 2: every declared unit has a trunk and a label"
$PY -I run_logging/build_extended_train_selection.py \
  --reclaim-journal training/runs/reclaim_train.json \
  --reclaim-journal training/runs/reclaim_train_wave1.json --require-ready >/dev/null \
  || { say "REFUSING: a declared unit is missing a trunk or a label"; exit 1; }
say "  OK"
say "gate 3: the alignment marker exists, passed, and covers THIS manifest"
[ -s "$MARKER" ] || { say "REFUSING: no alignment marker at $MARKER"; exit 1; }
$PY -I - <<EOF || exit 1
import json, sys
j = json.load(open("$MARKER"))
if not j.get("ok") or not j.get("complete"):
    sys.exit(f"REFUSING: {'$MARKER'} is not a pass (ok={j.get('ok')}, complete={j.get('complete')})")
print(f"  marker ok, {j.get('n_structures')} of {j.get('n_manifest_structures')} structure(s), "
      f"digest {j.get('manifest_digest')}, sharded={j.get('sharded')}")
EOF
say "gate 4: the rel labels exist and cover the trained population"
[ -d "$RELLAB" ] || { say "REFUSING: no rel labels at $RELLAB"; exit 1; }
RELLAB="$RELLAB" MANIFEST="$MANIFEST" $PY -I - <<'EOF' || exit 1
import csv, json, os, sys
rl, mf = os.environ["RELLAB"], os.environ["MANIFEST"]
skip = {"9mx1", "4zto", "9q5m"}
want = [r["pdb"].strip().lower() for r in csv.DictReader(open(mf))
        if r["pdb"].strip().lower() not in skip]
miss = [p for p in want if not os.path.exists(os.path.join(rl, f"{p}.npz"))]
m = os.path.join(rl, "manifest.json")
j = json.load(open(m)) if os.path.exists(m) else {}
print(f"  rel labels: {j.get('structures')} structure(s), {j.get('rows')} row(s), "
      f"unmapped {j.get('rows_unmapped')}, pool {os.path.basename(str(j.get('pool')))}")
if miss:
    sys.exit(f"REFUSING: {len(miss)} trained structure(s) have no rel label, e.g. {miss[:5]}")
if j.get("rows_unmapped"):
    sys.exit(f"REFUSING: the rel labels report {j['rows_unmapped']} unmapped row(s)")
print(f"  every one of {len(want)} trained structure(s) has a rel label")
EOF
# AND THE EXACT KEY SET, not just the files. Raised in review: a directory-level check cannot see a
# structure whose label file exists but is short of a seed, a sample or an interface, and the
# trainer only discovers that when the epoch reaches the pose -- up to 3.6 h in on this population.
$PY -I run_logging/check_rel_label_coverage.py --labels "$RELLAB" --manifest "$MANIFEST" \
  --pool "$POOL" --skip '9mx1,4zto,9q5m' --exclude-ifaces docs/unrecoverable_interfaces.csv \
  || { say "REFUSING: the rel label key set is incomplete"; exit 1; }
# A DRY RUN MUST NOT REQUIRE FREE CARDS. This gate sat before the DRY check, so validating a
# config while anything was training refused at "cards_busy reports '5'" and never printed the
# command line -- which made a dry run useless exactly when it is most wanted, and made my own
# test of the NOEVAL path report a false pass.
B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
if [ "$DRY" = "1" ]; then
  say "DRY RUN: cards_busy reports '$B' and MemAvailable $(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo) GB -- not gated on a dry run"
  B=0
fi
[ "${B:-1}" != "0" ] && { say "REFUSING: cards_busy reports '$B', not 0."; exit 1; }
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "gate 5: cards_busy 0, MemAvailable ${M} GB"
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable under the 250 GB floor."; exit 1; }

ARGS=(--rank-loss --rank-form rel --rel-labels "$RELLAB"
      --rank-cross-seed --rank-structs-per-step 2
      --alpha-rank 75 --rank-pool mean
      --fold training/fold
      --indices training/prep/all857_indices.csv
      --manifest-selection "$MANIFEST"
      --manifest-skip '9mx1,4zto,9q5m'
      --pae-scored-pairs "$POOL" --pae-scored-roles training/fold/chain_roles_all.csv
      --pae-only --lean-readouts --minimal-items
      --alignment-preflight "$MARKER"
      --epochs "$EPOCHS" --lr 0.01 --warmup-steps 20 --seed 0 --accum 30 --nsample 5
      --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0
      --prefetch-units 2 --prefetch-workers 4
      --blocks-per-ckpt 1 --ckpt-token-threshold "$CKPT_THRESH"
      --dist-backend gloo --dist-timeout-min 240
      --max-rss-gb 80 --preload-min-free-gb 250
      --save-every 250
      --require-replay training/extended_mixed_replay.log
      --allow-code-drift)
# WHAT DROPPING EVAL COSTS, stated so it is a choice and not a surprise: no held-out valfull
# Spearman, no selected DockQ, no acceptable-pose fraction at 0.23. What survives is the trainer's
# own per-epoch IN-SAMPLE lines -- WITHIN-INTERFACE SPEARMAN (predicted mean ipTM vs DockQ) and
# SCORE SPREAD against std(true iTM) -- which need no eval and must never be read as generalisation.
if [ "$NOEVAL" != "1" ]; then
  ARGS+=(--eval-pool training/valfull/pool.csv --eval-pool-fold validation/fold
         --eval-pool-seed-base 400000 --eval-pool-budget 30 --eval-pool-every 1)
else
  say "NOEVAL=1: no --eval-pool. IN-SAMPLE diagnostics only; no held-out number from this run."
fi

NOTE="$EPOCHS EPOCH(S), --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED TRAIN AND EVAL, over the 851-structure trained population (437 at 15 seeds, 416 at 1, less the three skips: 6,941 units in 231 cross-seed steps of 30). THE FIRST REAL RUN ON THIS DATA and the first exercise of the selection manifest's per-structure \`seeds\` column at world=3, where a wrong per-step unit count deadlocks on an all-reduce rather than failing. No crop on either side: every previous arm trained cropped and evaluated uncropped, and o2qczq's 0.6305 cropped against 0.5387 uncropped (shipped 0.5474) is what that gap looks like. rel regresses on the true iTM after mean-centring, so it uses every pose of an interface regardless of seed and has no pair family or tau. --indices all857_indices.csv because the labels were built against it and the indices path is part of their fingerprint. NOT a quality verdict at one epoch -- this measures that the path runs and what an epoch costs."

if [ "$DRY" = "1" ]; then
  say "DRY RUN"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
    --numa-map 0:0,1:0,2:1 --note "$NOTE" --dry-run -- "${ARGS[@]}"
  say "dry run exit $?"
  exit 0
fi
say "LAUNCHING on cards $CARDS"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}"
say "launch exit $?"
