#!/bin/bash
# THE DEFAULT ARM: --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED on both
# sides, over the 853-structure mixed-depth population.
#
# rel IS THE DEFAULT BY THE OWNER'S STANDING INSTRUCTION, 2026-11-06, and no other --rank-form is
# substituted without asking. It regresses each pose's ipTM on the TRUE iTM after subtracting the
# interface's own mean: L = (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2. So it uses every pose
# of an interface regardless of seed, which means there is no pair family to choose, no tau, and
# --rank-pairs and --rank-within-weight do not apply to it at all -- they are removed here rather
# than left in looking meaningful.
#
# WHY rel RATHER THAN A PAIRWISE FORM, measured not assumed: the head's failure is POSITIONAL, not
# collapse. On d32kqp epoch 66 the correlation between interface-MEAN ipTM and interface-MEAN DockQ
# was +0.866 while the median WITHIN-interface Spearman was +0.317, against +0.690 for the true
# iTM. Subtracting each interface's own mean from both sides removes the across-complex signal the
# head already has, leaving only the within-complex ordering that selection actually uses.
#
# ITS COEFFICIENTS SUM TO EXACTLY ZERO, so it cannot move an interface's level and its accumulated
# gradient cancels by construction. --alpha-rank must therefore be set from gradient NORMS via
# --grad-probe and never from loss values; 75 is the owner's figure, carried from the earlier
# rel arms.
#
# A NOTE ON HOW THIS SCRIPT CAME TO BE WRONG ONCE. It launched softplus because the rel labels did
# not exist yet, which was true at the time -- and then they were built an hour later and the
# substitution was never revisited. If a dependency ever forces another form again, the moment the
# blocker clears is the moment to re-raise it.
#
# WHY UNCROPPED BOTH SIDES. Every previous arm trained on a crop and evaluated uncropped, and that
# gap is visible in the results: o2qczq scored 0.6305 cropped and 0.5387 uncropped, below shipped
# 0.5474. A cropped reading carries forced ground-truth contacts and is never a deployment number.
# Training uncropped costs more per unit and removes the question.
#
# WHY softplus AND NOT rel. rel needs --rel-labels, which are rebuilt from itm_ceiling's output, and
# that pass is running now. softplus exercises the same cross-seed packing and the same mixed-depth
# manifest, so this epoch is not a placeholder: it is the first measurement of whether the path runs
# and what an epoch costs. The rel arm follows once its labels exist.
#
# WHAT THIS IS THE FIRST TEST OF. The selection manifest's per-structure `seeds` column -- 437
# structures at 15 seeds beside 416 at 1 -- packed into cross-seed steps of a constant 30 units by
# lr_schedule.pack_cross_seed_steps. 232 steps. Three downstream sites read a step's unit count as a
# per-epoch constant (the per-rank quotas, the PAE per-step divisor, the _xs_steps walk) and a wrong
# one DEADLOCKS on an all-reduce rather than failing, which is why world=3 matters here and why the
# one-card smoke test could not have proven it.
#
# --indices all857_indices.csv IS REQUIRED. phase_k_train and the fingerprint gate both DEFAULT to
# job_indices.csv, which names 501 structures; this population's labels were built against
# all857_indices.csv and the indices path is part of the label fingerprint. Left on the default the
# run rejects every label it loads -- the gate reported 2,010 such labels over 223 structures, with
# `code` and `loss` matching exactly, which is the signature of a wrong indices file rather than of
# a stale label.
#
# --rank-pairs all IS THE OBJECTIVE, AND --rank-cross-seed IS ONLY THE STEP SHAPE. I confused the
# two on the first launch and the help text says they are easy to confuse: --rank-cross-seed puts
# all seeds of a structure in one optimiser step, so the ranking GROUP is the whole interface;
# --rank-pairs decides which pairs drawn from that group enter the LOSS, and it defaults to
# `cross`. So run 9ghtvw trained on cross-seed pairs only, which is also why its 416 single-seed
# structures could contribute no ranking gradient at all -- a one-seed group has no cross-seed pair.
# `all` forms both families and lets every structure contribute.
#
# --rank-within-weight L FIXES A SHARE PROBLEM, NOT A STRENGTH ONE, and without it `all` would be
# nearly the same run. Within-seed |ipTM gaps| are tiny -- median 0.0036 against cross-seed -- so
# at L=1 within-seed pairs are 7.9% of pairs and only 7.0% of the pairwise gradient mass, and the
# orderings inside a seed are barely trained. Measured mass share at tau 0.0270:
#     L=1 -> 7.0%    L=3 -> 18.3%    L=8 -> 37.1%    L=20 -> 59.3%
# while mass reaching INVERTED pairs falls only 75.0% -> 71.6% at L=8. L multiplies within-seed
# pairs in BOTH the numerator and the normaliser, so it cannot act as a second --alpha-rank.
# Default 8 here: the largest share that still leaves most gradient on genuinely inverted pairs.
#
# --alignment-preflight IS MANDATORY AND THERE IS NO WAY AROUND IT. I tried dropping
# --minimal-items on the theory that the trainer would then run check_alignment itself per unit;
# it does not. The guard is `if True:` -- the full dataset-item path was removed, so the minimal
# path is the only path and the per-structure check no longer exists. The stale error message said
# "--minimal-items needs --alignment-preflight" and cost one launch attempt before I read the
# condition; it now says what it means.
#
# The marker is built by phases/phase_k_alignment_preflight.py over this exact manifest: ~45 min of
# single-threaded CPU for 851 structures at 3.2 s each, and its digest must equal the trainer's
# (568d3d3038004257 over 6,941 units) or the launch refuses.
#
# --pae-scored-roles IS chain_roles_all.csv, NOT THE DEFAULT. <fold>/chain_roles.csv names 501
# structures and none of this wave's 356, so the PAE mask found no role mapping for 660 of the
# 1,688 pool interfaces and refused rather than silently dropping their blocks. The union file
# covers 853 of 853. This is the same flag the rel ranking path needs, and the reason both of its
# interface_chain_pairs call sites were changed to honour it.
#
# THE REPLAY LOG IS FROM THIS POPULATION, not inherited. phase_k_train refuses to start without
# proof that a cached label reproduces its trunk, and the old arms passed
# training/ft3_pilot_replay.log, which was built for the 501-structure population under
# job_indices.csv. training/extended_mixed_replay.log is 12qa/free_500010 -- one of the NEW seeds,
# under all857_indices.csv -- and it reports max |stored - live| coordinate 0.00e+00 A, 0 mask
# differences, every loss term agreeing to 0.00e+00, and the gradient reaching 165 confidence-head
# tensors and 0 outside it.
#
# THREE STRUCTURES ARE SKIPPED, EACH FOR A MEASURED REASON:
#   9mx1  never folded -- 10 GPU units queued at training/fold/queue_9mx1.tsv, never run.
#   4zto  its ost_json does not associate a pose with a sample ("2.json describes , not sample 2"),
#         because it was scored by the full-complex route. The ceiling pass refuses it rather than
#         fall back to a static chain mapping, which is what --ost-mapping exists to prevent.
#   9q5m  its E-H interface has no rows at seeds 500010-500014 and no route can produce them: chain
#         E is the copy with only 81 of 532 residues resolved, so it ties at 1.00 against all four
#         model antigen copies and only geometry can say which -- and OST's chemical grouping
#         separated the 81-residue copy from the 372-residue ones, so it never offered a mapping.
#         Recorded with its evidence in docs/unrecoverable_interfaces.csv. The ranking-loss gate
#         found it independently: "5 group(s) have NO labels, e.g. ('9q5m', 500010, (0, 4))".
#         Skipping the whole structure costs its five healthy interfaces too; a per-INTERFACE
#         exclusion would be better and does not exist yet. The previous arm already skipped 9q5m.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-10}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
MARKER="${MARKER:-training/alignment_preflight_sharded.json}"
RELLAB="${RELLAB:-training/fold/itm_labels_extended}"
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
B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
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
      --blocks-per-ckpt 1 --ckpt-token-threshold 855
      --dist-backend gloo --dist-timeout-min 240
      --max-rss-gb 80 --preload-min-free-gb 250
      --save-every 250
      --require-replay training/extended_mixed_replay.log
      --eval-pool training/valfull/pool.csv --eval-pool-fold validation/fold
      --eval-pool-seed-base 400000 --eval-pool-budget 30 --eval-pool-every 1
      --allow-code-drift)

NOTE="$EPOCHS EPOCH(S), --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED TRAIN AND EVAL, over the 853-structure mixed-depth population (437 at 15 seeds, 416 at 1, 6,971 declared units, 232 cross-seed steps of 30 units). THE FIRST REAL RUN ON THIS DATA and the first exercise of the selection manifest's per-structure \`seeds\` column at world=3, where a wrong per-step unit count deadlocks on an all-reduce rather than failing. No crop on either side: every previous arm trained cropped and evaluated uncropped, and o2qczq's 0.6305 cropped against 0.5387 uncropped (shipped 0.5474) is what that gap looks like. rel regresses on the true iTM after mean-centring, so it uses every pose of an interface regardless of seed and has no pair family or tau. --indices all857_indices.csv because the labels were built against it and the indices path is part of their fingerprint. NOT a quality verdict at one epoch -- this measures that the path runs and what an epoch costs."

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
