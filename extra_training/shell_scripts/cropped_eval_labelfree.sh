#!/bin/bash
# SCORE A HEAD AND THE SHIPPED HEAD ON A LABEL-FREE UNION-OVER-POSES CROP, IN-SAMPLE AND HELD OUT.
#
#   ./shell_scripts/cropped_eval_labelfree.sh y8pyuq 20 posefree     # union only, label-free
#   ./shell_scripts/cropped_eval_labelfree.sh y8pyuq 20 posefree_m5  # union + 5 A, label-free
#   ./shell_scripts/cropped_eval_labelfree.sh y8pyuq 20 gt_ost       # the training crop, contrast
#   DRY=1 ./shell_scripts/cropped_eval_labelfree.sh y8pyuq 20        # print the passes, run nothing
#
# THREE EVAL CONDITIONS, and the primary one is not in this script. The UNCROPPED pass is the
# deployable result and already runs every 5 epochs as metric_valfull_ep*.json; everything here is
# a secondary diagnostic. The two cropped conditions bracket it:
#
#   union only   (margin 0)  the contact tokens and nothing else -- the harshest crop, and
#                            possibly too sparse: no surrounding context for the head to lean on.
#   union + 5 A  (margin 5)  every token within 5 A of a contact token as well, which is more
#                            spatially coherent and carries nearby support residues, while still
#                            using no ground truth anywhere.
#
# WHAT THE THREE-WAY COMPARISON BUYS: union-only winning says the head wants tight interface
# focus; union+5 A winning says nearby context matters; uncropped winning says cropping does not
# help at inference and the cropped gains were about the crop, not the ranking.
#
# THE MARGIN IS APPLIED PER POSE AND THEN UNIONED, not applied to the union. "Expand the union by
# 5 A" has no well-defined answer across 50 poses -- expand in WHICH pose's coordinates? So
# crop_from_coords takes each pose's own contacts, expands them in that pose's own frame, and the
# masks are OR-ed: union_i(expand_i(contacts_i)). That is also exactly what the trainer's
# interface_crop_tokens does, which --verify-rule confirms bit for bit on every structure.
#
# ONE SHARED MASK PER STRUCTURE EITHER WAY, applied to all of its poses, which is what keeps ipTM
# d0 and the token count constant across the candidates being ranked.
#
# THE QUESTION. Every cropped number this project has produced came off a crop that leaks.
# phase_k_train's --crop-margin cuts around label_coordinate, and the precomputed token masks are
# built with --force-gt, which unions the ground-truth interface contacts in on top of the pose
# union. Both tell the head where the true epitope is, so neither is a number anyone can deploy.
#
# AND THE LEAK IS NOT SMALL. The ground-truth-cropped arm (o2qczq, resumed as t4jtjr) reads
# +0.0163 held out on its label crop -- about 3.7 paired s.e. and the only figure in this project
# that clears the +0.009 significance bar -- against +0.0058 uncropped, which is inside the noise.
# Cropping roughly triples the apparent gain. This script exists to find out whether any of that
# survives when the crop cannot see the answer.
#
# WHAT LABEL-FREE MEANS HERE: the union, over all of a structure's candidate poses, of each pose's
# own PREDICTED interface -- contacts within --contact of ci_coordinate, which is what the fold
# produced. No experimental structure is read, so the same mask could be built at inference time.
# build_union_crop_masks.py --verify-rule re-runs the identical code on the LABEL coordinates and
# asserts it reproduces the trainer's interface_crop_tokens bit for bit, which is the only
# guarantee that the rule applied here is the rule the arm trained under.
#
# WHY ONE MASK PER STRUCTURE AND NOT PER POSE: ipTM's d0 is set from the chain pair's token count,
# so cropping each pose separately changes the scoring function between the candidates being
# ranked -- measured at d0 swinging 3.00 -> 4.81 A inside one interface. One shared mask gives
# every pose the same tokens, the same n and the same d0.
#
# FOUR PASSES, NOT TWO. results.csv's stored ipTM is the shipped head's UNCROPPED number, so using
# it as the cropped baseline would reintroduce exactly the asymmetry this removes (section 8c).
# The shipped head gets its own cropped pass with no --ckpt and none of the architecture flags.
#
# CARDS. This shares cards 4,5,7 with whatever is training, exactly as the per-epoch evals already
# do -- an epoch that ran both eval passes still took 43 min. It launches NOTHING and stops
# NOTHING; the trainer is untouched.
set -u
cd "$(dirname "$0")/.."
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=${1:?usage: cropped_eval_labelfree.sh <run_id> <epoch> [posefree|gt_ost]}
EP=${2:?usage: cropped_eval_labelfree.sh <run_id> <epoch> [posefree|gt_ost]}
VARIANT=${3:-posefree}   # posefree | posefree_m5 | gt_ost | gt_ost_m5
CARDS=${CARDS:-4,5,7}
OUT=training/runs/cropped_eval_${RID}
BUNDLE=training/runs/${RID}/resume_ep${EP}.pt
say() { echo "$(TZ=America/New_York date '+%F %I:%M:%S %p %Z') $*"; }

# MARGIN IS PART OF THE VARIANT, not a separate flag, so the mask filename, the check and the
# report heading can never disagree about which crop was scored.
case "$VARIANT" in
  posefree)    GT_FLAGS=""        ; SUF=posefree ; MARGIN=0.0 ; MTAG=0 ;;
  posefree_m5) GT_FLAGS=""        ; SUF=posefree ; MARGIN=5.0 ; MTAG=5 ;;
  gt_ost)      GT_FLAGS="forcegt" ; SUF=gt_ost   ; MARGIN=0.0 ; MTAG=0 ;;
  gt_ost_m5)   GT_FLAGS="forcegt" ; SUF=gt_ost   ; MARGIN=5.0 ; MTAG=5 ;;
  *) echo "STOP: variant must be posefree, posefree_m5, gt_ost or gt_ost_m5, got '$VARIANT'"
     exit 1 ;;
esac
[ -f "$BUNDLE" ] || { echo "STOP: $BUNDLE does not exist."; exit 1; }

# THE MASKS. Built if absent and verified either way -- a mask that exists is not a mask that is
# correct, and a stale one from an earlier margin would score a crop nobody chose. The two pools
# differ in every argument that matters: the training pool is 368 structures at seed-base 500000
# with budget 10 x nsample 5 = 50 poses per union; held out is 53 structures at 400000 with
# budget 30 x nsample 5 = 150. Mixing them up is the failure the per-pool checks in
# cropped_eval.py exist to catch.
IS_MASK=training/full_train/union_crop_m${MTAG}_${SUF}.npz
VF_MASK=training/valfull/union_crop_m${MTAG}_${SUF}.npz

build_mask() {
  local out=$1 pool=$2 fold=$3 sb=$4 budget=$5 labels=$6 trunks=$7
  local extra=""
  [ -n "$GT_FLAGS" ] && extra="--force-gt --ost-pairs --roles ${fold}/chain_roles.csv --fold ${fold}"
  say "building $out"
  # shellcheck disable=SC2086
  $PY -I run_logging/build_union_crop_masks.py \
    --pool "$pool" --trunks "$trunks" --labels "$labels" \
    --out "$out" --contact 5.0 --margin "$MARGIN" \
    --seed-base "$sb" --budget "$budget" --nsample 5 $extra \
    || { echo "STOP: building $out failed."; exit 1; }
}

[ -f "$IS_MASK" ] || build_mask "$IS_MASK" training/full_train/pool_winnable.csv \
  training/fold 500000 10 training/fold/labels training/fold/trunks
[ -f "$VF_MASK" ] || build_mask "$VF_MASK" training/valfull/pool.csv \
  validation/fold 400000 30 validation/fold/labels validation/fold/trunks

# VERIFY BOTH, EVERY TIME. check_crop_mask reads the mask's own recorded provenance, so it catches
# a file whose name says margin 0 while its contents say otherwise. --require-force-gt is asserted
# only for the gt_ost variant; for posefree the ABSENCE of forcing is the property under test, so
# it is checked directly below rather than waived.
CHECK_GT=""
[ -n "$GT_FLAGS" ] && CHECK_GT="--require-force-gt --require-ost-pairs"
# shellcheck disable=SC2086
$PY -I run_logging/check_crop_mask.py "$IS_MASK" --structures 368 --poses 50 \
  --margin "$MARGIN" --contact 5.0 $CHECK_GT || { echo "STOP: $IS_MASK failed its check."; exit 1; }
# shellcheck disable=SC2086
$PY -I run_logging/check_crop_mask.py "$VF_MASK" --structures 53 --poses 150 \
  --margin "$MARGIN" --contact 5.0 $CHECK_GT || { echo "STOP: $VF_MASK failed its check."; exit 1; }

if [ -z "$GT_FLAGS" ]; then
  # THE PROPERTY THE WHOLE SCRIPT RESTS ON, asserted rather than assumed: a posefree mask must
  # record force_gt=false. A mask built with forcing and renamed would otherwise sail through as
  # "label-free" and every number below would be a leaked number wearing the honest label.
  WANT_MARGIN="$MARGIN" $PY -I - "$IS_MASK" "$VF_MASK" <<'PYCHK' || exit 1
import json, os, sys
import numpy as np
for p in sys.argv[1:]:
    m = json.loads(str(np.load(p, allow_pickle=True)["__meta"][0]))
    if m.get("force_gt") or m.get("ost_pairs"):
        sys.exit(f"STOP: {p} records force_gt={m.get('force_gt')} "
                 f"ost_pairs={m.get('ost_pairs')}, so it is NOT label-free.")
    if abs(float(m["margin"]) - float(os.environ["WANT_MARGIN"])) > 1e-9:
        sys.exit(f"STOP: {p} records margin {m['margin']} but this run asked for "
                 f"{os.environ['WANT_MARGIN']}. The filename and the contents disagree.")
    print(f"  {p}: force_gt=False ost_pairs=False, margin {m['margin']} contact {m['contact']}")
PYCHK
fi

say "=== $RID epoch $EP, $SUF crop at margin $MARGIN, cards $CARDS ==="
DRY_FLAG=""
[ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"
mkdir -p "$OUT"
LOG=$OUT/eval_${SUF}.log
# THE FINGERPRINT IS READ BACK AND HANDED TO THE REPORT, which refuses to choose a file by
# modification time. Without this the report could print one variant's numbers under the other's
# heading, which is exactly what it did before review. pipefail so a failing eval is not masked
# by tee succeeding.
set -o pipefail
# shellcheck disable=SC2086
$PY -I run_logging/cropped_eval.py --bundle "$BUNDLE" \
  --insample-crop-tokens "$IS_MASK" --valfull-crop-tokens "$VF_MASK" \
  --cards "$CARDS" --out "$OUT" $DRY_FLAG 2>&1 | tee "$LOG" \
  || { echo "STOP: cropped_eval failed."; exit 1; }
set +o pipefail
CROP_ID=$(grep -oE '^CROP_ID=[0-9a-f]{8}$' "$LOG" | head -1 | cut -d= -f2)
[ -n "$CROP_ID" ] || { echo "STOP: no crop fingerprint in $LOG, so the report cannot be bound to "\
  "this crop. Refusing to report rather than reading whichever file is newest."; exit 1; }
say "crop fingerprint $CROP_ID"
[ "${DRY:-0}" = "1" ] && exit 0

# THE TABLE. Cropped ours-vs-shipped beside the UNCROPPED pair from the run's own metric files,
# because the comparison that matters is not the cropped number -- it is the DIVERGENCE. The
# launcher for this arm names that divergence as the failure to watch: in-sample climbing on
# cropped inputs while uncropped stalls is the signature of learning to rank crops rather than
# poses, which is exactly what the ground-truth-cropped arm did at 0.6305 cropped / 0.5387
# uncropped.
say "=== results ==="
$PY -I run_logging/cropped_eval_report.py --out "$OUT" --rid "$RID" --epoch "$EP" \
  --crop-id "$CROP_ID" --variant "$SUF crop, margin $MARGIN"
