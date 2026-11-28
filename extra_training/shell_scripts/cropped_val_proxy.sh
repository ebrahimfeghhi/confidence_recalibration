#!/bin/bash
# SCORE SEVERAL CHECKPOINTS ON THE CROPPED VAL POOL, to test whether a cheap proxy ranks them the
# way the uncropped metric does.
#
#     ./shell_scripts/cropped_val_proxy.sh <rid>:<ep> [<rid>:<ep> ...]
#
# WHY. An uncropped valfull83 pass is ~13 min on three cards, so a 12-epoch arm with per-epoch
# evals spends as long evaluating as training. The crops in validation/crop_final were selected so
# each one's DockQ regret against the UNCROPPED selection is within 0.05, which is exactly the
# property a ranking proxy needs: if the cropped pass picks the same checkpoint, it can choose
# which epochs are worth an uncropped pass.
#
# THE TEST IS AGAINST KNOWN ANSWERS. Every checkpoint named here already has an uncropped number,
# so this is falsifiable: if the cropped ranking disagrees, the proxy is not usable and the
# honest outcome is to say so rather than to adopt it because it was cheap.
#
# ONE CROP, FIXED, FOR EVERY UNIT AND EVERY CHECKPOINT (--crop-index 0). The trainer rotates
# `epoch % n` because it wants a different view each epoch; an evaluator must not, or two
# checkpoints would be scored on different inputs and the difference between them would be of the
# crops as much as of the heads.
#
# NOTHING HERE IS COMPARABLE WITH AN UNCROPPED NUMBER. The outputs are named _crop so they cannot
# be read into the uncropped series by accident.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

CARDS="${CARDS:-4,5,7}"
POOL="${POOL:-training/valfull83/pool.csv}"
FOLD="${FOLD:-validation/fold}"
CROPS="${CROPS:-validation/crop_final}"
CROP_INDEX="${CROP_INDEX:-0}"
BUDGET="${BUDGET:-30}"

[ $# -ge 1 ] || { echo "usage: cropped_val_proxy.sh <rid>:<ep> [...]"; exit 2; }
[ -d "$CROPS" ] || { say "REFUSING: $CROPS is not a directory"; exit 3; }

EXTRA="--fold $R/$FOLD --results $R/$FOLD/results.csv --pool $R/$POOL"
EXTRA="$EXTRA --roles $R/$FOLD/chain_roles_all83.csv --seed-base 400000"
EXTRA="$EXTRA --seed-manifest $R/training/valfull83/seed_manifest.json"
EXTRA="$EXTRA --pose-perblock --pose-perblock-copy --frame-direction"
EXTRA="$EXTRA --crop-pool $R/$CROPS --crop-index $CROP_INDEX"
# NO --join-cache. The join is by coordinate identity against the UNCROPPED coordinates, so the
# cache built by the uncropped passes is still valid -- but writing to it from a cropped run
# would mix provenances in one file for no gain. Deriving it costs ~3.6 min a shard, which is
# the price of keeping the uncropped cache untouched.
EXTRA="$EXTRA --pair-iptm-only --prefetch --pool-mean --select-by mean"

for SPEC in "$@"; do
    RID="${SPEC%%:*}"; EP="${SPEC##*:}"
    CP=$(ls "training/runs/$RID/head_ep${EP}_"*.pt 2>/dev/null | head -1)
    [ -n "$CP" ] || { say "no checkpoint for $RID ep$EP"; exit 4; }
    MJ="training/runs/$RID/metric_valfull83_crop_ep${EP}.json"
    if [ -s "$MJ" ]; then
        say "$RID ep$EP already scored cropped, skipping"
        continue
    fi
    B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    [ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
    say "$RID ep$EP (cropped) -> $MJ"
    LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
        --cards "$CARDS" --ckpt "$R/$CP" --budget "$BUDGET" \
        --work "training/runs/$RID/valcrop_shards_ep${EP}" --out "$R/$MJ" \
        --extra "$EXTRA" > "training/runs/$RID/eval_crop_ep${EP}.log" 2>&1
    rc=$?
    if [ "$rc" != "0" ] || [ ! -s "$MJ" ]; then
        say "FAILED on $RID ep$EP (exit $rc):"
        tail -6 "training/runs/$RID/eval_crop_ep${EP}.log" | sed 's/^/    /'
        exit 5
    fi
    $PY -I -c "
import json
d = json.load(open('$MJ'))
print('    CROPPED dockq_max %.4f  frac_max %.4f  dockq_mean %.4f'
      % (d['mean_dockq_select_max'], d['frac_acc_select_max'], d['mean_dockq']))"
done
say "ALL CROPPED PASSES DONE"
