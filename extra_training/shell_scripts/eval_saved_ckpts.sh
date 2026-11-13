#!/bin/bash
# SCORE CHECKPOINTS THAT ALREADY EXIST, instead of retraining to produce them again.
#
#     ./shell_scripts/eval_saved_ckpts.sh <rid> [<rid2> ...]
#
# WHY. The 1e-3 arm (fshpzx epochs 1-13, 7mo52r 14-18) saved a head every epoch but was evaluated
# only every 5th, then every 2nd -- 5 of 18. A fresh 1e-3 arm was launched to get the per-epoch
# curve and was a bit-identical repeat of training we had already paid for: same lr, alpha,
# structs-per-step, dropout, crop allowlist, rel labels, and no resume. The checkpoints are the
# training; only the eval was missing. This scores the gaps.
#
# THE COMMAND IS COPIED FROM THE TRAINER, not reconstructed. phase_k_train builds its train-eval
# as phase_k_eval_sharded with --cards/--ckpt/--budget/--work/--out and an --extra string holding
# --fold, --results, --pool, --roles, --seed-base, --seed-manifest, --join-cache,
# --pair-iptm-only, --prefetch, --pool-mean, --select-by mean, plus the pose flags the head needs
# (--pose-perblock --pose-perblock-copy --frame-direction). A hand-built argv here would produce a
# number that is not comparable with the ones the run wrote itself, which is the whole point.
#
# IT SKIPS WHAT IS ALREADY SCORED, so it is safe to re-run and resumes after an interruption.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

CARDS="${CARDS:-4,5,7}"
POOL="${POOL:-training/valfull83/pool.csv}"
FOLD="${FOLD:-validation/fold}"
BUDGET="${BUDGET:-30}"
SEED_BASE="${SEED_BASE:-400000}"
PRD="${PRD:-training/valfull83}"

[ $# -ge 1 ] || { echo "usage: eval_saved_ckpts.sh <rid> [<rid2> ...]"; exit 2; }

# ORDER: an explicit epoch sequence, highest-value first, instead of ascending.
#
# Ascending spends the first hours on epochs 1-4, which the curve already brackets, while the
# interesting region -- where the arm peaked at 16 and gave it back by 18 -- waits until the end.
# ORDER lets the odd epochs near the peak be scored first, so an interrupted sweep still answers
# the question it was run for. Epochs not named are scored afterwards in ascending order, so
# nothing is dropped by naming a partial list.
ORDER="${ORDER:-}"

EXTRA="--fold $R/$FOLD --results $R/$FOLD/results.csv --pool $R/$POOL"
EXTRA="$EXTRA --roles $R/$FOLD/chain_roles_all83.csv --seed-base $SEED_BASE"
[ -f "$R/$PRD/seed_manifest.json" ] && EXTRA="$EXTRA --seed-manifest $R/$PRD/seed_manifest.json"
EXTRA="$EXTRA --pose-perblock --pose-perblock-copy --frame-direction"
# --join-cache, WITHOUT WHICH EVERY PASS RE-DERIVES A JOIN THAT CANNOT HAVE CHANGED.
# The pose->sample mapping comes from the trunk coordinates and the pose cifs, neither of which
# moves between epochs; phase_k_eval's own help measures re-deriving it at 3.6 min per shard.
# Omitting it here made the first checkpoint take 15 min against the trainer's 11, which is the
# whole difference. The trainer derives this path as <dirname of the pool>/join_cache.json, and
# the sharded driver appends _shard<N>; the files already exist beside the pool with 2,211
# entries, so the first pass reuses them at 100%.
EXTRA="$EXTRA --join-cache $R/$(dirname "$POOL")/join_cache.json"
EXTRA="$EXTRA --pair-iptm-only --prefetch --pool-mean --select-by mean"

for RID in "$@"; do
    D="training/runs/$RID"
    [ -d "$D" ] || { say "no such run: $D"; continue; }
    # SPACE-SEPARATED, NOT NEWLINE. `sort -n` returns one epoch per line, and the `case " $HAVE "
    # in *" $E "*` tests below then compare against "\n17\n" and never match -- so ORDER was
    # silently ignored and the sweep ran ascending anyway. tr makes the membership test mean what
    # it reads as.
    HAVE=$(ls "$D"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tr '\n' ' ')
    SEQ=""
    for E in $ORDER; do
        case " $HAVE " in *" $E "*) SEQ="$SEQ $E";; esac
    done
    for E in $HAVE; do
        case " $SEQ " in *" $E "*) ;; *) SEQ="$SEQ $E";; esac
    done
    for EP in $SEQ; do
        CP=$(ls "$D"/head_ep${EP}_*.pt 2>/dev/null | head -1)
        [ -n "$CP" ] || continue
        MJ="$D/metric_valfull83_ep${EP}.json"
        if [ -s "$MJ" ]; then
            say "$RID ep$EP already scored, skipping"
            continue
        fi
        B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
        [ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
        say "$RID ep$EP -> $MJ"
        LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
            --cards "$CARDS" --ckpt "$R/$CP" --budget "$BUDGET" \
            --work "$D/valfull83_eval_shards_ep${EP}" --out "$R/$MJ" \
            --extra "$EXTRA" > "$D/eval_saved_ep${EP}.log" 2>&1
        rc=$?
        if [ "$rc" != "0" ] || [ ! -s "$MJ" ]; then
            say "FAILED on $RID ep$EP (exit $rc); tail:"
            tail -5 "$D/eval_saved_ep${EP}.log" | sed 's/^/    /'
            exit 4
        fi
        $PY -I -c "
import json
d = json.load(open('$MJ'))
print('    dockq_max %.4f  frac_max %.4f  dockq_mean %.4f'
      % (d['mean_dockq_select_max'], d['frac_acc_select_max'], d['mean_dockq']))"
    done
done
say "ALL REQUESTED CHECKPOINTS SCORED"
