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
# NAME SEPARATES ONE POOL'S OUTPUTS FROM ANOTHER'S. metric_valfull83_ep<N>.json and
# valfull83_eval_shards_ep<N>/ were hardcoded, so scoring ANY other pool with this script would
# have overwritten the held-out numbers of the run it was pointed at -- the same class of mistake
# as the five metric_train files that once held the only held-out numbers a run produced. It
# defaults to valfull83, so every existing invocation keeps its paths.
NAME="${NAME:-valfull83}"
# THE ROLES TABLE BELONGS TO THE POOL, not to the fold. The fold's own chain_roles.csv names 501
# structures and the full training pool spans 853, which leaves 356 of them unmappable; the
# trainer solves this by preferring <pool dir>/chain_roles.csv and so does this.
ROLES="${ROLES:-$PRD/chain_roles.csv}"
[ -f "$R/$ROLES" ] || ROLES="$FOLD/chain_roles_all83.csv"
# --select-by: THE TRAINER PICKS IT FROM THE ARM'S OBJECTIVE. A PAE-only arm (no ranking loss)
# selects by the MAX and reports both poolings; --select-by mean here would answer a different
# question from the run's own in-training passes and the two would not be comparable.
SELECT="${SELECT:-}"
# MIXED=1 for a pool whose folded depth is not uniform -- see phase_k_eval's --mixed-budget.
MIXED="${MIXED:-}"

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
EXTRA="$EXTRA --roles $R/$ROLES --seed-base $SEED_BASE"
if [ -f "$R/$PRD/seed_manifest.json" ]; then
    EXTRA="$EXTRA --seed-manifest $R/$PRD/seed_manifest.json"
    # DOUBLE-LOCKED, same as in the trainer: phase_k_eval stops when a mixed:true manifest
    # arrives without the flag and when the flag arrives without one, so neither half of a
    # mixed-depth pass can be reached by accident.
    [ -n "$MIXED" ] && EXTRA="$EXTRA --mixed-budget"
elif [ -n "$MIXED" ]; then
    echo "STOP: MIXED=1 but no $R/$PRD/seed_manifest.json; mixed depth is defined only there."
    exit 2
fi

# --join-cache, WITHOUT WHICH EVERY PASS RE-DERIVES A JOIN THAT CANNOT HAVE CHANGED.
# The pose->sample mapping comes from the trunk coordinates and the pose cifs, neither of which
# moves between epochs; phase_k_eval's own help measures re-deriving it at 3.6 min per shard.
# Omitting it here made the first checkpoint take 15 min against the trainer's 11, which is the
# whole difference. The trainer derives this path as <dirname of the pool>/join_cache.json, and
# the sharded driver appends _shard<N>; the files already exist beside the pool with 2,211
# entries, so the first pass reuses them at 100%.
EXTRA="$EXTRA --join-cache $R/$(dirname "$POOL")/join_cache.json"
EXTRA="$EXTRA --pair-iptm-only --prefetch --pool-mean"
[ -n "$SELECT" ] && EXTRA="$EXTRA --select-by $SELECT"

for RID in "$@"; do
    D="training/runs/$RID"
    [ -d "$D" ] || { say "no such run: $D"; continue; }
    # THE PATHWAY COMES FROM THE RUN, NOT FROM A CONSTANT. These three flags were hardcoded for the
    # arms that had pose injection sites; a NATIVE-head arm (Arm S, Arm L: no pose sites at all) would
    # then be scored with three modules its checkpoint has no weights for, which either refuses the
    # load or evaluates a head the training never had. Read from the run's own args.json, falling back
    # to the old constant when a run predates those fields.
    # NOT INDENTED, AND THAT IS THE WHOLE POINT. This heredoc body used to be indented four
    # spaces to match the shell around it, which is an IndentationError -- python never reached
    # the try, printed nothing, and PATHWAY came back EMPTY from the error path. Empty happens to
    # be correct for a native-head arm, which is why it survived: the first arm with pose
    # injection sites would have been scored with none of them and the state-dict guard would
    # have refused the checkpoint. A heredoc is not inside the shell's block structure.
    PATHWAY=$($PY -I - "$D/args.json" <<'PYPATH'
import json, sys
try:
    j = json.load(open(sys.argv[1])); a = j.get("args", j) or {}
except (OSError, ValueError):
    print("--pose-perblock --pose-perblock-copy --frame-direction"); raise SystemExit
f = []
if a.get("pose_perblock"): f.append("--pose-perblock")
if a.get("pose_perblock_copy"): f.append("--pose-perblock-copy")
if a.get("frame_direction"): f.append("--frame-direction")
if a.get("pose_readout"): f.append("--pose-readout")
if a.get("pose_readout_prenorm"): f.append("--pose-readout-prenorm")
if a.get("pose_readout_post_zero"): f.append("--pose-readout-post-zero")
# ONLY FLAGS phase_k_eval ACTUALLY HAS. --pose-block-init and --pose-readout-init are TRAINER
# options -- they say how an injected layer is initialised, which is meaningless to an evaluator
# loading trained weights -- and the evaluator has no such arguments. They were in this list
# anyway, and args.json records them as 'zero' for EVERY arm including ones with no injection at
# all, so the first pass that actually ran this block died with
#     phase_k_eval.py: error: unrecognized arguments: --pose-block-init zero --pose-readout-init zero
# on all three shards. That it had never fired before is the only reason it survived: the heredoc
# above used to be indented into an IndentationError, so this loop never ran.
#
# THE SURVIVING THREE ARE CHECKED AGAINST THE EVALUATOR'S OWN PARSER, not assumed:
#   grep -oE '"--pose[a-z-]*"' phases/phase_k_eval.py
# lists --pose-block-terms, --pose-readout-sites and --pose-readout-terms among its options and
# no *-init at all. An empty value is skipped, so a native-head arm still contributes nothing.
for k, flag in (("pose_block_terms","--pose-block-terms"),
                ("pose_readout_sites","--pose-readout-sites"),
                ("pose_readout_terms","--pose-readout-terms")):
    v = a.get(k)
    if v: f += [flag, str(v)]
print(" ".join(f))
PYPATH
)
    EXTRA_R="$EXTRA $PATHWAY"
    # SPACE-SEPARATED, NOT NEWLINE. `sort -n` returns one epoch per line, and the `case " $HAVE "
    # in *" $E "*` tests below then compare against "\n17\n" and never match -- so ORDER was
    # silently ignored and the sweep ran ascending anyway. tr makes the membership test mean what
    # it reads as.
    HAVE=$(ls "$D"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tr '\n' ' ')
    SEQ=""
    for E in $ORDER; do
        case " $HAVE " in *" $E "*) SEQ="$SEQ $E";; esac
    done
    # ONLY=1 SCORES JUST THE NAMED EPOCHS. Without it the loop appends every other saved epoch,
    # so asking for two checkpoints silently queued nine and took the cards for hours.
    if [ -z "${ONLY:-}" ]; then
        for E in $HAVE; do
            case " $SEQ " in *" $E "*) ;; *) SEQ="$SEQ $E";; esac
        done
    fi
    for EP in $SEQ; do
        CP=$(ls "$D"/head_ep${EP}_*.pt 2>/dev/null | head -1)
        [ -n "$CP" ] || continue
        MJ="$D/metric_${NAME}_ep${EP}.json"
        if [ -s "$MJ" ]; then
            say "$RID ep$EP already scored, skipping"
            continue
        fi
        B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
        [ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
        say "$RID ep$EP -> $MJ"
        LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
            --cards "$CARDS" --ckpt "$R/$CP" --budget "$BUDGET" \
            --work "$D/${NAME}_eval_shards_ep${EP}" --out "$R/$MJ" \
            --extra "$EXTRA_R" > "$D/eval_saved_${NAME}_ep${EP}.log" 2>&1
        rc=$?
        if [ "$rc" != "0" ] || [ ! -s "$MJ" ]; then
            say "FAILED on $RID ep$EP (exit $rc); tail:"
            tail -5 "$D/eval_saved_${NAME}_ep${EP}.log" | sed 's/^/    /'
            exit 4
        fi
        # THE KEYS DEPEND ON --select-by, SO THEY ARE READ DEFENSIVELY. With --select-by mean the
        # evaluator writes mean_dockq_select_max alongside the selected column; with the default
        # max selection it writes mean_dockq / frac_acc for the selection and
        # mean_dockq_select_mean for the other pooling, and there is no *_select_max key at all.
        # This line asked for it unconditionally, so every pass run at the default selector ended
        # in `KeyError: 'mean_dockq_select_max'` AFTER writing a perfectly good metric file --
        # noise that looks like a failed pass and hides a real one.
        $PY -I -c "
import json
d = json.load(open('$MJ'))
sel = d.get('select_by', 'max')
mx = d.get('mean_dockq_select_max', d.get('mean_dockq'))
fx = d.get('frac_acc_select_max', d.get('frac_acc'))
mn = d.get('mean_dockq_select_mean', d.get('mean_dockq'))
fn = d.get('frac_acc_select_mean', d.get('frac_acc'))
print('    n=%s  select_by=%s  max %.4f / %.1f%%   mean %.4f / %.1f%%   shipped %.4f / %.1f%%'
      % (d.get('n_interfaces', '?'), sel, mx, 100*fx, mn, 100*fn,
         d.get('baseline_mean_dockq', float('nan')),
         100*d.get('baseline_frac_acc', float('nan'))))"
    done
done
say "ALL REQUESTED CHECKPOINTS SCORED"
