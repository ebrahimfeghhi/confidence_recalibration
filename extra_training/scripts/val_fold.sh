#!/bin/bash
# Fold the validation set and cache what the confidence head needs, then hand the GPUs back to PXM.
#
#   val_fold.sh smoke   one complex, one seed -- proves the job json, the trunk dump and OST DockQ
#                       all work before 20 h of GPU is committed. Does NOT touch the PXM sweep.
#   val_fold.sh run     pause PXM, fold 79 structures x 20 seeds, resume PXM.
#
# WHY A PAUSE RATHER THAN SHARING: GPUs 0 and 1 are the only cards free to us (2-7 belong to other
# users), and both the fold workers and this want a whole card.
#
# WHAT IS SAVED per (structure, seed): the trunk dump z/s as bf16 bits -- bf16 because that is what
# the model runs in (--dtype bf16; docs line 110 call it the default) and because the confidence head
# runs inside autocast, so bf16 z is what it would see anyway. NOT fp16: a float16 trunk dump cost us
# a good pose once, 0.495 -> 0.056.
#
# SCORING is folder.py's default scorer via --dataset sabdab_val, which resolves DockQ references to
# extra_training/val_refs/<pdb>.cif through paths.py. OST does the chain mapping; nothing here
# hand-rolls structure code.
set -u
MODE="${1:-}"
R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
C=$R/diffusion_commit_clean_v2
E=$R/extra_training/val_fold
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
LOG=$R/extra_training/val_fold.log
SEEDS=20
GPUS="0 1"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

[ -n "$MODE" ] || die "usage: val_fold.sh smoke|run"
mkdir -p "$E" "$E/trunks" "$E/poses"

NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 1 ] || die "no validation job jsons -- run build_val_jobs.py after the MSAs exist"

# ---------------------------------------------------------------- queue
# One line per (complex, seed). folder.py keys work units on (pdb, tag, seed) and claims them with
# atomic mkdir, so two workers share one queue without coordination.
build_queue() {
    local n_seeds=$1 out=$2
    : > "$out"
    for f in $R/extra_training/val_jobs/*.json; do
        local pdb; pdb=$(basename "$f" .json)
        for i in $(seq 0 $((n_seeds-1))); do
            printf 'free\t%s\t*\tfree\t%d\n' "$pdb" $((400000+i)) >> "$out"
        done
    done
    wc -l < "$out"
}

run_workers() {
    local queue=$1 tag=$2
    for g in $GPUS; do
        # TRUNK_DUMP_DIR, *not* PROTENIX_DUMP_TRUNK: folder.py:831 turns the former into a per-fold
        # path <dir>/<pdb>/<tag>_<seed>.npz and refuses to overwrite an existing one. Pointing
        # PROTENIX_DUMP_TRUNK at a directory instead makes protenix.py:1362 write <dir>/trunk.npz for
        # every fold -- one file, overwritten 1,579 times.
        # PROTENIX_DUMP_CONF_INPUTS=1 adds what re-running the CONFIDENCE HEAD offline needs:
        # s_inputs, s_trunk in f32, the poses in model atom order, contact_probs and the six input
        # features the head and the ipTM summary read. ~6 MB per fold on top of z.
        tmux new-session -d -s vf_${tag}_$g "cd $P && CUDA_VISIBLE_DEVICES=$g \
          LAYERNORM_TYPE=openfold \
          TRUNK_DUMP_DIR=$E/trunks PROTENIX_DUMP_TRUNK_BF16=1 PROTENIX_DUMP_CONF_INPUTS=1 \
          $PY $C/folder.py --queue $queue --out $E/poses --csv $E/results.csv \
          --claim-dir $E/claims --dataset sabdab_val --ncycle 10 --nsample 5 \
          --dtype bf16 --overlap-scoring >> $E/w$g.log 2>&1"
        say "  launched $tag worker on GPU $g"
    done
    local last=-1 tick=0
    while :; do
        local alive=0
        for g in $GPUS; do tmux has-session -t vf_${tag}_$g 2>/dev/null && alive=1; done
        [ "$alive" -eq 0 ] && break
        tick=$((tick+1))
        if [ $((tick % 15)) -eq 0 ]; then
            local n; n=$(done_units)
            say "  progress: $n units folded$([ "$n" -eq "$last" ] && echo '  (no change in 15 min)')"
            last=$n
        fi
        sleep 60
    done
}

# How many (complex, seed) units are finished: folder.py writes one run.json per unit and skips
# units already in results.csv at startup, so this is also what a relaunch would resume from.
done_units() { find $E/poses -name run.json 2>/dev/null | wc -l; }

if [ "$MODE" = "smoke" ]; then
    say "=== SMOKE: one complex, one seed (PXM sweep untouched) ==="
    SQ=$E/queue_smoke.tsv
    head -1 <(for f in $R/extra_training/val_jobs/*.json; do
                  printf 'free\t%s\t*\tfree\t400000\n' "$(basename "$f" .json)"; done) > $SQ
    say "queue: $(cat $SQ)"
    run_workers "$SQ" smoke
    say "-- did it produce a pose?"; find $E/poses -name '*.cif' | wc -l | sed 's/^/   cifs: /' | tee -a "$LOG"
    say "-- did it dump a trunk?"; find $E/trunks -name '*.npz' -printf '   %p %s bytes\n' 2>/dev/null | tee -a "$LOG"
    say "-- did the default scorer give DockQ?"; head -3 $E/results.csv 2>/dev/null | tee -a "$LOG"
    say "=== SMOKE DONE. Inspect the three above, then: val_fold.sh run ==="
    exit 0
fi

[ "$MODE" = "run" ] || die "unknown mode '$MODE'"

# ---------------------------------------------------------------- pause PXM
say "=== RUN: pausing the PXM sweep ==="
say "pxm folds before pause: $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null)"
for g in $GPUS; do
    if tmux has-session -t fx_pxm2025_$g 2>/dev/null; then
        # kill the worker by PID, never by pattern: a pattern kill matches this script too
        for pid in $(tmux list-panes -t fx_pxm2025_$g -F '#{pane_pid}' 2>/dev/null); do
            for child in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
                say "  kill PID $child (pxm worker on GPU $g)"; kill "$child" 2>/dev/null
            done
        done
        sleep 5; tmux kill-session -t fx_pxm2025_$g 2>/dev/null
    fi
done
# the watcher would relaunch them straight into our GPUs
tmux kill-session -t fx_watch 2>/dev/null && say "  stopped fx_watch so it cannot relaunch PXM"
sleep 10
say "releasing claims the killed workers held"
$PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -2 | tee -a "$LOG"

# ---------------------------------------------------------------- fold validation
Q=$E/queue.tsv
TARGET=$((NJOB*SEEDS))
say "validation queue: $(build_queue $SEEDS $Q) units ($NJOB complexes x $SEEDS seeds)"

# ROUNDS, not one shot. A worker that dies (OOM, a bad cif, a node hiccup) ends its tmux session and
# run_workers returns with the queue unfinished. folder.py reconciles stale claims at startup and
# skips units already in results.csv, so relaunching simply continues. The loop stops when the target
# is met, or when a whole round adds nothing -- that is a real fault, not a transient one.
for round in 1 2 3 4 5; do
    before=$(done_units)
    [ "$before" -ge "$TARGET" ] && { say "all $TARGET units already folded"; break; }
    say "round $round: $before/$TARGET units done, launching workers"
    run_workers "$Q" val
    after=$(done_units)
    say "round $round ended at $after/$TARGET units (+$((after-before)))"
    [ "$after" -ge "$TARGET" ] && break
    if [ "$after" -le "$before" ]; then
        say "ESCALATE: round $round folded nothing. Stopping rounds; see $E/w0.log and $E/w1.log"
        break
    fi
done
say "validation folding finished"
say "  poses: $(find $E/poses -name '*.cif' 2>/dev/null | wc -l)"
say "  trunk dumps: $(find $E/trunks -name '*.npz' 2>/dev/null | wc -l)"
say "  scored rows: $(($(wc -l < $E/results.csv 2>/dev/null || echo 1)-1))"
say "  cache size: $(du -sh $E/trunks 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------- resume PXM
say "=== resuming the PXM sweep ==="
$PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -1 | tee -a "$LOG"
for g in $GPUS; do
    tmux new-session -d -s fx_pxm2025_$g "cd $P && CUDA_VISIBLE_DEVICES=$g LAYERNORM_TYPE=openfold \
      $PY $C/folder.py --queue $PXM/queue.tsv --out $PXM/poses --csv $PXM/results.csv \
      --claim-dir $PXM/claims --dataset pxm2025 --ncycle 10 --nsample 5 --dtype bf16 \
      --overlap-scoring >> $PXM/w$g.log 2>&1"
    say "  relaunched pxm worker on GPU $g"
done
tmux new-session -d -s fx_watch "bash $C/watch_free_extra10.sh" && say "  restarted fx_watch"
say "=== done ==="
