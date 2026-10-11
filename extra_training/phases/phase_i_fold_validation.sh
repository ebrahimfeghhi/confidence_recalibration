#!/bin/bash
# PHASE I -- fold the validation set and cache everything needed to re-run the confidence head.
#
#   phase_i_fold_validation.sh smoke    one complex, one seed. Asserts the mechanism in code, not by
#                                       eye. Does NOT pause the PXM sweep.
#   phase_i_fold_validation.sh run      pause PXM, fold every structure x every seed, resume PXM,
#                                       then verify. Refuses to start until smoke has passed.
#   phase_i_fold_validation.sh verify   the post-fold checks on their own.
#
# WHAT IS CACHED per (structure, seed), and why each piece. The point of this phase is not the poses:
# it is a cache complete enough to re-run Protenix's confidence head offline, over and over, without
# touching a GPU again. So alongside the pose and its scores we keep:
#     z_trunk, s_trunk              the pair and single representations the head reads
#     ci_s_inputs, ci_s_trunk_f32   its other two inputs
#     ci_coordinate                 the pose in MODEL atom order, not the cif's
#     ci_contact_probs              what the ipTM summary is computed from
#     ci_feat_*                     the six input features the head and the summary need
#     the RNG state                 python, torch and cuda, so a rollout can be reproduced
#
# bf16, NOT fp16. The model runs in bf16 (--dtype bf16) and the head runs inside autocast, so bf16 is
# what it would see anyway. The distinction is not cosmetic: bf16 has fp32's exponent range and fp16
# does not, and a float16 trunk dump once cost a good pose outright -- 8vqf seed 300048 went from
# DockQ 0.495 to 0.056.
#
# TRUNK_DUMP_DIR, NOT PROTENIX_DUMP_TRUNK. folder.py:831 turns the former into a per-fold path
# <dir>/<pdb>/<tag>_<seed>.npz and refuses to overwrite an existing one. Pointing the latter at a
# directory instead makes protenix.py:1362 write <dir>/trunk.npz for EVERY fold -- one file,
# overwritten 1,580 times, and a smoke test that only counts files would pass.
#
# SCORING is folder.py's own scorer via --dataset sabdab_val, which resolves DockQ references to
# extra_training/val_refs/<pdb>.cif through paths.py. OST does the chain mapping and the DockQ;
# nothing here hand-rolls structure code.
#
# TEMPLATES ARE REQUIRED, by decision. pxm2025 folds use them and the validation set is the arm
# compared against pxm2025, so it is built the same way. The consequence to state in the write-up is
# that FoldBench folds do not use templates, so a head calibrated here is evaluated there without
# them. Phase H must have run: the gate below refuses to fold without hmmsearch.a3m.
#
# ENVIRONMENT
#     GPUS="0 1"   which cards to use. The project ceiling is 4 and the allocation is the user's
#                  call, so this is never widened automatically.
#     SEEDS=20     seeds per structure.
#     ROUNDS=5     how many times to relaunch workers that died mid-queue.
#     NSAMPLE=5    poses per (structure, seed). The smoke test asserts the cache holds this many.
#     STALL_MIN=90 kill the workers if no unit completes for this many minutes. This phase holds the
#                  GPUs away from the PXM sweep, so a hung worker costs more than a lost round.
#     MAX_WAIT_MIN=2400  hard ceiling on one round's wall clock.
set -u
MODE="${1:-}"
R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
C=$R/diffusion_commit_clean_v2
E=$R/extra_training/val_fold
J=$R/extra_training/val_jobs
MSA=$R/extra_training/msa_work/mmcif_msa
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
LOG=$R/extra_training/phase_i.log
GPUS="${GPUS:-0 1}"
SEEDS="${SEEDS:-20}"
NSAMPLE="${NSAMPLE:-5}"
ROUNDS="${ROUNDS:-5}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

case "$MODE" in smoke|run|verify) ;; *) die "usage: phase_i_fold_validation.sh smoke|run|verify" ;; esac
mkdir -p "$E" "$E/trunks" "$E/poses"

# ---------------------------------------------------------------- inputs, checked before any GPU
# EVERY INPUT VERIFIED BEFORE 16 HOURS OF GPU IS COMMITTED, and by content rather than by count. A
# job json whose templatesPath or MSA path does not exist produces a fold that runs and scores
# nothing, which is the most expensive way to discover a missing file.
check_inputs() {
    NJOB=$(ls $J/*.json 2>/dev/null | wc -l)
    [ "$NJOB" -ge 1 ] || die "no job jsons in $J -- phase G has not run"
    POOL=$E/pool.csv
    [ -s "$POOL" ] || die "no $POOL -- folder.py's interfaces_by_pdb() reads it, and without it the
     fold runs and scores NOTHING"
    NPDB=$(tail -n +2 "$POOL" | cut -d, -f1 | sort -u | wc -l)
    NIF=$(($(wc -l < "$POOL")-1))
    [ "$NPDB" -eq "$NJOB" ] || die "pool.csv covers $NPDB structures but there are $NJOB jsons"
    # Dangling paths, by opening each json. Templates are required by decision, see the header.
    BAD=$($PY - "$J" <<'PYCHK'
import glob, json, os, sys
bad = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    for task in json.load(open(f)):
        for s in task["sequences"]:
            c = s.get("proteinChain")
            if not c:
                continue
            for key in ("pairedMsaPath", "unpairedMsaPath", "templatesPath"):
                p = c.get(key)
                if not p or not os.path.exists(p):
                    bad.append(f"{os.path.basename(f)} {key}={p}")
                    continue
                # THE SELF-TEMPLATING MARKER, and this is a structural check rather than a
                # procedural one. Phase H writes <templatesPath>.filtered holding the cutoff it
                # applied; without it the template file has never been date-filtered, and a
                # structure deposited before 2021-09-30 may be holding its own entry at 100%
                # identity. 46 of the 79 validation structures and 1,354 of the 2,886 training-pool
                # entries are in that window, so no fold -- validation or training -- may run on an
                # unmarked file.
                if key == "templatesPath" and not os.path.exists(p + ".filtered"):
                    bad.append(f"{os.path.basename(f)} templates not date-filtered: {p}")
print(len(bad))
for b in bad[:5]:
    print("  " + b)
PYCHK
)
    NBAD=$(printf '%s' "$BAD" | head -1)
    if [ "${NBAD:-1}" -ne 0 ]; then
        say "$BAD"
        die "$NBAD chain paths are missing or unfiltered. Templates are required by decision, so
     phase H must have run -- and its date filter must have run too: a template file with no
     .filtered marker may hand a pre-2021 structure its own entry at 100% identity. A dangling path
     folds without the file and nothing reports it."
    fi
    NTMPL=$(find "$MSA" -name hmmsearch.a3m 2>/dev/null | wc -l)
    say "inputs OK: $NJOB jsons, $NIF interfaces over $NPDB structures, $NTMPL hmmsearch.a3m, no dangling paths"
    TARGET=$((NJOB * SEEDS))
}

# How many (structure, seed) units are finished. folder.py writes one run.json per unit and skips
# units already in results.csv at startup, so this is also what a relaunch resumes from.
done_units() { find $E/poses -name run.json 2>/dev/null | wc -l; }

# in: a csv, a column NAME.  out: how many rows carry a numeric value there; exit 2 if no such column
csv_numeric() {
    awk -F, -v want="$2" '
        NR==1 { for (i=1; i<=NF; i++) if ($i == want) c = i; if (!c) exit 2; next }
        c && $c+0 == $c && $c != "" { n++ }
        END { print n+0 }' "$1"
}

build_queue() {
    local n=$1 out=$2
    : > "$out"
    for f in $J/*.json; do
        local pdb; pdb=$(basename "$f" .json)
        for i in $(seq 0 $((n-1))); do
            printf 'free\t%s\t*\tfree\t%d\n' "$pdb" $((400000+i)) >> "$out"
        done
    done
    wc -l < "$out"
}

run_workers() {
    local queue=$1 tag=$2 g last tick alive n
    for g in $GPUS; do
        tmux new-session -d -s vf_${tag}_$g "cd $P && CUDA_VISIBLE_DEVICES=$g \
          LAYERNORM_TYPE=openfold \
          TRUNK_DUMP_DIR=$E/trunks PROTENIX_DUMP_TRUNK_BF16=1 PROTENIX_DUMP_CONF_INPUTS=1 \
          $PY $C/folder.py --queue $queue --out $E/poses --csv $E/results.csv \
          --claim-dir $E/claims --dataset sabdab_val --ncycle 10 --nsample "$NSAMPLE" \
          --dtype bf16 --overlap-scoring >> $E/w$g.log 2>&1"
        say "  launched $tag worker on GPU $g"
    done
    # AN UPPER BOUND ON THE WAIT, which matters here more than anywhere else in the pipeline: this
    # is the only phase that takes the GPUs away from another job, so a hung worker does not merely
    # stall this run, it keeps PXM down for as long as it hangs. Two bounds -- total elapsed, and
    # time since the last completed unit -- and on either the workers are killed BY PID so the round
    # ends, the rounds loop sees no progress, and the trap restores PXM.
    last=-1; tick=0; stall=0
    while :; do
        alive=0
        for g in $GPUS; do tmux has-session -t vf_${tag}_$g 2>/dev/null && alive=1; done
        [ "$alive" -eq 0 ] && break
        tick=$((tick+1)); stall=$((stall+1))
        n=$(done_units)
        if [ "$n" -ne "$last" ]; then stall=0; last=$n; fi
        if [ $((tick % 15)) -eq 0 ]; then
            say "  progress: $n units$([ "$stall" -ge 15 ] && echo "   (no new unit for $stall min)")"
        fi
        if [ "$stall" -ge "${STALL_MIN:-90}" ] || [ "$tick" -ge "${MAX_WAIT_MIN:-2400}" ]; then
            say "  BOUND HIT: $tick min elapsed, $stall min since the last unit"
            say "  (limits: MAX_WAIT_MIN=${MAX_WAIT_MIN:-2400}, STALL_MIN=${STALL_MIN:-90})"
            for g in $GPUS; do
                tmux has-session -t vf_${tag}_$g 2>/dev/null || continue
                # BY PID, never by pattern: a pattern kill would match this script.
                for pid in $(tmux list-panes -t vf_${tag}_$g -F '#{pane_pid}' 2>/dev/null); do
                    for child in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
                        say "    kill PID $child (stalled $tag worker on GPU $g)"; kill "$child" 2>/dev/null
                    done
                done
                sleep 5; tmux kill-session -t vf_${tag}_$g 2>/dev/null
            done
            break
        fi
        sleep 60
    done
}

# ---------------------------------------------------------------- smoke
if [ "$MODE" = "smoke" ]; then
    check_inputs
    say "=== smoke: one complex, one seed. The PXM sweep is NOT paused. ==="
    SQ=$E/queue_smoke.tsv
    head -1 <(for f in $J/*.json; do printf 'free\t%s\t*\tfree\t400000\n' "$(basename "$f" .json)"; done) > $SQ
    say "queue: $(cat $SQ)"
    run_workers "$SQ" smoke

    NCIF=$(find $E/poses -name '*.cif' 2>/dev/null | wc -l)
    NTRUNK=$(find $E/trunks -name '*.npz' 2>/dev/null | wc -l)
    # NUMERIC dockq AND numeric iptm on at least one interface: proves the scorer ran, that OST
    # produced a score rather than discarding it (the --rigid-scores crash we have hit before), and
    # that the ipTM we will compare against after recalibration is actually being recorded.
    # BY HEADER NAME, NOT BY POSITION. $7 and $8 happen to be dockq and iptm today, but that is
    # folder.py's column order and nothing here owns it; a column inserted upstream would silently
    # make this count the wrong field and still pass.
    NDOCKQ=$(csv_numeric "$E/results.csv" dockq) || die "results.csv has no 'dockq' column"
    NIPTM=$(csv_numeric "$E/results.csv" iptm)  || die "results.csv has no 'iptm' column"
    NCONF=$(find $E/poses -name '*summary_confidence_sample_*.json' 2>/dev/null | wc -l)
    say "  cifs $NCIF   trunk npz $NTRUNK   numeric dockq $NDOCKQ   numeric iptm $NIPTM   conf json $NCONF"
    [ "$NCIF"   -ge 1 ] || die "no pose cif"
    [ "$NTRUNK" -ge 1 ] || die "no trunk npz -- TRUNK_DUMP_DIR did not fire"
    [ "$NDOCKQ" -ge 1 ] || die "no numeric DockQ -- scoring or the reference cif is wrong"
    [ "$NIPTM"  -ge 1 ] || die "no numeric ipTM"
    [ "$NCONF"  -ge 1 ] || die "no summary_confidence json -- per-interface ipTM would be lost"

    # OPEN THE NPZ. Counting files proves nothing about whether the head can be re-run from it: the
    # tensors are what matter, and a dump that silently dropped them looks identical on disk. This is
    # the assertion the whole phase exists to satisfy.
    $PY - "$E" "$NSAMPLE" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, sys, numpy as np
root, nsample = sys.argv[1], int(sys.argv[2])
f = sorted(glob.glob(f"{root}/trunks/*/*.npz"))
assert f, "no npz found"
d = np.load(f[0], allow_pickle=True)
need = ["z_trunk", "s_trunk", "ci_s_inputs", "ci_s_trunk_f32", "ci_coordinate",
        "ci_contact_probs", "ci_feat_atom_to_token_idx", "ci_feat_asym_id", "ci_N_cycle"]
missing = [k for k in need if k not in d.files]
print(f"npz {f[0]}")
for k in d.files:
    a = d[k]
    print(f"   {k:28s} {str(a.dtype):9s} {a.shape}")
assert not missing, f"MISSING CONFIDENCE-HEAD TENSORS: {missing}"
assert str(d["dtype"]) == "bf16", f"z/s stored as {d['dtype']}, expected bf16"

# THE PER-POSE DIMENSION, asserted rather than assumed. --nsample produces N poses per
# (structure, seed). z_trunk and s_trunk are rightly per seed -- one trunk, N samples drawn from it
# -- but ci_coordinate and ci_contact_probs are per POSE. If the npz holds only the last of N, then
# N-1 of every N poses have no cached head inputs and "re-run the head on every pose without a GPU"
# is 1/N met. It cannot be repaired later either: a repeat of the same seed does not reproduce the
# same poses (see the seed-nondeterminism note in the project memory), so the missing ones are gone.
# Leading batch dimensions of 1 are squeezed before counting.
def lead(a):
    sh = list(a.shape)
    while len(sh) > 1 and sh[0] == 1:
        sh.pop(0)
    return sh
for k in ("ci_coordinate", "ci_contact_probs"):
    sh = lead(d[k])
    print(f"   {k}: effective shape {sh}, expecting {nsample} poses in the leading dimension")
    assert sh and sh[0] == nsample, (
        f"{k} has leading dimension {sh[0] if sh else None}, expected nsample={nsample}. "
        f"The cache holds {sh[0] if sh else 0} of {nsample} poses per unit, and the missing ones "
        "cannot be recovered -- repeating a seed does not reproduce its poses.")
print("NPZ_OK all confidence-head tensors present, with one entry per pose")
PYEOF
    grep -q "NPZ_OK" "$LOG" || die "the trunk npz lacks tensors the confidence head needs (see above)"
    touch "$E/SMOKE_OK"
    say "=== smoke OK: pose, per-fold npz with every head tensor, DockQ, ipTM, confidence json ==="
    exit 0
fi

# ---------------------------------------------------------------- verify
verify() {
    local n rows trunks up g
    n=$(done_units)
    rows=$(($(wc -l < $E/results.csv 2>/dev/null || echo 1)-1))
    trunks=$(find $E/trunks -name '*.npz' 2>/dev/null | wc -l)
    say "  units $n/$TARGET   scored rows $rows   trunk dumps $trunks   cache $(du -sh $E/trunks 2>/dev/null | cut -f1)"
    # ONE NPZ PER FOLDED UNIT. If these diverge, dumps were overwritten or failed silently and the
    # cache cannot re-run the head on every pose -- which is the entire deliverable.
    [ "$trunks" -eq "$n" ] || say "ESCALATE: $trunks trunk npz for $n folded units; they must match"
    up=0
    for g in $GPUS; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && up=$((up+1)); done
    say "  pxm workers back up: $up   folds $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null)"
    [ "$n" -ge "$TARGET" ] && say "=== COMPLETE: all $TARGET units folded, scored and cached ===" \
        || say "ESCALATE: $((TARGET-n)) units short. Poses are kept; re-run 'run' to continue."
}
if [ "$MODE" = "verify" ]; then check_inputs; verify; exit 0; fi

# ---------------------------------------------------------------- run
check_inputs
# THE SMOKE GATE IS THIS PHASE'S OWN PRECONDITION, not the orchestrator's. 16 h of GPU and a paused
# PXM sweep should not be reachable by calling this script in the wrong order.
[ -f "$E/SMOKE_OK" ] || die "smoke has not passed. Run 'phase_i_fold_validation.sh smoke' first --
     it is the only check that the cache can actually re-run the confidence head."

# EVERY OTHER PHASE RESTORES WHAT IT CHANGED; this one must too. Without a trap, a run killed
# between the pause below and the resume at the end leaves the PXM sweep down and fx_watch dead --
# indefinitely, and silently, because the thing that would have restarted them is the script that
# just died. INT, TERM and EXIT all route through the same restore.
PXM_PAUSED=0
resume_pxm() {
    [ "$PXM_PAUSED" -eq 1 ] || return 0
    PXM_PAUSED=0
    say "=== restoring the PXM sweep ==="
    $PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -1 | tee -a "$LOG"
    for g in $GPUS; do
        tmux has-session -t fx_pxm2025_$g 2>/dev/null && continue
        tmux new-session -d -s fx_pxm2025_$g "cd $P && CUDA_VISIBLE_DEVICES=$g LAYERNORM_TYPE=openfold \
          $PY $C/folder.py --queue $PXM/queue.tsv --out $PXM/poses --csv $PXM/results.csv \
          --claim-dir $PXM/claims --dataset pxm2025 --ncycle 10 --nsample "$NSAMPLE" --dtype bf16 \
          --overlap-scoring >> $PXM/w$g.log 2>&1"
        say "  relaunched pxm worker on GPU $g"
    done
    tmux has-session -t fx_watch 2>/dev/null \
        || { tmux new-session -d -s fx_watch "bash $C/watch_free_extra10.sh" && say "  restarted fx_watch"; }
}
trap 'say "interrupted -- restoring PXM before exit"; resume_pxm; exit 130' INT TERM
trap 'resume_pxm' EXIT

say "=== run: $TARGET units ($NJOB structures x $SEEDS seeds) on GPUs $GPUS ==="
say "pausing the PXM sweep: $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null) folds so far"
PXM_PAUSED=1
for g in $GPUS; do
    if tmux has-session -t fx_pxm2025_$g 2>/dev/null; then
        # BY PID, NEVER BY PATTERN. A pattern kill matches this script too.
        for pid in $(tmux list-panes -t fx_pxm2025_$g -F '#{pane_pid}' 2>/dev/null); do
            for child in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
                say "  kill PID $child (pxm worker on GPU $g)"; kill "$child" 2>/dev/null
            done
        done
        sleep 5; tmux kill-session -t fx_pxm2025_$g 2>/dev/null
    fi
done
tmux kill-session -t fx_watch 2>/dev/null && say "  stopped fx_watch so it cannot relaunch PXM onto our cards"
sleep 10
say "releasing the claims those workers held"
$PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -2 | tee -a "$LOG"

Q=$E/queue.tsv
say "queue: $(build_queue $SEEDS $Q) units"

# ROUNDS, NOT ONE SHOT. A worker that dies -- OOM, a bad cif, a node hiccup -- ends its tmux session
# and run_workers returns with the queue unfinished. folder.py reconciles stale claims at startup and
# skips units already in results.csv, so relaunching simply continues. The loop stops on target, or
# on a round that adds nothing, which is a real fault rather than a transient one.
for round in $(seq 1 $ROUNDS); do
    before=$(done_units)
    [ "$before" -ge "$TARGET" ] && { say "all $TARGET units already folded"; break; }
    say "round $round: $before/$TARGET done, launching workers"
    run_workers "$Q" val
    after=$(done_units)
    say "round $round ended at $after/$TARGET (+$((after-before)))"
    [ "$after" -ge "$TARGET" ] && break
    if [ "$after" -le "$before" ]; then
        say "ESCALATE: round $round folded nothing. Stopping; see $E/w*.log"
        break
    fi
done

resume_pxm
sleep 30
verify
