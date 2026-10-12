#!/bin/bash
# PHASE I -- fold the validation set and cache everything needed to re-run the confidence head.
#
#   phase_i_fold_validation.sh smoke    one complex, one seed. Asserts the mechanism in code, not by
#                                       eye. Does NOT pause the PXM sweep.
#   phase_i_fold_validation.sh run      pause PXM, fold every structure x every seed, resume PXM,
#                                       then verify. Refuses to start until smoke has passed.
#   phase_i_fold_validation.sh verify   the post-fold checks on their own; nonzero if incomplete.
#   phase_i_fold_validation.sh rescore  score folded units that have no interface rows. A refold
#                                       CANNOT recover those: the trunk npz already exists and
#                                       folder.py refuses to overwrite it, and repeating a seed does
#                                       not reproduce its poses anyway.
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
PH=$R/extra_training/phases
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
LOG=$R/extra_training/phase_i.log
GPUS="${GPUS:-0 1}"
# folder.py resolves a job as <jobs>/<pdb>-final-updated.json, so the pdb id is the basename with
# that suffix removed -- not just ".json".
JOB_SUFFIX="-final-updated.json"
SEEDS="${SEEDS:-20}"
NSAMPLE="${NSAMPLE:-5}"
# SEED0 in one place. The queue builder, the smoke queue and the verifier must agree on which seeds
# are expected; three copies of the literal 400000 would let them drift apart silently.
SEED0="${SEED0:-400000}"
ROUNDS="${ROUNDS:-5}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

case "$MODE" in smoke|run|verify|rescore) ;; *) die "usage: phase_i_fold_validation.sh smoke|run|verify|rescore" ;; esac
mkdir -p "$E" "$E/trunks" "$E/poses"

# ---------------------------------------------------------------- inputs, checked before any GPU
# EVERY INPUT VERIFIED BEFORE 16 HOURS OF GPU IS COMMITTED, and by content rather than by count. A
# job json whose templatesPath or MSA path does not exist produces a fold that runs and scores
# nothing, which is the most expensive way to discover a missing file.
check_inputs() {
    # COUNTED BY THE NAME folder.py WILL LOOK FOR. Counting *.json would pass a directory full of
    # files the driver cannot find, which is exactly how the <pdb>.json mismatch survived review.
    NJOB=$(ls $J/*$JOB_SUFFIX 2>/dev/null | wc -l)
    NANY=$(ls $J/*.json 2>/dev/null | wc -l)
    [ "$NJOB" -ge 1 ] || die "no *$JOB_SUFFIX in $J (found $NANY other .json). folder.py:784 opens
     <jobs>/<pdb>$JOB_SUFFIX, so differently named files are invisible to it."
    [ "$NJOB" -eq "$NANY" ] || say "  note: $((NANY-NJOB)) json in $J do not match *$JOB_SUFFIX"
    POOL=$E/pool.csv
    [ -s "$POOL" ] || die "no $POOL -- folder.py's interfaces_by_pdb() reads it, and without it the
     fold runs and scores NOTHING"
    NPDB=$(tail -n +2 "$POOL" | cut -d, -f1 | sort -u | wc -l)
    NIF=$(($(wc -l < "$POOL")-1))
    [ "$NPDB" -eq "$NJOB" ] || die "pool.csv covers $NPDB structures but there are $NJOB jsons"
    # DANGLING PATHS AND THE TEMPLATE CUTOFF, in phase_i_inputs.py. It imports phase H and
    # recomputes the cutoff THIS job set requires, then demands the marker be at least that strict,
    # at the current filter version, with a hash matching the file. Verifying the hash alone -- what
    # this did before -- proved only that the file had not changed since filtering, not that it was
    # filtered for this query: one MSA directory is shared by every structure with that chain
    # sequence, so a marker written for a newer structure carries a later cutoff, and the pre-2021
    # structure sharing the directory would receive templates from its own era, its own entry at
    # 100% identity among them.
    $PY $PH/phase_i_inputs.py --jobs "$J" --phases "$PH" --suffix "$JOB_SUFFIX" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "input check failed (see above). Templates are required by
     decision, so phase H must have run -- and its date filter must have run for THESE jobs: a
     template file filtered at a later cutoff may hand a pre-2021 structure its own entry at 100%
     identity. A dangling path folds without the file and nothing reports it."
    # THE SCORER, RESOLVED HERE SO SMOKE EXERCISES IT. This used to sit beside verify(), below the
    # smoke block, with only $C on sys.path -- but paths.py lives at $R, not $C, so it raised
    # ModuleNotFoundError from any cwd, died on the `die` beneath it, and smoke never reached the
    # line. Smoke would pass and the full run would fail instantly. Both roots are added: $R for
    # paths.py, $C for the modules it pulls in.
    SCORER=$($PY -c "import sys; sys.path[:0]=['$R','$C']; import paths; print(paths.DATASETS['sabdab_val'].scorer)" 2>&1)
    case "$SCORER" in
        ost|pxm) say "  scorer for dataset sabdab_val: $SCORER (from paths.py)" ;;
        *) die "cannot resolve the scorer for dataset sabdab_val from paths.py: $SCORER" ;;
    esac
    NTMPL=$(find "$MSA" -name hmmsearch.a3m 2>/dev/null | wc -l)
    say "inputs OK: $NJOB jsons, $NIF interfaces over $NPDB structures, $NTMPL hmmsearch.a3m, no dangling paths"
    # THE TARGET COUNTS ACTIVE STRUCTURES. Using the raw job count made a recorded skip read as
    # permanently missing work: the rounds loop would keep relaunching for units that will never
    # arrive, and verify() would never reach COMPLETE.
    NACTIVE=$(active_pdbs | wc -l)
    TARGET=$((NACTIVE * SEEDS))
    [ "$NACTIVE" -eq "$NJOB" ] || say "  $((NJOB-NACTIVE)) structure(s) skipped by decision \
(val_fold/skipped.csv); folding $NACTIVE of $NJOB"
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
    local n=$1 out=$2 pdb i
    : > "$out"
    # OVER THE ACTIVE SET, so a recorded skip is not re-attempted 20 times a round for five rounds.
    for pdb in $(active_pdbs); do
        for i in $(seq 0 $((n-1))); do
            printf 'free\t%s\t*\tfree\t%d\n' "$pdb" $((SEED0+i)) >> "$out"
        done
    done
    wc -l < "$out"
}

# in: nothing.  out: the pdb ids to fold, one per line -- the job set MINUS the recorded skips
#
# ONE ACTIVE SET, used by the queue, the target, smoke and verification. skipped.csv was honoured by
# verification only, so a structure recorded as un-foldable stayed in every seed's queue and was
# re-attempted 20 times a round, five rounds deep -- which defeats the point of recording it and
# spends GPU on work already excluded by decision.
active_pdbs() {
    local skips=$E/skipped.csv f pdb
    for f in $J/*$JOB_SUFFIX; do
        pdb=$(basename "$f" "$JOB_SUFFIX")
        if [ -s "$skips" ] && awk -F, -v p="$pdb" 'NR>1 && $1==p {found=1} END{exit !found}' "$skips"; then
            continue
        fi
        echo "$pdb"
    done
}

run_workers() {
    # $3 is the gpu list, defaulting to all of them. Smoke passes ONE: it has a single work unit, so
    # a second worker only loads the model onto another card and then finds nothing to claim.
    local queue=$1 tag=$2 gpus=${3:-$GPUS} g last tick alive n
    for g in $gpus; do
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
        for g in $gpus; do tmux has-session -t vf_${tag}_$g 2>/dev/null && alive=1; done
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
            for g in $gpus; do
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

# ------------------------------------------------- PXM pause / restore, defined before both
# modes. SMOKE NEEDS THIS TOO, which is why it lives here rather than beside the run block:
# smoke launches a real worker that loads the model onto a real card, so running it while the
# PXM sweep holds GPUs 0 and 1 risks an OOM in both jobs and makes the smoke result say more
# about contention than about our inputs. Both modes therefore pause, work, and restore.

# EVERY OTHER PHASE RESTORES WHAT IT CHANGED; this one must too. Without a trap, a run killed
# between the pause below and the resume at the end leaves the PXM sweep down and fx_watch dead --
# indefinitely, and silently, because the thing that would have restarted them is the script that
# just died. INT, TERM and EXIT all route through the same restore.
PXM_PAUSED=0
PXM_WAS=""        # which fx_pxm2025_<g> sessions existed before we paused
WATCH_WAS=0       # whether fx_watch was running before we killed it

# in: a session name prefix and the gpu list.  out: nothing; kills those sessions BY PID
kill_sessions() {
    # $2 is the gpu list, defaulting to all of them -- smoke pauses only the one card it will use.
    local prefix=$1 gpus=${2:-$GPUS} g pid child
    for g in $gpus; do
        tmux has-session -t ${prefix}$g 2>/dev/null || continue
        for pid in $(tmux list-panes -t ${prefix}$g -F '#{pane_pid}' 2>/dev/null); do
            # BY PID, never by pattern: a pattern kill matches this script.
            for child in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
                say "    kill PID $child (${prefix}$g)"; kill "$child" 2>/dev/null
            done
        done
        sleep 3; tmux kill-session -t ${prefix}$g 2>/dev/null
    done
}

resume_pxm() {
    [ "$PXM_PAUSED" -eq 1 ] || return 0
    PXM_PAUSED=0
    # OUR WORKERS GO FIRST. The previous version restarted PXM without stopping the validation
    # workers, so an interrupted run put BOTH workloads on the same cards -- two folds per GPU,
    # competing for memory, with neither job aware of the other.
    say "=== stopping validation workers before restoring PXM ==="
    kill_sessions vf_val_
    kill_sessions vf_smoke_
    say "=== restoring the PXM sweep ==="
    $PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -1 | tee -a "$LOG"
    # ONLY WHAT WAS RUNNING BEFORE. Starting a worker on a card that had none is not a restore, it is
    # a new allocation -- and GPU allocation is not this script's call to make.
    for g in $PXM_WAS; do
        tmux has-session -t fx_pxm2025_$g 2>/dev/null && continue
        tmux new-session -d -s fx_pxm2025_$g "cd $P && CUDA_VISIBLE_DEVICES=$g LAYERNORM_TYPE=openfold \
          $PY $C/folder.py --queue $PXM/queue.tsv --out $PXM/poses --csv $PXM/results.csv \
          --claim-dir $PXM/claims --dataset pxm2025 --ncycle 10 --nsample "$NSAMPLE" --dtype bf16 \
          --overlap-scoring >> $PXM/w$g.log 2>&1"
        say "  relaunched pxm worker on GPU $g (it was running before)"
    done
    [ -n "$PXM_WAS" ] || say "  no pxm worker was running before; none started"
    if [ "$WATCH_WAS" -eq 1 ] && ! tmux has-session -t fx_watch 2>/dev/null; then
        tmux new-session -d -s fx_watch "bash $C/watch_free_extra10.sh" && say "  restarted fx_watch"
    fi
}

# ---------------------------------------------------------------- smoke
if [ "$MODE" = "smoke" ]; then
    check_inputs
    # ONE CARD, AND THE SWEEP PAUSED ON IT. Smoke used to launch a worker on every configured GPU
    # while declaring the PXM sweep untouched: two workers for one work unit, both loading the model
    # onto cards PXM already held. The second worker can only find nothing to claim, and the
    # contention risks an OOM in both jobs -- so a smoke failure would say more about scheduling
    # than about our inputs, which is the opposite of what a smoke test is for.
    SMOKE_GPU=$(echo $GPUS | awk '{print $1}')
    say "=== smoke: one complex, one seed, one worker on GPU $SMOKE_GPU ==="
    trap 'say "interrupted -- restoring PXM before exit"; resume_pxm; exit 130' INT TERM
    trap 'resume_pxm' EXIT
    PXM_PAUSED=1
    for g in $SMOKE_GPU; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && PXM_WAS="$PXM_WAS $g"; done
    tmux has-session -t fx_watch 2>/dev/null && WATCH_WAS=1
    if [ -n "$PXM_WAS" ] || [ "$WATCH_WAS" -eq 1 ]; then
        say "  pausing pxm on GPU(s)${PXM_WAS:- none} and fx_watch=$WATCH_WAS for the smoke unit"
        tmux kill-session -t fx_watch 2>/dev/null && say "  stopped fx_watch so it cannot relaunch pxm"
        kill_sessions fx_pxm2025 "$SMOKE_GPU"
        sleep 5
        $PY $C/reconcile_stale_claims.py $PXM --dataset pxm2025 2>&1 | tail -1 | tee -a "$LOG"
    fi
    SQ=$E/queue_smoke.tsv
    printf 'free\t%s\t*\tfree\t%d\n' "$(active_pdbs | head -1)" "$SEED0" > $SQ
    say "queue: $(cat $SQ)"
    run_workers "$SQ" smoke "$SMOKE_GPU"

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
    # EXIT STATUS, NOT A GREP OVER THE LOG. The validator was piped through tee without pipefail
    # and then grepped for NPZ_OK across the whole appended log, so a FAILING run passed as soon as
    # any earlier run had printed that marker. Its output is captured, shown, and its status checked.
    NPZOUT=$($PY - "$E" "$NSAMPLE" <<'PYEOF' 2>&1

import glob, sys, numpy as np
root, nsample = sys.argv[1], int(sys.argv[2])
f = sorted(glob.glob(f"{root}/trunks/*/*.npz"))
assert f, "no npz found"
d = np.load(f[0], allow_pickle=True)
# ALL SIX FEATURE TENSORS, not the two that happened to be listed. The dump writes
# distogram_rep_atom_mask, atom_to_token_idx, atom_to_tokatom_idx, asym_id, has_frame and is_ligand;
# the head and the ipTM summary read all of them, so a cache missing any one cannot be replayed.
need = ["z_trunk", "s_trunk", "ci_s_inputs", "ci_s_trunk_f32", "ci_coordinate",
        "ci_contact_probs", "ci_N_cycle", "ci_z_dtype", "ci_s_dtype", "ci_s_inputs_dtype",
        "rng_py", "rng_torch"] + [f"ci_feat_{k}" for k in (
            "distogram_rep_atom_mask", "atom_to_token_idx", "atom_to_tokatom_idx",
            "asym_id", "has_frame", "is_ligand")]
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
# COORDINATES ARE PER POSE and must all be there.
sh = lead(d["ci_coordinate"])
print(f"   ci_coordinate: effective shape {sh}, expecting {nsample} poses leading")
assert sh and sh[0] == nsample, (
    f"ci_coordinate has leading dimension {sh[0] if sh else None}, expected nsample={nsample}. "
    f"The cache holds {sh[0] if sh else 0} of {nsample} poses per unit, and the missing ones cannot "
    "be recovered -- repeating a seed does not reproduce its poses.")

# CONTACT PROBABILITIES ARE SHARED, NOT PER POSE. protenix.py:1284 computes them from the single
# trunk z with shape [N_token, N_token], and `per_sample_contact_probs` exists only when the
# permutation path stacked one per sample (permutation.py:447). Requiring nsample leading would
# therefore REJECT A VALID CACHE. Both shapes are accepted; which one it is gets printed, because
# an offline replay has to broadcast in the shared case and index in the other.
cp = lead(d["ci_contact_probs"])
ntok = lead(d["s_trunk"])[0]
if len(cp) == 2:
    assert cp == [ntok, ntok], f"ci_contact_probs {cp} is not [N_token, N_token] = [{ntok}, {ntok}]"
    print(f"   ci_contact_probs: shared across poses, {cp} -- broadcast when replaying")
elif len(cp) == 3:
    assert cp[0] == nsample and cp[1:] == [ntok, ntok], (
        f"ci_contact_probs {cp} is neither [N_token,N_token] nor [{nsample},{ntok},{ntok}]")
    print(f"   ci_contact_probs: one per pose, {cp}")
else:
    raise AssertionError(f"ci_contact_probs has unexpected rank: shape {cp}")
print("NPZ_OK all confidence-head tensors present, coordinates per pose")
PYEOF
)
    NPZRC=$?
    printf '%s\n' "$NPZOUT" | tee -a "$LOG"
    [ "$NPZRC" -eq 0 ] || die "the trunk npz validator failed (exit $NPZRC); see the output above"
    printf '%s' "$NPZOUT" | grep -q "NPZ_OK" \
        || die "the validator exited 0 but did not confirm NPZ_OK -- treat as a failure"
    touch "$E/SMOKE_OK"
    say "=== smoke OK: pose, per-fold npz with every head tensor, DockQ, ipTM, confidence json ==="
    exit 0
fi

# ---------------------------------------------------------------- verify / rescore
# BOTH LIVE IN phase_i_check.py. The shell versions of these two were the most error-prone code in
# this file and failed for reasons shell makes easy: `IFS=, read` kept the \r of a CRLF pool.csv in
# the last field and produced interfaces like "6mnq_AB\r" that scorer.py rejects; a crashed
# verification subprocess fell back to zero failures via ${1:-0}, turning a crash into a pass; and
# "a row exists" was accepted as a scored interface, so one scored sample of five, a NaN DockQ or a
# blank ipTM all read as complete. The checker returns a real exit status and this reports it.
#
# THE SCORER IS THE DATASET'S OWN, read from paths.py rather than named here. sabdab_val scores with
# 'ost'; an earlier rescore hardcoded 'pxm', which would have mixed two DockQ implementations into one
# results.csv and made the recovered rows incomparable to the rest.

check_py() {
    $PY $PH/phase_i_check.py "$1" --e "$E" --jobs "$J" --refs "$R/extra_training/val_refs" \
        --suffix "$JOB_SUFFIX" --seeds "$SEEDS" --seed0 "$SEED0" --nsample "$NSAMPLE" \
        --scorer "$SCORER" --py "$PY" --code "$C" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

verify() {
    local rc up g
    say "=== verify: coverage, finite scores, replayable caches (scorer=$SCORER) ==="
    check_py verify; rc=$?
    up=0
    for g in $GPUS; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && up=$((up+1)); done
    say "  pxm workers up: $up   folds $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null)"
    [ "$rc" -eq 0 ] || say "=== INCOMPLETE: poses are kept. 'rescore' recovers scores; 'run' continues folding. ==="
    return $rc
}

# RESCORE, because a refold cannot recover this. folder.py builds units_done from results.csv rows,
# not from run.json, so a unit that folded but failed to score is retried -- and then fails every
# time, because TRUNK_DUMP_DIR refuses to overwrite the npz the first attempt already wrote
# (folder.py:831). Refolding would be wrong even if it were possible: repeating a seed does not
# reproduce its poses, so new poses would not match the cached tensors. The only correct recovery is
# to score the poses already on disk -- per missing sample, so an interface with 3 of 5 samples gains
# the 2 it lacks instead of being skipped for having rows, or gaining 5 duplicates.
rescore() {
    say "=== rescore: scoring the (interface, sample) rows that are missing ==="
    check_py rescore
}
if [ "$MODE" = "rescore" ]; then check_inputs; rescore; verify; exit $?; fi
if [ "$MODE" = "verify" ]; then check_inputs; verify; exit $?; fi

# ---------------------------------------------------------------- run
check_inputs
# THE SMOKE GATE IS THIS PHASE'S OWN PRECONDITION, not the orchestrator's. 16 h of GPU and a paused
# PXM sweep should not be reachable by calling this script in the wrong order.
[ -f "$E/SMOKE_OK" ] || die "smoke has not passed. Run 'phase_i_fold_validation.sh smoke' first --
     it is the only check that the cache can actually re-run the confidence head."

trap 'say "interrupted -- restoring PXM before exit"; resume_pxm; exit 130' INT TERM
trap 'resume_pxm' EXIT

say "=== run: $TARGET units ($NJOB structures x $SEEDS seeds) on GPUs $GPUS ==="
say "pausing the PXM sweep: $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null) folds so far"
PXM_PAUSED=1
for g in $GPUS; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && PXM_WAS="$PXM_WAS $g"; done
tmux has-session -t fx_watch 2>/dev/null && WATCH_WAS=1
say "  running before the pause: pxm on GPU(s)${PXM_WAS:- none}, fx_watch=$WATCH_WAS"
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
