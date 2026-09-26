#!/bin/bash
# Staged GPU handoff, per GPU: fixed-seed arm -> free top-up (compute matching for arm A) ->
# variable-seed arm. A GPU moves on the moment its current worker exits.
#
# The variable-seed arm starts only once varseed_ready(): the top-up is fully folded and scored on
# every GPU, and the variable-seed queue rebuilt WITHOUT pending seeds is identical to the one queued --
# i.e. every seed it uses is now a scored free seed. If the rebuild differs the card is held and a
# REMEDIATE line logged instead. LAUNCH_VARSEED=0 holds every card after the top-up regardless.
#
# The run is capped at the four GPUs it holds now: 0, 1, 2, 5. Each GPU runs its own chain, so a GPU
# moves to the next arm the moment its current worker exits -- workers exit when their queue has nothing
# left to claim, so an arm that no longer needs a GPU hands it on without waiting for its slowest fold.
#
# Both self-template arms read the SAME template namespace, the fixed arm's (a template carries no seed).
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/Protenix/.venv/bin/python
D=/home/ebrahim/tts_abag/experimental_data
FIX=$D/self_template_fixed_50_seed
VAR=$D/self_template_variable_50_seed
FREE=$D/free_compute_matched
TOPUP=$FREE/queue_free_topup_arm_a.tsv     # the free folds arm A's extra donors need, 12
TMPL=$FIX/templates               # the orchestrator's template namespace
LOG=$D/stage_arms.log
GPUS="0 1 2 5"
LAUNCH_VARSEED=${LAUNCH_VARSEED:-1}
hold() { tmux has-session -t "hold$1" 2>/dev/null || tmux new-session -d -s "hold$1" \
    "CUDA_VISIBLE_DEVICES=$1 $PY /home/ebrahim/tts_abag/diffusion_commit_clean_v2/hold_gpus.py --gpus $1 --gb 1"; }
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
alive() { tmux ls 2>/dev/null | grep -q "^$1:"; }
# a paused card is held by hold_gpus.py in session hold<gpu>; release it only at the moment of launch
release_hold() { tmux kill-session -t "hold$1" 2>/dev/null && say "GPU $1: released hold"; }

launch() {   # launch <queue> <out> <csv> <claims> <logdir> <gpu> [extra call_folder flags]
    $PY diffusion_commit_clean_v2/call_folder.py --queue "$1" --out "$2" --csv "$3" \
        --claim-dir "$4" --log-dir "$5" --templates "$TMPL" --gpus "$6" "${@:7}" --run >> "$LOG" 2>&1
}

launch_free() {   # launch_free <queue> <gpu> -- untemplated folds into the compute-matched run
    $PY diffusion_commit_clean_v2/call_folder.py --queue "$1" --out "$FREE" --csv "$FREE/results.csv" \
        --claim-dir "$FREE/claims" --log-dir "$FREE" --gpus "$2" --run >> "$LOG" 2>&1
}

unclaimed() {   # unclaimed <queue> <claim dir> -- how many of the queue's units have no claim yet
    $PY - "$1" "$2" <<'PYEOF'
import os, sys
q, cd = sys.argv[1], sys.argv[2]
have = set(os.listdir(cd)) if os.path.isdir(cd) else set()
units = [l.rstrip("\n").split("\t") for l in open(q) if l.strip()]
print(sum(1 for f in units if f"{f[1]}_{f[2].replace('*', 'all')}_{f[3]}_{f[4]}" not in have))
PYEOF
}

varseed_ready() {   # 0 once the top-up is scored and the variable-seed queue checks out, 1 otherwise
    tmux ls 2>/dev/null | grep -q '^free_compute_matchedw' && return 1
    [ "$(unclaimed "$TOPUP" "$FREE/claims")" -eq 0 ] || return 1
    local tmp; tmp=$(mktemp)
    $PY diffusion_commit_clean_v2/build_varseed_queue.py --fixed "$FIX" --out "$tmp" >> "$LOG" 2>&1
    if cmp -s "$tmp" "$VAR/queue.tsv"; then rm -f "$tmp"; return 0; fi
    say "REMEDIATE variable-seed queue rebuilt from scored seeds differs from the queued one ($tmp)"
    return 2
}

chain() {   # chain <gpu>
    local g=$1
    while alive "free_compute_matchedw$g"; do sleep 120; done
    say "GPU $g: no free worker"
    if ! alive "self_template_fixed_50_seedw$g"; then
        local c=$(ls "$FIX/claims" | wc -l) n=$(wc -l < "$FIX/queue_tofold.tsv")
        if [ "$c" -lt "$n" ]; then
            release_hold "$g"
            launch "$FIX/queue_tofold.tsv" "$FIX/poses" "$FIX/results.csv" "$FIX/claims" "$FIX" "$g"
            say "GPU $g: launched fixed-seed worker ($c/$n claimed)"
        else
            say "GPU $g: fixed-seed queue fully claimed ($c/$n) -- skipping to variable-seed"
        fi
    fi
    while alive "self_template_fixed_50_seedw$g"; do sleep 120; done
    if [ "$(unclaimed "$TOPUP" "$FREE/claims")" -gt 0 ]; then
        release_hold "$g"
        launch_free "$TOPUP" "$g"
        say "GPU $g: launched free top-up worker"
        while alive "free_compute_matchedw$g"; do sleep 60; done
    fi
    if [ "$LAUNCH_VARSEED" != 1 ]; then
        hold "$g"
        say "GPU $g: fixed arm and top-up done -- holding the card for the variable-seed arm"
        return
    fi
    hold "$g"                      # keep the card while waiting for the other GPUs' top-up folds
    local ok retries=0
    while true; do
        varseed_ready; ok=$?
        [ $ok -eq 1 ] || break
        # a top-up fold that failed had its claim released and nobody is folding it: retry it here
        if ! tmux ls 2>/dev/null | grep -q '^free_compute_matchedw' \
                && [ "$(unclaimed "$TOPUP" "$FREE/claims")" -gt 0 ]; then
            retries=$((retries + 1))
            if [ $retries -gt 3 ]; then
                say "REMEDIATE GPU $g: free top-up still unfinished after 3 retries -- card held"
                return
            fi
            release_hold "$g"
            launch_free "$TOPUP" "$g"
            say "GPU $g: retrying unfinished free top-up folds (attempt $retries)"
            while alive "free_compute_matchedw$g"; do sleep 60; done
            hold "$g"
        fi
        sleep 120
    done
    if [ $ok -ne 0 ]; then
        say "GPU $g: variable-seed arm NOT launched (queue check failed) -- card held"
        return
    fi
    release_hold "$g"
    # the variable-seed arm scores each fold in the background while the next folds (~13% faster);
    # the fixed arm keeps the sequential path it was verified on
    launch "$VAR/queue.tsv" "$VAR/poses" "$VAR/results.csv" "$VAR/claims" "$VAR" "$g" --overlap-scoring
    say "GPU $g: launched variable-seed worker (overlapped scoring)"
}

say "=== stage_arms (per-GPU) starting, GPUs $GPUS ==="
for g in $GPUS; do chain "$g" & done
wait
say "=== all four GPUs handed to the variable-seed arm ==="
