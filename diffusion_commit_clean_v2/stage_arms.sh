#!/bin/bash
# Staged GPU handoff, per GPU: free top-up -> fixed-seed arm -> variable-seed arm. Replaces the old
# experimental_data/stage_arms.sh, which hardcoded GPUs 1,5,6 for the fixed arm and would have put it
# on 5 GPUs (0,2 + 1,5,6).
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
TMPL=$FIX/templates               # the orchestrator's template namespace
LOG=$D/stage_arms.log
GPUS="0 1 2 5"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
alive() { tmux ls 2>/dev/null | grep -q "^$1:"; }
# a paused card is held by hold_gpus.py in session hold<gpu>; release it only at the moment of launch
release_hold() { tmux kill-session -t "hold$1" 2>/dev/null && say "GPU $1: released hold"; }

launch() {   # launch <queue> <out> <csv> <claims> <logdir> <gpu>
    $PY diffusion_commit_clean_v2/call_folder.py --queue "$1" --out "$2" --csv "$3" \
        --claim-dir "$4" --log-dir "$5" --templates "$TMPL" --gpus "$6" --run >> "$LOG" 2>&1
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
    release_hold "$g"
    launch "$VAR/queue.tsv" "$VAR/poses" "$VAR/results.csv" "$VAR/claims" "$VAR" "$g"
    say "GPU $g: launched variable-seed worker"
}

say "=== stage_arms (per-GPU) starting, GPUs $GPUS ==="
for g in $GPUS; do chain "$g" & done
wait
say "=== all four GPUs handed to the variable-seed arm ==="
