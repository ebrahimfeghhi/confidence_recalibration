#!/bin/bash
# Staged GPU handoff: baseline -> fixed-seed arm -> variable-seed arm, in that priority order.
#
# Both self-template arms read the SAME template namespace, the fixed arm's. A template is a donor pose
# plus its MSA hit and carries no seed, so the two arms differ only in the seed column of their queue.
# Sharing the namespace is what makes them comparable, and it saved rebuilding 1.7 GB of templates.
#
# The variable-seed arm draws its seeds from 300100-300149 -- the same 50 seeds the compute-matched free
# run used -- so the extra folds are the same seeds spent a different way, not extra compute.
#
# tmux session names are not passed in: call_folder.py derives them from --log-dir's basename, as
# <basename>w<gpu>. That is what the waits below poll for.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/Protenix/.venv/bin/python
D=/home/ebrahim/tts_abag/experimental_data
FIX=$D/self_template_fixed_50_seed
VAR=$D/self_template_variable_50_seed
TMPL=$FIX/templates
LOG=$D/stage_arms.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

launch() {   # launch <queue> <out> <csv> <claims> <logdir> <gpu>
    $PY diffusion_commit_clean_v2/call_folder.py --queue "$1" --out "$2" --csv "$3" \
        --claim-dir "$4" --log-dir "$5" --templates "$TMPL" --gpus "$6" --run >> "$LOG" 2>&1
}

say "=== stage_arms starting ==="
say "waiting for the baseline (free_compute_matched) to drain"
while tmux ls 2>/dev/null | grep -q '^free_compute_matchedw'; do sleep 120; done
claims=$(ls "$D/free_compute_matched/claims" 2>/dev/null | wc -l)
want=$(wc -l < "$D/free_compute_matched/queue_free.tsv")
say "baseline workers gone: $claims/$want claims"
if [ "$claims" -lt "$want" ]; then
    say "REMEDIATE the baseline left $((want-claims)) folds unclaimed -- proceeding to the fixed arm"
fi

say "PRIORITY 1: the fixed-seed arm gets GPUs 1,5,6 (GPU 0 is already on it)"
for g in 1 5 6; do
    launch "$FIX/queue_tofold.tsv" "$FIX/poses" "$FIX/results.csv" "$FIX/claims" "$FIX" "$g"
    say "  launched fixed-seed worker on GPU $g"
done

# As each fixed-arm worker exits its GPU goes to the variable-seed arm. Workers exit when the queue is
# drained, so this is a per-GPU handoff rather than one barrier at the end.
for g in 0 1 5 6; do
    say "GPU $g: waiting for the fixed-seed worker to finish"
    while tmux ls 2>/dev/null | grep -q "^self_template_fixed_50_seedw$g:"; do sleep 120; done
    say "GPU $g: free -- PRIORITY 2, launching the variable-seed arm"
    launch "$VAR/queue.tsv" "$VAR/poses" "$VAR/results.csv" "$VAR/claims" "$VAR" "$g"
done
say "=== all four GPUs handed to the variable-seed arm ==="
