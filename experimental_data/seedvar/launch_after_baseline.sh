#!/bin/bash
# Start the seedvar workers once the free_matched baseline has drained.
#
# Waits for the baseline's workers to exit, then launches seedvar on the same GPUs. Escalates rather
# than launching if the baseline ended with work left unclaimed, so a crashed baseline does not get
# silently replaced by the next experiment.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS=1,5,6
BASE=experimental_data/free_matched
EXP=experimental_data/seedvar
LOG=$EXP/launch_after_baseline.log

say() { echo "$(date '+%F %T') $*" >> $LOG; }

say "waiting for baseline workers to exit"
while tmux ls 2>/dev/null | grep -q '^free_matchedw'; do sleep 120; done

claims=$(ls $BASE/claims 2>/dev/null | wc -l)
want=$(wc -l < $BASE/queue_free.tsv)
say "baseline workers gone: $claims/$want claims"
if [ "$claims" -lt "$want" ]; then
    say "REMEDIATE baseline left $((want-claims)) folds unclaimed -- NOT launching seedvar"
    exit 1
fi

say "launching seedvar on GPUs $GPUS"
$PY diffusion_commit_clean_v2/call_folder.py \
    --queue $EXP/queue.tsv \
    --out $EXP/out \
    --csv $EXP/results.csv \
    --claim-dir $EXP/claims \
    --log-dir $EXP \
    --templates $EXP/ns \
    --gpus $GPUS --run >> $LOG 2>&1
say "call_folder.py exited $?"
sleep 180
n=$(tmux ls 2>/dev/null | grep -c '^seedvarw')
say "seedvar worker sessions up: $n"
[ "$n" -eq 0 ] && say "REMEDIATE no seedvar workers came up"

say "scoring the baseline against the method"
$PY diffusion_commit_clean_v2/compare_arms.py \
    --out experimental_results/arms_compare.md >> $LOG 2>&1
say "compare_arms.py exited $?"
