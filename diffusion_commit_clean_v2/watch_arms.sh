#!/bin/bash
# Every 30 minutes, check that each live arm is BOTH folding and scoring, and remediate when it is not.
#
# Two separate things are checked because they fail separately, and the 603-failure episode was exactly
# this: a worker was up and staking claims while every fold died in featurisation, so "the worker is
# alive" read as progress while results.csv never moved.
#
#   claims growing   the worker is picking up work
#   rows growing     the folds are actually producing scored poses
#
# A worker alive with claims growing and rows flat is the dangerous state: it is burning GPU and
# producing nothing. That is escalated, not just logged.
#
# ARMS lists every arm watched, as label|directory|tmux session prefix. The label is what the log and
# the state file use; the prefix is how its workers are counted (call_folder.py names sessions
# <prefix><gpu>). pxm2025 and FoldBench both have a self_template_fixed_50_seed, hence the labels.
set -u
cd /home/ebrahim/tts_abag
D=/home/ebrahim/tts_abag/experimental_data
LOG=$D/watch_arms.log
STATE=$D/.watch_state
FB=$D/protenix/foldbench
ARMS=(
    "self_template_fixed_50_seed|$D/self_template_fixed_50_seed|self_template_fixed_50_seedw"
    "self_template_variable_50_seed|$D/self_template_variable_50_seed|self_template_variable_50_seedw"
    "free_compute_matched|$D/free_compute_matched|free_compute_matchedw"
    "foldbench_fixed|$FB/self_template_fixed_50_seed|fb_fixedw"
)
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

rows() { [ -f "$1/results.csv" ] && echo $(( $(wc -l < "$1/results.csv") - 1 )) || echo 0; }
claims() { ls "$1/claims" 2>/dev/null | wc -l; }

while true; do
    touch "$STATE"
    for entry in "${ARMS[@]}"; do
        IFS='|' read -r arm A prefix <<< "$entry"
        [ -d "$A" ] || continue
        w=$(tmux ls 2>/dev/null | grep -c "^${prefix}")
        c=$(claims "$A"); r=$(rows "$A")
        prev=$(grep "^$arm " "$STATE" 2>/dev/null | tail -1)
        pc=$(echo "$prev" | awk '{print $2}'); pr=$(echo "$prev" | awk '{print $3}')
        pc=${pc:-0}; pr=${pr:-0}
        fails=$(grep -c FAILED "$A"/w*.log 2>/dev/null | awk -F: '{t+=$2} END{print t+0}')
        say "$arm: workers=$w claims=$c (+$((c-pc))) rows=$r (+$((r-pr))) failed=$fails"

        if [ "$w" -eq 0 ]; then
            say "  $arm: no workers -- idle or finished"
        elif [ "$c" -gt "$pc" ] && [ "$r" -eq "$pr" ]; then
            say "  REMEDIATE $arm is claiming work but scoring NOTHING -- folds are failing"
            say "  last failure: $(grep -h FAILED "$A"/w*.log 2>/dev/null | tail -1)"
        elif [ "$c" -eq "$pc" ] && [ "$r" -eq "$pr" ]; then
            say "  REMEDIATE $arm has made no progress in 30 min with $w workers up"
        fi
        echo "$arm $c $r" >> "$STATE"
    done
    # disk: the arms keep every pose (~3.5 MB per fold); warn well before the 1024G quota
    used=$(quota -s 2>/dev/null | tail -1 | awk '{print $2}')
    usedg=${used%G}
    say "disk: $used of 1024G"
    if [ "${usedg%%.*}" -ge 950 ] 2>/dev/null; then
        say "  REMEDIATE disk at $used -- free space before the quota stops the arms"
    fi
    say "---"
    sleep 1800
done
