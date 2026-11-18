#!/bin/bash
# CHAIN HEALTH: one line per arm, A through D, and a verdict. Read-only.
#
# WHY THIS EXISTS. Four arms run unattended behind three chained queues. The failure that matters
# is not a crash -- it is a queue that died quietly while the arm it was meant to launch never
# started, which is exactly what an undefined variable under `set -u` did once already. This
# checks the two things that break a chain: is each queue process still alive, and is the arm it
# is waiting for actually progressing.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
STALL=${STALL:-2700}
now=$(date +%s)
bad=0

printf '%-22s %-8s %-10s %-22s %s\n' ARM RID EPOCH "LOG AGE" STATE
for spec in "Arm A:training/runs/pose6_paeonly_rid:queue_pose6_paeonly.sh:20" \
            "Arm B:training/runs/pose6_paeln_rid:queue_pose6_paeln.sh:20" \
            "Arm C (lora4):training/runs/pose6_lora4_rid:queue_lora_chain.sh:20" \
            "Arm D (lora8):training/runs/pose6_lora8_rid:queue_lora_chain.sh:20"; do
    lab=${spec%%:*}; rest=${spec#*:}
    rf=${rest%%:*}; rest=${rest#*:}
    qs=${rest%%:*}; want=${rest##*:}
    q=$(pgrep -f "shell_scripts/$qs" 2>/dev/null | grep -v "^$$\$" | head -1 || true)
    if [ ! -s "$rf" ]; then
        if [ -n "$q" ]; then
            printf '%-22s %-8s %-10s %-22s %s\n' "$lab" "-" "-" "-" "queued, not started"
        else
            printf '%-22s %-8s %-10s %-22s %s\n' "$lab" "-" "-" "-" "NO QUEUE AND NO RUN"
            bad=1
        fi
        continue
    fi
    rid=$(cat "$rf")
    ep=$(ls "training/runs/$rid"/head_ep*.pt 2>/dev/null \
        | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
    ep=${ep:-0}
    if [ -f "training/runs/$rid/run.log" ]; then
        age=$(( now - $(stat -c %Y "training/runs/$rid/run.log") ))
    else
        age=-1
    fi
    if [ "$ep" -ge "$want" ] 2>/dev/null; then
        st="done"
    elif [ "$age" -ge 0 ] && [ "$age" -lt "$STALL" ]; then
        st="training"
    else
        st="STALLED (${age}s)"; bad=1
    fi
    printf '%-22s %-8s %-10s %-22s %s\n' "$lab" "$rid" "$ep/$want" "${age}s" "$st"
done

C=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
echo "cards busy: ${C:-unknown}"
for qs in queue_pose6_paeonly.sh queue_pose6_paeln.sh queue_lora_chain.sh; do
    if pgrep -f "shell_scripts/$qs" 2>/dev/null | grep -qv "^$$\$"; then echo "  queue alive: $qs"; else echo "  queue gone:  $qs"; fi
done
[ "$bad" = 0 ] && echo "VERDICT: ok" || echo "VERDICT: ATTENTION"
exit 0
