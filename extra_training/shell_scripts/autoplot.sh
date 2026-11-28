#!/bin/bash
# WATCH FOR THE RUN THIS LAUNCH IS ABOUT TO CREATE, then keep its figure redrawn every epoch and
# its health checked -- without anyone remembering to start a watcher.
#
#     ./shell_scripts/autoplot.sh            # waits for the next new run directory
#     ./shell_scripts/autoplot.sh <rid>      # attaches to one that already exists
#
# WHY A WATCHER CANNOT JUST BE STARTED AFTER THE LAUNCH. run_logging/launch.py calls
# subprocess.run, so launch_pae_crop_arm.sh does not return until the ARM HAS FINISHED -- every
# chain that started its watchers on the line after the launcher started them hours late or never.
# On 2026-11-14 that left a resumed arm unwatched for 45 minutes and a 20-epoch arm with no
# per-epoch figure at all. So this starts BEFORE the launch and discovers the run itself.
#
# IT IDENTIFIES THE RUN BY DIRECTORY MTIME, NOT BY SCRAPING A LOG. The launcher prints no run id
# in a form worth matching, and a chain that waited on `grep "^run <rid>"` waited forever. The
# newest directory under training/runs that did not exist when this started IS the run.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RUNS=training/runs
WAIT_S="${WAIT_S:-1800}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') autoplot: $*"; }
newest() { ls -td $RUNS/*/ 2>/dev/null | grep -vE "/latest/$" | head -1 | xargs -r basename; }

RID="${1:-}"
if [ -z "$RID" ]; then
    BEFORE=$(newest)
    say "waiting for a new run directory (newest now: ${BEFORE:-none})"
    for _ in $(seq 1 $((WAIT_S / 10))); do
        N=$(newest)
        [ -n "$N" ] && [ "$N" != "$BEFORE" ] && { RID="$N"; break; }
        sleep 10
    done
    [ -n "$RID" ] || { say "no new run appeared within ${WAIT_S}s; nothing to plot"; exit 2; }
fi
say "attached to $RID"

# args.json IS THE HANDSHAKE. The plotter reads the epoch count from it, and health_watch refuses
# without it, so both wait here rather than racing a directory that exists but is still empty.
for _ in $(seq 1 120); do [ -f "$RUNS/$RID/args.json" ] && break; sleep 10; done
[ -f "$RUNS/$RID/args.json" ] || { say "$RID never wrote args.json; it died in its gates"; exit 3; }

echo 0 > "$RUNS/${RID}_last_plotted"
nohup ./shell_scripts/plot_on_epoch.sh "$RID" > "$RUNS/$RID/plot_on_epoch.log" 2>&1 &
nohup ./shell_scripts/health_watch.sh  "$RID" > "$RUNS/$RID/health.log" 2>&1 &
say "plot_on_epoch and health_watch started on $RID"
say "  figure: figures/rel_arm_$RID.png   per-epoch csv: $RUNS/$RID/spread.r0.csv"
