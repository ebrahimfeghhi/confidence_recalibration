#!/bin/bash
# RETRAIN EPOCH 5 WITH THE SPEEDUPS, AND EVALUATE IT PROPERLY.
#
# WHY RESUME FROM EPOCH 4 AND NOT 5. The epoch-5 eval already ran, but without --pool-mean, so it
# wrote only the max-pooled column -- no mean_dockq_select_mean and no iptm_mean, which is also
# why rho came back nan (eval_curve builds it from that column). Re-running just the evaluator
# would mean hand-building phase_k_eval's argv, and retyping argv has cost this session four
# failed launches in twenty minutes. Resuming from epoch 4 lets the TRAINER retrain epoch 5 and
# evaluate it through its own tested path, which now asks for --pool-mean.
#
# WHAT ELSE CHANGES, STATED PLAINLY: the retrained epoch 5 will NOT be bit-identical to the one
# being replaced, because --epoch-balance changes which units each rank walks. It is a fresh
# epoch 5 under the new deal, not a re-evaluation of the old weights.
#
# THE OLD EPOCH-5 ARTEFACTS ARE MOVED, NOT DELETED, at the user's request to remove them "so it
# doesn't cause confusion later". Archiving satisfies that and stays reversible; nothing in this
# project deletes a checkpoint.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:-axrorr}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

BUNDLE="training/runs/$RID/resume_ep4.pt"
[ -s "$BUNDLE" ] || { say "REFUSING: $BUNDLE missing or empty"; exit 2; }
say "resume point $BUNDLE ($(stat -c %s "$BUNDLE") bytes)"

say "stopping $RID by explicit pid (never a pattern: pkill -f has self-killed this session twice)"
PARENT=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk -F, '{print $1}' | sort -u); do
  [ -d "/proc/$p" ] || continue
  c=$({ tr '\0' ' ' < /proc/$p/cmdline; } 2>/dev/null)
  case "$c" in *phase_k_train*"runs/$RID"*) PARENT=$(awk '/^PPid:/{print $2}' /proc/$p/status);; esac
done
[ -n "$PARENT" ] || say "  no live trainer -- already stopped, continuing"
say "  torchrun parent $PARENT -- SIGTERM"
kill -TERM "$PARENT" 2>/dev/null
for _ in $(seq 1 30); do
  B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$B" = "0" ] && { say "  cards released"; break; }
  sleep 10
done
B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "REFUSING: cards still busy ('$B'). NOT relaunching."; exit 4; }

ARCH="training/runs/$RID/superseded_ep5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH" || exit 5
say "archiving the superseded epoch-5 artefacts to $ARCH"
for f in head_ep5_step725.pt resume_ep5.pt metric_valfull83_ep5.json valfull83_eval_shards_ep5; do
  [ -e "training/runs/$RID/$f" ] && { mv "training/runs/$RID/$f" "$ARCH/"; echo "    moved $f"; }
done
cat > "$ARCH/WHY.txt" <<EOF
Superseded $(TZ=America/New_York date '+%Y-%m-%d %-I:%M %p %Z').

These are the FIRST epoch 5 of $RID. Its eval ran without --pool-mean, so it carries only the
max-pooled selection: no mean_dockq_select_mean, no iptm_mean column, and therefore rho is nan.
Epoch 5 was retrained from resume_ep4.pt with --trunk-memmap and --epoch-balance, and evaluated
with both poolings. The replacement is NOT bit-identical: --epoch-balance changes which units
each rank walks.

Kept rather than deleted. metric_valfull83_ep5.json here reads mean_dockq 0.5850, frac_acc
0.7234, select_by max, against the shipped max-pooled baseline of 0.5813 / 0.7234.
EOF
say "archived; $(ls "$ARCH" | wc -l) item(s)"
sleep 30

PREV=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
say "LAUNCH: resume from epoch 4, with memmap + balance, 16 workers"
RESUME="$BUNDLE" PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 CROP_THREADS=3 \
  RESUME_OVERRIDE=epoch_balance \
  RESUME_NOTE="epoch_balance off->on at the user's direction. It reorders which units each rank \
walks within a step; it does not change the loss, the population, the batch size or the schedule, \
and epoch 5 is being retrained from the epoch-4 bundle rather than continued mid-epoch. AdamW \
moments are per-tensor and do not encode which rank saw which structure. Named field only, never \
--resume-allow-drift, so any OTHER divergence still stops the run." \
  ./shell_scripts/launch_pae_crop_arm.sh > "training/runs/redo_ep5_launch.log" 2>&1

NEW=""
for _ in $(seq 1 60); do
  C=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
  [ -n "$C" ] && [ "$C" != "${PREV:-}" ] && { NEW="$C"; break; }
  sleep 10
done
[ -n "$NEW" ] || { say "DID NOT START:"; grep -aE "REFUS|STOP|Traceback" training/runs/redo_ep5_launch.log | tail -8; exit 6; }
say "new run $NEW"
for _ in $(seq 1 90); do
  grep -qaE "steps/epoch|STOP:|Traceback" "training/runs/$NEW/run.log" 2>/dev/null && break
  sleep 20
done
grep -aE "RESUMED from|continuing at epoch|epoch plan \(cover-all\)|steps/epoch" "training/runs/$NEW/run.log" | head -4
say "done -- $NEW retrains epoch 5 then evaluates it with both poolings"
