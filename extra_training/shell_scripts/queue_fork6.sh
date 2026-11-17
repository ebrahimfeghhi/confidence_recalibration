#!/bin/bash
# QUEUE THE SIX-SITE FORK: resume jyz3g2 at epoch 40 with the post-LayerNorm site added, 30 more
# epochs (41-70). Waits for the five-site arm to release the cards, then runs four gated steps and
# refuses at the first one it cannot verify.
#
# THREE DEFECTS FROM THE FIRST VERSION, all found in review and all fatal, are fixed here:
#
#   THE PROBE SUPPRESSED THE LIST IT EXISTED TO CAPTURE. phase_k_train.py:3613 raises on a wrong
#   --expect-trainable, and the print loop that names every trainable tensor is at :3619, AFTER
#   it. So EXPECT_TRAINABLE=999 produced only a six-name "kept e.g." repr and the parse got zero
#   lines. The probe now passes the CORRECT count so the list prints, and is stopped by pid once
#   the names are on disk.
#
#   --pose-readout-copy-distance IS NOT AN EVALUATOR ARGUMENT. phase_k_eval.py has no such flag
#   and says why: every head tensor comes from the checkpoint, so a copy-initialisation would be
#   overwritten. Passing it killed all three shards with an argparse error. POSE_EVAL_FLAGS now
#   carries only the three the evaluator takes, and is the single place they are written.
#
#   THE RESUME GUARD COMPARES pose_readout_post_zero (phase_k_train.py:4202). The bundle records
#   False and the fork sets True, so the run stops unless that field is named in
#   --resume-override with a note. Both are now passed. This is the one field that differs: every
#   other compared field matches the bundle.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
SRC=${SRC:-training/runs/jyz3g2/resume_ep40.pt}
PARENT_ROWS=${PARENT_ROWS:-training/runs/jyz3g2/valfull83_eval_shards_ep40}
EPOCHS=${EPOCHS:-70}
NTRAIN=${NTRAIN:-18}
# THE EVALUATOR'S THREE, NOT THE TRAINER'S FOUR. Written once, used once.
POSE_EVAL_FLAGS="--pose-readout --pose-readout-prenorm --pose-readout-post-zero"
TRAIN6="POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 POSE_READOUT_POST_ZERO=1"
TONLY='frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_'
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest(){ ls -td training/runs/*/ 2>/dev/null | head -1 | xargs -r basename; }

say "waiting for the cards (the five-site arm holds them until epoch 60)"
for i in $(seq 1 480); do
    N=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$N" = "0" ] && { say "cards clear"; break; }
    [ "$i" = 480 ] && { say "GIVING UP: cards busy after 8 h"; exit 2; }
    sleep 60
done

# ---- 1 ORDER ----------------------------------------------------------------------------------
# THE CORRECT COUNT, SO THE LIST PRINTS, then stopped by pid. The probe is killed as soon as the
# names are on disk -- it is only here to enumerate parameters, and letting it train would both
# waste a card-hour and leave a checkpoint that looks like an arm.
say "step 1/4: capturing the six-site trainable order (probe will be stopped once it prints)"
BEFORE=$(newest)
OL=$SD/fork6_order_probe.log
env TRAIN_ONLY="$TONLY" $TRAIN6 EXPECT_TRAINABLE=$NTRAIN EPOCHS=1 LR=1e-4 PAIR_DROPOUT=0.25 \
    ALPHA_RANK=50 REL_LABELS=training/fold/itm_labels_crop EVAL_EVERY=0 AUTOPLOT=0 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > "$OL" 2>&1 &
LP=$!
PRID=""
for i in $(seq 1 90); do
    N=$(newest)
    [ -n "$N" ] && [ "$N" != "$BEFORE" ] && [ -f "training/runs/$N/run.log" ] && { PRID="$N"; break; }
    grep -qE "REFUS|STOP:" "$OL" && { say "ABORT: probe refused"; grep -E "REFUS|STOP:" "$OL" | head -3; exit 3; }
    [ -d "/proc/$LP" ] || { say "ABORT: probe launcher exited before a run appeared"; tail -5 "$OL"; exit 3; }
    sleep 10
done
[ -n "$PRID" ] || { say "ABORT: no probe run directory appeared"; exit 3; }
say "  probe run $PRID"
ORD=$SD/fork6_order.txt
for i in $(seq 1 60); do
    grep -oE '^    [A-Za-z0-9_.]+\.weight' "training/runs/$PRID/run.log" 2>/dev/null \
        | sed 's/^ *//' | awk '!seen[$0]++' | sed 's/^/confidence_head./' > "$ORD"
    [ "$(wc -l < "$ORD")" -ge "$NTRAIN" ] && break
    grep -qE "STOP:|Traceback" "training/runs/$PRID/run.log" 2>/dev/null \
        && { say "ABORT: probe stopped before printing the list"
             grep -E "STOP:" "training/runs/$PRID/run.log" | head -2; exit 3; }
    sleep 10
done
# STOP THE PROBE BY EXPLICIT PID, never by pattern.
for p in $(grep -oE "rank [0-9] pid [0-9]+" "training/runs/$PRID/run.log" 2>/dev/null | grep -oE "[0-9]+$"); do
    kill "$p" 2>/dev/null
done
kill $LP 2>/dev/null
for p in $(ps -eo pid,ppid --no-headers | awk -v l=$LP '$2==l{print $1}'); do kill "$p" 2>/dev/null; done
sleep 20
say "  order: $(wc -l < "$ORD") tensor(s); probe stopped"
[ "$(wc -l < "$ORD")" = "$NTRAIN" ] || { say "ABORT: expected $NTRAIN tensors, got $(wc -l < "$ORD")"; exit 3; }
for i in $(seq 1 30); do
    [ "$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)" = "0" ] && break
    sleep 10
done

# ---- 2 FORK -----------------------------------------------------------------------------------
say "step 2/4: forking the bundle"
FORK=training/runs/fork6_resume_ep40.pt
$PY -I run_logging/fork_add_postnorm_site.py --src "$SRC" --out "$FORK" --order "$ORD" \
    || { say "ABORT: fork refused"; exit 4; }

# ---- 3 PREFLIGHT ------------------------------------------------------------------------------
say "step 3/4: pre-flight, the fork must reproduce the parent's logits"
HEAD=$SD/fork6_head_ep40.pt
$PY -I -c "
import torch
b = torch.load('$FORK', map_location='cpu', weights_only=False)
torch.save(b['head'], '$HEAD')
print('  head-only copy:', len(b['head']), 'tensor(s)')" \
    || { say "ABORT: could not write the head"; exit 5; }
CKPT="$HEAD" TAG=fork6_pre SIDES=val SD="$SD" POSE_FLAGS="$POSE_EVAL_FLAGS" \
    ./shell_scripts/score_checkpoint.sh > "$SD/fork6_preflight.log" 2>&1
# score_report asserts 141x150 and runs even on a val-only pass; the rows are what matters, so a
# non-zero exit is reported but the gate below decides.
$PY -I - "$SD/val30_fork6_pre" "$PARENT_ROWS" <<'PYEOF' || { say "ABORT: pre-flight MISMATCH"; exit 7; }
import csv, glob, sys
def load(d):
    out = {}
    for f in glob.glob(d + "/rows_*.csv"):
        for r in csv.DictReader(open(f)):
            out[(r["iface"], r["seed"], r["sample"])] = float(r["iptm"])
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
both = set(a) & set(b)
print(f"  fork {len(a)} rows, parent {len(b)}, shared {len(both)}")
# EVERY ROW, NOT SOME. A pass that lost a shard would leave ~2/3 of the keys matching and a
# subset gate would call that a pass.
if len(a) != len(b) or len(both) != len(b) or not both:
    print("  FAIL: the two passes do not cover the same candidates")
    sys.exit(1)
d = max(abs(a[k] - b[k]) for k in both)
print(f"  max |delta iptm| over {len(both)} rows: {d:.3e}")
# A TOLERANCE, SO THE FAILURE IS READABLE. Zero-init means bit-identical, but two separate GPU
# passes can differ in the last bits from kernel selection. A non-inert sixth site would move
# iptm by orders of magnitude more than this.
if d > 1e-6:
    print("  FAIL: the sixth site is NOT inert at step 0")
    sys.exit(1)
print("  PRE-FLIGHT PASS")
PYEOF

# ---- 4 LAUNCH ---------------------------------------------------------------------------------
say "step 4/4: launching 30 epochs from the fork, to epoch $EPOCHS"
BEFORE=$(newest)
LOG=training/runs/fork6_launch.log
env RESUME="$(pwd)/$FORK" \
    RESUME_OVERRIDE=pose_readout_post_zero \
    RESUME_NOTE="six-site fork of jyz3g2 at epoch 40: the post-LayerNorm readout site is ADDED, zero-init, with fresh AdamW state; every other setting and all 15 existing weights and moments are the parent's." \
    TRAIN_ONLY="$TONLY" $TRAIN6 EXPECT_TRAINABLE=$NTRAIN EPOCHS=$EPOCHS LR=1e-4 \
    PAIR_DROPOUT=0.25 ALPHA_RANK=50 REL_LABELS=training/fold/itm_labels_crop EVAL_EVERY=2 \
    EXTRA_EVAL="name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > "$LOG" 2>&1 &
LP=$!
for i in $(seq 1 90); do
    grep -qE "LAUNCH|REFUS|STOP:" "$LOG" 2>/dev/null && break
    [ -d "/proc/$LP" ] || break
    sleep 20
done
if grep -qE "REFUS|STOP:" "$LOG"; then
    say "NOT LAUNCHED: the launcher refused"
    grep -E "REFUS|STOP:" "$LOG" | head -3
    exit 8
fi
# THE RUN ID COMES FROM A DIRECTORY THAT DID NOT EXIST BEFORE THIS STEP. The first version wrote
# whatever was newest, which after a refusal was the dead probe.
RID=""
for i in $(seq 1 30); do
    N=$(newest)
    [ -n "$N" ] && [ "$N" != "$BEFORE" ] && { RID="$N"; break; }
    sleep 10
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; exit 8; }
echo "$RID" > training/runs/fork6_rid
say "run $RID -- 6 sites, resumed at epoch 40, to epoch $EPOCHS"
