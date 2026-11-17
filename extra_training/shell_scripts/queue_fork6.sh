#!/bin/bash
# QUEUE THE SIX-SITE FORK: resume jyz3g2 at epoch 40 with the post-LayerNorm site added, 30 more
# epochs (41-70). Waits for the five-site arm to release the cards, then runs four gated steps and
# refuses at the first one that fails rather than training from a bundle it could not verify.
#
#   1 ORDER   build the six-site model and capture its trainable named_parameters() order. AdamW
#             state is keyed by position (phase_k_train.py:4151), so the fork is only safe if the
#             three new tensors land after the fifteen existing ones. Done with a deliberately
#             wrong --expect-trainable so the trainer prints the list and hard-stops in ~1 min.
#   2 FORK    extend resume_ep40.pt: 15 weights and 15 AdamW moments copied at their original
#             indices, step counter kept, three zero tensors appended with no optimiser state.
#   3 PREFLIGHT  the new matrices are zero, so the forked head MUST reproduce the parent's PAE
#             logits. Score it on valfull83 WITH the sixth site installed and require the iptm
#             column to match jyz3g2's own epoch-40 val rows exactly. If it does not, the site is
#             not zero-init in this build and nothing measured afterwards means anything.
#   4 LAUNCH  only then, 30 epochs from the fork.
#
# EPOCHS IS A STOPPING POINT, NOT A COUNT. The trainer stops AT --epochs, so 30 more from 40 is
# EPOCHS=70.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
SRC=${SRC:-training/runs/jyz3g2/resume_ep40.pt}
PARENT_ROWS=${PARENT_ROWS:-training/runs/jyz3g2/valfull83_eval_shards_ep40}
EPOCHS=${EPOCHS:-70}
POSE6="--pose-readout --pose-readout-prenorm --pose-readout-copy-distance --pose-readout-post-zero"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

say "waiting for the cards (the five-site arm holds them until epoch 60)"
for i in $(seq 1 480); do
    N=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$N" = "0" ] && { say "cards clear"; break; }
    [ "$i" = 480 ] && { say "GIVING UP: cards busy after 8 h"; exit 2; }
    sleep 60
done

# ---- 1 ORDER ----------------------------------------------------------------------------------
say "step 1/4: capturing the six-site trainable order"
OL=$SD/fork6_order_probe.log
TRAIN_ONLY='frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_' \
POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 POSE_READOUT_POST_ZERO=1 \
EXPECT_TRAINABLE=999 EPOCHS=1 LR=1e-4 PAIR_DROPOUT=0.25 ALPHA_RANK=50 \
REL_LABELS=training/fold/itm_labels_crop EVAL_EVERY=0 DRY=0 \
    ./shell_scripts/launch_pae_crop_arm.sh > "$OL" 2>&1
PRID=$(ls -td training/runs/*/ | head -1 | xargs basename)
ORD=$SD/fork6_order.txt
grep -oE '^    [A-Za-z0-9_.]+\.weight' "training/runs/$PRID/run.log" | sed 's/^ *//' \
    | awk '!seen[$0]++' | sed 's/^/confidence_head./' > "$ORD"
say "  order from run $PRID: $(wc -l < "$ORD") tensor(s)"
[ "$(wc -l < "$ORD")" = "18" ] || { say "ABORT: expected 18 trainable tensors, got $(wc -l < "$ORD")"
    tail -5 "training/runs/$PRID/run.log"; exit 3; }

# ---- 2 FORK -----------------------------------------------------------------------------------
say "step 2/4: forking the bundle"
FORK=training/runs/fork6_resume_ep40.pt
$PY -I run_logging/fork_add_postnorm_site.py --src "$SRC" --out "$FORK" --order "$ORD" \
    || { say "ABORT: fork refused"; exit 4; }

# ---- 3 PREFLIGHT ------------------------------------------------------------------------------
say "step 3/4: pre-flight, the fork must reproduce the parent's logits"
HEAD=$SD/fork6_head_ep40.pt
$PY -I -c "
import torch, sys
b = torch.load('$FORK', map_location='cpu', weights_only=False)
torch.save(b['head'], '$HEAD')
print('  wrote a head-only copy,', len(b['head']), 'tensor(s)')
" || { say "ABORT: could not write the head"; exit 5; }
CKPT="$HEAD" TAG=fork6_pre SIDES=val SD="$SD" POSE_FLAGS="--pose-readout --pose-readout-prenorm --pose-readout-copy-distance --pose-readout-post-zero" \
    ./shell_scripts/score_checkpoint.sh > "$SD/fork6_preflight.log" 2>&1 \
    || { say "ABORT: pre-flight scoring failed"; tail -5 "$SD/fork6_preflight.log"; exit 6; }
$PY -I - "$SD/val30_fork6_pre" "$PARENT_ROWS" <<'PYEOF' || { say "ABORT: pre-flight MISMATCH"; exit 7; }
import csv, glob, sys
def load(d):
    out = {}
    for f in glob.glob(d + "/rows_*.csv"):
        for r in csv.DictReader(open(f)):
            out[(r["iface"], r["seed"], r["sample"])] = r["iptm"]
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
both = set(a) & set(b)
bad = [k for k in both if a[k] != b[k]]
print(f"  fork {len(a)} rows, parent {len(b)}, shared {len(both)}, differing iptm {len(bad)}")
if not both or bad:
    for k in bad[:3]:
        print(f"    {k}: fork {a[k]} parent {b[k]}")
    sys.exit(1)
print("  PRE-FLIGHT PASS: the sixth site is zero and changes nothing at step 0")
PYEOF

# ---- 4 LAUNCH ---------------------------------------------------------------------------------
say "step 4/4: launching 30 epochs from the fork, to epoch $EPOCHS"
LOG=training/runs/fork6_launch.log
RESUME="$(pwd)/$FORK" \
TRAIN_ONLY='frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_' \
POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 POSE_READOUT_POST_ZERO=1 \
EXPECT_TRAINABLE=18 EPOCHS=$EPOCHS LR=1e-4 PAIR_DROPOUT=0.25 ALPHA_RANK=50 \
REL_LABELS=training/fold/itm_labels_crop EVAL_EVERY=2 \
EXTRA_EVAL="name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10" \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > "$LOG" 2>&1 &
say "launcher pid $!, log $LOG"
for i in $(seq 1 90); do
    grep -qE "LAUNCH|REFUS|STOP:" "$LOG" 2>/dev/null && break
    sleep 20
done
grep -E "LAUNCH|REFUS|STOP:" "$LOG" | head -3
sleep 60
RID=$(ls -td training/runs/*/ | head -1 | xargs basename)
echo "$RID" > training/runs/fork6_rid
say "run $RID"
