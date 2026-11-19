#!/usr/bin/env bash
# SUPERVISOR FOR THE REL-ALPHA SWEEP. Replaces queue_rel_alpha_sweep.sh + overnight_rel_sweep.sh,
# which an audit found would end the whole chain on any single transient refusal and could not
# tell a finished arm from a dead one.
#
# WHAT CHANGED, AND WHY EACH ONE MATTERS OVERNIGHT:
#
#   1. A REFUSAL RETRIES THE SAME ALPHA instead of exiting the sweep. The launcher refuses if
#      cards_busy != 0, if any of cards 4/5/7 has <80 GB free, or if MemAvailable < 250 GB. This
#      is a SHARED box -- other users hold cards 0,1,2,3,6 right now. A neighbour straying onto
#      one of ours for sixty seconds at 3 AM used to convert "wait for the box" into "sweep over,
#      remaining alphas silently dropped".
#
#   2. A DEATH IS DETECTED AND NAMED. The old finish test was "cards_busy went to 0", which is
#      equally true of an OOM at epoch 2, an --expect-trainable mismatch, and a failed eval. It
#      then printed "alpha N finished" either way. Now the epoch-10 checkpoint and the val metric
#      are required, and an arm that lacks them is logged as DIED with its epoch count. Moving on
#      to the next alpha is still correct -- the other arms are still wanted -- but the morning
#      table must not show a two-epoch arm as a result.
#
#   3. A WEDGED ARM IS NOT CALLED FINISHED. The finish wait used to fall out of its loop after
#      ~8 h and print "finished" with no check, so a trainer wedged while holding the cards would
#      be reported as a completed arm and the NEXT alpha would then burn its own 4 h card wait and
#      give up. Falling out of the loop now says STALLED and stops.
#
#   4. IT RESUMES. Any alpha whose arm already has an epoch-10 val is skipped, so this can be
#      restarted after a crash, or pointed at the full list, without redoing finished work.
#
# IT DOES NOT KILL ANYTHING, EVER -- not a wedged arm, not a neighbour. It reports and stops.
set -u
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python
ALPHAS="${ALPHAS:-100 200 50 250 300 25}"
TRIES="${TRIES:-6}"

cards(){ n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
         case "$n" in ''|*[!0-9]*) n=99;; esac; echo "$n"; }

# AN ARM COUNTS AS DONE ONLY WITH BOTH ITS EPOCH-10 CHECKPOINT AND ITS VAL METRIC.
done_alpha(){
    local f="training/runs/rel_alpha$1_rid" rid
    [ -f "$f" ] || return 1
    rid=$(cat "$f")
    [ -f "training/runs/$rid/metric_valfull83_ep10.json" ] || return 1
    ls "training/runs/$rid"/head_ep10_*.pt >/dev/null 2>&1
}
epochs_done(){ ls "training/runs/$1"/head_ep*.pt 2>/dev/null \
               | sed -E 's|.*head_ep([0-9]+)_.*|\1|' | sort -n | tail -1; }

wait_free(){                      # $1 = minutes to wait
    local i
    for i in $(seq 1 $(( $1 * 2 ))); do
        [ "$(cards)" -eq 0 ] && return 0
        sleep 30
    done
    return 1
}

for ALPHA in $ALPHAS; do
    if done_alpha "$ALPHA"; then
        say "alpha $ALPHA already complete ($(cat training/runs/rel_alpha${ALPHA}_rid)) -- skipping"
        continue
    fi
    # AN ALPHA ALREADY ON THE CARDS IS ADOPTED, NOT RELAUNCHED. The first arm of the chain was
    # started by the previous supervisor; launching a second copy of it would be refused by the
    # card gate, and treating that refusal as failure would drop the alpha that is running fine.
    RID=""
    if [ -f "training/runs/rel_alpha${ALPHA}_rid" ] && [ "$(cards)" -ne 0 ]; then
        RID=$(cat "training/runs/rel_alpha${ALPHA}_rid")
        if [ -d "training/runs/$RID" ]; then
            say "alpha $ALPHA is already running as $RID -- adopting it"
        else
            RID=""
        fi
    fi

    if [ -z "$RID" ]; then
        ok=0
        for try in $(seq 1 "$TRIES"); do
            say "alpha $ALPHA: waiting for the cards (attempt $try/$TRIES)"
            wait_free 240 || { say "alpha $ALPHA: cards busy for 4 h; retrying"; continue; }
            SEEN=$SD/rel_${ALPHA}_before.txt
            ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" \
              | xargs -r -n1 basename | sort > "$SEEN"
            LOG=training/runs/rel_alpha${ALPHA}_launch.log
            say "alpha $ALPHA: launching"
            ALPHA_RANK="$ALPHA" DRY=0 nohup ./shell_scripts/launch_armE_rel.sh > "$LOG" 2>&1 &
            for i in $(seq 1 120); do          # 20 min, not 10: a cold trunk scan can overrun
                RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" \
                      | xargs -r -n1 basename | sort | comm -13 "$SEEN" - | head -1)
                [ -n "$RID" ] && break
                if grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null; then
                    say "alpha $ALPHA: REFUSED -- $(grep -E 'REFUS|STOP:' "$LOG" | head -1)"
                    break
                fi
                sleep 10
            done
            [ -n "$RID" ] && { ok=1; break; }
            say "alpha $ALPHA: no run directory on attempt $try; backing off 10 min"
            sleep 600
        done
        [ "$ok" -eq 1 ] || { say "alpha $ALPHA: GIVING UP after $TRIES attempt(s); continuing to the next alpha"; continue; }
    fi

    echo "$RID" > "training/runs/rel_alpha${ALPHA}_rid"
    echo "$RID" > training/runs/pose6_norms_rid
    printf 'rel a%s' "$ALPHA" > "training/runs/${RID}_name"
    echo 73rf4v > training/runs/plot_reference_rid
    [ -f "training/runs/${RID}_last_autoplot" ] || echo 0 > "training/runs/${RID}_last_autoplot"
    say "alpha $ALPHA is run $RID"

    for i in $(seq 1 120); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
    if [ -s "training/runs/$RID/args.json" ]; then
        "$PY" -I - "$RID" "$ALPHA" <<'PYEOF'
import json, sys
rid, alpha = sys.argv[1], float(sys.argv[2])
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"expect_trainable": 53, "lr": 0.001, "epochs": 10.0, "pair_dropout": 0.25,
        "pae_split_accept": 0.23, "pae_balance_accept": 0.0, "lora_rank": 0,
        "pose_readout_post_zero": True, "train_eval_every": 10,
        "rank_loss": True, "rank_form": "rel", "alpha_rank": alpha}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
if bad:
    print("  ARGS MISMATCH:")
    for k, (got, exp) in bad.items(): print(f"    {k}: got {got!r}, expected {exp!r}")
    raise SystemExit(6)
print(f"  args confirmed: alpha {a.get('alpha_rank')}, {a.get('epochs')} epochs, "
      f"{a.get('expect_trainable')} tensors, labels {a.get('rel_labels')}")
PYEOF
        [ $? -eq 0 ] || say "alpha $ALPHA: ARGS MISMATCH on $RID -- it is running anyway; READ THE LINES ABOVE"
    else
        say "alpha $ALPHA: $RID has no args.json after 20 min"
    fi

    sleep 120
    if wait_free 480; then
        if done_alpha "$ALPHA"; then
            say "alpha $ALPHA COMPLETE ($RID, 10 epochs + val)"
        else
            say "alpha $ALPHA DIED ($RID) after $(epochs_done "$RID") epoch(s) -- no epoch-10 val; continuing"
        fi
    else
        say "alpha $ALPHA STALLED ($RID): cards still held after 8 h and no completion. STOPPING the sweep."
        exit 9
    fi
done

say "sweep done. status:"
for ALPHA in $ALPHAS; do
    if done_alpha "$ALPHA"; then say "  alpha $ALPHA  OK   $(cat training/runs/rel_alpha${ALPHA}_rid)"
    elif [ -f "training/runs/rel_alpha${ALPHA}_rid" ]; then
        say "  alpha $ALPHA  DIED $(cat training/runs/rel_alpha${ALPHA}_rid)"
    else say "  alpha $ALPHA  NEVER STARTED"; fi
done
