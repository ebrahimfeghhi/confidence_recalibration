#!/bin/bash
# Seeds 500011-500014 for the 437 non-degenerate training structures, one seed at a time.
#
# FOUR SEPARATE RUNS, NOT ONE 1,748-UNIT JOB. Each seed is a complete, independently verifiable
# increment: if the third one is interrupted the first two are whole and the training set is
# coherent at 13 seeds rather than ragged at 12.7. It also means the chain can be stopped between
# seeds without leaving half a seed behind, which matters because this is ~5.5 h of GPU.
#
# THE STRUCTURE LIST IS FROZEN, in docs/extra_seed_structures.txt. Degeneracy is measured from
# DockQ spread, and every seed added changes the spread -- so recomputing the list per seed would
# make which structures get seed N depend on what seed N-1 produced. The list is the one computed
# from seeds 500000-500009, the same classification the 290.6 GiB trim acted on.
#
# EACH SEED COSTS ABOUT 106 GiB: measured over these 437 structures' existing ten seeds, 4,370
# trunks hold 1,056.6 GiB, so 105.7 GiB of trunks at 248 MiB/unit plus 0.8 GiB of poses. Four
# seeds is ~426 GiB. The headroom check before each seed is what stops the fourth one discovering
# the quota mid-write, since the quota is a hard limit with no grace period.
#
# IT WAITS FOR THE CURRENT FOLD, and waits on COMPLETENESS rather than on the sessions going away:
# a worker that dies leaves no session either, and treating that as success is how a seed would be
# recorded as folded with units missing.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration
TG=/home/ebrahim/confidence_recalibration/extra_training/training/fold
LIST=docs/extra_seed_structures.txt
SEEDS="${SEEDS:-500011 500012 500013 500014}"
WAIT_FOR="${WAIT_FOR:-500010}"
MIN_HEADROOM_GB="${MIN_HEADROOM_GB:-200}"
MIN_MEM_GB=250
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

N=$(grep -cv '^#' $LIST)
say "chain: seeds [$SEEDS] over $N frozen structures, ~106 GiB each"

headroom(){ $PY -I -c "
import subprocess
o = subprocess.run(['quota','-u','ebrahim'],capture_output=True,text=True).stdout.split()
for i,t in enumerate(o):
    if t.startswith('/dev/'):
        print(int((int(o[i+3])-int(o[i+1]))/2**20)); break
"; }

# out: 0 when every frozen structure has nsample rows at this seed
complete(){ SEED=$1 $PY -I - <<'EOF'
import csv, collections, os, sys
seed = os.environ["SEED"]
want = [l.strip() for l in open("docs/extra_seed_structures.txt") if l.strip() and l[0] != "#"]
got = collections.Counter()
with open("training/fold/results.csv") as fh:
    for r in csv.DictReader(fh):
        if r["seed"] == seed:
            got[r["iface"].split("_")[0]] += 1
miss = [p for p in want if got[p] == 0]
print(f"  seed {seed}: {len(want)-len(miss)} of {len(want)} structure(s) scored")
sys.exit(1 if miss else 0)
EOF
}

say "waiting for seed $WAIT_FOR to be complete before starting"
while ! complete "$WAIT_FOR" >/dev/null 2>&1; do
  S=$(tmux ls 2>/dev/null | grep -cE 'w1fold|w1val' || true)
  if [ "$S" -eq 0 ]; then
    say "seed $WAIT_FOR is INCOMPLETE and no fold session is running -- refusing to chain onto it"
    complete "$WAIT_FOR" || true
    exit 1
  fi
  sleep 120
done
complete "$WAIT_FOR"
say "seed $WAIT_FOR complete"

for SEED in $SEEDS; do
  H=$(headroom); M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  say "--- seed $SEED --- quota headroom ${H} GiB, MemAvailable ${M} GB"
  if [ "${H:-0}" -lt "$MIN_HEADROOM_GB" ]; then
    say "STOPPING: ${H} GiB headroom is under the ${MIN_HEADROOM_GB} GiB floor for a ~106 GiB seed."
    exit 1
  fi
  if [ "${M:-0}" -lt "$MIN_MEM_GB" ]; then
    say "STOPPING: MemAvailable ${M} GB is under the ${MIN_MEM_GB} GB floor for other users."
    exit 1
  fi
  if complete "$SEED" >/dev/null 2>&1; then say "seed $SEED already complete, skipping"; continue; fi

  Q=$TG/queue_seed$SEED.tsv
  grep -v '^#' $LIST | awk -v s=$SEED 'NF{printf "free\t%s\t*\tfree\t%s\n", $1, s}' > "$Q"
  say "queue: $(wc -l < "$Q") unit(s) -> $Q"

  B=$($PY -I run_logging/cards_busy.py | tail -1)
  if [ "${B:-1}" != "0" ]; then say "STOPPING: cards_busy reports '$B', not 0."; exit 1; fi
  for c in 4 5 7; do
    tmux new-session -d -s w1fold_$c "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$c \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$TG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $Q --out $TG/poses \
      --csv $TG/results.csv --claim-dir $TG/claims --pool $TG/pool.csv --dataset sabdab_train \
      --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $TG/w1fold_seed${SEED}_c$c.log 2>&1"
  done
  sleep 45
  UP=$(tmux ls 2>/dev/null | grep -c w1fold || true)
  say "workers up: $UP of 3"
  if [ "$UP" -lt 3 ]; then
    say "STOPPING: only $UP worker(s) came up for seed $SEED."
    exit 1
  fi

  # WAIT ON COMPLETENESS, NOT ON THE SESSIONS. A crashed worker also leaves no session.
  while true; do
    sleep 180
    S=$(tmux ls 2>/dev/null | grep -c w1fold || true)
    if complete "$SEED" >/dev/null 2>&1; then say "seed $SEED complete"; break; fi
    if [ "$S" -eq 0 ]; then
      say "STOPPING: seed $SEED is incomplete and every worker has exited."
      complete "$SEED" || true
      tail -4 $TG/w1fold_seed${SEED}_c4.log
      exit 1
    fi
  done
  complete "$SEED"
done

say "CHAIN COMPLETE: seeds [$SEEDS] folded and scored over $N structures"
say "REMINDER: phase_k_train assumes a uniform seed range under --rank-cross-seed. The set is now"
say "437 structures at 15 seeds against 416 at 1; nothing should train on this until that is fixed."
