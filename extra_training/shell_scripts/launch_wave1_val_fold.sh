#!/bin/bash
# Wave 1 VALIDATION fold: 30 new structures x 100 seeds = 3,000 units, folded AND scored inline.
#
# WHY NOT --fold-only. The first attempt at this ran --fold-only, which (a) deleted the pose cifs
# and left 42 unscoreable units, and (b) auto-launched while the training fold still held the
# cards, putting two folds on each card and roughly doubling per-unit time. Inline scoring costs
# ~3-4 s on a ~25 s unit because --overlap-scoring runs the scorer in a background thread behind
# the next fold; that is far cheaper than a separate rescore pass over 3,000 units.
#
# THE POOL GATE is the check that was missing. folder.py builds its interface registry from
# --pool, so a structure absent from the pool yields "no interfaces to score" and the whole
# queue is skipped silently -- which is exactly what happened on the training side. Refuse to
# launch unless every queued structure is pooled.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration
# ABSOLUTE, BECAUSE THE WORKER RUNS FROM /home/ebrahim/Protenix. Every path below is passed to a
# process whose cwd is the Protenix checkout, so a relative "validation/fold" would resolve there --
# the queue would not be found, and the trunk dump, poses, csv, claims and log would be written
# under Protenix. The pool preflight above runs in THIS shell, from extra_training, so it reads the
# right pool.csv and would not catch the mistake.
EG=/home/ebrahim/confidence_recalibration/extra_training
VG=$EG/validation/fold
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

B=$($PY -I run_logging/cards_busy.py | tail -1)
if [ "${B:-1}" != "0" ]; then
  say "REFUSING: cards_busy reports '$B', not 0. Something is still on cards 4/5/7."
  exit 1
fi

say "pool gate: every queued structure must appear in $VG/pool.csv"
$PY -I - <<'EOF' || exit 1
import csv, sys
q = set()
for ln in open("/home/ebrahim/confidence_recalibration/extra_training/validation/fold/queue.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5: q.add(p[1])
pool = {r["pdb"] for r in csv.DictReader(open("/home/ebrahim/confidence_recalibration/extra_training/validation/fold/pool.csv"))}
missing = sorted(q - pool)
print(f"  queued {len(q)} structure(s), pooled {len(pool)}, missing {len(missing)}")
if missing:
    print(f"  REFUSING: not in the pool -> {missing[:20]}")
    print("  folder.py would report 'no interfaces to score' and skip every unit.")
    sys.exit(1)
EOF

say "launching 3 workers on cards 4, 5, 7 -- fold AND score, no --fold-only"
for c in 4 5 7; do
  tmux new-session -d -s w1val_$c "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$c \
    LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$VG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
    PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $VG/queue.tsv --out $VG/poses \
    --csv $VG/results.csv --claim-dir $VG/claims --pool $VG/pool.csv --dataset sabdab_val2 \
    --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $VG/w1val_c$c.log 2>&1"
done
sleep 60
say "workers up: $(tmux ls 2>/dev/null | grep -c w1val)"
for c in 4 5 7; do echo "  c$c: $(grep -cE 'interfaces in' $VG/w1val_c$c.log 2>/dev/null) unit(s) scored"; done
say "VALIDATION FOLD LAUNCHED"
