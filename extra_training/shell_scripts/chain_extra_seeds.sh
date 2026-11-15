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
LIST="${LIST:-docs/extra_seed_structures.txt}"
export LIST   # the python heredocs below read it; they used to hard-code the wave-1 path
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

# out: 0 when this seed is complete. STRICT_IFACE=1 asks PER INTERFACE, against the interfaces a
# reference seed proves are scorable -- pool.csv names three (6gv4_BE, 8dv6_BD, 9bbn_FH) that no
# route has ever scored, so counting against pool.csv demands rows nothing can produce. Default is
# the wave-1 behaviour: a structure with zero rows at this seed is missing, nothing finer.
complete(){
  if [ "${STRICT_IFACE:-0}" = "1" ]; then
    $PY -I run_logging/seed_completeness.py --csv "$TG/results.csv" --structures "$LIST" \
        --seed "$1" --ref-seed "${REF_SEED:-500000}" --nsample "${NSAMPLE:-5}"
    return $?
  fi
  SEED=$1 $PY -I - <<'EOF'
import csv, collections, os, sys
seed = os.environ["SEED"]
want = [l.strip() for l in open(os.environ["LIST"]) if l.strip() and l[0] != "#"]
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

# THE INLINE SCORER CANNOT SCORE EVERY INTERFACE, so a seed is not complete when the fold exits.
# Measured on seed 500010: the fold left 1wej, 4p3c and 9blx with zero rows while all three had a
# trunk and five sample cifs, and the chain's own gate then refused to start 500011 -- correctly,
# but it meant the chain stopped after ONE seed and waited for a human. 1wej and 4p3c are in
# docs/recovery_manifest_wave1.csv: OST groups the antibody H and L chains into one chemical class,
# leaves one reference chain unassigned and silently drops the interface, so the inline scorer has
# never scored them at ANY seed and never will. Their earlier rows all came from a recovery sweep.
#
# So the recovery is part of folding a seed, not a repair of it. This runs the pair-isolation route
# over whatever the fold left unscored, merges it under the usual gates, and only then asks whether
# the seed is complete. If the sweep cannot close the gap the gate still refuses, which is the
# behaviour that caught this in the first place.
#
# --roles IS chain_roles_all.csv, NOT chain_roles.csv. The latter names 501 structures and none of
# this wave's 356, and the sweep then fails every sample with "roles expect ('?', '?')" -- which is
# what happened on the first attempt at 1wej and 4p3c. chain_roles_all.csv covers 853 of 853.
# out: 0 when every structure in LIST has a trunk npz at this seed. A ROW IS NOT A TRUNK: the whole
# point of this wave is that rows survived a prune and trunks did not, and complete() reads rows. Off
# unless TRUNK_GATE=1, so the wave-1 chain's behaviour is unchanged.
trunks_present(){
  [ "${TRUNK_GATE:-0}" = "1" ] || return 0
  local SEED=$1 MISS=0 P
  while read -r P; do
    [ -s "$TG/trunks/$P/free_$SEED.npz" ] || { MISS=$((MISS+1)); }
  done < <(grep -v '^#' "$LIST" | awk 'NF')
  if [ "$MISS" -gt 0 ]; then say "  seed $SEED: $MISS structure(s) have NO trunk npz"; return 1; fi
  return 0
}

recover_seed(){
  local SEED=$1
  local SC="$TG/.consolidate"
  mkdir -p "$SC"
  local T
  if [ "${STRICT_IFACE:-0}" = "1" ]; then
    T=$($PY -I run_logging/seed_completeness.py --csv "$TG/results.csv" --structures "$LIST" \
          --seed "$SEED" --ref-seed "${REF_SEED:-500000}" --nsample "${NSAMPLE:-5}" --mode targets)
  else
  T=$(SEED=$SEED $PY -I - <<'EOF'
import csv, collections, os
seed = os.environ["SEED"]
want = [l.strip() for l in open(os.environ["LIST"]) if l.strip() and l[0] != "#"]
got = collections.Counter()
for r in csv.DictReader(open("training/fold/results.csv")):
    if r["seed"] == seed:
        got[r["iface"].split("_")[0]] += 1
miss = {p for p in want if got[p] == 0}
# THE POOL NAMES THE INTERFACES, and only sorted single-letter pairs are addressable by the sweep.
out = []
for r in csv.DictReader(open("training/fold/pool.csv")):
    if r["pdb"] in miss:
        c = "".join(sorted((r["chain1"], r["chain2"])))
        if len(c) == 2:
            out.append(f"{r['pdb']}:{r['pdb']}_{c}")
print(",".join(sorted(set(out))))
EOF
)
  fi
  if [ -z "$T" ]; then say "  seed $SEED: nothing left unscored, no recovery needed"; return 0; fi
  local N; N=$(echo "$T" | tr ',' '
' | wc -l)
  say "  seed $SEED: $N interface(s) unscored by the fold; running the isolated route"
  $PY -I phases/phase_j_rescore_unmapped.py --targets "$T"     --fold "$TG" --refs training/refs --roles "$TG/chain_roles_all.csv"     --seed0 "$SEED" --seeds 1 --nsample 5 --route isolated     --rows-out "$SC/rows_$SEED.csv" || { say "  sweep exit $?"; return 1; }
  [ -s "$SC/rows_$SEED.csv" ] || { say "  sweep produced no rows"; return 1; }
  local R; R=$(($(wc -l < "$SC/rows_$SEED.csv") - 1))
  # THE MANIFEST NAMES WHICH INTERFACES MAY BE ADDED, which the merge requires with --replace.
  { echo "iface,route"; echo "$T" | tr ',' '
' | cut -d: -f2 | sort -u | sed 's/$/,iso/'; }     > "$SC/manifest_$SEED.csv"
  local I; I=$(($(wc -l < "$SC/manifest_$SEED.csv") - 1))
  # --out MUST SIT BESIDE THE CSV. os.replace is atomic only within one filesystem and raises
  # EXDEV across devices; pointing it at the session scratchpad failed exactly that way AFTER
  # every check had passed.
  $PY -I phases/phase_l_merge_recovered.py --csv "$TG/results.csv"     --rows "iso=$SC/rows_$SEED.csv" --manifest "$SC/manifest_$SEED.csv"     --expect-rows "$R" --expect-ifaces "$I" --seeds 1 --seed0 "$SEED" --nsample 5     --out "$SC/merged_$SEED.csv" --replace || { say "  merge exit $?"; return 1; }
  say "  seed $SEED: merged $R recovered row(s) over $I interface(s)"
}

if [ "$WAIT_FOR" = none ]; then
  say "WAIT_FOR=none: not gating on a previous seed (this list was never in the wave-1 fold)"
else
say "waiting for seed $WAIT_FOR to be complete before starting"
while ! complete "$WAIT_FOR" >/dev/null 2>&1; do
  S=$(tmux ls 2>/dev/null | grep -cE 'w1fold|w1val' || true)
  if [ "$S" -eq 0 ]; then
    say "seed $WAIT_FOR: no fold session running and still incomplete -- attempting recovery"
    recover_seed "$WAIT_FOR" || true
    if complete "$WAIT_FOR" >/dev/null 2>&1; then break; fi
    say "seed $WAIT_FOR is INCOMPLETE after recovery -- refusing to chain onto it"
    complete "$WAIT_FOR" || true
    exit 1
  fi
  sleep 120
done
complete "$WAIT_FOR"
say "seed $WAIT_FOR complete"
fi

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
  if complete "$SEED" >/dev/null 2>&1 && trunks_present "$SEED"; then
    say "seed $SEED already complete, skipping"; continue
  fi

  TAG_RUN="${TAG_RUN:-}"   # names the queue and log files, so two waves cannot collide
  Q=$TG/queue_seed${SEED}${TAG_RUN}.tsv
  grep -v '^#' $LIST | awk -v s=$SEED 'NF{printf "free\t%s\t*\tfree\t%s\n", $1, s}' > "$Q"
  say "queue: $(wc -l < "$Q") unit(s) -> $Q"

  B=$($PY -I run_logging/cards_busy.py | tail -1)
  if [ "${B:-1}" != "0" ]; then say "STOPPING: cards_busy reports '$B', not 0."; exit 1; fi
  for c in 4 5 7; do
    tmux new-session -d -s w1fold_$c "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$c \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$TG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $Q --out $TG/poses \
      --csv $TG/results.csv --claim-dir $TG/claims --pool $TG/pool.csv --dataset sabdab_train \
      --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $TG/w1fold_seed${SEED}${TAG_RUN}_c$c.log 2>&1"
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
    if complete "$SEED" >/dev/null 2>&1; then
      if trunks_present "$SEED"; then say "seed $SEED complete"; break; fi
      say "STOPPING: seed $SEED is scored but is MISSING trunk npz files -- see the fold log for"
      say "  'trunk file already exists' or a fold that died after the dump."
      exit 1
    fi
    if [ "$S" -eq 0 ]; then
      # THE WORKERS ARE DONE; the gap is now a SCORING gap, so try the sweep before refusing.
      say "seed $SEED: workers exited while incomplete -- attempting recovery before refusing"
      recover_seed "$SEED" || true
      if complete "$SEED" >/dev/null 2>&1 && trunks_present "$SEED"; then
        say "seed $SEED complete after recovery"; break
      fi
      say "STOPPING: seed $SEED is still incomplete after the recovery sweep."
      complete "$SEED" || true
      tail -4 $TG/w1fold_seed${SEED}${TAG_RUN}_c4.log
      exit 1
    fi
  done
  complete "$SEED"
done

say "CHAIN COMPLETE: seeds [$SEEDS] folded and scored over $N structures"
say "REMINDER: phase_k_train assumes a uniform seed range under --rank-cross-seed. The set is now"
say "437 structures at 15 seeds against 416 at 1; nothing should train on this until that is fixed."
