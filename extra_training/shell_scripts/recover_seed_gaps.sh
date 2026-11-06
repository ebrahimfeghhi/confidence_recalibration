#!/bin/bash
# Close the SCORING gap on seeds 500010-500014, using the exact key set as the definition of the gap.
#
# WHY THIS EXISTS. chain_extra_seeds.sh judged a seed complete by counting rows per PDB and asking
# whether the count was zero. 83.1% of the frozen 437 have more than one scoreable interface and
# folder.py appends per INTERFACE, so a structure whose first interface scored and whose second did
# not passed the gate. Measured against training/combined/pool.csv, the pool the trainer reads:
#
#     seed 500010   422 of 437 exact      seed 500012   423 of 437
#     seed 500011   423 of 437            seed 500013   423 of 437
#
# and the chain reported 437 of 437 for all of them. recover_seed() inside the chain compounded it
# by selecting only structures with ZERO rows, so it repaired 1wej/4p3c/9blx and never looked at
# the ~14 partial structures that recur at every seed.
#
# THE GAP IS DEFINED BY THE GATE, NOT RE-DERIVED HERE. seed_coverage.py --targets-out writes the
# sweep target string from the same key-set comparison that decides pass or fail. Two independent
# notions of "what is missing" is the defect itself -- the chain had one and the gate had another.
#
# NO GPU. The poses, the trunks and the OST json all exist for every one of these; what is missing
# is rows in results.csv. Checked on 7sk4 and 9vaw at the failing seeds: dockq_interfaces and
# chain_mapping are identical to a seed that worked, five finite DockQ values are present, and the
# chain_pair_iptm matrix is complete. So this is CPU-only rescoring from poses already on disk.
#
# IT WAITS FOR THE FOLD. phase_l_merge_recovered.py refuses to replace results.csv while any
# w1fold/w2_ session exists, and it is right to: an append landing between its row count and its
# os.replace would be lost. So this blocks rather than failing late.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
TG=/home/ebrahim/confidence_recalibration/extra_training/training/fold
SC="$TG/.consolidate"
SEEDS="${SEEDS:-500010 500011 500012 500013 500014}"
MIN_MEM_GB="${MIN_MEM_GB:-250}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
mkdir -p "$SC"

# --out MUST SIT BESIDE results.csv. os.replace is atomic only within one filesystem and raises
# EXDEV across devices; pointing it at the session scratchpad failed exactly that way AFTER every
# check had passed, which is documented in consolidate_training_scores.sh and cost a whole run.
if [ "$(stat -c %d "$SC")" != "$(stat -c %d "$TG")" ]; then
  say "STOP: $SC is on a different filesystem from $TG/results.csv; the atomic swap would EXDEV."
  exit 1
fi

M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "MemAvailable ${M} GB"
[ "${M:-0}" -lt "$MIN_MEM_GB" ] && { say "REFUSING: under the ${MIN_MEM_GB} GB floor."; exit 1; }

say "waiting for every session that writes $TG/results.csv"
while [ "$(tmux ls 2>/dev/null | grep -cE '^(w1fold|w2_)')" -gt 0 ]; do sleep 120; done
say "no fold session remains"

FAIL=""
for SEED in $SEEDS; do
  say "--- seed $SEED ---"
  # THE GATE NAMES THE GAP. Exit 0 here means nothing to do, and that is the common case later.
  if $PY -I run_logging/seed_coverage.py --seed "$SEED" --limit 0 \
       --targets-out "$SC/targets_$SEED.txt"; then
    say "  seed $SEED already exact, nothing to recover"; continue
  fi
  T=$(cat "$SC/targets_$SEED.txt" 2>/dev/null)
  if [ -z "$T" ]; then
    # INCOMPLETE WITH NO ADDRESSABLE TARGET is not success. Multi-character chain names are not
    # representable in the sweep's pdb:pdb_C1C2 form and are filtered out by the gate, so a seed
    # can be incomplete and have nothing this script can act on. Say so rather than pass.
    say "  seed $SEED is INCOMPLETE but no addressable target: needs a look, not a sweep"
    FAIL="$FAIL seed$SEED(no-target)"; continue
  fi
  N=$(echo "$T" | tr ',' '\n' | wc -l)
  say "  $N interface(s) to rescore by the isolated route"

  # --roles IS chain_roles_all.csv. chain_roles.csv names 501 structures and none of this wave's
  # 356; with it the sweep fails every sample as "roles expect ('?', '?')". The union covers 853/853.
  nice -n 19 ionice -c 3 $PY -I phases/phase_j_rescore_unmapped.py --targets "$T" \
    --fold "$TG" --refs training/refs --roles "$TG/chain_roles_all.csv" \
    --seed0 "$SEED" --seeds 1 --nsample 5 --route isolated \
    --rows-out "$SC/gap_rows_$SEED.csv" || { say "  sweep exit $?"; FAIL="$FAIL seed$SEED(sweep)"; continue; }
  [ -s "$SC/gap_rows_$SEED.csv" ] || { say "  sweep produced no rows"; FAIL="$FAIL seed$SEED(norows)"; continue; }
  R=$(($(wc -l < "$SC/gap_rows_$SEED.csv") - 1))

  # THE MANIFEST NAMES WHICH INTERFACES MAY BE ADDED, which --replace requires. It is derived from
  # the same target string, so it cannot name an interface the gate did not ask for.
  { echo "iface,route"; echo "$T" | tr ',' '\n' | cut -d: -f2 | sort -u | sed 's/$/,iso/'; } \
    > "$SC/gap_manifest_$SEED.csv"
  I=$(($(wc -l < "$SC/gap_manifest_$SEED.csv") - 1))
  nice -n 19 ionice -c 3 $PY -I phases/phase_l_merge_recovered.py --csv "$TG/results.csv" \
    --rows "iso=$SC/gap_rows_$SEED.csv" --manifest "$SC/gap_manifest_$SEED.csv" \
    --expect-rows "$R" --expect-ifaces "$I" --seeds 1 --seed0 "$SEED" --nsample 5 \
    --out "$SC/gap_merged_$SEED.csv" --replace \
    || { say "  merge exit $?"; FAIL="$FAIL seed$SEED(merge)"; continue; }
  say "  merged $R row(s) over $I interface(s)"

  # RE-GATE, because a merge that succeeded is not the same as a seed that is complete: the sweep
  # can legitimately fail to score an interface OST cannot map at all.
  $PY -I run_logging/seed_coverage.py --seed "$SEED" --limit 6 || FAIL="$FAIL seed$SEED(still-short)"
done

say "final coverage over [$SEEDS]"
for SEED in $SEEDS; do $PY -I run_logging/seed_coverage.py --seed "$SEED" --limit 4 || true; done
if [ -n "$FAIL" ]; then
  say "RECOVERY INCOMPLETE --$FAIL"
  say "Backups are beside results.csv (.pre_recovered_merge). Each swap is atomic: the file is"
  say "either the old one or the new one, never half-written."
  exit 1
fi
say "RECOVERY COMPLETE: every seed in [$SEEDS] passes the exact key-set gate"
