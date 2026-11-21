#!/usr/bin/env bash
# STEP 0 of the ladder, and it runs FIRST: how reproducible is the LEGACY path against ITSELF?
#
# Row-wise dropout is live at p_drop 0.25 and `_xs_prepare` saves and restores both the CPU and
# the CUDA RNG state precisely so pass B REDRAWS pass A's masks. So phase_k_train.py is not
# guaranteed bit-identical to itself, and any tolerance chosen before this number is measured is
# either tighter than the floor -- failing on noise and sending someone hunting a bug that is not
# there -- or looser, and certifying nothing.
#
# Runs the LEGACY trainer twice on the same fixture, same seed, same card, and reports the
# largest per-pose difference. That number is the floor for parity_compare.py --tol.
set -uo pipefail
. "$(dirname "$0")/parity_common.sh"
require_idle_card || exit 3

mapfile -d '' -t ARGV < <(fixture_argv)
for RUN in 1 2; do
  OUT=$SD/replay_$RUN
  say "legacy pass $RUN of 2 on 8yvm (15 seeds x 5 poses), card $CARD"
  rm -rf "$OUT"
  CUDA_VISIBLE_DEVICES=$CARD LAYERNORM_TYPE=openfold \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout 2400 $PY "$ROOT/phases/phase_k_train.py" "${ARGV[@]}" \
      --limit-units 15 --loss-only --out "$OUT" > "$SD/replay_$RUN.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] || { say "legacy pass $RUN FAILED (exit $RC)"; tail -5 "$SD/replay_$RUN.log"; exit 1; }
done

say "--- legacy vs legacy ---"
$PY -I - "$SD/replay_1" "$SD/replay_2" <<'PY'
import csv, sys
def rows(d):
    out=[]
    with open(f"{d}/trajectory_rank0.csv") as fh:
        for r in csv.DictReader(fh):
            try: out.append((r["pdb"], r["unit"], int(float(r["pose"])), float(r["pae_loss"])))
            except Exception: continue
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
print(f"  poses: {len(a)} vs {len(b)}")
if len(a) != len(b) or [x[:3] for x in a] != [x[:3] for x in b]:
    print("  ORDER DIFFERS between two runs of the SAME code -- that is the finding, not a"
          "\n  tolerance question. Stop and explain it before comparing anything to the copy.")
    raise SystemExit(1)
d = max(abs(x[3]-y[3]) for x, y in zip(a, b)) if a else 0.0
print(f"  max |d PAE| legacy-vs-legacy: {d:.3e}")
print()
if d == 0.0:
    print("  The legacy path IS bit-identical to itself on this fixture. Use --tol 0 and require")
    print("  the copy to match exactly; anything looser would hide a real difference.")
else:
    print(f"  FLOOR = {d:.3e}. parity_compare.py --tol must be AT LEAST this. A tighter value")
    print("  would fail on the legacy path's own noise.")
PY
