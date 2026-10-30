#!/bin/bash
# GATE THE 5-DAY UNCROPPED RUN ON THE MEMORY PROBE. Launches ONLY if all three checks pass.
#
# WHY A SCRIPT RATHER THAN A JUDGEMENT CALL. The thresholds were agreed BEFORE the numbers were
# known, which is the only time thresholds mean anything. Writing them down here stops them being
# renegotiated against a result that lands just short, and makes the decision reviewable after the
# fact -- the launch either happened because these conditions held, or it did not happen.
#
# THE THREE CHECKS, all on the nine-largest-structures probe with expandable_segments:True:
#
#   1. THREE OPTIMISER STEPS. A probe that survives the forward proves nothing about backward or
#      the step; --rank-structs-per-step 3 over 9 structures gives exactly 3, so this is checkable
#      rather than assumed. Non-finite gradients also fail it.
#   2. PEAK RESERVED BELOW THE PREVIOUS PROBE'S 91.94 GB. If expandable_segments buys nothing
#      there is no reason to prefer it, and no reason to believe the margin changed.
#   3. AT LEAST 5 GB GENUINELY FREE at the tightest sampled moment on the tightest card. NOT
#      total minus torch's reserved: torch counts only its own allocator, so the CUDA context,
#      this project's GPU holder (1.16 GiB a card) and any other resident process are invisible
#      to it. gpu_free_sampler.sh records what the DEVICE reports free, every 5 s, and the
#      minimum over the whole run is what has to clear the bar.
#
# MISSING DATA IS A FAILURE, never a pass. A probe that OOMed, a sampler file that never got
# written, a rank whose memory line is absent -- each leaves the cards alone and exits non-zero.
#
# Usage: greenlight_uncropped.sh <probe_rid> <sampler_csv> [sampler_marker]
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: greenlight_uncropped.sh <probe_rid> <sampler_csv> [marker]}"
CSV="${2:?}"
MARK="${3:-}"
LOG="training/runs/$RID/run.log"
PREV_RESERVED_GB=91.94
MIN_FREE_MIB=5120
NEED_STEPS=3

echo "waiting for probe $RID to settle"
for _ in $(seq 1 240); do
  grep -qE "memory: peak allocated|OutOfMemory|CUDA out of memory|STOP:" "$LOG" 2>/dev/null && break
  sleep 15
done
# THE SAMPLER IS STOPPED HERE, not earlier, so its minimum covers the whole probe including the
# final optimiser step -- which is where the peak has been.
[ -n "$MARK" ] && rm -f "$MARK"
sleep 6

$PY -I - "$RID" "$CSV" "$PREV_RESERVED_GB" "$MIN_FREE_MIB" "$NEED_STEPS" <<'PYEOF'
import csv, os, re, sys
rid, csv_p, prev_gb, min_free, need_steps = sys.argv[1:6]
prev_gb, min_free, need_steps = float(prev_gb), int(min_free), int(need_steps)
log = f"training/runs/{rid}/run.log"
txt = open(log, errors="replace").read() if os.path.exists(log) else ""
fail = []

def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fail.append(name)

check("the probe did not OOM", "OutOfMemory" not in txt and "CUDA out of memory" not in txt)
check("the probe did not STOP on a guard", "\nSTOP:" not in txt)

# 1. optimiser steps, from the trajectory the run wrote
tp = f"training/runs/{rid}/trajectory.csv"
rows = list(csv.DictReader(open(tp))) if os.path.exists(tp) else []
taken = [r for r in rows if r.get("step_taken") == "1"]
nonfin = [r for r in rows if r.get("nonfinite_grad") not in ("0", "", None)]
check(f"{need_steps} optimiser step(s) completed", len(taken) == need_steps,
      f"{len(taken)} taken over {len(rows)} pose row(s)")
check("no non-finite gradients", len(nonfin) == 0, f"{len(nonfin)} event(s)")

# 2. peak reserved, every rank
res = [float(m) for m in re.findall(r"peak reserved ([0-9.]+) GB", txt)]
check("every rank reported its peak", len(res) >= 3, f"{len(res)} rank line(s)")
worst = max(res) if res else float("inf")
check(f"peak reserved below the previous probe's {prev_gb} GB", worst < prev_gb,
      f"worst {worst:.2f} GB" + (f" (was {prev_gb})" if res else ""))

# 3. the device's own free memory, at its tightest moment
mins = {}
if os.path.exists(csv_p):
    rr = list(csv.DictReader(open(csv_p)))
    if rr:
        ks = list(rr[0])
        for i, c in ((1, "card4"), (2, "card5"), (3, "card7")):
            v = [int(r[ks[i]]) for r in rr if r[ks[i]].strip().isdigit()]
            if v:
                mins[c] = min(v)
check("the sampler recorded all three cards", len(mins) == 3, str(mins))
tight = min(mins.values()) if mins else -1
check(f"at least {min_free} MiB genuinely free at the tightest moment", tight >= min_free,
      "  ".join(f"{c} {v}" for c, v in sorted(mins.items())) or "no samples")

print()
if fail:
    print(f"  NO LAUNCH -- {len(fail)} check(s) failed: {', '.join(fail)}")
    sys.exit(1)
print("  ALL CHECKS PASS")
sys.exit(0)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "greenlight: refusing to launch; the cards are left free"
  exit "$rc"
fi
BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "greenlight: REFUSING -- cards busy ($BUSY)"; exit 1; }
echo "greenlight: launching the uncropped beta-75 arm at $(TZ=America/New_York date '+%-I:%M %p %Z')"
exec ./shell_scripts/launch_rel_itm_beta75_uncropped.sh
