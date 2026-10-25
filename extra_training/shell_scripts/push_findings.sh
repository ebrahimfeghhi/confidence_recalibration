#!/bin/bash
# PUSH AN ARM'S FINDINGS TO GITHUB. Called at every 5th epoch by the plot monitor.
#
#   ./shell_scripts/push_findings.sh <rid> <epoch>
#
# WHAT IT ADDS, AND WHY THE LIST IS EXPLICIT. Never `git add -A` here. The run directory holds
# 61 MiB resume_ep*.pt checkpoints at this parameter count, eval shards, and multi-megabyte logs;
# a broad add would commit them and the history could not be undone cheaply. Only these paths go
# in, matching what the repo already tracks for earlier arms (args.json + the rank_rho cache):
#
#   docs/<rid>_results.md                      regenerated from the run's own records
#   docs/confidence_head_reachability.md       the finding behind this arm's trainable set
#   training/runs/<rid>/args.json              what was configured
#   training/runs/<rid>/config.json            the launcher's note, verbatim
#   training/runs/<rid>/rank_rho_b30_v*.csv   the held-out metric cache the doc is rendered from.
#                                             GLOBBED, not pinned to a version: the cache is
#                                             bumped when what the numbers MEAN changes, and a
#                                             pinned name would keep pushing the stale file while
#                                             the doc was rendered from the new one.
#   run_logging/render_findings.py             the renderer, so the doc is reproducible
#   shell_scripts/*.sh                         the launchers, so the arm is reproducible
#
# The doc is REGENERATED before every push rather than hand-edited, so what lands on GitHub always
# agrees with the records in the same commit.
#
# IT REFUSES rather than pushing something wrong: no cached epoch, a dirty index it did not create,
# or a detached HEAD all abort. A failed push is reported and left for a human; it is not retried
# in a loop and never forced.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: push_findings.sh <rid> <epoch>}"
EP="${2:?usage: push_findings.sh <rid> <epoch>}"
ROOT=/home/ebrahim/confidence_recalibration

BR=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
[ "$BR" = "HEAD" ] && { echo "PUSH REFUSED: detached HEAD"; exit 1; }

# Regenerate the doc FIRST: if this fails there is nothing worth committing.
$PY -I run_logging/render_findings.py --rid "$RID" || {
    echo "PUSH REFUSED: render_findings.py failed for $RID"; exit 1; }

FILES=(
  "extra_training/docs/${RID}_results.md"
  "extra_training/docs/confidence_head_reachability.md"
  "extra_training/run_logging/render_findings.py"
  "extra_training/shell_scripts/push_findings.sh"
  "extra_training/shell_scripts/queue_fullhead_after_24u34d.sh"
  "extra_training/shell_scripts/queue_pairtrans_after_24u34d.sh"
  "extra_training/shell_scripts/queue_fullfold_after_24u342.sh"
  "extra_training/shell_scripts/resume_24u34d.sh"
  "extra_training/shell_scripts/watch_fullfold.sh"
  "extra_training/run_logging/plot_three.py"
  "extra_training/training/runs/${RID}/args.json"
  "extra_training/training/runs/${RID}/config.json"
  extra_training/training/runs/${RID}/rank_rho_b30_v*.csv
)
ADD=()
for f in "${FILES[@]}"; do [ -f "$ROOT/$f" ] && ADD+=("$f"); done
[ "${#ADD[@]}" -gt 0 ] || { echo "PUSH REFUSED: none of the expected files exist"; exit 1; }

git -C "$ROOT" add -- "${ADD[@]}" || { echo "PUSH REFUSED: git add failed"; exit 1; }
if git -C "$ROOT" diff --cached --quiet; then
    echo "nothing changed since the last push; not committing"
    exit 0
fi

# The subject line states the epoch and the two criterion numbers, so `git log --oneline` alone
# shows whether the arm was beating shipped at each checkpoint.
read -r DQ FR SH_DQ SH_FR <<<"$($PY -I - "$RID" <<'INNER'
import sys, csv, glob, math
sys.path.insert(0, '.')
from run_logging import rank_rho as RHO
rid = sys.argv[1]
d = RHO.load('training/runs', rid, 30)
r = dict(zip(RHO.COLS, d[max(d)]))
rows = []
for f in glob.glob(f'training/runs/{rid}/train_eval_shards_ep{max(d)}/rows_b30_*.csv'):
    rows += list(csv.DictReader(open(f)))
best = {}
for x in rows:
    k, v = x['iface'], float(x['stored_iptm'])
    if k not in best or v > best[k][0]:
        best[k] = (v, float(x['dockq']))
sv = [v[1] for v in best.values()]
print(f"{r['dockq_mean']:.4f} {r['frac_mean']:.4f} {sum(sv)/len(sv):.4f} "
      f"{sum(1 for v in sv if v > 0.23)/len(sv):.4f}")
INNER
)"

git -C "$ROOT" commit -q -F - <<MSG
${RID} epoch ${EP}: DockQ ${DQ} vs shipped ${SH_DQ}, frac ${FR} vs ${SH_FR}

Regenerated docs/${RID}_results.md from the run's own records. Shipped baselines in
both the doc and this subject are derived from the same eval shards via their
stored_iptm column, not carried over by hand.

Criterion is selected DockQ and the fraction above 0.23 on the val10 interfaces;
Spearman is reported but is not the criterion.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014t55QPNWnMkjqFi95YVBtG
MSG
rc=$?
[ "$rc" = "0" ] || { echo "PUSH REFUSED: commit failed ($rc)"; exit 1; }

if git -C "$ROOT" push -q origin "$BR"; then
    echo "pushed $RID epoch $EP to origin/$BR: $(git -C "$ROOT" rev-parse --short HEAD)"
else
    echo "PUSH FAILED (the commit is local, on $BR). Not retrying, not forcing."
    exit 1
fi
