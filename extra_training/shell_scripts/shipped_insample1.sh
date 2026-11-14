#!/bin/bash
# THE SHIPPED HEAD'S OWN IN-SAMPLE PASS, BOTH POOLINGS.  Usage: ./shell_scripts/shipped_insample1.sh [cards]
#
# WHY THIS EXISTS. results.csv stores ONE ipTM column per pose -- the max-pooled shipped value --
# so phase_k_eval's baseline_* fields can only ever report max-pooled shipped numbers
# (phase_k_eval.py:625-647 builds them from stored_iptm and ignores --select-by). The MEAN-pooled
# shipped in-sample number therefore does not exist anywhere in the repo and is not derivable from
# stored scores; it needs the shipped head actually forwarded with --pool-mean. That is this pass.
#
# NO --ckpt IS THE SHIPPED HEAD, and no pose flags: the shipped head is stock Protenix, so the
# --pose-perblock/--frame-direction architecture of our arms must NOT be passed. Copied from
# shipped_rho_valfull83.sh, which is the held-out equivalent.
#
# BUDGET 1 BECAUSE THAT IS WHAT THE PRUNED TRUNK STORE COVERS. Measured 2026-11-11: seed
# free_500000 survives for all 368 pool structures, free_500001..500014 for only 274 -- the
# 2026-11-04 prune took 846 trunks (94 shallow structures x 9 seeds). Budget 1 is the only depth
# that covers every structure at EQUAL depth. Matches run_logging/insample_eval.py --pool-name
# insample1 exactly, minus --ckpt and the pose flags.
#
# BUILT-IN CONTROL: this pass writes its own baseline_* from stored_iptm, which must reproduce the
# 0.4587 / 0.6030 that metric_insample1_ep11.json already recorded. If it does not, the pass is
# not scoring the units it claims and the mean-pooled number it reports cannot be trusted either.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
R=/home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS="${1:-4,5,7}"
OUT=training/runs/insample_eval/metric_insample1_shipped.json
WORK=training/runs/insample_eval/insample1_shards_shipped

[ -f "$OUT" ] && { echo "REFUSING: $OUT exists -- delete it to recompute"; exit 1; }
BUSY="$($PY -I run_logging/cards_busy.py 2>/dev/null)"
[ "${BUSY:-9}" = "0" ] || { echo "REFUSING: cards_busy reports $BUSY -- not launching onto busy cards"; exit 1; }
mkdir -p "$(dirname "$OUT")"

LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
  --cards "$CARDS" --budget 1 --out "$R/$OUT" --work "$R/$WORK" \
  --extra "--fold $R/training/fold --results $R/training/fold/results.csv \
--pool $R/training/full_train/pool_winnable.csv --roles $R/training/fold/chain_roles.csv \
--seed-base 500000 --labels $R/training/fold/labels \
--join-cache $R/training/full_train/join_cache_winnable.json \
--pair-iptm-only --prefetch --pool-mean --select-by mean"
rc=$?
echo "sharded eval exit $rc"
[ "$rc" -eq 0 ] || exit 4
$PY -I -c "
import json
d=json.load(open('$OUT'))
print()
print('  shipped head, in-sample, budget 1, %d interfaces' % d['n_interfaces'])
print('    mean-pooled ipTM : %.4f  / %.4f' % (d['mean_dockq'], d['frac_acc']))
print('    max-pooled  ipTM : %.4f  / %.4f' % (d['mean_dockq_select_max'], d['frac_acc_select_max']))
print('    CONTROL baseline_* (stored max-pooled, must be 0.4587 / 0.6030): %.4f / %.4f'
      % (d['baseline_mean_dockq'], d['baseline_frac_acc']))
"
