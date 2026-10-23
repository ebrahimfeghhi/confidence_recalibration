#!/bin/bash
# LABEL CACHE for the 43 validation structures val10 never needed.
#
# The held-out PAE monitor reads validation/fold/labels/<pdb>/free_<seed>.npz for every unit it
# evaluates, and refuses to start if any is absent -- which is how the full-val launch died:
# 1,290 of 1,590 missing, because only val10's 10 structures had ever been built.
#
# NOT THE SAME ARTIFACT AS THE DockQ LABELS. validation/fold/results.csv carries DockQ for all 92
# interfaces and was complete; these are the per-pose FEATURE caches the loss reads. Checking the
# first does not establish the second, and conflating them is what let the launch get that far.
#
# Modelled on analysis/oneoff/01_build_val10_labels.sh, which took ~20 s per structure for 30 units.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LIST="$(cat "${1:?pass a file holding the pdb list}")"
n=0
for P in $LIST; do
    n=$((n+1))
    echo "== [$n] $P =="
    LAYERNORM_TYPE=openfold $PY -I phases/phase_j_build_cache.py \
      --fold validation/fold \
      --out-dir validation/fold/labels \
      --indices validation/prep/job_indices.csv \
      --bio-dir validation/prep/job_bioassembly \
      --mmcif-dir validation/prep/job_cifs \
      --only "$P" --limit 30 \
      --report "validation/fold/label_build_$P.csv" 2>&1 | grep -E "built|skipped|STOP|s/unit" | tail -3
done
echo "=== totals ==="
for P in $LIST; do printf "%s %s\n" "$P" "$(ls validation/fold/labels/$P 2>/dev/null | wc -l)"; done
du -sh validation/fold/labels
