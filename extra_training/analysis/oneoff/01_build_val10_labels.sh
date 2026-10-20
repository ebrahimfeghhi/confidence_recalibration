#!/bin/bash
# Build PAE labels for the val10 structures: 30 seeds each, lowest-numbered (the budget the val metrics now use), matching the
# cadence training uses. CPU only -- it runs beside the live trainer.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
for P in 7qu2 7r58 8og0 8omz 9bx7 9c7x 9cfd 9fvb 9ima 9qrv; do
    echo "== $P =="
    $PY -I phases/phase_j_build_cache.py \
      --fold validation/fold \
      --out-dir validation/fold/labels \
      --indices validation/prep/job_indices.csv \
      --bio-dir validation/prep/job_bioassembly \
      --mmcif-dir validation/prep/job_cifs \
      --only "$P" --limit 30 \
      --report "validation/fold/label_build_$P.csv" 2>&1 | grep -E "built|skipped|STOP|s/unit" | tail -3
done
echo "=== totals ==="
for P in 7qu2 7r58 8og0 8omz 9bx7 9c7x 9cfd 9fvb 9ima 9qrv; do
    printf "%s %s\n" "$P" "$(ls validation/fold/labels/$P 2>/dev/null | wc -l)"
done
du -sh validation/fold/labels
