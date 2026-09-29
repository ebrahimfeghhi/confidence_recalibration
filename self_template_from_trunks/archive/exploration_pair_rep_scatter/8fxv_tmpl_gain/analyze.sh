#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/8fxv_tmpl_gain
NS=/home/ebrahim/tts_abag/experimental_data/protenix/foldbench
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2/analysis
for a in g2 g1; do
  echo "[$(date +%T)] fold_scatter $a"
  OMP_NUM_THREADS=1 /home/ebrahim/tts_abag/envs/pxmeter/bin/python fold_scatter.py --dataset foldbench --exp $D/$a --poses $D/$a --pdbs 8fxv --method ost --jobs 8 --out $D/fold_scatter_$a.csv 2>&1 | grep -E "templated folds|rows|skipped"
done
echo "[$(date +%T)] tables"
python3 $D/tables.py
echo "[$(date +%T)] DONE"
