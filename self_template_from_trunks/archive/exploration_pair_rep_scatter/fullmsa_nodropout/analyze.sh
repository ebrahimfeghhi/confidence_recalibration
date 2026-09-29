#!/bin/bash
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2/analysis
for p in 9fww 8fxv; do
  echo "[$(date +%T)] fold_scatter $p"
  OMP_NUM_THREADS=1 /home/ebrahim/tts_abag/envs/pxmeter/bin/python fold_scatter.py --dataset foldbench --exp /home/ebrahim/tts_abag/pair_rep_scatter/fullmsa_nodropout/$p --poses /home/ebrahim/tts_abag/pair_rep_scatter/fullmsa_nodropout/$p/out --pdbs $p --method ost --jobs 8 --out /home/ebrahim/tts_abag/pair_rep_scatter/fullmsa_nodropout/fold_scatter_$p.csv 2>&1 | grep -E "templated folds|rows|skipped"
done
echo "[$(date +%T)] tables"; python3 /home/ebrahim/tts_abag/pair_rep_scatter/fullmsa_nodropout/tables.py; echo "[$(date +%T)] DONE"
