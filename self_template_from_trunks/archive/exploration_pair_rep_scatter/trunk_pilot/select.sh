#!/bin/bash
export FOLDBENCH_DATA=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot/dataset OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
H=/home/ebrahim/tts_abag/diffusion_commit_clean_v2
PY=/home/ebrahim/Protenix/.venv/bin/python
echo "[$(date +%T)] shadow dataset"; python3 $D/build_shadow.py || exit 1
$PY -c "import sys; sys.path.insert(0,'/home/ebrahim/tts_abag'); import paths; d=paths.DATASETS['foldbench']; print('foldbench free =', d.free); assert str(d.free).startswith('/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot/dataset')" || exit 1
echo "[$(date +%T)] pool + anchor + interfaces"; $PY $H/build_experiment.py --dataset foldbench --out $D/select --stop-after interfaces --jobs 8 || exit 1
echo "[$(date +%T)] pose-to-pose DockQ (CPU)"; CUDA_VISIBLE_DEVICES= $PY $H/pose_to_pose_dockq.py --poses $D/select/poses.csv --maps $D/select/chain_frame/chain_maps_anchor.csv --roles $D/select/interfaces.csv --out $D/select/pose_dockq_matrices --device cpu --dataset foldbench || exit 1
echo "[$(date +%T)] diversity filter"; $PY $H/structural_diversity_filter.py --matrices $D/select/pose_dockq_matrices --out $D/select/queue.tsv --cap 50 --seed 430000 --dataset foldbench || exit 1
echo "[$(date +%T)] DONE"
