#!/bin/bash
export FOLDBENCH_DATA=/home/ebrahim/tts_abag/pair_rep_scatter/batch2/dataset OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
B=/home/ebrahim/tts_abag/pair_rep_scatter/batch2; H=/home/ebrahim/tts_abag/diffusion_commit_clean_v2; PY=/home/ebrahim/Protenix/.venv/bin/python
echo "[$(date +%T)] shadow dataset"; python3 $B/build_shadow.py || exit 1
$PY -c "import sys; sys.path.insert(0,'/home/ebrahim/tts_abag'); import paths; d=paths.DATASETS['foldbench']; assert str(d.free).startswith('/home/ebrahim/tts_abag/pair_rep_scatter/batch2'), d.free" || exit 1
echo "[$(date +%T)] pool + anchor + interfaces"; $PY $H/build_experiment.py --dataset foldbench --out $B/select --stop-after interfaces --jobs 8 || exit 1
echo "[$(date +%T)] pose-to-pose DockQ (CPU)"; CUDA_VISIBLE_DEVICES= $PY $H/pose_to_pose_dockq.py --poses $B/select/poses.csv --maps $B/select/chain_frame/chain_maps_anchor.csv --roles $B/select/interfaces.csv --out $B/select/pose_dockq_matrices --device cpu --dataset foldbench || exit 1
echo "[$(date +%T)] diversity filter"; $PY $H/structural_diversity_filter.py --matrices $B/select/pose_dockq_matrices --out $B/select/queue.tsv --cap 50 --seed 430000 --dataset foldbench || exit 1
echo "[$(date +%T)] build_templates"; $PY $H/build_templates.py --queue $B/select/queue.tsv --ns $B/select/templates --maps $B/select/chain_frame/chain_maps_anchor.csv --roles $B/select/interfaces.csv --jobs 8 --dataset foldbench || exit 1
echo "[$(date +%T)] preflight"; $PY $H/preflight.py --ns $B/select/templates --queue $B/select/queue.tsv --maps $B/select/chain_frame/chain_maps_anchor.csv --roles $B/select/interfaces.csv || exit 1
for p in 7yxu 7zoz 8be3 8ux6; do grep -P "^tmpl\t$p\t" $B/select/queue.tsv > $B/select/queue_$p.tsv; echo "donors $p: $(wc -l < $B/select/queue_$p.tsv)"; done
echo "[$(date +%T)] STAGE B DONE"
