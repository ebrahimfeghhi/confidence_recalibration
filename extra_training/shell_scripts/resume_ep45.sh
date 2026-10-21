#!/bin/bash
# Resume run 9mfb4y for epochs 4-5 on cards 4/5/7, with --minimal-items.
#
# WHY THESE CARDS. Cards 0/1/3 were the run's own, but the SIGHUP that killed it at 09:56 PT also
# killed the three GPU holders, the cards sat unguarded for three hours, and `john` took 0-3. 4/5/7
# were idle and are now held (pids from shell_scripts/gpu_hold.sh, stop file HOLD_STOP_457). The
# resume guard compares `world` (3), not card indices, so moving cards is not drift.
#
# WHY --minimal-items. The dataset item is never built, so the per-structure item cache -- measured
# at ~518 MB per structure, which was the whole 85-95 GB per rank -- does not exist. It needs the
# alignment preflight marker, because skipping the item also skips the per-unit check_alignment.
#
# WHY --allow-code-drift. The labels were built by an older checkout and HEAD has moved again since
# the run started; the data provenance is unchanged, only the builder's source hash differs.
set -euo pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
R=training/runs/9mfb4y

# THE GATE: refuse to launch unless the preflight actually passed over the whole population.
/home/ebrahim/Protenix/.venv/bin/python -I - "$R/alignment_preflight.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("ok") or not r.get("complete"):
    raise SystemExit(f"STOP: preflight not usable: ok={r.get('ok')} complete={r.get('complete')} "
                     f"{r.get('n_structures')}/{r.get('n_manifest_structures')} "
                     f"mismatches={len(r.get('mismatches', []))}")
print(f"preflight OK: {r['n_structures']}/{r['n_manifest_structures']} structures, "
      f"digest {r['manifest_digest']}, 0 mismatches")
PYEOF

export LAYERNORM_TYPE=openfold
export CUDA_VISIBLE_DEVICES=4,5,7
exec /home/ebrahim/Protenix/.venv/bin/torchrun --nproc_per_node=3 --master_port=29571 \
  phases/phase_k_train.py \
  --resume "$R/resume_ep3.pt" \
  --alignment-preflight "$R/alignment_preflight.json" \
  --minimal-items \
  --out "$R" \
  --epochs 5.0 --lr 0.001 --warmup-steps 20 --seed 0 --accum 256 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 24 --eval-every-epoch --eval-budget 100 \
  --shuffle structure --seeds 500000-500009 \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log \
  --allow-code-drift
