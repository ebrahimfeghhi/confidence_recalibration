#!/bin/bash
# Wait for the val10 loss pass to free card 7, then score the SAME checkpoint on the 30 TRAINING
# structures. If selection improved on the data it trained on while val10 stayed flat, the problem
# is generalisation; if it did not move even on train, PAE cross-entropy is not reordering poses.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
P=$(ps -eo pid,cmd | grep -E "python3? -I .*phase_k_train\.py" | grep -v ugrep | awk '{print $1}' | head -1)
if [ -n "$P" ]; then echo "waiting on loss-only pid $P"; while [ -d /proc/$P ]; do sleep 15; done; fi
echo "card free, extracting the ep142 head"
$PY -I - <<'PYX'
import torch
b = torch.load("training/runs/2u8tf4/resume_ep142.pt", map_location="cpu", weights_only=False)
ep, st = int(b["epoch"]), int(b["step"])
out = f"training/runs/2u8tf4/head_ep{ep}_step{st}.pt"
torch.save(dict(b["head"]), out)
print(f"wrote {out} ({len(b['head'])} tensors)")
PYX
CK=$(ls -t training/runs/2u8tf4/head_ep142_step*.pt | head -1)
echo "scoring $CK on the 30 training structures, budget 10"
LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
  --cards 4,5,7 --ckpt "$CK" --budget 10 \
  --work training/overfit30/traineval_ep142 \
  --out training/overfit30/metric_train_ep142.json \
  --extra "--fold training/fold --results training/fold/results.csv --pool training/overfit30/pool.csv --roles training/fold/chain_roles.csv --seed-base 500000 --join-cache training/overfit30/join_cache.json --pair-iptm-only --prefetch"
echo "driver exit $?"
