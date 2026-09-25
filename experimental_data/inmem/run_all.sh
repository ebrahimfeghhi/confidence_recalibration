#!/usr/bin/env bash
# 11 -> 12 -> 13 recycles, in series on ONE card.
#
# WHY IN SERIES, not three cards: the cap is 3 GPUs and two are on the 100-seed sweep. Sharing one
# card between arms would also make the fold timings incomparable, and timing is half the point --
# the claim being tested is that a continuation costs ~4 s where a refold costs ~35 s.
#
# WHY SEPARATE OUT/CSV PER ARM: the arm tag is the same ("im") in all three, so one csv would make
# 11-, 12- and 13-recycle rows indistinguishable, and the completion check would skip arms 12 and 13
# as already done.
set -u
D=/home/ebrahim/tts_abag/experimental_data/inmem
PY=/home/ebrahim/Protenix/.venv/bin/python
G=${G:-7}
cd /home/ebrahim/tts_abag/diffusion_commit || exit 1
for N in 11 12 13; do
  if [ -f "$D/n$N.done" ]; then echo "n$N already done"; continue; fi
  echo "=== $(date +%H:%M:%S) ncycle=$N"
  CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold $PY fold.py \
    --queue "$D/queue.tsv" --out "$D/n$N" --csv "$D/n$N.csv" --claim-dir "$D/.claims$N" \
    --dtype bf16 --ncycle "$N" --hook self --hook-after 9 --drop-cifs >> "$D/n$N.log" 2>&1
  rc=$?
  # A missing "queue complete" means the worker died mid-arm; do NOT mark it done, or the next
  # relaunch silently skips an incomplete arm.
  if grep -q "queue complete" "$D/n$N.log"; then touch "$D/n$N.done"; else
    echo "!! ncycle=$N did not complete (rc=$rc) -- leaving it unmarked"; exit 1; fi
done
echo "=== all arms complete $(date +%H:%M:%S)"
