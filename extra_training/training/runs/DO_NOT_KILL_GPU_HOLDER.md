# pid 1263388 is a deliberate GPU holder on card 4 — leave it alone

Started 2026-11-24 8:40 PM EST by `shell_scripts/gpu_hold.sh --gib 10 --stop
training/runs/HOLD_STOP_card4`, log in `training/runs/hold_card4_single.log`.

**Why it exists.** We cut our GPU use from three cards to one so other users on this shared box
get cards 5 and 7 back. The holder keeps card 4 visibly ours in `nvidia-smi`, because a card
reading 0 MiB between our runs has already been taken by a neighbour once (2026-09-10) and killed
our next launch with CUDA OOM.

**So:** `run_logging/cards_busy.py` returning `1` is the intended state. It is not a stray process
and not a leak. Do not `kill` it, and do not treat the single occupancy as something to clean up.

**To release it** (only when the owner asks): `touch training/runs/HOLD_STOP_card4`.
