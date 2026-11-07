# Two speedups for the confidence-head arms: freeze the stack, then cache it

Every figure below is **measured** on `mqird6` (rel + PAE at alpha-rank 75, uncropped, 841
structures, `--ckpt-token-threshold 855`) on 2026-11-07, rank 0, epoch 3. The two speedup
numbers derived from them are **ANALYTICAL AND UNVALIDATED** — nothing in either section has
been run.

## The baseline we are cutting into

Epoch 3, rank 0, 232.2 min total. Epoch wall clock came in at 3.83 / 3.87 / 3.75 h for epochs
2, 3 and 4; call it **3.87 h**.

| bucket | min | per | share |
|---|---|---|---|
| `gpu_backward` | 110.0 | 11,500 calls @ 0.574 s | **47.4%** |
| `gpu_forward` | 69.2 | 23,000 calls @ 0.181 s | 23,000 = 11,500 pass A + 11,500 pass B |
| `unit_setup` | 37.0 | 4,600 units | |
| `loss_stats_collective_plus_wait` | 13.5 | | |
| `rank_gather` | 8.2 | | |
| `fingerprint_hash` | 5.5 | | |
| `apply_step_total` | 1.8 | | |
| `grad_collective` | 1.1 | | |
| **accounted** | **196.2** | | |
| **residual** | **35.9** | | |

Population: 841 structures, 6,903 units, 230 steps, 5 poses per unit (verified 2300/2300 units in
epoch 1), 34,515 poses per epoch across all ranks. Tokens per unit: min 128, median 582, p90 1004,
max 1650. Peak allocated **81.30 GB**, identical across all 3 ranks and all epochs, against a
~90 GB card budget.

The head trains 19.8M of 464.4M params (4.3%), all under `confidence_head`. **The 4 pairformer
blocks are UNTIED** — `self.blocks = nn.ModuleList()` with a distinct block appended per layer in
`/home/ebrahim/Protenix/protenix/model/modules/pairformer.py` — at 4,749,184 params each, 24.0% of
the head each. Everything outside the stack is 755,534 params (3.8%). `c_z = 256` in this model,
**NOT** the 128 in the `confidence.py` docstring; verified from
`confidence_head.linear_no_bias_pae.weight` of shape (64, 256). `pairformer_dropout = 0.0` — the
run's `config.json` has no dropout key, so it takes the `ConfidenceHead` default of 0.0.

**77% of the epoch is the model's own forward and backward on real poses.** That is the number that
decides where to look: the two ideas below both attack forward/backward, and the ruled-out list is
mostly things that attack the other 23%.

## Idea 1 — freeze blocks 0-2, train only block 3

**ONE FLAG, NO CODE CHANGE:** `--train-only 'pairformer_stack\.blocks\.3\.'`.

Verified that `--train-only` genuinely sets `p.requires_grad_(...)` at
`extra_training/phases/phase_k_train.py:2745`, not merely filtering the optimiser. So with only
`blocks.3.` trainable, block 0's inputs come from the frozen trunk and frozen input projections,
nothing below block 3 requires grad, and **autograd builds no graph through blocks 0-2 at all**.

### Estimate

| | min | why |
|---|---|---|
| `gpu_forward` | 69.2 → **69.2** | UNCHANGED — all 4 blocks still run; you need block 3's output |
| `gpu_backward` | 110.0 → **~28** | graph through 1 block instead of 4 |
| overhead | 53 → **53** | unchanged |
| epoch | 232 → **~150** | **1.55x** |

The backward division by 4 is fair because the 4 blocks are byte-identical in size and everything
outside the stack is 3.8% of the head.

**Second-order win.** Measured backward is 3.2x forward per pose (0.574 vs 0.181 s) where theory
says ~2x. The gap is the checkpoint re-forward on the 696 of 4,600 units above the 855-token
threshold. Freezing cuts retained activations ~4x, which should let checkpointing go off entirely
and recover that gap, pushing toward **1.67x**.

Quote the range as **1.5-1.7x**: epoch 3.87 h → ~2.4 h (plan on 2.5 h), a 10-epoch arm 38.7 h →
~25 h, saving **~14 h per arm**.

### Why ~20 prior arms did not already get this

~20 prior arms already use `--train-only`, but **every one targets INPUT-SIDE parameters**
(`linear_no_bias_d`, `pose_perblock`, `frame_dir`) which sit at the **BOTTOM** of the stack. The
gradient still has to traverse all 4 blocks to reach them, so those arms pay the full backward
while updating almost nothing. This proposal is the first configuration that puts the trainable set
at the **TOP**, which is exactly where the saving comes from.

### Costs and risks

- **This is a CAPACITY CUT, not a free optimisation.** 4.75M trainable params instead of 19.75M. It
  needs its own held-out control arm; it is **not** comparable against `mqird6`.
- It may win on generalisation anyway — 19.8M params against 840 structures.
- It makes per-block injection work untrainable in blocks 0-2 (`--pose-perblock`, `frame_dir` at
  all 4 depths).
- **The estimate is analytical and unvalidated.** An earlier same-method speed projection made this
  same day predicted 1.35x where the measured value was 1.10x — 25% optimistic. Discount
  accordingly.
- **Falsification:** run ~20 steps with the flag and compare s/step against `mqird6`'s measured
  **62.1 s/step** mean. If backward collapses you see ~40 s immediately. Needs a free card.

## Idea 2 — cache blocks 0-2's output between pass A and pass B (DEPENDS ON IDEA 1)

The trainer runs **two passes per step**. Pass A is a `torch.no_grad()` forward per pose that yields
the scores the relative-iTM coefficients need — a pose's coefficient depends on every other pose in
its interface, so all scores must exist before any backward. Pass B then rebuilds each pose and
backwards it. So **every pose is forwarded TWICE within the same step**; that is why `gpu_forward`
shows 23,000 calls against 11,500 backwards.

With blocks 0-2 frozen their output is **identical in both passes**, so stash it in pass A and start
pass B from it. Validity rests on `pairformer_dropout = 0.0`: no dropout means pass A and pass B
compute a bit-identical representation for the same pose. **This is EXACT, not an approximation.**
Pass A already computes the tensor, so the stash is free — you just don't free it.

Saves blocks 0-2's forward in pass B = 3/4 of pass B's 34.6 min ≈ **26 min/epoch**.

### Memory, which is the whole difficulty

The pair rep is n² × 256 in bf16, and a rank holds 5 seeds × 5 poses = 25 poses of each of the 2
structures in a deep cross-seed step.

| tokens | per pose | per structure | per step |
|---|---|---|---|
| median 582 | 165 MiB | 4.0 GiB | 8 GiB |
| p90 1004 | — | ~12 GiB | ~24 GiB |
| max 1650 | 1329 MiB | 32.5 GiB | 65 GiB |

Freezing drops peak memory a long way (only block 3 retained), so expect ~55-60 GB of headroom.
**Median and p90 fit; the largest structures do not.** So it wants a **TOKEN THRESHOLD**, the same
pattern as the existing `--ckpt-token-threshold`: cache below the line, recompute above it.
Coverage by budget: ≤409 tok = 12.3% of units, ≤579 = 49.3%, ≤819 = 82.9%, ≤1158 = 93.9%.

### Stacked estimate

Freeze alone 232 → ~150 min; plus cache at ~85% coverage → **~128 min**, about **1.8x** overall,
10 epochs in ~21 h instead of 38.7.

### Caveats

- **This is the first item needing real code rather than a flag** — it changes the two-pass
  structure.
- It is an estimate stacked on an estimate. Trust it less than Idea 1.
- **It is strictly dependent on Idea 1.** Without frozen blocks 0-2, pass B must run them anyway to
  build the graph, so there is nothing to cache.

## Ruled out — do not re-propose

1. **Caching blocks 0-2 across EPOCHS to disk. DEAD TWICE OVER.** Capacity: 34,515 poses × n² × 256
   × 2 B ≈ **~7.8 TB** in bf16, against 721 GiB free and a 3,072 GiB hard quota — it would not fit
   on an empty disk. Bandwidth: even granting infinite disk, reading ~7.8 TB back each epoch exceeds
   the ~52 min of forward compute it would save.
2. **Skipping unused confidence outputs. ALREADY DONE** and live in `mqird6`: `lean_readouts = True`
   and `pae_only = True`. The log prints "LEAN READOUTS: pde, pLDDT and resolved are NOT COMPUTED"
   and "pae-only: all-atom lddt SKIPPED; measured 3.4 ms and up to 1.66 GB per pose". Worth
   recording why it mattered: at the shipped weights PAE is only ~30% of the loss (pLDDT 42%,
   PDE 27%), so most of the gradient used to go to outputs ipTM never reads.
3. **Restricting the PAE readout to the scored interface submatrix.** Worth only 1-3% — a single
   linear against an O(n³) stack. Not worth the code.
4. **Engineering overhead generally.** 77% of the epoch is the model's own forward and backward on
   real poses, so the ceiling on pure overhead elimination is about **1.3x**. There is no large win
   hiding in the data path.
