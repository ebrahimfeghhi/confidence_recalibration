# Experiment plan: baseline, C, A, C+A

Four arms, one variable at a time, on a shared protocol. Written 2026-10-19.

The question is whether the confidence head can be retrained to select better antibody-antigen
poses than the shipped head does, and whether giving it more of the predicted pose helps. Arm 1
establishes what the current recipe achieves. Arms 2 and 3 add one pose pathway each. Arm 4
combines them.

## The data

### Train — 30 structures, 77 interfaces

`training/overfit30/structures.csv`, interfaces in `training/overfit30/pool.csv`:

```
35uj 7sfw 8dfh 8ebp 8ez7 8jkh 8jn1 8jn4 8tui 8wnp 9b6o 9cq8 9cyh 9dz4 9e3l
9e60 9hls 9js7 9kod 9md3 9oqd 9p4f 9pix 9q26 9qqe 9qto 9shd 9uax 9ulm 9x61
```

Chosen for **headroom with zero baseline success**: the shipped head's selected pose is under
DockQ 0.23 for every one of the 77 interfaces (max 0.2100), while an acceptable pose exists for
68 of them. So improvement is unmissable and the ceiling is high.

| | mean DockQ | frac acceptable |
|---|---|---|
| shipped head | 0.0420 | 0.0000 (0 of 77) |
| true-PAE reference | 0.4216 | 0.6883 (53) |
| oracle | 0.5098 | 0.8831 (68) |

Seeds **500000-500009**, 5 poses each, so 10 seeds x 5 = 50 poses per interface and 1,500 poses
per epoch. Trunks and labels in `training/fold/`, per-pose DockQ and chain-pair ipTM in
`training/fold/results.csv`.

### Validation — a DIAGNOSTIC subset: 10 held-out structures, 18 interfaces

`training/val10/pool.csv`:

```
7qu2 7r58 8og0 8omz 9bx7 9c7x 9cfd 9fvb 9ima 9qrv

7qu2_AC 7qu2_BC 7r58_AB 7r58_AC 8og0_AC 8og0_BC 8omz_AB 9bx7_AC 9bx7_BC
9c7x_AC 9c7x_BC 9cfd_AC 9cfd_BC 9fvb_BD 9ima_AD 9ima_BD 9qrv_AB 9qrv_AC
```

The 10 validation structures with the most total headroom, ranked by oracle minus selected summed
over each structure's interfaces. Drawn from the 53-structure held-out validation fold
(`validation/fold/`), which is held out **by cluster**, so no relative of these appears in
training. Seeds **400000-400099**, 5 poses each — 100 seeds, 500 poses per interface.

| | mean DockQ | frac acceptable |
|---|---|---|
| shipped head | 0.2626 | 0.5000 (9 of 18) |
| oracle | 0.6662 | 1.0000 (18 of 18) |
| headroom | 0.4036 | |

Every interface has an acceptable pose available and the shipped head finds 9, so **9 are
winnable** and anything above 0.2626 / 0.5000 is the head beating the baseline on structures it
has never trained on.

**This is a diagnostic subset, not an estimate of validation performance.** Selecting structures
for high headroom makes the pool sensitive -- an improvement shows up here that would be lost in
the average over all 53 -- but a gain on it says nothing about the full validation population,
because the population was chosen to exclude the cases where there is nothing to gain. **Whichever
arm wins must be confirmed on all 53 validation structures (92 interfaces) before the result is
stated.** That full pass is ~25 min sharded at budget 100 and is the number that goes in a paper;
val10 is the instrument used to decide which arms deserve it.

Three things that must travel with every number from this pool:

- **`9qrv` and `9bx7` are two of the three interfaces that carried 85-88% of every earlier arm's
  validation gain.** Ranking by headroom necessarily selects them, so a gain must be decomposed
  per interface before it is believed. 18 interfaces also means one flip moves frac_acc by 0.056.
- **The baseline moves with the seed budget.** At 20 seeds it is 0.3041 / 0.5556 against an oracle
  of 0.5554 / 0.9444 — the shipped head scores *higher* with fewer candidates, because more
  candidates give ipTM more chances to pick badly while the oracle only improves. Never quote the
  100-seed baseline against a reduced-budget run.

Note the two sets are not equally hard and the difficulty is in the structures, not the selector:
validation's oracle is 0.182 above the training fold's. `docs/train_val_difficulty.md` has the
measurements.

## Optimisation

Every arm is identical except for the pathway it adds.

| | | why |
|---|---|---|
| loss | PAE only, masked to the scored chain pairs | ipTM derives entirely from PAE; the mask gave 2.76x gradient concentration. `--pae-only --pae-scored-pairs` |
| learning rate | **1e-2** | chosen by experiment — see below |
| warmup | **20 steps** | a fresh optimiser's first step moves every weight by ~lr regardless of gradient size. 20 is the trainer's default and ~2% of the run, matching Protenix's own finetune |
| batch | `--accum 85` → 84 poses/step, 18 steps/epoch | inherited; the micro-batch is one pose, so batch buys gradient averaging, not throughput |
| grad clip | 25.0 | never engaged: max observed norm 3.23 |
| crop | interface + 10 A margin, 5 A contact seed, cap 3,400 tokens | median 354 kept tokens |
| length | 50 epochs ≈ 900 steps, early stop **OFF** | the early-stop rule was ending runs mid-descent; fixed in `6cc69d30` and left off here |
| start | fresh from the shipped head | the pathways add trainable tensors, so no existing bundle can be resumed |
| checkpoints | a resume bundle every epoch | any epoch can be scored afterwards |
| **eval** | **val10, every 2 epochs, budget 100**, sharded over 3 cards | 5.1 min each with the join cache, down from 8.9 |
| final check | the winning arm, **all 53 validation structures**, budget 100 | val10 is diagnostic; the full pass is the reportable number, ~25 min |

Cost: **~6.5 h per arm** — 4.3 h training at 5.2 min/epoch plus 2.1 h of evals.

### Why lr 1e-2

Settled by running it. The identical protocol at **lr 1e-1** (`e4tvxe`+`f8rqck`) was worse: val10
0.2151/0.4444, 0.2420/0.3889, 0.2753/0.5556, 0.2222/0.3889, 0.2006/0.3333 at ep2..10, so its one
point above baseline did not hold and the next three declined; PAE sat flat near 2.35-2.42 from
ep4, about 0.15 behind 1e-2. It never misbehaved numerically — 185 steps, max grad norm 3.58,
nothing skipped — so 1e-1 is not unstable, it optimises worse. **5e-2** is the untested middle:
`ovqdv8` showed it stable on the full 488-structure fold over 128 steps (max grad norm 7.77) but
was stopped before producing a loss trajectory. It is the natural next rate to try against arm 1.

### Rules for reading the arms

- **Ten-epoch minimum.** No arm is stopped, relaunched, or called stable / unstable / plateaued
  before 10 complete epochs. One epoch's rise is not instability. Exceptions: non-finite
  gradients, skipped optimiser steps, the gradient norm hitting the clip, a crash.
- **Epoch 50 is the primary comparison.** Matched-epoch curves are for watching, not deciding:
  each arm gets the same 50 epochs and the comparison is made at the end. An arm that looks ahead
  at ep6 and behind at ep20 has told us nothing yet.
- **The two signals answer different questions.** Training PAE diagnoses OPTIMISATION — is the arm
  fitting at all, is it faster, has it stalled. Held-out selection on val10 decides WHICH ARM
  HELPS. They can disagree, and when they do the held-out metric is the one that matters. Use
  pose-weighted epoch PAE, `sum(batch_pae_loss * batch_n) / sum(batch_n)` over steps — never a
  mean of per-step means, never a partial epoch.
- **One clipping event is not instability.** Gradient clipping working as designed is not a
  failure, and gradient norm is a weak detector anyway under AdamW, whose update is roughly
  `lr x sign(gradient)` per parameter. Stop for: persistent divergence of the loss, non-finite
  gradients, or failed optimiser updates (`step_ok == 0`). Not for a single clip.

## Arm 1 — baseline

**RUNNING as `335psh`** (launched 2026-10-19 17:37 PT). The protocol above, nothing added.

It exists because no historical lr 1e-2 arm can serve as a reference: warmup 0 rather than 20,
every masked arm cut short by the early-stop bug, scored on its own training pool rather than held
out, and `dapghi`/`krah8z` are byte-for-byte the same experiment — which made 1e-2 look like it
had two independent supports when it has one.

Launcher: `shell_scripts/launch_overfit30_lr1e2.sh`.

## Arm 2 — C, frame-relative direction

**QUEUED** behind arm 1 — `shell_scripts/queue_C_after_anchor.sh`, fires only if the anchor exits
0 and only after confirming no training or eval process survives.

`--frame-direction`. For each ordered pair (i, j), the **unit direction of j in i's backbone
frame** — three numbers — through one zero-initialised `LinearNoBias(3 -> c_z)` added into
`z_pair` beside the two distance paths that already exist.

PAE is not a distance: it is "superpose on token i's local frame, then ask how far token j ended
up from where it should be." The head's only view of the pose is `cdist` over
`distogram_rep_atom_mask`, which selects **CB**, so none of N, CA or C is in its input and the
frame's orientation is absent rather than implicit. C supplies the quantity the label is actually
built from, which is why it adds **information** and runs before A.

Design and the corrections review made to it: `docs/pose_pathways.md`.

## Arm 3 — A, per-block injection

**BUILT, NOT QUEUED.**

`--pose-perblock`. Re-inject the same one-hot distance embedding before **every** pairformer block
instead of only before the first: one zero-initialised `LinearNoBias(num_bins -> c_z)` per
injected block, 3 matrices at 39x128 = **14,976 parameters**, 0.08% of the head's 19.8M. Block 0
is left alone because its injection already exists as the pretrained `linear_no_bias_d`.

A adds **access and capacity, not information** — the one-hot is the same tensor each time. Its
hypothesis is that the pose signal degrades crossing the stack, and with only **4** blocks that is
thin; nothing measured shows attenuation. Hence C first.

`--pose-perblock-tied` shares one matrix across the injected blocks, which is the sharper test of
that hypothesis: the identical embedding re-presented at each depth. Untied also lets each depth
learn a different projection of distance, so an untied win is ambiguous between the two
explanations. Untied is the default. Tying is **not** equivalent to scaling a single injection —
each block transforms `z` before the next add.

## Arm 4 — C + A

**Conditional** on at least one of C or A helping. Both flags together, protocol otherwise
unchanged. Both pathways are zero-init, so any subset leaves the head bit-identical at step 0 and
a later ablation is just zeroing one group. If both help individually, this arm says whether the
gains are independent.

## Code

**Pathways**
- `phases/pose_direction.py` — arm 2. Also `derive_frames`, which reconstructs
  `frame_atom_index` and `pae_rep_atom_mask` from the trunk's own `atom_to_token_idx` /
  `atom_to_tokatom_idx`, verified exact against the label npz on all 500 training structures.
  That is what lets C be scored on `validation/fold`, which has trunks and no labels.
- `phases/pose_perblock.py` — arm 3. Installed by wrapping each block inside
  `PairformerStack._prep_blocks` rather than inserting callables, which would add checkpoint
  boundaries.

**Checks, CPU-only, both must pass before an arm runs**
- `phases/phase_k_frame_direction_check.py` — five properties of C: rigid-motion invariance, unit
  norm, zero self-entry, asymmetry, exact zero-init.
- `phases/phase_k_perblock_check.py` — six properties of A: exact zero-init, bit-identity at step
  0, the right blocks wrapped, a trained weight changing the output, tied sharing one matrix, and
  the shared one-hot released after the forward.

**Training and scoring**
- `phases/phase_k_train.py` — the pathway flags, `--pae-only`, `--pae-scored-pairs`, and the
  `--train-eval-*` flags including `--train-eval-fold` and `--train-eval-seed-base`, which let the
  every-2-epochs eval score a pool living in another fold.
- `phases/phase_k_eval.py` — the scorer. Reads one field, `chain_pair_iptm`, and selects
  `max(v, key=lambda t: (t[0], -t[2], -t[3]))`.
- `phases/phase_k_eval_sharded.py` — the multi-card driver.

**Launchers**
- `shell_scripts/launch_overfit30_lr1e2.sh` — arm 1.
- `shell_scripts/queue_C_after_anchor.sh` — arm 2.
- `shell_scripts/launch_overfit30_lr1e1_val10.sh` — the lr 1e-1 arm that settled the rate.

**Plots**
- `run_logging/plot_val10.py`, `shell_scripts/refresh_val10_plot.sh` — three panels: training PAE
  loss, held-out mean DockQ, held-out frac acceptable, with the shipped head and oracle drawn in.
  Deliberately separate from `run_logging/plot_train_eval.py`, which overlays arms scored on their
  own training pool and must not share axes with held-out numbers.

**Data**
- `training/overfit30/` — the 30 training structures, their 77 interfaces, the alignment marker.
- `training/val10/pool.csv` — the 10 held-out structures and 18 interfaces, plus the per-shard
  join caches that cut each eval from 8.9 min to 5.1.
- `training/fold/results.csv`, `validation/fold/results.csv` — per-pose DockQ and chain-pair ipTM.
