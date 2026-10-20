# Experiment plan: baseline, C, A, C+A

Four arms, one variable at a time, on a shared protocol. Written 2026-10-19.

The question is whether the confidence head can be retrained to pick better antibody-antigen
poses than the shipped head does, and whether giving it more of the predicted pose helps. Arm 1
establishes what the current recipe achieves. Arms 2 and 3 add one pose pathway each. Arm 4
combines them, and is only worth running if at least one of the two helped.

## The shared protocol

Every arm is identical except where its row says otherwise.

| | |
|---|---|
| training set | the 30 `overfit30` structures, 77 interfaces (`training/overfit30/structures.csv`) |
| loss | PAE only, masked to the scored chain pairs (`--pae-only --pae-scored-pairs`) |
| learning rate | 1e-2, warmup 20 steps |
| batch | `--accum 85` → 84 poses per optimiser step, 18 steps/epoch |
| crop | interface plus 10 A margin, 5 A contact seed, cap 3,400 tokens |
| length | 50 epochs ≈ 900 steps, early stop OFF |
| checkpoints | a resume bundle every epoch, so any epoch can be scored afterwards |
| **eval** | **every 2 epochs on 10 HELD-OUT validation structures**, `training/val10/pool.csv`, 18 interfaces, budget 100 seeds, sharded over three cards |
| start | fresh from the shipped head |

Cost: ~6.5 h per arm — 4.3 h of training at 5.2 min/epoch plus 2.1 h of evals at 5.1 min each.

### Why the eval is held out

Earlier arms were scored on the pool they trained on, so they could only ever demonstrate
memorisation. `training/val10/pool.csv` is the 10 validation structures with the most total
headroom, ranked by oracle minus selected summed over each structure's interfaces.

| | mean DockQ | frac acceptable |
|---|---|---|
| shipped head | 0.2626 | 0.5000 (9 of 18) |
| oracle | 0.6662 | 1.0000 (18 of 18) |
| headroom | 0.4036 | |

Every interface has an acceptable pose available and the shipped head finds 9, so **9 are
winnable** and anything above 0.2626 / 0.5000 is the head beating the baseline on structures it
has never trained on.

**The caveat that travels with every result here.** `9qrv` and `9bx7` are two of the three
interfaces that carried 85-88% of every earlier arm's validation gain. Ranking by headroom
necessarily selects them, so a gain on this pool must be decomposed per interface before it is
believed. 18 interfaces also means one flip moves frac_acc by 0.056.

**The baseline moves with the seed budget.** At 20 seeds it is 0.3041 / 0.5556 with an oracle of
0.5554 / 0.9444 — the shipped head scores *higher* with fewer candidates, because more candidates
give ipTM more chances to pick badly while the oracle only improves. Never quote the 100-seed
baseline against a reduced-budget run.

### Rules agreed for reading the arms

- **Ten-epoch minimum.** No arm is stopped, relaunched, or called stable / unstable / plateaued
  before 10 complete epochs. One epoch's rise is not instability. The exceptions are non-finite
  gradients, skipped optimiser steps, the gradient norm hitting the 25.0 clip, and a crash.
- **Compare at matched epochs**, not at best-ever. Every arm runs 18 steps/epoch, so epoch number
  is a fair axis. "Best seen" has previously just tracked which arm was allowed to run longest.
- **The loss is the smoother signal.** DockQ on 18 interfaces is noisy; pose-weighted epoch PAE
  is not. Pose-weighted means `sum(batch_pae_loss * batch_n) / sum(batch_n)` over steps, never a
  mean of per-step means, and never a partial epoch.
- Gradient norm is a **weak** instability detector under AdamW, whose update is roughly
  `lr x sign(gradient)` per parameter, so the clip only catches extreme outliers.

## Arm 1 — baseline

**Status: RUNNING as `335psh`** (launched 2026-10-19 17:37 PT).

The protocol above with nothing added. It exists because no historical lr 1e-2 arm can serve as a
reference: they had warmup 0 rather than 20, every masked arm was cut short by the early-stop bug
fixed in `6cc69d30`, they were scored on their own training pool, and `dapghi` and `krah8z` are
byte-for-byte the same experiment — which made 1e-2 look like it had two independent supports when
it has one.

Early numbers: val10 ep2 **0.3045 / 0.5556**, ep4 0.2673 / **0.6111** (11 of 18) — both above the
shipped head. PAE 2.3248, 2.2748, 2.2267, 2.2210, 2.2232.

Launcher: `shell_scripts/launch_overfit30_lr1e2.sh`.

### Why 1e-2 and not something higher

Settled by experiment, not assumption. `e4tvxe`+`f8rqck` ran the identical protocol at **lr 1e-1**
and was worse: val10 0.2151/0.4444, 0.2420/0.3889, 0.2753/0.5556, 0.2222/0.3889, 0.2006/0.3333 at
ep2..10, so the one point above baseline did not hold and the next three declined. PAE sat flat
around 2.35-2.42 from ep4, about 0.15 behind 1e-2. It never misbehaved numerically — 185 steps,
max grad norm 3.58 against a 25.0 clip, nothing skipped — so 1e-1 is not unstable, it simply
optimises worse. `ovqdv8` showed **5e-2** stable on the full 488-structure fold over 128 steps
(max grad norm 7.77) but was stopped before it produced a loss trajectory; 5e-2 remains the
untested middle and is the natural next rate to try against this anchor.

## Arm 2 — C, frame-relative direction

**Status: QUEUED** behind arm 1 (`shell_scripts/queue_C_after_anchor.sh`), fires only if the
anchor exits 0 and only after confirming no training or eval process survives.

Adds `--frame-direction`: for each ordered pair (i, j), the **unit direction of j in i's backbone
frame** — three numbers — through one zero-initialised `LinearNoBias(3 -> c_z)` added into
`z_pair` beside the two distance paths that already exist.

**The argument.** PAE is not a distance. It is: superpose on token i's local frame, then ask how
far token j ended up from where it should be. The head's only view of the pose is `cdist` over
`distogram_rep_atom_mask`, which selects **CB** — so none of N, CA or C is in its input, and the
frame's orientation is absent rather than merely implicit. C supplies exactly the quantity the
label is built from. This adds **information**, which is why it goes before A.

Details and the corrections that shaped it are in `docs/pose_pathways.md`, including: the target
atoms are the PAE representative (CA, the token centre), not the distogram CB, which also makes
the self-entry exactly zero; the input is the **unit** vector because zero-init bounds the output
but not the first update, and the magnitude is already supplied by the existing paths; and the
earlier symmetry argument for C was wrong and has been replaced.

## Arm 3 — A, per-block injection

**Status: BUILT, NOT QUEUED.**

Adds `--pose-perblock`: re-inject the same one-hot distance embedding before **every** pairformer
block instead of only before the first. One zero-initialised `LinearNoBias(num_bins -> c_z)` per
injected block — 3 matrices at 39x128 = **14,976 parameters**, 0.08% of the head's 19.8M.

Block 0 is deliberately left alone: its injection already exists as the pretrained
`linear_no_bias_d` applied before the stack.

**What it does and does not test.** A adds **access and capacity, not information** — the one-hot
is the same tensor each time. Its hypothesis is that the pose signal degrades crossing the stack,
and with only **4** blocks that story is thin. Nothing measured so far shows attenuation. That is
the whole reason C runs first.

`--pose-perblock-tied` shares one matrix across the injected blocks. That is the sharper test of
A's actual hypothesis — the identical embedding re-presented at each depth — where untied also
lets each depth learn a different projection of distance, so an untied win is ambiguous between
the two explanations. Untied is the default because it is how every other weight in the stack
works and is strictly more expressive. Tying is **not** equivalent to scaling a single injection:
each block transforms `z` before the next add.

## Arm 4 — C + A

**Status: conditional.** Run only if C or A helped; both flags together, protocol otherwise
unchanged. Both pathways are zero-init, so any subset leaves the head bit-identical at step 0 and
a later ablation is just zeroing one group.

If both arms help individually, this one says whether the gains are independent. If neither does,
it is not worth the 6.5 h.

## What is NOT in the plan, and why

- **Pathway B**, injecting pose into the single track, is **ruled out**. `PairformerBlock.forward`
  updates the pair track from `z` and `pair_mask` only, then does
  `s = s + attention_pair_bias(a=s, s=None, z=z)`. Information flows `z -> s` and never back, so
  the single track is *already* pose-aware and geometry injected into `s_trunk` could never reach
  `pae_pred`, which is computed from `z_pair`.
- **A capacity control.** C and A are both representation and access changes. If the binding
  constraint is capacity — 19.8M parameters, 4 blocks, a 128-dim pair channel — neither helps
  much, and `n_blocks 4 -> 8` is the control that would say so. It cannot resume from any existing
  bundle, since the new blocks have no pretrained weights.
- **A ranking loss** on chain-pair ipTM (`docs/ranking_loss.md`), deprioritised in favour of
  driving PAE lower. If it returns: `calculate_iptm` is differentiable end to end
  (`protenix/model/sample_confidence.py:813`), and pairs must be formed **across seeds**, because
  within one seed's 5 poses the DockQ spread is median 0.053 while across an interface's ~50 poses
  it is median 0.509.

## Code

### The pathways
- `phases/pose_direction.py` — arm 2. The projection, the frame geometry, and `derive_frames`,
  which reconstructs `frame_atom_index` and `pae_rep_atom_mask` from the trunk's own
  `atom_to_token_idx` / `atom_to_tokatom_idx`. That derivation was verified exact against the
  label npz on all 500 training structures and is what lets C be scored on `validation/fold`,
  which has 53 trunks and no labels.
- `phases/pose_perblock.py` — arm 3. The per-block projections, installed by wrapping each block
  inside `PairformerStack._prep_blocks` rather than inserting callables, which would add
  checkpoint boundaries.

### The checks, both CPU-only and both required to pass before an arm runs
- `phases/phase_k_frame_direction_check.py` — five properties of C: rigid-motion invariance, unit
  norm, zero self-entry, asymmetry, exact zero-init. A missing input file is a hard stop, because
  an earlier version printed SKIP and then "all checks passed".
- `phases/phase_k_perblock_check.py` — six properties of A: exact zero-init, bit-identity at step
  0, the right blocks wrapped, a trained weight actually changing the output, tied sharing one
  matrix, and the shared one-hot released after the forward. Bit-identity is measured on a single
  instance before and after install, because two `PairformerStack`s built under the same seed do
  not agree.

### Training and scoring
- `phases/phase_k_train.py` — `--frame-direction`, `--pose-perblock`, `--pose-perblock-tied`,
  `--pae-only`, `--pae-scored-pairs`, and the `--train-eval-*` flags including
  `--train-eval-fold` and `--train-eval-seed-base`, which are what let the every-2-epochs eval
  score a pool living in another fold.
- `phases/phase_k_eval.py` — the scorer. Reads one field, `chain_pair_iptm`, and selects
  `max(v, key=lambda t: (t[0], -t[2], -t[3]))`. Needs the same pathway flags as training, which
  the trainer passes automatically.
- `phases/phase_k_eval_sharded.py` — the multi-card driver.
- `phases/phase_k_derive_preflight_subset.py` — derives an alignment marker for a subset of an
  already-verified manifest, which turns a 2.3 h re-check into seconds.

### Launchers and queues
- `shell_scripts/launch_overfit30_lr1e2.sh` — arm 1.
- `shell_scripts/queue_C_after_anchor.sh` — arm 2, armed behind arm 1.
- `shell_scripts/launch_overfit30_lr1e1_val10.sh` — the lr 1e-1 arm that settled the rate.

### Plots
- `run_logging/plot_val10.py` and `shell_scripts/refresh_val10_plot.sh` — three panels: training
  PAE loss, held-out mean DockQ, held-out frac acceptable, with the shipped head and oracle drawn
  in. Deliberately separate from `run_logging/plot_train_eval.py`, which overlays arms scored on
  their own training pool and must not share axes with held-out numbers.

### Data
- `training/overfit30/` — the 30 training structures, their 77 interfaces, and the alignment
  marker.
- `training/val10/pool.csv` — the 10 held-out validation structures and their 18 interfaces, plus
  the per-shard join caches that cut each eval from 8.9 min to 5.1.
- `training/fold/results.csv` — per-pose DockQ and chain-pair ipTM for training;
  `validation/fold/results.csv` for validation, at seeds 400000-400099.

### Background
- `docs/pose_pathways.md` — the design of C and A, and what review corrected in both.
- `docs/train_val_difficulty.md` — validation is easier than training by 0.157 mean DockQ at the
  shipped head, and the oracle is 0.182 higher, so the difficulty is in the structures. Size
  explains ~15%; the antigen filter and deposition era were checked and explain none.
- `docs/unscored_interfaces.md` — 17 of 28 antibody-antigen interfaces that never got a DockQ were
  recovered; five remain.
- `docs/ranking_loss.md` — the deprioritised loss-side alternative.
