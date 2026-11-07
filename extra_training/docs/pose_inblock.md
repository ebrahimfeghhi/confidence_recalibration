# In-block pose injection: putting the pose where a frozen stack stays frozen

**Nothing here has been run on a GPU.** The cards are occupied by the live run.

## The problem this solves

Freezing pairformer blocks 0-2 (`docs/training_speedups.md`, item 3) is only a speedup if autograd
**builds no graph** through them. Autograd starts building at the **first op whose parameters or
inputs require grad** — so *where the pose enters* decides how much backward there is.

`--pose-perblock`, the existing mechanism, injects at each block's **input**. With block 3 otherwise
frozen that makes the `z` entering `tri_mul_out` require grad, **both triangle multiplications rejoin
the backward**, and the saving evaporates with **no error and no warning** — a slower epoch is the
only symptom, and nobody reads epoch times against a prediction.

`PairformerBlock.forward`'s order is `tri_mul_out` → `tri_mul_in` → `tri_att_start` → `tri_att_end`
→ `pair_transition`. **After `tri_mul_in` and before `tri_att_start`** is the earliest point at which
both frozen tri-muls are already behind us. That is where `--pose-inblock` injects, and that is the
whole design.

## The two injections

**1 — in-block, after `tri_mul_in`:**

```
z = z + pose_inblock_d(one_hot(d)) + pose_inblock_d_wo_onehot(d)
```

**Both** terms, not either. That is what the native pre-stack injection is
(`confidence.py:297-305`): `linear_no_bias_d` over a 39-bin one-hot of the pairwise distance **and**
`linear_no_bias_d_wo_onehot` over the raw distance. Same shapes — `(c_z, num_bins)` and `(c_z, 1)`.

**2 — at the readout, after `pae_ln`:**

```
pae = linear_no_bias_pae(pae_ln(z) + pose_inblock_pae_d(one_hot(d)) + pose_inblock_pae_d_wo_onehot(d))
```

**After** the layer norm, which is the owner's choice made knowing the alternative. **The consequence
is the thing to review:** it adds an **unnormalised** quantity to a **normalised** one, and nothing
downstream rescales it — `linear_no_bias_pae` is the next and last op. So the injection's effective
influence is set by its initialisation and by whatever the optimiser does to it, *not* by the layer
norm. Zero-init means it starts with none; after that its scale relative to the normalised
activations it is added to is unconstrained. Injecting **before** the norm would have put it inside
the normalisation and bounded it.

It is implemented as a **forward hook on `pae_ln`**, not a wrapper module: wrapping would rename
`pae_ln.weight` to `pae_ln.ln.weight` and break `phase_k_eval`'s set-equality check on head tensors.
A hook that returns a modified output adds no keys of its own, and the projections are separate
children of the head exactly as `pose_perblock_proj` is.

## Zero-init, verified

Both injections are zero-initialised and **probed**, not trusted: a random input is pushed through
each projection and the result must be **exactly** zero, following `pose_direction`. So **step 0 is
bit-identical to the un-injected head** and any later divergence is attributable to training rather
than to installation.

The graph test checks this two ways — that the mirrored block reproduces the unmirrored block
bit-identically at zero-init (`max|diff| = 0.000e+00`), **and** that a non-zero injection *does* move
`z`. Without the second, the first would pass for a mirror that never applies the projections at all.

## The freeze

Expressed through the existing `--train-only`, so no new freezing mechanism is invented. Trainable:

```
pairformer_stack\.blocks\.3\.tri_att_start\.
pairformer_stack\.blocks\.3\.tri_att_end\.
pairformer_stack\.blocks\.3\.pair_transition\.
^pose_inblock_
```

**Not** `blocks.3.tri_mul_out` and **not** `blocks.3.tri_mul_in`. `INBLOCK_READOUT=1` appends
`^linear_no_bias_pae\.,^pae_ln\.`, which costs **no backward depth at all** — they sit above the
stack, so their gradient needs only their input and their output gradient.

**`INBLOCK_READOUT` now defaults to 1, changed from 0 after review**
(`shell_scripts/launch_default_arm.sh:336-348`, pattern appended at `:359-361`). The reason is the
asymmetry described two sections up: the second injection is added **after** `pae_ln`, so it hands
an **unnormalised** quantity to `linear_no_bias_pae`, and with the readout frozen **nothing
downstream can adapt its scale**. The injection's entire influence would then be fixed by its
zero-init and by whatever the optimiser did to the injection itself, with the layer that consumes it
held at its pretrained scale. Letting that layer learn to consume it costs **exactly 16,896
parameters** — `linear_no_bias_pae`'s 64x256 weight (16,384) plus `pae_ln`'s two 256-vectors (512) —
and no backward depth, which is a small price for making the second injection's scale learnable
rather than accidental.

`INBLOCK_READOUT=0` is still there and is a **legitimate ablation**: it is the arm whose capacity is
*exactly* the injections, which is the cleaner attribution claim. It is simply not the default
reading of the two-injection design, because a frozen readout is a constraint on the injection
rather than a neutral choice.

`INBLOCK=1` sets `--train-only` itself and **refuses to run alongside `FREEZE_BLOCKS=1`**, before
appending anything: both switches set the same flag, argparse lets the second win silently, and the
run would then freeze whatever the loser's pattern said while the banner described the winner's.

## It is a mirror, so it is pinned

Injecting *between* two ops inside `PairformerBlock.forward` means reimplementing that function for
one block. A mirror can go stale: if Protenix reorders the block, this copy would keep the old order
while the rest of the stack moved — and the injection would land somewhere the freeze was never
reasoned about, **raising nothing**. The sha256 of `PairformerBlock.forward` is pinned and a mismatch
is a hard stop, the same contract `lean_readouts.py` uses. Only the `inplace_safe=False` branch is
mirrored, because that is the one the trainer takes; `inplace_safe=True` is refused.

Protenix is **not** forked. Everything is in-repo runtime patching, and the install refuses to run
twice.

Two further hazards taken from `pose_perblock`'s history rather than rediscovered:

- **The geometry is captured into a local in `_prep_blocks`**, not read from a module-level dict
  inside the block. With `--blocks-per-ckpt 1` a block's forward is **re-run during backward**, by
  which time the wrapper's `finally` has cleared the dict — which is the `_NoPose`-on-backward bug
  review reproduced in `pose_perblock`.
- **Each delta is added and its reference dropped** before the next op, rather than kept live across
  the block. Holding an `[N_token, N_token, c_z]` tensor across a block's forward *and* backward is
  the mistake review caught in pathway C, where it was 5.5 GiB at the 3,400-token crop cap.

## Scoring it: the pathway had no evaluator at all

Until review, `--pose-inblock` existed only in the trainer. **`phase_k_eval.py` carried 29
references to the `--pose-perblock` and `--frame-direction` pathways and ZERO to this one**, so there
was no flag with which to install the injection before the checkpoint was loaded — and the
evaluator's guard on head tensors is **exact set equality**, not a subset test
(`phase_k_eval.py:1123-1160`). `_head_ckpt_tensors` saves **every** head tensor, so
`pose_inblock_d.weight`, `pose_inblock_d_wo_onehot.weight` and, under `--pose-inblock-pae`, their two
readout counterparts are **in** the checkpoint, and a stock head does not have them. **Every
in-block checkpoint would have been refused as carrying tensors the model does not have.** An
in-block arm could have trained to completion with **no held-out DockQ at all**, the symptom arriving
as dead eval shards after the first epoch — which is how `2g65a6` died on the per-block pathway, for
a different reason, at the same place. Found before any in-block arm was launched, so this one cost
no GPU time.

`phase_k_eval.py:805-837` now takes `--pose-inblock`, `--pose-inblock-block` (default **-1**,
meaning the **last** block, resolved against the live stack length exactly as the trainer resolves
it) and `--pose-inblock-pae`, and installs at `phase_k_eval.py:1104-1122` — **before** the
checkpoint load, beside `pose_direction` and `pose_perblock`, below the line that forces
`blocks_per_ckpt = None` for the whole evaluator.

**The block index has to be an argument, because the guard cannot see it.** `pose_inblock_d*` carry
no depth in their names, so an evaluator that installed at a *different* block than the trainer
trained at would pass set equality and then score **a different operator** — the same silent-wrong
class as `--pose-perblock-share`, which adds no tensors at all and which the guard's own error
message now says it cannot catch.

**Both of the trainer's refusals are repeated verbatim**, so a combination that is a hard stop when
training is a hard stop when scoring: `--pose-inblock` with `--pose-perblock` (both replace
`_prep_blocks`, so one injection is silently dropped depending on install order), and
`--pose-inblock-pae` without `--pose-inblock` (the readout injection reads geometry the in-block
wrapper stashes, so on its own it never fires and the flag is a no-op). An evaluator more permissive
than its trainer is how a number gets reported for an architecture that was never trained.

### What the trainer forwards to each child, and the one flag it deliberately does not

`eval_checkpoint` builds **one** `extra` list that feeds both the sharded driver's `--extra` and the
single-card argv, so the single site at `phase_k_train.py:6677-6687` covers both paths: it appends
`--pose-inblock`, `--pose-inblock-block <n>` and, when the parent trained with it,
`--pose-inblock-pae`.

**`--lean-single` is deliberately NOT forwarded to the evaluator.** It sets `c_s = 0` on blocks that
are **already built**, leaves every module in the tree, and so changes neither the checkpoint's key
set nor the PAE path's mathematics; the evaluator has no such flag and needs none, and the only
difference is that it runs a single track nothing reads. `docs/dead_compute.md` has the measurement.

`val_loss_checkpoint`, the `--loss-only` held-out-PAE child, is the other case, and a stricter one,
because that child is **`phase_k_train.py` itself** resuming the epoch's bundle.
`phase_k_train.py:6204-6229` now forwards `--pose-inblock`, `--pose-inblock-block`,
`--pose-inblock-pae`, `--lean-readouts` whenever the parent has it, and `--lean-single` — **the
eighth and ninth instances of the "--loss-only inheritance class"**, the running list in that file
of flags a child must inherit or the monitor measures a model the run does not have.

The resume guard compares `lean_single`, `pose_inblock`, `pose_inblock_pae` **and**
`pose_inblock_block` (`phase_k_train.py:3367-3381`), so a child launched without them **refuses the
bundle**. That refusal prints — the shard's last six lines come out as a `WARNING` and the epoch
returns `None` — but it is silent in the **data**: `val_loss.csv` just has no row, for every epoch,
for the whole arm, and the monitor's absence looks like a monitor that was never switched on.
`--lean-readouts` is the one no guard would have caught at all; `docs/dead_compute.md` records why.

## The test, and what it had to be changed to

`run_logging/test_pose_inblock_graph.py`, **seven sections, all passing on CPU.** The one that
matters:

| section | what it asserts |
|---|---|
| 1 | `z` entering block 3 does **not** require grad; `z` leaving it does. Every parameter in blocks 0-2 has `.grad is None`. `blocks.3.tri_mul_out` and `tri_mul_in` have `.grad is None` **and their backward never ran** (`full_backward_hook` did not fire). `tri_att_start`, `tri_att_end`, `pair_transition` and **both** injections have grads, some non-zero. |
| 2 | **The freeze is the other half.** With the tri-muls *trainable* they get gradient at **both** placements, so the placement only matters once they are frozen — and the freeze only pays once the placement is right. Both, or neither. |
| 3 | **The negative control.** Frozen tri-muls, injection moved to the block **input**: both tri-muls' backward **runs**, and the control's backward strictly covers more ops (5 vs 3). Blocks 0-2 still get nothing, so the failure is confined to block 3. |
| 4 | Zero-init ⇒ bit-identical to the unmirrored block; a non-zero injection *does* move `z`. |
| 5 | `inplace_safe=True` and an unknown `inject_at` are refused. |
| 6 | The pin matches the installed Protenix. |
| 7 | The `--train-only` pattern matches block 3's three trained ops and both injections, and matches **neither** tri-mul **nor** blocks 0-2. |

**The control had to be corrected, and that correction is the most useful thing in this document.**
The first version asserted that moving the injection to the block input makes the tri-muls **receive
gradient** — and it **failed**, correctly. Their parameters are *frozen*, so they never get a `.grad`
wherever the pose enters. **The cost of the wrong placement is not a parameter gradient; it is that
autograd builds the graph through both tri-muls and runs their backward to propagate dL/dz.**
`register_full_backward_hook` fires exactly when a module's input gradients are computed, so it sees
that and a `.grad` check cannot. Had the control been left as written and "fixed" by relaxing it, the
file would have been asserting the wrong thing about the whole change.

### Why there is a second test file

`run_logging/test_pose_inblock_integration.py` exists because the graph test **never calls
`install()`**. It reaches into `pose_inblock.block_forward` directly, on a stand-in head and a block
it constructed itself (`test_pose_inblock_graph.py:128`, `:218-240`), which is the right shape for
proving the autograd boundary of **one standalone block** — and it therefore exercises none of the
machinery that puts that block inside a real trainer: the `memory_efficient_forward` wrapper, the
`_prep_blocks` replacement that carries the geometry down to the block, the forward hook on `pae_ln`,
and `--blocks-per-ckpt 1`, under which **a block's forward is re-run during backward**. The graph
test would pass unchanged on an installation that lost the injection on that recompute — which is
precisely the `_NoPose`-on-backward failure `pose_perblock` already had once. The second file is for
the installed path; this one is for the boundary.

### Two traps for anyone writing a CPU test against this head

Both verified on 2026-11-07, and both can make a test pass while proving nothing:

- **A freshly constructed `ConfidenceHead` zero-initialises `linear_no_bias_pae.weight`**
  (`protenix/model/modules/confidence.py:127-129`, alongside `linear_no_bias_pde`, `plddt_weight`
  and `resolved_weight`). Its PAE logits are therefore **identically zero regardless of what the
  pairformer stack does**, so any assertion of the form "X changes the PAE logits" or "X leaves the
  PAE logits unchanged" is **vacuously true** on such a head. Give the readout weight a non-zero
  value before asserting anything about PAE, or the test is measuring the zero map rather than the
  change.
- **`ConfidenceHead` construction is NOT deterministic under `torch.manual_seed(0)`.** Two
  consecutive constructions under the same seed differ in **84 tensors**. Any test comparing two
  separately constructed heads has to make them equal with `load_state_dict`; a bit-identity claim
  between two seeded heads is reporting the RNG, not the property under test.

## What this costs the experiment

**It is a capacity change on top of a freeze, so it is a different arm and needs its own held-out
control.** It is not comparable against `mqird6` or against a full-head arm. The launcher writes that
into the run's `NOTE`, the same rule `MEMLOG` and `FREEZE_BLOCKS` follow — a run recorded under the
default note would be a false record of what was trained.

Trainable under `INBLOCK=1`: block 3's `tri_att_start` (330,240), `tri_att_end` (330,240) and
`pair_transition` (786,944) — **1,447,424** between them — plus the injections and, with
`INBLOCK_READOUT=1`, the readout.

The injection arithmetic is easy to get wrong, so spelled out: each site holds TWO tensors, a
`39 × 256` one-hot term and a `1 × 256` raw-distance term, which is `10,240` per site. With
`INBLOCK_PAE=1` there are two sites, so **20,480** covers all four tensors — not two of them.
`INBLOCK_READOUT` is a SEPARATE switch and adds no injection at all: it decides whether the
pre-existing `linear_no_bias_pae` (`64 × 256`) and `pae_ln` (two 256-vectors) stay trainable,
**16,896** together.

So the default two-injection arm is `1,447,424 + 20,480 + 16,896 =` **1,484,800** trainable
parameters, and the `INBLOCK_READOUT=0` ablation is `1,447,424 + 20,480 =` **1,467,904**. The gap
is exactly the readout's 16,896, or 1.15%, which is why the flip is a default change and not a new
arm. `run_logging/test_pose_inblock_integration.py` asserts the 1,467,904 figure and the 25-tensor
trainable set directly, so a change to either fails a test rather than quietly changing the arm's
advertised capacity. Against the head's 19,752,270 and the 9,200,718 live under `--lean-single`
(`docs/dead_compute.md`), both numbers round to the same **~1.5M**.

## What could not be verified statically

- **Everything on a GPU.** No timing, no memory, no throughput, no loss curve. The expected 1.5-1.7x
  is the `docs/training_speedups.md` freeze estimate, which is analytical and whose method ran **25%
  optimistic** on 2026-11-07.
- **That the two `cdist` calls agree.** The wrapper recomputes the pairwise distance so the injection
  has geometry, while the inner forward computes its own for the native pre-stack injection. Both are
  `torch.cdist` on the same fp32 representative coordinates outside autocast, so they should be
  identical — but that is reasoned, not measured on a real unit.
- **Interaction with `--pose-perblock` and `--frame-direction`.** Both wrap the same
  `memory_efficient_forward` and `--pose-perblock` also replaces `_prep_blocks`. The install order in
  `phase_k_train` puts this module between `lean_readouts` and those two, and `lean_readouts`'
  refusal list now carries this module's marker. **`--pose-inblock` with `--pose-perblock` is now
  REFUSED outright**, before anything is installed: both replace `_prep_blocks`, so one injection
  would be silently dropped depending on install order. `--frame-direction` is *not* refused — it
  wraps `memory_efficient_forward` and installs after this module, so the chain holds — but that
  combination has not been run either, **and the evaluator's chain is the reverse of the
  trainer's.** `phase_k_train.py:1805-1830` installs this module **before** `pose_direction`, so
  `pose_direction`'s wrapper is the outer one; `phase_k_eval.py:1091-1122` installs `pose_direction`
  first and this module **last**, so the nesting is inverted. Nothing enforces the relative order:
  `lean_readouts`' refusal list only pins `lean_readouts` as the innermost patch, and the evaluator
  never installs it at all. Both wrappers set their own per-pose state and then delegate, and
  `pose_direction` does not touch `_prep_blocks`, so the inversion is *believed* numerically
  irrelevant — it is not checked, and the combination has not been run at either nesting. A known
  divergence, not a verified equivalence.
