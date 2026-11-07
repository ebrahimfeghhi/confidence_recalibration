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

## What this costs the experiment

**It is a capacity change on top of a freeze, so it is a different arm and needs its own held-out
control.** It is not comparable against `mqird6` or against a full-head arm. The launcher writes that
into the run's `NOTE`, the same rule `MEMLOG` and `FREEZE_BLOCKS` follow — a run recorded under the
default note would be a false record of what was trained.

Trainable under `INBLOCK=1`: block 3's `tri_att_start` (330,240), `tri_att_end` (330,240),
`pair_transition` (786,944), plus the injections — `2 × (39 + 1) × 256 = 20,480` without the readout
injection and `4 × ... = 40,960` with it. So roughly **1.47M parameters**, against the head's
19,752,270 and the 9,200,718 that are live at all (`docs/dead_compute.md`).

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
  combination has not been run either.
