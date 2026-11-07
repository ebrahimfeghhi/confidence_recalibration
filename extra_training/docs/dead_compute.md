# The dead compute under `--pae-only --lean-readouts`: what was still running, and what it was worth

`--pae-only` stubs three loss terms. `--lean-readouts` stops *computing* the three readouts those
terms would have read. **Neither touched the compute that FEEDS those readouts**, and this document
is the audit of what was left.

Every parameter count here is **measured** by instantiating the live configuration on CPU
(`n_blocks 4, c_z 256, c_s 384, c_s_inputs 449, b_pae 64, b_pde 64, b_plddt 50, b_resolved 2,
max_atoms_per_token 24, pairformer_dropout 0.0, hidden_scale_up True`), which reproduces the head's
**19,752,270** parameters and a block's **4,749,184** exactly. `hidden_scale_up=True` is what makes a
block 4,749,184 rather than 4,091,264, and is how that configuration was identified.

**Nothing in this document is tested on a GPU.** The cards are occupied by the live run.

## The headline, and it is smaller than it sounds

| what | share | what it actually costs |
|---|---|---|
| the pairformer's **single track**, dead | **10,053,120 params = 50.90% of the head** | **2.5-4.0% of a block's saved activations; time below CPU noise** |
| `plddt_weight` | 460,800 (2.33%) | optimiser state + collective bytes only |
| `resolved_weight` | 18,432 (0.09%) | same |
| `linear_no_bias_pde` | 16,384 (0.08%) | same |
| `input_strunk_ln`, `plddt_ln`, `resolved_ln`, `pde_ln` | 2,816 (0.01%) | same |
| **total dead** | **10,551,552 of 19,752,270 = 53.42%** | |
| **live** | **9,200,718** | |

**More than half the "trainable" parameters receive no gradient.** That is the finding. It is *not*
half the compute, and conflating the two is the trap this document exists to close.

## Why the single track is dead

`PairformerBlock.forward` updates `z` through `tri_mul_out`, `tri_mul_in`, `tri_att_start`,
`tri_att_end` and `pair_transition` — **every one a function of `z` alone**. Only then does it update
`s` through `attention_pair_bias` and `single_transition`, **both of which read `z`**. So the
dependency is strictly one-directional: **`z` never depends on `s`.**

Under `--pae-only` the readout is `linear_no_bias_pae(pae_ln(z_pair))`, which reads `z` only, and `s`
feeds nothing but the pLDDT and resolved heads that `--lean-readouts` already stubs.
`lean_readouts.py` says so itself — *"s_single is now unused by anything that is kept"* — and then
keeps computing it in all four blocks, for every pose of every unit of every epoch.

**The project already had the evidence.** The `--grad-probe` path prints *"of 258 tensors marked
trainable, PAE reaches 100, the rel term reaches 100, and NEITHER reaches 158"* and adds
*"EFFECTIVE trainable set is … not …"*. It diagnosed this and nothing acted on it.

## What it is worth — measured, and the measurement corrected itself

**Time was the wrong instrument.** A first pass timed one block's forward+backward on CPU with the
track on and off at N = 96, 144, 208, 288 and got **+0.6%, −2.6%, +9.2%, −4.7%** — pure noise. The
reason is structural: the pair track is **O(N³)** (triangle multiplication and attention) and the
single track's dominant term is **O(N²)** (one `c_z → n_heads` projection over the whole pair map),
so the difference is under the measurement floor and *falls* with N. Projecting that fit gave a
negative saving, which is the fit telling you it has no signal.

**Saved-activation bytes are exact**, measured with `torch.autograd.graph.saved_tensors_hooks`
summing unique saved tensors for one block, fp32, checkpointing off:

| N | full | `c_s=0` | saved | share |
|---|---|---|---|---|
| 128 | 1,234 MB | 1,184 MB | 49.9 MB | **4.0%** |
| 256 | 5,407 MB | 5,248 MB | 159.2 MB | **2.9%** |
| 384 | 13,342 MB | 13,004 MB | 338.0 MB | **2.5%** |

Fitting `saved ≈ 2021·N² + 108347·N` bytes projects 0.75 GB per block fp32 at the median 582 tokens
(3.0 GB over four blocks, ~1.5 GB in bf16) and 5.7 GB per block at the 1,650-token maximum
(22.7 GB over four, ~11 GB in bf16).

**But the arm runs with `blocks_per_ckpt=1`, so those activations are not retained** — they are
recomputed in backward. The saving is therefore on the **recompute**, not on the peak, which is why
the honest estimate for the epoch is **~1-2%, roughly 2-4 min of 232**, and not the 2.5-4.0% of a
block that the table shows.

### The parameter side, which is exact and small

The dead parameters are **already not updated**: autograd produces no gradient for them, and
`_apply_step` then materialises an explicit zeros tensor for the collective (*"a rank with no
gradient for a parameter still has to contribute a tensor, or the collective's shapes do not line
up"*). With `weight_decay=0.0` and a permanently zero gradient, AdamW's update is exactly zero. So
removing them is **bit-identical** and buys:

- **~84 MB of VRAM** — AdamW's `exp_avg` + `exp_avg_sq` for 10.55M fp32 parameters.
- **~42 MB of the 79 MB flat collective buffer, every step.** gloo stages that buffer through host
  memory, so it is a 53% cut in bytes copied out and back, 230 times an epoch. Measured
  `grad_collective_plus_wait` is only **1.1 min/epoch**, so cap the expectation at ~0.5 min.
- **An honest number in the log.** Every `--pae-only` arm in this project has printed
  *"258 trainable tensor(s), 19.8M of 464.5M parameters (4.3%)"* while 158 of those tensors received
  no gradient. The effective capacity is **9.2M, not 19.8M** — which matters for any claim about
  overfitting 19.8M parameters to 841 structures.

## How it is turned off: one attribute, no fork, no checkpoint change

`PairformerBlock` gates the single track on `if self.c_s > 0:` in **both** `__init__` (so the modules
are not built) **and** `forward` (so they are not called). Setting `block.c_s = 0` on a **built**
block stops the calls without touching the module tree:

- the parameters stay where the checkpoint expects them,
- nothing is renamed and nothing is dropped from the state dict,
- `_head_ckpt_tensors` still saves **every** head tensor, so **`phase_k_eval`'s set-equality check
  on head tensors still passes**, and
- Protenix is **not forked**.

So the brief's waiver on checkpoint compatibility was not needed: **no key is created or destroyed.**

Both `PairformerBlock.forward` and `PairformerStack.forward` are **pinned to a sha256**, because what
this rests on is a guard (`if self.c_s > 0`) and a property (`z` never depends on `s`) that a
Protenix upgrade could remove silently. A changed sha is a hard stop, exactly as
`memory_efficient_forward` already is.

`--lean-single` requires `--lean-readouts` **and** `--pae-only`, and is refused otherwise: without
them the pLDDT and resolved heads *are* trained, and they would be trained against a representation
that is no longer updated — silently, since the shapes still line up.

It is also a **resume-guard field**. `--lean-single` drops 158 tensors from the optimiser's parameter
group, and AdamW's `state_dict` is keyed by the parameter's **index** in that group — so resuming a
`--lean-single` bundle without the flag would load `exp_avg`/`exp_avg_sq` onto *different tensors*,
with no shape error (many are the same shape) and no way to notice afterwards. It changes no
mathematics in the forward, which is precisely why nothing else would have caught it.

## The trunk load: audited, and the premise was wrong

The brief expected this to be the biggest item — "mean 243 MB per unit, ~1.1 TB per rank per epoch".
It is not. Of the 11 tensors `RC.load_unit` uploads, **9 are live and 2 are dead**, and the two dead
ones are **tiny**:

| key | verdict | read by |
|---|---|---|
| `z_trunk` | **live** | the pair track and the PAE readout — it *is* the representation |
| `ci_s_inputs` | **live** | `confidence.py:198-202`, `z_init = s1(s_inputs) + s2(s_inputs)` added into `z_trunk` |
| `ci_s_trunk_f32` | **live**, but dead under `--lean-single` | the single track; `input_strunk_ln` is applied in `ConfidenceHead.forward`, which the mirror does not replace |
| `ci_coordinate` | **live** | every pose is used |
| `ci_feat_distogram_rep_atom_mask` | **live** | `confidence.py:192-195`, picks the token representative atoms for `cdist` |
| `ci_feat_asym_id` | **live** | `rank_loss.py:952` (ipTM anchors) **and** `_set_pae_mask` → the PAE term's kept pairs |
| `ci_feat_has_frame` | **live** | `rank_loss.py:953`, selects the frame-bearing anchor rows |
| `ci_feat_is_ligand` | **live** | `rank_loss.py:954-955`, excludes ligand anchors |
| `ci_feat_atom_to_token_idx` | **live** | `rank_loss.py:954-955`; the lean mirror's read of it is shape-only |
| `ci_feat_atom_to_tokatom_idx` | **DEAD** | only `confidence.py:326-347` (the `plddt_weight`/`resolved_weight` gathers the mirror drops) and `pose_direction.derive_frames`, which `--frame-direction False` never installs |
| `ci_N_cycle` | **DEAD** | bound as `_nc` and never read again; its only consumers are `phase_j_rescore_cache.confidence` and `pair_iptm_only`, which the trainer never calls |

**Measured bytes, on a real 191.8 MB trunk (`9mnq`):** `z_trunk` is **187,404,928 B — 97.7% of the
file**. `atom_to_tokatom_idx` is **36,928 B** and `ci_N_cycle` is **136 B**: together **0.02%** of the
unit. Dropping them is worth nothing and costs a conditional in a function `phase_k_eval` also calls
— where `atom_to_tokatom_idx` **is** read, because the eval child does not install
`lean_readouts`. **Declined, deliberately.**

`ci_s_trunk_f32` (929,408 B, **0.38%**) becomes dead the moment `--lean-single` is on, but
`ConfidenceHead.forward` still feeds it to `input_strunk_ln` before the mirror is reached, so
skipping the upload would mean substituting a dummy into a Protenix forward on the strength of a
shape nothing validates. **0.38% is not worth that. Declined, deliberately.**

One thing the audit did turn up that is *not* dead compute: `load_unit` uploads **all five poses at
once** (`ci_coordinate`, 276,128 B) and `_one_pose` then slices `poses[i:i+1]`, so `.to(dev)` there is
a no-op. A per-pose upload would hold 1/5 of it resident. Correctness-neutral, and **0.11% of the
unit** — recorded, not done.

**The real trunk-I/O win is not here.** It is in `docs/training_speedups.md`: reading each unit
*once* per step instead of twice, and getting that read behind the GPU.

## Also fixed: the per-epoch head checkpoint was only written when an eval ran

The single `torch.save` of `head_ep<N>_step<S>.pt` sat **inside** the evaluation function, before it
spawns `phase_k_eval`. Under `NOEVAL=1` that function is never called, so **no epoch-boundary head
was ever written**. The live arm `mqird6` has only `head_step250/500/750/1000.pt` from the periodic
`--save-every 250` path, and at 230 steps per epoch those land at arbitrary points **mid-epoch**:
there is no head checkpoint corresponding to any completed epoch of that run.

The head is the artifact; the eval is one consumer of it. It is now written at every epoch boundary
under `is_main and write_traj`, beside the resume bundle, with the **same filename** so every
downstream scorer, plot and `eval_dirs` reader keeps working — the evaluator's own save becomes a
no-op rewrite of an identical file instead of the only copy. ~79 MB per epoch against a 3,072 GiB
quota. An `OSError` warns and continues: the resume bundle already carries the head.

Nothing is lost on the current run, because `resume_ep<N>.pt` is written unconditionally and contains
the head — but it has to be extracted, which is the friction this removes.

## What was tested (CPU only)

`run_logging/test_lean_single.py`, seven sections, all passing:

1. **`z` is BIT-identical** with the single track on and off — `max|diff| = 0.000e+00`. Exact
   equality, not a tolerance: this is op removal, not reordering, so anything else means the premise
   is wrong.
2. **The single track got no gradient even when it RAN** — 0 of its 16 tensors, with `c_s=384`. This
   is the proof that dropping 10.55M parameters from AdamW is a no-op and not a capacity cut.
   Together with the control that the pair track *does* get gradients (41 tensors), so the backward
   was real. (Only 5 of those 41 are non-zero at init, because several Protenix output projections
   are zero-initialised — init, not a dead path.)
3. **The modules are NOT CALLED**, by forward hook, rather than merely unused — the failure mode
   `--blocks-per-ckpt` had for its whole life.
4. **`s` comes back as the same tensor object**, so the mirror's `s_single.to(torch.float32)` is safe.
5. **`PairformerStack` passes `s` through untouched**, so one attribute per block is enough.
6. **Both pins match** the installed Protenix.
7. **The dead-parameter list is exactly 10,551,552 parameters across 75 tensors**, contains no
   pae-path tensor and no pair-track tensor, and leaves the single track in the optimiser when
   `single_dropped=False` — so the helper cannot silently change pre-existing behaviour.

## What could not be verified statically

- **Everything on a GPU.** No timing, no memory, no throughput. The 1-2% epoch estimate is a
  projection from CPU activation bytes and an O(N³)-vs-O(N²) argument.
- **Whether the trunk's `has_frame` / `is_ligand` / `distogram_rep_atom_mask` are numerically
  identical to the label's `lf_` copies.** `ALIGN_FEATS` claims the preflight asserts agreement; if
  that holds, the trunk copies are redundant and could be dropped in favour of the label's. Not
  pursued: they are ~37 KB each.
- **How often `iptm_pooled` returns `None` early**, in which case `has_frame`/`is_ligand` were read
  and produced nothing. The log's "all 13975 groups hold 5 finite DockQ" suggests it is rare, but
  settling it would need a counter in a live run.
