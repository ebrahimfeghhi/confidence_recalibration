# Protenix's four training stages

Code-anchored explainer for how Protenix is trained, with the detail concentrated on **FT3**, the
stage we re-run. Sections 1 and 2 cover initial training, fine-tune 1 and fine-tune 2 -- what they
share per step, what differs, and why that determines what FT3's poses look like. Sections 3 onward
are FT3 itself. Every claim cites a file and line in
`~/Protenix` as of 2026-10-12. Read this before changing anything in `phases/`.

**Notation.** `N_token` tokens (one per residue, or per atom for ligands), `N_atom` atoms,
`N_sample` poses drawn per fold, `c_s = 384` single-channel width, `c_z = 128` pair-channel width,
`c_s_inputs = 449`. `s_inputs` is the input-embedder single representation, `s_trunk` and `z_trunk`
are the pairformer's single and pair representations after recycling. "Trunk" means input embedder +
template embedder + MSA module + pairformer stack. A "pose" is one sampled set of coordinates.

---

## 1. The four stages, and which one FT3 is

`docs/training_inference_instructions.md:241` gives Protenix's own table, following AF3:

| | initial training | fine-tune 1 | fine-tune 2 | **fine-tune 3** |
|---|---|---|---|---|
| `train_crop_size` | 384 | 640 | 768 | 768 |
| `diffusion_batch_size` | 48 | 32 | 32 | 32 |
| `loss.weight.alpha_diffusion` | 4.0 | 4.0 | 4.0 | **0** |
| `loss.weight.alpha_distogram` | 0.03 | 0.03 | 0.03 | **0** |
| `loss.weight.smooth_lddt` | 1.0 | 0 | 0 | 0 |
| `loss.weight.alpha_bond` | 0 | 1.0 | 1.0 | 0 |
| `loss.weight.alpha_confidence` | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `loss.weight.alpha_pae` | 0 | 0 | 0 | **1.0** |
| `train_confidence_only` | False | False | False | **True** |
| throughput (A100-80G) | ~12 s/step | ~30 | ~44 | **~13** |

Two facts worth stating plainly because they are easy to get backwards:

- **The confidence head trains in every stage, not just FT3.** `alpha_confidence = 1e-4` throughout,
  and `loss.py:1464-1466` weights `plddt_loss`, `pde_loss` and `resolved_loss` by
  `alpha_confidence * alpha_except_pae`.
- **The PAE head trains only in FT3.** `loss.py:1467` weights `pae_loss` by
  `alpha_confidence * alpha_pae`, and `alpha_pae` is 0 until FT3. This matters for us: ipTM is
  derived from the PAE head (§5), so the signal we are recalibrating is the one with the least
  training history.

## 2. The three stages before it

FT3 is a finetune *of* something, and what the first three stages do determines what its poses look
like. All three share one training loop (`protenix.py:1847` `main_train_loop`) and differ only in
crop size, batch size and which loss weights are nonzero.

### What every stage does per step

**A random number of recycles.** `protenix.py:1842`, train mode only:

    nc_rng = np.random.RandomState(current_step)
    N_cycle = nc_rng.randint(1, self.N_cycle + 1)

So a step sees anywhere from 1 to `N_cycle` trunk passes, drawn uniformly and seeded by the step
number. We infer at a fixed `N_cycle = 10`, which is the *top* of that range -- the model is trained
across depths and we use it at the deepest.

**A crop, by one of three methods.** `train_crop_size` with `method_weights = [0.2, 0.4, 0.4]` over
`["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"]`
(`data_pipeline.py:251,265-267`). So 80% of crops are spatial and 40% are centred on an interface.
The default base config is 256 tokens (`configs_base.py:58`); the stages raise it to 384, 640, 768.

**Diffusion training on noised ground truth, not on a rollout.**
`generator.py:289` `sample_diffusion_training`: the ground truth is randomly rotated and translated
into `N_sample` copies (`diffusion_batch_size`, 48 then 32), each gets independent noise at a level
drawn from a log-normal (`p_mean = -1.2`, `p_std = 1.5`, `sigma_data = 16.0`), and the denoiser is
asked to recover it in **one** step. This is a denoising objective, not a trajectory: the model never
unrolls 200 steps during training.

**The confidence head, on a separate 20-step mini-rollout.** Described in section 4; it runs in every
stage, which is why `alpha_confidence = 1e-4` throughout.

### What changes between them

| | initial | FT1 | FT2 |
|---|---|---|---|
| `train_crop_size` | 384 | 640 | 768 |
| `diffusion_batch_size` | 48 | 32 | 32 |
| `smooth_lddt` | **1.0** | 0 | 0 |
| `alpha_bond` | 0 | **1.0** | **1.0** |
| `alpha_diffusion` | 4.0 | 4.0 | 4.0 |
| `alpha_distogram` | 0.03 | 0.03 | 0.03 |
| s/step (A100-80G) | ~12 | ~30 | ~44 |

The crop is the cost: 384 to 768 tokens is 4x the pair tensor, and the throughput drops by 3.7x
across the three.

### The losses, and what each is for

| loss | where | what it penalises |
|---|---|---|
| **diffusion (MSE)** | `loss.py:1063` | weighted squared error after a weighted rigid alignment to the augmented ground truth. Not uniform: `weight_mse = 1/3` with 5.0 for DNA, 5.0 for RNA and 10.0 for ligand atoms, so a ligand atom counts 30x a protein atom |
| **smooth lDDT** | `loss.py:63` | a differentiable lDDT over atom pairs within 15 A (30 A for nucleic). Active ONLY in initial training |
| **bond** | `loss.py:299` | AF3's Formula 5, bonded-atom distance error. Switched on for FT1 and FT2, off again in FT3 |
| **distogram** | `loss.py:515` | cross-entropy over 64 bins from 2.3125 to 21.6875 A, read off the trunk's pair representation. The only loss that trains the trunk directly rather than through the diffusion module |
| **confidence** (pLDDT, PDE, resolved) | section 6 | weighted `1e-4` in every stage |

So the shape of the schedule is: learn geometry at a small crop with a soft structural prior
(smooth lDDT), then grow the crop and swap that prior for explicit bond geometry, then freeze
everything and fit confidence.

### Why this matters for our FT3

- **Our poses come from a 200-step inference rollout at `N_cycle = 10`, `N_sample = 5`.** Nothing in
  training ever produced such a pose: the diffusion objective is single-step denoising, and the
  confidence head's training poses come from a 20-step mini-rollout with one sample. Our pose
  distribution is therefore *out of distribution* for the head we are re-fitting, which is the
  argument for re-fitting it.
- **Training crops to 768 tokens; we fold whole complexes.** 37 of our 40 validation structures are
  under 768 tokens, so only 3 exceed what FT2's crop ever covered -- but a confidence head fitted on
  crops has never seen the pair geometry of a complete assembly, cropped or not.
- **The trunk we freeze was trained with 1-to-10 recycles and we always use 10.** The cached
  `z_trunk` is a depth-10 representation, which is at the edge of what training sampled.

## 3. What `train_confidence_only = True` actually changes

Three distinct things, in three different files.

**The trunk is put in eval mode, every forward.** `protenix.py:283-287` calls `.eval()` on
`input_embedder`, `template_embedder`, `msa_module` and `pairformer_stack` before the trunk runs, and
`:922-926` calls `.train()` on them again afterwards. So dropout and any other train-mode behaviour
is off inside the trunk while the rest of the model stays in train mode.

**The diffusion and distogram losses are asserted to zero.** `protenix.py:191-193`:

    if self.train_confidence_only:  # the final finetune stage
        assert configs.loss.weight.alpha_diffusion == 0.0
        assert configs.loss.weight.alpha_distogram == 0.0

So the run fails loudly rather than silently training the generator.

**The forward returns before the diffusion training path.** `protenix.py:1750-1752` returns
`pred_dict, label_dict, log_dict` immediately after the confidence head, skipping
`sample_diffusion_training` entirely.

Net effect: nothing that produces a pose receives a gradient. A given seed produces bit-identical
poses before and after FT3 — which is why the cached trunk tensors in `val_fold/trunks/` are a valid
substitute for re-folding, and why an FT3 result is a controlled A/B on fixed poses.

## 4. Where the poses the head is trained on come from: the mini-rollout

The head cannot be trained on ground-truth coordinates — at inference it only ever sees sampled ones.
So each training step generates poses on the fly, called the mini-rollout
(`protenix.py:1680-1710`):

- `N_sample_mini_rollout = 1` and `N_step_mini_rollout = 20` (`configs_base.py:183-184`), so **one**
  pose per step, denoised in 20 steps rather than the 200 used at inference.
- `self.diffusion_module.eval()` before it and `.train()` after (`:1682`, `:1707`) — the sampler runs
  in eval mode.
- Every input is detached: `s_inputs.detach()`, `s.detach()`, `z.detach()`, and the output is
  detached in place with `coordinate_mini.detach_()` (`:1709`). No gradient flows back from the pose.
- The ground truth is then **permuted to match the pose**, not the reverse:
  `symmetric_permutation.permute_label_to_match_mini_rollout(...)` (`:1712-1720`). With homomeric
  chains there is no canonical correspondence, so the label is relabelled to whichever assignment the
  pose implies.

This is the single most important structural fact about FT3: the head learns to score *the
generator's own output distribution*. Our pose distribution differs from Protenix's — different
MSAs, templates, `N_cycle = 10`, `N_sample = 5` at 200 diffusion steps — which is the entire premise
for re-running this stage on our poses.

## 5. The confidence head itself

`protenix/model/modules/confidence.py`, config at `configs_base.py:341-354`: `n_blocks = 4`,
`pairformer_dropout = 0.0`, distance bins from 3.25 Å to 52.0 Å in 1.25 Å steps,
`stop_gradient = True`.

**Inputs, and the second stop-gradient.** `forward()` at `:134` takes `s_inputs`, `s_trunk`,
`z_trunk`, `x_pred_coords` and the feature dict. With `stop_gradient = True`, `:179-182` detaches
`s_inputs`, `s_trunk` and `z_trunk` again inside the head. So the trunk is protected twice: eval mode
in §2, detached tensors here.

**How the pose enters.** `:279-307`, per pose:

1. Keep one representative atom per token via `distogram_rep_atom_mask`, giving `[N_token, 3]`.
2. `torch.cdist` in float32 (autocast explicitly disabled) for the `N_token x N_token` distance map.
3. Embed it twice and add both to the pair representation — once one-hot over the distance bins
   (`linear_no_bias_d`) and once as the raw scalar (`linear_no_bias_d_wo_onehot`).

Before that, `:197-203` builds `z_init` from `s_inputs` as an outer sum of two projections and adds
`z_trunk` to it. `s_trunk` is layer-normed after `torch.clamp(s_trunk, min=-512, max=512)` (`:184`).

**The pose-conditioned pairformer.** `:310-318` runs the 4-block pairformer stack on
`(s_trunk, z_pair)`. This is the only place the pose affects anything — every head output below is
read off its result.

**Poses are processed one at a time.** The loop at `:211-240` calls `memory_efficient_forward` per
pose and stacks afterwards; `N_sample` PAE maps at once would be 11.6 GB in bf16 (`:245` comment).

**The four outputs** (`:330-348`), all computed with autocast disabled after upcasting to float32:

| output | shape | from |
|---|---|---|
| `pae` | `[N_sample, N_token, N_token, 64]` | `linear_no_bias_pae(pae_ln(z_pair))` |
| `pde` | `[N_sample, N_token, N_token, bins]` | `linear_no_bias_pde(pde_ln(z_pair + z_pair.transpose(-2,-3)))`, symmetrised |
| `plddt` | `[N_sample, N_atom, bins]` | `s_single` broadcast token→atom, per-tokatom weights |
| `resolved` | `[N_sample, N_atom, 2]` | same broadcast, per-tokatom weights |

Note `pde` is explicitly symmetrised and `pae` is not — PAE is directional, the error in j's position
when the structure is aligned on i's frame.

## 6. How ipTM comes out of the PAE head

`protenix/model/sample_confidence.py`. `calculate_ptm` at `:423`:

1. Softmax the PAE logits into `pae_prob` over 64 bins spanning 0–32 Å.
2. `ptm_norm = 1.24 * (max(N,19) - 15)**(1/3) - 1.8` (`:333-335`), the TM-score normalisation for N
   tokens.
3. Per-bin weight `1 / (1 + (bin_center / ptm_norm)**2)` — the TM-score kernel evaluated at each
   bin's centre.
4. `token_token_ptm = (pae_prob * per_bin_weight).sum(-1)`, then
   `ptm = token_token_ptm.mean(dim=-1)[..., has_frame].max(dim=-1).values`: average over j, then take
   the **best** anchor token i that has a frame.

`calculate_iptm` restricts the token pairs to those spanning different chains; `calculate_chain_based_ptm`
at `:474` produces `chain_ptm`, `chain_iptm`, `chain_pair_iptm` and `chain_pair_iptm_global` by
masking to each chain pair's tokens. **We select on `chain_pair_iptm` for the interface in question,
never the global ipTM** — a complex with one good interface and one bad one has a global ipTM that
describes neither.

The overall ranking score is `0.8 * iptm + 0.2 * ptm` (`:173-174`).

So the chain of dependence is: pose → distance map → pose-conditioned pairformer → PAE logits →
softmax → TM kernel → per-chain-pair ipTM. Recalibrating ipTM means changing the PAE head, which
means FT3.

## 7. The losses

`loss.py:1442` `ProtenixLoss` assembles the weights; `:1464-1467` is the table in §1 made concrete.

**PAE** (`loss.py:813`, `PAELoss`): 64 bins over 0–32 Å. The label is computed in
`calculate_label` (`:842`) from the pose, the true coordinates, and `frame_atom_index` — three atoms
per token defining that token's frame — gated by `has_frame`. The loss is
`softmax_cross_entropy(logits[..., has_frame, :, :], labels=true_bins)`, i.e. only tokens with a
valid frame serve as anchors.

**pLDDT** (`:1283`), **PDE** (`:638`), **experimentally-resolved** (`:1010`): all cross-entropy
against labels computed from the mini-rollout pose versus the permuted ground truth.

**The diffusion and distogram losses are not merely zero-weighted, they are not built.**
`loss.py:1648-1650` branches on `self.configs.train_confidence_only and mode == "train"` and sets
`loss_fns = {}`, skipping the whole smooth-lDDT / bond / MSE block that the `else` arm assembles. So
FT3 saves the cost of constructing and evaluating them, not just their contribution — which is most
of why it is ~13 s/step against FT2's ~44.

## 8. What this means for our re-run

| fact | consequence for us |
|---|---|
| trunk frozen and detached twice (§2) | a seed's poses are bit-identical before and after; the cached `z_trunk`/`s_trunk` in `val_fold/trunks/` can replay the head offline |
| head sees one pose at a time (§4) | per-pose rescoring is the natural inference-time operation |
| trained on mini-rollout poses, 1 per step, 20 steps (§3) | Protenix's head was fitted to a *different* pose distribution than our 200-step, `N_sample = 5`, `N_cycle = 10` folds |
| PAE trains only in FT3 (§1) | ipTM has the least training history of the four heads |
| ipTM is a TM-kernel reduction of PAE (§5) | we cannot recalibrate ipTM directly; we change PAE and ipTM follows |
| labels need ground truth and frames (§6) | every training structure needs a reference cif with resolvable frames, which is what phase G stages |

## 9. Pointers

| | |
|---|---|
| stage table | `~/Protenix/docs/training_inference_instructions.md:241` |
| what the flag changes | `~/Protenix/protenix/model/protenix.py:191`, `:283`, `:922`, `:1750` |
| mini-rollout | `~/Protenix/protenix/model/protenix.py:1680-1720` |
| the head | `~/Protenix/protenix/model/modules/confidence.py`, config `configs/configs_base.py:341` |
| ipTM | `~/Protenix/protenix/model/sample_confidence.py:423`, `:474`, `:333` |
| losses | `~/Protenix/protenix/model/loss.py:813`, `:1442`, `:1464-1467` |
| our pose cache | `phases/phase_i_fold_validation.sh`, the npz validator in its smoke block |
| our data | [`training_data_overview.md`](training_data_overview.md), [`cluster_composition.md`](cluster_composition.md) |
