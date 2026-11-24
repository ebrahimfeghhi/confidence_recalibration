# Joint end-to-end PAE pose reranker — implementation and launch handoff

**Status:** code implemented and pushed as an **opt-in library**, with unit
tests and a sequential 3-GPU capacity runner. **NOT YET wired to
`phase_k_train.py`**; the real-Protenix forward adapter and GPU-probe factory
must be implemented and verified before running. No existing run or checkpoint
was modified, resumed, stopped, or launched by these commits.

## Scientific specification

The target population is a set of K independently sampled models for **one
scored antibody–antigen interface**, not all interfaces of a PDB pooled
together. K varies naturally with crop size and available VRAM.

1. The same trainable Protenix confidence head predicts a [F,T,64] PAE-logit
   tensor for **each pose**, using its own crop. F can differ from T.
2. Define directed antibody–antigen pair mask from **MODEL asym IDs** and
   has_frame; no DockQ, native/true contacts, crystal-derived epitope or OST
   reference mapping enters feature extraction or inference. For multiple
   antigen copies, provide deployment-time model-chain IDs for the *specific*
   interface. Do not reproduce the reference-dependent `abag_rank/chain_map.py`
   join in the new feature path.
3. `softmax(PAE logits)` per pair -> shared `64->64->64` MLP (SiLU).
   Mean, max and learned-attention pooling over **all scored AB/AG pairs**
   -> 192-D; Linear(192,128) -> 128-D pose representation. A 1500-pair
   truncation is **not** the default; if required for memory, make it an
   explicitly measured ablation. Pair chunking retains the complete autograd
   graph and is not a gradient-memory magic trick.
4. One 4-head self-attention block over the K pose representations, with no
   positional embeddings. This is **permutation-equivariant**: shuffling
   the input poses shuffles the scores. The block works with K=2..N.
5. Shared 128->64->1 pose scorer. `s_i = ipTM_i + delta_i`. The final
   correction layer is zero-initialized, so **at initialization** output
   scores equal the input ipTM values. The ipTM is computed with gradient
   from the **same trainable confidence head** and from the same scored
   pair; use the uncropped pair's d0 for continuity with the current code.
6. The original per-pose Protenix PAE CE is **unchanged**, and averages
   across all K poses. Add:
   `L_rank = logsumexp(s/tau) - logsumexp(s[DockQ>=0.23]/tau)`
   and `L_total = mean(L_PAE) + alpha*L_rank`.
   If all K poses are acceptable or all are unacceptable, the rank term is
   zero and the PAE CE still runs. DockQ is a TRAINING LABEL ONLY.

No reference-based chain mapping, label-derived pair masks, true contact
regions or DockQ-dependent feature selection at inference. Sampling groups
with known positives/negatives is a legitimate **training** sampling policy
but must be separate from deployable inputs.

## Initial architecture decision: preserve pose injections, do not feed them separately to reranker

For the **first** joint end-to-end arm, keep the same upstream confidence-head pose-injection
configuration as the cropped `7mo52r` recipe:
`--pose-perblock --pose-perblock-copy --frame-direction`, with its matching trainable-parameter
scope and other architecture settings. These are trainable, never frozen. The new learned
ranking loss backpropagates through the PAE logits into the head **and** the injected pose
layers, so the pose pathway receives the new supervision without any extra reranker input.

The reranker should initially consume only the **current** head's differentiable PAE
probability distributions and live per-interface ipTM as a residual baseline; do NOT
route intermediate pose-injection outputs, geometry tensors, or frame-direction features
directly to the reranker. Adding those would confound the basic hypothesis (a learned
readout from PAE vs a fixed ipTM readout) and introduce another memory/architecture axis.

For a **clean** attribution test, initialize the new arm and its matched control from the
same shipped-head / pose-injection initialization. Warming from a trained `7mo52r`
checkpoint is allowed as a separately labelled speed/transfer experiment, but should
not be presented as a fresh comparison to `7mo52r` because that checkpoint has already
been updated by the old ranking loss.

The 256/384/512 GPU capacity sweep MUST include these same pose-injection pathways.
If memory is tight, reduce the poses per group using the measured safe K before deleting
injections. Once the joint model works, run an explicit no-pose-injection ablation if the
paper needs to distinguish the two mechanisms.

## Files

- `model.py`: differentiable PAE-pair encoder, pose attention, residual
  scorer; streaming `encode_pose`/`score_embeddings` inference.
- `training.py`: acceptable-mass loss, joint loss, **shared head** gradient
  calibration, alpha ratio helper.
- `integration.py`: head-agnostic bridge: per-pose original PAE CE + dynamic
  ipTM + PAE logits + model-chain mask -> group loss with live graphs.
- `gpu_probe_step.py`: CUDA-only worker testing the **actual model** and
  backpropagation through head/reranker and optimizer state.
- `gpu_capacity_sweep.py`: sequential, separate-subprocess tests for all
  crop sizes; reports absolute maximum tested success and safe group size.
  On OOM, tries the *next crop size*; a hard integration error stops the
  sweep. Timeout kills torchrun workers, not merely the parent.
- `test_joint_reranker.py`, `test_integration.py`: CPU-only correctness gates.

## Agent: implement the missing Protenix adapter

In a new opt-in training launcher (do **not** alter a running legacy arm),
write `predict_one(pose) -> PosePrediction` following `integration.py`:

- Forward the **trainable** confidence head with grad enabled, using
  `model.run_confidence_head(...)`, unchanged trunk/pose-injection settings
  from the selected baseline.
- Squeeze its sample/frame axes correctly to live `[F,T,64]` logits.
- Compute **original** differentiable PAE cross-entropy using the existing
  `loss_fn.pae_loss` and its original label/mask rules, without changing
  its denominator or weighting; return the scalar as `pae_ce`.
- Compute **differentiable max-pooled** per-interface ipTM with
  `rank_loss.iptm_pooled(..., pool="max", n_pair_tokens=uncropped_pair_count)`.
  **Adapter prerequisite:** extend `iptm_pooled`'s `pool="max"` branch to
  support an explicit fixed d0 via `pae_row_weight.row_t`, returning
  `t[cnt > 0].max()`; retain the existing vendor `iptm()` path when
  `n_pair_tokens is None`. Never `.item()` or detach the returned scalar.
- Build `antibody_antigen_pair_mask` solely from the model's chain metadata,
  using `has_frame`; never read `ost_json` to choose a chain from a true
  reference at inference.
- Assemble K **unique** training poses from the same interface, carrying
  per-pose DockQ only as supervision. Use the batch's unique pose IDs to
  guard against sample-index/pose-index swaps. Call
  `joint_group_loss`; backprop ONCE after the complete K group has been
  forwarded. Do not update the confidence head while the group is open.
- Both the head and reranker parameters must be in the optimizer. If running
  3-rank DDP, each rank owns **one complete K-pose group per forward**;
  all-reduce gradients only after those local group losses. Do not split a
  group's poses across ranks unless explicitly implementing distributed
  cross-rank attention.
- Use real final crop policy and sampled structure trunks. Some structures
  need 384/512 crops: reduce K based on measured capacity, do not force 256
  and thereby destroy important interface pairs.
- Include frozen/recomputed shipped baseline on **identical** candidate
  sets, correct seed/sample tie-break, 139 leak-free val interfaces,
  fraction acceptable, mean selected DockQ, top-5 retrieval, first acceptable
  rank, and ordinary good/bad-pose PAE CE.

For a standalone integration call after the adapter is written:

```python
from extra_training.joint_reranker import PAEPoseReranker
from extra_training.joint_reranker.integration import joint_group_loss
reranker = PAEPoseReranker().cuda()
# predict_one must be implemented against the actual Protenix training state.
joint = joint_group_loss(reranker, pose_inputs, predict_one, dockq,
                         alpha=alpha, temperature=tau)
joint.total.backward()  # head and reranker both update (optimizer.step later)
```

The `gpu_probe_step.py` requires a concrete `build_probe` factory (see its
module docstring); **no runnable Protenix-specific factory is included yet**.

## Decided ipTM residual: max pooling, FIXED uncropped d0 (agent update)

**The previous suggestion to use mean pooling for v1 is superseded.** Use
max-pooled ipTM plus a learned correction for the initial joint reranker,
with the d0 normalization pinned to the uncropped scored model-chain pair
length for all crops of that interface. The existing `rank_loss.iptm_pooled`
max path refuses `n_pair_tokens` because it dispatches to the Protenix
vendor function which chooses d0 from cropped asym_ids. Do not remove the
guard and continue calling the vendor function: that would silently ignore
our fixed-d0 requirement. Implement this minimal nonvendor max path instead:

```python
# extra_training/phases/rank_loss.py, inside iptm_pooled(...)
if pool == "max" and n_pair_tokens is None:
    return iptm(pae_logits, feat, configs, pair)  # preserve old behavior
if pool not in ("mean", "max"):
    raise ValueError(...)
# Share the existing mean-path mask / live / row_t setup, including ligand
# exclusion and uncropped n_pair_tokens when provided.
t, cnt = rw.row_t(pae_logits, live, tm, n_pair_tokens=_npt,
                  min_bin=bp["min_bin"], max_bin=bp["max_bin"])
if t is None or not bool((cnt > 0).any()):
    return None
valid = t[cnt > 0]
return valid.max() if pool == "max" else valid.mean()
```

Here `pae_logits` is the ORIGINAL full-token-grid tensor accepted by
`rank_loss.iptm_pooled`, **not** the frame-row-sliced `[F,T,64]` tensor
passed to `PAEPoseReranker`; compute both from the **same live head output**.
Ensure all masks use model-chain IDs from deployable metadata and match the
original scorer's exclusion of ligand tokens from anchor rows. A one-interface
ranking group must share a single fixed uncropped token count and never mix
other interfaces or reference-derived chain maps.

Numerical review gates: with no override, old `pool=max` behavior must be
unchanged; with `n_pair_tokens` equal to the actual chain-pair token count,
`row_t(...).max()` must match the original Protenix max selector on an
uncropped fixture (including its correct chain remap and bin parameters);
with a 256/384/512 crop and fixed uncropped count, assert the same d0 is
actually used. Prove nonzero gradients to the live PAE/head parameters,
correct handling of `cnt==0` rows, and batch/sample/crop parity. The
zero-init residual yields the **fixed-d0 cropped max score**, which can
differ from the shipped *uncropped* max selector: compare shipped separately
on the same candidate poses for evaluation, and do not claim literal
uncropped shipped parity at step 0.

## CPU gates, before consuming GPUs

From the repository root (the GitHub checkout):

```bash
python -m unittest discover -s extra_training/joint_reranker -p 'test_*.py' -v
python -m compileall -q extra_training/joint_reranker
```

They must be executed and pass on the target machine, not merely exist.
The first integration smoke test also verifies live gradients reach both
module parameter sets and the exact-zero residual reproduces the current
ipTM **before training**.

## 3 x 96 GiB GPU capacity protocol — back to back

**After user/agent safely checkpoints and stops the current job**, reserve
the **same three 96 GiB GPUs** for the entire sweep. Run all trials
sequentially within that allocation; each candidate trial is an independent
torchrun subprocess so an OOM does not poison later tests or lose the
reservation. Example assuming those cards are exposed as 4,5,7:

```bash
export CUDA_VISIBLE_DEVICES=4,5,7

# First implement a real, importable Protenix factory, e.g.
#   extra_training.joint_reranker.protenix_probe_adapter:build_probe
# Must construct the exact head, original PAE CE, rank loss, optimizer,
# full K-pose backward, and realistic PAE masks and original training trunks.
python -m extra_training.joint_reranker.gpu_capacity_sweep \
  --ranks 3 --steps 3 --headroom-gib 5 \
  --cases representative,worst \
  --grids '256:4,8,12,16,20,24;384:2,4,6,8,10,12;512:2,3,4,5,6,8' \
  --out-dir training/joint_reranker_capacity \
  --command-template 'torchrun --standalone --nproc_per_node=3 -m extra_training.joint_reranker.gpu_probe_step --factory extra_training.joint_reranker.protenix_probe_adapter:build_probe --crop {crop} --poses {poses} --case {case} --steps {steps} --prefix {prefix}'
```

`--dry-run` prints commands without running GPUs. Never call its output a
measured capacity. At each trial, verify >=3 real backward/AdamW steps, live
nonzero head and reranker gradients, peak reserved VRAM, actual free VRAM,
step time and identical input crop size. A failed trial must leave the
next trial safe; the driver kills all torchrun workers if one hangs.
Non-OOM runtime errors are **blockers**, not a smaller-memory verdict.

The driver writes `trials.csv`, `report.md` and `report.json`:
- **Largest completed tested K** (can be near OOM)
- **Largest K with >=5 GiB unreserved VRAM per GPU** (launch recommendation)

Do not infer K from the naive square-law alone. For a 96 GiB GPU, 5 GiB
headroom means ideally staying at or below ~91 GiB of overall device usage;
verify the actual per-rank free memory, not only the PyTorch allocator peak.
If large-crop groups only support K=2–5, retain those structures and use
masked variable-K group processing rather than discarding them.

The runner's allocator-based headroom is approximate: CUDA/NCCL allocations
outside PyTorch's caching allocator, GPU sharing, and real-data extremes can
consume extra VRAM. Verify actual device free memory and a short sustained
training smoke, especially before using a near-OOM count.

## Alpha and temperature calibration (NO recycled `alpha_rank=59`)

Start with `tau=0.05` in score units, but treat it as an initial hypothesis.
The zero-initialized score residual means the first gradient is dominated
by current ipTM; after training starts, reranker corrections alter it.

1. Run several **mixed-label** 20-pose (or variable-K) real training groups
   covering multiple structures and different win-rate bins, but **do not**
   update weights during calibration.
2. Compute `g_PAE = grad(mean(PAE CE), head_trainable_params)` and
   `g_rank = grad(listwise_rank, head_trainable_params)`, both accumulated
   over the SAME effective batch/window. Also record cosine similarity
   and gradient ratio **after** any DDP all-reduce; the mean of per-pose
   norms is not the norm of the sum.
3. Initialize `alpha = r * ||g_PAE|| / ||g_rank||` with `r=0.3`.
   A first compact sweep is `r in {0.1, 0.3, 1.0}`.
   Take a robust median across representative windows, not a single pose.
4. Re-measure after the reranker's residual output stops being zero.
   Log the actual shared gradient ratio per epoch and check for conflict.
   Choose based on held-out **top-1 acceptable fraction** first, mean
   DockQ and top-5/mid-rank second; don't optimize rho alone.
5. If ranks collapse after lowering tau, try `tau=0.1` (keeping the
   alpha-gradient ratio matched). Small tau can saturate and make only the
   winning or nearly winning poses contribute; report probability mass
   on acceptable poses and score spread. Never tune temperature and alpha
   by raw loss magnitudes alone.
6. Track **unweighted PAE CE on good and bad poses** relative to shipped,
   because the prior TM-weighted arms sacrificed good-pose PAE despite
   improving their weighted objectives.

For a new objective, the previous `alpha_rank=59` is not transferable:
its gradient units, candidate-set size and score derivative are different.

## Acceptance gates and rollback

1. CPU tests, shape and numerical gradient checks pass.
2. Baseline parity at step zero, using the **same current head**
   and the same (seed, sample, scored-chain-pair, d0) keys.
3. Reranker scores permute with input poses; mask construction uses no
   ground-truth structure or reference-only mappings.
4. Confirm ordinary PAE CE and listwise gradient both reach head; rank
   gradient also reaches attention/pair encoder after final-layer updates.
5. Record capacity per 256/384/512 via **real full-step GPU tests**.
6. Launch a brief in-sample overfit/sanity check and one val pass on the
   existing leak-free split, then a controlled run. Freeze NOTHING.
7. Preserve existing experimental branches/checkpoints; if a gate fails,
   stop only this opt-in arm and leave legacy paths untouched.

This is a working model/loss/integration *interface*, not a claim that the
full current Protenix trainer already runs this architecture. The missing
factory and adapter are intentionally explicit to prevent false GPU claims.
