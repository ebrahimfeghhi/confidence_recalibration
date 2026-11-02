# Response to the review of 1c88e5a7 / da0a4f9f

Written 2026-11-02 evening. Nothing has been launched; the green light is still held.

**Summary: findings 2, 4 and 5 are fixed, and the review was right about all three. Finding 1 was
already measured and I had the number. Finding 3 is correct and I have made it worse, not better
— the measurement below is bigger than the review assumed, and the exact measurement it asks for
is not cheap. Findings 6 and 7 are judgment calls and belong to the user.**

Fixing finding 5 also surfaced a bug I had introduced in the same patch series that would have
killed the run at startup. That is recorded below rather than quietly corrected.

---

## Finding 2 (HIGH, forced coverage uses the wrong pairs) — CONFIRMED and FIXED

The review is right, and this was the one that mattered. `--force-gt` built its forced contact set
from the **static** pairs in `pool_winnable.csv` + `chain_roles.csv`. The trainer does not
supervise static pairs: `phase_k_train._pose_pairs()` resolves the chain pair **per pose through
OST**, and its own docstring records why and by how much:

> the two mappings disagree on 1,095 of 18,400 (structure, seed, pose) checks across 52 of 368
> structures, and 7stf resolves 7stf_BC to (3,1) statically against (3,0) per pose — a different
> chain, not a reordering

So on up to 52 structures the forced set could protect a pair the trainer is not scoring, leaving
the supervised one partly cropped. That is exactly the hole `--force-gt` exists to close, and §16's
"coverage 1.0 by construction" was true of static pairs and unproven for supervised ones.

**FIX.** `build_union_crop_masks.py` gains `--ost-pairs` (requires `--force-gt` and `--fold`),
which resolves each pose's pair the way the trainer does and forces the contacts of the **union of
both mappings**. Covering both is strictly safer than choosing between them and costs almost
nothing, since they agree on 316 of 368 structures. The builder now also:

* reports how many structures had an OST-resolved pair the static mapping does not name — i.e.
  exactly the structures whose forced set would otherwise have been wrong;
* **names** any structure whose OST mapping cannot be read and that therefore fell back to static
  pairs, rather than silently leaving it uncovered.

The rebuild with `--ost-pairs` was still running when this was written, so the count of affected
structures is not yet in this file. **The arm should not launch on the static-only mask.**

---

## Finding 5 (MEDIUM, guard checks the filename) — CONFIRMED and FIXED

The guard recorded `os.path.basename(a.crop_tokens)`. Rebuilding the mask with a different margin,
contact cutoff or forced-pair rule under the same name would have resumed as "unchanged" while the
whole training input distribution had moved.

**FIX.** A content digest computed at startup: sha256 over the per-structure masks in sorted order
(`np.packbits` of each) plus the builder's `__meta` provenance blob, truncated to 16 hex. Both
guard sides compare that digest instead of the basename, the crop banner prints it so a log says
which mask the run actually trained on, and pre-flag bundles get `""` as a migration default. It is
computed at startup rather than beside the loader because the guard dict is built first.

### A bug this surfaced, which would have killed the run at startup

Two patches in this series failed their own assertions and wrote nothing, but a later patch applied
by line number succeeded. The result: both guard lines referenced `_CROP_TOK_DIGEST` while the
block defining it did not exist, and a second, redundant digest under the old name
`_CROPTOK_DIGEST` survived in the loader. The trainer would have raised `NameError` before pose 0.

`test_guard_fields.py` passed throughout, because it checks that compared and saved field *names*
agree — not that the values are defined. `test_crop_tokens.py` now asserts `_CROP_TOK_DIGEST` is
**defined** and that the definition precedes both uses, and the three new startup guards are
exercised end-to-end against the real trainer:

```
STOP: --crop-tokens and --crop-margin are two crop rules at once ...
STOP: --crop-tokens /nope.npz does not exist ...
STOP: --crop-tokens needs --minimal-items ...
```

---

## Finding 4 (MEDIUM, two-pass gradient under dropout) — mechanism existed, now TESTED

The review was right that this needed verifying and right that the design was already correct.
`phase_k_train.py:~4531` saves `torch.get_rng_state()` and `torch.cuda.get_rng_state_all()` before
Pass A and restores both in a `finally`, with a comment naming dropout as the reason. What had
never happened is dropout being **on** — `ConfidenceHead` ships at 0.0, so the save/restore has
been a no-op in every arm to date.

`run_logging/test_two_pass_dropout.py` now exercises it on a real `ConfidenceHead` with dropout
live, and the hazard turns out to be severe:

| check | result |
|---|---|
| Pass A and Pass B agree pose-for-pose **with** the restore | `max|dA-dB| 0.000e+00` |
| **without** the restore they disagree | `1.213e-03` |
| coefficients sum to zero, so the centring term cancels | `-5.96e-08` |
| two-pass gradient vs one-pass autograd on the real objective | rel `9.98e-07` |
| **a Pass B that redraws gives a different gradient** | rel **`4.06e-01`** |
| the restore is absolute, so extra Pass-A draws do not shift Pass B | exact |

A 40% relative gradient error is what the guard is preventing. The test also pins that both the
CPU and CUDA halves are saved and restored in a `finally`, so a raising Pass A cannot leak state.

The test is CPU-only, so the CUDA restore itself is pinned in source rather than executed. A GPU
run of the same comparison with checkpointing on and off is still worth having, as the review
asked; this closes the arithmetic, not the device.

---

## Finding 1 (HIGH, 855 threshold not validated) — fair, but the number existed

The review is right that the launch script omits `--lean-readouts` while 855 was chosen from the
lean measurement. But both arms were measured on 2026-11-02, uncheckpointed pass-B peak at 855
tokens:

| configuration | peak |
|---|---|
| lean readouts ON | 73.51 GB |
| lean readouts OFF — the script as written | **76.18 GB** |
| ceiling an epoch is demonstrated to survive | 80.22 GB |

So the script as written has ~4 GB of margin. And the crop makes this **safer**, not riskier —
post-crop token counts are median 368, p90 654, max 1208, so **354 of 368 structures run
uncheckpointed and 14 are checkpointed**. Dropout's own cost is the row-wise masks, `[1, N, 256]`
per site, ~15 MB total at N=900 against 76 GB.

I am taking the recommendation to enable `--lean-readouts` anyway: it is free, buys 2.7 GB, and
`test_lean_readouts.py` proves the PAE path is bit-identical. A real epoch's peak line remains the
confirmation, and `kz536p` is producing one right now under threshold 855 + lean on the uncropped
manifest.

---

## Finding 3 (HIGH, full vs cropped true-iTM) — CORRECT, and the new measurement is worse

The review is right on the facts and right that my commit text said "global DockQ scalar" loosely:
the target is true iTM, the cached `g_mean`, computed on the full structure.

**I measured how much of each scored chain the crop actually keeps, and it is a bigger effect than
the review assumed:**

| fraction of a scored chain retained | |
|---|---|
| median | **0.620** |
| mean | 0.635 |
| p10 | 0.368 |
| min | 0.057 |
| chains kept whole (≥0.99) | 9.5% |
| chains kept ≥0.90 | 13.8% |
| **chains kept <0.50** | **22.7%** |

iTM for a chain pair runs over token pairs spanning the two chains, normalised by a `d0` set from
the pair's token count. At a median 62% per chain that is roughly 38% of the original token pairs
and a materially smaller `d0`. So the prediction's support moves a great deal while the label does
not.

**Two things that bound the concern, neither of which makes it go away.**

*The mean-centring absorbs the level, not the ordering.* The loss is
`[(s_i − mean_s) − (g_i − mean_g)]²` with one shared mask per structure, so `d0` and every other
per-interface constant cancel exactly. Only a change in the **ordering or the gaps** between a
structure's poses can hurt. That is the narrower question, and it is the one the review's proposed
Spearman measurement would answer.

*This mismatch is the status quo of the best result in the project.* `o2qczq` trained with the same
full-structure `--rel-labels` at **53.3%** of tokens — more aggressive than this crop — for 31
epochs, and produced the project's best in-sample numbers (+0.0733 / +0.1314, §9a). So the
objective is demonstrably trainable under it. The review's Option A, recomputing labels on the
crop, would be the **larger** departure from what has worked, not the conservative choice.

**Why I have not run the exact measurement.** `training/fold/itm_labels/*.npz` holds precomputed
scalars — `iface, seed, sample, g_mean, g_max` — produced upstream by `itm_ceiling.py`. There are
no per-token-pair true error bins stored to re-reduce under a mask. Computing a cropped true-iTM
means re-running the label pipeline with a token mask, which is a pipeline job rather than a quick
script, and I would rather scope it honestly than hand-wave a proxy. **It is the one finding I have
not closed.** If the user wants it before launch, that is the cost to price.

---

## Findings 6 and 7 — judgment, and the user's call

Finding 6 (too many changes at once) is the ablation point I raised before the user chose to
combine them. The cost is real: a null result will not attribute. Two things make it less bad than
it reads — L2-SP is at 1e-4, an order below the 1e-3 that produced −0.0084, so the crop and dropout
are the live variables; and the crop is motivated by speed, so it is not competing for credit.

Finding 7 restates the caution the launch script already names as the thing to watch, and the
review and I agree on the test: **in-sample selected DockQ measured on UNCROPPED inputs.** If
cropped training climbs while that stays flat, stop early. `insample_eval.py` refuses a cropped
bundle, so it is the right tool and it needs no new code.

---

## State

| item | status |
|---|---|
| Finding 2 | fixed; mask rebuild with `--ost-pairs` in flight |
| Finding 4 | fixed and tested (CPU); GPU variant still open |
| Finding 5 | fixed, plus the `NameError` it surfaced |
| Finding 1 | answered from existing measurement; `--lean-readouts` to be added to the script |
| Finding 3 | **open**; scoped, cost stated |
| Findings 6, 7 | user's call; 7's monitor is agreed |

Nine CPU suites green: `test_lean_readouts`, `test_lean_readouts_arch`, `test_ckpt_threshold`,
`test_epoch_order`, `test_pair_dropout`, `test_crop_tokens`, `test_two_pass_dropout`,
`test_guard_fields`, `test_lr_schedule`.

**Not launched.**
