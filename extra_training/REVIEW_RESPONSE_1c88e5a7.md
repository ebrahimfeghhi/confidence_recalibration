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

*The mean-centring absorbs additive offsets — and NOT d0.* **I had this wrong and the second
review corrected it.** I wrote that one shared mask makes `d0` "cancel exactly". It does not. iTM's
per-pair contribution is `1 / (1 + (d/d0)²)`, so `d0` enters **nonlinearly**: changing it reweights
how different distance errors count, which can move the relative gaps and the ordering between
poses even though every pose is scored under the identical formula. A shared mask guarantees the
poses are *comparable to each other*; it does not make the cropped and full-structure targets
order-equivalent. What mean-centring removes is a per-interface additive offset, which is a
strictly weaker statement than the one I made.

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

---

# Round 2: response to the review of e0ee07b5

Both blockers fixed, the `--lean-readouts` omission fixed, and one piece of my own mathematics
retracted. Still not launched.

## Blocker 1 — the launcher could still use the old mask. FIXED.

The review is right: the response said "the arm should not launch on the static-only mask" and the
launcher only checked `[ -f "$MASK" ]`. A content digest proves the file has not *changed*; it
cannot prove the file was *correct*. A static-only mask hashes just as stably as the right one.

* `build_union_crop_masks.py` now records `force_gt`, `ost_pairs`,
  `ost_poses_per_structure` and the list of `ost_extra_structures` in `__meta`.
* New `run_logging/check_crop_mask.py` reads that metadata and refuses a mask that does not
  establish the required configuration — missing `__meta`, absent `force_gt` or `ost_pairs`, the
  wrong structure count, any structure built from the wrong pose count, or the wrong
  margin/contact.
* The launcher points at `union_crop_m5_gt_ost.npz`, runs that check as a gate, and its documented
  build command now carries `--force-gt --ost-pairs --fold`.

Verified by refusing the two masks that exist: both the static-only one and the first `--ost-pairs`
build (which predates the metadata) are rejected, each naming exactly what it fails.

## Blocker 2 — partial OST resolution was accepted. FIXED.

Right again, and the invariant is stated better in the review than it was in my code: not "at
least one mapping exists" but "every supervised pose's mapping was examined". A structure with 49
of 50 poses resolved was being accepted, and the unexamined pose can reference a pair absent from
the forced set.

* `ost_pairs_for()` returns a `failures` list of `(seed, sample, why)` instead of silently
  `continue`-ing, and a seed whose unit dir cannot be resolved counts **all** `nsample` of its
  poses as failures rather than zero.
* The caller refuses the whole build unless `n_resolved == len(seeds) * nsample` with no failures,
  naming the first offenders and their reasons.
* **The fallback is gone, not merely reported.** "Named but accepted" still shipped a mask whose
  guarantee did not hold for the named structures.
* `--ost-pairs` now requires `--seed-base` and `--budget`, so an empty seed range cannot examine
  zero poses while the metadata claims otherwise.

## `--lean-readouts` — FIXED

It is now in the launcher's argv, with a header note giving the reason: 2.7 GB at the 855-token
boundary for bit-identical PAE logits.

## WHAT FINDING 2 ACTUALLY COST, now measured

The first `--ost-pairs` build reported it:

> **37 structure(s) had an OST-resolved pair the static pool mapping does NOT name**

So on **37 of 368 structures (10%)** the forced set would have protected the wrong chain pair.
Finding 2 was not hypothetical.

## The d0 argument — I was wrong, and the correction stands

See the retraction inline in finding 3 above. `d0` enters iTM as `1 / (1 + (d/d0)²)`, so it is not
an additive offset and mean-centring does not cancel it. A shared mask makes the poses comparable
to one another; it does not make the cropped and full-structure targets order-equivalent. My
"cancel exactly" was too strong.

I accept the review's framing: this is an **intentional approximation** — predict full-structure
quality from partial context — and should be documented as such rather than described as
crop-consistent supervision. It is not a claim that the targets agree.

## Proposed next steps — NOT agreed

The reviewer proposed these and I think they are sensible. **The user has not agreed to any of
it** and holds the decision on all GPU time. An earlier version of this section called it an
"agreed plan", which it was not; that was me putting a decision in the user's mouth.

1. Mask rebuilt under the current code, so coverage is enforced rather than asserted. *(in flight;
   this one needs no cards and no decision)*
2. One GPU canary in the production configuration — cropped input, lean readouts, dropout 0.25,
   both pose pathways, cross-seed relative regression, checkpointing on and off across the
   threshold — checking finite gradients, real peak memory, and Pass A/Pass B agreement.
3. A **short pilot**, not 60 epochs, with uncropped held-out and uncropped in-sample selected
   DockQ as the primary readouts and cropped training Spearman as secondary.

The reviewer is willing to defer finding 3's label experiment past a pilot. Whether anything runs
at all, and in what order, is the user's call.

Nine CPU suites green. **Not launched.**


---

# Round 3: response to the review of c686a6bf

**This round named no new code bugs**, and said so: "I don't see another major code-level blocker
in the latest changes." Two items were actionable; both are done. No runs were authorised and none
were started.

## The mask rebuild — PASSES

Rebuilt under the coverage-enforcing code and validated by the launcher's own gate:

```
mask OK: 368 structure(s), margin 5.0 A, contact 5.0 A, force_gt=True, ost_pairs=True,
built 2026-11-03T00:15:12Z by c686a6bf; 37 structure(s) needed an OST-resolved pair the
static mapping does not name
```

The build did not refuse, which is the substantive part: **every one of 368 structures had all 50
poses resolve through OST**, so the exact-coverage invariant holds on real data rather than in
principle. Kept fraction median 0.5980, 50 poses per union on every structure, rule verified
against `interface_crop_tokens` on 368 of 368.

The reviewer cannot verify this from GitHub because the npz is gitignored, so the gate output above
and `check_crop_mask.py` are the auditable artefacts.

## The L2-SP penalty — the report now answers the question asked

The review's point was that "λ = 1e-4" does not establish weak regularization when the squared
displacement is large. It was already reported per epoch as `||W-W0||^2` and
`penalty = λ·||W-W0||^2`, but in absolute terms only, which does not answer it:

| ‖ΔW‖² | penalty at λ=1e-4 | share of a 2.9 loss | penalty at λ=1e-5 | share |
|---|---|---|---|---|
| 100 | 0.010 | 0.3% | 0.001 | 0.03% |
| 1,000 | 0.100 | 3.4% | 0.010 | 0.3% |
| 10,000 | 1.000 | 34.5% | 0.100 | 3.4% |
| 67,052 *(what the unconstrained head reached)* | 6.705 | **231%** | 0.671 | 23% |

So at the displacement the unconstrained full head actually reached, 1e-4 would be more than twice
the objective — the review is right that the coefficient alone settles nothing. The epoch line now
prints the penalty **as a percentage of that epoch's own mean loss**, plus what 67,052 would cost
at the configured λ, so the question is answered in the log instead of by a reader doing
arithmetic.

**AND THE ARM IS NOW λ = 1e-5, NOT 1e-4.** The user lowered it on seeing the table above: two
orders below tv6hue's 1e-3 rather than one, and ~23% of the objective at the worst displacement on
record instead of 231%. The intent was always "anchor lightly, do not carry the arm", and 1e-4 did
not reliably deliver that.

## The dropout-across-poses variance — recorded, not changed

The review raised it and classified it correctly as a property rather than a bug: the RNG restore
aligns Pass A with Pass B *per pose*, which is what gradient correctness needs, but each pose still
draws its own mask, so the relative-regression loss compares poses evaluated under different
realisations of the head.

Nothing is changed for that. Recording the suggested alternative here so it is not lost: **if
training proves unstable, share one dropout mask across a structure's poses within a step.** That
would make the 50 poses differ only in their coordinates, which is closer to what the loss means to
compare. It is a real experiment, not a fix, and it is not in this arm.

## State

Mask passes. Nine CPU suites green. The launcher's eight test gates and the mask provenance check
all pass. **No runs authorised, none started, and the user holds every decision about GPU time** —
including the canary and pilot the reviewer recommends.
