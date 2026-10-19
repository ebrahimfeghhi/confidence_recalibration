# The train/validation split, and why they are not equally hard

**THE single file for how the two sets differ in difficulty.** `docs/validation_set.md` is the
authority on how the validation set was *chosen*; this doc is about what that choice did to the
difficulty of the resulting sets, and how to read a number measured on one against a number
measured on the other.

Status: written 2026-10-19, corrected the same day. The difficulty gap is measured. Its cause is
NOT explained: size accounts for about 15% of it, and the first replacement hypothesis — a
validation-only antigen filter — turned out not to exist. See *What does not explain it*.

## The three sets

| set | structures | interfaces | drawn from |
|---|---|---|---|
| training fold | 500 | 1,011 | SAbDab2, the clusters not held out |
| validation fold | 53 | 92 | 40 held-out clusters, one or two representatives each |
| `overfit30` | 30 | 77 | a headroom-selected subset of the training fold |

Seeds 500000-500009 (10 seeds x 5 samples = 50 poses per interface) throughout, except the
validation baseline, which is quoted at 100 seeds.

## The gap

Shipped confidence head, selecting by `chain_pair_iptm`:

| set | n | shipped mean DockQ | shipped frac acc | oracle mean DockQ | oracle frac acc | headroom |
|---|---|---|---|---|---|---|
| validation | 92 | **0.5775** | 0.7283 | 0.7171 | 0.8696 | 0.1396 |
| training fold | 1,011 | **0.4202** | 0.5344 | 0.5401 | 0.7023 | 0.1199 |
| `overfit30` | 77 | 0.0420 | 0.0000 | 0.5098 | 0.8831 | 0.4678 |

Validation is easier by 0.157 mean DockQ at the shipped head. Crucially the **oracle** is also
0.177 higher, so this is not the selector doing better on validation — the poses themselves are
better. The difficulty is in the structures.

`overfit30` is a third thing and must not be read as a difficulty statement about training data.
It was selected for headroom — interfaces where an acceptable pose exists and the shipped head
fails to find it — so its 0.0420 is a property of the selection, by construction. Its oracle
(0.5098 / 0.8831) is close to the training fold's, which is the right reference for it.

## What this means for reading results

- **A gain on `overfit30` does not transfer at face value.** Headroom there is 0.4678; on
  validation it is 0.1396. A method that closes half the gap on the capacity probe has a third
  as much room to show it on validation.
- **The validation set is small enough to be moved by single interfaces.** 92 interfaces, so one
  flipping from 0 to 1.0 moves mean DockQ by ~0.011. Both earlier 1e-3 arms' validation gains
  came from the same 3 interfaces (`9qrv_AB`, `29tj_BG`, `9bx7_BC`) carrying 85-88% of the total,
  with identical deltas — the same pose selected. Any validation claim has to survive that check.
- **12 of 92 validation interfaces have no acceptable pose at all**, and 13 are "winnable" — an
  acceptable pose exists and the shipped head misses it. The reachable gain is bounded by those
  13, not by the 92.

## Size explains some of it, and not most of it

The validation selection caps representatives at 1,000 tokens (rule R1 in
`docs/validation_set.md`, because `z_trunk` is `[N, N, 256]` and the cache grows as N^2). The
training fold has no such cap. Measured token counts (`z_trunk.shape[0]`, one unit per
structure):

| set | n | median | mean | p90 | max | frac > 1000 |
|---|---|---|---|---|---|---|
| validation | 53 | 481 | 500 | 699 | 799 | 0% |
| training (120 sampled) | 120 | 673 | 715 | 1,138 | 2,289 | 13% |

And size does track difficulty inside the training fold — oracle DockQ by structure size, over
all 1,011 interfaces:

| tokens | n | oracle DockQ | oracle frac acc |
|---|---|---|---|
| < 400 | 74 | 0.5849 | 0.7703 |
| 400-600 | 288 | 0.5904 | 0.7674 |
| 600-800 | 327 | 0.5885 | 0.7339 |
| 800-1000 | 138 | 0.4584 | 0.6014 |
| > 1000 | 184 | 0.4188 | 0.5924 |

`pearson(token count, oracle DockQ) = -0.225`. There is a clear step down above 800 tokens.

**But restricting the training fold to the validation size regime barely closes the gap.**
Training interfaces at <= 1,000 tokens: n=827, oracle 0.5671 / 0.7267, against validation's
0.7171 / 0.8696. So of the 0.177 oracle gap, size accounts for about 0.027 — roughly 15%. The
remaining 0.150 is something else.

## What does not explain it

Size is the only candidate measured so far, and it is not enough. The other differences between
the two selections, all from `docs/validation_set.md`, are **untested hypotheses**:

1. **Antigen-type composition, not filtering.** An earlier version of this doc claimed the
   polypeptide-antigen filter was validation-only and called it the strongest candidate. **That
   was wrong.** `phase_a_define_train_set.py:119` defines `POLY = ("PROTEIN", "PEPTIDE")` and
   line 287 applies it inside `training_clusters`, which that file calls "THE DEFINITION OF THE
   TRAINING SET". Both sets require a polypeptide antigen. The 2026-10-12 correction recorded in
   `validation_set.md` fixed an *inconsistency* — validation's representative-picker had omitted
   a filter training already had, so a hapten-only pairing could have been picked — rather than
   adding a filter validation alone has.
   What survives is weaker: `POLY` admits `PEPTIDE` as well as `PROTEIN`, so short peptide
   antigens are in both sets, and the two could still differ in how much of each they drew.
   Measurable by splitting both sets' oracle DockQ by `agtypes`.
2. **The 150-token filter** on validation representatives. Check whether the training definition
   applies an equivalent minimum before treating this as a difference — the antigen-filter error
   above came from assuming a validation rule had no training counterpart.
3. **R2: clusters with <= 5 post-cutoff structures.** Validation is deliberately built from
   *small* clusters, to preserve training data. `validation_set.md` already flags that "any
   performance claim must be stated for that population, not for antibody-antigen complexes in
   general."
4. **R3: 24 all-new + 16 mixed clusters.** A "mixed" cluster is one Protenix has *older*
   structures of. Those 16 could be systems the trunk has effectively seen relatives of, which
   would make them easier to fold — in the direction of the observed gap. Worth separating the
   validation baseline by this flag: if the 16 mixed clusters score well above the 24 all-new
   ones, part of the gap is leakage-adjacent rather than intrinsic.
5. **One representative per cluster.** Validation takes the cluster's representative (preferring
   the smallest by token count among eligible); the training fold keeps every structure in a
   cluster, hard ones included.

Note that 2 and 5 push the same way: validation is filtered toward the easier end of the
distribution and training is not. None of this is a flaw in the validation design — R1 and R2
exist to keep the cache affordable and to preserve training data — but it does mean the two sets
are not samples from one population, and a difference between them is not evidence about a
method.

## Measurements that would settle it

In rough order of value per minute, none needing a GPU:

1. Validation baseline **split by the all-new / mixed flag**. Tests hypothesis 4, now the
   strongest candidate, and it is the one with implications beyond difficulty.
2. Oracle DockQ on **both** sets split by `agtypes` (PROTEIN vs PEPTIDE). Tests hypothesis 1 in
   its surviving composition form. Split both sets, not just training — the error this replaced
   came from comparing one set against an assumption about the other.
3. Oracle DockQ on training interfaces split by **antigen length**, for hypothesis 2.
4. A size- and antigen-type-matched subsample of the training fold, scored against validation.
   If a matched subsample still scores well below validation, the remaining gap is in the cluster
   selection (hypotheses 3-5) and the two sets should be described as different populations
   rather than reconciled.

## Where the numbers come from

- Validation baseline: `training/ft3_pilot_baseline.json` (shipped head, 100 seeds, 92
  interfaces, written by `phases/phase_k_eval.py`, which also prints it as the number to beat).
  `docs/pose_provenance.md` records that the evaluator reads 0.5775 where `results.csv` reads
  0.5779, because `results.csv` rounds ipTM to 4 dp.
- Training fold shipped head: the two-row training-set comparison, 975 interfaces at the time it
  was measured (the pool is 1,011 at seeds 500000-500009; 9dzj's seeds 500000/500001 still need an
  OST re-score, which is why counts of 975/999/1011 all appear in the record).
- Training fold oracle, token counts and the size bands: computed 2026-10-19 from
  `training/fold/results.csv` and `z_trunk.shape[0]` per structure.
- `overfit30`: `training/overfit30/ceiling/ceiling.json`.
