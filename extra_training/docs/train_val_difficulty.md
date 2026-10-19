# The train/validation split, and why they are not equally hard

**THE single file for how the two sets differ in difficulty.** `docs/validation_set.md` is the
authority on how the validation set was *chosen*; this doc is about what that choice did to the
difficulty of the resulting sets, and how to read a number measured on one against a number
measured on the other.

Status: written 2026-10-19, corrected twice the same day. The difficulty gap is measured. Its
cause is NOT explained: size accounts for about 15% of it, and two replacement hypotheses — a
validation-only antigen filter, and an era/mixed-cluster imbalance — were both checked and do
not exist. See *What does not explain it*, which also records why both were wrong in the same
way.

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

Three candidates have been checked. None closes the gap.

**1. Structure size — about 15% of it.** Measured above: restricting the training fold to
validation's <= 1,000-token regime moves its oracle from 0.5401 to 0.5671, against validation's
0.7171.

**2. The polypeptide-antigen filter — applies to both sets.** An earlier version of this doc
called this the strongest candidate, on the reading that validation required a protein antigen
and training did not. Wrong: `phase_a_define_train_set.py:119` defines
`POLY = ("PROTEIN", "PEPTIDE")` and line 287 applies it inside `training_clusters`, which that
file calls "THE DEFINITION OF THE TRAINING SET". The 2026-10-12 correction in
`validation_set.md` fixed an *inconsistency* — validation's representative-picker had omitted a
filter training already had — rather than adding a validation-only rule.

**3. Deposition era and the mixed/all-new split — matched.** The next version of this doc
proposed that R3's 16 "mixed" clusters (systems Protenix holds older structures of) could be
easier and inflate validation. Also wrong, twice over. "Mixed" is a property of a *cluster*: 107
of the 533 pool clusters are mixed, validation took 16, and the other 91 are in training. And
both folds are post-cutoff only — all 500 training structures have `release_date >= 2021-09-30`
(`training/prep/job_indices.csv`), as validation is by construction. If anything the era
argument runs the wrong way: pre-cutoff structures are the ones Protenix trained on, so a
training fold containing them would be *easier*, and it contains none.

### The mistake those two have in common

Both wrong hypotheses came from reading `validation_set.md` — which documents validation's rules
in detail and training's not at all — and treating each rule as absent from training because that
doc does not mention it. **A rule described in the validation doc is not evidence of a difference.
Check the training definition before counting one.**

So the honest state is: the gap is real, size accounts for roughly 15% of it, and nothing checked
so far accounts for the rest.

### What is left, and the one measurement that would settle it

Still-unchecked differences, all from `docs/validation_set.md`:

- **R2**: validation clusters hold <= 5 post-cutoff structures, so validation is deliberately
  built from *small* clusters. `validation_set.md` already warns that "any performance claim must
  be stated for that population, not for antibody-antigen complexes in general."
- **The 150-token filter** on validation representatives — check for a training counterpart
  first.
- **One representative per cluster**, preferring the smallest eligible by token count, against
  every structure in a training cluster.
- **Antigen-type composition** within `POLY`: it admits `PEPTIDE` as well as `PROTEIN`, so the
  two sets could still differ in the mix.

Rather than testing these one at a time from the validation side — which is what produced two
wrong answers — the measurement to run is a single **side-by-side attribute comparison**: for
every structure in both folds, pull token count, antigen type, antigen length, cluster size and
deposition date, and compare the distributions. That says in one pass which attributes actually
differ, and only those are worth a difficulty breakdown. No GPU needed.

Until that exists, treat the two sets as different populations and do not read a train-versus-val
difference as evidence about a method.

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
