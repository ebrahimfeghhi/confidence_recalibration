# Review round 3 — the cache design, 2026-10-13

Reviewer's pass on `docs/cached_training.md` at 695 structures × 10 seeds × 5 poses. Their verdict on
the shape — several trunk seeds per structure, five inference-length poses per trunk — stands.

| # | finding | verdict | status |
|---|---|---|---|
| 1 | selector does not apply the polypeptide-antigen filter | **CONFIRMED, worse than stated** | fixed |
| 2 | MSA recovery workspace does not fit beside a 1.61 TB cache | **MOOT after 1** | n/a |
| 3 | storage counts only `z_trunk` | **CONFIRMED, +2.6%** | measured |
| 4 | large jobs: backward-pass memory untested | **CONFIRMED** | OPEN, blocks their caching |
| 5 | representatives chosen alphabetically; per-PDB dedup discards distinct jobs | **CONFIRMED** | OPEN |
| 6 | a cached training step needs more than confidence-head inputs | **CONFIRMED, design gap** | OPEN, blocks the build |

---

## 1. Eligibility — confirmed, and larger than reported

> The selector does not apply the polypeptide-antigen filter. The manifest contains 29 single-chain
> jobs, and 143 entries have no antigen cluster recorded.

Both counts reproduce exactly, and the real number is bigger: **186 of 695 rows had no polypeptide
antigen at all.** Antigen types present across the selection were PROTEIN 561, **SUGAR 175, HAPTEN
121, ION 63, RNA 19, DNA 1**, PEPTIDE 75. A hapten-only row is not an antibody–antigen complex for
this purpose — there is no antigen chain to fold, so there is no interface for the head to be
calibrated on.

`phase_a_define_train_set.py:263` applies this filter when it builds the pool and phases C and G apply
it again when they emit sequences; the selector read `abag_split.csv` directly and applied neither.

Fixed: `eligible()` now requires an antibody chain plus at least one polypeptide antigen, and reports
what it dropped. **695 → 536 structures, 207 → 193 clusters**, 548 rows dropped.

**A consistency check fell out of it.** All 140 structures previously blocked on missing MSAs were
ineligible rows — phase C only ever built MSAs for rows with a polypeptide antigen. After the filter,
**0 of 536 are MSA-blocked**. The MSA store and the eligibility rule agree exactly, which is evidence
both are right.

## 2. MSA recovery workspace — moot

> Caching the 555 ready structures at ten seeds uses about 1.61 TB, leaving only 430 GB — too little
> for the stated ~1 TB MSA recovery workspace.

The arithmetic was right and the conflict is real, but finding 1 dissolves it: there is nothing to
recover. No uniref30/envdb re-download, no 1 TB workspace, no sequencing problem.

## 3. Storage counted only `z_trunk` — confirmed, and quantified

> I reproduced the 1,703 GB total exactly from Σ N² × 512 × 10. That excludes single representations,
> coordinates, features, and file overhead.

Correct. Measured against 55 complete cache files rather than estimated:

| | |
|---|---|
| Σ N² × 512 over those files | 7.56 GB |
| actual bytes on disk | 7.75 GB |
| **ratio** | **1.0255** |

So the omission is **+2.6%**, not an order of magnitude — `z_trunk` really is ~97% of a unit, because
`s_trunk` is N×384, `s_inputs` N×449 and the five poses are N_atom×3 floats, all linear in N while
`z_trunk` is quadratic. All figures below use the measured ratio.

| | structures | 5 seeds | 10 seeds |
|---|---|---|---|
| selection after the eligibility fix | 536 | **848 GB** | **1,695 GB** |

Of 2,039 GB free, so 10 seeds leaves 344 GB.

## 4. Large jobs — confirmed, and this blocks caching them

> In the current manifest, 76 structures above 1,000 tokens consume about 730 GB — 43% of the
> pair-tensor budget.

Reproduced on the corrected selection: **77 structures above 1,000 tokens, 45% of the bytes, 765 GB
at 10 seeds.** The largest is 2,941 tokens.

Agreed that the backward pass has to be established first. The head's 4 pairformer blocks hold
N×N×256 activations with gradients; at 2,941 tokens one such tensor is 4.4 GB in bf16 and a block
holds several. Paying ~765 GB to cache structures we then cannot train on is the worst outcome, and
the test is cheap: one forward and backward through the head at 1,000, 1,500, 2,000 and 2,941 tokens
on a free card. **Not run — the cards are folding.** Until it is, the cache build should be ordered
small-first so the ≤1,000-token structures exist whatever the test says.

## 5. Representative choice and deduplication — confirmed

> Currently, the selector takes candidates in alphabetical order. [...] global PDB deduplication can
> discard distinct antibody–antigen instances from the same entry.

Both confirmed. The cap iterates `sorted(cand[cl])`, so within a (antibody, antigen) pairing the two
kept structures are whichever sort first by antibody cluster id — no diversity criterion at all.

And the per-PDB rule measurably discards real jobs. Of the 536 selected structures, **277 have more
than one eligible instance, 143 have instances that differ in actual input** (heavy, light or antigen
sequence), and per-PDB dedup throws away **201 distinct jobs**. Those are different complexes that
happen to share a deposition.

Both are open. The fix for the second is to key deduplication on the input tuple
(Hseq, Lseq, sorted polypeptide antigens) rather than the PDB id; the fix for the first is to prefer
candidates that differ in folded sequence, which needs a criterion chosen deliberately rather than
invented here.

## 6. A cached training step needs more than confidence-head inputs — confirmed, and it is a design gap

> The listed tensors cover confidence-head inputs, but training also needs correctly mapped reference
> coordinates, masks, frames, and per-pose symmetry handling.

This is the finding that blocks the build, and the doc was wrong to imply otherwise. The cache as
specified stores what `run_confidence_head` consumes. The **loss** needs more, and none of it is in
the npz:

- ground-truth coordinates for the cropped/selected atoms, in the model's atom ordering;
- `coordinate_mask`, the `pae_rep_atom_mask` and `distogram_rep_atom_mask`, `frame_atom_index` and
  `has_frame` for the PAE loss's per-residue frames (`loss.py:880-940`);
- the **symmetric permutation** of the ground truth to match each pose
  (`protenix.py:1712-1720`), which is per pose, not per structure.

The permutation is the subtle one: with homomeric chains there is no canonical correspondence, so
FT3 relabels the label to whichever assignment the pose implies. Five poses per trunk means five
permutations. Caching the pose without the permuted label would train the head against a
mis-assigned truth for exactly the symmetric cases.

So the cache contents have to be extended, and before any of it is generated the right test is a
**numerical replay**: run one cached unit through the loss and confirm it reproduces, to within
floating point, the loss computed by the uncached path on the same structure and seed. That is the
check that the cache is precomputation rather than a different experiment.

---

## Open after this round

| # | finding | severity | blocks |
|---|---|---|---|
| 6 | cache lacks labels, masks, frames and per-pose symmetry permutation | **high** | the build |
| 4 | backward-pass memory untested above ~1,000 tokens | **high** | caching 77 large structures |
| 5 | per-PDB dedup discards 201 input-distinct jobs; representatives chosen alphabetically | medium | the selection |

Accepted without change: 10 seeds × 5 poses as the target, and the reviewer's note that more poses
can be drawn later from a saved trunk if the sampler inputs are preserved — so `N_sample=5` is a
default, not a ceiling.
