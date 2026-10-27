# abag_rank/

Training an ABAG-Rank-style reranker on OUR poses: a small Deep Set over interface residue pairs
that regresses DockQ directly, instead of regressing PAE with Protenix's 19.8M-parameter
confidence head.

Upstream is <https://github.com/tadteo/ABAG-Rank> (code MIT, checkpoints CC BY-NC 4.0), cloned
read-only to `/home/ebrahim/abag_rank_upstream/ABAG-Rank`, OUTSIDE this repo so nothing of theirs
is committed here and there is no nested git checkout.

## Why this exists

Measured on our data (`../paper_results.md`):

- the best full-training-set arm is **+0.0024 mean DockQ** over shipped on valfull, ~2% of headroom
- `ocqtyt`, the unconstrained full head, captures **+0.0012** on its OWN training set -- it does not
  fit 487 structures in selection terms at all
- `jy6sah` on 30 structures reaches **+0.1965 in-sample** and **-0.0355 on valfull** with the same
  checkpoint

So a PAE objective converts into pose selection only with enough passes per complex, and what it
learns there does not cross complexes. ABAG-Rank changes the supervision (DockQ directly) and the
capacity (a small Deep Set, far less able to memorise per-complex).

## What we take from upstream, and what we do not

WE TAKE THE MODEL AND EVERY PREPROCESSING STEP THAT CHANGES A TENSOR. We do NOT reproduce their
file format. Their chain is AF3 JSON -> stage-1 HDF5 -> stage-3 stats -> shard packer -> their
dataloader -> the model; we have no AF3 JSON, so stage 1 was always going to be ours. Matching the
rest cost per-sample `inter_idx` duplication, a `tm_normalized_reference` we do not have, their
`train_test_split`, and a fixed-length-per-complex constraint that forced an elaborate union
design -- none of which touches a tensor.

Ported in full, from `preprocess_shards.py:133-275`:

| step | what it does |
|---|---|
| X filter | drop pairs whose residue is unknown, remap indices |
| distance cutoff | CA-CA <= 12 A, keeping the single closest pair when none qualifies |
| sort + truncate | order pairs, cap at `max_n` |
| PAE centring | `pae_centered = pae - pae_col_mean`, the per-complex column mean over its poses |
| triangular encoding | `cantor_norm`, `dist_norm` from the pair indices |
| one-hot residue | for both residues of each pair |
| label transform | DockQ -> `log(p/(1-p))` |

Per-pair feature vector: `[cantor_norm, dist_norm, pae_col_mean, pae_centered, ca_distance]` plus
the two one-hots.

Dropped: HDF5 schema, shard packer, manifest, their split. We use OUR split -- 487 training
structures, valfull's 92 interfaces held out -- or no number here compares to `paper_results.md`.

## Facts about our data that this code must respect

Each was established by measurement, and each has already caused or nearly caused a wrong result.

- **CA, not CB.** `ci_feat_distogram_rep_atom_mask` is CB for ordinary residues (CA only for Gly);
  upstream uses CA. On 10fj the masks differ at 1,158 of 4,795 atoms, 1.42 A apart on average, and
  at a 12 A cutoff the CB convention loses 198 real pairs and adds 430 spurious ones. The CA mask
  is `lf_pae_rep_atom_mask` in the LABEL npz.
- **Residue letters are not in the cache.** The trunk npz has six feature arrays and no restype;
  `ci_s_inputs` has no restype one-hot (400 of its 449 columns are identically zero). They come
  from phase_j's training dataset, the same featuriser that built the cache, with
  `phase_j_build_cache.check_alignment()` proving the token order elementwise.
- **One PDB is not one complex.** 382 of 487 training structures have more than one labelled
  interface (311 have two, 40 three, 29 four, 2 six). The unit is the interface, `7zra_AC`.
- **The chain mapping is decided PER POSE.** `phase_k_eval.ost_chain_pairs` reads
  `ost_json/<k>.json` because "the first antigen is not determined by sequence". OST disagreed with
  the static `chain_roles` table on 145 poses across 2 structures, one of them 7zra -- the
  structure whose ipTM join already failed.
- **Pose -> sample is a coordinate-identity join**, `BC.resolve_seed_dir`, never a
  `(pdb, seed, sample)` guess.
- **PAE must be expected error in Angstroms**, via
  `logits_to_score(**get_bin_params(configs.loss.pae))` -- Protenix's own function, not a
  reimplementation of the bin centres, and never raw logits or softmax probabilities.
- **Budget 10 on training data.** The training fold has 10 seeds x 5 samples = 50 poses; validation
  has 30 x 5 = 150. Training-set numbers are never comparable to budget-30 valfull numbers.

## Contents

| file | state | what it is |
|---|---|---|
| `README.md` | built | this |
| `restypes.py` | **built** | per-token residue letters, alignment proved against the cache (CPU) |
| `features.py` | planned | upstream's seven transforms, ported (CPU) |
| `dataset.py` | planned | a torch Dataset yielding what their Deep Set consumes (CPU) |
| `train.py` | planned | training loop over our split |
| `eval.py` | planned | selected-DockQ on valfull, by phase_k_eval's rules |

## Smoke test of the PAE dump, 2026-10-27

One unit (`22gi`, seed 400000, 5 poses) through `phase_k_eval.py --dump-pae`, on card 4 alongside
training. 174 KB for 5 poses, ~4,700 pairs each.

    PAE  (A): min 0.37 med 1.04 max 14.21, all finite
    CA-CA(A): min 4.38 med 13.36 max 16.00 (= the dump cutoff)
    all pairs inter-chain: True

CA distances were then recomputed INDEPENDENTLY from `ci_coordinate`, by code that does not call
`_ca_coords`, resolving CA as within-token atom index 1 by hand. Every dumped sample reproduces
**exactly, max |diff| 0.000000 A** -- and the pose-to-sample mapping is a NON-TRIVIAL PERMUTATION:

    sample 0 <- pose 2    sample 1 <- pose 0    sample 2 <- pose 1    samples 3,4 identity

So a naive `(pdb, seed, sample)` join would have mislabelled **3 of these 5 poses**. That is the
7zra failure class, reproduced on a different structure, and the dump is immune to it because it
rides `BC.resolve_seed_dir`'s coordinate-identity join rather than an index assumption.

NOT YET EXERCISED by this smoke test: the closest-pair fallback (0 pairs beyond the cutoff here,
so every chain pair had contacts within 16 A) and any multi-interface structure. `7zra` remains
the fixture to run.

The PAE dump itself lives in `phases/phase_k_eval.py --dump-pae`, because it rides that file's
coordinate-identity join; putting it anywhere else would reintroduce the 7zra class of bug.
