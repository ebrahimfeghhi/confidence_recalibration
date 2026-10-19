# Train/validation: how each set was built, and how differently they score

**THE single file for the performance disparity between the two sets.** `docs/validation_set.md`
is the authority on the validation selection procedure; this file records what each set is, the
code that produced it, and the measured difficulty gap.

Status: written 2026-10-19. The gap is measured. Its cause is **not** established — see
*What has been ruled out*.

## Summary metrics

Shipped Protenix confidence head, selecting the highest `chain_pair_iptm` of the available poses,
reporting that pose's DockQ. Seeds 500000-500009 (10 seeds x 5 samples = 50 poses per interface),
except the validation baseline, which is quoted at its full 100-seed budget.

| set | structures | interfaces | shipped mean DockQ | shipped frac acc | oracle mean DockQ | oracle frac acc | headroom |
|---|---|---|---|---|---|---|---|
| **validation** | 53 | 92 | **0.5775** | 0.7283 | 0.7171 | 0.8696 | 0.1396 |
| **training fold** | 500 | 1,011 | **0.4202** | 0.5344 | 0.5298-0.5401 | 0.6855-0.7023 | ~0.11-0.12 |
| `overfit30` | 30 | 77 | 0.0420 | 0.0000 | 0.5098 | 0.8831 | 0.4678 |

*frac acc* = share of interfaces whose selected pose has DockQ >= 0.23. *oracle* = the best pose
available, i.e. a perfect selector. *headroom* = oracle minus shipped.

**Validation scores 0.157 higher at the shipped head, and its oracle is 0.177 higher.** Since the
oracle moves with it, this is not the selector performing better on validation — the available
poses are better. The difficulty is in the structures, not in the ranking.

`overfit30` is not a difficulty statement about training data. It was selected for headroom —
interfaces where an acceptable pose exists and the shipped head fails to find it — so its 0.0420
is a property of that selection. Its oracle (0.5098 / 0.8831) is close to the training fold's,
which is the right reference for it.

### The training-fold oracle is a range, not a point

`training/fold/results.csv` holds **59,145 rows for 50,490 distinct `(iface, seed, sample)`
keys**: 8,655 keys carry two rows, left behind by a refold. For most the two agree to rounding
(median DockQ spread 0.002), but 111 interfaces have a disagreement over 0.1, and the worst is
0.944 — one row is a superseded pose and the other is current.

`results.csv` alone cannot say which. The production path resolves it by coordinate identity
against the trunk (`resolve_seed_dir`, and the pose->sample join in `phases/phase_k_eval.py`),
which is why the evaluator's own numbers are unaffected. Any oracle computed straight from
`results.csv` is therefore bounded, not exact:

| | oracle mean DockQ | oracle frac acc |
|---|---|---|
| taking the max of each duplicated key | 0.5401 | 0.7023 |
| taking the min of each duplicated key | 0.5298 | 0.6855 |

190 of the 1,011 interfaces are touched at all. The spread is ~0.010, so it does not change the
train/val story — the oracle gap is 0.177 — but a single number here would be the optimistic
bound presented as fact. The shipped-head figure 0.4202 comes from the evaluator, not from
`results.csv`, so it is not affected. The same caveat applies to the size bands below, which are
computed on the optimistic bound.

## Consequences for reading any result

- Headroom is 0.4678 on `overfit30` and 0.1396 on validation. A method that closes a given
  fraction of the gap on the capacity probe has about a third as much room to show it on
  validation.
- Validation is 92 interfaces, so one flipping from 0 to 1.0 moves mean DockQ by ~0.011. Both
  earlier lr 1e-3 arms' entire validation gain came from the same 3 interfaces (`9qrv_AB`,
  `29tj_BG`, `9bx7_BC`), carrying 85-88% of the total with identical deltas — the same pose
  selected. Any validation claim has to survive that decomposition.
- 12 of the 92 validation interfaces have no acceptable pose at all, and 13 are winnable (an
  acceptable pose exists and the shipped head misses it). The reachable gain is bounded by those
  13.
- The two sets are built by different procedures with different filters. A train-versus-val
  difference is not by itself evidence about a method.

## The code that produced each set

### Training set

| step | file | what it does |
|---|---|---|
| cluster definition | `phases/phase_a_define_train_set.py` | `training_clusters()` is **the definition of the training set**: the SAbDab2 splits minus clusters holding a benchmark complex, minus those homologous to the 39 unlisted ones. `POLY = ("PROTEIN", "PEPTIDE")` at line 119, applied at line 287 and 566 |
| structure list | `phases/phase_j0_select_structures.py` | freezes the structure list for the FT3 cache; step 0 excludes every cluster in the live validation split |
| Protenix format | `phases/phase_j_prepare_training_data.py` | bioassembly pickles plus `training/prep/job_indices.csv`, one row per sampleable chain/interface |
| trunk + label cache | `phases/phase_j_build_cache.py` | `LOSS_FEATS` (lines 68-69) is what the label npz stores as `lf_<name>`; `ALIGN_FEATS` is what the trunk npz holds for the head |
| folding | `phases/phase_j_queue_now.py`, `phases/phase_j_labels_and_audit.sh` | pose generation and label build |

Output: `training/fold/` — `trunks/<pdb>/free_5000NN.npz`, `labels/<pdb>/free_5000NN.npz`,
`results.csv` (per pose: `iface, seed, arm, donor_pose, tmpl_chains, sample, dockq, iptm`),
`chain_roles.csv`.

### Validation set

| step | file | what it does |
|---|---|---|
| first split | `phases/phase_b_define_val_clusters.py` | the original 50-cluster draw, stratified by median deposition date |
| live split | `phases/phase_b2_propose_val_split.py` | **the authority on the current 40-cluster split.** R1 token cap (default 1000), R2 withholding cap, R3 composition quota, R4 redundancy flag |
| second reps | `phases/phase_b3_pick_second_reps.py` | prefers a second representative with a different antigen, then the smallest by token count |
| job jsons | `phases/phase_g_build_val_jobs.py` | one Protenix job json per validation structure; lines 154-165 are what is actually folded, and therefore what the token count must be priced on |
| folding | `phases/phase_i_fold_validation.sh`, `phases/phase_i_inputs.py`, `phases/phase_i_check.py` | pose generation and verification |

Output: `validation/fold/` — trunks only; **there are 53 trunks and no labels**, which is why
`phase_k_train --frame-direction` cannot score the validation set (it needs
`lf_frame_atom_index` and `lf_pae_rep_atom_mask`).

### Scoring, shared by both

`phases/phase_k_eval.py` is the only scorer. It reads exactly one field, `chain_pair_iptm`,
selects `max(v, key=lambda t: (t[0], -t[2], -t[3]))` and reports that pose's DockQ. Its module
docstring carries the validation baseline, and `phases/phase_k_eval_sharded.py` is the multi-card
driver. The baseline json is `training/ft3_pilot_baseline.json`.

## Structure size: measured, and it explains about 15% of the gap

Validation caps representatives at 1,000 tokens (R1 in `phase_b2_propose_val_split.py`, because
`z_trunk` is `[N, N, 256]` and cache cost grows as N^2). The training fold has no cap. Token
counts from `z_trunk.shape[0]`, one unit per structure:

| set | n | median | mean | p90 | max | frac > 1000 |
|---|---|---|---|---|---|---|
| validation | 53 | 481 | 500 | 699 | 799 | 0% |
| training (120 sampled) | 120 | 673 | 715 | 1,138 | 2,289 | 13% |

Size does track difficulty inside the training fold — oracle DockQ over all 1,011 interfaces:

| tokens | n | oracle DockQ | oracle frac acc |
|---|---|---|---|
| < 400 | 74 | 0.5849 | 0.7703 |
| 400-600 | 288 | 0.5904 | 0.7674 |
| 600-800 | 327 | 0.5885 | 0.7339 |
| 800-1000 | 138 | 0.4584 | 0.6014 |
| > 1000 | 184 | 0.4188 | 0.5924 |

`pearson(token count, oracle DockQ) = -0.225`, with a clear step down above 800 tokens.

But restricting the training fold to validation's size regime barely closes the gap: training
interfaces at <= 1,000 tokens are n=827, oracle 0.5671 / 0.7267, against validation's 0.7171 /
0.8696. **Of the 0.177 oracle gap, size accounts for about 0.027.**

## What has been ruled out

Measured, not argued:

| candidate | verdict |
|---|---|
| structure size | explains ~0.027 of the 0.177 oracle gap |
| polypeptide-antigen filter | **applies to both sets.** `phase_a_define_train_set.py:119,287` applies the same `POLY = ("PROTEIN", "PEPTIDE")` filter inside `training_clusters()` |
| deposition era | **matched.** All 500 training structures have `release_date >= 2021-09-30` (`training/prep/job_indices.csv`); validation is post-cutoff by construction |
| mixed vs all-new clusters | **not set-specific.** "Mixed" is a cluster property; 107 of the 533 pool clusters are mixed, validation took 16 and the other 91 are in training |

The remaining differences between the two procedures are visible in the code tables above. The
measurement that would settle which of them matters is a single side-by-side attribute comparison
— token count, antigen type, antigen length, cluster size and deposition date for every structure
in both folds — rather than testing one rule at a time from the validation side. No GPU needed.

## Provenance of every number here

- Validation baseline: `training/ft3_pilot_baseline.json`, written by `phases/phase_k_eval.py`
  (shipped head, 100 seeds, 92 interfaces). `docs/pose_provenance.md` records that the evaluator
  reads 0.5775 where `results.csv` reads 0.5779, because `results.csv` rounds ipTM to 4 dp.
- Training fold shipped head: the two-row training-set comparison, measured over 975 interfaces.
  The pool is 1,011 at seeds 500000-500009; `9dzj`'s seeds 500000/500001 still need an OST
  re-score, which is why counts of 975, 999 and 1,011 all appear in the record.
- Training fold oracle, token counts and size bands: computed 2026-10-19 from
  `training/fold/results.csv` and `z_trunk.shape[0]` per structure.
- `overfit30`: `training/overfit30/ceiling/ceiling.json`.
