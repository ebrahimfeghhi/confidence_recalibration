# Training ABAG-Rank on our training set — plan for review

Upstream: <https://github.com/tadteo/ABAG-Rank> (code MIT; checkpoints and examples CC BY-NC 4.0).
Cloned read-only to `/home/ebrahim/abag_rank_upstream/ABAG-Rank`, **outside this repo**, so it is
not a nested git checkout and nothing of theirs is committed here.

## Why

Every arm we have trained optimises Protenix's confidence head on a masked PAE regression and is
then judged on which single pose it picks. Measured today (`paper_results.md`):

- the best full-training-set arm is **+0.0024 mean DockQ** over shipped, ~2% of valfull's headroom,
  with `frac_acceptable` unmoved;
- the overfit30 chain reached **+0.1965 in-sample** and **-0.0355 on valfull** with the same
  checkpoint, so the objective converts into selection but does not cross complexes.

ABAG-Rank differs on the axis we have not tried: it regresses **DockQ directly** with a small Deep
Set over interface residue pairs, instead of regressing PAE with a 19.8M-parameter head. If it
transfers where our head does not, that is the result; if it fails the same way, the wall is the
data, not the objective.

## What it consumes

`config_ABAG_rank_no_esm.yaml`, `input_dim: 4`, per inter-chain residue pair within 12 A:

| feature | flag | our source |
|---|---|---|
| inter-chain PAE | `use_interchain_pae` | confidence head forward — **the only GPU cost** |
| inter-chain CA distance | `use_interchain_ca_distances` | `ci_coordinate` in the trunk npz |
| residue one-hot | `use_one_hot_residue` | fold metadata |
| ipTM / pTM | `use_iptm_ptm` | ipTM in `results.csv`; pTM from the same forward |
| ESM embeddings | `use_esm_embeddings: false` | not needed |

**Contact probability is NOT used by this model.** I said earlier that `ci_contact_probs` being
cached helped here; it does not — that feature belongs to their AF3 ingest, not to the no-ESM
feature set. Recorded because the error is easy to repeat.

Their loss is smooth-L1 on DockQ plus a ranking term (`one_minus_rho`) plus a distance-preservation
term, which is exactly the top-1-oriented objective our PAE arms lack.

## Their pipeline, and the one stage we replace

1. `src/data/01_process_af3_predictions.py` — AF3 JSON -> per-sample HDF5. **We replace this.**
2. `src/data/03_prepare_dataset.py` — adds per-complex normalisation stats. **Unchanged.**
3. `src/data/preprocess_shards.py` — packs shards. **Unchanged.**
4. `src/data/06_generate_manifest.py` — manifest.csv. **Unchanged, but fed our split.**
5. `train.py --config configs/config_ABAG_rank_no_esm.yaml`.

### Stage-1 schema we must emit, per `<complex>/<sample>`

Read out of `01_process_af3_predictions.py` (lines 1164-1228), not from the README:

    pae_matrix            float32 [T, T]
    token_chain_ids       utf-8 [T]
    token_res_ids         int32  [T]
    ptm, iptm, ranking_score   scalars
    inter_idx, inter_jdx  int32  [P]      every (i, j) with different chain ids
    interchain_pae_vals   float32 [P]
    ca_distance_matrix    float32 [T, T]
    residue_one_letter    utf-8 [T]
    residue_three_letter  utf-8 [T]
    inter_idx_dist, inter_jdx_dist  int32 [P]
    interchain_ca_distances float32 [P]
    abag_dockq, abag_lrmsd, abag_irmsd   scalars
    chain_mapping, pdb_id, seed_name, sample_name   strings

We have `abag_dockq` (results.csv) but **not** lrmsd/irmsd; those are written as NaN, which is what
their own code does when DockQ fails. Nothing in the no-ESM dataloader reads them.

## The GPU step

`phase_j_rescore_cache.pair_iptm_only()` already computes the full token-pair PAE (line 158) and
discards everything but `chain_pair_iptm`. So this is **not a new forward pass** -- it is returning
a tensor that is already in hand. Plan: add `--dump-pae <dir>` to `phase_k_eval.py`, writing one
npz per pose with the expected PAE matrix and pTM/ipTM.

Cost: 24,350 training poses + 13,800 valfull poses = 38,150, which is one train-pool eval pass
(~35 min on 3 cards) plus one valfull pass (~10 min). Call it **~1 GPU-hour**, plus write time.

**Storage: RESOLVED, store sparse.** Measured on 10fj (T=625, 3 chains): 259,600 of 390,625 token
pairs are inter-chain (66.5%), but only **1,894 fall within the 12 A cutoff the loader applies** --
0.48% of T^2. So:

| | per pose | 38,150 poses |
|---|---|---|
| dense PAE matrix | 1.5 MB | ~60 GB |
| inter-chain pairs at 12 A | ~8 KB | **~0.3 GB** |

200x smaller, and it is exactly the set the model consumes.

**But a PER-POSE cutoff breaks their stage 3, and this is the detail that would have cost a GPU
hour.** `03_prepare_dataset.py` does `np.stack(pae_list, axis=0)` over a complex's samples to get
`pae_col_mean/median/std`, which requires every pose of a complex to have the SAME pair count.
Per-pose 12 A sets do not: measured on 10fj they range 1,784 to 2,018 across its 50 poses. Their
own stage 1 avoids this by writing ALL inter-chain pairs and letting the dataloader apply the
cutoff -- but that is 259,600 pairs per pose for 10fj, ~2 MB/pose for PAE+CA, **~76 GB** over
38,150 poses, which is worse than dense.

**Resolution: a FIXED pair set per complex, the union over its poses.** Measured on 10fj: the union
of 12 A inter-chain pairs across all 50 poses is **4,096** -- only 2.2x the per-pose mean, and 1.05%
of T^2.

| scheme | per pose (PAE+CA) | 38,150 poses | stage 3 stacks? |
|---|---|---|---|
| dense `pae_matrix` | 1.5 MB | ~60 GB | yes |
| all inter-chain pairs (their stage 1) | ~2 MB | ~76 GB | yes |
| per-pose 12 A cutoff | ~15 KB | ~0.6 GB | **NO** |
| **union over the complex's poses** | **~33 KB** | **~1.3 GB** | **yes** |

So the exporter runs in two passes: a CPU pass that fixes each complex's pair set from the
coordinates, then the GPU pass that dumps PAE on that fixed set. `inter_idx`/`inter_jdx` are
written once per complex rather than per pose. The dataloader's own per-sample `valid_pair_mask`
still filters by actual distance afterwards, so the model sees exactly what it would have.

## Split

Their manifest carries a `split` column. We use **our** split, not a new one: the 487 structures of
`structures_no9dzj.csv` for training, valfull's 92 interfaces held out. Anything else makes the
result incomparable to every number in `paper_results.md`.

## Environment

Theirs is conda (`environment.yml`, env `abag`). We will build a **separate** venv rather than
touch `/home/ebrahim/Protenix/.venv`, which every training and eval path in this repo depends on.

## Order of work

1. `--dump-pae` in `phase_k_eval.py`, with the sparse-vs-dense decision made first. *(no GPU)*
2. `run_logging/abag_export.py`: trunk npz + poses + results.csv -> their stage-1 HDF5. *(no GPU)*
3. The dump pass over train + valfull. *(~1 GPU-hour, queued behind ocqtyt)*
4. Their stages 3-4, then train.

## Open questions for review

- `samples_per_complex: 10` against our 50 poses per training structure -- subsample per epoch, or
  raise it?
- Is `ranking_score` (AF3's 0.8*ipTM + 0.2*pTM) worth reproducing, or set to ipTM and noted?
