# ABAG-Rank on our data — results

Run `training/abag/run_full`, started 2026-10-27 5:36 PM Eastern. Regenerate the plot with:

    python -I abag_rank/plot_valfull.py training/abag/run_full/train.log figures/abag_rank_valfull.png

## Setup

| | |
|---|---|
| train | all **975** interfaces over 487 structures, 50 poses each (budget 10) |
| validation | **valfull**, 92 interfaces over 53 structures, 150 poses each (budget 30) |
| held out | FoldBench and PXM-2025, untouched |
| model | upstream `DeepSet`, phi [512,256,128,256], rho [2048,1024,512,256,128], `attn_pool_concat_stats`, 5,180,545 parameters |
| features | 47 per pair: cantor_norm, dist_norm, pae_col_mean, pae_centered, ca_distance, two 21-wide residue one-hots |
| sample-level | **ipTM only** — `results.csv` carries no pTM and recovering it needs another forward pass over 38,150 poses |
| losses | smooth-L1 on the DockQ logit + soft Spearman (tau 0.05, probability space) + distance preservation (lambda 0.1), relevance-weighted by clamp(std/0.07) |
| hyperparameters | upstream's, except batch 8 (their own config comments cap concat_stats at 6) |

Speed: ~1.6 s/epoch including a valfull evaluation every 5 epochs, after caching valfull's
features once (1.20 GB). Before that cache a single evaluation cost 35 s, rebuilding 13,800
identical pose feature sets from disk each time.

## Result so far (212 epochs)

| epoch | selected DockQ | shipped | delta | frac acceptable |
|---|---|---|---|---|
| 1 | 0.5317 | 0.5831 | -0.0514 | 0.6630 |
| 5 | 0.5440 | 0.5831 | -0.0392 | 0.6957 |
| **10** | **0.5825** | 0.5831 | **-0.0006** | 0.7391 |
| 50 | 0.5613 | 0.5831 | -0.0218 | 0.7065 |
| 100 | — | — | — | — |
| 180 | 0.5608 | 0.5831 | -0.0223 | 0.7065 |

shipped 0.5831, oracle 0.6926, frac acceptable for shipped 68/92 = 0.7391.

**The reranker is not beating shipped.** It rose from well below to level at epoch 10
(-0.0006), then settled at **-0.02 to -0.04** and has stayed there while the training loss kept
falling, 2.14 -> ~1.09. The best epoch remains epoch 10 and has not been approached in 170
epochs, so that peak reads as noise rather than a trend.

**It is losing to a signal it was given.** ipTM is an input feature and verified to change the
model's output, yet the model selects worse than ipTM alone does. That is a sharper negative than
if it had never seen ipTM.

## How this relates to the confidence-head arms

Same 92 interfaces, same budget, same shipped baseline (`../paper_results.md`):

| | selected DockQ vs shipped |
|---|---|
| best confidence-head arm (ip8ga4, level over 20 epochs) | **+0.0024** |
| ABAG-Rank, best epoch | -0.0006 |
| ABAG-Rank, settled | -0.02 to -0.04 |

So a direct-DockQ objective with a small Deep Set does not do better here than a PAE objective
with a 19.8M-parameter head. Both sit within ~0.01 of shipped while their own training objectives
improve substantially. That the two approaches fail the same way, with different supervision,
capacity and architecture, points at the data rather than at either design.

## Not yet done

- The run is still going; these are early numbers and the 10,000-epoch horizon is far off.
- pTM is absent, so this is not exactly upstream's no-ESM baseline.
- Budget 30 of the 100 available seeds, chosen because every figure in `../paper_results.md` is
  at budget 30.

## The decisive test: ABAG-Rank does not beat ipTM on its own training data

A held-out run was launched to ask whether the failure above is cross-complex transfer: 70% of the
training structures to fit, the other 30% to evaluate, split by structure, `run_7030.sh`, seed 0,
membership recorded in `split.json`. The split does put near-neighbours on both sides -- 104 of the
146 held-out structures share a SAbDab2 `ab_ag_cluster` with a training structure, and 74 share an
`agclusters` antigen cluster -- so this is not the cluster-distance regime valfull sits in.

It generalises badly. It also never worked in-sample, which is the finding that matters.

| model | scored on | ipTM top-1 | model top-1 | delta |
|---|---|---|---|---|
| run_7030, epoch 220 | held-out 30%, 299 interfaces | 0.4252 | 0.4034 | **-0.0218** |
| run_7030, epoch 220 | its own fitted 70%, 676 interfaces | 0.4126 | 0.4133 | **+0.0006** |
| run_full, epoch 505 | its own fitted 975 interfaces | 0.4117 | 0.4020 | **-0.0097** |

So there is no in-sample performance for the held-out set to fail to inherit. The generalisation
gap is 0.022 sitting on top of an in-sample delta of zero. Calling this a generalisation failure
would be wrong: the method does not fit the selection objective at all.

Three things sharpen it:

- `sample_level_dim=1` feeds ipTM to the model as an input feature. It could match ipTM by passing
  that one number through, and instead scores below its own input.
- Held-out soft Spearman is +0.77. The model orders poses well and still loses top-1, which is the
  rho-is-not-the-metric result of `../paper_results.md` section 5 in a much starker form than the
  confidence-head arms showed it (rho 0.2301 -> ~0.255 for +0.0024).
- Train 1.0394 vs val 1.1357 at epoch 220 is almost no gap. A 5.2M-parameter model on 676
  interfaces for 220 epochs is UNDERfitting, not memorising.

Method, so the comparison cannot be waved off:

- Every baseline is paired on the SAME candidates. The run's `val top-1` is not over all 50 poses
  of an interface: `dataset.__getitem__` draws K=10 from a stateful Generator and the DataLoader
  re-forks workers each epoch from a parent whose rng never advances, which is why the logged
  oracle is pinned at 0.4933 every epoch. `val_baseline.py` therefore iterates the SAME loader and
  reads ipTM and DockQ out of its batches rather than reconstructing the draw, and asserts it
  reproduces that oracle to four decimals.
- The model is scored with all K poses in ONE forward, because `attn_pool_concat_stats` normalises
  across the K axis and scoring poses singly zeroes every pooled feature.
- Cross-check: the computed held-out 0.4034 equals the logged epoch-220 `val top-1 DockQ` exactly.
- Caveat: the interfaces are fully in-sample, but the specific 10-pose subsets scored here are not
  guaranteed to be the ones the training loop saw, since its train loader shuffles and worker draws
  vary. The claim is interface-level.
- Budget 10 (training structures were folded at 10 seeds x 5 samples), so these are NOT comparable
  to the budget-30 valfull figures elsewhere. Read them against their own oracle, 0.4847-0.4933.
