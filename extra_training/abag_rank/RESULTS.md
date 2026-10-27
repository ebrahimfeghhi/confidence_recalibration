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
