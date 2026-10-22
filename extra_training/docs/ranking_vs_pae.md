# We were training PAE and selecting by ipTM

## The measurement

Selection is an argmax: for each interface, take the pose with the highest per-interface
`chain_pair_iptm` over the seed budget (`phases/phase_k_eval.py:3`), and DockQ is whatever that one
pose scores. So everything rests on how well ipTM orders the poses of an interface.

Mean within-interface Spearman(ipTM, DockQ) over the 18 val10 interfaces, budget 30, computed from
each run's own `train_eval_shards_ep*/rows_b30_*.csv` -- every row carries the trained head's
`iptm`, the fold's `stored_iptm` and `dockq` side by side, so no new compute was needed:

| head | rho(ipTM, DockQ) |
|---|---|
| shipped | **0.196** |
| pose-share `8eiek8` ep1 | 0.222 |
| pose-share `8eiek8` ep50 | **0.271** (monotone, still rising) |
| pose-share + L2-SP `4x8t44` ep20 | 0.261 (then flat for 25 epochs) |
| LoRA `eewbr5` ep13 | 0.285 |
| LoRA `eewbr5` ep30 | 0.199 |
| full head `mdv9x7` ep1 | 0.267 |
| full head `mdv9x7` ep30 | **0.078** |

The statistic we select by correlates 0.196 with pose quality. That single number is the whole
0.2909-vs-0.5968 oracle gap: an argmax over ~150 poses ranked almost at random.

## What this corrects

Two claims that were in the record and are wrong.

**"frac_acc never exceeded the untrained 10 of 18 in any epoch of any arm."** The pose-share arm
reached 11 of 18 at epochs 10 and 18 (DockQ 0.3246 and 0.3110) and matched 10 of 18 in fourteen
more. Distribution over its 50 scored epochs: 8/18 x8, 9/18 x26, 10/18 x14, 11/18 x2.

**"The pose-share arm was the worst of the four."** True on held-out PAE (best 2.7690 against the
full head's 2.6881) and backwards on the ordering, where it is the only arm that improved
monotonically and the only one above the shipped baseline at every epoch.

The arms that rearranged the ordering most are the ones that wrecked it -- rho to the shipped
ordering fell to 0.22 for the full head and 0.32 for LoRA, and both ended at or below untrained.
Pose-share held at ~0.75, nudging rather than rewriting, and that nudge was in the right direction.

## Why DockQ could not show this

18 interfaces. frac_acc moves in steps of 1/18 = 0.0556, and mean DockQ is an average of 18 argmax
outcomes over a near-random ranker. Measured: `mdv9x7` swung DockQ 0.2072 -> 0.2018 -> 0.2904 across
three ADJACENT epochs. A rho gain of 0.075 is invisible in that. Panel 5 of
`run_logging/plot_full_train.py` now plots rho per epoch, cached per run in `rank_rho_b30.csv`; its
epoch-0 point and dashed baseline are derived from each run's own `stored_iptm` column rather than
hardcoded, and all six run directories agree at 0.195841.

## The external prior

Fromm, Ludaic and Elofsson, *Evaluating deep learning based structure prediction methods on
antibody-antigen complexes*, Bioinformatics 2026, compare ipTM against pDockQ2 for ranking
antibody-antigen predictions and find ipTM slightly better. pDockQ2 is also PAE-derived, so the
obvious move -- rank by a PAE-derived score, since PAE is what we train -- is the one that paper
argues against. We already use the better of the two confidence rankers.

## What follows

`--rank-loss` (`phases/rank_loss.py`) trains the ordering directly: for two poses of one interface,
the one with the higher DockQ should get the higher ipTM. See that module for the loss, the
two-pass gradient factorisation that keeps it inside the one-pose-at-a-time memory budget, and the
scalar self-check.

It is NOT the only hypothesis. A ranking loss fixes the objective; it cannot create information.
The head sees pose only as `cdist` over one representative atom per residue and never a backbone
frame, so it carries no orientation. If the deficit is orientation, a better loss plateaus higher
and stops. `--frame-direction` (pathway C) is the experiment that separates those two, and it is
still unrun.

**Nothing in the ranking path has run on a GPU.** `shell_scripts/launch_poseshare_rank_overfit30.sh`
is written and pre-flighted offline (labels cover all 30 trained structures, all 10 seeds, exactly
5 poses per group) but unlaunched. Smoke-test two epochs before trusting it.
