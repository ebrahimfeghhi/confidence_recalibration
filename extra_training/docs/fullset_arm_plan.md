# Full-set arm: protocol, and how to read it

Written before the arm runs, so the decision rules are not chosen after seeing the numbers.

## Goal

Improve top-1 selection on **val**, then test once. The test set stays untouched until exactly one
checkpoint has been chosen on val.

## The fixed training slice

`training/slice280/items.csv` -- 280 items, 178 structures, built once by
`run_logging/build_train_slice.py --seed 0` and never rebuilt. Every in-sample number comes from
this file.

* Only items whose structure has **15 loadable seeds** (75 candidates). An item with one loadable
  seed offers 5 candidates and its bin means something else.
* Thin bins taken whole: **214 survive** the 15-seed filter -- 97 at 1/5, 44 at 2/5, 44 at 3/5,
  29 at 4/5. That is just under the 215 the specification expected, so the slice is filled to 280
  with easy items in the 25/20/20 proportion: 25 with one all-good seed, 20 with two or three,
  21 with four or more.
* Bins are stored in the file. They are a property of the generator's output and are never
  recomputed per scoring run, or an item could change bin because the selector changed.

## The arm

* **Shipped init** (protenix-v2 + a fresh confidence head), not a resume of any earlier arm.
* **The whole training set**, pre- and post-cutoff. The manifest holds 853 structures and the pool
  1,711 interfaces, but the launcher's skip list removes 7 and 32 pool rows name structures the
  manifest does not, so what the arm ACTUALLY trains is **846 structures, 1,679 interfaces, 7,230
  units (456 x 15 seeds + 390 x 1), 74,685 candidates**. `training/fullset/structures.csv` has its
  seed spans read off the trunks on disk.
* Same loss, same interface-only mask, same per-item class balance (0.5/n_good, 0.5/n_bad, mean of
  the two for a straddler, renormalised per structure).
* **Items with no good candidates, or no bad ones**: every candidate falls in one class and takes
  the same 0.5/n, which the per-structure normaliser turns into exactly 1.0. The undefined case is
  therefore unweighted, which is the intended behaviour and needed no new code.
* **Constant lr 1e-3** after the 20-step warmup. No staircase, no decay window.
* **Pair dropout 0.25** in training, off in every scoring pass.

## Scoring

* The trainer has ONE held-out trigger, `(ep+1) % EVAL_EVERY == 0`, so the two cadences are two
  stages of the same arm rather than one run: **STAGE=open** runs 2 epochs at `EVAL_EVERY=2`,
  producing the epoch-2 val point the lr check needs; **STAGE=main** resumes from its epoch-2
  bundle and runs to 60 at `EVAL_EVERY=10`.
* Val and the slice fire on the SAME epochs -- both triggers are `(ep+1) % every == 0`, val first,
  slice second, off the same saved head -- so the side-by-side table compares one checkpoint.
* Dropout is off: `phase_k_eval` builds its own model and does not accept `--pair-dropout`.
* Val: 141 items, 30 seeds, 150 candidates. Slice: the 280 fixed items, 15 seeds.
* Each pass emits our score and the shipped score for the same candidate, so the shipped reference
  row costs no extra GPU and is measured on identical candidates.
* The in-training passes write their own shards under
  `training/runs/<rid>/{valfull83,slice280}_eval_shards_ep<N>/`; the report reads THOSE. Re-scoring
  a checkpoint with `shell_scripts/score_checkpoint.sh` costs a full pass per side and is for
  checkpoints the run never scored.
* The reporter refuses to print a table over a subset: it asserts 280 slice items at 75 candidates
  and 141 val items at 150, so a pass that lost a shard fails loudly instead of reporting a rate
  over two thirds of the set.
* Reported per point by `run_logging/score_report.py`: overall top-1 and mean quality of the pick,
  the same by bin, and item-level flips against shipped (gained, lost).
* Training loss stays split by good and bad candidates every epoch, as now.

## Choosing the checkpoint

**Best val mean quality of the pick**, with flips against shipped as the tiebreak (prefer fewer
losses). One checkpoint only. The full one-hour pass runs on that checkpoint and nothing else.

## How to read it

| observation | reading | action |
| --- | --- | --- |
| epoch-2 val easy bin well below shipped's 96% | the learning rate is too high | stop, lower it, restart |
| slice improves in the thin bins, val does not | generalisation gap; data is the limit | stop training, carve a better held-out set |
| slice does not improve in the thin bins | the loss is the limit, not the data | revisit the objective |
| val beats shipped | the thing we were after | stop, go to test |

## Known gaps, written down rather than discovered later

* **Bin labels only line up in the thin regime.** The slice splits 5/5 three ways by all-good-seed
  count; val has no such split, and val additionally has a 0/5 row that the slice cannot have
  (items with no good candidate are excluded from the slice by construction). The reporter emits an
  aggregate `5/5` row on both sides so the easy bin is comparable.
* **Val bins are recomputed per pass.** Only the slice has stored bins. Val's come from the pass's
  own rows; they key on quality rather than on our score, so they cannot move with the selector,
  but they would move if rows went missing.
* **Release dates are documentation only.** `run_logging/fetch_release_dates.py` records that 317
  of 853 training structures are pre-cutoff, but nothing in this arm gates, strata-fies or reports
  on that. The pre/post framing is not measurable from what the arm emits.

## What this arm cannot answer

Val has **21 hard items** and its thin bins are 13/5/1/2. A change of one item is ~8 points in the
1-of-5 bin. So val can resolve the easy bin and the overall rate; it cannot resolve the thin bins,
which is the regime the loss is aimed at. The slice can -- in-sample. That asymmetry is the whole
reason for reading the two side by side.
