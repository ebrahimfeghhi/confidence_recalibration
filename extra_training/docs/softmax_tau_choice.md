# Choosing the softmax temperature for `--rank-form expected`

Reproduce every number here with:

```
cd /home/ebrahim/confidence_recalibration/extra_training
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/measure_softmax_tau.py
```

## The loss, and what tau actually controls

```
L        = - sum_i softmax(ipTM_i / tau) * DockQ_i      over every pose of one interface
dL/ds_i  = -(p_i / tau) * (DockQ_i - E[DockQ])          p = softmax(ipTM / tau)
```

**The direction of the push does not depend on tau.** Every pose whose DockQ is above the group's
expected DockQ is pushed up, every pose below it is pushed down, at any temperature. What tau
changes is the magnitude `p_i` — and `p_i` is large exactly where ipTM is already large. So a cold
tau concentrates gradient on the poses the model already ranks top and starves the poses whose
ipTM is low, which are precisely the ones that must be **lifted** for selection to improve.

## Data

Scores are the **shipped head's**, which is the correct starting point: all candidate arms begin
from the shipped weights, so this is the gradient allocation at step 0. The `iptm` column of
`training/fold/results.csv` and `validation/fold/results.csv` was verified **exactly** equal to the
`stored_iptm` the evaluator writes — 2,700 overlapping poses, max |diff| 0.0 — so no eval shards
are needed and the full sets are usable.

Poses are pooled across **all seeds** of an interface, because deployment picks argmax over all of
them regardless of which seed produced each one.

| set | interfaces | poses/interface (median) | poses |
|---|---|---|---|
| training | 1,027 | 50 | 51,300 |
| validation | 92 | 500 | 46,000 |
| combined | 1,119 | 50 | 97,300 |

Interfaces with fewer than 2 poses or a constant DockQ are dropped — they have no gradient at any
temperature and would dilute every column with zeros.

## Columns

- **E[DockQ]** — mean over interfaces of `-L`: the loss's own view of selection quality.
- **argmax** — mean DockQ of the highest-ipTM pose. **The real metric.**
- **gap** — `E[DockQ] - argmax`. Numerical fidelity of the surrogate to the metric. **Not the
  criterion** (see the warning below).
- **eff#** — median `1 / sum(p^2)`, the effective number of poses receiving gradient, against
  poses/interface. Small means a handful of poses drive the step; approaching poses/interface
  means the softmax is nearly uniform.
- **top5%** — share of total `|gradient|` on the five poses carrying most of it.
- **underrated%** — **share of `|gradient|` reaching poses with DockQ above and ipTM below the
  interface median.** These are the poses that must be lifted. **Choose on this column.**
- **oracle%** / **oracle_rank** — share of `|gradient|` on the single highest-DockQ pose, and that
  pose's median rank when poses are ordered by `|gradient|`. Rank 1 means the loss pushes hardest
  on the best available pose.

## Training set — 1,027 interfaces, 50 poses each (the group size the arms actually train on)

| tau | E[DockQ] | argmax | gap | eff# (of 50) | top5% | **lift%** | median-cut err | oracle% | oracle_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.4069 | 0.4078 | −0.0009 | 4.9 | 80.4% | **2.9%** | 0.0% | 5.79% | 16 |
| 0.01 | 0.4062 | 0.4078 | −0.0017 | 6.9 | 70.3% | **4.6%** | 0.1% | 6.38% | 11 |
| 0.02 | 0.4049 | 0.4078 | −0.0029 | 12.5 | 58.3% | **7.9%** | 0.2% | 6.98% | 7 |
| 0.03 | 0.4034 | 0.4078 | −0.0044 | 17.5 | 51.0% | **10.5%** | 0.3% | 7.47% | 5 |
| 0.05 | 0.4002 | 0.4078 | −0.0076 | 26.5 | 43.3% | **14.2%** | 0.5% | 8.20% | 3 |
| **0.10** | 0.3921 | 0.4078 | −0.0157 | 39.3 | 35.4% | **18.2%** | 0.9% | 8.76% | **2** |
| 0.20 | 0.3779 | 0.4078 | −0.0300 | 46.4 | 32.0% | **20.8%** | 1.2% | 8.95% | 1 |
| 0.50 | 0.3586 | 0.4078 | −0.0492 | 49.4 | 31.2% | **22.4%** | 1.6% | 9.11% | 1 |

## Validation set — 92 interfaces, 500 poses each (30 seeds, NOT the training group size)

| tau | E[DockQ] | argmax | gap | eff# (of 500) | top5% | **lift%** | median-cut err | oracle% | oracle_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.5765 | 0.5779 | −0.0014 | 42.7 | 41.7% | **3.9%** | 1.34% | 86 |
| 0.01 | 0.5776 | 0.5779 | −0.0003 | 113.0 | 30.3% | **6.2%** | 1.65% | 54 |
| 0.02 | 0.5758 | 0.5779 | −0.0021 | 222.3 | 19.8% | **9.0%** | 0.2% | 1.29% | 28 |
| 0.03 | 0.5732 | 0.5779 | −0.0047 | 281.7 | 15.9% | **10.9%** | 1.23% | 20 |
| 0.05 | 0.5696 | 0.5779 | −0.0083 | 331.8 | 12.7% | **13.0%** | 0.4% | 1.26% | 12 |
| 0.10 | 0.5619 | 0.5779 | −0.0160 | 422.1 | 9.1% | **16.1%** | 0.7% | 1.34% | 10 |
| 0.20 | 0.5461 | 0.5779 | −0.0318 | 472.9 | 7.5% | **18.6%** | 0.9% | 1.36% | 10 |
| 0.50 | 0.5226 | 0.5779 | −0.0553 | 493.6 | 7.2% | **20.3%** | 1.28% | 6 |

## Both combined — 1,119 interfaces

| tau | E[DockQ] | argmax | gap | eff# | top5% | **lift%** | median-cut err | oracle% | oracle_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.4208 | 0.4218 | −0.0010 | 5.0 | 77.2% | **3.0%** | 5.42% | 17 |
| 0.01 | 0.4203 | 0.4218 | −0.0016 | 7.7 | 67.0% | **4.8%** | 5.99% | 13 |
| 0.02 | 0.4190 | 0.4218 | −0.0028 | 13.6 | 55.1% | **8.0%** | 0.2% | 6.51% | 8 |
| 0.03 | 0.4174 | 0.4218 | −0.0045 | 19.5 | 48.1% | **10.7%** | 6.95% | 6 |
| 0.05 | 0.4142 | 0.4218 | −0.0077 | 29.3 | 40.8% | **14.1%** | 0.5% | 7.63% | 4 |
| 0.10 | 0.4061 | 0.4218 | −0.0157 | 41.1 | 33.3% | **18.1%** | 0.9% | 8.15% | 2 |
| 0.20 | 0.3917 | 0.4218 | −0.0301 | 47.2 | 30.0% | **20.6%** | 1.2% | 8.33% | 2 |
| 0.50 | 0.3721 | 0.4218 | −0.0497 | 49.5 | 29.2% | **23.0%** | 8.47% | 1 |

## The threshold correction (review of 9798aa1c)

The `lift%` column originally cut on **median DockQ**, but the loss raises a pose's ipTM only
where `DockQ_i > E[DockQ]` -- that is where `dL/ds_i` is negative. The median is not E[DockQ], and
E[DockQ] **moves with tau** while the median does not: on `overfit30` the median is 0.0383 at every
temperature while E[DockQ] runs 0.0505 at tau 0.02 up to 0.0773 at tau 0.20. So the old cut
counted some poses the loss was pushing *down* as if they were being lifted, and the error grew
with tau -- biased in favour of the warm temperatures this doc recommends.

Corrected, with the miscounted mass kept visible as `median-cut err`. The error is small and the
ordering is unchanged: on the training set `lift%` goes 7.9 / 14.2 / 18.2 / 20.8 across tau 0.02 /
0.05 / 0.10 / 0.20 against 7.9 / 14.5 / 18.8 / 21.5 before, with 0.2-1.2 points miscounted. On
`overfit30`, 30.0% at tau 0.10 becomes **28.7%** with 1.5 points miscounted. The script now
computes the corrected column, so the numbers in every table above are the corrected ones.

## Recommendation: tau = 0.10 as a first choice, 0.05 as the comparison -- NOT a declared winner

Read off the training table, which is the one that governs — training groups are 50 poses, and tau
interacts with group size (the validation set's 500-pose groups shift every column, which is why
the two tables disagree on the same tau and why the val table must not be used to pick it).

- **tau = 0.10** puts **18.8%** of the gradient on the underrated poses and makes the single best
  pose of an interface the **2nd**-largest recipient of gradient out of 50. That is the behaviour
  wanted: the poses that need lifting are the ones being pushed.
- **tau = 0.05** gives 14.5% and rank 3, with a more selective `p` (eff# 26.5 of 50 against 39.3).
  Worth running as the contrast.
- **tau = 0.02 is lower priority.** Only 7.9% of the gradient lifts low-ipTM poses and the best
  pose ranks 7th. It is cold enough to spend most of its gradient on poses already on top, which
  is the defect the whole form was meant to fix.

**This is not evidence enough to declare a winner.** Every number here is the gradient allocation
at step 0 from the shipped weights -- it says where the loss will push first, not what 50 epochs
of pushing produces. The allocation argument favours 0.10, the correction above shifted it
slightly, and nothing in it rules out 0.05 being better in practice. Both have to be trained and
judged on selected DockQ and fraction above 0.23.

### The 30 structures the arms actually train on say the same thing, more loudly

```
... run_logging/measure_softmax_tau.py --restrict-to training/overfit30/structures.csv \
    --taus 0.02,0.05,0.10
```

78 interfaces, 50 poses each, 3,900 poses:

| tau | E[DockQ] | argmax | gap | eff# (of 50) | top5% | **lift%** | median-cut err | oracle% | oracle_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.0505 | 0.0417 | +0.0088 | 7.1 | 63.8% | **11.2%** | 0.3% | 5.32% | 9 |
| 0.05 | 0.0629 | 0.0417 | +0.0213 | 16.4 | 47.7% | **23.5%** | 0.8% | 10.80% | 2 |
| **0.10** | 0.0725 | 0.0417 | +0.0308 | 30.2 | 41.5% | **28.7%** | 1.5% | 13.44% | **1** |

At tau 0.10, **30% of the gradient** reaches the underrated poses and the best pose of an
interface is the **largest** single recipient. The ordering of the recommendation is unchanged and
the margin is wider.

Note the sign flip in `gap`: it is **positive** here because `overfit30` is a hard subset where the
shipped head's argmax scores only **0.0417** mean DockQ, so even a soft average over poses beats
what selection currently achieves. That is a statement about how much room there is on this
training set, not about tau — but it does mean `gap` carries no information at all here, which is
a second reason not to choose on it.

### Why not go warmer than 0.10, even though underrated% keeps rising

It flattens — 18.8% to 21.5% from tau 0.10 to 0.20 — and `eff#` goes to 46.4 of 50, i.e. `p`
becomes nearly uniform. In that limit `p_i -> 1/n` and the gradient tends to
`-(1/(n*tau)) * (DockQ_i - mean DockQ)`, which no longer depends on ipTM at all. The loss stops
being about selection and becomes a plain "raise ipTM in proportion to DockQ deviation" — which is
close to the **linear** form that failed in `ntwjfp` with spread running away without limit. The
`eff#` column approaching poses/interface is the signature of that degeneracy, and it is the reason
to stop at 0.10 rather than chase the column upward.

## Warning: the `gap` column is a trap, and it caught me

Choosing tau to minimise `|gap|` — making `E[DockQ]` numerically match the argmax DockQ — selects
the **coldest** temperature, because a cold softmax is a better numerical stand-in for argmax. On
the training set `gap` is −0.0009 at tau 0.005 and −0.0157 at tau 0.10. I recommended tau 0.02 on
exactly this basis and it was wrong: matching the number rewards a loss that only looks at poses
already on top, which is the opposite of what improves selection. `underrated%` moves the other
way and is the column that reflects the goal.

## Retrospective check on the same metric

`rank_rho` now records `exloss_*`, the held-out expected-DockQ objective at each run's own tau
(added in `defcece1` after review). Recomputed over `h7msge`, the pairwise softplus arm: its
pairwise held-out loss improved 47.3% (0.07412 to 0.03905) while **E[DockQ] fell from 0.2611 to
0.2202**, a 15.7% drop, ending **0.0370 below the shipped head** on that metric. The pairwise arm
improved the objective it was given and degraded the one that matters, which is the case for
changing the loss.
