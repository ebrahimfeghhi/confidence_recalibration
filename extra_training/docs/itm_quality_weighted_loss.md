# Quality-weighted asymmetric regression of mean-row ipTM onto DockQ

**Status:** proposed, not yet run. Reference implementation in `phases/itm_quality_loss.py`,
gate in `run_logging/test_itm_quality_loss.py` (29 assertions, all passing).

This document is for review. Every design choice below is traced to a measurement in this repo;
where a choice is a judgement call rather than a measurement, it says so.

> **THE TARGET CHANGED FROM THE FIRST VERSION: it is now DockQ itself, not the true iTM.** The
> first version's global minimum selected the wrong pose on 9.6% of mixed interfaces and was
> capped at 65.9% of oracle headroom; targeting DockQ makes the zero-loss optimum the oracle
> (0/415 wrong, 100% of headroom). The reasoning that led to the iTM target was a false
> constraint, explained under *Why DockQ and not the true iTM*.
>
> **One measured defect still stands:** the direction that steals the argmax is diluted — one bad
> pose taking the top costs 4.7% of what calling the good pose bad costs. See
> *What this does not do*.

---

## The loss

For one scored chain pair of one structure, over its P candidate poses:

```
s_p = mean over anchor rows of t_i(predicted PAE logits)       # the model's score
y_p = dockq_p                                                  # the target, detached
w_p = sigmoid((dockq_p - 0.23) / 0.05)                         # how acceptable the pose is

up   = sum_p   w_p   * relu(y_p - s_p)^2  /  max(sum_p   w_p,   1)
down = sum_p (1-w_p) * relu(s_p - y_p)^2  /  max(sum_p (1-w_p), 1)
L    = up + down
```

In words: **push acceptable poses up toward their DockQ, push unacceptable poses down toward
theirs, and ignore the other two directions** — those cannot change an argmax.

The target and the weight come from the same quantity. That is deliberate, and it is what removes
the entire class of defects the first version had.

## The transformation from PAE to DockQ

There isn't a new one. `s_p` is `pae_row_weight.row_t` exactly as it already exists: softmax the
PAE logits, weight each bin by `1/(1+(c_b/d0)²)`, average over the pair's cross-chain targets,
then average over anchor rows.

**No new parameters.** The model changes `s_p` only by changing its predicted errors, and it has
full range to do so: errors ≪ `d0` send `s_p` → 1, errors ≫ `d0` send it → 0.

**Why not a learned head on top of the PAE features:** selection would no longer read
`chain_pair_iptm`, so the entire eval path changes — and that is ABAG-Rank, measured at +0.0006
in-sample and −0.0218 held out. Reusing the formula leaves inference untouched.

**One burden this imposes**, stated because it is not obvious: `d0 = 1.24·(max(N,19)−15)^(1/3) −
1.8` depends on the interface's token count and DockQ does not, so the same predicted errors give
different `s_p` on a large and a small complex. The model must compensate for `d0` to hit the same
DockQ on both. Learnable, but it is work the iTM target did not require.

**And it forces the pooling decision.** `s_p` here is the MEAN over anchor rows, while shipped
ipTM and every eval in this repo select by the MAX. Training the mean while selecting by the max
trains a quantity nobody reads, so mean pooling has to land with this — and the existing
checkpoints should be rescored under mean pooling *first*, so the two changes can be told apart.
`itm_ceiling.py` puts the pooling alone at ~7 points of headroom.

## Why DockQ and not the true iTM

The first version regressed onto `g_p`, the iTM computed from the *true* error bins, on the
reasoning that `s_p` is a bounded structured function of the logits and so could not be asked for
arbitrary values.

**That reasoning was wrong.** `s_p = mean of 1/(1+(err/d0)²)` and the model controls `err`, so
`s_p` spans (0,1) for *any* structure. Nothing constrained it to a sub-range.

The false constraint cost exactly what it bought, and it was measured:

| | zero-loss optimum picks a bad pose | mean DockQ | % headroom |
|---|---|---|---|
| target `g_p` (true iTM) | **40/415** mixed groups | 0.6492 | 66.1% |
| target DockQ | **0/415** | 0.7015 | **100%** |

Within-group Kendall tau(`g_p`, DockQ) is only **+0.48**. On `8ce4_BF` a DockQ-0.069 pose has
`g` = 0.1642 while the best acceptable pose (DockQ 0.816) has `g` = 0.1018 — so the old loss
trained the bad pose *up* and held the good one *down*, and since `s == g` gives `L = 0` it was
perfectly satisfied doing so.

**An intermediate design was rejected** and is recorded so it is not re-proposed: quantile-map
DockQ into iTM's range, keeping DockQ's ordering with "reachable" values. It also gave 0/415 and
100%, but it permutes values *between* poses, so two poses with identical DockQ get different
targets depending on where they fell in a global sort — not even a function of the pose. An
isotonic `φ(dockq)` would fix that, but both were solving the non-problem above.

**What the retarget costs**, unchanged: `s_p` stops being an error estimate. The model emits
whatever logits make the row-mean transform equal DockQ, so ipTM becomes a selection score and
nothing downstream may read it as a calibrated confidence.

---

## Why this and not the alternatives

### Why not PAE alone

This is what the current arm does, and the measured result at epoch 55 on 738 winnable
interfaces, budget 10:

| | mean DockQ | frac acc | vs shipped ipTM |
|---|---|---|---|
| shipped ipTM | 0.5477 | 0.7060 | — |
| row-weighted PAE | 0.5609 | 0.7304 | +0.0131 ± 0.0052 (t=2.51) |
| unweighted control | 0.5546 | 0.7317 | +0.0067 ± 0.0060 (t=1.12) |

Paired head-to-head the two arms differ by **+0.0063 ± 0.0055 (t=1.14)** — not separable. Held out
on 92 valfull interfaces at budget 100 both arms fell *below* the shipped baseline
(−0.0123 ± 0.0139 and −0.0140 ± 0.0149) and differed by +0.0017 ± 0.0137.

Training PAE fell from 2.9877 to 1.9016 over those 55 epochs. So a 0.9 drop in the objective
bought nothing that survives out of sample.

Three structural reasons, in increasing order of how much they matter:

1. **PAE is separable across poses.** It is a sum of per-pose terms, and ranking is a property
   *between* poses. No per-pose weighting can express it.
2. **PAE is indifferent to which poses it is accurate on.** Every pose contributes equally, so the
   average is the same whether the accuracy comes from the three contenders or the forty-seven
   hopeless poses. It drifts to whatever is easiest to fit, which is the hopeless ones.
3. **The loss is far from the readout.** PAE grades a full token×token matrix; selection reads one
   scalar summary of it. Accuracy in the matrix does not imply accuracy in the summary.

This loss fixes (3) by acting on the readout directly, and fixes (2) by the weighting and the
separate normalisation. It does **not** fix (1) — see *What this does not do*.

### Why not a ranking loss

Measured, in `docs/dntxmn_results.md`:

> The ranking arms were later shown to be lowering their loss by **MARGIN COLLAPSE** rather than
> by ordering — `x5frz6` drove within-seed std 0.00593 → 0.00147 and the top1–top2 gap
> 0.00205 → 0.00015 while its loss fell

The mechanism: a pairwise loss penalises confidently-wrong orderings, and the penalty grows with
the margin. Collapsing every score to nearly the same value makes every margin ≈ 0, so no pair is
confidently wrong and each pays the same middling cost. **Being indifferent is cheaper than being
wrong.** The loss falls and nothing was learned.

A softmax/listwise form has the same escape: a flat predicted distribution is a fixed, moderate
cost.

Regression toward `g_p` cannot be gamed this way, because `g_p` is an *absolute* target —
flattening the scores increases the loss. `test_collapse_is_penalised` and
`test_collapse_is_worse_than_truth_even_when_mostly_bad` pin this.

Prior ranking-only arms, for completeness: `xmz9ge` stopped at epoch 9 having put 89% of its loss
fall into epoch 1→2 then plateaued while selection swung ±0.07 DockQ/epoch; `4y3ve6` overfit from
epoch 3; `hronsr` managed +0.0116 paired. All on 20 structures.

### Why not gap-matching

`((s_p − s_q) − (g_p − g_q))²` is collapse-proof and was considered. But it is *algebraically*
regression with a free per-structure offset — minimising it over all pairs equals minimising
`(s_p − g_p − c)²` with `c` free. So it is strictly less supervision than regression for no gain,
unless there is a reason to let the absolute level float. There is not: `g_p` is known.

It also inherits the indifference problem, and worse: with 50 poses of which 3 are good, 1,081 of
the 1,225 pairs (88%) are bad-vs-bad. Weighting pairs by `|g_p − g_q|` makes this worse, not
better, because it emphasises far-apart pairs, which are the easy ones.

### Why the MEAN over rows and not the max

Measured by `run_logging/itm_ceiling.py` on the same 738 interfaces at 50 candidates each —
poses ranked by their **true** iTM:

| selector | mean DockQ | frac acc | % of oracle headroom |
|---|---|---|---|
| shipped ipTM (baseline) | 0.5479 | 0.7060 | — |
| true iTM, **max** over rows | 0.6385 | 0.8767 | **59.0%** |
| true iTM, **mean** over rows | 0.6492 | 0.8821 | **65.9%** |
| oracle (best DockQ) | 0.7015 | 0.9363 | 100% |

Mean minus max, paired: **+0.0107 ± 0.0039 (t=+2.70)**. So roughly 7 points of reachable headroom
are lost to the pooling choice alone, before any question of how well PAE is learned. That is more
than either trained arm has achieved.

The mean also removes a failure diagnosed directly. On `9gvn_BF` the row-weighted head compressed
its ipTM range to 0.0787 (shipped: 0.5966) and its correlation with DockQ went to **−0.306**
(shipped: +0.994), then selected the worst of 50 poses at DockQ 0.028 on an interface where 40 of
50 were acceptable. With a max, the whole pose's score rests on one row; in the saturated tail of
`1/(1+(c/d0)²)` the rows are nearly equal and which one wins becomes arbitrary.

**Two caveats on that table, both of which make 59–66% a floor rather than a ceiling:**
- It uses the static `chain_roles` join, not the per-pose OST join `phase_k_eval` uses. 298 of 368
  structures have more than two pooled chains, where a swap scores the wrong pair. That adds
  ranking noise.
- A point mass on the true bin is the best a 64-bin head can emit; scoring the continuous distance
  would give a slightly higher, unreachable ceiling.

**The headline implication:** a perfect PAE recovers 59–66% of the headroom, and the row-weighted
arm reached 8.5% of it. So ~86% of what is reachable is unclaimed by *optimisation*, not by
specification. PAE is the right target; we are far from using it.

---

## Each piece of the loss

### `relu`, splitting by direction

Only two mistakes can change which pose is selected:

- an acceptable pose scored **too low** → it loses the argmax
- an unacceptable pose scored **too high** → it steals the argmax

A bad pose scored even lower still loses; a good pose scored even higher still wins. The relus
keep only the two that matter.

**`relu` comes before the square.** Squaring first discards the sign and makes the split
meaningless.

### `w_p` and `(1 - w_p)`, the asymmetry

They sum to 1, so quality slides attention between the two directions rather than scaling the
loss. A clearly acceptable pose puts all its weight on "do not underrate me"; a clearly bad pose
puts all of it on "do not overrate me". At DockQ 0.23 the split is 50/50, which is plain
regression for that pose.

### The weight and the target are both DockQ

An earlier draft used `g_p` (true iTM) as the target and DockQ for the weight, arguing the two
jobs were different. They are different jobs, but pointing them at different *quantities* is what
let them disagree — see *Why DockQ and not the true iTM*. Both now come from DockQ.

One consequence worth naming: `w_p` is now a function of the target itself, so the two knobs are
no longer independent and the sigmoid width is the only thing controlling the asymmetry.

### The sigmoid width, 0.05

The width is a dial between separation and calibration: → 0 is a hard step at DockQ 0.23; large
sends every `w_p` toward 0.5, which is plain symmetric regression.

| DockQ | `w` at width 0.05 |
|---|---|
| 0.05 | 0.027 |
| 0.18 | 0.269 |
| **0.23** | **0.500** |
| 0.28 | 0.731 |
| 0.35 | 0.917 |
| 0.80 | 1.000 |

**0.05 is a judgement call, not a measurement.** It was chosen against the dilution table below.

### Separate normalisation, and the `max(mass, 1)` floor

If the loss were a mean over all P poses, an interface with one acceptable pose out of 50 would
give that pose **1/50 = 2%** of the gradient, so "call everything bad" would be nearly free. That
is the indifference problem again. Normalising each direction by its own weight mass raises that
pose's share to **43%** at width 0.05 — about 20× more.

The dilution that remains is the known limitation of a soft weight:

| width | `w` of a DockQ-0.05 pose | lone good pose's share of the up-term |
|---|---|---|
| 0.02 | 0.0001 | 99% |
| 0.05 | 0.027 | **43%** |
| 0.10 | 0.14 | 13% |

With 49:1 imbalance even a small per-pose weight adds up: 49 × 0.027 outweighs one pose at 1.0.
Width 0.02 fixes it but is effectively the hard threshold.

**The floor is the one non-obvious part.** Dividing a direction by its own mass makes it a
weighted *mean*, which is scale-free in `w` — so a direction whose total weight is tiny gets
amplified back to full strength and the asymmetry does nothing *between* the two directions. The
first gate run measured this: a lone pose at DockQ 0.80 has `1-w` = 1.1e-5, but `m_down` was also
1.1e-5, so **overrating it cost 0.16000, identical to underrating it** — the exact thing this loss
exists to prevent. Flooring each denominator at 1.0 fixes it without touching the
lone-acceptable-of-50 case, whose `m_up` = 2.30 is well above the floor.

**But 1.0 is not pinned, and the first draft's "forced by the gate" overstated it.** Every floor
in **[0.05, 10]** passes the whole gate — a 200× range. The gate forces that *some* floor above
~0.05 exists; 1.0 is a judgement call.

**And the floor is not inert.** It switches each term between a weighted-*sum* regime (mass < 1)
and a weighted-*mean* regime (mass > 1), so the per-group loss scale depends on group composition.
Over the 738 real groups at width 0.05: `sum(1-w)` is **below** the floor in **262 (35.5%)**,
where the down term is therefore a plain sum, while `sum(w)` is below it in only **11 (1.5%)**. So
the up:down balance is a per-group accident rather than a chosen constant. An explicit balance
coefficient would at least make it a decision — that is the real answer to open question 1.

---

## Implementation

The gradient is a per-pose scalar that depends on the other poses *only* through the two weight
masses:

```
dL/ds_p = -2 w_p relu(g_p - s_p) / max(sum(w), 1)
          +2 (1-w_p) relu(s_p - g_p) / max(sum(1-w), 1)
```

Both masses are computable with no graph. So this drops into the **existing** cross-seed path in
`phase_k_train.py`: `score_only=True` (pass A) already does a no-grad forward per pose collecting
scores, `_xs_coef` already holds `dL/ds` keyed `(pdb, seed, sample, pair)`, and backward already
adds `c_p · s_p` with one pose live at a time. `dL_ds` in the reference implementation is the
closed form, checked against autograd over 25 random groups (worst deviation 1.5e-08).

Cost: one extra no-grad forward per pose. From this run's own timers, forward is 4.4 min/epoch
against backward's 9.4, so about **+28% per epoch** — 15.5 min to roughly 20.

`g_p` is precomputed once by the same code path as `itm_ceiling.py` (CPU only, 9 workers, ~8 min
for all 36,900 poses) and cached keyed `(pdb, seed, sample, pair)`.

**Per chain pair, not per structure.** `d0` and the row set differ per interface, and 298 of 368
structures score two or more pairs. Pooling them ranks rows by a quantity no ipTM reads — this was
a review finding on the row-weight arm, where only 11 of the top 20 rows agreed between the two
groupings.

---

## What this does not do

### The down direction is still diluted, and it is the one that steals the argmax

The separate normalisation fixes the *up* term. A bad pose's coefficient is
`(1-w_p)/sum(1-w)` ≈ **1/P**, so one bad pose lifting itself above the good one is cheap and gets
cheaper as P grows:

| P | good pose coefficient | bad pose coefficient | ratio |
|---|---|---|---|
| 2 | 0.974 | 0.973 | **1.0** — asymmetry gone |
| 10 | 0.807 | 0.111 | 7.3 |
| 50 | 0.434 | 0.020 | 21.3 |

At the training P=50 with one acceptable pose: lifting a single bad pose to 0.901 costs **0.0131**
against **0.2779** for calling the good pose bad — **4.7%**. The cheapest way to lose the
selection costs 5% of what the gate calls expensive, and the argmax only needs one pose.

Nuance worth keeping: lifting *all 49* costs 0.6416, so the loss does defend against a wholesale
lift. It is specifically the single-pose steal that is nearly free.
`test_down_direction_dilution_is_pinned` pins the ratio so a change is noticed.

### alpha is not O(1)

Gradient-norm parity against a PAE cross-entropy on the same logits needs **alpha ≈ 3,000** at the
mean real `|dL/ds|`, ~12,200 at the median, ~260 at the p99. Two compounding causes: `s_p` is a
mean over `N_rows` of a bounded quantity, and the squared penalty shrinks the coefficient at the
realistic error scale (median `|s − g|` = 0.078).

The loss **value** is median **0.0106** against PAE's ~2.0, so the total-loss column in
`trajectory.csv` will not visibly move. `up`, `down` and both weight masses must be logged
separately or there is no way to tell the term is live — the same class of failure as the
row-weight bool cast, which logged itself as working while being an exact no-op.

### The rest

- **It does not fix separability.** It is still a per-pose loss. The claim is only that the
  quantity being fit *is* the readout, and that the weighting puts the pressure where selection is
  decided.
- **It does not distinguish two acceptable poses**, by design. An acceptable pose is unconstrained
  from above, so acceptable poses may all saturate near 1. Harmless for selection, but it means
  the trained score is **not a calibrated confidence for good poses**, and **mean DockQ among
  acceptable poses is unguided**. This loss targets frac acceptable. If mean DockQ matters more,
  `w_p` should be `dockq_p` itself rather than a sigmoid around the threshold.
- **It cannot speak to out-of-distribution calibration.** Like the row-weight arm, it trains on
  the 368 winnable structures; the 119 dropped ones are the only place where the correct answer is
  "I am wrong, be unconfident".
- **It inherits the row-pooling change.** Switching the readout to mean-over-rows is a second
  variable. The ceiling says it is worth ~7 points on its own, so the two should be separated:
  **score the existing checkpoints with mean pooling first** (one eval pass, no training) before
  attributing anything to the loss.

---

## Open questions for review

1. **The down-direction dilution.** The single-pose steal costs 4.7% of the up-direction's
   penalty. Fix by normalising the down term differently, or by an explicit up:down coefficient?
3. **`max(mass, 1)`.** The gate admits [0.05, 10] and the floor silently changes the down term's
   scale in 36% of real groups. An explicit balance coefficient with `P` in the denominator may be
   the cleaner design.
4. **Width 0.05 is unmeasured.** Worth a cheap sweep on `g_p` alone before training. Note widths
   ≤ 0.03 previously underflowed `1-w` to exactly zero in float32; fixed by computing the down
   weight with its own `sigmoid(-z)`.
5. **Should PAE stay on as a low-weight anchor?** The record says the ranking loss alone produced
   per-pose gradients "as good as mutually independent" (alignment 0.107 against a 1/√N baseline
   of 0.100) while PAE's are nearly parallel — PAE supplied the coherent direction. This loss is a
   regression and may not need that, but it is untested.
6. **alpha ≈ 3,000 is now measured** (open question 4 of the first draft). Confirm by gradient
   parity at init as with `alpha_rank`, and log the terms separately since the total will not move.

---

## Corrections to the first draft

Found by adversarial review, verified independently, fixed in place:

1. *"The two harmless mistakes contribute no gradient at all rather than a small one"* — **false**.
   The relu chooses which *weight* multiplies the error, not whether there is one. 0 of 36,900 real
   poses get a zero coefficient; the ratio is 3.4e-05, not zero.
2. *"The floor was forced by the gate"* — the gate admits any floor in [0.05, 10].
3. *"Flooring at 1.0 fixes both without touching the case the normalisation was added for"* — true
   for `m_up`, but it changes `m_dn`'s scale in 36% of real groups.
4. *"Separating the weight from the target costs nothing"* — it cost the tau +0.48 / 9.6%
   disagreement, which is why the target is now DockQ.
5. *"`s_p` is a bounded structured function and cannot be asked for arbitrary values"* — **false**,
   and it is the claim that produced the wrong target. The model controls the predicted errors, so
   `s_p` spans (0,1) for any structure.

Gate holes found and closed: `test_asymmetry_is_real` used P=1 groups, where **both masses hit the
floor**, so it never exercised the normalisation — 7 of 23 mutations passed the original gate,
including dropping the down relu and setting `m_dn := sum(w)`. And the headline 43% figure was
asserted about `quality_weight`, never through the loss. One assertion was named "the other 49
hold a MINORITY of it" while printing that they hold 57%, and its condition reduced to a
tautology.
