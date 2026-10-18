# A ranking loss for the confidence head

Status: proposal, nothing implemented. Written 2026-10-18 for review.

## The problem this is trying to solve

We re-run Protenix's FT3 stage (confidence head only) on our own pose distribution,
hoping to improve DockQ on antibody-antigen interfaces by selecting a better pose out
of a fixed budget. Three learning rates have now been tried (5e-3, 1e-3, 5e-4). The
best validation result of any arm is mean DockQ 0.5891 against a shipped-head baseline
of 0.5775 — about a 2% relative gain, which then decays with further training.

The decisive measurement was evaluating the epoch-6 head of run `425fct` (lr 1e-3)
on its **own training data**: the same 975 interfaces, the same 10 seeds x 5 samples,
the same poses it had already seen six times.

| | mean DockQ | frac >= 0.23 |
|---|---|---|
| stock head | 0.4202 | 0.5344 (521/975) |
| epoch-6 head | 0.4217 | 0.5303 (517/975) |
| difference | +0.0014 | -0.0041 |

Flat on memorised data. Meanwhile the training loss did move:

| ep | total | pLDDT | PAE | PDE | grad |
|---|---|---|---|---|---|
| 1 | 8.0038 | 3.4057 | 2.6753 | 1.8979 | 3.20 |
| 2 | 7.8280 | 3.3080 | 2.6391 | 1.8599 | 2.89 |
| 3 | 7.7915 | 3.3009 | 2.6267 | 1.8429 | 2.83 |
| 4 | 7.7657 | 3.2871 | 2.6192 | 1.8391 | 2.50 |
| 5 | 7.7069 | 3.2508 | 2.6090 | 1.8272 | 2.37 |
| 6 | 7.6965 | 3.2387 | 2.6118 | 1.8262 | 2.38 |

Total down 3.8%, PAE down 2.4%, and PAE rose between epochs 5 and 6. `alpha_pae` and
`alpha_confidence` are both 1.0 in this run, so PAE is fully in the gradient — the flat
curve is real, not a weighting artefact.

**The loss moved and the ranking did not.** That is the signature of an objective that
is pointed at the wrong quantity, not of an optimisation failure. FT3's loss is
per-pose absolute regression on PAE/pLDDT/PDE/resolved. The metric is a within-interface
argmax. Every pose's calibration can improve without a single pair of poses swapping
order. Three learning rates are three points on an axis that cannot reach the target.

## What the metric actually selects on

`phase_k_eval.aggregate` reads exactly one field, `chain_pair_iptm`, and selects

```python
max(v, key=lambda t: (t[0], -t[2], -t[3]))   # top pair ipTM, ties by lowest seed then sample
```

then reports that row's DockQ. ipTM derives entirely from PAE.

## Is the score differentiable?

Yes, and it is worth being precise because there are two argmaxes here.

**Inside the score.** `chain_pair_iptm` is: PAE logits -> softmax -> weighted sum over
bins (the TM-score term) -> mean over the chain pair's token block -> `.max(dim=-1)`
over the alignment token (`sample_confidence.py:871`). Every step is differentiable;
the final `max` backprops like max-pooling, gradient to the winning element only,
non-smooth on a measure-zero set.

**The pose selection.** "Pick the highest-ipTM of 50 poses, report its DockQ" is a hard
argmax into a lookup table. There is no gradient and we never differentiate it. The
ranking loss replaces it with a smooth surrogate in the scores.

## The loss

For one interface, with poses `i` scored `s_i` (predicted pair ipTM) and labelled with
true `DockQ_i`:

```
L_rank = (1/Z) * sum over pairs with DockQ_i > DockQ_j of
             w_ij * -log sigmoid( (s_i - s_j) / T )

w_ij = |DockQ_i - DockQ_j|
Z    = sum of w_ij over the same pairs
```

### Worked example

Four poses, `T = 0.1`:

| pose | true DockQ | score `s` |
|---|---|---|
| A | 0.82 | 0.61 |
| B | 0.45 | **0.74** |
| C | 0.11 | 0.52 |
| D | 0.05 | 0.33 |

The model ranks B first, so we select B and score 0.45 where A would have given 0.82.
The ordering is wrong in exactly one place.

| pair | `w` | `s_i - s_j` | `-log sigmoid(./T)` | weighted |
|---|---|---|---|---|
| A>B | 0.37 | **-0.13** | 1.55 | **0.58** |
| A>C | 0.71 | +0.09 | 0.34 | 0.24 |
| A>D | 0.77 | +0.28 | 0.06 | 0.05 |
| B>C | 0.34 | +0.22 | 0.11 | 0.04 |
| B>D | 0.40 | +0.41 | 0.02 | 0.01 |
| C>D | 0.06 | +0.19 | 0.14 | 0.01 |

The single violated pair carries 63% of the total.

### Why the gradient behaves well

From one pair, the gradient on `s_i` is `-w_ij * (1 - sigmoid(d/T)) / T`. For the
violated A>B pair that is about **-2.9**: a hard shove up on A, an equal shove down on
B. For B>D, already correct by a wide margin, **-0.07**. The loss spends almost all its
gradient on pairs that are wrong or nearly wrong and leaves settled pairs alone — which
is exactly what per-pose regression fails to do.

### The two knobs

- **`T`** sets what counts as "comfortably ahead", in ipTM units. Within-interface ipTM
  spreads are small, so `T` must match that scale. Measure the within-interface standard
  deviation of pair ipTM on training data and start there. Too large and nothing
  saturates; too small and every pair reads as violated.
- **`w_ij = |dDockQ|`** makes the loss care about swaps that cost real DockQ. Without it,
  fixing a 0.42-vs-0.41 inversion counts as much as fixing 0.82-vs-0.45.

Normalising by `Z` per interface stops an interface with a wide DockQ spread from
drowning out fifty others.

### Edge cases

- Pairs with equal DockQ: skipped.
- Interfaces where every pose is bad (12 of 92 in validation have no acceptable pose at
  all): kept, but every `w_ij` is tiny so they contribute almost nothing. That is the
  right behaviour — nothing we do there can help.

## Variant: threshold-straddling pairs only

Restrict the pair set to pairs where exactly one of the two poses is `>= 0.23`.

This targets `frac_acc` — the headline metric — directly, instead of mean DockQ. Each
such pair is a case where getting the order right flips an interface from unacceptable
to acceptable.

Trade-offs to think about:

- **Much less signal per interface.** Most pairs are dropped. On an interface where all
  50 poses are acceptable, or none are, the set is empty and the interface contributes
  nothing at all.
- **Possibly better aligned.** Only 13 of 92 validation interfaces are winnable (an
  acceptable pose exists, the shipped head does not find it) and the head currently wins
  2. Those 13 are precisely the interfaces with straddling pairs, so this variant puts
  all its weight on them.
- **Weighting question.** With a threshold restriction, `|dDockQ|` may be the wrong
  weight — a 0.24-vs-0.22 pair flips acceptability as surely as 0.9-vs-0.1 does, but
  gets ~35x less weight. Consider uniform weights for this variant, or
  `w = 1` plus a small `|dDockQ|` term.
- **Worth running both** and comparing on both metrics, since they optimise different
  things and we report both.

## Keeping the FT3 terms

```
L = L_FT3 + lambda * L_rank
```

A pure ranking loss is invariant to any monotone rescaling of `s`, so on its own it
would be free to destroy absolute calibration — and pLDDT/PDE/resolved share the head.
FT3 becomes the anchor. Pick `lambda` so the two terms' gradient norms are comparable;
measurable by logging each separately for a few steps.

## Engineering: the memory problem and the two-pass trick

The loss couples all of an interface's poses, but we cannot hold 50 autograd graphs.

Fortunately `dL/ds_i` depends only on the scalar scores, so:

1. forward all 50 poses under `no_grad`, collect `s`
2. compute `L_rank` and `dL/ds_i` analytically (50 numbers, negligible cost)
3. re-forward each pose with grad and backward seeded with that scalar

Peak memory stays at one pose. Forward cost roughly doubles: ~0.83 -> ~1.4 min/step.

### The data layout already supports this

`shard_by_structure` in `phase_k_train.py` splits across ranks by **whole structure**,
and `--shuffle structure` keeps a structure's units consecutive. So all 10 seeds x 5
samples of an interface land on the same rank, in a row. No pipeline rework is needed —
only accumulation of a structure's scores before the ranking term is formed.

Note a structure can host more than one interface (e.g. `9dzj` has `C-E` and `D-E`).
Group by (structure, chain pair), since ipTM is per pair.

## Cheap pre-check before implementing anything

~20 minutes, no training. Using the frozen shipped head's PAE output, ask whether **any**
reweighting of the chain pair's token block reorders poses better than the current ipTM
formula does. Fit a simple linear reweighting directly against DockQ order on training
data.

If a hand-optimised reweighting cannot beat the shipped ranking, the information is not
present in the PAE output and no loss function will find it. That would redirect the
work towards unfreezing more of the model, or towards a different input representation,
rather than towards a better objective.

## Where this might disappoint

- **ipTM is a rigid function of PAE.** The head may not have the freedom to reorder poses
  without fighting the FT3 anchor. That tension is the experiment.
- **The ceiling is bounded.** Headroom is 0.1272 on training and 0.0857-0.1392 on
  validation. Of 92 validation interfaces, 13 are winnable and 12 have no acceptable pose
  at all. A perfect reranker gains a bounded amount.
- **The previous gains were concentrated and possibly noise.** Both 1e-3 arms' validation
  gains came from the same 3 interfaces (`9qrv_AB`, `29tj_BG`, `9bx7_BC`) carrying 85-88%
  of the total, with identical deltas — meaning the same pose was selected. A 92-interface
  validation set moves 0.0109 in mean DockQ when one interface flips from 0 to 1.0.
  Any result from this work needs to survive that scrutiny.

## Open questions for review

1. Pairwise vs listwise (softmax over the interface's poses, target `softmax(DockQ/T')`).
   Pairwise is easier to debug and degrades more gracefully when no pose is good.
2. Should the ranking term see all 50 poses, or only the top-k by predicted score?
   Top-k focuses on the decision that actually gets made, but biases the gradient.
3. Is `T` better fixed, or set per interface from that interface's score spread?
4. Does the threshold variant want uniform weights (see above)?
