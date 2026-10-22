# Three-term pose-share arm: PAE + ranking + L2-SP

**Status: written, pre-flighted offline, NEVER RUN ON A GPU.** Launcher is
[`shell_scripts/launch_poseshare_rank_l2_overfit30.sh`](../shell_scripts/launch_poseshare_rank_l2_overfit30.sh).
Needs a two-epoch smoke test before it is trusted; `docs/next_restart.md` records that a NameError
at an epoch boundary once cost 77.9 minutes.

## Why

Selection is an **argmax of per-interface `chain_pair_iptm`** over the seed budget
([`phases/phase_k_eval.py:3`](../phases/phase_k_eval.py#L3)). The FT3 loss minimises **PAE**. Those
are different functionals of the same logits, and five arms showed they move independently —
held-out PAE improved 0.18–0.29 in every arm while the ordering went nowhere or backwards. See
[`docs/ranking_vs_pae.md`](ranking_vs_pae.md) for that measurement and the two corrections it
forced on the record.

So this arm trains the ordering directly, on top of the two terms already in play.

## Which arm to run first

**Run PAE + ranking, without L2-SP.** Adding both at once is not a one-variable comparison against
`8eiek8`: if it improves, you cannot separate ranking supervision from L2-SP stabilising the
mapping from an interaction between them. The doc previously called `8eiek8` a one-variable control
while describing a two-term change; review of `cd8e6ab7` was right to object.

That matters more than usual here, because L2-SP is 0-for-2 on this problem: `4x8t44` (λ=1e-2,
10,240 params) was a null with the penalty never above 2.0% of the loss, and `wpg274` (λ=1e-5,
19.8M params) did not prevent the ordering collapse — ρ fell to 0.040–0.055 by epoch 30–32, below
the unpenalised arm's 0.078 floor, penalty never above 1.3%.

| arm | PAE | ranking | L2-SP | launcher |
|---|---|---|---|---|
| `8eiek8` (done, 50 ep) | ✓ | | | the control |
| **run this** | ✓ | ✓ | | [`launch_poseshare_rank_overfit30.sh`](../shell_scripts/launch_poseshare_rank_overfit30.sh) |
| only if ranking shows score inflation or drift | ✓ | ✓ | ✓ | [`launch_poseshare_rank_l2_overfit30.sh`](../shell_scripts/launch_poseshare_rank_l2_overfit30.sh) |

Architecture is `--pose-perblock --pose-perblock-share` with
`--train-only linear_no_bias_d,linear_no_bias_d_wo_onehot` — 10,240 parameters, 0.052% of the head.
**`8eiek8` is the one-variable control**: identical in every respect but the two added terms.

## The ranking loss

[`phases/rank_loss.py:148`](../phases/rank_loss.py#L148) (`pair_terms`), with ipTM computed at
[`phases/rank_loss.py:123`](../phases/rank_loss.py#L123).

```
L  =  − Σ (s_i − s_j)(d_i − d_j)  /  Σ |d_i − d_j|
```

over the 10 pairs of a seed's 5 poses. `s` = the head's ipTM, `d` = DockQ (label only). Pairs with
equal DockQ are dropped. It is the normalised covariance between score gaps and quality gaps.

Per-pose gradient, verified against finite differences to 5.6e-17:

```
∂L/∂s_i  =  − n (d_i − mean d) / Σ |d_i − d_j|
```

Each pose is pushed by how far its DockQ sits from the group mean, and the coefficients sum to
zero — so **the loss is invariant to a uniform shift of the scores within a seed.**

That is a statement about score space, and an earlier version of this doc wrongly carried it into
parameter space ("the group's mean ipTM is never pushed"). It does not follow. The parameter
gradient is `∇_θ L = Σ cᵢ ∇_θ sᵢ`, and the five Jacobians `∇_θ sᵢ` are not equal, so a step can
move the group's mean score and can move other seeds' scores too. **Within-seed supervision can
therefore change cross-seed ordering through the shared weights** — in either direction, and
unsupervised. It has to be measured, not assumed. Caught in review of `cd8e6ab7`.

### Why linear and not softplus

The first version (pushed as `429dd891`) used `softplus(−(s_i − s_j)·sign(d_i − d_j))`, i.e.
RankNet. That was a units error. Measured on 5,400 real within-seed pairs:

| | within-seed \|ipTM gap\| |
|---|---|
| median | 0.0081 |
| 90th pct | 0.0339 |
| max of 5,400 | 0.3311 |

softplus only bends over arguments of order 1. At a margin of 0.02 its gradient magnitude is
**0.495 for a correct pair and 0.505 for an inverted one** — indistinguishable. Every property
softplus is chosen for was inactive, and the loss was a straight line through ln2 with all the
curvature outside the data. So it is written as the straight line it already was.

Two alternatives considered and rejected:

- **A multiplier σ.** σ≈30 would stop pushing at a gap of 0.1, just above the 90th percentile. It
  works, but it is a hyperparameter invented to compensate for a scale, and the linear form needs
  none.
- **Per-group standardisation** (divide margins by the group's own std). Scale-free and adaptive,
  but the implied multiplier ranges **12 to 1,262** across 540 seed groups, because the divisor is
  a standard deviation of five points. It amplifies hardest exactly where the head has no opinion.

### What linear gives up

Any stopping point: it pays to widen the gap forever. ipTM is structurally in [0,1], so nothing
diverges, but **the within-seed spread is the thing to watch** and it is now logged per epoch.

There is a second, subtler cost. Because `cᵢ` depends only on DockQ, a pose already ranked above
everything still gets pushed up, and a bottom pose still gets pushed down. So **the loss can fall
purely by widening gaps that were already correct, with the inverted fraction unchanged.** This is
a DockQ-weighted covariance surrogate for ranking, not a measure of ranking mistakes, and the two
must be read separately. If the loss improves substantially while the inversion rate stands still,
that is the signal to revisit a properly scaled logistic loss — the σ=1 measurement rules out
unscaled softplus, not a calibrated one.

### Why the DockQ gap is the weight

With a linear term, `w·sign(d_i − d_j)` collapses to `(d_i − d_j)`, so no sign function is needed.
It also fixes a real defect in the unweighted form, whose per-pose gradient is the pose's **rank**:
two poses at DockQ 0.136 and 0.124 are near-identical in quality, but the rank form pushes the
last-placed one twice as hard. The weighted form treats them alike.

## Why within a seed

A unit is one structure, one seed, `--nsample 5` poses — the only poses co-scored under identical
weights in one step. Measured variance decomposition over the 18 val10 interfaces, 150 poses each:

| | between seeds | within a seed |
|---|---|---|
| **ipTM** | **82.3%** | 17.7% |
| **DockQ** | 45.4% | **54.6%** |

ipTM is largely a per-seed quantity; DockQ is mostly pose-level. And **44.8% of within-seed pairs
are inverted** (coin flip 50%), essentially unchanged across every arm — 44.8% shipped, 45.0% full
head, 45.6% pose-share. So the head is near-random exactly where most of the real quality
variation lives.

**The limitation, stated plainly.** Selection is a global argmax over all 150 poses, and the loss
only forms pairs within a seed. Note what this does *not* establish: because the weights are
shared, within-seed updates can shift cross-seed scores as well, so the 17.7% figure is **not a
bound on the possible benefit** — it bounds what is directly supervised, not what can change.
Cross-seed *pairs* would need several
~200 MB trunks co-resident (trunks are stored per `(pdb, seed)`:
`training/fold/trunks/<pdb>/free_<seed>.npz`), and that is the same I/O amplification which is the
leading hypothesis for the `ggfhj5` epoch-12 stall. Not attempted here.

An earlier version of this doc claimed a measurable DockQ gain from fixing within-seed ordering.
That claim was withdrawn: it held cross-seed selection fixed while changing within-seed ordering,
but both come from the same weights, so it is a counterfactual that cannot occur.

## How the gradient is computed — one pass, no extra forward

[`phases/rank_loss.py`](../phases/rank_loss.py) `coeffs()`, applied at
[`phases/phase_k_train.py:2490`](../phases/phase_k_train.py#L2490).

Because the loss is **linear** in the scores, `L = Σ cᵢ sᵢ`, its derivative contains no score at
all — only DockQ:

```
c_i  =  − Σ_{j valid} (d_i − d_j)  /  Σ_{(i,j) valid} |d_i − d_j|
```

Verified: the coefficients are byte-identical for score vectors as different as
`[0.70, 0.69, 0.69, 0.69, 0.66]`, `[0.1, 0.9, 0.5, 0.2, 0.7]` and all-zeros.

So the coefficients are computed **before the model runs**, from the labels, and each pose's
existing forward/backward carries its own `cᵢ · sᵢ`. One graph at a time, exact gradient, **zero
extra forwards.** The loss value is reconstructed as `Σ cᵢ sᵢ` from the scores the forward already
produced — no second evaluation of anything.

### What this replaced, and why it was there

The version pushed as `429dd891` used softplus, whose derivative `sigmoid(−m)` *does* depend on
every score. That needed a no-grad forward over all five poses first, costing ~33% more GPU time
per pose, plus save/restore of the dropout RNG so the two passes evaluated the same function, plus
a per-pose scalar equality check (`--rank-scalar-tol`) to catch it when they didn't.

Switching the loss to linear made all of that dead, and it was not removed at the time — the
machinery was not re-derived when the function it served changed. Review of `cd8e6ab7` caught it.
The pass, the RNG handling and the tolerance flag are gone.

The memory constraint that motivated one-pose-at-a-time is unchanged and still real: one pose's
backward graph is ~45 GB on a 98 GB card. Note the 5.5 GiB pair-logit figure quoted in several code
comments assumes the 3,400-token crop **cap** — real crops are 409 tokens median, 656 max over
9,180 units, where that tensor is 0.10 GiB. The constraint is the full graph, not those tensors.

## ipTM is computed the evaluator's way

[`phases/rank_loss.py:123`](../phases/rank_loss.py#L123) mirrors
[`phases/phase_j_rescore_cache.py:130`](../phases/phase_j_rescore_cache.py#L130)
(`pair_iptm_only`) — same `sample_confidence.calculate_chain_based_ptm`, same
`get_bin_params(configs.loss.pae)`, same fp32 softmax — minus the `no_grad`. `phase_k_eval`'s
self-check validates that path against the fold's stored ipTM to 2e-3. Optimising a near-miss of
the selection statistic would be a different experiment.

## Labels

`training/fold/results.csv`, 51,350 labelled poses over 1,028 interfaces, keyed
`(pdb, seed, asym pair) -> {sample: DockQ}` at
[`phases/rank_loss.py:99`](../phases/rank_loss.py#L99). **Per interface, not per structure** — a
structure holds several, and reading one interface's ipTM against another's DockQ is the error
`phase_k_eval` refuses to run.

Pre-flight for this arm, run offline: all 30 trained structures have labels, all 10 seeds
(500000–500009) present, exactly 5 poses per `(pdb, seed, interface)` group. The trainer refuses to
start if any scored structure lacks labels
([`phases/phase_k_train.py:983`](../phases/phase_k_train.py#L983)) — otherwise those poses would pay
a forward and contribute no ranking gradient, making it a partial experiment reported as a whole one.

## L2-SP

λ = 0.05, reasoned from measurement but **not tuned**. Arm 2 (`4x8t44`) ran this exact parameter
set at λ=1e-2 and the drift equilibrated at ‖W−W₀‖² ≈ 4.5–4.8, pricing the penalty at 2.0% of the
loss — inert for all 50 epochs. λ=0.05 puts that same drift near 10%. It will come out lower, since
the penalty suppresses the drift it prices.

For context on L2-SP's track record here: `4x8t44` (λ=1e-2, 10,240 params) was a null, and
`wpg274` (λ=1e-5, 19.8M params) did **not** prevent the ordering collapse — ρ fell to 0.040–0.055
by epoch 30–32, below the unpenalised arm's 0.078 floor, with the penalty never exceeding 1.3% of
the objective. λ is in this arm because it is cheap to carry, not because it has earned a place.

## What is NOT settled

- **`--alpha-rank 1.0` is a guess.** The loss value is ~0.01–0.25 against a PAE loss of 2.3, but
  gradient magnitude on the weights is what matters and that is not inferable from loss values.
  Two scale constants picked earlier were wrong (`--l2-sp 1e-2` for 10,240 params, and softplus's
  implicit σ=1).
  My first plan — compare `max grad_norm` across two short runs — is not good enough, and review
  of `cd8e6ab7` said why: the combined norm can be *smaller* than the PAE-only norm if the two
  gradients oppose each other, and max norms depend heavily on individual batches. The right probe
  is, on a few **identical** batches at the **same** initial weights, measure `‖g_PAE‖`,
  `‖g_rank‖` and **`cos(g_PAE, g_rank)`** — which answers both how large the ranking update is and
  whether it cooperates with or fights the existing objective. Separately for `g_L2SP` if that arm
  is run.
- **Cross-seed ordering is not directly supervised**, and between-seed variation is 82.3% of
  ipTM's. But it is reachable: shared-parameter updates can move seed means, so the effect is
  unsupervised rather than absent, and `rho_trained` is what measures it.
- **Nothing has run on a GPU.** Offline checks passed: autograd vs finite differences 5.6e-17,
  coefficients identical across unrelated score vectors, gradients summing to zero, label coverage
  770/770, no unbound names. **The untested path is the gradient through
  `calculate_chain_based_ptm` into the two trainable matrices** — a label-coefficient test cannot
  establish it. The smoke test must confirm both matrices receive finite, non-zero ranking
  gradients and that an optimiser step completes.
- **How to read the outcome**, since the loss falling is not by itself evidence of better ranking:

  | observation | reading |
  |---|---|
  | loss falls, inversions fall | within-seed discrimination genuinely improved |
  | loss falls, inversions flat | margin expansion only — revisit a *scaled* logistic loss |
  | inversions fall, ρ flat | local gain, cross-seed effects limiting selection |
  | ρ **and** selected DockQ improve | the result that matters |

## Monitoring — four quantities, because they move independently

The loss supervises one of these and reports none of them, so all four are logged. Computed
post-hoc from the eval shards by [`run_logging/rank_rho.py`](../run_logging/rank_rho.py), cached
per run in `rank_rho_b30.csv`, no trainer involvement.

| column | what it is | shipped head |
|---|---|---|
| `rho_trained` | mean within-interface Spearman(ipTM, DockQ) over all ~150 poses — what selection needs | **0.1958** |
| `inv_trained` | **within-seed pairwise inversion rate** — the thing `--rank-loss` actually supervises | **0.4481** |
| `spread` | mean within-seed ipTM std — the quantity a linear loss can inflate without bound | — |
| DockQ / frac_acc | the downstream objective at the fixed budget | 0.2909 / 10 of 18 |

These really do dissociate, and already have. Across every PAE-only arm the within-seed inversion
rate has barely moved from the shipped 0.4481 — `8eiek8` went 0.4600 at epoch 1 to 0.4557 at 50,
`mdv9x7` sat at 0.4503 at epoch 30 — while the within-seed spread roughly doubled or tripled
(0.0059 → 0.0103 for `8eiek8`, 0.0165 for `mdv9x7`). **So nothing has ever trained the within-seed
ordering**, which is the gap this arm aims at. It also means a successful intervention could move
`inv_trained` without moving global `rho` promptly, or move `rho` through cross-seed score shifts
without touching `inv_trained` — hence both.

Panel 5 of [`run_logging/plot_full_train.py:150`](../run_logging/plot_full_train.py#L150) plots
`rho_trained`. Panels 3 and 4 are noise at n=18 — `mdv9x7` swung DockQ 0.2072 → 0.2018 → 0.2904
across three adjacent epochs — so ρ and the inversion rate are the readouts to steer by.

**One plotting caveat that is not yet fixed.** Panel 5 places the *shipped* head's ρ at epoch 0,
but the pose-share architecture is not bit-identical to the shipped head at initialisation — blocks
1–3 gain an injection they never had. So for these arms epoch 0 is a *shipped-head reference line*,
not the architecture's own starting value. Measuring the initialised pose-share head once before
training would separate the architectural change from the training effect; until then the first
segment of those curves conflates the two.

## Scope

This is a **development study**: 30 training structures, 18 validation interfaces. Hyperparameters
and checkpoints are being chosen on that same 18-interface set, so it cannot also serve as the
reported result. A paper-level claim needs a separate, larger held-out set scored once.
