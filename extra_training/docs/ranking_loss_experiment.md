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

## The three terms

| term | flag | what it does |
|---|---|---|
| PAE | `--alpha-pae 1.0` | the shipped FT3 objective, unchanged |
| ranking | `--rank-loss --alpha-rank 1.0` | orders the 5 poses of a seed by DockQ |
| L2-SP | `--l2-sp 0.05` | prices distance from the shipped head |

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

Each pose is pushed by how far its DockQ sits from the group mean. The gradients sum to zero, so
the group's **mean ipTM is never pushed** — only the spread.

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

Any stopping point: it pays to widen the gap forever. Two things bound the damage — ipTM is
structurally in [0,1], and the gradients sum to zero so the group mean cannot drift. **The spread
is the thing to watch**, and if it inflates, σ and softplus come back.

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

**The limitation, stated plainly.** Selection is a global argmax over all 150 poses. This loss only
orders within a seed, which is 17.7% of ipTM's variance. Cross-seed pairs would need several
~200 MB trunks co-resident (trunks are stored per `(pdb, seed)`:
`training/fold/trunks/<pdb>/free_<seed>.npz`), and that is the same I/O amplification which is the
leading hypothesis for the `ggfhj5` epoch-12 stall. Not attempted here.

An earlier version of this doc claimed a measurable DockQ gain from fixing within-seed ordering.
That claim was withdrawn: it held cross-seed selection fixed while changing within-seed ordering,
but both come from the same weights, so it is a counterfactual that cannot occur.

## How the gradient is computed

[`phases/phase_k_train.py:2490`](../phases/phase_k_train.py#L2490) (pass A),
[`:2481`](../phases/phase_k_train.py#L2481) (the added backward term),
[`:2368`](../phases/phase_k_train.py#L2368) (the scalar self-check).

One pose's backward graph is ~45 GB on a 98 GB card, so five cannot be held at once — and the
trainer deliberately runs forward *and* backward one pose at a time. But the gradient factors:

```
∂L/∂θ  =  Σ_i (∂L/∂s_i)(∂s_i/∂θ)
```

`∂L/∂s_i` is one number per pose, computable from the five ipTM **values** alone. So:

1. **Pass A** — forward all 5 poses under `no_grad`, collect the scores. No graph.
2. Autograd on a 5-element leaf gives the five coefficients exactly.
3. **Pass B** — the existing per-pose forward; each backward carries its own `c_i · s_i`.

Exact gradient, one graph at a time, one extra forward per pose (~33% more epoch time).

**The RNG state is captured before pass A and restored before pass B**, because the head has
dropout: the two passes must evaluate the same function or `c_i` belongs to a different one. Each
scalar is then re-checked against `--rank-scalar-tol` (default 2e-3, matching the evaluator's
`--iptm-tol`). A silent disagreement would be a wrong gradient that still trains and still looks
plausible, so it is a `SystemExit`, not a warning.

Note the 5.5 GiB pair-logit figure quoted in several code comments assumes the 3,400-token crop
**cap**. Real crops are 409 tokens median, 656 max over 9,180 units, where a pair-logit tensor is
0.10 GiB. The memory constraint is the full backward graph, not those tensors.

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
  Two scale constants picked today were wrong (`--l2-sp 1e-2` for 10,240 params, and softplus's
  implicit σ=1). The honest way to set it: two epochs at `--alpha-rank 0`, two at `1.0`, compare
  `max grad_norm`. That doubles as the GPU smoke test.
- **The within-seed ipTM spread is not yet logged.** It is the quantity to watch with a linear
  loss. Computable post-hoc from the eval shards the same way ρ is
  ([`run_logging/rank_rho.py:80`](../run_logging/rank_rho.py#L80)), so it should be a second column
  there rather than a fifth trainer patch.
- **Cross-seed ordering is untouched**, and it is 82.3% of ipTM's variance.
- **Nothing has run on a GPU.** Offline checks passed: autograd vs finite differences 5.6e-17,
  gradients sum to zero, label coverage complete, no unbound names. The untested path is
  `calculate_chain_based_ptm` with a graph attached, and the pass-A/pass-B scalar agreement under
  dropout.

## Monitoring

Panel 5 of [`run_logging/plot_full_train.py:150`](../run_logging/plot_full_train.py#L150) plots mean
within-interface Spearman(ipTM, DockQ) per epoch, cached per run in `rank_rho_b30.csv`. Reference
values at budget 30, 18 interfaces: shipped ρ **0.1958**, baseline DockQ 0.2909 / 10 of 18, oracle
0.5968 / 18 of 18. Panels 3 and 4 are noise at n=18 — `mdv9x7` swung DockQ 0.2072 → 0.2018 → 0.2904
across three adjacent epochs — so ρ is the readout to steer by.
