# The straddle arm: ranking loss restricted to threshold-crossing pairs

Full training fold (487 structures) -> valfull (92 held-out interfaces).
Launched 2026-10-24, ~11:57 PM EDT.

## The idea

The reported criterion is the **fraction of interfaces whose selected pose clears DockQ 0.23**.
That number moves only when selection crosses 0.23. A pair of two acceptable poses, or a pair
of two failures, can be reordered in any way at all and the fraction does not change.

So restrict the ranking loss to pairs that **straddle** the threshold: exactly one pose above
0.23, one below. `--rank-straddle 0.23`.

Over the full training fold, that keeps **143,680 of 1,056,035 pairs (13.6%)**. The other 86.4%
of the loss's weight was being spent on orderings the criterion cannot see.

## What it costs, stated rather than hidden

**Mean DockQ is not invariant to the dropped pairs.** Promoting 0.61 over 0.38 raises mean DockQ
and leaves the fraction untouched -- and that gradient is now discarded. This arm optimises the
fraction and treats mean DockQ as a free variable. The two can move in opposite directions, so
both are reported and the fraction is named in advance as the primary number.

**589 of 1028 training interfaces go silent.** Only 438 are usable; the rest have every pose on
one side of 0.23 -- all acceptable, or all failures -- so they form no straddling pair and
contribute nothing to the loss. This is the single biggest risk in the arm and the first thing
to check if it fails.

## Tau

Tau is measured over the pairs this arm actually trains, not over all pairs:

    python -I run_logging/measure_tau.py --pairs all --straddle 0.23

| pair set | pairs | usable ifaces | median ipTM gap |
|---|---|---|---|
| all cross-seed, no straddle | 1,056,035 | 1027 | 0.0305 (DEFAULT_TAU) |
| cross-seed, straddling | 138,006 | 438 | 0.1585 |
| **all families, straddling** | **143,680** | **438** | **0.1429  <- this arm** |

Tau is 4.7x DEFAULT_TAU and that is not an error. Poses on opposite sides of 0.23 are genuinely
different structures and the shipped head already separates them more; training them at 0.0305
would put nearly all of them deep in saturation.

## Configuration

| | |
|---|---|
| pairs | `--rank-pairs all` -- cross-seed **and** within-seed, within_weight 1 |
| straddle | `--rank-straddle 0.23` |
| tau | `--rank-tau 0.1429` |
| form | softplus |
| PAE | `--alpha-pae 0` -- off |
| orientation | **dropped**, no `--frame-direction` |
| trainable | `linear_no_bias_d,pose_perblock` = 8 tensors / 40,960 params |
| L2-SP | on **every** trainable tensor, no `--l2-sp-exclude`; lambda derived by `probe_straddle.sh` |
| epochs | 15, ten-epoch minimum before any trend claim |

The straddle restriction is orthogonal to the family split: a within-seed pair that crosses 0.23
is exactly as selection-relevant as a cross-seed one, so both families are included.

There is no `--l2-sp-exclude` because the only tensors `zg5fnw` excluded were `frame_dir`, and
`frame_dir` is not trained here. The anchored set is 40,960 parameters -- the same count
`zg5fnw` anchored -- so `l2sp_lambda.py`'s size factor is exactly 1 and lambda transfers on the
measured gradient ratio alone, from `zg5fnw`'s 1e-05 at |g_rank| 2.4260e-03.

**Training and held-out loss are the same function.** `rank_rho.run_rank_objective()` reads
`rank_straddle` out of the run's own `args.json` and passes it to the held-out `group_loss`, so
the held-out curve is the twin of the training curve and the gap between them is generalisation.
The cache is bumped to v6 for that change of meaning.

## Control

`zg5fnw` -- same fold, same family (it additionally trained `frame_dir`, 44,032 params),
cross-seed at tau 0.0305 with `--alpha-pae 1` and lambda 1e-05. Reached paired +0.00052,
positive in 4 of 7 epochs, before being stopped at epoch 7 of 12.

Its training ranking loss never descended: 0.05525 at epoch 1 to 0.05865 at epoch 7, with the
best held-out DockQ at epoch 5 and falling after. Its held-out ranking loss was below shipped at
every epoch (0.0524-0.0593 against shipped 0.0606).

## Criterion and falsifier

**Primary:** fraction above 0.23 over valfull's 92 interfaces, against shipped **0.7391**.
**Secondary, explicitly not what this optimises:** selected mean DockQ against shipped **0.5831**.
13 of the 92 interfaces (14.1%) have no acceptable pose at any rank and are censored.

**Falsifier:** fraction at or below shipped over epochs 10-15, which would say that
concentrating the gradient on threshold-crossing pairs buys nothing even when it is the metric
being optimised.

## Reproduce

    python -I run_logging/measure_tau.py --pairs all --straddle 0.23
    ./shell_scripts/chain_straddle.sh      # probe -> derive lambda -> launch -> watch
