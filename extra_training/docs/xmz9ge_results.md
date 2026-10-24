# xmz9ge: L8 PLUS BLOCK 3's pair_transition, 50 epochs

Regenerate with `python -I run_logging/render_findings.py --rid xmz9ge`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-1 (1 of 50.0).  **Interfaces:** 18.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.027 |
| alpha_rank / alpha_pae | 1.0 / 0.0 |
| trainable (from the run's own log) | **17 tensor(s), 0.8M of 464.5M parameters (0.2%)** |
| `--train-only` patterns | 4 |
| L2-SP lambda | 5e-05 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 13 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 50.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2650 +- 0.0599 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5000 +- 0.1213 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2121 +- 0.0650 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.04540 | 0.04433 +- 0.00763 | 0.2293 | 3.5 | 0.0002 | 3.1916 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2650 +- 0.0599 | 0.5000 +- 0.1213 | 0.2121 +- 0.0650 | 0.0067 |

## Rank of the first acceptable pose, ordering by ipTM

`frac acceptable` only asks whether rank 1 happens to be acceptable, so it cannot see an
arm that lifts the best pose from rank 90 to rank 2. This can. Rank is 1-indexed and
**lower is better**; rank 1 everywhere is frac 1.0.

Scored over 150 poses per interface, threshold DockQ >= 0.23, 0 interface(s) with no acceptable pose anywhere (those have no rank and are excluded; a non-zero count here would bias the mean optimistically, since the hardest interfaces would be the ones dropped).

The unpaired s.e. is large because a few interfaces first become acceptable in the 90s
while most are acceptable at rank 1, so between-interface variance swamps the effect.
**The paired column is the one that can resolve a change** -- every interface is scored
by both orderings over the same poses, so the pairing is exact.

| epoch | mean rank +- s.e. | median | at rank 1 | paired vs shipped | better/worse/same |
|---|---|---|---|---|---|
| 1 | 44.89 +- 14.84 | 1.5 | 9/18 | +7.44 +- 4.01 | 1/8/9 |

Shipped on the same interfaces: mean rank 37.44 +- 12.54, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep1.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 5.4949e-01 |
| `pose_perblock_frame_dir.0.weight` | 5.5103e-01 |
| `pose_perblock_frame_dir.1.weight` | 4.2926e-01 |
| `pose_perblock_frame_dir.2.weight` | 4.1112e-01 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> L8 PLUS BLOCK 3's pair_transition, 50 epochs. ONE VARIABLE AGAINST hronsr: the trainable set goes from 12 tensors / 44,032 parameters to 17 / 830,976, an 18.9x increase, by adding pairformer_stack.blocks.3.pair_transition (layernorm1.weight+bias 512, linear_no_bias_a 1024x256, linear_no_bias_b 1024x256, linear_no_bias 256x1024 = 786,944). Everything else matches hronsr exactly: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270, --rank-topk-decay 0, softplus, --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, overfit30 train, val10 held out every epoch at budget 30, 2 structures x 10 seeds x 5 poses = 100 poses/step, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009, --blocks-per-ckpt 0. WHY THIS FAMILY: ipTM reads linear_no_bias_pae(pae_ln(z_pair)) and pair_transition acts on z_pair, so unlike the 73.1% of the head that cannot affect ipTM at all every one of these parameters is reachable; across all four blocks pair_transition is 3,147,776 parameters = 59% of the whole 5,313,536-parameter reachable set, and this arm takes the last block's share. LAMBDA IS 5e-05, NOT hronsr's 2.5e-4, derived rather than carried: L2-SP strength relative to the task is 2*lambda*||dW||/|g_task| and both factors moved, the anchored set going 40,960 -> 827,904 parameters (||dW|| grows as its square root) and |g_rank| measuring 5.1398e-02 here against 4.0207e-02 at the old set, so run_logging/l2sp_lambda.py gives 2.5e-4 * 1.2783 / 4.4958 = 7.11e-05, used as 5e-05. Keeping 2.5e-4 would have pinned the new weights ~3.5x harder than hronsr's. PROBE n24xyz also measured rank ALIGNMENT 0.1072 against a 1/sqrt(N) baseline of 0.1000, so the ranking gradient remains as good as mutually independent across poses even at 19x the parameters -- the ~91% cancellation tax is UNCHANGED by capacity. THE NEW TENSORS ARE ANCHORED and the orientation ones still are not: L2-SP prices distance from the SHIPPED weights, a prior worth holding for a pretrained matrix, and pair_transition is pretrained, while frame_dir stays in --l2-sp-exclude because those four tensors are ZERO-INIT and anchoring them to their init is plain decay toward zero. CONTROLS: hronsr (this arm minus the new tensors, 15 epochs, paired +0.0116 against shipped over epochs 8-15, positive in 8 of 8, final DockQ 0.3024 vs shipped 0.2909 at frac parity 0.5556) and nzqhdf (all 5.3M reachable parameters, mostly pair_transition, which drove the held-out ranking loss to the lowest on record at 0.03624 while selection collapsed to DockQ 0.1816 / frac 0.3333 -- the standing warning that capacity here buys loss, not selection). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing on hronsr's +0.0116 within error, which would say the extra 786,944 parameters buy nothing the 44,032 did not already provide.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0958, 0 non-finite gradient(s).

