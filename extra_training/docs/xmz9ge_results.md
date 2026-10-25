# xmz9ge: L8 PLUS BLOCK 3's pair_transition, 50 epochs

Regenerate with `python -I run_logging/render_findings.py --rid xmz9ge`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-6 (6 of 50.0).  **Interfaces:** 18.  **Budget:** 30.

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
| selected DockQ | 0.2464 +- 0.0509 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5000 +- 0.1213 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.1329 +- 0.0697 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.04540 | 0.04433 +- 0.00763 | 0.2293 | 3.5 | 0.0002 | 3.1916 |
| 2 | 0.03745 | 0.03824 +- 0.00286 | 0.2165 | 9.8 | 0.0005 | 3.7247 |
| 3 | 0.03736 | 0.03704 +- 0.00221 | 0.2165 | 11.5 | 0.0006 | 4.0295 |
| 4 | 0.03707 | 0.03878 +- 0.00282 | 0.2124 | 10.5 | 0.0005 | 3.5779 |
| 5 | 0.03710 | 0.04025 +- 0.00271 | 0.2083 | 8.6 | 0.0004 | 3.6425 |
| 6 | 0.03657 | 0.04142 +- 0.00269 | 0.2049 | 7.7 | 0.0004 | 3.7432 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2650 +- 0.0599 | 0.5000 +- 0.1213 | 0.2121 +- 0.0650 | 0.0067 |
| 2 | 0.2692 +- 0.0554 | 0.5000 +- 0.1213 | 0.2291 +- 0.0662 | 0.0054 |
| 3 | 0.3034 +- 0.0596 | 0.5556 +- 0.1205 | 0.2599 +- 0.0612 | 0.0042 |
| 4 | 0.2364 +- 0.0528 | 0.4444 +- 0.1205 | 0.1848 +- 0.0744 | 0.0062 |
| 5 | 0.2483 +- 0.0518 | 0.5000 +- 0.1213 | 0.1532 +- 0.0702 | 0.0059 |
| 6 | 0.2464 +- 0.0509 | 0.5000 +- 0.1213 | 0.1329 +- 0.0697 | 0.0057 |

## Paired per-interface test, epoch 1 -> 6

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **-0.0186 +- 0.0443** (0.42 sigma)
- 7 up, 8 worsened, 3 unchanged

| interface | ep1 | ep6 | delta |
|---|---|---|---|
| `9qrv_AB` | 0.598 | 0.040 | -0.558 |
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `8og0_AC` | 0.655 | 0.370 | -0.285 |
| `9c7x_BC` | 0.071 | 0.241 | +0.170 |
| `7r58_AB` | 0.011 | 0.087 | +0.076 |

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
| 2 | 41.50 +- 13.97 | 1.5 | 9/18 | +4.00 +- 3.17 | 3/6/9 |
| 3 | 40.22 +- 13.36 | 1.0 | 10/18 | +2.78 +- 2.53 | 4/3/11 |
| 4 | 33.72 +- 12.37 | 2.0 | 8/18 | -3.78 +- 3.87 | 5/6/7 |
| 5 | 34.89 +- 12.71 | 1.5 | 9/18 | -2.56 +- 3.83 | 5/5/8 |
| 6 | 34.28 +- 12.69 | 1.5 | 9/18 | -3.22 +- 4.01 | 5/5/8 |

Shipped on the same interfaces: mean rank 37.50 +- 12.56, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep6.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 3.6998e+00 |
| `pose_perblock_frame_dir.0.weight` | 3.7015e+00 |
| `pose_perblock_frame_dir.1.weight` | 3.5043e+00 |
| `pose_perblock_frame_dir.2.weight` | 3.2297e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> L8 PLUS BLOCK 3's pair_transition, 50 epochs. ONE VARIABLE AGAINST hronsr: the trainable set goes from 12 tensors / 44,032 parameters to 17 / 830,976, an 18.9x increase, by adding pairformer_stack.blocks.3.pair_transition (layernorm1.weight+bias 512, linear_no_bias_a 1024x256, linear_no_bias_b 1024x256, linear_no_bias 256x1024 = 786,944). Everything else matches hronsr exactly: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270, --rank-topk-decay 0, softplus, --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, overfit30 train, val10 held out every epoch at budget 30, 2 structures x 10 seeds x 5 poses = 100 poses/step, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009, --blocks-per-ckpt 0. WHY THIS FAMILY: ipTM reads linear_no_bias_pae(pae_ln(z_pair)) and pair_transition acts on z_pair, so unlike the 73.1% of the head that cannot affect ipTM at all every one of these parameters is reachable; across all four blocks pair_transition is 3,147,776 parameters = 59% of the whole 5,313,536-parameter reachable set, and this arm takes the last block's share. LAMBDA IS 5e-05, NOT hronsr's 2.5e-4, derived rather than carried: L2-SP strength relative to the task is 2*lambda*||dW||/|g_task| and both factors moved, the anchored set going 40,960 -> 827,904 parameters (||dW|| grows as its square root) and |g_rank| measuring 5.1398e-02 here against 4.0207e-02 at the old set, so run_logging/l2sp_lambda.py gives 2.5e-4 * 1.2783 / 4.4958 = 7.11e-05, used as 5e-05. Keeping 2.5e-4 would have pinned the new weights ~3.5x harder than hronsr's. PROBE n24xyz also measured rank ALIGNMENT 0.1072 against a 1/sqrt(N) baseline of 0.1000, so the ranking gradient remains as good as mutually independent across poses even at 19x the parameters -- the ~91% cancellation tax is UNCHANGED by capacity. THE NEW TENSORS ARE ANCHORED and the orientation ones still are not: L2-SP prices distance from the SHIPPED weights, a prior worth holding for a pretrained matrix, and pair_transition is pretrained, while frame_dir stays in --l2-sp-exclude because those four tensors are ZERO-INIT and anchoring them to their init is plain decay toward zero. CONTROLS: hronsr (this arm minus the new tensors, 15 epochs, paired +0.0116 against shipped over epochs 8-15, positive in 8 of 8, final DockQ 0.3024 vs shipped 0.2909 at frac parity 0.5556) and nzqhdf (all 5.3M reachable parameters, mostly pair_transition, which drove the held-out ranking loss to the lowest on record at 0.03624 while selection collapsed to DockQ 0.1816 / frac 0.3333 -- the standing warning that capacity here buys loss, not selection). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing on hronsr's +0.0116 within error, which would say the extra 786,944 parameters buy nothing the 44,032 did not already provide.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0284, 0 non-finite gradient(s).

