# oe6sru: SIMPLE RANKING LOSS + PAE + L2-SP at the pair_transition-block-3 trainable set, 25 epochs

Regenerate with `python -I run_logging/render_findings.py --rid oe6sru`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-5 (5 of 25.0).  **Interfaces:** 18.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.0305 (the `softplus` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 5.347 / 1.0 |
| trainable (from the run's own log) | **17 tensor(s), 0.8M of 464.5M parameters (0.2%)** |
| `--train-only` patterns | 4 |
| L2-SP lambda | 0.0005 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 13 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 25.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2966 +- 0.0600 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5556 +- 0.1205 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.1963 +- 0.0677 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.05297 | 0.04451 +- 0.00861 | 0.2338 | 7.3 | 0.0036 | 2.9085 |
| 2 | 0.04329 | 0.03928 +- 0.00453 | 0.2208 | 13.1 | 0.0066 | 2.9863 |
| 3 | 0.04230 | 0.03961 +- 0.00472 | 0.2194 | 13.8 | 0.0069 | 2.9286 |
| 4 | 0.04204 | 0.04065 +- 0.00542 | 0.2205 | 14.2 | 0.0071 | 2.9169 |
| 5 | 0.04274 | 0.04086 +- 0.00551 | 0.2207 | 14.2 | 0.0071 | 2.9110 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2639 +- 0.0584 | 0.4444 +- 0.1205 | 0.2305 +- 0.0688 | 0.0065 |
| 2 | 0.2918 +- 0.0599 | 0.5556 +- 0.1205 | 0.2146 +- 0.0687 | 0.0065 |
| 3 | 0.2956 +- 0.0596 | 0.5556 +- 0.1205 | 0.2018 +- 0.0696 | 0.0062 |
| 4 | 0.3029 +- 0.0588 | 0.5556 +- 0.1205 | 0.2001 +- 0.0676 | 0.0062 |
| 5 | 0.2966 +- 0.0600 | 0.5556 +- 0.1205 | 0.1963 +- 0.0677 | 0.0067 |

## Paired per-interface test, epoch 1 -> 5

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **+0.0327 +- 0.0226** (1.44 sigma)
- 5 up, 0 worsened, 13 unchanged

| interface | ep1 | ep5 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `9bx7_BC` | 0.172 | 0.274 | +0.102 |
| `9ima_BD` | 0.259 | 0.306 | +0.047 |
| `7qu2_AC` | 0.033 | 0.068 | +0.035 |
| `9c7x_AC` | 0.067 | 0.069 | +0.002 |

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
| 1 | 37.94 +- 13.11 | 2.0 | 8/18 | +0.50 +- 3.69 | 2/7/9 |
| 2 | 36.61 +- 12.69 | 1.0 | 10/18 | -0.89 +- 3.04 | 4/4/10 |
| 3 | 36.17 +- 12.76 | 1.0 | 10/18 | -1.28 +- 3.43 | 4/4/10 |
| 4 | 36.56 +- 12.71 | 1.0 | 10/18 | -0.94 +- 3.07 | 4/4/10 |
| 5 | 36.28 +- 12.75 | 1.0 | 10/18 | -1.17 +- 3.76 | 4/4/10 |

Shipped on the same interfaces: mean rank 37.44 +- 12.54, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep5.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 2.7252e+00 |
| `pose_perblock_frame_dir.0.weight` | 2.7518e+00 |
| `pose_perblock_frame_dir.1.weight` | 2.7135e+00 |
| `pose_perblock_frame_dir.2.weight` | 2.4308e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> SIMPLE RANKING LOSS + PAE + L2-SP at the pair_transition-block-3 trainable set, 25 epochs. DROPS from xmz9ge: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270 -- the ranking term is plain cross-seed softplus at DEFAULT_TAU 0.0305 again. ADDS --alpha-pae 1, where every arm since h7msge ran alpha_pae 0 and made ordering the only term. TRAINABLE SET UNCHANGED at 17 tensors / 830,976 parameters: the 12 orientation+distance tensors plus pairformer_stack.blocks.3.pair_transition. ALL THREE WEIGHTS MEASURED AT THESE WEIGHTS from probe jn2pg7, not guessed. alpha_pae 1 is the reference every PAE arm uses. alpha_rank 5.347 is GRADIENT PARITY AT INIT: the probe reports |g_PAE| 2.5742e-01 against |g_rank| 4.8142e-02 in one accumulation window at identical weights and prints the parity ratio itself, so the two terms contribute equal gradient magnitude at step 0 -- the same method that set arm B's 2.058. lambda 5e-04 is ~10x the ranking-only value because L2-SP strength is 2*lambda*||dW||/|g_task| and |g_task| is now COMBINED: |g_PAE + 5.347 g_rank| = 0.4225 against 0.0481 for ranking alone, so l2sp_lambda.py transfers 24u34d's 2.5e-4 as 2.5e-4 * (0.4225/0.0402) / sqrt(827904/40960) = 5.84e-4, used as 5e-04. THE TWO OBJECTIVES AGREE: cos(PAE, rank) = +0.3466, so the ranking term reinforces the PAE direction rather than fighting it. That is the reason to expect anything here -- the ranking loss ALONE produces per-pose gradients as good as mutually independent (alignment 0.107 against a 1/sqrt(N) baseline of 0.100, ~90% of its magnitude destroyed by cancellation every step) while PAE's are nearly parallel, so PAE supplies a coherent direction the ranking term can sharpen. PARITY HOLDS AT INITIALISATION ONLY: in 4y3ve6 the ranking gradient decayed ~22x over 11 epochs as pairs became ordered, so the ranking term fades relative to PAE as training proceeds -- an accidental annealing schedule that points the right way, but alpha_rank is a STARTING ratio. THE LAMBDA TRANSFER COMPARES UNLIKE QUANTITIES, stated rather than hidden: the old side is ranking-only at alpha_pae 0, the new side combined; the invariant preserved is still anchor strength relative to the task gradient, but 5e-04 is a principled starting point and not a tuned value. CONTROLS: xmz9ge (same trainable set, within-seed weighting, NO PAE -- stopped at epoch 9 having put 89% of its total training-loss fall into epoch 1->2 and then plateaued at 0.0365 while selection swung +-0.07 DockQ per epoch) and hronsr (44,032 parameters, within-seed weighting, no PAE, paired +0.0116 over epochs 8-15). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim; reported at 15 as well so it stays comparable to the three 15-epoch arms. FALSIFIER: the same 0.037 plateau at epoch 2 with selection still swinging, which would say PAE's coherent direction does not rescue a ranking signal that cancels itself.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.9758, 0 non-finite gradient(s).

