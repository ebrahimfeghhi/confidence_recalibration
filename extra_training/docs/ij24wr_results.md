# ij24wr: FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-3 ANCHORING EVERYTHING EXCEPT THE POSE PARAMETERS

Regenerate with `python -I run_logging/render_findings.py --rid ij24wr`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-1 (1 of 20.0).  **Interfaces:** 92.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `linear`, tau 0.0305 (the `linear` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 1.0 |
| trainable (from the run's own log) | **254 tensor(s), 19.8M of 464.5M parameters (4.3%)** |
| `--train-only` patterns | none (whole head) |
| L2-SP lambda | 0.001 |
| L2-SP excluded | `linear_no_bias_d` -- 8 tensor(s) out of the anchor, 246 anchored |
| pose injection | per-block=True copies=True orientation=False |
| lr / warmup / epochs | 0.01 / 20 / 20.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5736 +- 0.0385 | 0.5831 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7283 +- 0.0466 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2120 +- 0.0343 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | nan | -0.09644 +- 0.01932 | 0.5658 | 59.2 | 0.0592 | 2.8207 |

Shipped, on these same interfaces: held-out ranking loss -0.10567, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5736 +- 0.0385 | 0.7283 +- 0.0466 | 0.2120 +- 0.0343 | 0.0071 |

## Rank of the first acceptable pose, ordering by ipTM

`frac acceptable` only asks whether rank 1 happens to be acceptable, so it cannot see an
arm that lifts the best pose from rank 90 to rank 2. This can. Rank is 1-indexed and
**lower is better**; rank 1 everywhere is frac 1.0.

Scored over 150 poses per interface, threshold DockQ >= 0.23, 13 interface(s) with no acceptable pose anywhere (those have no rank and are excluded; a non-zero count here would bias the mean optimistically, since the hardest interfaces would be the ones dropped).

The unpaired s.e. is large because a few interfaces first become acceptable in the 90s
while most are acceptable at rank 1, so between-interface variance swamps the effect.
**The paired column is the one that can resolve a change** -- every interface is scored
by both orderings over the same poses, so the pairing is exact.

| epoch | mean rank +- s.e. | median | at rank 1 | paired vs shipped | better/worse/same |
|---|---|---|---|---|---|
| 1 | 10.32 +- 3.44 | 1.0 | 67/79 | -0.94 +- 0.88 | 7/5/67 |

Shipped on the same interfaces: mean rank 11.25 +- 3.58, median 1.0, 68/79 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude linear_no_bias_d` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep1.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `linear_no_bias_d.weight` | 1.8060e+01 |
| `linear_no_bias_d_wo_onehot.weight` | 3.3625e+00 |
| `pose_perblock_linear_no_bias_d.0.weight` | 1.7731e+01 |
| `pose_perblock_linear_no_bias_d.1.weight` | 1.8042e+01 |
| `pose_perblock_linear_no_bias_d.2.weight` | 1.9119e+01 |
| `pose_perblock_linear_no_bias_d_wo_onehot.0.weight` | 3.3317e+00 |
| `pose_perblock_linear_no_bias_d_wo_onehot.1.weight` | 3.5832e+00 |
| `pose_perblock_linear_no_bias_d_wo_onehot.2.weight` | 3.5628e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-3 ANCHORING EVERYTHING EXCEPT THE POSE PARAMETERS. The 8 pose tensors (block 0's linear_no_bias_d and linear_no_bias_d_wo_onehot plus three copies of each, 40,960 parameters) train FREE with no anchor; the other 246 head tensors go from FROZEN to trainable-but-priced at 1e-3 * ||W - W_shipped||^2. The exclusion is an unanchored regex, so it selects exactly the set dntxmn trained, and the launch log must print 8 excluded / 246 anchored or the arm is not what it claims. WHY: every capacity increase so far went into the generalisation gap. The unconstrained full head (wpg274, 50 ep) ended with val PAE 3.0990 against 2.7463 at epoch 1 while train fell to 1.7854, and its best epoch of 50 was still below shipped on paired DockQ (-0.0020 +- 0.0292 on val10's 18 interfaces). share and copy kept val falling but converted train gains to val at about half rate, and copy's gap was already wider than share's by +0.0067 +- 0.0017 over 12 matched epochs. So the hypothesis is not less capacity but capacity ONLY WHERE IT IS NEW: no reason to want the 246 pretrained tensors far from shipped, every reason to want the pose injection free. LAMBDA IS NOT DERIVED -- 1e-3 is the user's choice (lowered from 1e-2 before launch, because 1e-2 would price wpg274's measured drift of 3,801 at 38.0 against a loss near 2.6 and likely pin the 246 tensors; 1e-3 prices it at 3.8) and l2sp_lambda.py's transfer formula is deliberately not used, since on wpg274 it produced a penalty of 0.0380 against a loss near 2.0 by sizing lambda for an assumed drift 18x larger than what occurred. Read the measured ||W-W0||^2 and penalty printed each epoch against train and val PAE. ONE VARIABLE AGAINST dntxmn: same fold, selection, skip list, crop, lr, seeds, accum, nsample, valfull, 20 epochs, pose-copy injection; the trainable restriction removed and the anchor added. CONTROL: dntxmn through epoch 14 ran paired dDockQ +0.0016 mean on valfull's 92 interfaces, rho 0.2478 vs shipped 0.2218, val PAE bottoming at 2.7580 at epoch 13. TEN-EPOCH MINIMUM does not apply -- the user retracted it. VAL10 AND VALFULL ARE NOT COMPARABLE.

## Trajectory, latest epoch

244 optimiser step(s), max `grad_norm` 7.4285, 0 non-finite gradient(s).

