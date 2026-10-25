# aup7ku: PAE READOUT UNFROZEN, pair_transition FROZEN, lambda 0

Regenerate with `python -I run_logging/render_findings.py --rid aup7ku`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-1 (1 of 15.0).  **Interfaces:** 18.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.0305 (the `softplus` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 7.854 / 1.0 |
| trainable (from the run's own log) | **15 tensor(s), 0.1M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 5 |
| L2-SP lambda | 0.005 |
| L2-SP excluded | `frame_dir,pae_ln` -- 6 tensor(s) out of the anchor, 9 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 15.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2749 +- 0.0596 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5000 +- 0.1213 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2096 +- 0.0693 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.05169 | 0.04386 +- 0.00791 | 0.2281 | 5.3 | 0.0265 | 2.9278 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2749 +- 0.0596 | 0.5000 +- 0.1213 | 0.2096 +- 0.0693 | 0.0041 |

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
| 1 | 38.72 +- 13.22 | 1.5 | 9/18 | +1.28 +- 2.99 | 3/4/11 |

Shipped on the same interfaces: mean rank 37.44 +- 12.54, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir,pae_ln` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep1.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 5.6853e-01 |
| `pae_ln.bias` | 2.0437e+00 |
| `pae_ln.weight` | 3.8685e+00 |
| `pose_perblock_frame_dir.0.weight` | 5.6907e-01 |
| `pose_perblock_frame_dir.1.weight` | 5.0078e-01 |
| `pose_perblock_frame_dir.2.weight` | 4.2287e-01 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> PAE READOUT UNFROZEN, pair_transition FROZEN, lambda 0.005, alpha_rank 7.854, alpha_pae 1.0, 15 epochs. THE QUESTION: is the PAE pathway the limit? oe6sru ran this exact objective with the readout FROZEN and DID NOT LEARN THE PAE LOSS AT ALL -- held-out PAE 2.9085 -> 2.9029 over seven epochs, a 0.2% change, while bouncing more than that between epochs (2.9863 at epoch 2); training PAE 2.29-2.56 with no trend, rising at epochs 3, 5 and 7. Both terms were stuck, not only the ranking one. The structural suspect is that pae_ln and linear_no_bias_pae were frozen, fixing the map from z_pair to PAE logits so the only route to the PAE loss was through z -- the same narrow pathway every ranking arm in this project has failed to move. WHAT CHANGES against oe6sru: pairformer_stack.blocks.3.pair_transition is REMOVED from --train-only (-786,944 parameters) and pae_ln.weight+bias (+512) and linear_no_bias_pae.weight 64x256 (+16,384) are ADDED, giving 15 tensors / 60,928 parameters against oe6sru's 17 / 830,976. L2-SP COVERS THE READOUT BUT NOT THE LAYERNORM: --l2-sp-exclude 'frame_dir,pae_ln' leaves pae_ln trainable and out of the anchor while linear_no_bias_pae stays anchored, so 9 tensors / 57,344 parameters are priced against shipped. The readout is a pretrained 64x256 matrix and distance from shipped is a prior worth holding for it; a LayerNorm's 512 scale-and-bias numbers set the scale that readout sees, and anchoring them would fight the adjustment this arm exists to allow. frame_dir stays excluded because it is zero-init and anchoring it to its own init is plain decay toward zero. ALL THREE WEIGHTS MEASURED AT THIS TRAINABLE SET by probe_readout.sh, none defaulted: unfreezing a 16,384-parameter readout changes both |g_PAE| and |g_rank|, so neither oe6sru's alpha_rank 5.347 nor its lambda 5e-04 carries over. CONTROLS: oe6sru (same objective, readout frozen, pair_transition trainable) and 24u342 (same pool, cross-seed ranking only, no PAE, 50 epochs). NOTE ntwjfp and qiodxz trained pae_ln and linear_no_bias_pae before, but with the pre-flag LINEAR within-seed loss, no tau, no L2-SP and no orientation input, so neither is a control for this. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards; held-out PAE from val_loss.csv is the secondary diagnostic this arm was built to move. TEN-EPOCH MINIMUM. FALSIFIER: held-out PAE still flat, which would move the suspicion off the frozen readout and onto the objective or the data, every pathway into the PAE head having then been opened.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 2.8700, 0 non-finite gradient(s).

