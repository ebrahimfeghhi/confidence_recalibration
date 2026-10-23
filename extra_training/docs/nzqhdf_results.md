# nzqhdf: FULL CONFIDENCE HEAD TRAINABLE AND ANCHORED, under the cross-seed softplus ranking loss

Regenerate with `python -I run_logging/render_findings.py --rid nzqhdf`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-6 (6 of 50.0).  **Interfaces:** 18.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.0305 (the `softplus` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 0.0 |
| trainable (from the run's own log) | **100 tensor(s), 5.3M of 464.5M parameters (1.2%)** |
| `--train-only` patterns | 75 |
| L2-SP lambda | 0.00025 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 96 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 50.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.1778 +- 0.0351 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.3889 +- 0.1182 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.0824 +- 0.0808 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.04587 | 0.04285 +- 0.00286 | 0.1953 | 38.7 | 0.0097 | 3.3002 |
| 2 | 0.04048 | 0.03624 +- 0.00253 | 0.2255 | 41.6 | 0.0104 | 3.7286 |
| 3 | 0.04137 | 0.03862 +- 0.00212 | 0.2080 | 27.4 | 0.0069 | 3.6454 |
| 4 | 0.04044 | 0.04721 +- 0.00382 | 0.1906 | 24.9 | 0.0062 | 3.3493 |
| 5 | 0.03889 | 0.04154 +- 0.00208 | 0.1951 | 20.1 | 0.0050 | 3.2092 |
| 6 | 0.03942 | 0.04372 +- 0.00296 | 0.1972 | 15.9 | 0.0040 | 3.2970 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2265 +- 0.0529 | 0.4444 +- 0.1205 | 0.0765 +- 0.0872 | 0.0065 |
| 2 | 0.2799 +- 0.0593 | 0.5000 +- 0.1213 | 0.2409 +- 0.0755 | 0.0033 |
| 3 | 0.2815 +- 0.0613 | 0.5556 +- 0.1205 | 0.1897 +- 0.0755 | 0.0042 |
| 4 | 0.1816 +- 0.0367 | 0.3333 +- 0.1143 | 0.0165 +- 0.0905 | 0.0082 |
| 5 | 0.2352 +- 0.0534 | 0.4444 +- 0.1205 | 0.0714 +- 0.0832 | 0.0046 |
| 6 | 0.1778 +- 0.0351 | 0.3889 +- 0.1182 | 0.0824 +- 0.0808 | 0.0065 |

## Paired per-interface test, epoch 1 -> 6

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **-0.0487 +- 0.0387** (1.26 sigma)
- 8 up, 8 worsened, 2 unchanged

| interface | ep1 | ep6 | delta |
|---|---|---|---|
| `9qrv_AC` | 0.476 | 0.059 | -0.417 |
| `9bx7_AC` | 0.587 | 0.184 | -0.403 |
| `8og0_BC` | 0.699 | 0.384 | -0.315 |
| `9c7x_BC` | 0.069 | 0.278 | +0.209 |
| `9cfd_AC` | 0.361 | 0.278 | -0.083 |

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep6.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 1.8312e+00 |
| `pose_perblock_frame_dir.0.weight` | 1.8122e+00 |
| `pose_perblock_frame_dir.1.weight` | 1.7876e+00 |
| `pose_perblock_frame_dir.2.weight` | 1.7704e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> FULL CONFIDENCE HEAD TRAINABLE AND ANCHORED, under the cross-seed softplus ranking loss. All 248 head tensors train (19.8M parameters, 4.3% of the model) with no --train-only; the anchor holds them near the shipped weights instead of freezing them. WHY: 24u34d plateaued with its held-out ranking loss at ~0.040 AND its training loss flat at the same value (0.04191 -> 0.04189 over epochs 6-7), both curves flattening together -- the signature of UNDERFITTING, so 44,032 parameters cannot express the correction the loss asks for. This is also the first full-head arm ever run under the ranking loss: all ~44 earlier ones used PAE (rank_form None). LAMBDA WAS MEASURED, NOT GUESSED, and that matters more here than in any previous arm: wpg274 ran this exact parameter set at l2_sp 1e-5 for 50 epochs under the PAE loss and ended at DockQ 0.1616 / frac 0.2778 against shipped 0.2909 / 0.5556, with drift at 3,801 and the penalty at 1.9% of the loss -- the anchor was not binding and the head moved freely to a much worse place. The sqrt factor alone (40,960 -> 19.84M anchored, 22.0x) would put lambda at 1.1e-5, to one digit that same failed value; it is not necessarily the same setting, because the ranking gradient is ~25x smaller than the PAE gradient (alignment probes: ||sum_i g_i|| 1.4e-2 rank vs 3.6e-1 PAE at the same weights), so the same lambda is ~25x more binding against this objective -- but that factor was measured by a fresh --grad-probe at this trainable set rather than assumed. The 4 zero-init orientation tensors stay OUT of the anchor via --l2-sp-exclude 'frame_dir', since anchoring a zero-init tensor to zero is plain decay on the new term. EVERYTHING ELSE MATCHES 24u34d so capacity is the only variable: tau 0.0305, --alpha-pae 0, distance AND orientation at all four pairformer blocks, overfit30 train, val10 held out every epoch at budget 30, 50 epochs, 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped (DockQ 0.2909, frac 0.5556, oracle 0.5968/1.0000). Spearman is reported but is NOT the criterion. THE RISK, STATED: 19.8M parameters against 30 training structures, and wpg274 is what this looks like when the anchor is too loose -- watch drift and the metrics together from epoch 1, not just the loss. CONTROLS: 24u34d (same arm, 44,032 parameters), wpg274 (same parameter set, loose anchor, PAE loss).

**The note above is STALE on one point.** It describes all 248 head tensors / 19.8M
parameters, which is what was intended at the time it was written. `--grad-probe`
then REFUSED that set, because 158 of those tensors receive no ranking gradient at
all, and the arm was narrowed before launch to the 100 tensors /
5.3M parameters the table above reports from the run's own log. See
`docs/confidence_head_reachability.md`. The note is left verbatim rather than edited,
because a run's recorded config should say what was configured, not be rewritten after
the fact -- but the number to trust is the logged one.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0949, 0 non-finite gradient(s).

