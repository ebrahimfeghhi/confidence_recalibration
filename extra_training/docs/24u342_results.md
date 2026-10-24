# 24u342: RESUME of 24u34d from resume_ep8

Regenerate with `python -I run_logging/render_findings.py --rid 24u342`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-22 (21 of 50.0).  **Interfaces:** 18.  **Budget:** 30.

**Resume chain:** `24u34d` -> `24u342`. A resume writes to a fresh directory
while continuing the parent's epoch numbering, so every table here spans the chain; the
parent is found from `args.json`'s recorded `--resume` checkpoint path, not from the
run id. An epoch trained but stopped before its held-out pass finished has no row here
rather than a carried-forward one.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.0305 (the `softplus` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 0.0 |
| trainable (from the run's own log) | **12 tensor(s), 0.0M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 3 |
| L2-SP lambda | 0.00025 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 8 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 50.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2814 +- 0.0549 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.6111 +- 0.1182 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2291 +- 0.0694 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.05395 | 0.04871 +- 0.01091 | 0.2370 | 1.1 | 0.0003 | 3.1478 |
| 2 | 0.04434 | 0.04068 +- 0.00524 | 0.2226 | 3.9 | 0.0010 | 3.4982 |
| 3 | 0.04322 | 0.03862 +- 0.00365 | 0.2182 | 5.3 | 0.0013 | 3.6831 |
| 4 | 0.04252 | 0.03978 +- 0.00453 | 0.2198 | 4.0 | 0.0010 | 3.5343 |
| 5 | 0.04234 | 0.04005 +- 0.00472 | 0.2191 | 3.8 | 0.0010 | 3.4454 |
| 6 | 0.04191 | 0.03947 +- 0.00400 | 0.2162 | 4.5 | 0.0011 | 3.5394 |
| 7 | 0.04189 | 0.04026 +- 0.00457 | 0.2165 | 4.4 | 0.0011 | 3.3473 |
| 9 | 0.04156 | 0.03999 +- 0.00447 | 0.2174 | 3.5 | 0.0009 | 3.3774 |
| 10 | 0.04154 | 0.03921 +- 0.00379 | 0.2157 | 4.4 | 0.0011 | 3.4476 |
| 11 | 0.04115 | 0.03962 +- 0.00432 | 0.2172 | 4.3 | 0.0011 | 3.3266 |
| 12 | 0.04131 | 0.03968 +- 0.00447 | 0.2171 | 5.0 | 0.0012 | 3.2262 |
| 13 | 0.04107 | 0.03922 +- 0.00399 | 0.2165 | 4.6 | 0.0012 | 3.3823 |
| 14 | 0.04129 | 0.04003 +- 0.00451 | 0.2162 | 5.1 | 0.0013 | 3.2475 |
| 15 | 0.04079 | 0.03943 +- 0.00417 | 0.2163 | 5.1 | 0.0013 | 3.3319 |
| 16 | 0.04102 | 0.03974 +- 0.00447 | 0.2177 | 4.3 | 0.0011 | 3.3649 |
| 17 | 0.04095 | 0.03935 +- 0.00404 | 0.2162 | 4.8 | 0.0012 | 3.3877 |
| 18 | 0.04095 | 0.03936 +- 0.00404 | 0.2161 | 4.4 | 0.0011 | 3.3877 |
| 19 | 0.04092 | 0.03989 +- 0.00446 | 0.2165 | 4.7 | 0.0012 | 3.3036 |
| 20 | 0.04092 | 0.03944 +- 0.00408 | 0.2158 | 4.7 | 0.0012 | 3.3565 |
| 21 | 0.04127 | 0.04072 +- 0.00522 | 0.2182 | 4.6 | 0.0011 | 3.2030 |
| 22 | 0.04083 | 0.03904 +- 0.00356 | 0.2143 | 5.3 | 0.0013 | 3.4617 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2616 +- 0.0587 | 0.5000 +- 0.1213 | 0.1994 +- 0.0703 | 0.0069 |
| 2 | 0.2834 +- 0.0619 | 0.5556 +- 0.1205 | 0.2082 +- 0.0716 | 0.0052 |
| 3 | 0.2607 +- 0.0570 | 0.5556 +- 0.1205 | 0.2263 +- 0.0704 | 0.0047 |
| 4 | 0.2849 +- 0.0577 | 0.5556 +- 0.1205 | 0.2222 +- 0.0663 | 0.0049 |
| 5 | 0.2819 +- 0.0553 | 0.6111 +- 0.1182 | 0.2330 +- 0.0665 | 0.0050 |
| 6 | 0.2835 +- 0.0543 | 0.6111 +- 0.1182 | 0.2251 +- 0.0683 | 0.0049 |
| 7 | 0.2893 +- 0.0531 | 0.6111 +- 0.1182 | 0.2150 +- 0.0673 | 0.0051 |
| 9 | 0.2808 +- 0.0550 | 0.6111 +- 0.1182 | 0.2222 +- 0.0678 | 0.0052 |
| 10 | 0.2859 +- 0.0541 | 0.6111 +- 0.1182 | 0.2224 +- 0.0687 | 0.0049 |
| 11 | 0.2872 +- 0.0536 | 0.6111 +- 0.1182 | 0.2257 +- 0.0668 | 0.0051 |
| 12 | 0.2998 +- 0.0548 | 0.6111 +- 0.1182 | 0.2290 +- 0.0671 | 0.0050 |
| 13 | 0.2871 +- 0.0540 | 0.6111 +- 0.1182 | 0.2242 +- 0.0678 | 0.0050 |
| 14 | 0.2892 +- 0.0541 | 0.6111 +- 0.1182 | 0.2219 +- 0.0666 | 0.0052 |
| 15 | 0.2949 +- 0.0557 | 0.6111 +- 0.1182 | 0.2256 +- 0.0680 | 0.0050 |
| 16 | 0.2993 +- 0.0548 | 0.6111 +- 0.1182 | 0.2262 +- 0.0671 | 0.0051 |
| 17 | 0.2949 +- 0.0557 | 0.6111 +- 0.1182 | 0.2272 +- 0.0678 | 0.0050 |
| 18 | 0.2959 +- 0.0560 | 0.6111 +- 0.1182 | 0.2284 +- 0.0681 | 0.0050 |
| 19 | 0.2972 +- 0.0565 | 0.6111 +- 0.1182 | 0.2306 +- 0.0667 | 0.0052 |
| 20 | 0.3016 +- 0.0548 | 0.6111 +- 0.1182 | 0.2326 +- 0.0682 | 0.0050 |
| 21 | 0.3017 +- 0.0556 | 0.6111 +- 0.1182 | 0.2367 +- 0.0651 | 0.0054 |
| 22 | 0.2814 +- 0.0549 | 0.6111 +- 0.1182 | 0.2291 +- 0.0694 | 0.0048 |

## Paired per-interface test, epoch 1 -> 22

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **+0.0199 +- 0.0300** (0.66 sigma)
- 6 up, 4 worsened, 8 unchanged

| interface | ep1 | ep22 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `8og0_AC` | 0.655 | 0.370 | -0.285 |
| `9c7x_BC` | 0.071 | 0.241 | +0.170 |
| `7r58_AB` | 0.011 | 0.087 | +0.076 |
| `9bx7_BC` | 0.274 | 0.231 | -0.043 |

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
| 1 | 45.50 +- 14.03 | 1.5 | 9/18 | +8.06 +- 5.09 | 3/6/9 |
| 2 | 42.94 +- 14.06 | 1.0 | 10/18 | +5.44 +- 3.04 | 3/5/10 |
| 3 | 41.83 +- 13.98 | 1.0 | 10/18 | +4.39 +- 3.12 | 3/4/11 |
| 4 | 43.11 +- 14.07 | 1.0 | 10/18 | +5.61 +- 3.03 | 1/6/11 |
| 5 | 37.78 +- 12.87 | 1.0 | 11/18 | +0.33 +- 2.88 | 4/4/10 |
| 6 | 37.67 +- 12.80 | 1.0 | 11/18 | +0.17 +- 3.15 | 4/4/10 |
| 7 | 37.39 +- 12.78 | 1.0 | 11/18 | -0.11 +- 3.27 | 3/4/11 |
| 9 | 37.39 +- 12.54 | 1.0 | 11/18 | -0.06 +- 3.70 | 4/4/10 |
| 10 | 38.33 +- 12.68 | 1.0 | 11/18 | +0.83 +- 3.99 | 4/4/10 |
| 11 | 38.22 +- 12.70 | 1.0 | 11/18 | +0.72 +- 3.91 | 4/4/10 |
| 12 | 37.00 +- 12.64 | 1.0 | 11/18 | -0.44 +- 3.60 | 3/4/11 |
| 13 | 38.17 +- 12.69 | 1.0 | 11/18 | +0.67 +- 3.49 | 4/4/10 |
| 14 | 37.72 +- 12.69 | 1.0 | 11/18 | +0.28 +- 3.63 | 4/4/10 |
| 15 | 37.28 +- 12.59 | 1.0 | 11/18 | -0.17 +- 3.56 | 4/4/10 |
| 16 | 38.11 +- 12.71 | 1.0 | 11/18 | +0.61 +- 3.35 | 4/4/10 |
| 17 | 37.83 +- 12.63 | 1.0 | 11/18 | +0.33 +- 3.44 | 4/4/10 |
| 18 | 37.06 +- 12.52 | 1.0 | 11/18 | -0.44 +- 3.10 | 4/4/10 |
| 19 | 37.06 +- 12.56 | 1.0 | 11/18 | -0.44 +- 3.23 | 4/4/10 |
| 20 | 36.44 +- 12.52 | 1.0 | 11/18 | -1.00 +- 2.87 | 4/4/10 |
| 21 | 36.94 +- 12.62 | 1.0 | 11/18 | -0.56 +- 3.26 | 4/4/10 |
| 22 | 36.78 +- 12.47 | 1.0 | 11/18 | -0.67 +- 3.10 | 4/4/10 |

Shipped on the same interfaces: mean rank 37.44 +- 12.54, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep22.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 1.3064e+01 |
| `pose_perblock_frame_dir.0.weight` | 1.3045e+01 |
| `pose_perblock_frame_dir.1.weight` | 1.1519e+01 |
| `pose_perblock_frame_dir.2.weight` | 1.2040e+01 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> RESUME of 24u34d from resume_ep8.pt, continuing to epoch 50. Identical in every setting: cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0 so ordering is the only term, distance AND orientation injected at ALL FOUR pairformer blocks with untied per-block copies, --train-only 'linear_no_bias_d,pose_perblock,frame_dir' = 12 tensors / 44,032 parameters, L2-SP at lambda 2.5e-4 over 8 anchored tensors with --l2-sp-exclude 'frame_dir' keeping the 4 zero-init orientation tensors trainable but out of the anchor, overfit30 train, val10 held out every epoch at budget 30, 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0. THE ANCHOR WAS COPIED FROM THE PARENT and the log must say 'anchor READ from'; a 'SNAPSHOT at init' line would mean the copy lost the race and the arm is anchored to its own 8-epoch state instead of the shipped head. WHY IT WAS INTERRUPTED: stopped at epoch 8 to free cards 4,5,7 for nzqhdf, which tested whether the plateau was a capacity limit by training the 5.3M reachable head parameters. It is not: nzqhdf drove the held-out ranking loss to 0.03624 by epoch 2, through the 0.042 wall, while selection fell to the worst on record (DockQ 0.1816, frac 0.3333, rho 0.0165 at epoch 4) with the training loss still falling as the held-out loss rose. 24u34d is the only arm ever to beat shipped on frac acceptable -- 0.6111 at epochs 5-7 against 0.5556 -- with DockQ 0.2893 against 0.2909 and both losses flattening together rather than diverging. PARENT'S STATE AT THE STOP: DockQ 0.2893 +- 0.0531, frac 0.6111 +- 0.1182, rho 0.2150 +- 0.0673, held-out ranking loss 0.04026, E[DockQ] 0.2165 against shipped 0.2572, drift 3.8, orientation norms 5.66/5.66/5.14/5.02 from exact zero. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968. Spearman is reported but is NOT the criterion. THE OPEN DISAGREEMENT: frac improved while E[DockQ] moved monotonically AWAY from shipped every epoch, and the frac gain is one interface (9c7x_BC crossing 0.23 by 0.011) -- which is exactly why this needs to run past the ten-epoch minimum instead of being judged at 7.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0103, 0 non-finite gradient(s).

