# Experiment tracker

One file. What is running, what ran, where the code is, where the numbers land.
Last updated by the session that launched `ntwjfp`.

---

## WHERE THINGS STAND

`ntwjfp` ran 8 epochs and is being abandoned, not for lack of signal but because the LINEAR
ranking objective is unsound. See THE LINEAR LOSS IS UNSOUND below. The replacement -- a scaled
softplus over cross-seed pairs -- is implemented but NOT launched.

## THE RUN THAT FAILED

**`ntwjfp`** — pose-share + ranking loss + **trainable PAE readout**, α_rank 250, 50 epochs, no L2-SP.

| | |
|---|---|
| launcher | [`shell_scripts/launch_poseshare_rank_readout.sh`](../shell_scripts/launch_poseshare_rank_readout.sh) |
| trainable | 5 tensors, **27,136 params** (0.137% of the head) |
| | `linear_no_bias_d` 9,984 · `linear_no_bias_d_wo_onehot` 256 · `linear_no_bias_pae` 16,384 · `pae_ln` weight+bias 512 |
| architecture | `--pose-perblock --pose-perblock-share` — block 0's own pose injection re-applied at all 4 pairformer blocks, zero new parameters |
| loss | `PAE + 250 × ranking` |
| data | 30 structures (`training/overfit30`), 10 seeds, 5 poses/seed |
| held out | val10 — 10 structures, **18 interfaces**, budget 30 |
| outputs | `training/runs/ntwjfp/` |

### Why the readout was unfrozen

Measured on the shipped head (`protenix-v2.pt`):

- **35 of 256** `pae_ln` γ channels sit at |γ| ≤ 5.7e-9, with |β| ≤ 6.1e-10
- `linear_no_bias_pae` reads those 35 with **4.3e-8** of column weight against **230.4** for the other 221
- Spearman(|γ|, projection column mass) = **+0.929** — LayerNorm and the readout agree almost perfectly about which channels matter
- **9.2%** of the two pose projections' output weight lands on those dead channels

Frozen, the pose injection cannot recover any of that. This arm lets it.

### α = 250, measured not guessed

`--grad-probe` (run `qiodxz`), one accumulation window, 3 ranks, gradients summed as `_apply_step` sums them:

| | frozen readout (`zbbci6`) | **trainable readout (`ntwjfp`)** |
|---|---|---|
| ‖g_PAE‖ | 4.41e-01 | **2.59e+00** |
| ‖g_rank‖ at α=1 | 8.61e-03 | 1.04e-02 |
| ratio | 0.0195 | **0.0040** |
| **cos(g_PAE, g_rank)** | −0.15 (oppose) | **+0.30 (agree)** |
| parity α | 51.25 → used 50 | 250.1 → **used 250** |

Rule fixed before the number was seen: use parity unless cos < −0.5.

---

## THE CONTROL IS NOT `8eiek8`

`8eiek8` differs from `ntwjfp` in **both** the ranking term and the trainable set. The nearest
control is **`ggfhj5+zcuiab`** — the same five tensors minus `pae_ln`, 26,624 params, no ranking
term, best val10 2.7510.

---

## All arms

Budget 30, 18 interfaces. "best" is the best epoch of that arm, which flatters arms that ran longer.

| rid | arm | trainable | eps | best val10 | @ep | end | best ρ | best top-1 | best rankloss |
|---|---|---|---|---|---|---|---|---|---|
| `ggfhj5+zcuiab` | pose-focused | 26,624 | 21 | 2.7510 | 15 | 2.7521 | 0.2752 | 0.3444 | −0.00253 |
| `eewbr5` | LoRA r=8 | 286,600 eff | 30 | 2.7057 | 13 | 2.7929 | **0.3066** | 0.3685 | **−0.00810** |
| `mdv9x7` | full head | 19,752,270 | 30 | **2.6881** | 12 | 2.8182 | 0.2674 | 0.3667 | −0.00730 |
| `8eiek8` | pose-share | 10,240 | 50 | 2.7690 | 42 | 2.8047 | 0.2800 | 0.3407 | −0.00360 |
| `4x8t44` | pose-share + L2-SP 1e-2 | 10,240 | 50 | 2.8145 | 34 | 2.8476 | 0.2690 | 0.3481 | −0.00253 |
| `wpg274` | full head + L2-SP 1e-5 | 19,752,270 | 50 | 2.6912 | 22 | 3.0990 | 0.2675 | **0.3759** | −0.00762 |
| `zbbci6` | pose-share + rank α50 | 10,240 | 16 (stopped) | 2.8252 | 14 | 2.8291 | 0.2558 | 0.3259 | −0.00204 |
| `ntwjfp` | + trainable readout, α250 | 27,136 | running | | | | | | |
| shipped | — | — | — | 2.9756 | — | — | 0.1958 | 0.3037 | +0.00022 |

### At matched epoch 16 — the honest comparison

`zbbci6` only reached 16, so the "best" column above is not a fair read.

| rid | val10 | rankloss | ρ | top-1 | DockQ |
|---|---|---|---|---|---|
| `ggfhj5+zcuiab` | 2.7633 | −0.00154 | 0.2381 | 0.3167 | 0.2798 |
| `eewbr5` | 2.7394 | **−0.00597** | **0.2713** | **0.3444** | 0.2373 |
| `mdv9x7` | **2.6969** | −0.00353 | 0.1356 | 0.3352 | 0.2693 |
| `8eiek8` | 2.8117 | −0.00313 | 0.2558 | 0.3259 | **0.3103** |
| `4x8t44` | 2.8286 | −0.00216 | 0.2625 | 0.3296 | 0.2858 |
| `wpg274` | 2.7222 | −0.00205 | 0.1412 | 0.3278 | 0.2889 |
| **`zbbci6`** | 2.8291 | **−0.00127** | 0.2433 | **0.2981** | 0.2845 |
| shipped | 2.9756 | +0.00022 | 0.1958 | 0.3037 | 0.2909 |

**The negative result that motivated `ntwjfp`.** At matched epoch 16 the one arm with an explicit
ranking loss had the **worst held-out rank loss of all seven** and the **only top-1 below the
shipped head**. Training the ordering objective directly, with the readout frozen, produced worse
ordering than not training it at all. That is consistent with the measured cos = −0.15: the
ranking term was fighting the PAE term through a fixed bottleneck.

---

## Metrics, and where they come from

| metric | meaning | shipped | source |
|---|---|---|---|
| val10 PAE | FT3 objective, held out | 2.9756 | `<rid>/val_loss.csv` |
| **rankloss** | the ranking objective, held out. 0 = no alignment, negative better | +0.000218 | `<rid>/rank_rho_b30.csv` |
| **ρ** | mean within-interface Spearman(ipTM, DockQ) over ~150 poses | 0.1958 | same |
| **top-1** | per seed group of 5, is the highest-ipTM pose also highest-DockQ? Random floor **0.20** | 0.3037 | same |
| spread | within-seed ipTM std — a linear loss has no stopping point, so this can run away | — | same |
| DockQ / frac_acc | selected-pose quality, with per-interface standard errors | 0.2909 / 0.5556 | same, cross-checked against `metric_train_ep*.json` to 3e-5 |
| oracle | best pose available | 0.5968 / 1.0000 | — |

`inv_trained` (within-seed pairwise inversion rate, shipped 0.4502) is still computed and stored
but is **no longer reported** — top-1 replaced it as the headline because it states in a sentence.

### Resolution limit — read before claiming anything

Paired over the same 18 interfaces, the standard error on mean DockQ is **0.026**. The set
resolves ~0.05 DockQ and nothing smaller.

| true effect | interfaces needed |
|---|---|
| +0.10 | ~6 |
| +0.05 | ~20 ← we have 18 |
| +0.02 | ~130 |

`8eiek8`'s 50-epoch result was a mean paired difference of **+0.0003** — not "no effect", but "no
effect above the noise floor". top-1 s.e. ≈ 0.039, ρ s.e. ≈ 0.066.

**And we select on the same 18** — loss form, α, eventually the checkpoint. Even a 2σ effect found
that way is optimistically biased.

---

## Code

| file | what |
|---|---|
| [`phases/rank_loss.py`](../phases/rank_loss.py) | the ranking loss, its label-only coefficients, the ipTM computation, the chain_roles join |
| [`phases/phase_k_train.py`](../phases/phase_k_train.py) | `--rank-loss`, `--alpha-rank`, `--rank-labels`, `--grad-probe`, `--pose-perblock-share`, `--l2-sp`, `--train-only` |
| [`phases/pose_perblock.py`](../phases/pose_perblock.py) | the per-block re-injection |
| [`run_logging/rank_rho.py`](../run_logging/rank_rho.py) | all held-out diagnostics, cached per run in `<rid>/rank_rho_b30.csv` |
| [`run_logging/plot_val_panels.py`](../run_logging/plot_val_panels.py) | the 8-panel figure (6 held out + 2 marked `[TRAIN]`) |
| [`run_logging/plot_head_diagram.py`](../run_logging/plot_head_diagram.py) | the architecture figure |
| [`shell_scripts/probe_rank_gradients_readout.sh`](../shell_scripts/probe_rank_gradients_readout.sh) | the α probe for the current trainable set |
| [`shell_scripts/refresh_val_five.sh`](../shell_scripts/refresh_val_five.sh) | regenerates `training/runs/val_panels.png` |

Background: [`docs/ranking_loss_experiment.md`](ranking_loss_experiment.md) (the loss, the
gradient factorisation, the limitations) and [`docs/ranking_vs_pae.md`](ranking_vs_pae.md) (the
measurement that started this: PAE and ordering move independently).

## Outputs

```
training/runs/<rid>/
  val_loss.csv                 held-out PAE per epoch
  trajectory.csv               per-step train loss, grad_norm, step_taken
  rank_rho_b30.csv             all held-out diagnostics (see table above)
  metric_train_ep*.json        DockQ / frac_acc per epoch, budget-tagged
  train_eval_shards_ep*/       per-pose iptm, stored_iptm, dockq -- every diagnostic derives here
  run.log                      everything, including the per-epoch `rank-loss:` line
  resume_ep*.pt, head_*.pt     checkpoints
training/runs/val_panels.png   the 8-panel figure, this arm only
training/runs/head_diagram.png the architecture figure
```

## Run dirs to ignore

`o7u2ih` `kbkj9v` `jjvdhk` `m8rote` `a4afho` `wd656y` `xtq4vh` `fdvu8v` `s7dh7o` — failed launches
or abandoned attempts. **Keep**: `ppxxgm` and `qiodxz` (the two successful gradient probes, no
checkpoints), `zbbci6` (stopped at epoch 16), `ocqtyt` (full set, stopped at `resume_ep26.pt`).

---

## Open, not started

1. **An unbiased number.** `validation/fold` holds **92 interfaces over 53 structures**, already
   folded and scored — against the 18 in use. Scoring the chosen checkpoint once on all 92 takes
   the paired s.e. from 0.026 to ~0.011 and carries no selection bias, since the checkpoint is
   chosen on the 18. Nothing is built for it. **This is the highest-value next step.**
2. **Cross-seed ranking pairs.** The linear loss makes these *free* — coefficients are label-only,
   so poses never need co-scoring and no trunks need to be co-resident. An earlier claim that this
   needed several 200 MB trunks was **wrong** and applied only to the softplus version. Note it is
   a different objective, not a bigger one: cross-seed `c_i` reduces to "this pose's DockQ minus
   the interface mean", and ipTM is already 82.3% between-seed while DockQ is 54.6% pose-level.
3. **`--frame-direction` (pathway C).** The only experiment that gives the head orientation. In
   every arm above, the pose is `cdist` over one representative atom and nothing more. If the
   deficit is orientation, none of this can reach it. Still unrun.
4. A scaled logistic loss, if rankloss improves while top-1 does not — that pattern means margin
   expansion rather than discrimination. Unscaled softplus was measured inert: gradient 0.495 for a
   correct pair against 0.505 for an inverted one at the real margin scale (median |ipTM gap|
   0.0081).


---

## THE LINEAR LOSS IS UNSOUND

`L = -sum (s_i - s_j)(d_i - d_j)/sum|dd|` is **linear in the scores**, so it collapses to
`L = sum_i c_i s_i` with `c_i` from the DockQ labels alone. That is why it needed no scoring pass,
and it is also the defect: a gradient that ignores the current scores is **constant** in them, so
the loss has no stopping point and cannot distinguish an already-correct pair from an inverted one
when deciding how hard to push.

Scaling every score by `lambda` scales `L` by `lambda`, so **widening the spread pays without
limit** -- and it pays far better than reordering. At the real within-seed gap scale, saturating
the spread to [0,1] is worth **~71x** more loss reduction than going from fully-inverted to
fully-correct order. Dropping the DockQ weighting to pure `sign` does not help: sign only changes
the constants, and spread still pays exactly proportionally.

Two fair qualifications, neither of which rescues it: the constancy is with respect to the SCORES,
not the parameters, so `d/dtheta` can still change or vanish; and 71x is specific to that
score-range comparison. The argument survives both.

**`ntwjfp` is the measured confirmation.** 8 epochs, alpha_rank 250, trainable readout:

| ep | val10 | held-out rankloss | rho | top-1 | spread | grad_norm |
|---|---|---|---|---|---|---|
| 1 | 2.9370 | +0.00018 | 0.1574 | 0.2444 | 0.0114 | 12.3 |
| 2 | 2.9762 | +0.00058 | 0.1503 | 0.2352 | 0.0202 | 7.0 |
| 3 | 3.1286 | +0.00031 | 0.1473 | 0.2407 | 0.0175 | 15.1 |
| 4 | 3.0255 | **-0.00066** | 0.1637 | 0.2444 | 0.0200 | 20.5 |
| 5 | 3.0483 | +0.00236 | **0.0496** | 0.2481 | 0.0318 | 22.7 |
| 6 | 3.2679 | -0.00136 | 0.1379 | 0.2444 | 0.0241 | 42.7 |
| 7 | 3.0574 | -0.00136 | 0.1730 | 0.2537 | 0.0320 | 15.1 |
| 8 | **3.3985** | **+0.00288** | 0.0725 | 0.2204 | **0.0371** | 37.0 |

Spread roughly tripled while held-out rank loss went the WRONG way and `rho` twice fell below the
0.08 watch line. Never beat shipped on any metric. Compare Arm 1 (`ggfhj5+zcuiab`, the same
tensors minus `pae_ln`, **no ranking term at all**), which had rankloss -0.00178 at epoch 1 and
rho 0.24-0.28 throughout -- the arm without a ranking loss ranked better than the arm with one.

Caveat that still holds: `ntwjfp` changed three things at once against Arm 1 (`pae_ln` trainable,
the ranking term, and per-block pose sharing), so the divergence cannot be attributed to any single
one. Two of the three are exonerated elsewhere -- per-block sharing (`8eiek8`, `zbbci6`) and a
ranking term at alpha 50 (`zbbci6`) both had val10 FALLING. `pae_ln` has never been tested alone.

---

## THE REPLACEMENT: SCALED SOFTPLUS OVER CROSS-SEED PAIRS

    m_ij = sign(d_i - d_j)(s_i - s_j)
    L    = (2 tau / W) sum_ij w_ij softplus(-m_ij / tau),   w_ij = |d_i - d_j|,  W = sum w_ij

Per-pair gradient weight is `sigmoid(-m/tau)`: ~1 when badly inverted, 1/2 at a tie, decaying to 0
once correct by a few tau. So spreading buys a bounded amount and then stops, while every inverted
pair keeps generating full-strength gradient until fixed.

**tau = 0.0305, measured not chosen** (`run_logging/measure_tau.py`). Median |ipTM gap| over
cross-seed pairs of the full training set: 497 structures, 1,028 interfaces, 1,056,035 pairs after
dropping DockQ ties. The within-seed median is 0.0081 and does NOT transfer -- cross-seed pairs are
3.9x wider. For reference the 20-structure overfit30 subset gives 0.0429 and val10 gives 0.0313;
the training number is the right one, since tau shapes the training gradient.

At that tau, inverted pairs are 43.7% of pairs **by count** and take **81.6% of the pairwise
score-space gradient mass**, against **40.4%** under the DockQ-weighted linear loss -- a
redistribution of **+41.3 points**. That redirection is the whole point.

| objective | inverted pairs' share of score-space gradient mass |
|---|---|
| DockQ-weighted LINEAR (the baseline that ran) | 40.4% |
| DockQ-weighted SOFTPLUS, tau=0.0305 | **81.6%** |
| unweighted LINEAR | 44.7% |
| unweighted SOFTPLUS | 72.7% |

Two things that is NOT, both of which an earlier version of this note got wrong and which a
blocking review of `018b9e50` caught. It is **not** the parameter gradient -- per-pair mass is
`2|dd| sigmoid(-m/tau)/W` for the softplus and `|dd|/W` for the linear form, with parameter
Jacobians and cross-pose cancellation excluded (a probe once measured 65x cancellation between
per-pose and accumulated norms). And the baseline is **not** the raw inversion count: with the gap
as the weight the linear loss does not give every pair equal gradient magnitude, so 43.7% is the
right comparison only for `--no-rank-weight-by-gap`. The first published figures, 43.7% -> 71.3%,
summed bare sigmoids and compared against the count fraction; both were wrong, and the corrected
redistribution is larger, not smaller.

THE POPULATION IS PART OF THE NUMBER. All figures above: rows deduplicated by (seed, sample),
CROSS-SEED pairs only, pairs with |dDockQ| < 1e-6 dropped, interfaces with at least 10 surviving
pairs kept -- 1,027 interfaces, 1,056,035 pairs. The 0.0315 over 1,154,400 pairs quoted earlier
predates the DockQ-tie filter and should not be used.

ONE GLOBAL TAU. Per-interface median gap spans ~1,700x, which sounds fatal but is not: wide
interfaces have their CORRECT pairs saturate (right -- already ordered) while their inversions
still push, and tight interfaces look like ties only because the head has no opinion there. A
per-interface tau would amplify hardest where there is least signal.

THE 2*tau PREFACTOR reproduces the linear loss's score-space gradient AT A TIED PAIR exactly, so a
form change is not also a silent ~33x learning-rate change. Asserted in `test_rank_loss.py`.

**Cross-seed grouping.** Deployment selects argmax over ALL poses of an interface, so cross-seed
ordering is the objective. `--rank-cross-seed` regroups the epoch so all seeds of a structure share
one step. Measured agreement between a subsample's gradient direction and the full group's, cosine
over shared poses (400 interfaces, 8 draws each):

| group | cross-seed pairs | mean | median | p10 |
|---|---|---|---|---|
| 3 seeds (15 poses) | 75 | 0.856 | 0.949 | **0.544** |
| 5 seeds (25 poses) | 250 | 0.925 | 0.981 | 0.823 |
| 10 seeds (50 poses) | 1,125 | exact | | |

Taking all 10 seeds costs nothing in batch size: S=2 structures x 10 seeds x 5 poses = **100 poses
per step**, against the ~100 the normal schedule already used. It trades structures per step (2
instead of ~6) for an exact ranking signal. No seed sits out, where the normal schedule drops one
per structure per epoch.

THE SHORT RANK ROTATES. 20 units over 3 ranks is 7/7/6, and a permanently-short rank would take a
different number of steps and HANG on the first mismatched all-reduce. The extra units go to ranks
`(k .. k+rem-1) mod world` at step k, which gives every rank 100 units over the 15 steps. When the
arithmetic does not come out even the existing cross-rank count check refuses to start.

### Implemented, NOT launched

| file | what |
|---|---|
| [`run_logging/measure_tau.py`](../run_logging/measure_tau.py) | the tau measurement, with `--results` / `--restrict-to` / `--tau` |
| [`phases/rank_loss.py`](../phases/rank_loss.py) | `pair_terms_softplus`, `coeffs_softplus`, `group_coeffs` (cross-seed) |
| [`phases/test_rank_loss.py`](../phases/test_rank_loss.py) | 10 property groups, all passing |
| [`phases/phase_k_train.py`](../phases/phase_k_train.py) | `--rank-form`, `--rank-tau`, `--rank-cross-seed`, `--rank-structs-per-step`, `build_epoch_cross_seed` |

### Still to build

1. Pass A: score every pose of the step under no_grad, then `all_gather` the scalars so every rank
   holds the whole group. Each unit's ~205 MB trunk is read twice per step, the second from page
   cache; the alternative, holding 20 trunks resident, is worse for RSS.
2. The quota-driven step boundary, replacing `n_in_accum >= accum_local`.
3. Coefficient lookup keyed `(pdb, seed, sample, pair)` at the per-pose site.
4. `_rank_div` -> normalise per interface-group, not per unit.
5. `_rank_check_coverage` for cross-seed groups -- this plus `_rank_key` is where every ranking bug
   so far has lived, so it gets re-verified rather than edited.
6. The gradient probe (to set alpha) and the launcher.

### The three arms to compare, on identical architecture and trainable set

A. PAE only.  B. PAE + cross-seed softplus.  C. cross-seed softplus only.

Arm A is nearly `ggfhj5+zcuiab` but that arm had no per-block pose sharing, so a clean three-way
comparison means re-running A too. Evaluate all three on global Spearman, within-seed top-1,
cross-seed ordering, selected-pose DockQ and held-out PAE.
