# Within-seed vs across-seed ranking, paired inside each chain pair

`run_logging/paired_interface_slopes.py`, run 2026-11-26 on
`training/fold/abag_all_results.csv` (antibody-to-antigen chain pairs only, original 5 poses per
seed, **no seed cap**). 2,260 chain pairs read, **1,530 used** (730 single-seed chain pairs have no
across-seed cell and are dropped), **95,003,339 pose pairs** built.

## Why this measurement replaces the pooled regression

`run_logging/pair_gap_regression.py` fits one within slope and one across slope over ~11M pairs
with no chain-pair fixed effect. Pairs are built inside a single chain pair, so interface
*difficulty* cannot separate the two populations — but interface *weight* can. Inside a chain pair
with `S` seeds there are `10S` within-seed pose pairs and `~12.5S²` across-seed ones, so a heavily
sampled chain pair dominates the across estimate far more than the within estimate; if the
ipTM-vs-DockQ slope varies by chain pair, the pooled difference is part real and part reweighting
and the fit cannot say which.

Measured size of that leak on the master CSV:

| dataset | per-pair across:within ratio (min / med / max) | top decile by seed count holds |
|---|---|---|
| pxm | 23.75 / 23.75 / 23.75 | 9.9% of within, 9.9% of across |
| foldbench | 3.75 / 23.75 / 23.75 | 9.7% / 9.8% |
| sabdab | 0.00 / 17.50 / 23.75 | 19.2% / 24.1% |

PXM is immune by construction. SAbDab is not: single-seed chain pairs (ratio 0.00) contribute
within pairs and no across pairs at all.

**The design.** One number per chain pair per cell, then a paired t-test across chain pairs. Each
chain pair is its own control and counts once regardless of seed count; a chain pair enters a row
only if it has both cells in that row. Composition and weighting drop out algebraically rather
than being adjusted for.

**Gap-matched by stratum.** The two cells have different gap distributions inside a chain pair too,
and the relation is curved, so a slope fitted on small gaps differs from one fitted on large gaps
with no within/across effect present. Both cells are restricted to the same `|ΔDockQ|` stratum
before anything is compared.

**Per-cell statistics.** Through-origin slope `b = Σ(d·y) / Σ(d²)` of the signed ipTM gap `y` on
the DockQ gap `d`, pairs oriented so `d > 0`; and `P(correct) = mean 1[y>0]` with an ipTM tie
counting 0.5.

## Result

The effect is real and it survives the pairing — composition was not the explanation. But it is
**narrower than the pooled fit implied**: it is a large-gap phenomenon, and the unstratified `ALL`
row is inflated by gap skew in every dataset.

Slope (signed ΔipTM per unit DockQ gap), within vs across, by stratum:

| stratum | foldbench | pxm | sabdab train | sabdab val83 |
|---|---|---|---|---|
| [0.00,0.02) | 0.212 / 0.344 | 1.171 / 2.021 | 0.007 / 0.107 | −0.190 / −0.255 |
| [0.02,0.05) | −0.075 / −0.185 | 0.131 / 0.063 | 0.020 / 0.128 | −0.009 / 0.159 |
| [0.05,0.10) | −0.027 / 0.145 | 0.049 / 0.089 | 0.022 / 0.154 | −0.049 / 0.046 |
| [0.10,0.20) | 0.033 / 0.277 | 0.070 / 0.092 | 0.011 / 0.205 | −0.020 / 0.162 |
| [0.20,0.40) | 0.047 / 0.334 | 0.061 / 0.166 | 0.014 / 0.156 | 0.053 / 0.212 |
| **[0.40,1.01)** | **0.052 / 0.441** (t −10.50) | **0.121 / 0.354** (t −5.84) | **0.055 / 0.255** (t −12.31) | **0.056 / 0.314** (t −4.97) |

Within a seed, ipTM is essentially **flat against DockQ even across a 0.4–1.0 quality gap**
(slope 0.05–0.12), while across seeds the same gap moves ipTM by 0.25–0.44 per unit. That is the
finding, and it is the same in all four pools.

`P(correct)`, within − across:

| stratum | foldbench | pxm | sabdab train | sabdab val83 |
|---|---|---|---|---|
| [0.00,0.02) | −0.0024 | **+0.0107** (t +4.20) | −0.0067 | −0.0002 |
| [0.02,0.05) | +0.0065 | **+0.0271** (t +3.95) | +0.0008 | +0.0028 |
| [0.05,0.10) | −0.0145 | **+0.0325** (t +3.04) | +0.0003 | +0.0049 |
| [0.10,0.20) | −0.0314 | **+0.0382** (t +2.52) | −0.0109 | +0.0190 |
| [0.20,0.40) | −0.0592 (t −3.06) | +0.0182 | −0.0438 (t −3.22) | −0.0135 |
| [0.40,1.01) | −0.1344 (t −4.73) | −0.0778 (t −2.71) | −0.1010 (t −5.76) | −0.1274 (t −2.53) |

Two things to read here. First, below a 0.2 gap the within-seed *decision* is no worse than the
across-seed one anywhere except PXM, where it is reliably **better** — four consecutive strata,
all significant, the same anomaly the per-dataset regressions showed, now paired and gap-matched.
Second, the `ALL` rows (sabdab train −0.0699, t −14.65; foldbench −0.0711, t −7.38) are larger
than every stratum below 0.20, which is the residual-gap-skew artifact made visible: pooled over
gaps, most of the unstratified difference is the gap distribution rather than seed sharing.

`foldbench extra` (28 chain pairs, the regenerated top-up) is a consistent null throughout and is
underpowered rather than contradictory.

## Where the pose information goes

`run_logging/pose_injection_ablation.py`, 10 chain pairs × 1 seed × 5 poses, each seed chosen so
its 5 poses span near-0 to near-1 DockQ. The 3D pose reaches `ConfidenceHead` through exactly two
layers — `linear_no_bias_d` (one-hot distance bins) and `linear_no_bias_d_wo_onehot` (raw
distance), `confidence.py:279-305` — and ipTM is a function of the PAE probabilities plus static
masks, so zeroing both makes ipTM pose-independent **by construction**. Confirmed: within-seed
pose-free ipTM spread `0.000e+00` on all 10 units, so no other pose path exists.

The pose channel is therefore measured as `delta_i = ipTM(pose i, full) − ipTM(pose-free)`:

- mean `max|delta|` per unit **0.2377**, against a full-head within-seed ipTM spread of **0.1498** —
  the geometry is a **dominant** input whose effects largely cancel, not an ignored one
- within-seed pairwise accuracy, full head **0.5139**; mean Spearman(ipTM, DockQ) **+0.057**
- 4 of 10 units strongly anti-correlated (3b9k −0.975, 8ut2 −0.872, 2p42 −0.667)

So the head reads the pose hard and reads it wrong, which is consistent with the flat within-seed
slopes above.

**Caveat, unresolved.** The full-head pass was meant to reproduce the stored ipTM and does not:
max `|recomputed − stored|` **0.2814**, mean 0.0521, against the 2e-3 the cache path claims. This
does not touch `delta` (computed entirely inside one recomputation) or the magnitude conclusion,
but it puts the *direction* numbers in doubt — the likely cause is the pose→sample join via the
labels npz `sample_index`, i.e. DockQ attached to the wrong pose. The fix is to assign poses to
samples by optimal matching on `chain_pair_iptm`, as `phase_j_rescore_cache.py:545-560` does.

## The top stratum WAS skewed — matched pair by pair, 2026-11-26

`run_logging/matched_gap_paired.py`. The `[0.40,1.01)` stratum is 0.61 wide, and neither statistic
above is immune to where inside it a cell sits: `P(correct)` has no gap normalisation at all, and
the through-origin slope evaluates to `B + G·Σd³/Σd²` under `y = B·d + G·d²`, which depends on the
gap distribution — and the pooled fit put a large positive `G` on this relation. **The skew is
real.** Unmatched median gap inside that stratum, within vs across:

| dataset | within | across |
|---|---|---|
| foldbench | 0.525 | 0.642 |
| pxm | 0.594 | 0.654 |
| sabdab train | 0.582 | 0.663 |
| sabdab val83 | 0.495 | 0.764 |

So each within-seed pair is matched to the across-seed pairs of the **same chain pair** whose gap
is within ±0.01, and the raw signed ΔipTM is compared with no normalisation (the gap is matched,
so there is nothing left to normalise). Audit: mean gap within/across agree to four decimals
(e.g. foldbench 0.5404 / 0.5406).

**The ΔipTM response survives intact.** Mean signed ΔipTM, within vs across, at matched gap:

| stratum | sabdab train | foldbench | pxm | val83 |
|---|---|---|---|---|
| [0.00,0.02) | 0.0000 / 0.0005 | 0.0006 / 0.0008 | 0.0020 / 0.0026 | −0.0006 / 0.0007 |
| [0.05,0.10) | 0.0007 / 0.0099 (t −3.8) | −0.0023 / 0.0065 | 0.0038 / 0.0061 | −0.0044 / 0.0036 |
| [0.10,0.20) | 0.0006 / 0.0227 (t −6.2) | 0.0044 / 0.0383 (t −4.1) | 0.0099 / 0.0072 | −0.0023 / 0.0191 |
| [0.20,0.40) | 0.0050 / 0.0383 (t −8.2) | 0.0133 / 0.0851 (t −5.4) | 0.0151 / 0.0351 (t −2.1) | 0.0150 / 0.0476 (t −2.8) |
| **[0.40,1.01)** | **0.0273 / 0.1123** t −10.9, p 5e-24 | **0.0349 / 0.1981** t −9.0, p 7e-14 | **0.0677 / 0.1678** t −5.9, p 5e-08 | **0.0309 / 0.1449** t −5.0 |

For the *same* DockQ gap, within-seed ipTM moves **2.5–5× less** than across-seed ipTM, with a
clean monotone ladder in sabdab train (−0.0005, −0.0022, −0.0092, −0.0221, −0.0333, −0.0850).

**`P(correct)` does NOT survive intact — much of it was the skew.** Within − across after matching,
against the binned value it replaces:

| dataset, [0.40,1.01) | binned | matched |
|---|---|---|
| sabdab train | −0.1010 (t −5.76) | **−0.0582** (t −3.70) |
| foldbench | −0.1344 (t −4.73) | −0.0454 (t −2.06, marginal) |
| pxm | −0.0778 (t −2.71) | **−0.0129 (t −0.56, gone)** |
| sabdab val83 | −0.1274 (t −2.53) | −0.0858 (t −2.09, marginal) |

So the correct claim is about **sensitivity, not accuracy**: within a seed ipTM barely responds to
real quality differences, but at a matched gap it does not reliably pick the wrong pose more often
than an across-seed comparison does, except in sabdab train (−0.058) where the effect is real but
half the size previously reported. The earlier `P(correct)` numbers in this document are superseded
by the matched ones.

**PXM's reversal strengthens under matching** — within-seed decisions are *better*, four strata,
all significant: +0.0118 (t 4.60), +0.0261 (t 3.69), +0.0318 (t 3.03), +0.0574 (t 3.88) at
[0.00,0.02) through [0.10,0.20).
