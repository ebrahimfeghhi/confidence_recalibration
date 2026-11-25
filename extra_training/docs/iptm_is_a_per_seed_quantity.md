# ipTM is effectively a per-seed quantity

**What this document is.** The evidence that Protenix-v2's ipTM barely distinguishes the five
diffusion samples drawn from one trunk, while ground-truth DockQ and ground-truth iTM vary a great
deal among exactly those five poses. The consequence is that extra diffusion samples per seed
generate quality the confidence head cannot spend, so at a fixed pose budget breadth beats depth.

**Reproduce everything below with**

```
python -I run_logging/within_seed_variance.py --reps 10        # sections 1-4
python -I run_logging/within_seed_iptm_dockq.py                # section 5
python -I run_logging/pose_pair_signal_sites.py --units 50     # section 6
python -I run_logging/true_pae_iptm.py --out rows.csv          # section 7
python -I run_logging/compare_true_pae_ranking.py --rows rows.csv
```

**The data.** `training/fold/results_with_trunks.csv` (predicted ipTM and ground-truth DockQ per
pose), `training/fold/itm_labels/<pdb>.npz` (ground-truth iTM per pose). Sections 1–4 restrict to
the **947 interfaces that carry all 15 seeds × 5 samples**, out of 1,688, so that a within-seed and
a between-seed term are both estimable on the same interfaces. 587 of those also carry ground-truth
iTM on ≥ 2 complete seeds (median 10 seeds; the iTM store covers seeds 500000–500009 only).

---

## 1. Inside one interface, ipTM's variance is between seeds. DockQ's is not.

Within each interface, each quantity's variance splits exactly into a **between-seed** part (the
variance of the per-seed means) and a **within-seed** part (the mean of the per-seed variances) —
exactly, because the groups are equal-sized at 5 samples per seed. The within-seed **share** is the
fraction of what the quantity does that a selector choosing among one seed's five samples can see.

| quantity | interfaces | mean within-seed share | median |
|---|---|---|---|
| ground-truth DockQ | 947 | 0.4310 | **0.4484** |
| **predicted ipTM** | 947 | 0.1477 | **0.0269** |
| ground-truth iTM (`g_max`) | 587 | 0.4607 | **0.5158** |

The typical interface puts **2.7%** of its ipTM variance among a seed's five samples and **44.8%**
of its DockQ variance there. Read the medians: ipTM's mean share of 0.148 is inflated by a
right tail of interfaces where ipTM does move within a seed, and the median is the typical case.

**The iTM row is the control that matters.** iTM is the quantity ipTM is an estimate *of*, computed
on the same poses from the deposited structure, and **51.6%** of its variance is within-seed. So
this is not a property of the ranking metric and not a DockQ-threshold artefact: the thing ipTM is
trying to predict varies within a seed about as much as DockQ does. ipTM is the only one of the
three that is flat.

## 2. ipTM picks the best of five 26.6% of the time, against 20% for a coin

Over all 14,205 (interface, seed) groups, taking ipTM's argmax among the seed's five samples
(ties broken highest score, then lowest seed, then lowest sample — `phase_k_eval.py:795`):

| | DockQ |
|---|---|
| ipTM's pick | 0.4176 |
| a blind pick (the group mean) | 0.4080 |
| the seed's best pose (oracle) | 0.4841 |

* ipTM selects the seed's best pose **26.6%** of the time against **20.0%** for a blind pick.
* It beats a blind pick by **+0.0097 DockQ**, and converts **12.7%** of the +0.0664 within-seed
  headroom.

So the head is not anti-correlated and not random — it is *weakly* right, and nearly all of the
within-seed headroom is left on the table.

## 3. The stakes are real in 22% of seeds, and in a third of those ipTM is flat

| | |
|---|---|
| seeds whose five poses span DockQ > 0.2 | 3,146 / 14,205 = **22.2%** |
| of those, seeds whose ipTM spans < 0.02 | 1,087 / 3,146 = **34.6%** |

In roughly one seed in fourteen, the five poses differ by more than 0.2 DockQ and ipTM reports
them as essentially identical. That is not a calibration error — it is a failure to resolve.

## 4. The cost: at 15 poses, breadth beats depth by +0.050 DockQ

Fifteen poses per interface, drawn three ways, averaged over 10 random draws each. Equal compute in
every row (the trunk cost differs — 15 trunks against 3 — so the 3 × 5 row is the *cheapest*, which
only strengthens the conclusion).

| allocation | ipTM's pick | oracle | blind | picks the best |
|---|---|---|---|---|
| **15 seeds × 1 sample** | **0.5345** | 0.6498 | 0.4077 | 0.1681 |
| 5 seeds × 3 samples | 0.5071 | 0.6232 | 0.4096 | 0.1187 |
| 3 seeds × 5 samples | 0.4842 | 0.6001 | 0.4105 | 0.1075 |

And at a **fixed** 15 seeds, adding the other four samples of each seed:

| | ipTM's pick | oracle |
|---|---|---|
| 15 seeds × 1 sample | 0.5360 | 0.6500 |
| 15 seeds × 5 samples | 0.5383 | **0.7066** |

**This is the headline.** Five times the diffusion compute raises the oracle by **+5.66 points**
and what we actually select by **+0.23 points**. The quality is being generated; the confidence
head cannot find it. Converting that +5.66 at ipTM's own cross-seed efficiency would be worth more
than the entire selection gain of the five-predictor ensemble we have been benchmarking against.

## 5. The within-seed response is correctly *signed* and nearly uninformative

`run_logging/within_seed_iptm_dockq.py`, mean-centring both quantities inside each
(interface, seed) group — which removes exactly the between-seed part that per-seed selection
cannot use — over **14,946 groups / 74,730 poses**:

| subset | groups | pooled Pearson | mean per-group Spearman | pairwise ranking accuracy |
|---|---|---|---|---|
| all | 14,946 | +0.1581 | +0.0811 | **0.5357** |
| train | 9,708 | +0.1909 | +0.0923 | 0.5408 |
| held out | 5,238 | +0.0978 | +0.0603 | **0.5265** |

Restricted to groups whose five poses span a large ground-truth DockQ range (pairwise accuracy; an
ipTM tie counts as half a hit, not a hit):

| minimum DockQ spread | train | held out |
|---|---|---|
| ≥ 0.2 | 0.5931 | 0.5322 |
| ≥ 0.5 | 0.6531 | 0.5387 |
| ≥ 0.695 | 0.6990 | **0.5583** |

On held-out structures ipTM orders a seed's poses correctly **52.7%** of the time against 50% for a
coin, and only **55.8%** even where DockQ spans 0.695 or more. The train/held-out gap is large and
nothing here was fitted — this is the shipped ipTM — so the leading explanation is Protenix's own
pretraining exposure, our split being temporal. **That is a hypothesis, not a result.**

## 6. It is not an information-flow problem: the pose signal survives the stack

`run_logging/pose_pair_signal_sites.py` on 50 (structure, seed) units whose five samples span
0.695–0.945 DockQ. Each pose is run through the confidence head on its own, and at each site the
antibody × antigen sub-block of the pair representation is captured. The statistic is the fraction
of that sub-block's activation energy that is pose-dependent,
`r = sqrt( mean_e Var_p(z_e) / mean_e mean_p z_e² )`, normalised by the activation magnitude at that
site so it is comparable across layers.

| site | mean r | median | vs injection |
|---|---|---|---|
| after the two distance terms | 0.0952 | 0.0882 | 1.000 |
| after Pairformer block 0 | 0.0985 | 0.0906 | 1.035 |
| after block 1 | 0.1096 | 0.1003 | 1.151 |
| after block 2 | 0.1101 | 0.1014 | 1.156 |
| after block 3 | 0.1000 | 0.0991 | 1.051 |
| PAE logits | 0.1635 | 0.1484 | **1.718** |
| token–token iTM | 0.1364 | 0.1248 | 1.432 |
| pooled ipTM | 0.1364 | 0.1074 | 1.433 |

The geometry is **not** suppressed — only 18 of 50 units lose any ground from injection to block 3,
and the PAE readout *amplifies* the pose-dependent component 1.7×. Taken with section 5, the
network responds strongly to the pose and the response carries little docking-quality
information: it is reacting to geometry changes that are largely irrelevant to docking accuracy.

## 7. Ground-truth PAE → ipTM: the prediction is the bottleneck, and the metric loses the rest

**All 7,279 units, 74,730 poses, 0 failures.** `run_logging/true_pae_iptm.py` replaces the head's
predicted PAE distribution with the **experimentally derived** aligned error (the one-hot label
`PAELoss.calculate_label` returns, from a real frame alignment of the pose onto the deposited
structure) and runs the *identical* aggregation,
`sample_confidence.calculate_chain_based_ptm` — the same call `rank_loss.iptm` makes. No model is
loaded and no network runs.

| subset | score | pooled Pearson | Spearman | pairwise | picks best | mean DockQ |
|---|---|---|---|---|---|---|
| all | predicted ipTM | +0.1581 | +0.0811 | 0.5357 | 0.2684 | 0.4229 |
| all | **ground-truth-PAE ipTM** | **+0.7362** | **+0.3858** | **0.6732** | **0.4650** | **0.4601** |
| all | oracle | | | | 1.0000 | 0.4867 |
| all | blind | | | 0.5000 | 0.2000 | 0.4136 |
| held out | predicted ipTM | +0.0978 | +0.0603 | 0.5265 | 0.2574 | 0.5358 |
| held out | **ground-truth-PAE ipTM** | **+0.7232** | **+0.3439** | **0.6534** | **0.4366** | **0.5764** |
| held out | oracle | | | | 1.0000 | 0.6059 |
| held out | blind | | | 0.5000 | 0.2000 | 0.5316 |

The predicted-ipTM row reproduces section 5 exactly (+0.1581 / +0.0811 / 0.5357), which is the
cross-check that the join is right this time.

**A perfect PAE predictor through the existing aggregation recovers 58.3% of the within-seed
headroom** — selected DockQ 0.4229 → 0.4601 against an oracle of 0.4867 — and 57.9% held out. So:

* **The prediction is the dominant problem.** Most of what ipTM is failing to do, it is failing
  because the PAE it is read from is wrong, not because the pooling is the wrong pooling.
* **But the metric is not blameless.** Even with the error known exactly, pairwise accuracy is
  0.673, not 1.0, and 42% of the headroom stays on the table. `max over anchor rows of the mean`
  genuinely discards ordering.
* **The metric generalises and the prediction does not.** Ground-truth-PAE ipTM scores 0.683
  pairwise on train and 0.653 held out — essentially flat — while predicted ipTM falls from 0.541
  to 0.527. Whatever makes the head worse on unseen structures lives in the PAE prediction.

**A bug worth recording.** The first version of this measurement reported the association as
*negative* (−0.088 pooled, 0.481 pairwise). The pose index is **not** the sample index:
`results_with_trunks.csv` is keyed by the fold's sample number while the trunk and label npz files
are in pose order, and the permutation lives in the label's `sample_index` — non-identity in 3,971
of 4,000 training label files (99.3%), e.g. `10fj/free_500000 → [0, 4, 3, 2, 1]`. Joining by
position paired each pose's true-PAE ipTM with another pose's DockQ. This is the same trap
`phase_k_train.py:7086` documents for the rel target.

---

## What this establishes, and what it does not

**Established.**

1. ipTM is effectively a per-seed quantity: a median 2.7% of its within-interface variance lies
   among a seed's five samples, against 44.8% for DockQ and 51.6% for the ground-truth iTM it
   estimates.
2. The headroom is large and unclaimed: +0.0664 DockQ per seed, of which ipTM converts 12.7%.
3. At a fixed 15-pose budget, 15 seeds × 1 beats 3 seeds × 5 by **+0.050 DockQ**, and the extra
   samples raise the oracle by +5.66 points while moving the selection by +0.23.
4. The failure is not information flow. The pose-dependent component of the pair representation
   survives all four Pairformer blocks and is amplified by the PAE readout.
5. The within-seed response is correctly signed and close to uninformative on held-out structures:
   52.7% pairwise against 50% chance.
6. A perfect PAE predictor through the same aggregation recovers **58.3%** of the within-seed
   headroom (57.9% held out) and reaches 0.673 pairwise. The prediction is the dominant problem;
   the aggregation loses the remaining 42%. The metric generalises (0.683 train / 0.653 held out)
   and the prediction does not (0.541 / 0.527).

**Not established.**

* **Why** the head responds to pose changes that do not track docking accuracy. Sections 5 and 6
  together say it does; neither says what it is responding to instead.
* Whether the train/held-out gap in section 5 is Protenix's pretraining exposure. Plausible,
  untested.
* Whether a retrained readout on the frozen final pair representation can recover DockQ — the
  frozen-representation probe is the open experiment, and sections 5–6 are the reason to run it:
  if the information is present but the wrong projection is being taken, only the readout needs
  retraining, which is far more tractable at our data scale than changing how the head processes
  geometry.
