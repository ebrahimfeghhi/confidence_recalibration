# paper_results.md

Results worth keeping, with enough provenance to recompute each one. Numbers here are copied from
a command that is named beside them; if a number has no command, it is not in this file.

**Every selected-pose number here uses phase_k_eval's rules**, and nothing else is allowed in:
`max(v, key=(iptm, -seed, -sample))` for the pick and `dockq >= 0.23` for acceptability. The
tie-break is not cosmetic -- the evaluator's own comment records that without it, "two poses with
equal ipTM selected different DockQ depending on how the work was split". On 2026-10-27
`train_eval_metrics.py` was found using `max(key=iptm)` and `> 0.23` instead, which moved ocqtyt's
shipped baseline by 0.0003 and its frac by one interface in 975. Fixed, and every number below was
regenerated.

**Two distinctions that every number in here depends on.**

1. **val10 (18 interfaces) and valfull (92 interfaces) are never comparable.** Different pools,
   different oracles. Any table mixing them is wrong.
2. **Budget 30 (150 poses) and budget 10 (50 poses) are not comparable either.** The validation
   fold was folded at 30 seeds x 5 samples; the training fold only at 10 seeds x 5. So any
   training-set number is budget 10 and must be read against a budget-10 validation number, which
   `run_logging/train_eval_metrics.py` produces by re-aggregating existing valfull rows.

---

## 1. The headline in-sample result: a PAE objective does drive pose selection

**`jy6sah`, the overfit30 chain** (`pykssn -> x3888d -> tgfiqx -> 89uwjx -> owdyyp -> hhzddi ->
jy6sah`), 138 epochs, full confidence head, no L2, 30 structures. Scored **in-sample** on the same
30 structures, 77 interfaces, budget 10.

| | epoch 2 | epoch 136 | shipped | oracle |
|---|---|---|---|---|
| mean DockQ | 0.1135 | **0.2386** | 0.0420 | 0.5098 |
| fraction DockQ > 0.23 | 0.1299 | **0.3896** | **0.0000** | — |
| training PAE loss | ~2.9 | **1.3438** (min, ep135) | — | — |

paired dDockQ vs shipped on the same rows: **+0.1965 +- 0.0306** (t ~ +6.4).
**Captures 42% of the headroom (0.0420 -> 0.5098), and that is a FLOOR, not a ceiling** -- the
chain was stopped at epoch 136 with training PAE still falling and both DockQ and frac still
rising. Shipped puts **0 of 77** interfaces above 0.23; the trained head puts **39%** above it.
Against the ~2% of headroom the full-training-set arms capture, this is a 20x difference.

**Why it matters:** DockQ and frac rise monotonically with training PAE across the whole 138
epochs -- not a late jump -- so the PAE objective genuinely converts into pose selection. This is
the strongest evidence against the "PAE is the wrong objective" reading, and it is what makes
cross-complex transfer, not objective design, the live problem.

**Scope:** in-sample, 30 structures. It establishes what the objective can do when it has enough
signal per complex; it does not yet show the same holds across complexes. Running this chain
further would very likely push all three numbers higher, since nothing had converged.

    python -I run_logging/plot_overfit30.py jy6sah figures/overfit30_chain.png

---

## 1b. ...and it does NOT transfer. The same checkpoint is WORSE than shipped on valfull

**The problem is cross-complex generalisation.** `jy6sah/head_ep136_step2448.pt` -- the exact
checkpoint that produced the result above -- scored on valfull's 92 held-out interfaces at budget
30, sharded over 3 cards, 9.1 min:

| | jy6sah | shipped | delta |
|---|---|---|---|
| mean DockQ | **0.5476** | 0.5831 | **-0.0355** |
| fraction DockQ > 0.23 | **0.6957** | 0.7391 | **-0.0434** (4 fewer interfaces of 92) |
| oracle | 0.6926 | | |

Same checkpoint, two pools:

| | in-sample (overfit30, 77 ifaces) | held out (valfull, 92 ifaces) |
|---|---|---|
| paired dDockQ vs shipped | **+0.1965 +- 0.0306** | **-0.0355** |
| frac vs shipped | 0.0000 -> 0.3896 | 0.7391 -> 0.6957 |

So a head that captures 42% of in-sample headroom **destroys** 0.036 DockQ out of sample -- about
15x larger in magnitude than the best full-training-set arm's gain (+0.0024). Training the PAE
objective to convergence on a complex makes selection much better ON THAT COMPLEX and worse
everywhere else.

**This had never been measured.** The whole overfit30 chain ran with `val_loss_pool=None` and zero
validation eval directories across all 7 runs -- it only ever scored itself on its own 30
structures. The transfer failure was assumed, not known, until 2026-10-27.

Scope: 30 training structures is the most extreme version of the gap. It does not predict what 487
structures do -- that is section 2's question.

Join was clean: 13,800 of 13,800 intended keys, 0 duplicates, 0 unresolvable, 0 non-finite.

    LAYERNORM_TYPE=openfold python -I phases/phase_k_eval_sharded.py \
      --cards 4,5,7 --ckpt training/runs/jy6sah/head_ep136_step2448.pt --budget 30 \
      --work training/runs/jy6sah/valfull_eval --out training/runs/jy6sah/metric_valfull.json \
      --extra "--fold validation/fold --results validation/fold/results.csv \
               --pool training/valfull/pool.csv --roles validation/fold/chain_roles.csv \
               --seed-base 400000 --join-cache training/valfull/join_cache.json \
               --pair-iptm-only --prefetch"

---

## 2. Capacity is not the limit on the full training set

Training PAE loss, final value, all arms on `full_train` (487-488 structures), same objective,
same fold, same crop -- **training curves are comparable across these; held-out curves are not**
(`ocqtyt` monitored val10, the rest valfull).

| run | trainable | pose layers | L2-SP | train PAE start -> end | held-out |
|---|---|---|---|---|---|
| `ip8ga4` | 10,240 | share | none | 2.7693 -> 2.6443 | 2.8305 -> 2.7923 (valfull) |
| `dntxmn` | 40,960 | copy | none | 2.7654 -> 2.6256 | 2.8283 -> 2.7770 (valfull) |
| `tv6hue` | 19.8M | copy | 1e-3 | 2.7005 -> 2.6095 | 2.8207 -> 2.7444 (valfull) |
| `jer222` | 19.8M | copy | 1e-2 | 2.7243 -> 2.6596 (7 ep) | 2.9191 -> 2.7501 (valfull) |
| **`ocqtyt`** | **19.8M** | **none** | **none** | **2.7384 -> 2.1007** | 2.6953 -> **3.0841** (val10) |

The unconstrained full head reaches **2.10** while every constrained arm plateaus near **2.61**.
So the arms are not capacity-limited. `ocqtyt`'s held-out loss *rose* 2.6953 -> 3.0841 over the
same run: textbook overfitting.

    python -I run_logging/plot_train_loss.py figures/train_loss.png

---

## 2b. THE FULL HEAD DOES NOT FIT 487 STRUCTURES IN SELECTION TERMS

`ocqtyt/head_ep26_step6620.pt` -- full head, no L2, no pose injection, training PAE driven 2.7384
-> 2.1007 -- scored on **its own 975 training interfaces** at budget 10:

| | trained | shipped | delta |
|---|---|---|---|
| mean DockQ | 0.4214 | 0.4202 | **+0.0012 +- 0.0042** |
| fraction DockQ >= 0.23 | 0.5436 | 0.5344 | +0.0092 |
| oracle | 0.5489 | | headroom 0.1275 |

**0.9% of the in-sample headroom, and indistinguishable from zero.** The PAE loss fell by 0.64 and
the pose selection did not move.

Set against the only other in-sample measurement we have:

| run | structures | passes each | in-sample paired dDockQ |
|---|---|---|---|
| `jy6sah` (overfit30) | 30 | ~136 | **+0.1965 +- 0.0306** |
| `ocqtyt` (full train) | 487 | 26 | **+0.0012 +- 0.0042** |

So at 487 structures the head is not failing to TRANSFER -- it is not fitting the training set at
all, in the only sense the benchmark measures. This is a different failure from the one section 1b
documents, and it constrains how section 1 may be read: the overfit30 conversion from PAE to pose
selection is a 30-structure, 136-pass phenomenon, not a general property of the objective.

**The live test:** `ocqtyt` resumed 26 -> 50 on 2026-10-27, which roughly doubles passes per
complex, the axis separating those two rows. If the in-sample gain appears by epoch 50 it is a
capacity/passes story; if it stays flat while PAE keeps falling, PAE and selection decouple at
scale and the 30-structure result says little about 487.

    python -I run_logging/train_eval_metrics.py ocqtyt --no-run

---

## 3. What the full-training-set arms actually deliver on valfull

Measured as each arm's **level over all 20 epochs**, not its best epoch -- picking the max of 20
noisy draws inflates an apparent effect by roughly the per-epoch spread.

| arm | mean DockQ (20 ep) | vs shipped 0.5830 | t | frac above/equal/below shipped |
|---|---|---|---|---|
| `ip8ga4` share | 0.5854 | **+0.0024** | +4.11 | 3 / 0 / 17 |
| `dntxmn` copy | 0.5842 | +0.0012 | +1.69 | 1 / 0 / 19 |
| `tv6hue` full+L2 1e-3 | 0.5725 | -0.0105 | -7.58 | 2 / 9 / 9 |

The best arm is **reliably** above shipped and the effect is **+0.0024 DockQ against 0.1096 of
headroom -- about 2%**. No arm moved `frac_acceptable`: averaged over 20 epochs they get -0.1 to
-0.6 interfaces of 92 right. Real, far too small to matter.

Top-1 rate (the pipeline already tracks it): shipped picks the best pose on **26.45%** of
interfaces; the arms average 26.9-27.2%. Essentially unmoved.

---

## 4. Budget 10 vs budget 30 on valfull: the arms help MORE with fewer poses

Same arms, same final checkpoints, valfull rows re-aggregated.

| arm | paired dDockQ @ b30 (150 poses) | paired dDockQ @ b10 (50 poses) |
|---|---|---|
| `ip8ga4` | +0.0073 +- 0.0040 | **+0.0118 +- 0.0051** |
| `dntxmn` | +0.0029 +- 0.0032 | +0.0075 +- 0.0053 |
| `tv6hue` | -0.0084 +- 0.0054 | +0.0020 +- 0.0055 |

shipped at b30 0.5831 / oracle 0.6926; at b10 0.5763 / oracle 0.6504.

    python -I run_logging/train_eval_metrics.py ip8ga4 dntxmn tv6hue --no-run

---

## 5. rho is a poor proxy for the metric, and ipTM is worse than its rho implies

Simulated rankers on valfull's actual eval rows: rank poses by **true DockQ plus Gaussian noise**,
sweep the noise, measure mean within-interface Spearman and the resulting selected DockQ.

| ranker rho | selected DockQ | vs shipped |
|---|---|---|
| 0.175 | 0.5895 | +0.0054 |
| **0.218** | **0.6043** | **+0.0202** |
| 0.259 | 0.6279 | +0.0438 |
| 0.319 | 0.6408 | +0.0567 |

**Shipped ipTM has rho 0.2301 and scores 0.5841.** An unbiased ranker at *lower* rho (0.218)
scores 0.6043 -- **+0.020 better DockQ at worse rank correlation**. ipTM's errors are concentrated
at the top of the list, which is the only place the metric looks.

Consequence: our arms moved rho 0.2301 -> ~0.255, which on this curve should be worth ~+0.024
DockQ. They delivered **+0.0024, one tenth of it**. Improving mean rank correlation is not the
same as improving the top pick.

Caveat: this shows rho and PAE are weak proxies for top-1 selection. Section 1 shows a PAE
objective nonetheless converts into selection when pushed far enough in-sample. Both are true.

---

## 6. Where valfull's headroom lives

977-interface analysis of the shipped head's misses on valfull (92 interfaces, 150 poses each):

- total headroom (oracle - shipped) = 0.1085 per interface on average
- shipped already picks the best pose on **1 of 92** interfaces
- **47 of 92** interfaces have more than 0.05 of headroom
- the top 20 interfaces (22%) hold **67%** of all headroom; half the interfaces hold 90%

So the task is not hopeless-by-concentration: broad headroom exists.

---

## 7. ABAG-Rank: a direct-DockQ reranker does not beat ipTM IN-SAMPLE either

Sections 1b and 2b are about a PAE objective that fits in-sample and then fails to transfer.
ABAG-Rank -- a 5.2M-parameter Deep Set over inter-chain residue pairs, regressing DockQ directly
with a soft-Spearman and distance-preservation loss -- fails one step earlier. It never fits the
selection objective at all.

Paired on the SAME K=10 candidate poses per interface, model top-1 against the ipTM top-1 it is
handed as an input feature:

| model | scored on | ipTM top-1 | model top-1 | delta |
|---|---|---|---|---|
| run_7030, epoch 220 | held-out 30% of training structures, 299 interfaces | 0.4252 | 0.4034 | **-0.0218** |
| run_7030, epoch 220 | its own fitted 70%, 676 interfaces | 0.4126 | 0.4133 | **+0.0006** |
| run_full, epoch 505 | its own fitted 975 interfaces | 0.4117 | 0.4020 | **-0.0097** |

run_7030 held out 146 of 487 structures at random (`abag_rank/run_7030.sh`, seed 0, split by
structure, membership in `split.json`). The held-out set is NOT the cluster-distance regime valfull
sits in: 104 of the 146 share a SAbDab2 `ab_ag_cluster` with a training structure and 74 share an
antigen cluster. So "our splits are too hard" does not explain this.

**The held-out number is not the result.** The in-sample delta is zero, so there is nothing for the
held-out set to fail to inherit, and the honest statement is that the method does not work rather
than that it does not generalise. Supporting detail:

- The model is given ipTM as a sample-level feature and scores BELOW it. Passing one number
  through would have matched it.
- Held-out soft Spearman +0.77, and top-1 still loses. This is section 5's rho-is-not-the-metric
  finding in a far starker form than the confidence-head arms showed it.
- Train 1.0394 vs val 1.1357 at epoch 220: almost no gap. 5.2M parameters on 676 interfaces for
  220 epochs is UNDERfitting, not memorising.

Taken with sections 1b and 2b: two methods with different supervision (PAE vs DockQ), capacity
(19.8M vs 5.2M) and architecture now sit within ~0.01 of ipTM on our antibody-antigen interfaces
while their own training objectives improve substantially.

Method notes, numbers from `abag_rank/val_baseline.py`, full version in `abag_rank/RESULTS.md`:
budget 10, not 30, so read against the local oracle 0.4847-0.4933 and not against the valfull
figures above; every baseline paired on the same candidates by iterating the training loop's own
DataLoader, asserted by reproducing its pinned oracle 0.4933 to four decimals; the computed
held-out 0.4034 equals the logged epoch-220 value exactly; interfaces are in-sample but the exact
10-pose subsets are not guaranteed to be those seen in training, so the claim is interface-level.

## 8. Mean-centred iTM regression: the ranking metric moves and selection does NOT follow

The loss was built against a measured gap. On `d32kqp` epoch 66 over the 738 winnable interfaces,
`corr(interface MEAN predicted ipTM, interface MEAN DockQ) = +0.866` while the median WITHIN-interface
Spearman against DockQ was only +0.2994 (mean-pooled), against +0.6898 for the TRUE iTM. The head had
learned which complexes are easy and not which pose is best, and only the second is used in selection.
Centring both prediction and target on the interface mean removes the across-complex signal from the
objective, so the within-complex ordering is the only thing left to earn:

    L = PAE + 75 * (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2

**IT WORKED ON ITS OWN METRIC.** Over 27 epochs (`o2qczq`, beta 75, mean pooling, 368 winnable
structures) the in-sample median within-interface Spearman went **+0.2994 -> +0.5392**, and the
predicted-score spread ratio went 0.433 -> 0.867 against the true iTM's. That is 43% of the distance
from the shipped head to the true-iTM ceiling, on the exact quantity the loss targets.

**IT DID NOT CONVERT INTO SELECTION, IN-SAMPLE, AT ALL.** Scored at epoch 27 on the SAME 738
interfaces it trains on, at budget 10 (= all 50 training poses per interface), 25.4 min on 3 cards:

| selector | mean DockQ | frac acc | vs shipped | % headroom |
|---|---|---|---|---|
| shipped ipTM | 0.5474 | 0.7060 | -- | -- |
| oracle | 0.7015 | 0.9363 | +0.1541 | 100% |
| rel ep27, MEAN-selected | 0.5387 | 0.6992 | **-0.0087** | **-5.6%** |
| rel ep27, MAX-selected | 0.5466 | 0.7127 | -0.0008 | -0.5% |

Set against the other arms scored on this identical pool and budget (all MAX-selected):

| arm | epoch | mean DockQ | vs shipped | % headroom |
|---|---|---|---|---|
| lineage | 55 | 0.5546 | +0.0072 | +4.7% |
| row-weighted | 55 | 0.5609 | +0.0135 | +8.8% |
| row-weighted | 66 | 0.5614 | +0.0140 | +9.1% |
| **rel (mean-centred)** | **27** | **0.5466** | **-0.0008** | **-0.5%** |

**NOT EPOCH-MATCHED, and that is the one real caveat**: 27 against 55 and 66. Some of the gap to the
other arms could be training time. The sign is not a matter of training time, though -- this arm is
BELOW the shipped head it started from, on its own training data, having improved the ranking
correlation by +0.24.

WHAT THAT RULES OUT. "The head cannot represent within-complex pose quality" is dead -- it can, and
this loss teaches it. What does not follow is the step from a better median rank correlation to a
better argmax. Selection reads ONE pose per interface; a loss that improves the ordering across all
50 can do so by sorting the poses nobody picks. The held-out numbers over the same stretch say the
same thing from the other side: held-out within-interface Spearman went +0.2629 (ep5) -> +0.1377
(ep25), BELOW the shipped head's +0.1709, while the in-sample version climbed.

THE MEASUREMENT THAT MATTERS NEXT is therefore not another ranking metric. It is whether any loss
of this family moves the TOP of each interface's distribution rather than its middle.

### 8b. WHERE the ranking gain sits: among the poses nobody picks

`run_logging/rank_gain_location.py` splits the same in-sample rows the -0.0087 came from. "The top"
is the 10 poses with the highest TRUE DockQ -- the only ones a selector could usefully order:

| selector | rho all 50 | rho top 10 | rank of best pose | picks best | best in top 5 |
|---|---|---|---|---|---|
| ours, MEAN-pooled | +0.2819 | **+0.0326** | 18.0 / 50 | 3.79% | 20.7% |
| ours, MAX-pooled | +0.2857 | **+0.0667** | 17.0 / 50 | 4.74% | 22.0% |
| shipped ipTM (MAX) | +0.1931 | **-0.0246** | 21.0 / 50 | 3.98% | 18.2% |
| random | 0 | 0 | 25.5 / 50 | 2.00% | 10.0% |

**The +0.089 of Spearman is real and it is entirely outside the top.** Restricted to the ten poses
worth picking, every selector is uncorrelated with DockQ: +0.03, +0.07, -0.02. Spearman over 50
poses is dominated by the ~40 nobody would select, so sorting those correctly moves the statistic
and cannot move the argmax. At the top, where it would pay, the arm picks the best pose 3.79% of
the time against the shipped head's 3.98% -- both a whisker above the 2.00% of random choice.

That also says the TARGET shares the defect, not just the learned approximation: the true iTM this
loss regresses onto ranks the whole field far better than it ranks the top, so a perfect fit to it
would inherit the same blindness. Section 5's "rho is a poor proxy for the metric" is the same
finding one level up.

TIES ARE WHY THIS NEEDED CARE. stored_iptm is recorded at 4 decimal places, so the rank of the best
pose and the top-1 rate must be computed from counts of strictly-better and tied poses, not from a
sort: resolving ties by row order moved the shipped head's median rank between 21 and 22 and its
top-1 rate between 3.98% and 4.3% depending only on the glob order of three shard files.

### 8c. THE IN-SAMPLE EVAL IS UNCROPPED AND THE ARM TRAINED CROPPED

`phases/phase_k_eval.py` has no crop code at all -- no flag, no path -- so every number in sections
8 and 8b was measured on FULL structures. The trainer crops every pose:

  199,800 pose-crop records over epochs 1-27, 100.0% marked cropped
  tokens, full structure   median 617    mean 649    max 1543
  tokens, kept after crop  median 333    mean 331    max  801
  kept fraction            median 0.534, and 1.0 on 0.0% of poses

**The head has never once seen a full structure during this training.** "In-sample" in section 8
means the STRUCTURES are the training structures; it does not mean the INPUTS are the training
inputs, and on this arm they are not -- the median pose arrives at the head with 47% of its tokens
removed.

WHAT THAT DOES AND DOES NOT CHANGE.

It does NOT rescue the selection result as a statement about USEFULNESS. The goal is DockQ on full
antibody-antigen complexes, inference runs uncropped, and uncropped is where the arm is -0.0087
against the shipped head. A method that only helps on cropped inputs helps nobody.

It DOES undercut the result as a statement about LEARNING. The comparison in section 8 is
ours-uncropped against shipped-uncropped, which sounds paired and is not symmetric: the shipped
head's recorded ipTM comes from the ordinary full-structure pipeline, so it is being read on its
own distribution while this arm is read off its. The gap between +0.5392 cropped and +0.2819
uncropped for the SAME checkpoint is the size of that asymmetry, and it is larger than every
effect in sections 8 and 8b put together.

So "the loss raises the ranking metric and does not convert into selection" is established on
uncropped inputs only, and the honest form of section 8's conclusion is: ON THE INPUTS IT WILL
ACTUALLY BE USED WITH, this arm is not better than the shipped head, and we do not yet know whether
that is because the loss teaches nothing useful or because the arm cannot be read off its training
distribution.

THE MEASUREMENT THAT SEPARATES THEM is the shipped head's CROPPED within-interface Spearman on the
same cropped inputs -- one loss-only pass through the trainer, no optimiser steps. If cropping
lifts both heads equally, the cropped curve is a measurement artefact and section 8 stands as
written. If it lifts this arm specifically, the arm learned something crop-dependent and the
train/eval input mismatch is a CAUSE of the non-transfer rather than a footnote on it.

A CAVEAT ON THE PER-EPOCH SPEARMAN CURVE. The in-sample diagnostic printed every epoch (+0.2994 ->
+0.5392) is computed inside the training loop on CROPPED inputs, crop 3400 with a 10-residue
interface margin. This eval is uncropped. Same checkpoint, same metric, same poses: +0.5392 cropped
against +0.2819 uncropped. The curve therefore overstates the arm's ranking gain, and the defensible
figure for it is the +0.089 over the shipped head measured here.

## 9. MATCHED CROPPING: the arm is +0.0733 in-sample, not -0.0087

Section 8 scored a cropped-trained head on full structures and section 8c said why that is not a
fair reading. `phases/phase_k_eval.py` now crops (`--crop-margin`), calling the same two library
functions the trainer calls, with every setting -- margin, contact, all-poses, the pose pathways,
the pooling -- read from the scored checkpoint's own guard. `run_logging/test_eval_crop.py`
reproduces the trainer's recorded `kept_tok` exactly on 25 of 25 sampled units.

FOUR PASSES, because the rows' `stored_iptm` is the shipped head's RECORDED ipTM from the
full-structure pipeline; using it as the cropped baseline would rebuild the very asymmetry being
removed. The shipped head gets its own cropped pass.

**IN-SAMPLE, 738 winnable interfaces, budget 10 (= all 50 training poses), both heads at 50.9% of
tokens kept, epoch 30 of the beta-75 arm:**

| head | input | mean DockQ | frac acc |
|---|---|---|---|
| shipped | uncropped (recorded) | 0.5474 | 0.7060 |
| shipped | **cropped** | 0.5572 | 0.7209 |
| ours ep30 | uncropped | 0.5387 | 0.6992 |
| ours ep30 | **cropped** | **0.6305** | **0.8523** |
| oracle | -- | 0.7015 | 0.9363 |

**Matched cropped, ours minus shipped: +0.0733 mean DockQ and +0.1314 frac acceptable.** The same
checkpoint read uncropped is -0.0087. Cropping does flatter the shipped head, by +0.0098, so that
confound is real and it is small; it does not account for ours moving +0.0918 on the identical
interfaces and candidates.

AND THE GAIN IS NOW AT THE TOP OF THE LIST, which is what section 8b said it was not:

| | uncropped | cropped |
|---|---|---|
| Spearman, all 50 poses | +0.2819 | +0.5810 |
| Spearman, top 10 by true DockQ | +0.0326 | **+0.2040** |
| picks the truly best pose | 3.79% | **15.99%** |
| best pose inside its top 5 | 20.7% | **41.7%** |

(random picks best 2.00%). The cropped +0.5810 also lands beside the training loop's own +0.5593
at this epoch, so the evaluator's crop reproduces the trainer's in effect and not only in token
counts.

SO SECTION 8's CONCLUSION WAS AN ARTEFACT OF THE INPUT MISMATCH, not a property of the loss. The
mean-centred iTM objective taught the head a great deal; reading it on full structures threw that
away. Section 8b's "the gain is entirely among poses nobody picks" was measuring the same artefact.

### 9b. HELD OUT, MATCHED CROP: Spearman AND selected DockQ both improve

valfull, 92 interfaces, budget 30 (150 candidates), both heads cropped identically, paired per
interface:

| selector | mean DockQ | frac acc | rho | vs shipped | t |
|---|---|---|---|---|---|
| shipped, MAX-pooled | 0.5802 | 0.7283 | +0.2645 | -- | -- |
| shipped, MEAN-pooled | 0.5865 | 0.7391 | +0.3119 | -- | -- |
| ours ep30, MAX-pooled | **0.6178** | **0.7826** | **+0.4017** | **+0.0376** | **+2.86** |
| ours ep30, MEAN-pooled | 0.6028 | 0.7717 | +0.3492 | +0.0164 | +1.19 |

Within-interface Spearman, ours minus shipped, paired over the 92 interfaces:
**MAX +0.0845 (t=+4.10)**, MEAN +0.0390 (t=+1.97).

**Both of the things that had to move, moved.** Held-out ranking improves and held-out selection
improves with it, which is what every previous arm here failed to do -- the recorded finals are
ip8ga4 +0.0073, dntxmn +0.0029, tv6hue -0.0084 at this budget.

CROPPING DOES NOT FLATTER THE SHIPPED HEAD HERE. Its cropped 0.5802/0.5865 sits on top of its
uncropped 0.5832, i.e. it gains nothing from the crop on valfull, while this arm goes 0.5897 ->
0.6178 under MAX and 0.5836 -> 0.6028 under MEAN. The effect is specific to the trained head.

MAX BEATS MEAN HELD OUT, for this arm, although the loss optimises the MEAN: 0.6178 against
0.6028, and +0.0845 against +0.0390 on Spearman. In-sample MEAN is (slightly) ahead, 0.6305 to
0.6261. Not explained; it is the opposite of the ceiling argument the arm was configured on
(mean-pooled true iTM reaches 67.7% of oracle headroom against max's 58.6%) and it should not be
designed around until it is understood.

### 9c. THE CROP IS DEFINED BY GROUND TRUTH, so this is not a deployable configuration

`interface_crop_tokens` finds interface tokens "by ground-truth coordinates" -- it reads
`label_coordinate` from the label npz. The window handed to the head is therefore chosen using the
answer. At inference there is no label, so **no number in sections 9 and 9b is a deployment
number**, including the held-out ones.

What they ARE is a clean head-to-head: both heads got the identical oracle-informed window, the
shipped head gained nothing from it on valfull, and this arm gained +0.0376. That isolates what
the loss taught from the input mismatch that was hiding it, which is exactly what sections 8-8c
could not separate.

WHAT WOULD MAKE IT DEPLOYABLE is a crop that does not read the label -- the obvious candidate
being the PREDICTED pose's own inter-chain contacts, which are available at inference. Whether the
gain survives that substitution is unmeasured and is the next thing to measure. "Train uncropped"
is the other direction and does not have this problem at all.

WHAT THIS DOES NOT YET SHOW. These are the structures the arm trained on, so memorisation is
available to it, and the held-out pair (valfull, budget 30, both heads cropped) is the measurement
that matters. It also does NOT make the arm useful as it stands: deployment is uncropped, and
uncropped it is not better than shipped. What it argues for is training uncropped, or otherwise
closing the train/eval input gap, rather than abandoning the objective.


## Open questions as of 2026-10-27

- **2026-10-30 (section 9): sections 8 and 8b were confounded by the train/eval input
  mismatch.** Matched cropping moves the same epoch-30 checkpoint from -0.0087 to
  +0.0733 in-sample and moves the top-10 Spearman from +0.03 to +0.20. The held-out
  matched pair is running; it is the number that decides whether to train uncropped.

- **ANSWERED 2026-10-30 (section 8): a within-interface ranking loss can move the
  ranking metric a long way and still not move selection, in-sample.** +0.24 of median
  within-interface Spearman bought -0.0087 of in-sample mean DockQ. Rank correlation over
  all 50 poses is not the quantity selection reads.

- **ANSWERED 2026-10-27 (section 1b): the in-sample conversion does NOT transfer at 30
  structures -- it inverts, to -0.0355 on valfull.** The live question is whether 487 structures
  change that; `ocqtyt` scored on the training pool (budget 10) tests whether the fitting even
  happens at that scale.
- **ANSWERED 2026-10-27 (section 7): a direct-DockQ reranker does not beat ipTM even in-sample**,
  so for ABAG-Rank the question of cross-complex transfer does not arise yet.
- `jy6sah` was stopped, not converged -- both curves still climbing at epoch 136.
- No run exists that is full head + 4 pose-copy layers + no L2 on the full training set. `ocqtyt`
  is the closest and differs only by lacking the pose layers (plus 9dzj, 1 structure of 488).
