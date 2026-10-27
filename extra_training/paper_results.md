# paper_results.md

Results worth keeping, with enough provenance to recompute each one. Numbers here are copied from
a command that is named beside them; if a number has no command, it is not in this file.

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
| `ip8ga4` | +0.0074 +- 0.0040 | **+0.0124 +- 0.0051** |
| `dntxmn` | +0.0030 +- 0.0032 | +0.0081 +- 0.0052 |
| `tv6hue` | -0.0082 +- 0.0054 | +0.0027 +- 0.0055 |

shipped at b30 0.5830 / oracle 0.6926; at b10 0.5756 / oracle 0.6504.

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

## Open questions as of 2026-10-27

- **ANSWERED 2026-10-27 (section 1b): the in-sample conversion does NOT transfer at 30
  structures -- it inverts, to -0.0355 on valfull.** The live question is whether 487 structures
  change that; `ocqtyt` scored on the training pool (budget 10) tests whether the fitting even
  happens at that scale.
- `jy6sah` was stopped, not converged -- both curves still climbing at epoch 136.
- No run exists that is full head + 4 pose-copy layers + no L2 on the full training set. `ocqtyt`
  is the closest and differs only by lacking the pose layers (plus 9dzj, 1 structure of 488).
