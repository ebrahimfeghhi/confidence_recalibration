# Why the ranking loss stops moving the weights: measured

Reproduce with:

```
cd /home/ebrahim/confidence_recalibration/extra_training
./shell_scripts/probe_alignment_trend.sh <rid> <ep> [<ep> ...]
```

One `--grad-probe` per checkpoint. Each resumes, takes one accumulation window, prints the
numbers below and exits without writing a checkpoint, so the source run resumes afterwards
untouched.

## The question

`zfahpb`'s parameter gradient fell **8×** over six epochs, 0.0280 → 0.0033, while on val10:

- **40.1% of cross-seed pairs were still inverted** (shipped head: 42.3%), so the loss had not
  run out of mistakes;
- the mean per-pair gradient weight `sigma(-m/tau)` had **risen**, 0.2977 → 0.3465;
- total coefficient magnitude `sum|c_i|` had **risen**, 8,354 → 9,669.

I concluded the 40,960 parameters "cannot express the correction the loss is asking for". That
was an overclaim: a shrinking parameter gradient has two causes and I had measured neither.

| | mechanism | fix implied |
|---|---|---|
| **cancellation** | per-pose gradients keep magnitude but oppose each other and destroy themselves when summed | change the objective's grouping, or the reduction |
| **sensitivity** | poses individually stop producing gradient, because the trainable parameters no longer move ipTM | change what is trainable, or where the pose enters |

## The measurement

```
alignment = ||sum_i g_i|| / sum_i ||g_i||
```

`sum_i ||g_i||` is the total magnitude **before** cancellation; `||sum_i g_i||` is what the
optimiser receives. **Read it against `1/sqrt(N)`, not against 1** — N independent directions
already give ~`1/sqrt(N)`, so a value near that baseline is not evidence of anything and only a
**fall across checkpoints** is. The probe prints the baseline on the same line.

```
alignment falls while sum_i ||g_i|| holds up  ->  cancellation
sum_i ||g_i|| itself collapses                ->  sensitivity
```

## Result: sensitivity, not developing cancellation

`zfahpb`, probes at epochs 1 / 3 / 6, 8 trainable tensors, 40,960 parameters,
`1/sqrt(N)` ~ 0.098:

| epoch | **ranking** alignment | `sum_i ||g_i||` | `||sum_i g_i||` | **PAE** alignment |
|---|---|---|---|---|
| 1 | 0.0935 | 0.1496 | 0.0140 | 0.7662 |
| 3 | 0.0723 | 0.0390 | 0.0028 | 0.9594 |
| 6 | 0.1134 | 0.0509 | 0.0058 | 0.9607 |

The ranking alignment sits **at the independent-direction baseline at every epoch and does not
fall** — it ends higher than it started. What changed is `sum_i ||g_i||`, down ~3× from epoch 1.
So the collapse is **lost sensitivity**, and the cancellation, while severe, is not what
developed.

## The finding worth keeping: the ranking loss pays a constant ~10× cancellation tax

At the same weights, in the same accumulation window, PAE's per-pose gradients are **nearly
parallel** (0.96) while the ranking loss's are **as good as mutually independent** (0.07–0.11,
against a 0.098 baseline). The ranking objective therefore destroys roughly **91% of its own
gradient magnitude** every step, and PAE destroys almost none.

That is structural, not incidental: the ranking coefficients **sum to zero within each
interface group** by construction — raising one pose's ipTM means lowering another's — so the
per-pose contributions are built to oppose each other. PAE has no such constraint; every pose's
PAE gradient points toward "predict this pose's error better", and those agree.

It compounds the separate problem already measured in `h7msge`: pairs involving the selected
pose are only **2.29%** of the pairwise loss. So the ranking objective spends 97.7% of its
signal on pairs that cannot change selection, and then loses 91% of what remains to
cancellation.

## What this does not establish

- The probes are one accumulation window at one parameter state each, on the **training** set.
- The val10 inversion and `sum|c_i|` figures are **held-out**, so they speak to the loss rather
  than exactly to the training gradient that stalled.
- "Lost sensitivity" names the symptom, not the cause. It does not say *why* the parameters stop
  moving ipTM — the frozen readout (`pae_ln` and `linear_no_bias_pae`) is the obvious shared
  suspect, since the plateau lands at 0.038–0.042 for both the 10,240 single-site and the 40,960
  per-block parameter sets, but that has not been tested.

## Provenance

Probes `7boqww` (ep1), `29h75f` (ep3), `sjxrh2` (ep6), each 1.2 min, all exit 0, each noting in
`resume_drift.json` why it ran at `alpha_pae 1` and without `--l2-sp`: measuring both terms is
free and `|g_rank|` does not depend on alpha_pae, while omitting the anchor makes the reported
gradient the **pure task gradient**, which is what the ratio needs. The run was then resumed as
`zfahp2` from `resume_ep7.pt` with the anchor copied across (`anchor READ from`, not
`SNAPSHOT at init`).
