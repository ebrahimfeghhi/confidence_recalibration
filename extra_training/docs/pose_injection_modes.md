# Pose injection: four modes, and the words to use for them

The flag names say how the weights are *stored*, not what they *do*, which is why "share" keeps
reading as "inject once and share the result". It does not. **Every mode below injects the pose
embedding at all four pairformer blocks.** They differ only in how many distinct weight sets those
four injection sites draw from.

Two numbers describe any of them: **sites × weight-sets**.

| flag | sites | weight sets | new params | step 0 vs shipped |
|---|---|---|---|---|
| *(none)* — inject once, before the stack | 1 | 1 | 0 | identical |
| `--pose-perblock` | **4** | 4 | 14,976, **zero-init** | identical (zero-init) |
| `--pose-perblock-tied` | **4** | **1** (new) | one new matrix | not identical |
| `--pose-perblock-share` | **4** | **1** (block 0's own, pretrained) | **0** | **not** identical |
| `--pose-perblock-copy` | **4** | 4 (untied, each init from block 0) | 30,720 | not identical |

## Say it this way

- `--pose-perblock-share` → **"4 sites, 1 shared weight set (block 0's pretrained)"**, or
  "4x1-shared". NOT "shared injection", which sounds like one site.
- `--pose-perblock-copy` → **"4 sites, 4 independent weight sets"**, or "4x4-untied".
- `--pose-perblock-tied` → **"4 sites, 1 new weight set"**, or "4x1-new".
- plain `--pose-perblock` → **"4 sites, 4 zero-init weight sets"**.

## Two traps the names hide

**`share` adds nothing but is not a no-op.** It re-applies `linear_no_bias_d` and
`linear_no_bias_d_wo_onehot` -- block 0's *already trained, non-zero* matrices -- at blocks 1-3,
which never had an injection before. Zero new parameters, but **step 0 is NOT the shipped head**.
The run's own log says so: "blocks 1-3 gain an injection they never had, from weights that are
already non-zero. That is inherent to this mode, not a bug." So part of any share arm's epoch-1
movement is that initial perturbation rather than training, and a share arm is **not** a clean
baseline against shipped at epoch 0.

**`copy` and `share` have identical step 0.** copy initialises each site from block 0, so against a
share arm the only variable is capacity -- 4 weight sets instead of 1. Against plain
`--pose-perblock`, which is zero-init, the variable would be capacity AND initialisation at once.

## Where this bit

4y3ve6, 8eiek8 and j583i8 are all **4 sites, 1 shared weight set**, 10,240 parameters.
24u34d, zg5fnw, 8fkrsu and hronsr are **4 sites, 4 weight sets**, 40,960+.
Describing the first group as "pose-share" repeatedly invited the reading that they inject once.
