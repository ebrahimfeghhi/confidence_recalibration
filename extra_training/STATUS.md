# STATUS — what is running, what exists but is off, what is planned

Regenerated 2026-11-09, source `b70fa96b`. If this disagrees with a run's own `args.json`, the
`args.json` is right and this file is stale.

## 1. LIVE

**`87ja6a`**, launched 10:17 AM EST 2026-11-09 on cards 4/5/7, 20 epochs, lr 1e-3.

| | |
|---|---|
| trainable | **105 tensors / 5,358,000 params** — the whole confidence head except the 75 never-called LEAN_SINGLE tensors and the 81 behind the 10 certified-dead pair modules |
| pose enters | the **four pairformer block inputs**: distance as an untied copy of block 0's pretrained injection, plus a zero-init orientation term at each |
| checkpointing | off at or below **1,250 tokens** (measured, see below); above that, on |
| pool | **845** structures (8 skipped; 3 of them for the memory cap) |
| prefix cache | **off** — no frozen prefix exists once all four blocks train |
| eval | held-out 83-structure fold at epochs 5 / 10 / 15 / 20 |
| pace | ~95 min/epoch, worst peak 81.88 GB of 94.97 |

## 2. EXISTS, BUT OFF IN THE LIVE ARM

| code | why off |
|---|---|
| `pose_readout.py` (site 5, after `pae_ln`) | removed at the owner's direction after `73vv52`. `--pose-readout-copy-distance` copies two arms and only the one-hot arm is bounded; the raw-distance arm is `||w|| * d` with d in angstroms — 474% of the normalised signal at 5 A, 1,895% at 20 A. `73vv52` opened at PAE **46.27** over its first 23 steps against a normal 2.95. **If reinstated: copy the one-hot arm only, leave raw-distance at zero.** |
| `pose_inblock.py` (block 3 only) | superseded. It was the previous arm (`peae97`, 20 trainable tensors). Mutually exclusive with `--pose-perblock`. |
| `prefix_cache.py` | caches a frozen prefix; there is no frozen prefix now. |
| `lora.py`, `itm_quality_loss.py` | dropped by the owner; slated for removal. |

## 3. MEASURED, NOT FITTED

Memory, one rank, the live config, checkpointing OFF:

| tokens | pass A | pass B |
|---|---|---|
| 1,004 | 13.95 GB | 56.81 GB OK |
| 1,254 | 20.78 GB | **87.62 GB OK** |
| 1,499 | 28.93 GB | OOM |
| 1,822 | 41.91 GB | OOM |

The old fit `peak = 2.02 + 2.896e-05 * tokens^2` predicted 67 GB at 1,499 where the truth was an
OOM, so it does **not** transfer to this architecture. Runtime token counts equal the manifest's
`tokens` column, so 1,250 is a true ceiling.

Epoch time, `73vv52` epoch 1, 94.7 min: GPU-busy 86.9 min = **92%**. Largest host bucket is
`trunk_h2d` at 10.1 min.

## 4. THE ORGANIZED TRAINER IS **NOT RUNNABLE**

`phases/trainer/` holds all 8,402 lines of `main()`, extracted into 12 modules, and 8 CPU tests
of which 7 pass. **It has never executed a forward pass and cannot complete a step.**

**All 8 CPU tests now pass, including the context contract: every `rt.<attr>` read has a
writer.** It still has NOT executed a forward pass -- that is the next gate, not a passing
suite. What follows is the history of how far off the earlier claims were.

The contract test used to fail: **99 of the 115 `rt.<attr>` reads were written by no stage** -- `TERMS`, the module handles (`BC`, `PFC`,
`DOPS`, `RC`), the caches and the per-run callbacks. Those live in the ~1,300 lines of `main()`
*between* the extracted regions, which have not been extracted.

The other seven tests pass on this state. That is precisely why none of them may be read as
evidence that it runs: parsing, name resolution and extraction fidelity are all true of a
skeleton. I reported this package ready twice before this was measured.

## 5. PLANNED, NOT WRITTEN

`phases/trainer/` holds `cli`, `config`, `checkpoints`, `step_plan` and 4 passing CPU tests.
Still to port out of `phase_k_train.py`'s 8,402-line `main()`: `bootstrap` (~823),
`unit` (~721), `loop` (~977), `evaluate` (~690). **Nothing in `trainer/` runs the experiment.**
