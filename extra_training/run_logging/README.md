# run_logging

Reports for a training run, rendered from the files the trainer already writes. **No wandb, no
TensorBoard server, no account** — and no viewer instrumentation inside the trainer.

## Why it is built this way

The trainer writes plain files. Readers come and go. Coupling the trainer to a viewer — a
`SummaryWriter` call, a wandb handle — means a restart whenever the viewer changes, and the FT3
pilot spent four restarts on 2026-10-16 already. So the records are the contract:

| file | written by | one row/entry is |
|---|---|---|
| `<out>/trajectory.csv` | `phase_k_train.py`, flushed every 20 rows | one **pose**; `step_taken=1` marks the pose the optimiser stepped on |
| `<out>/metric_ep<N>.json` | the per-epoch evaluation | one **checkpoint** scored on all 5,300 validation units |
| `<out>/args.json` | the trainer at startup | every argument the run was launched with |
| `training/ft3_pilot_baseline.json` | GATE 1 | the **shipped head** scored by the same evaluator |

## Launching a run

```
python -I run_logging/launch.py --note "why this launch exists" --cards 0,1,3 -- \
    --fold training/fold --labels training/fold/labels --epochs 3 --accum 256 --lr 1e-3 ...
```

Everything after `--` is passed to `phase_k_train.py` unchanged. The launcher mints a **random run
id**, creates `training/runs/<id>/`, writes `config.json` **before** starting (so a run that dies in
its first second still says what it was), appends a row to `training/runs/index.csv`, and points
`latest` at it.

| in `training/runs/<id>/` | what it is |
|---|---|
| `config.json` | the settings, the **commit** and whether tracked files were modified, the exact command, cards, host, library versions |
| `status.json` | exit code and wall time, written after |
| `trajectory.csv` | one row per pose; the per-step series is `step_taken == 1` |
| `metric_ep<N>.json` | the validation metric per epoch |
| `run.log` | stdout of every rank |
| `head_*.pt` | checkpoints — kept because the per-epoch evaluation has to score something |

**Why a random id per launch.** Launches used to share one directory keyed on the learning rate. On
2026-10-16 four of them collided in `training/ft3_sweep/lr1e-3` and **three of the four loss curves
were lost**, because preserving them depended on renaming files by hand before each restart. A run's
identity is a launch, not a learning rate.

**`dirty` means tracked files changed**, not that untracked files exist. The repo always holds
untracked data directories, so counting those made every run dirty and the flag stopped meaning
anything. What bears on reproducibility is whether the code differs from the commit.

The commit is part of the config for a concrete reason: labels carry a fingerprint over the builder's
source, and `--allow-code-drift` lets a run proceed when that source has changed. A curve whose
commit is unknown cannot be compared with another.

## Usage

```
python -I run_logging/render_report.py --run training/ft3_sweep/lr1e-3 \
    --baseline training/ft3_pilot_baseline.json --out run_logging/reports/lr1e-3.html

python -I run_logging/render_report.py --run-id latest --out run_logging/reports/latest.html

bash run_logging/watch.sh training/runs/latest               # re-render every 5 min
```

`watch.sh` refreshes the file on disk. It does **not** publish: publishing goes through Claude's
tool, so the published page updates when Claude republishes, and the local file is at most 5 minutes
stale. The watcher exits on its own once no training process remains, after one final render.

## The three charts, and what a point is

1. **Training loss per batch.** x = optimiser step; y = the summed confidence loss. One point is one
   step, covering `--accum` poses (255 = 85 per rank × 3 ranks), **not** one pose — a per-pose curve
   is 85× denser than the thing being optimised and reads as noise. The four terms are drawn
   separately, because the total is their sum and a change in one is the interesting case. PAE is the
   one that matters: ipTM is computed entirely from PAE, and Protenix ships its weight at 0.
2. **Validation mean DockQ per epoch.** x = epoch; y = the mean over 92 interfaces of the DockQ of
   the pose the head ranked highest by per-interface ipTM, over the 100-seed budget.
3. **Validation fraction acceptable per epoch.** Same axes; y = the fraction of those interfaces
   whose selected pose reaches DockQ ≥ 0.23 (CAPRI acceptable).

Both validation charts carry two reference lines: the **shipped head** under the same evaluator, and
the **oracle** — the best pose the sampler produced, whether or not confidence found it. The gap
between them is the headroom this project exists to claim.

## The baseline is the evaluator's own number

`0.5775`, not the `0.5779` that comes from `results.csv`. `results.csv` stores ipTM rounded to four
decimals, so near-ties break differently there — measured, that moves the selection on 5 of 92
interfaces. Comparing a recomputed metric against a rounded one would bake in a 0.0003 offset that
has nothing to do with the head.

## On TensorBoard

`to_tensorboard.py` is not written yet, deliberately. The whole run is 291 optimiser steps and 6
validation points; TensorBoard needs a server plus SSH port forwarding to view, and its event files
are binary, so a reviewer reading this repo cannot open them. If live curves over SSH are wanted,
the converter reads these same records — `pip install tensorboard` plus one script — and the trainer
still does not change.
