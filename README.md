# tts_abag

Benchmarking pipeline for **test-time scaling (TTS)** on the FoldBench
antibody–antigen task, across two structure-prediction models: **ESMFold2-Fast**
and **Protenix-v2**.

The pipeline is: **run** a model over many seeds → **eval** every diffusion
sample with OST/DockQ → **plot** the scaling curves and per-interface results.
The eval/plot/analysis code is model-agnostic — it reads a generic
`<complex>/seed_N/predictions/` layout, so it works on either model's output.

## ⚠️ Where things go — read this before creating any file

**Everything this project produces lives inside this repo.** Scripts, configs, docs,
notes, run output, vendored third-party code, handoff bundles. Do not create project
files at `/home/ebrahim/` top level or leave them in `/tmp`.

**Large outputs go in the repo too — and get gitignored.** "Too big to track" is not a
reason to store something outside the repo. Put it in the right directory, add a
`.gitignore` entry, and note it. Currently gitignored:

| path | size | what |
|---|---|---|
| `confident_decoy_experiments/artifacts/` | ~91 GB | scored results, OST caches, sweep output |
| `confident_decoy_experiments/predictions/` | ~12 GB | prediction trees |

Moving data here is cheap: on one filesystem `mv` is an instant metadata operation even
for 100 GB (`df --output=source` to check). **Leave a symlink at the old path** so
absolute paths baked into running jobs, old logs and configs keep resolving —
`/home/ebrahim/confident_decoy_experiments/{artifacts,predictions}` are symlinks here for
exactly that reason, and the move was done while a job was actively writing.

Paths are centralized — edit [`paths.py`](paths.py) and
[`confident_decoy_experiments/paths.py`](confident_decoy_experiments/paths.py) (or set the
matching env vars) rather than hardcoding in individual scripts.

**Working across machines:** a fresh clone has code and docs but none of the gitignored
output. Small metadata needed to run elsewhere is committed under
`confident_decoy_experiments/handoff/` — see
[`docs/HANDOFF_nomsa_split.md`](confident_decoy_experiments/docs/HANDOFF_nomsa_split.md)
for the pattern. Checkpoints and the model repos are external and not tracked.

> **Steering experiment:** the active work is label-free **distogram-repulsion steering**
> in [`steer/`](steer/README.md) — push ESMFold2 out of confident *wrong* antibody–antigen
> orientations without using the ground truth. Superseded steering/analysis experiments
> (including the earlier ConforNet line) are archived under `archive/`.

## ⚠️ READ THIS BEFORE RUNNING ANYTHING — the benchmark is mis-configured

**Every ESMFold2 number in this repo was produced by `biohub/ESMFold2-Fast` run SINGLE-SEQUENCE.**
That is a handicapped setup, and it invalidates the comparison against Protenix that most of the
analysis rests on:

- `msa_encoder.enabled = **false**` in the ESMFold2-Fast checkpoint. Base `biohub/ESMFold2` has it
  **true** (already cached).
- [`archive/run/esm/run_foldbench_abag.py`](archive/run/esm/run_foldbench_abag.py) never passes an MSA — and at
  `:46-48` it explicitly *filters out* the MSA-bearing JSONs.
- Protenix uses MSAs (10k–16k sequences).

**And running single-sequence is what CAUSES the confident-decoy trap this project is built on.**
Confidence discriminates well across the pool and then **inverts inside its own top decile** —
exactly where `argmax` reads. Measured (OST-scored, identical seeds and complexes, 100-pose pools;
δ = Cliff's delta of confidence between acceptable and bad poses; **negative at the top = the
disease**):

| | δ, all poses | **δ, top quartile** | selector | **trap rate** |
|---|:--:|:--:|:--:|:--:|
| ESMFold2-**Fast**, no MSA — *what this repo benchmarks* | +0.359 | **−0.139** | 40% | **60%** |
| base ESMFold2, no MSA | +0.209 | **−0.200** | 25% | **60%** |
| base ESMFold2, **with MSA** | +0.464 | **+0.327** | **62%** | **33%** |
| Protenix, with MSA | +0.623 | **+0.568** | 86% | **13%** |

**Both no-MSA arms invert. Both with-MSA arms are healthy. An MSA cures it** — selector 25% → 62%
on the *same* checkpoint (**+38%, 95% CI [+15%, +62%], P = 0.998**), and it **halves the trap rate**
(60% → 33%), curing 44% of traps outright.

**Every headline number in this repo is inflated by the missing MSA** — including the 47-interface
confident-decoy set and the 6/22 clustering result. **Re-baseline on base ESMFold2 + MSA before
doing any more method work.**

→ **[`docs/overnight_2026_07_14_summary.md`](docs/overnight_2026_07_14_summary.md)** — read first.
→ [`docs/confidence_head_hypothesis.md`](docs/confidence_head_hypothesis.md) — the full study.

**The one finding that survives, and is worth more than the hypothesis was:** a confident decoy and
a confident truth are **structurally indistinguishable** (same cohesion, same basin count, same
concentration — only correctness differs). **So no structure-only, label-free selector can separate
them.** That is why every clustering method tried here hits the same ceiling — a *perfect* oracle
clustering scores 3/22, *worse* than the real method's 6/22. Stop tuning τ.

## DockQ: there are exactly TWO code paths. Do not add a third.

Every DockQ bug this project has ever had was in **residue correspondence** — never in the
DockQ math. So the two jobs are split, and only one of them does the hard part:

| job | tool | why |
|---|---|---|
| **GROUND TRUTH** — a prediction vs the **crystal structure** | [`eval/eval_ost_foldbench.py`](eval/eval_ost_foldbench.py) → **OST** | Correspondence is **hard**: the crystal has unresolved loops, antibody chains use non-sequential Kabat numbering, chain names differ (crystal `H/L/A` vs model `A/B/C`), and the model predicts residues the crystal lacks. OST solves this, and it is what FoldBench uses. **Score every model with it — ESMFold2, Protenix, anything new.** |
| **CLUSTERING** — a prediction vs **another prediction** | [`steer/experiments/dockq_torch.py`](steer/experiments/dockq_torch.py) → our GPU implementation | Correspondence is **free**: two predictions of the same sequence by the same model have identical residues in identical order. No alignment, so none of the above can go wrong. **Verified to reproduce OST exactly** (diffs ≤ 0.0005 = OST's 3-decimal rounding), and the reference DockQ python package on 7,354,375 pairs. |

`dockq_torch.py` is the **only** file in the repo containing the DockQ formula. Everything else
either calls OST or reads a `dockq` column from a CSV. **Do not reimplement pose-vs-crystal DockQ**
— that path was tried, and deleted.

**Traps, all of which have bitten us:**
1. `chain1`/`chain2` in every scoring CSV are **native** chain ids, not model ones. The
   `chain_mapping` column translates, and on multi-copy assemblies it **changes per pose**.
2. Native chains are named by **`label_asym_id`**, not `auth_asym_id` (8ts0's heavy chain is auth
   `H` but label `A`).
3. **`results/csv_files/Protenix_dockq.csv` is WRONG — do not use it.** It never came from OST.
   Use `results/csv_files/dockq_ost_protenix.csv`.

Full detail, including the verification numbers: [`docs/dockq_ground_truth_and_gpu.md`](docs/dockq_ground_truth_and_gpu.md).

## Layout

```
paths.py            # single source of truth for every absolute path
archive/run/        # SUPERSEDED launchers, kept for provenance. Nothing imports them; the Protenix
                    # one predates the claim-based worker and cannot share a queue or resume.
  esm/              # ESMFold2-Fast inference (needs the esm venv)
    run_foldbench_abag.py            # fold all complexes for a set of seeds
    run_foldbench_abag_parallel.py   # fan the above out across GPUs
    capture_reps.py                 # capture baseline ESMC rep for a complex (steering prereq)
    loop_diffusion.py       # per-loop diffusion + load_complex helper (used by steer/core)
  protenix/         # Protenix-v2 inference (needs the Protenix venv + checkpoint)
    run_benchmark_parallel.py        # 113-structure benchmark, seeds split across GPUs
eval/               # model-agnostic
    eval_ost_foldbench.py            # OST/DockQ + confidence scores per sample -> CSV
plot/
    plot_scaling_ost.py              # TTS scaling curve + per-interface barplot
steer/              # label-free distogram-repulsion steering  (see steer/README.md)
    core/           # engine: forward_grad.py (differentiable trunk) + basins.py
    experiments/    # repel_proven.py (the method), decode_dockq.py (steered -> CIF -> DockQ)
results/            # default output dir for plots (git-ignored)
archive/            # superseded run/eval/plot/steer one-offs (nothing imports it)
```

## Which environment runs what

There is no single venv — each stage uses the environment that has its model /
tools installed. Scripts find `paths.py` by walking up from their own location,
so you can invoke them from anywhere; just use the right interpreter.

| Stage | Interpreter / env | Why |
|-------|-------------------|-----|
| `archive/run/esm/*` | `~/esm/.venv/bin/python` (`paths.ESM_PYTHON`) | imports the `esm` package |
| `archive/run/protenix/run_benchmark_parallel.py` | `~/Protenix/.venv/bin/python` + `~/checkpoint` | calls Protenix `runner/inference.py` |
| `eval/eval_ost_foldbench.py` | python with `tqdm` + `biotite` (else falls back to a `PROTENIX_PYTHON` subprocess); needs the `foldbench` conda env's **OST** binary (`paths.OST_BIN`) | runs `ost compare-structures` |
| `plot/*` | any python with numpy/matplotlib | pure post-processing |
| `steer/experiments/*` | `~/esm/.venv/bin/python` | imports the steering engine (`steer/core`) + `esm` |

## End-to-end example (ESMFold2 TTS)

```bash
cd ~/tts_abag

# 1) RUN — 50 seeds across 3 GPUs, 10% mask + 30% dropout (TTS settings)
~/esm/.venv/bin/python archive/run/esm/run_foldbench_abag_parallel.py \
    --run_name esmfold2fast_tts --gpus 0 1 2 \
    --seeds $(seq 0 49) --lm_mask_pct 0.1 --lm_dropout 0.3
# writes to paths.ESM_PREDICTIONS/esmfold2fast_tts/<complex>/seed_N/predictions/

# 2) EVAL — DockQ for every sample (needs OST on PATH / via paths.OST_BIN)
python eval/eval_ost_foldbench.py \
    --pred_dir ~/esm/foldbench_predictions/esmfold2fast_tts \
    --out_csv  ~/esm/foldbench_predictions/dockq_ost_esmfold2fast_tts_all_samples.csv

# 3) PLOT — scaling curve + the per-interface barplot
python plot/plot_scaling_ost.py \
    --csv ~/esm/foldbench_predictions/dockq_ost_esmfold2fast_tts_all_samples.csv \
    --out results/tts_esmfold2fast_tts.png
# -> results/tts_esmfold2fast_tts.png  and  results/tts_esmfold2fast_tts_barplot.png
```

For Protenix, replace step 1 with the parallel launcher (same seeds-axis design as
ESM — each GPU runs all structures for its share of seeds, one checkpoint load per
GPU, resume-aware):

```bash
~/Protenix/.venv/bin/python archive/run/protenix/run_benchmark_parallel.py \
    --gpus 0 1 3 4 --seeds $(seq 101 110)
# add --dry_run to preview per-GPU seeds, structure counts, and commands first
```

Then point `eval` at `paths.PROTENIX_PREDICTIONS`. Steps 2–3 are identical.

## Run configs

Every run records its exact parameters. Both launchers call `save_run_config`
(`runlog.py`), writing one JSON snapshot per invocation to
`<run_dir>/configs/<timestamp>__<uuid>.json` — params + timestamp, host, pid, and
the `tts_abag` git commit. The unique filename means concurrent GPU processes never
clobber each other (the launcher logs the run once; inner workers get
`--no_config_log`). So any prediction dir is self-describing; no external tracker
needed. (If you later want cross-run dashboards, the natural place is the eval stage,
logging config + DockQ pass-rate once results exist.)

## Overriding paths

Every constant in `paths.py` reads an env var of the same name first, e.g.:

```bash
FOLDBENCH_DIR=/data/FoldBench RESULTS_DIR=/tmp/plots \
    python plot/plot_scaling_ost.py --csv my_dockq.csv
```

Run `python paths.py` to print all resolved paths and whether each exists.
