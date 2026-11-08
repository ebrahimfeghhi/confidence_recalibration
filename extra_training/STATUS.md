# Current experiment

Everything below is read from the running process and its `args.json`, not from prose that may
have gone stale. Regenerate with `python -I phases/trainer/status.py`.

## The question

At equal or less compute than a free 50-seed budget, can we improve DockQ on FoldBench/PXM
antibody–antigen interfaces by making the confidence head rank poses better?

The head predicts ipTM; selection is an argmax over 150 poses *within* one interface. Pooled over
all poses ipTM correlates with DockQ at **+0.83**, but *within* an interface — the only comparison
the argmax makes — it is **+0.15**. That gap is the headroom.

## What is running

Two arms, identical in everything but the learning rate, so one is the other's control.

| | `qip9ta` (live) | `hv2kqa` (control, stopped at epoch 7) |
|---|---|---|
| lr | **0.003** | 0.01 |
| staircase | 20 → 1e-3, 40 → 1e-4 | same |
| epochs | 60 | 60 |
| seed / accum / nsample | 0 / 30 / 5 | same |
| loss | PAE + `rel` (relative iTM, mean-centred per interface), `alpha_rank` 75 | same |
| held out | `training/valfull83/pool.csv`, budget 30, every 5 epochs | same |

Both start fresh from the shipped `protenix-v2` head. Cards 4, 5, 7.

**Trainable: 20 tensors, 1,154,560 params — 7.3% of the live confidence head.** Block 3's
`tri_att_start` and `pair_transition`, the pose injection (20,480 params), and the PAE readout.
Everything else is frozen.

## The held-out set

83 structures / 141 interfaces, budget 30 seeds. Two disjoint waves: the "old 53" (92 interfaces,
what every number in `paper_results.md` §3–4 was measured on) and a "new 30" (49 interfaces, much
more headroom: 0.2173 vs 0.1299).

**The seeds are TRIMMED, and this matters.** 30 kept out of a 100-wide sweep, chosen per structure
by `anchored-coverage-v2` to preserve the oracle, so the kept ids are sparse — only 5 of 83
structures kept a contiguous set. `training/valfull83/seed_manifest.json` is authoritative;
assuming `400000..400029` reports 321 of 1,590 units as "missing trunks" and stops the run.

## Reproduce the figures

    ./run_logging/refresh_plots.sh qip9ta,hv2kqa figures
    # -> figures/qip9ta_train.png        PAE, rel, within-interface rho, score spread
    # -> figures/qip9ta_validation.png   DockQ, frac acceptable, rho vs the shipped baseline

Those two are the only supported plotters. Both draw only; every number comes from
`run_logging/epoch_curve.py` (training) and `eval_curve.py` (held out).

**The training curve is the ALL-RANKS mean.** `trajectory.csv` is rank 0's alone, so averaging its
per-pose column reports the third of the structures that rank walked. `epoch_curve` uses the
cross-rank `batch_<term>` weighted by `batch_n`. This defect recurred three times before the logic
was centralised; `run_logging/test_epoch_curve.py` fails if a plotter stops using the helper.

## Tests that must pass

    python -I run_logging/test_epoch_curve.py          all-ranks curve, per-rank trajectories
    python -I run_logging/test_seed_manifest.py        the trimmed seed set, 83-structure pool
    LAYERNORM_TYPE=openfold python -I phases/trainer/test_cli_equivalence.py
    LAYERNORM_TYPE=openfold python -I phases/trainer/test_config_equivalence.py
    python -I phases/trainer/test_checkpoints_equivalence.py
    python -I phases/trainer/test_step_plan.py

## Known, and deliberate

- `protenix-v2` ships **81 dead tensors / 3.9M params** in the confidence head — every
  `tri_att_end`, all of block 0's pair track. They hold ~4.9e-37 on a 1000-spaced integer lattice
  with a constant exponent, which is neither trained weights nor an initialisation. `--skip-dead-ops`
  declines to compute them; the arithmetic is bit-identical because a zero on a residual branch is
  the identity. Every baseline carries the same dead modules.
- `phases/trainer/` is a **separate minimal copy** of the trainer's config surface. Nothing runs it.
  `phase_k_train.py` is what executes.
