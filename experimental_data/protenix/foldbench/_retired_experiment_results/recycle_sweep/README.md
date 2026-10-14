# Recycle-ladder sweep: is pooled ipTM across recycles a better selector than final ipTM?

Launched 2026-10-06 20:00. **3 GPUs (0, 1, 2).** Watcher `rswatch` every 5 minutes.

## The question

A fold's final ipTM reports which basin it committed to — but commitment is a per-run coin flip, not a
seed property. Six runs of seed 143 on 8zp2_AB at identical settings gave DockQ 0.783, 0.004, 0.062,
0.761, 0.060, 0.355, and ipTM separated the two good runs (0.950, 0.956) from the four bad ones
(0.349–0.393) perfectly. So ipTM is a reliable *detector* of the basin, sampled once per fold.

This asks whether the **trajectory** carries signal the endpoint does not: do decoys keep changing
their mind across recycles while solved interfaces settle early?

## Design

Per target complex, per seed (101–150): fold to recycle 5 and decode 5 poses, then resume **+1 recycle
and decode 5 poses**, five more times — rungs 5, 6, 7, 8, 9, 10. **30 poses per seed**, each with its
own DockQ and ipTM.

**One trajectory, not six folds.** Six separate folds at `--ncycle 5..10` would differ by run-to-run
non-determinism as much as by recycle count — the exact confound that makes a single ipTM unreliable.
The trunk dump carries z, s **and the RNG streams**, so every rung continues the same stream and the
rungs differ only in how many recycles preceded them.

**Not `DUMP_CYCLE_DECODE`.** That hook decodes at every recycle but saves coordinates only — it never
runs the confidence head, so it cannot produce ipTM, which is the point here.

**The tag is always `free`.** `TRUNK_DUMP_DIR` writes `<dir>/<pdb>/<tag>_<seed>.npz` while
`TRUNK_LOAD_DIR` reads `<dir>/<pdb>/free_<seed>.npz` — the loader hardcodes `free`, so renaming the
tag breaks every resume. The six rungs are separated by their own csv/out/claim directories instead:
`results_c<k>.csv_<pdb>`.

## Scope

| phase | interfaces | complexes | status |
|:---|---:|---:|:---|
| 1 | 20 (10 decoy, 10 solved) | 17 | running |
| 2 | 20 more (10 decoy, 10 solved) | 15 further | queued, only if phase 1 finishes early |

Smallest complexes first (1k–6k atoms) to control cost. `targets.tsv` carries phase, class, interface,
complex, atom count.

## Mechanics

- **Work stealing, not slicing.** The six rungs must run in order against one set of trunks, so a
  worker claims a whole COMPLEX with an atomic `mkdir claim_<pdb>`. Three workers process three
  complexes concurrently; the watcher releases a claim whose owner pid is gone.
- **Disk.** 50 trunks per rung, two rungs alive at once for one complex (rung k cannot free rung k−1
  until its own csv is complete, or an interruption would leave nothing to resume from). ~32 MB per
  trunk at 8aok's size. All of a complex's trunks are deleted when it finishes.
- **Resumable.** A rung whose csv already has 50 scored seeds is skipped. A partially-written rung has
  its dump dir cleared and is rerun from the previous rung's trunks, which are still present.

## Reading it

```
./report.py --phase 1
```
Scores five candidate selectors per interface — final ipTM, pooled max, pooled mean, rung spread
(rung 10 minus rung 5), rung std — and prints the solved-minus-decoy separation for each. A selector
is judged by the DockQ of the pose it picks, not by its own value.

## Files

| file | what |
|:---|:---|
| `run.sh` | one worker, `$1 = GPU`; claims a complex, runs six rungs, cleans up |
| `watch.sh` | 5-minute health/progress; remediates dead workers and stale complex claims |
| `report.py` | the selector comparison |
| `results_c<k>.csv_<pdb>` | scored poses for rung k of that complex |
| `poses_c<k>/` | the poses themselves, with `run.json` provenance |
| `w<gpu>.log`, `watch.log` | worker and watcher logs |

## RESULT — closed 2026-10-07, negative

**Pooling ipTM across recycles is not a better selector than the final recycle's ipTM.**

Phase 1, 20 interfaces (10 decoy, 10 solved), 50 seeds x 6 rungs x 5 samples:

| selector | solved − decoy separation |
|:---|---:|
| **final ipTM** (rung 10 only) | **+0.294** |
| pooled mean over 30 poses | +0.291 |
| pooled max over 30 poses | +0.259 |
| rung spread (rung 10 − rung 5) | +0.013 |
| rung std across rungs | −0.011 |

On the actionable metric the two selections are **identical** — decoy 0/10, solved 8/10 under both,
with no interface where they disagree. So the recycle trajectory carries no confidence signal the
endpoint lacks, and "decoys keep changing their mind across recycles" is not in the data.

**The one real finding, and why it is not usable.** DockQ *does* move across rungs — median
within-trajectory swing 0.105 — and the drift is class-dependent: solved rises c5 0.564 -> c10 0.602,
decoys fall 0.184 -> 0.169, a +0.054 separation in the trend. The best rung is near-uniform
(c5 20%, c6 17%, c7 15%, c8 18%, c9 13%, c10 18%), so recycling wanders rather than refines. But that
signal lives in DockQ, which inference does not have, and its confidence analogue (rung spread)
carries only +0.013 of it. The trajectory knows something ipTM cannot see.

Consistent with the same day's other negative: resuming seed 125 of 8zp2 past recycle 10 made it
worse, not better (ipTM 0.566 -> 0.431 -> 0.389 at rungs 10/11/12).

**Stopped and cleaned.** All trunk checkpoints deleted (99 GB reclaimed; they were 87% of this
directory's footprint and are resume state, not results). Kept: 6 pose directories, 132 result csvs,
47,460 scored poses with DockQ and ipTM. Phase 2 reached 4 of 13 complexes before being stopped.
