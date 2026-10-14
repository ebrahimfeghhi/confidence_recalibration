# FoldBench mask-ON self-templating, 10 recycles

The nc10 arm. Identical to `../selftmpl_mask_on` in every respect — same donors, same seeds, same
template files, `PROTENIX_MULTICHAIN_MASK=1` — except `--ncycle 10` instead of 5.

**Headline result (phase 1, frozen): the solved-interface breakages at ncycle 5 were mostly a recycle
artifact, not the template mask. 5 of 7 recovered at 10 recycles.**

## Why this run exists

The nc5 arm (`../selftmpl_mask_on`) broke 9 solved interfaces while rescuing 7 decoys, netting −2
acceptable. But it ran at `--ncycle 5` while the free baseline ran at `--ncycle 10`, so *every*
breakage was confounded: the damage could have been the halved recycle budget rather than the mask.
Protenix ships `N_cycle=10` as its default, presumably tuned, and we had measured only the 0.66–0.68
compute cost of halving it, never the quality cost.

## Two phases, one results.csv

Phase 1 folded only the 6 complexes holding the 9 broken solved interfaces (183 folds). Phase 2
expands to the whole benchmark (3,209 folds). Phase 1's 183 folds are a **strict subset** of phase 2's
queue (verified: `comm -23 <(sort queue_6complex.tsv) <(sort ../selftmpl_mask_on/queue.tsv)` → 0
lines), so phase 2 reuses them — `folder.py` skips an `(iface, seed, arm)` already in `results.csv`.
One csv covers both phases; nothing was refolded.

| file | what it is |
|:---|:---|
| `queue_6complex.tsv` | PHASE 1, pinned: the 183 folds over 6 complexes. `report.py` reads this, never the live queue |
| `queue_full.tsv` | PHASE 2 source: all 3,209 folds, ordered (see below) |
| `build_phase2_queue.py` | builds `queue_full.tsv` from the nc5 arm's queue. Asserts the reorder is a permutation — same folds, none added, none dropped. Verified byte-identical to the queue actually launched |
| `sizes.json` | atom count per complex, the ordering key; cached so a rebuild is instant |
| `phase2_handoff.sh`, `phase2_handoff.log` | the phase-1 → phase-2 switch and its log (waits, freezes, swaps, relaunches) |
| `queue.tsv` | the LIVE queue the workers read. Phase 1 = the 183; phase 2 = a copy of `queue_full.tsv` |
| `results_phase1_6complex.md` | frozen snapshot of the phase-1 table, as of the queue swap |
| `report.py` | phase-1 recovery table: free+matched / nc5 / nc10 / recovered? / oracle / pooled |
| `results.csv` | every scored pose, both phases |
| `poses/<pdb>/<seed>-<run_id>/` | poses + `run.json` provenance (pins the exact donor file by md5) |
| `w{0,1,3}.log`, `watch.log` | worker logs; watcher ticks |

Final report lands in `../../../../experimental_results/fb_mask_on_nc10.md` (the nc5 arm's equivalent
is `fb_mask_on_vs_free_x068.md`).

## Phase 2 queue order

Solved-containing complexes first (the risk this run exists to measure), then decoy-only, then
nocov-only; size ascending within each group; complexes ≥14,000 atoms pulled to the global tail so a
monster cannot block early progress.

| phase | complexes | folds | cumulative | atoms |
|:---|---:|---:|---:|---:|
| solved-containing | 77 | 2063 | 2063 | 1,618–13,996 |
| decoy-only | 16 | 642 | 2705 | 1,930–12,713 |
| nocov-only | 5 | 231 | 2936 | 5,575–12,024 |
| tail ≥14k atoms | 6 | 273 | 3209 | 14,419–19,652 |

Tail: 8fym, 8xnh, 8r4u (solved), 7z2m, 7zau (decoy), 8oud (nocov). **8xnh is also the all-NaN
complex** — it emitted 17,661/17,661 NaN coordinates on seeds 180 and 193 in an earlier arm, so it is
both the largest and the most likely to need a refold. Deferring it is deliberate.

A complex with BOTH solved and decoy interfaces is grouped as solved-containing, because a fold scores
every interface of its complex at once: putting mixed complexes first resolves all 75 solved
interfaces earliest, which is what "do the solved ones first" asks for.

## Reading the result: the compute match changes

**Compare against `--match-factor 1.0`, not 0.68.** At ncycle 5 a method fold cost 0.68 of a free
fold, so the baseline was granted 0.68 extra free seeds per method fold. At ncycle 10 a method fold
costs the SAME as a free fold, so the baseline gets a full extra seed per method fold. This arm
therefore faces a STRICTER bar than the nc5 arm did — it removes the confound and the 32% compute
discount at the same time. Quoting an nc10 number against 0.68 would flatter it.

```
cd ../../../../diffusion_commit_clean_v2 && /home/ebrahim/Protenix/.venv/bin/python compare_arms.py \
  --dataset foldbench --exp ../experimental_data/protenix/foldbench/selftmpl_mask_on_nc10 \
  --match-factor 1.0 --per-interface
```

## Phase 1 result

`report.py` for the live table; `results_phase1_6complex.md` for the frozen one. The split was clean:
every interface whose nc5 DockQ had collapsed to ~0.01–0.05 came back at nc10, and both interfaces
sitting at ~0.22 did not.

| nc5 DockQ | interfaces | recovered at nc10 |
|:---|---:|---:|
| ~0.01–0.05 (total mis-dock) | 5 | 5 |
| ~0.22 (committed to a wrong site) | 2 | 0 |

**Pooling free + nc10 does not fix the two that failed, it makes them worse.** On 8fdd the pool held
the free pose at 0.579/0.799 and top-pooled-ipTM selected the templated 0.221/0.216 instead -- on an
ipTM margin of **0.0006**. `pair_iptm` is not comparable across regimes, but **not** because templated folds run hotter -- measured 2026-10-06 over 110 solved interfaces, templated ipTM is LOWER than free at every DockQ level (-0.013 to -0.044; mean 0.507 vs 0.592). The mechanism is the UPPER TAIL: a templated pose beats free's own best ipTM on **76%** of interfaces by a median **+0.021**, while extra FREE poses manage it on only 30% (median -0.005). Templated poses are 31% of the pooled arm but win **76% of the argmaxes** -- 2.5x their share, against extra free's 30% which is exactly its share. Templating is effectively bimodal (a confident pose or a poor one) and an argmax over 250+ poses only ever sees the confident mode. So pooling cannot guard against breakage when the broken
pose is the one it picks. Note the pooled ORACLE on 8fdd is
0.915/0.949 against an nc10 oracle of 0.606/0.821 — that 0.9 pose is a *free* pose that free's own
top-ipTM pick also missed. 8fdd is a selection failure on both arms, not a coverage failure.

## Reproduce

```
bash launch.sh <gpu>        # one worker; reads queue.tsv, --ncycle 10, mask ON
GPUS="0 1 3" PREFIX=nc10_ REPORT=.../fb_mask_on_nc10.md bash ../../../../diffusion_commit_clean_v2/watch_fb_mask_on.sh
```

The watcher is shared with the nc5 arm and takes `E` / `PREFIX` / `REPORT` from the environment.
`PREFIX` must match the live tmux session names (`nc10_0`, `nc10_1`, `nc10_3`) or its REMEDIATE step
will start a duplicate worker on a GPU that already has one.
