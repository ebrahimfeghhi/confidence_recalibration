# Pose provenance: which run wrote this trunk?

**Notation.** A *unit* is one `(pdb, seed)` fold. Each unit writes a *pose directory*
`<fold>/poses/<pdb>/<seed>-<runid>/` holding `predictions/<pdb>_sample_<k>.cif` for k=0..4 and a
`run.json` written only on success, plus a *trunk* `<fold>/trunks/<pdb>/<arm>_<seed>.npz` whose
`ci_coordinate` array holds the same 5 poses. *Join 2* is the map from a trunk's pose index i to
the sample index k of the cif it was written from; pose order is NOT sample order.

## The defect

A refold writes a **second** finished pose directory for a seed that was already folded. Both hold
a `run.json`, so neither is distinguishable by completeness, and the two runs are different folds
of the same seed (nondeterminism), not copies.

| where | count |
|---|---|
| validation units (pdb, seed) | 5,300 |
| validation units with 2 finished pose directories | 1,600 across 16 structures |
| training units with 2 finished pose directories | 0 |

The 16 structures are exactly the validation refold stages (13 in `fold_deferred` + 3 in
`fold_trunkfix`). Training never refolded, which is why its 5,000 cached labels are unaffected.

## Why the two obvious resolvers are wrong

`poses/index.tsv` records one `run_dir` per `(pdb, seed)`, but it was written by the original fold
and never updated by the refold, so it names the **superseded** run. Measured on `22gi`:

| seed | directory matching the trunk | `index.tsv` names | distance of the other directory |
|---|---|---|---|
| 400003 | `epk4ora3i7` | `fllxiphw1y` (wrong) | 0.0259 Å |
| 400004 | `1enbwnjro0` | `glonkmt4kh` (wrong) | 1.3787 Å |
| 400005 | `fa3a003bke` | `rpa2k4ufb6` (wrong) | 1.9177 Å |
| 400008 | `csecjkarrh` | `ey6joldtso` (wrong) | 3.1175 Å |
| 400010 | `g0rhuqb619` | `1f2zrb86ob` (wrong) | 0.0726 Å |

Wrong in 5 of 5. No code reads `index.tsv`; it is retained as a fold artifact and must not be used
for provenance.

Filesystem order is not evidence either. `phase_k_eval.py` took `glob(...)[0]`, which joined 95 of
`22gi`'s 100 seeds by luck and failed the other 5 — and a join failure there is fatal to the
evaluator's gate, so the LR sweep would not start.

## The resolver

`phase_j_build_cache.resolve_seed_dir(fold, pdb, seed, poses, tol, margin)` uses **the trunk as the
arbiter**: `ci_coordinate` was written by exactly one run, so exactly one directory's cifs join to
it under `sample_indices`' existing identity contract (one pose within `tol`, every other beyond
`margin`; defaults 0.01 Å / 0.5 Å). It introduces no threshold of its own — a directory qualifies
only if `sample_indices` already accepts it, and the true directory matches at 0.0000 Å while the
superseded one sits 0.026–3.118 Å away, two to three orders of magnitude outside `tol`.

Zero or two qualifying directories is **refused**, not guessed. Callers:

| caller | before | after |
|---|---|---|
| `phase_k_eval.py` | `glob(...)[0]` | `resolve_seed_dir` |
| `phase_j_build_cache.py` | `seed_dir` + `sample_indices` | `resolve_seed_dir` |
| `phase_j_stage_complete.py` | `>1 dir` = broken | `>1 dir` broken only if the trunk resolves to no single one |

On a unit with one directory `resolve_seed_dir` does exactly what `seed_dir` + `sample_indices`
did, so **no training label changes**.

## Evidence

| check | result |
|---|---|
| `phase_k_resolve_seed_dir_selftest.py` (6 cases: 1 match, 2 match, wrong run, no dir, real pair, no run.json) | 6 of 6 behave as designed |
| join 2 re-run over every validation unit on CPU | see table below |

| join 2, every validation unit | units | resolved | refused |
|---|---|---|---|
| all | 5,300 | 5,300 | 0 |
| of which ambiguous (2 directories) | 1,600 | 1,600 | 0 |

In all 1,600 ambiguous units the trunk selected the **newer** directory, which is what the refold
story predicts and an independent confirmation that the refold wrote both the trunk and the second
pose directory. The check costs 0.13 s/unit on CPU (`+0.065 s` only for ambiguous units) and needs
no GPU, so it can be re-run as a pre-flight before any evaluation.

`results.csv` was checked separately and shares the refold's provenance: 46,000 rows, 46,000
distinct `(iface, seed, sample)` keys, no duplicates, and a refolded interface's rows are byte-equal
to `fold_deferred/results.csv`. So the stored ipTM the evaluator checks itself against came from the
same run as the trunk.

## The second defect from the same merge: score rows

STAGE 0's merge moved `trunks/` and `poses/` and **nothing else**, so `validation/fold` ended up
holding the refold's trunks and poses beside the ORIGINAL fold's score rows for three structures.
This is the same "a new trunk must not be paired with an old pose" error in score-row form, and the
resolver above **cannot** detect it: the trunk and the cifs agree, so join 2 passes, and the wrong
DockQ is read from `results.csv` with no symptom.

| root | rows | identical in `fold/results.csv` before the repair | different |
|---|---|---|---|
| `fold_deferred` (13 structures) | 13,000 | 13,000 | 0 |
| `fold_trunkfix` (3 structures: 7tzh, 8v8f, 9cn2) | 3,000 | 853 | **2,147** |

The deferred merge did it correctly, which is why only one stage was affected. Repaired with
`phase_j_merge_results.py`, which replaces rows by `(iface, seed, sample)` and refuses a src key the
destination does not already have; after the repair both roots agree with `fold/results.csv` on all
16,000 rows, which still holds 46,000 rows and 46,000 distinct keys. The destination was copied to
`results.csv.pre_merge` first.

Because 2,147 rows changed, **the validation baseline (0.5779 / 0.7283 at 100 seeds over 92
interfaces) must be recomputed** before it is used as a comparison point.

## Restoring the invariant

The one-pose-directory-per-seed invariant is the real fix; the resolver is a guard. Both merge paths
are corrected (`queue_ft3_pilot.sh` now keys its collision test on `(pdb, seed)` and merges score
rows), and `phase_j_drop_superseded_poses.py` removes the directories no trunk was computed from:
1,600 directories, 2.8 GiB, 0 seeds where the trunk singled out neither or both. It recomputes the
evidence itself rather than reading a list, keeps the directory the trunk matches, and re-checks that
survivor against its trunk immediately before removing its rival, so no seed is left without
readable poses. This deletion was explicitly requested by the user on 2026-10-16, overriding the
standing never-delete-poses rule for this one case; the removed folds had no trunks (deleted in the
earlier 381 GB reclaim) and no score rows in `results.csv`.
