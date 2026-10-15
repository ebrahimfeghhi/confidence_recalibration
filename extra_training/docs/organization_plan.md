# Organization plan

Deferred work from the 2026-10-14 experimental code review. The review's eight prioritized findings
and its first organization tip are fixed in commit `027ddcac`; tips 2-6 are recorded here instead of
being done, because none of them blocks training and the largest one risks the run in flight.

## Status

| Tip | What it asks for | Status |
|---|---|---|
| 1 | one configuration source; reject empty expected populations | **done** (`027ddcac`) |
| 2 | share the critical contracts; keep phase CLIs thin | open |
| 3 | separate frozen definitions from generated state | open |
| 4 | a regression-fixture set | **unblocked 2026-10-15**, and started: `phases/phase_j_label_audit_selftest.py` is the first instance -- it builds its fixtures from the repo, damages them in 5 known ways and exits non-zero if any goes unnoticed. The geometry rebuild it was waiting on has finished, so the class membership below is now measurable rather than invented |
| 5 | make the runtime reconstructable | partly done |
| 6 | one current experiment status page | open |

## Tip 1, for the record

Repointed 12 live references across 8 files: `train_fold` -> `training/fold`, `val2_fold` ->
`validation/fold`, `val_jobs` -> `validation/jobs`, `train_prep` -> `training/prep`, and PXM's
`free_pool/results.csv` -> `pxm2025_baseline_results.csv`. Three of those directories did not exist.
Each new path was checked to resolve on disk.

The empty-population half mattered more than the paths. PXM's dataset spec kept `keep="rule"` against
a merged table carrying no `tier` column, so `--counting pool` asked for `tier == "pool"`, matched
nothing, and printed a seed-depth curve over zero interfaces rather than failing. `keep` is now
`None` and an empty population raises.

## Tip 4 — the regression fixtures (do this first)

Wanted: ordinary Fabs, nanobodies, split scFvs, identical antigen copies, homodimers, missing
residues, partial score output, malformed JSON, confidence ties. CPU-only except one small GPU replay
that must consume the stored `lf_*` loss features.

NO LONGER DEFERRED: the geometry rebuild has finished. `phase_j_label_audit_selftest.py` covers the
label-audit phase and is the pattern to copy -- fixtures built from the repo's own files, damage
applied deliberately, non-zero exit when a damage is missed. It also demonstrated the value
immediately, by failing on its first run for a reason that turned out to be a WRONG EXPECTATION
rather than a bug: deleting the END of a chain is correctly invisible, because SAbDab2 crops to the
variable region and cannot verify residues it does not contain. A truncation test must be aimed.

Candidates already measured:

| Class | Count in the selection | Example |
|---|---|---|
| split single-chain constructs | 21 selected, 15 folded | `8h3c` (238 tokens), `8cz8` |
| identical antigen copies | 40 folded | `9e7l`, `7ugq` |
| geometry-veto cases | 139 of 501 | `8h3c` (32 A), `11zw` (10.8 A, marginal) |
| ordinary Fab control | 287 folded | `9g13` (139 tokens, smallest) |
| partial score output | 2 groups in the FoldBench table | `8b7h` seed 128, sample 2 absent on both interfaces |
| confidence ties | 274 FoldBench / 598 PXM groups | top `pair_iptm` tied across different DockQ |

## Tip 2 — shared contracts

The contracts that currently differ between folding, replay, rescoring and analysis: candidate
identity, cache schemas, completeness rules, score selection and tie rules, artifact fingerprints.
Partial progress exists — `seed_dir()` and `sample_cifs()` are now imported by the rescorer from
`phase_j_build_cache` rather than reimplemented, and the tie rule is identical in `folder.report()`
and `three_classes()`.

Deferred because it means editing `folder.py` while three workers are running it, plus every analysis
script, in one change with no test to anchor it. The fixes in `027ddcac` were each surgical with a
measured check; this one is not, and a break lands on the run producing data.

## Tip 3 — definitions versus generated state

Inputs (frozen, hashed): the selection, the train/val splits, the queues. Artifacts (regenerated):
progress, manifests, logs, results, labels. Each artifact should carry a schema version and an
explicit dependency list.

`027ddcac` did the label artifacts: every label npz now stores `schema` plus a fingerprint over the
job cif, bioassembly, indices, trunk, upstream code and loss weights, and a label whose inputs
changed is rebuilt rather than reused. The same treatment is not yet applied to the score tables or
the prepared data.

## Tip 5 — reconstructable runtime

Done: `runmeta.external_provenance()` records the external Protenix commit, whether its tree is
dirty, the local diff's hash and size, the checkpoint's sha256, python and torch into every
`run.json`; the 35,155-byte local diff is checked in as `run_model/protenix_local.patch`.

Open: the project metadata still declares Python >= 3.13 with a minimal dependency list, which does
not describe the Protenix runtime the folds actually use (3.12.3, torch 2.7.1+cu128).

## Tip 6 — one status page

Commands, artifact locations, expected counts, unresolved issues, with counts derived from manifests
rather than written down. The phase docstrings still carry historical narrative with superseded
counts -- `cached_training.md` says 536 structures where the live selection is 501.

## Not an organization tip, but open and owned by the user

Cluster 64 holds 77 of the 501 selected structures (15.4%) against a median cluster size of 1.
Uniform structure sampling spends about one update in seven on that single cluster. The cap is per
(antibody, antigen) pairing, which preserves distinct pairings and does not balance clusters. The
sampler and the loss reductions should state the choice explicitly; cluster-level uncertainty is
needed for any aggregate claim.
