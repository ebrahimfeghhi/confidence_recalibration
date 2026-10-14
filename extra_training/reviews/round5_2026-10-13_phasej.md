# Review round 5 — phase J and phase I, 2026-10-13

Reviewer's pass at `ee69ca5`. Verdicts below are what I reproduced on this box.

| # | finding | verdict | status |
|---|---|---|---|
| J1 | splitting a chain leaves two auth chains sharing a subchain and entity | **CONFIRMED** | **closed round 8** |
| J2 | sequence matching can pick an antigen copy that does not contact the chosen antibody | **CONFIRMED, 71 of 501** | **closed round 6** |
| J3 | resolved-first trimming can shorten the label's sequence frame | **PARTLY CONFIRMED** | **closed round 8** |
| J4 | defaults point at whole-entry cifs, and a missing one is downloaded | **CONFIRMED** | fixed |
| J5 | `rank_bad` never incremented; pose identity cannot come from scores | **CONFIRMED** | fixed |
| I1 | phase I can report COMPLETE when a job is absent from pool.csv | **reproduced round 8 on a synthetic tree** | **closed round 8** |
| I2 | the cache validator accepts NaN coordinates and a bad trunk shape | **reproduced round 8, 8 controls** | **closed round 8** |
| DE1 | phases D/E leave READY markers after a failure, no trap | open since round 2 | **closed round 7** |

---

## J5 — the ranking check never ran. A reported result was vacuous.

`rank_bad` is initialised at `:168` and printed at `:221`; `got_r` and `want_r` are computed at
`:203-204` and discarded. The increment was lost when the optimal-assignment fix was swapped in.

**"pose ranking within a seed changed on 0 of 1,767 units" was therefore unconditional**, and it was
reported to the user as a verification result. Withdrawn. The companion figure,
`max |stored - recomputed| = 0.000000` over 17,665 poses, IS computed and compared and stands.

Fixed: the comparison runs, and the summary now states plainly that matching poses by confidence
score cannot establish identity -- distinct poses share scores (9fve/400073) -- so the mapping must
not be used to join poses to DockQ or to labels. The cache writer records the sample index.

## J4 — defaults pointed at whole-entry cifs

All three of `phase_j_prepare_training_data.py`, `phase_j_dataset.py` and `phase_j_job_cifs.py`
defaulted to `data/rcsb_training_cifs`. A whole entry builds the deposited assembly, which is not
what we fold, so a label built from it cannot be compared to a pose. The runs that produced what is
on disk passed explicit paths, so the artifacts are right -- but the default was a trap, and
`prepare_training_data` would DOWNLOAD a missing entry rather than fail, silently substituting the
assembly for the job.

Fixed: the defaults are the job cifs and job bioassemblies, and fetching is opt-in via `--fetch`
rather than opt-out.

## J1 and J3 — the label's chain decomposition and sequence frame

Measured on the written job cifs:

| structure | what the metadata says |
|---|---|
| 9p4c (trimmed full-length antibody) | entities 216 / 211 / 17, one subchain each — **correct** |
| 9vb4 antigen | `full_sequence` 354 with residues at label_seq 4–354 — **correct**, the 3 unresolved positions stay masked |
| 9vb4 split construct | auth chains D and C **share subchain F and entity 6**, whose `full_sequence` is the 259-residue fusion — **wrong** |

So ordinary chains come out right, and the 9 split constructs do not: two auth chains under one
label_asym is a wrong chain decomposition, and every chain-pair quantity the confidence head produces
depends on it. It is right by accident elsewhere -- gemmi preserved the original entity because
label_seq survived -- and accident is not an invariant.

Fix in progress: build each output chain's entity explicitly, with `full_sequence` set to the FOLD's
expected sequence for that role and `label_seq` from the alignment, so the label always describes the
expected sequence with unresolved positions masked; then assert it against the job json.

## J2 — not measured

Sequence identity cannot distinguish copies of an antigen, and the tie-break is alphabetical, so a
selected antigen may not be the copy that contacts the selected antibody. Not yet checked
geometrically. This is the next measurement, not an assumption either way.

## Phase I and phases D/E

Reported this round, reproduced by the reviewer with synthetic inputs, not yet reproduced here:

- `phase_i_check.py` can report COMPLETE when a job is absent from `pool.csv`, because the pose and
  scoring checks are skipped for it. Every active job should have to present its expected interfaces,
  and pose files should be checked independently of `pool.csv`.
- the cache validator accepts NaN coordinates and an invalid trunk shape. It should check finiteness,
  dimensions, and consistency between the atom and token mappings.
- phases D and E leave READY markers behind on a `die` path, so a retry can skip an incomplete
  database build. Open since round 2. Needs a trap that removes only the markers that invocation
  created.

Neither phase I finding establishes that existing outputs are corrupted.
