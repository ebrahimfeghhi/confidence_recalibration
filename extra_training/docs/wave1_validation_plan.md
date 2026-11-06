# Wave 1 validation: status and plan

Written 2026-11-04, ~9:40 PM EST. Branch `perf/ckpt-by-size`.

## Where things stand

The training fold is running normally on cards 4, 5 and 7 — fold and score inline, ~28 s/unit,
3,560 units, ETA ~9 h. **Nothing is running or armed for validation.**

The 30 new validation structures are fully prepped except for two artifacts, and both come from
the same missing file.

| | |
|---|---|
| selected | 30 |
| job jsons | 30/30 |
| per-job CIFs | 30/30 |
| bioassemblies | 30/30 |
| pool rows | 13/30 |
| ground-truth refs | 0/30 |

The existing 53-structure benchmark is untouched: 53 structures, 1,590 units, verified.

## The one root cause

17 of the 30 have no SAbDab2 instance CIF in the local corpus (`SAbDab2_data/training_cifs`
holds 4,903 of the splits' 15,810 instances). That single file feeds two things:

1. **The pool's chain names.** `build_wave1_pool.py` reads auth → `label_asym_id` from it. The
   scorer keys interfaces by the *reference* label, not the model letter, which is why the
   training pool names `10or`'s interfaces B-S and F-S rather than its positional A-D.
2. **The DockQ reference itself.** `phase_g_build_val_jobs.py:410` stages that same instance CIF
   as `validation/refs/<pdb>.cif`, which is what `paths.py:225` points the scorer at.

So the refs and the pool labels are derived from one file, which is exactly why they agree today.
Any replacement has to supply both from one source or they will not join.

### What is NOT a problem

The 17's **labels are sound**. A missing instance CIF only endangers a role resolved by
*sequence*; `pick_contacting_copies` states that roles resolved by SAbDab's own chain names are
never moved, and `9djw` failed precisely because SAbDab named chains (`AG`, `AE`) the entry does
not contain, so both its roles fell to sequence. Measured on all 17:

```
all roles named (copy choice authoritative, instance cif adds nothing): 17 of 17
any role by sequence (the 9djw risk is real):                           0
```

That includes the 6 with two copies of the complex in the entry (`8q7o`, `8v52`, `8weo`, `9dl0`,
`9hvi`, `9hvk`). No job CIFs need regenerating. The warning that implied otherwise fired on file
absence without checking whether the names resolved, and now distinguishes the two cases.

`build_instance_cifs.py` is **not** a fix and is not wired in anywhere. It reconstructs an
instance CIF as the deposited entry restricted to the splits row's chains, and its own verify
gate rejects it at 9 of 13: the corpus also trims chains to the Fv domain (`9ghi` H is 120
residues against 257) and sometimes keeps chains the splits row does not name (`8yyz` holds D
and J, `8z7j` holds S1 and S2). It is committed as the record of a rejected route.
`validation/prep/instance_cifs` does not exist.

## Proposed resolution for the 17

Use the **deposited entry** as the single source for both artifacts, resolving the one ambiguity
explicitly rather than guessing.

`ref_auth_to_label` refuses an entry where one auth chain covers several polymer label subchains,
and says so: *"resolve it explicitly rather than guessing"*. That is the only obstacle, and the
role's own sequence resolves it. Measured on the 17:

```
resolve directly from the entry:                4   (8ya0, 9hvi, 9hvk, 9vca)
resolve after matching the role sequence
  to one of the candidate label subchains:     13
unresolved:                                     0
```

**The gate, already run.** Applying the same resolution to the 13 structures that *do* have
instance CIFs and comparing against the committed `pool_val13.csv`:

```
identical interface sets: 12 of 13
  8z7j  mine []  vs instance-derived [('A','B'), ('A','C')]
```

`8z7j` fails **closed**, not wrong — SAbDab names chains `S1`/`S2` that the entry does not have,
so the resolver returns nothing rather than a wrong label. That is the behaviour to keep: any
role that does not resolve to exactly one label subchain drops the structure.

### Still to be validated before use

Entry-as-reference changes what the scorer compares against: the entry carries extra copies and
chains the instance CIF does not. DockQ scores a named chain pair, so this should not matter, but
it has not been measured. **The measurement is cheap and uses existing data**: the old 53-structure
benchmark has 1,590 folded units with poses on disk *and* instance-CIF refs. Re-score a sample
against each structure's deposited entry and compare DockQ to the stored values. If they agree,
entry-as-reference is validated on real poses; if they do not, this route is rejected the way the
reconstruction was. This is CPU-only (the scorer is `ost`).

### Alternatives if that gate fails

- **Re-select the 17** from the 1,962 candidates that have an instance CIF, MSAs, eligibility and
  fit the 2,560-token budget. Keeps 30 structures; costs one re-run of jobs → per-job CIFs →
  bioassemblies → pool for 17 structures, all CPU.
- **Carry 13** and treat the expansion as smaller than planned.

## Plan, in order

1. **Validate entry-as-reference on the old 53.** Re-score a sample of already-folded units
   against the deposited entry instead of the staged instance CIF; compare DockQ to the stored
   values. CPU only, no refolding. *Gate: agreement, or this route is rejected.*
2. **Implement the explicit resolver** in the pool builder — resolve a multi-subchain auth chain
   by matching the role sequence, and fail closed on anything that does not resolve to exactly
   one. Gate it on reproducing `pool_val13.csv` (currently 12/13 with the 13th failing closed).
3. **Stage the refs for all 30.** The 13 get their instance CIF staged as
   `validation/refs/<pdb>.cif`; the 17 get their deposited entry, from the same file the pool
   labels were derived from. Every queued PDB's ref must exist before anything launches.
4. **Rebuild and merge the pool for all 30**, then verify: every queued structure has pool rows,
   chain roles, and a ref on disk.
5. **Rescore the 240 recovered training units.** `training_queue_drained.py` found 241 units
   folded but unscored, 240 with poses still on disk. Those need a rescore, not a refold — the
   GPU work is already paid for. CPU only, runs alongside the fold.
6. **Fold validation behind the training drain.** `arm_val13_after_train.sh` gates on worker pids
   gone, no tmux session, `cards_busy` 0, and `training_queue_drained.py` passing — not on a log
   line. Nothing launches until all four hold.

Steps 1, 2, 3, 4 and 5 are CPU-only and can run while the training fold holds the cards. Only
step 6 needs the GPUs.

## Fixed since the last review

- `--fold-only` **removed entirely** (`9b9570d1`). It produced three rounds of the same bug: v1
  deleted `<rdir>/predictions`; v2 removed that deletion, which was the wrong line, because the
  poses live in `<rdir>/<name>/seed_<seed>/predictions` until `normalise_poses` lifts them, so the
  `rmtree` of `<rdir>/<name>` destroyed them; v3 lifted and verified them and still returned
  "failed" on both new paths without releasing the claim. Its reason for existing is gone — the
  pool now covers all 856 training and 66 validation structures, and scoring inline costs ~3-4 s
  on a ~25 s unit. Scored path verified after removal: `8qz6` seed 500000, 1/1 interfaces in 43 s,
  DockQ 0.8230, 5 poses, 0 failed.
- The arming gate reads **state, not log lines** (`a8ecaa90`). `folder.py` prints "queue complete"
  with its failed tally included and then runs a second pass, so that string appears while the
  worker is still folding; an earlier version waited on tmux sessions reaching zero, which a
  manual stop satisfies — that is what fired and cost 42 units.
- `build_wave1_pool.py` committed (it existed only on the server).
- Both CPU chains fail closed, with completeness checks rather than exit codes alone.
- `launch_wave1_val_fold.sh` uses absolute paths — the worker runs from the Protenix checkout, so
  relative paths resolved there, and the pool preflight runs in the launching shell and could not
  catch it.

## Measurement corrections worth keeping

- **Inline scoring is close to free.** Matched on `8qz6`: 23-25 s fold-only, 28 s fold+score.
  Earlier reports of 1.77x and 5x were artifacts — a single-unit test that omitted
  `--overlap-scoring`, then two fold processes sharing each card after a validation fold
  auto-launched. Both were measured on `9dvr`, 869 tokens against a 474-token queue median.
- `9dvr` has now produced three bad extrapolations (disk, runtime, and this). It is not a
  representative unit and should not be used as one.

---

# Step 1 result: entry-as-reference is REJECTED

Run 2026-11-04 ~10:05 PM EST. CPU only, 318 OST calls over all 53 old benchmark structures,
3 poses each, fresh json per reference. Nothing was written into the benchmark; the new refs were
not staged and validation was not launched.

Script: `run_logging/compare_entry_as_reference.py`.
Per-comparison rows: `validation/full_val/entry_as_reference_report.csv`.

## What passed

The two things review was most worried about came back clean:

```
ref->model chain mapping differs:        0 of 176
ipTM read from a different model pair:   0 of 176
```

Where the entry scores an interface at all, it maps it to the same model chains and ipTM is read
from the same pair. The reference *naming* is not the problem.

## What failed

**1. Multi-copy structures lose interfaces outright.**

```
multi-copy structures:  13 lose interfaces, 10 clean   (57% fail)
single-copy structures:  1 lose interfaces, 24 clean   ( 4% fail)
```

67 interface-pose rows scored against the staged instance CIF and produced *nothing* against the
entry — 61 of those 67 on multi-copy structures. The entry holds several copies of the complex, so
OST's automatic mapper lays the model over whichever copy fits best, and the interface we asked
for by label is simply absent from the result. `ost_dockq.interface()` has the `<X>-2` fallback for
FoldBench-style copy naming, but a deposited entry's copies carry ordinary distinct labels, so
there is nothing to fall back on. This is exactly the failure mode review predicted from extra
chains, and it shows up as a missing interface rather than a wrong number — which is worse, because
no row is written and the structure quietly contributes less data.

**This is the class that matters most for the 17: six of them are multi-copy** — `8q7o`, `8v52`,
`8weo`, `9dl0`, `9hvi`, `9hvk` — the same six flagged earlier.

**2. DockQ shifts even where everything maps correctly.**

Restricted to single-copy structures that lost no interface (n=123):

```
|dDockQ|: max 0.099  median 0.0030  over 0.01: 44 of 123
```

Signed, over all 176 scored both ways: mean −0.0062, median 0.0000, sd 0.0245, max |Δ| 0.121
(`8tbq` L-ag0: 0.810 staged against 0.689 entry). 53 of 176 agree exactly; the rest do not.

The cause is residue content, not naming. The corpus instance CIFs are trimmed — measured earlier
on the same structures, `9ghi`'s H chain is 120 residues against 257 in the entry, and antigens are
trimmed too. DockQ is defined over the reference's interface residues, so an untrimmed reference
scores a different interface. A median of 0.003 is small, but a benchmark mixing two reference
conventions would compare structures scored against trimmed references with structures scored
against untrimmed ones, and the paired error bar on this benchmark is ~0.005 uncropped. A
systematic reference difference of that size across 17 of 83 structures is not acceptable.

**3. Five structures cannot use the entry at all**, and fail closed as designed:
`29tj`, `7tzh`, `8v8f`, `9cn2`, `9nr6`. SAbDab names chains the deposited entry does not carry —
`9nr6`'s instance is `pdb_00009nr6_F1_F2` and the entry has no `F1`/`F2` — so the resolver returns
nothing rather than a wrong label. 30 rows excluded on that basis, which is the correct behaviour
and confirms the fail-closed rule works on real data.

## Conclusion and revised plan

Entry-as-reference is rejected on evidence, the same way the reconstruction was. The resolver idea
was sound for *naming* — 0 of 176 mapping differences proves that — but the reference file itself
cannot be swapped, because the instance CIF's trimming is part of what the stored DockQ means, and
because multi-copy entries lose interfaces 57% of the time.

**Revised step 2: re-select the 17** from the 1,962 candidates that have an instance CIF, MSAs,
eligibility, and fit the 2,560-token budget. That keeps the validation set at 30 new structures,
keeps one reference convention across the whole benchmark, and costs one CPU re-run of jobs →
per-job CIFs → bioassemblies → pool for 17 structures. The alternative is carrying 13.

What this run also settles: the explicit sequence resolver for a multi-subchain auth chain is
correct and worth keeping for the *pool* side wherever an instance CIF exists, since it reproduced
the committed pool for 12 of 13 with the 13th failing closed. It is the reference swap that is
dead, not the resolver.

---

# RESOLVED: the 17 were never missing, only never unpacked

2026-11-04 ~10:21 PM EST. All 30 new validation structures are now fully prepared.

## Why those 17 specifically

Nothing about them. The question "why do those 17 lack the information" has a dull answer that
three earlier hypotheses got wrong:

- **Not the split.** The corpus holds 3,905 train and 998 test instances; the missing ones are
  8,743 train and 2,164 test. Both sides are mixed.
- **Not the release date.** Present and absent instances are spread across 2021-2026 in similar
  proportions.
- **Not a different instance of the same entry.** The corpus has *zero* instances for all 17
  entries, so it is missing the entry, not our chosen copy. 5,743 of 8,796 entries have none.
- **Not MSA coverage.** All 17 have MSAs; those were generated for the Wave 1 validation set.

`SAbDab2_data/training_cifs` is simply a **partial extraction**: 4,903 of 15,810 instance cifs,
covering 3,053 of 8,796 PDB entries. It happens to cover 603 of 603 structures the project had
already folded, and the Wave 1 *training* selection landed entirely inside it (356 of 356), so the
gap never surfaced until the validation selection reached outside it. No step checked for corpus
membership, and MSAs were extended to the new validation structures while the instance cifs were
not.

**The archive with all 15,810 was on disk the whole time**:
`SAbDab2_data/zenodo_splits/splits_v0.2.0.tar.gz`, 889 MB, 15,810 cif members, including all 30
of ours. Extracting the 17 was a local operation, not a download.

## What was done

Every step is in `shell_scripts/val30_cpu_chain.sh`, which now runs end to end from the archive —
the first pass had the extraction, the queue json and the pool merge done by hand, which is not
reproducible and left the script failing on inputs it could not build for itself.

1. `run_logging/extract_instance_cifs.py` — extract missing instance cifs from the release
   archive. Each one's chain set must equal the chains the splits row names, or nothing is
   installed; an existing corpus file is never overwritten. 17 extracted, **17 of 17 verified**.
2. Per-job cifs regenerated for all 30, now with the instance cifs present, so the copy choice is
   anchored by `pick_instance_copies` rather than resting on names alone. The job-cif log no longer
   emits any instance-cif note at all.
3. Bioassemblies rebuilt: 83 of 83.
4. Pool: **49 rows over 30 of 30 structures**, 79 chain roles, 1 relabelled. Regression check: the
   13 previously pooled are **unchanged, 13 of 13**.
5. `run_logging/stage_val_refs.py` — stage the DockQ reference per structure, taking the
   *relabelled* copy where `relabel_unique` wrote one (`8z7j`), which is the staleness class
   `phase_g` records costing 8evg all four of its interfaces. Every pool chain must appear as a
   `label_asym_id` in the reference actually staged: **30 of 30 verified**.
6. Pool merged into `validation/fold/pool.csv` (141 rows over 83 structures), with the launcher's
   own gate applied in the chain: **0 unpooled, 0 without a reference**.

| | before | now |
|---|---|---|
| instance cif | 13/30 | **30/30** |
| pool rows | 13/30 | **30/30** |
| ground-truth ref | 0/30 | **30/30** |
| queue | 1,300 units / 13 | **3,000 units / 30** |

The old 53-structure benchmark is untouched: 1,590 trunk units, 53 structures, its 58 references
still in place.

## What this changes about the step 1 result

Nothing about the measurement — entry-as-reference is still rejected, and for the reasons measured:
multi-copy entries lose interfaces 57% of the time and DockQ shifts up to 0.121 because the
instance cifs are trimmed. It is simply no longer needed. The re-selection fallback is also
unnecessary: the validation set keeps all 30 originally chosen structures, with one reference
convention across the whole benchmark.

The sequence resolver is kept in `compare_entry_as_reference.py` as the measurement that rejected
the swap, and `build_instance_cifs.py` remains the record of the earlier rejected reconstruction.
Neither is in the live path.

## Remaining before validation folds

1. **Rescore the 240 recovered training units** — folded, unscored, poses on disk. CPU only.
2. **Fold validation behind the training drain**, via `arm_val13_after_train.sh`, which gates on
   worker pids gone, no tmux session, `cards_busy` 0, and `training_queue_drained.py` passing.
   3,000 units over 30 structures, ~8 h on three cards at the measured 28 s/unit.
