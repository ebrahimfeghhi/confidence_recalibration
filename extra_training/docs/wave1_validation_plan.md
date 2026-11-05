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
