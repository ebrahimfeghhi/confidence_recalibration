# Audit of the self-templating pipeline — 2026-09-25

Every change made, why, how it was checked, and its commit. Findings that changed results come first.

## Summary

| # | finding | effect | status |
|:---|:---|:---|:---|
| 1 | Selection read the crystal in five places | donors chosen partly by the crystal | fixed — rebuilt crystal-free (`d69dfd3`, `d971b9d`) |
| 2 | Ranking ipTM was read at different chains than the ones compared and templated | ~25% of poses ranked on the wrong copy | fixed (`c120983`, then subsumed by 1) |
| 3 | Folds shared run directories and overwrote each other's poses | 154 / 181 fixed-arm folds lost poses; scores intact | fixed (`2fd17b8`) |
| 4 | Dead claims were never released in templated arms | a fold killed mid-run was skipped forever (4 stranded) | fixed (`378f1c4`) |
| 5 | Six "predicted-only" interfaces shared names with crystal interfaces | their matrices mixed two pools; one pair never ranked | gone — one naming scheme (`d69dfd3`) |
| 6 | The two reporters used different baselines, classes and completeness rules | reports could disagree | fixed (`63ca94c`) |
| 7 | Template cache and preflight could not catch a wrong donor or a dangling path | latent wrong template | fixed (`4cbe6ca`, `f3d8e05`) |

The old fixed-seed arm is kept as `experimental_data/self_template_fixed_50_seed_old_crystal_iptm/`.
Of its folds, 2,144 are still selected with the same template and are carried over (25,060 rows,
values identical); 650 are no longer selected and are excluded; 1,380 new folds are being run.

## 1. The crystal on the selection path

**Rule.** Two facts may come from the crystal, as a biologist would know them before folding: which
complexes form the benchmark, and which chains are antibody and which antigen. Nothing about binding
— which chains interface, which copy, the ranking, the similarity, the donors — may.

**Where the old pipeline broke it.**

| where | what it read |
|:---|:---|
| `anchor_relabel.py` | the anchor was the pose with the highest mean pool-csv ipTM, measured at crystal-matched chains over crystal interfaces |
| `anchor_relabel.py` | interface names and roles came through the anchor's crystal map; complexes without one were skipped |
| `pose_interfaces.py` | candidate pairs from crystal roles; single-interface complexes fell back to crystal maps |
| `fetch_spurious_interface_iptm.py` | fell back to crystal maps, and read an older anchor run (fixed `0ffaa19` before the redesign) |
| `structural_diversity_filter.py` | pinned single-interface complexes in the crystal frame (latent) |
| `pose_to_pose_dockq.py` | ranking ipTM copied from the crystal-scored pool csv (finding 2) |

**Now** (`d69dfd3`, `d971b9d`, `8f54e26`):

1. Antibody/antigen labels come from the benchmark's role file through each pose's crystal chain map,
   and **every copy of a molecule takes the same label** (identical residues). The benchmark often
   annotates one copy and the crystal map lands it on whichever copy superposes best; labelling only
   that chain would say which copy binds. Chains the benchmark does not annotate (33 entities in 22
   complexes) get no label and join no interface.
2. The anchor is the pose with the highest mean `chain_pair_iptm`, from its own confidence json, over
   the antibody–antigen pairs in contact in that pose.
3. Interfaces are anchor antibody × antigen pairs in contact (5 Å) in at least one pose: 432.
4. Ranking ipTM is read from each pose's json at the chains it is compared on and templated on.
5. `fetch_spurious_interface_iptm.py`, the "spurious" branch, the filter's crystal-frame and reuse
   branches, and the crystal defaults for maps and roles are deleted.

**Checks.** A 6-complex end-to-end test (8ror, 8w5e, 8w5v, 9ban, 9h39, 9hi4 — duplicate copies,
modified residues, name collisions): 1,500 poses, 0 failures, every copy labelled alike. The full
build: 123 anchors, 30,750 poses mapped, 0 failures; 414 matrices (18 interfaces below the
5-contacting-pose minimum, logged); 3,524 donors; preflight clean. 8ror (single copy) selects the same
50 donors, tags and template chains as before.

**Tried and dropped.** Labelling from ANARCI on the input sequence (misses 9kqc's shark VNAR), and
matching pose chains to input entities by sequence (breaks on modified residues; 9hi4 has two
identical entities told apart only by a phosphoserine). Both were more machinery than the rule needs.

**Open.** The 33 unannotated entities are unannotated because the benchmark's scored crystal
interfaces did not include them (e.g. 8uzp's light chain), so which chains may form interfaces is still
slightly shaped by the crystal. To close it: label every chain from 1-D metadata (entity
descriptions), checked by hand, and measure how the donors in those 22 complexes change.

## 2. Ranking at the wrong chains

`pose_to_pose_dockq.py` stored each pose's `pair_iptm` from the pool csv, read at crystal-matched
chains, while the matrix compared — and the template pinned — the anchor-frame chains. On
duplicate-copy complexes these differed for 16,854 of 69,936 poses.

- Measured on 9h4r_AC: the stored value equalled the ipTM at the crystal-matched chains for 238 / 238
  poses, and at the compared chains for 51 / 238.
- An independent re-selection with corrected ipTM changed donors in 43 complexes, all duplicate-copy;
  the pipeline's corrected selection matched it exactly, 123 / 123 complexes.
- Fix `c120983`: `pool_index.pair_iptm_at()` reads `chain_pair_iptm` from the pose's json at the
  compared chains; identical to the independent re-implementation, DockQ unchanged.

## 3. Overwritten poses

`runmeta.run_id()` drew from the global `random`, which Protenix re-seeds with the fold seed before
every fold. Every templated fold uses seed 430000, so directory names repeated: 173 folds in 27
directories. Scores were intact (each fold is scored from its own subdirectory before its poses move);
154 folds lost their poses (listed in the old arm's `poses_overwritten_2026-09-25.tsv`).

Fix `2fd17b8`: `run_id()` uses `SystemRandom`; the run directory is created exclusively and pose moves
refuse to overwrite. `tests/test_folder.py` passes on the fix and fails on the old code. Every new fold
since has its own directory and keeps its five poses.

## 4. Claims

`folder.reconcile_claims` decided whether a claimed fold was scored by `(complex, seed)`. Every
templated fold shares seed 430000, so one scored fold made every dead claim in its complex look scored,
and a fold killed mid-run was never retried (4 stranded). Fix `378f1c4`: keyed by
`(complex, tag, seed)`; test in `tests/test_folder.py`.

## 5. Name collisions ("spurious" interfaces)

Crystal interfaces were named in crystal letters and predicted-only ones in anchor letters, so six
names meant two different chain pairs: `merge_roles` kept the crystal one, every pose appeared twice in
their matrices, and the predicted-only pair was never ranked. With one naming scheme (finding 1) the
distinction and the collisions no longer exist.

## 6. Reporters

`score_experiment.py` took its baseline from the selection matrices (mixing chain choices after finding
2), labelled classes on the 95-seed pool, and read global matrix directories; `compare_arms.py` used
the first-50-seed pool csv. Both counted an interface finished once the folds *motivated by it* were
scored, though every fold scores every interface. Fix `63ca94c`: one `pool_by_seed`, `classes_at` and
`load_scored`, shared; finished = every queued fold of the complex has rows. On the old arm the two
reporters now agree exactly.

## 7. Guards that were missing

- `build_templates.py` reused any cached namespace; ids are re-assigned on every build, so a
  re-selection could pair a job with another donor's cif. Now reused only if donors and ids match
  (`4cbe6ca`, tested).
- `preflight.py` now checks that every template is the queued donor and every `templatesPath` is
  absolute and exists (`f3d8e05`).
- The orchestrator's DockQ step counted a non-empty directory as done; an interrupted build skipped 315
  of 339 interfaces until its validator caught it. It now always runs and resumes per interface
  (`6b05044`).
- `build_free_queue.py` wrote the queue before refusing, checked the wrong seeds, and did not avoid the
  seeds of the run it was extending (`5b3cea3`, `a5d0bd9`).

## Checks behind the current results

| check | result |
|:---|:---|
| template features equal the donor pose (8ror + 20 jobs) | residues 1,897 / 1,897, atoms 0.0000 Å, distogram 100% incl. cross-chain |
| carried rows vs the old arm | 25,060 / 25,060 identical; no duplicates; carried ∩ to-fold = ∅ |
| selection rule | all 3,524 donors in seeds 300000–300049, legacy directory, none repeated |
| first new folds | own directory, 1 template = queued donor, 5 poses kept, rows match queue and template |

## Compute matching

Arm A spends 3,524 donor folds. Five complexes had fewer extra scored free seeds than donors (checked
per interface, as `compare_arms.py` does): 8w5v 1, 9e7n 1, 9g4f 1, 9myl 2, 9nn9 7. A 12-fold free top-up
(`free_compute_matched/queue_free_topup_arm_a.tsv`, seeds that run had not used) runs on each GPU after
its fixed-arm worker, before the variable-seed arm (`a5d0bd9`).

## Throughput: overlapped scoring

Measured on the first 558 fixed-arm folds: the GPU was idle ~13% of wall time between folds while the
CPU scored the previous one (13-22 s per fold); feature building inside a fold is ~3%.
`folder.py --overlap-scoring` scores each unit on one background thread while the next folds -- at
most one unit is being scored, only that thread writes the csv, the claim is held until scoring ends,
and a scoring exception is a FAILED unit with its claim released. Tested in `tests/test_folder.py`
(overlap happens, scorings never overlap each other, failure releases the claim). Opt-in: the stager
passes it to the variable-seed arm only; the fixed arm runs the sequential path it was verified on.

**Scorer launch failures (2026-09-26, `cdec046`).** On the overlapped path, 14 of the variable-seed
arm's first 787 folds failed to START the scorer: `OSError: [Errno 14] Bad address`, from the background
thread while the main thread folded. The folds succeeded; no rows were written; each claim was released,
so nothing was lost, but a worker passes each queue line once and would have left them unfolded. The
cause is a guess, not tested: Python 3.12 launches subprocesses with vfork. Scorer launches now use
fork + exec (`subprocess._USE_VFORK = False`) and `spawn()` retries a launch that fails with OSError
(safe: nothing ran). The four workers were stopped at 17:08 and relaunched; the failed folds, earlier in
the queue than where the workers had reached, were the first they claimed. Their first, unscored poses
stay on disk in their own run directories.

## Variable-seed seeds = the compute-matched free seeds

The variable-seed queue drew a random subset of 300100-300149 per complex, while free + matched counts
the lowest-numbered extra free seeds (including the deeper pool, 300050-300099): identical seed sets
for only 16 / 123 complexes. Now `build_varseed_queue.py` gives each complex's donors exactly the seeds
`compare_arms.matched_seeds()` counts -- one function for both -- so the two arms spend the same folds;
123 / 123 match. The rule also counts a free seed only if it is scored on every interface of its
complex (9hvk's 300083 had rows for 1 of 4 interfaces). The orchestrator's `--varseed-out` is removed.

## Analysis added

`analysis/crystal_frame_selection.py` → `analysis/crystal_frame_selection.md` (`e245acd`): donor counts
and donor quality for the old arm, A (crystal-free) and B (the crystal's 292 interfaces and chain maps,
everything else A's).

## Tidy-ups

`self_template_50seeds_pxm.py` renamed `build_experiment.py` (2026-09-27): it builds either dataset, and
the `_pxm` suffix dated from when pxm2025 was the only one. Old `build.log` files keep the old name.

Stager and watcher moved into v2, stager capped at our four GPUs (`7ca54cf`, `cc487fa`); two leftover
`watch.sh` auto-restarters stopped and `watch.sh` removed (`8c85264`); `run.json` records the code the
worker loaded (`2e8c961`); orchestrator output names made descriptive (`25e04ea`); `folder.py
--force`, which could not refold anything, removed (`b5dd74e`); `paths.py` duplicates and
`dockq_torch.py` dead code removed (`dedc709`); tests in one file (`ab41c70`); one-off audit scripts
removed (`d3187da`).
