# Handoff — self-templating vs free sampling, four arms

Written 2026-09-25 17:04. Read this first, then `git log` for the reasoning behind each fix.

## The question

Does self-templating on a free pose rescue a confident decoy, *at matched compute*, better than
spending the same folds on more free seeds? Four arms, all selected the same way — top pooled
`chain_pair_iptm`, DockQ read off the selected pose, acceptable at DockQ >= 0.23.

| arm | directory | what it is |
|:---|:---|:---|
| free-50 | `experimental_data/free/` | the 50-seed pool, seeds 300000-300049. Also the donor pool every templated fold templates on. NEVER rename: 44,769 queue lines and 33,290 results rows point at it by absolute path. |
| free-extra | `experimental_data/free_compute_matched/` | extra free seeds 300100-300149, compute-matched to what the method spends |
| self-tmpl-fixed | `experimental_data/self_template_fixed_50_seed/` | every templated fold on seed 430000 |
| self-tmpl-varseed | `experimental_data/self_template_variable_50_seed/` | one seed per donor, drawn from 300100-300149 — the same 50 the free-extra arm used |

`experimental_data/` also holds `archive/`, the shared input pool `pxm2025/`, the `matrices*` and
`anchor_frame/` the pipeline reads, and `reffree_50/` (provenance only, see below). Everything else was
deleted on 2026-09-25 — eleven superseded runs, 66,690 poses, on the user's explicit instruction.

## State as of 17:04

| arm | progress | workers | GPUs |
|:---|---:|---:|:---|
| free-extra, baseline | 1,296 / 1,297 DONE | — | — |
| free-extra, top-up | 109 / 215 | 2 | 1, 5 |
| self-tmpl-fixed | 162 / 757 | 2 | 0, 2 |
| self-tmpl-varseed | 0 / 3,342 | 0 | queued behind the fixed arm |

Disk 815 G of a 1,024 G quota. **Hard rule: never more than 4 GPUs without asking.** GPU 4 belongs to
another user — never count it as available.

`self_template_fixed_50_seed/results.csv` holds 36,185 rows. **33,290 of those are carried** from
`reffree_50`, retagged to the new selection's arm labels; the rest are folded here. That is why
`reffree_50/` is kept: its `results.csv` is the provenance for a quarter of this arm's numbers, and it
holds 531 folds whose scores exist nowhere else. Its `ns/` (1.1 GB) is regenerable and can be deleted.

### Remaining work, measured

30.1 GPU-hours for the fixed arm: 15 h on 2 GPUs, 7.5 h on 4. Concentrated in a few complexes —
**9vor is 8.0 h of it alone** (24 folds x 1,205 s). Per-fold cost ranges 20 s to 20 min, so a
folds-per-hour rate is meaningless without knowing which complex is in flight. Recompute the total the
way the 17:00 estimate was made: median measured seconds per complex x folds still unclaimed.

## What is running

- `tmux stagearms` — `diffusion_commit_clean_v2/stage_arms.sh` (replaced 2026-09-25 17:23). Runs one chain per
  GPU on 0, 1, 2, 5 only: free top-up -> fixed arm -> varseed arm, each GPU moving on as soon as its own
  worker exits. Never holds more than 4 GPUs. Log `experimental_data/stage_arms.log`. The old
  `experimental_data/stage_arms.sh` hardcoded GPUs 1,5,6 and would have put the fixed arm on 5 GPUs.
- `tmux watcharms` — `diffusion_commit_clean_v2/watch_arms.sh`, every 30 min. Checks claims AND scored rows
  **separately**, because they fail separately: a worker alive with claims growing and `results.csv`
  flat is the state that hid 603 failed folds. Log `experimental_data/watch_arms.log`.
- Four folding workers, sessions `<arm-dir-basename>w<gpu>`.

Relaunch a worker with absolute paths only — `call_folder.py` does `cd diffusion_commit_clean_v2`
first, so relative paths silently break:

    PY=/home/ebrahim/Protenix/.venv/bin/python
    R=/home/ebrahim/tts_abag/experimental_data/self_template_fixed_50_seed
    $PY diffusion_commit_clean_v2/call_folder.py --queue $R/queue_tofold.tsv --out $R/poses \
      --csv $R/results.csv --claim-dir $R/claims --log-dir $R --templates $R/templates --gpus 0 --run

**Never pass `--drop-cifs`.** It is off by default; `reffree_50` was run with it and its poses are gone.

## The selection, and how it was built

`self_template_50seeds_pxm.py` runs steps 1-7 (pool -> anchor -> interfaces -> spurious -> dockq ->
filter -> templates -> preflight) and then diffs against an earlier run. The rule is hardcoded: **seeds
300000-300049, `legacy-no-provenance` fold directory only**. It validates after every step and stops on
violation. Two interpreters, because no single env has everything: `anchor` uses `paths.PXMETER_PYTHON`,
the rest `paths.PROTENIX_PYTHON`. Step 5 shards DockQ across `DOCKQ_GPUS` with a 30-minute deadline,
after which extras stop and the first GPU finishes alone.

Current selection: 3,342 folds, 123 complexes, 308 interfaces, cap 50 donors per complex. Diff against
`reffree_50`: **2,585 carry over, 757 to fold, 531 omit** — of which 302 are the out-of-range donors the
whole exercise was meant to remove. 229 of the omissions are unexplained; that is bookkeeping only, the
new selection stands on its own and does not need them.

The varseed arm **reuses the fixed arm's `templates/`**. A template is a donor pose plus its MSA hit and
carries no seed, so sharing the namespace is what makes the two arms comparable — and it saved
rebuilding 1.7 GB. Its queue assigns each complex a random permutation of the 50 seeds, one per donor,
seeded per complex (`random.Random(f'varseed-{pdb}')`) so it is reproducible.

## Bugs fixed on 2026-09-25 — do not reintroduce

Every one of these was invisible from the outside: workers alive, claims growing, no errors.

1. **Spurious roles never merged.** Roles are written by two steps — `anchor_relabel.py` for the 292
   reference interfaces, `fetch_spurious_interface_iptm.py` for the 53 the predictions invent.
   `pose_to_pose_dockq.py` builds a matrix only for an interface it can find roles for, so passing only
   the anchor file silently dropped all 53 — the confident-decoy case itself. `merge_roles()` now builds
   the merge in both frames and a guard asserts every queued interface has roles before any GPU time.
2. **Relative `templatesPath`.** `build_templates.py` wrote it relative; `folder.py` runs with its own
   cwd, so Protenix never found the per-donor a3m. 603 folds failed, none succeeded, and it was reported
   as healthy twice. Absolute now, which also survives a rename.
3. **`os.replace` on a non-empty directory.** A unit scores every interface of its complex and each job
   writes its own `*_json`; the second collided and the exception killed the whole worker. Merged
   file-by-file now.
4. **Claim stealing.** `reconcile_claims` released any unscored claim as "killed mid-fold" — which is
   indistinguishable from a live worker's in-flight claim, so every launch stole a fold and folded it
   twice. Claims now record host+pid and are released only when the owner is gone.
5. **Header-only csv counted as a finished step**, so the pipeline skipped it and fed an empty chain map
   downstream.
6. **Two vacuous validators** — `dockq` counted its own matrices (292 looked fine when 345 interfaces
   had been declared upstream); `templates` counted `ns/`'s three top-level entries, so it always
   passed. Both now compare against the previous step.

Lesson worth keeping: a check that only inspects a step's own output cannot catch a step that lost data.

## Open items

- **Overwritten poses (fixed 2fd17b8).** `runmeta.run_id()` drew from the global `random`, which Protenix
  re-seeds per fold, so every seed-430000 fold reused a few directory names. As of 2026-09-25 18:10,
  154 of 181 fixed-arm folds have their poses overwritten; every one still has its scores in
  `results.csv` (scored from its own subdirectory before the move). Per-fold list:
  `experimental_data/self_template_fixed_50_seed/poses_overwritten_2026-09-25.tsv`. Workers started
  before 2fd17b8 (GPUs 0 and 2) keep the old code until restarted. Poses are not needed; not refolded.

- **Run data in git.** `results.csv` and `index.tsv` stay tracked (a few MB each; provenance for every
  number). Commit them at milestones -- an arm finishing -- not while workers are writing them.
  `.watch_state` is gitignored.

- **Renaming a directory under running workers kills them.** That is how the baseline died: three
  workers held absolute paths to `free_matched/`. Their logs moved with the directory, so the
  tracebacks in `free_compute_matched/w1.log` and `w5.log` are pre-rename history, not new failures.
- `paths.SELFTMPL_DIR` still defaults to the deleted `self_template/`.
- `experimental_data/STRUCTURE.md` is regenerated by a script inlined in the git history; it is current
  as of the eleven-directory deletion.
- `compare_arms.py` defaults now point at `self_template_fixed_50_seed` and `free_compute_matched`.
  Run it when the fixed arm reaches 757/757. It excludes interfaces whose folds are unfinished, so it
  will under-report until then — at 162/757 it excluded 87.
- **Compute matching is fair to within 2%.** Measured across six complexes folded by both arms, method
  folds average 179.2 s against free's 182.7 s (ratio 0.98) — the single template costs slightly less
  than free's 3-4 PDB templates, but the trunk dominates. One complex came out 1% the other way.
- A design question the user set aside: `point_templates_at()` **replaces** Protenix's 3-4 PDB templates
  with the one self-template rather than adding to it. So a rescue could come from the self-template
  helping or from removing the PDB templates. Separating those needs a third condition.

## Working rules for this user

Lead with the answer, keep it short. Report n on every row. Per-interface `chain_pair_iptm`, never
global ipTM. Markdown tables, not box-drawn. Measure before claiming — no cost, performance or
causality assertions without evidence; say "I cannot measure that yet" rather than estimating.
Never launch a run, kill a process, or exceed 4 GPUs without explicit approval. Never `pkill -f`; kill
by PID. Never delete poses. Everything lives in `tts_abag`, never `$HOME`.
