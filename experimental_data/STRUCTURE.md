# experimental_data

The arms, the inputs the pipeline reads, the paper analyses, and an archive. Most of this is not in
git: poses (`.cif`), DockQ matrices (`.npz`), MSAs and per-pose confidence json are excluded by size.
Queues, selections, scores and diffs are committed — enough to read an experiment without rerunning it.
Code lives in `../diffusion_commit_clean_v2/`; see its `README.md` and `HANDOFF.md`.

**ONE DIRECTORY PER BENCHMARK.** Everything belonging to a benchmark lives under
`protenix/<benchmark>/`, and its pooled free scores are `protenix/<benchmark>/free_pool/results.csv`,
reachable without knowing any of this as `paths.DATASETS[ds].free_pool`. Six rows of this table used
to name top-level directories that had already been migrated under `protenix/pxm2025/`, which is how
an analysis came to read half a benchmark; if you move something, update this table in the same
commit.

| directory | size | what it is |
|:---|---:|:---|
| `protenix/foldbench/` | 68 G | **benchmark root.** Every FoldBench arm, its job jsons, references, and `free_pool/` |
| `protenix/pxm2025/` | 74 G | **benchmark root.** Every PXM-2025 arm, plus `free/` — the free pose pool, seeds 300000–300049, every template's donor. **Never rename** |
| `pose_dockq_matrices/` | 166 M | pose-to-pose DockQ over the free-50 pool, one npz per interface, one subdirectory per (dataset, frame) — **read its `README.md` first**. `pxm2025_crystal` (292, symlink into `analysis/`) and `foldbench_crystal` (160) are COMPLETE for their pools; get them as `paths.DATASETS[name].matrices` and never rebuild them. `pxm2025/`, `foldbench/` are the anchor-frame sets. One other 292-npz set elsewhere looks identical and is WRONG — the README names it |
| `analysis/` | 52 M | intermediates of `diffusion_commit_clean_v2/analysis/` (e.g. the crystal-frame selection B) |
| `cost_model/` | 12 K | the fold-time fit, `t ≈ 0.00182·N^1.57` s |
| `archive/` | 15 G | earlier eras; `archive/superseded_2026-09-25/` holds the crystal-framed pipeline's outputs (`anchor_frame/`, `matrices*/`) and `reffree_50/` |

### Still loose at the top level — all six are PXM-2025 arms

They belong under `protenix/pxm2025/` and have not been moved because they store their own absolute
path in their own files: `redock_branch` in 1,812 files, `redock_probe` in 111, `selftmpl_mask_on_9g9p`
in 43, the rest in 4–5 each. A plain `mv` breaks those; moving them means rewriting the paths, which
is what `diffusion_commit_clean_v2/migrate_pxm_layout.py` exists for.

| directory | size | what it is |
|:---|---:|:---|
| `redock_branch/` | 2.0 G | ARM: does a masked 10th recycle add anything (8w5d) — see below |
| `redock_probe/` | 662 M | ARM: mechanism test for "Redock" (8w5f) — see below |
| `selftmpl_mask_on_9g9p/` | 97 M | ARM: mask-ON self-templating on 9g9p |
| `refold_fp32_9g9p_s50/` | 27 M | ARM: 9g9p refolded in **fp32**, 50 seeds |
| `cycle_decode_9g9p/` | 58 M | ARM: per-recycle decode on 9g9p |
| `refold_fp32_9g9p/` | 5.4 M | ARM: 9g9p refolded in **fp32** |

The two `refold_fp32_*` arms are the only fp32 folds outside `protenix/pxm2025/free`'s
`freefp32r1/r2` — the evidence to look at before spending GPU on an fp32 determinism test.

Top-level logs: `stage_arms.log` (GPU handoffs), `watch_arms.log` (30-minute checks).

## `protenix/foldbench/` — the mask-ON self-templating arms

`PROTENIX_MULTICHAIN_MASK=1` masks the template's cross-chain features, so a donor's ARRANGEMENT is
not transmitted — only each chain's internal conformation. (`folder.py`'s `TMPL_ENV` pins this to `0`,
pose carried, for the ordinary self-templating arms, so these are the deliberate control.)

| directory | folds | what it is |
|:---|---:|:---|
| `free/` | — | the FoldBench free pose pool; every donor and every baseline number comes from here |
| `free_compute_matched/` | — | extra free seeds (151–200) so the baseline can be granted the method's fold cost back |
| `selftmpl_mask_on/` | 3209 | mask ON at **`--ncycle 5`**. PAUSED at 2032/3209. Result: 7/27 decoys rescued, 9/75 solved broken, net −2 acceptable. **Confounded** — the free baseline runs 10 recycles, so every breakage could be the recycle cut rather than the mask. Report: `experimental_results/fb_mask_on_vs_free_x068.md` |
| `selftmpl_mask_on_nc10/` | 3209 | the same arm at **`--ncycle 10`**, which removes that confound. **Read its `README.md`.** Phase 1 (183 folds, the 6 broken complexes) found 5 of 7 breakages recover at 10 recycles; phase 2 is the full benchmark. Report: `experimental_results/fb_mask_on_nc10.md` |

**Compare the two arms at different `--match-factor`.** A 5-recycle method fold costs 0.68 of a free
fold, so `selftmpl_mask_on` is read with `--match-factor 0.68`; a 10-recycle fold costs the same as a
free fold, so `selftmpl_mask_on_nc10` needs `--match-factor 1.0`. Using 0.68 on the nc10 arm hands it
a 32% compute discount it did not earn.

## Inside an arm

| name | what it holds |
|:---|:---|
| `poses.csv` | the pool the selection drew from |
| `chain_frame/` | per-pose contacts and ipTM, the anchor per complex, chain maps onto it |
| `interfaces.csv` | the interfaces, in anchor letters |
| `pose_dockq_matrices/` | a symlink into the top-level `pose_dockq_matrices/<dataset>/` (since 2026-10-02 the npz live there, not per arm). Names are in the arm's ANCHOR chain letters (`chain_frame/`, `interfaces.csv`) — a crystal-frame name looked up here silently misses |
| `queue.tsv` | every selected fold: kind, pdb, interface, tag, seed, donor pose |
| `queue_tofold.tsv` | the folds not carried over from an earlier build |
| `carry_over_folds.tsv`, `diff_vs_*.md` | what was carried, by fold identity (donor, seed, template chains) |
| `templates/` | one namespace per fold: job json, chosen chains, a3m hits; template mmcifs |
| `poses/<pdb>/<seed>-<run id>/` | one directory per fold: `predictions/` (5 poses + confidence) and `run.json` |
| `claims/` | one directory per claimed fold; the workers' locking |
| `results.csv` | one row per scored pose and crystal interface — also the to-do list |
| `w<gpu>.log`, `build.log` | worker and build logs |

## `redock_probe/` — mechanism test for "Redock" (BUILT, NOT RUN)

Does erasing the antigen x antibody block of the recycling carry-over `z` and running 1-3 more
recycles produce a DIFFERENT dock? Implemented as `REDOCK_MASK_AG_ASYM` / `REDOCK_MASK_AB_ASYM` in
`Protenix/protenix/model/protenix.py` (default off, applied after `PROTENIX_LOAD_TRUNK`). Target is
9g9p, which is one antigen and TWO independent nanobodies -- so only A x C (the decoy) is masked and
A x B (solved) stays as a control inside every pose. **Read its `README.md` first**; it carries the
reasoning for zeroing over noise, why no rate parameter is needed, and the three outcomes the probe is
built to distinguish.

### `protenix/pxm2025/selftmpl_mask_on_nc10/` — the pxm2025 twin (queued 2026-10-05)

Same mask-ON intervention at `--ncycle 10`, on pxm2025 instead of FoldBench, to test whether the
FoldBench result (pooled `free + method` oracle +0.067 on decoys at equal compute and pose count)
replicates. Reuses `self_template_variable_50_seed`'s queue and `self_template_fixed_50_seed`'s
template namespace unchanged — nothing rebuilt. Read with `--match-factor 1.0` and the
`free + method` column. **Read its `README.md` first.**

### `redock_branch/` — does a masked 10th recycle add anything? (running 2026-10-05)

20 interfaces (10 decoy, 10 solved, 242-537 tokens), 50 seeds each. Per seed: fold to 9 recycles, then
take the 10th TWICE — once normally (branch A, which IS a normal fold) and once with the antigen x
antibody pair-rep block erased (branch B). Pool the 6 poses and compare against the free pool's 5.
Needs no new code: `folder.py`'s `TRUNK_DUMP_DIR`/`TRUNK_LOAD_DIR` already resume per fold with the
checkpoint loaded once. Trunks are dumped in bf16 and deleted per complex — in fp32 and kept, the
1,000 trunks would have been 150-200 GB against 91 GB free. **Read its `README.md`.**
