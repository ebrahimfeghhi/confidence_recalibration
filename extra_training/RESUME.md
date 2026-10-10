# RESUME — state at 2026-10-10 13:10 PST, before a planned server restart

Written because the server restarts **~17:00 PST on 2026-10-10** (~4 h from this snapshot).
Everything below is either safe to lose or has a documented restart command. Read this top to bottom
before restarting anything.

**The PXM generation run will not finish first.** At 13:09 it was at 218/1230 folds, having started at
10:25 — about 79 folds/h on 2 GPUs. By 17:00 expect roughly 500/1230, so it will be interrupted
mid-run and needs the resume procedure in §1.

---

## 1. What is RUNNING and will die on restart

**Free-seed generation, 10 extra seeds per complex, both datasets.** Launched 2026-10-09 16:22.

| | |
|---|---|
| FoldBench | **done** — 1042/1040 folds, 7,995 rows, 0 failures |
| PXM-2025 | **in progress** — 218/1230 folds, 3,220 rows, 0 failures |
| tmux sessions | `fx_chain`, `fx_pxm2025_0`, `fx_pxm2025_1`, `fx_watch` |
| arm dirs | `experimental_data/protenix/{foldbench,pxm2025}/free_extra10` |
| watcher log | `experimental_data/protenix/free_extra10_watch.log` (0 ESCALATEs) |

Poses and scored rows already written are safe — they are on disk, not in memory. Only the in-flight
folds (at most 2) are lost.

### How to resume it after the restart

The worker claims work with atomic `mkdir` under `<arm>/claims/`, so a hard kill leaves **stale claim
directories** for folds that never finished. PXM currently has 220 claims against 218 finished folds,
so expect ~2 stale. Clear them first or those complexes are skipped forever:

```bash
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2
/home/ebrahim/Protenix/.venv/bin/python reconcile_stale_claims.py \
    --exp ../experimental_data/protenix/pxm2025/free_extra10
```

Then relaunch the two PXM workers (GPU 0 and 1) exactly as `run_free_extra10.sh` does. The FoldBench
phase is complete and must **not** be relaunched. `fx_chain` has already handed off to PXM, so the
chain script does not need to run again — launch the PXM workers directly.

Check `run_free_extra10.sh` for the exact worker invocation rather than reconstructing it.

### Standing constraints on any relaunch

- Never more than 4 GPUs; 2 is what this run uses.
- Max 64 CPU cores total.
- Kill by PID, never by pattern.
- Never delete poses.

---

## 2. What is NOT running, and must not be started before the restart

The MSA database build. A 107 GB download would be killed mid-flight, and `/var/tmp` may be cleared on
boot, so there is nothing to gain by starting now.

**Nothing has been downloaded.** An earlier launch of `setup_databases.sh` was stopped and fully
cleaned up: no `/var/tmp/colabfold_db`, no partial tarballs, quota unchanged.

### What IS already in place

| | |
|---|---|
| `tools/mmseqs/bin/mmseqs` | MMseqs2 17-b804f, static build |
| `tools/hmmer-3.1b2-linux-intel-x86_64/binaries` | HMMER 3.1b2 (ANARCI needs it; system 3.4 breaks ANARCI's parser) |
| `tools/anarci_hmms/ALL.hmm` | ANARCI's HMMs, re-pressed with 3.1b2 |
| `tools/colabfold/setup_databases.sh` | ColabFold DB setup script, downloaded |
| `envs/colabfold/bin/colabfold_search` | installed from the ColabFold repo, `--help` verified |
| `extra_training/data/` | SAbDab2 v0.2.0 tarball + 4 CSVs, 39 RCSB mmCIFs |

---

## 3. The plan, in order

Goal: fine-tune Protenix v2's confidence head on **our own pose distribution**, using their own FT3
stage, and measure whether pose *selection* improves. See `README.md` in this folder for why.

### Phase 0 — verify staging works (minutes, no big download)

`DEBUG_MINI_DB=1 bash setup_databases.sh <dir>` substitutes a 280 MB `mini_swissprot` for both large
databases. Use it to confirm the three-phase sequence below actually runs with only one database
resident at a time — by moving the other database's files aside between phases.

This was about to run when the restart was announced. **It has not run yet.**

What the code review already established (`colabfold/mmseqs/search.py:129-160`):

- The uniref search is skipped when `uniref.a3m` exists in the output dir → phase 2 is resumable.
- The env search is driven from `prof_res`, written to the **output** dir in phase 1, not from the
  uniref database → so uniref does not need to be resident in phase 2.
- **Caveat to test:** the *pairing* path is a separate function that searches uniref itself, so
  pairing must happen in phase 1 while uniref is resident.

### Phases 1-4 — the real build

| phase | action | peak disk |
|---|---|---|
| 1 | download + unpack `uniref30_2302`, search, keep `uniref.a3m` + `prof_res`, delete DB | 367 GB |
| 2 | download + unpack `colabfold_envdb_202108`, search (resumes from `prof_res`), delete DB | 459 GB |
| 3 | download `pdb100` + PDB mmCIF (92.3 GB measured), template search, delete | ~136 GB |
| 4 | convert to Protenix per-chain `pairing.a3m` / `non_pairing.a3m` / `hmmsearch.a3m` | ~25 GB permanent |

**Sequences to search: 6,048** = 4,672 unique from the 557 training clusters + 1,376 from our existing
benchmarks (632 FoldBench + 744 PXM).

Why re-do the benchmark MSAs: ours were built on `uniref30_2103`, which is **no longer downloadable**,
so new data would be on `2302`. Rebuilding both on 2302 removes that confound, and the databases are
transient — doing it later means re-downloading 236 GB. Marginal cost now is +29% sequences, ~7 GB.

**Do not overwrite the existing 2103 MSAs** (user instruction). Write benchmark output to a new
directory alongside the old, so every number already reported stays interpretable.

### Decisions already made (do not revisit)

- Training set: **all 557 clusters**, not the 162 novel-only subset.
- **Templates are included.** Do not pass `SKIP_TEMPLATES`.
- Databases live **in `tts_abag`**, not `/var/tmp`.
- Do **not** clear `~/.cache/huggingface` — it holds the ESMFold2 / ESM-C / ESM3 weights and a gated
  auth token. Protenix weights are elsewhere (`~/checkpoint/protenix-v2.pt`) and unaffected.

### Disk

Quota is **420 GB of 1024 GB**, so 604 GB headroom. Phase 2 peaks at ~459 GB, which fits. 461 GB was
freed earlier today by deleting trunk `.npz` dumps (`self_distill_dropout/{anchor10,offline10}/trunks`,
`pose_branch_300046/trunks`, and scattered `.npz` under `steer_8w5q_*` and `refold_fp32_9g9p_s50`).
All poses were preserved and verified.

If phases turn out to need ~800 GB, the user has offered to arrange more space — ask.

---

## 4. After the MSAs: the training plan

### Data prep

Protenix's preprocessor (`Protenix/docs/prepare_training_data.md`) turns CIFs into one
`[pdb_id].pkl.gz` bioassembly dict plus an **indices CSV**. Put SAbDab2's `ab_ag_cluster` in the
`cluster_id` column — the dataloader samples by cluster weight, which is mandatory here: cluster 171
alone holds 1,001 structures of the same antibody, and cluster 2 holds 850.

### FT3, unmodified, as the baseline

Their recipe (`Protenix/docs/training_inference_instructions.md:238-250`): `train_confidence_only=true`,
`alpha_pae=1.0`, `alpha_diffusion=0`, `alpha_distogram=0`, crop 768, `diffusion_batch_size=32`,
lr 0.0018 cosine with 1,000 warmup, `max_steps` 20,000. `finetune_demo.sh` is a working launch
template. Only the confidence head updates — the trunk is frozen three ways (`loss_fns={}`,
trunk modules in `.eval()`, and `stop_gradient` detaching `s_inputs`/`s_trunk`/`z_trunk`).

Step count is the only knob being tuned. Two couplings to respect: the LR schedule is defined over
`max_steps`, so stopping early is not the same as training with a smaller `max_steps`; and 1,000
warmup steps is a large fraction of a short run.

### Validation

Hold out ~50 clusters, chosen **randomly by cluster**. Protenix itself has no early stopping and
validates on complex-level lDDT over `recentPDB` + PoseBusters — neither antibody-specific nor
DockQ-based, so we are adding our own criterion.

Stop on the **paired per-interface change in selected DockQ** and within-fold Spearman of `pair_iptm`
against DockQ — not on the confidence loss, and not on the binary acceptable rate (underpowered at
n≈25 informative interfaces). Drop interfaces whose pool max `pair_iptm` exceeds 0.8.

Because the trunk is frozen, `s` and `z` are identical across checkpoints: generate the held-out poses
**once**, caching `z`, `s_trunk`, the poses and their DockQ together so they stay matched, then every
checkpoint rescores from that cache. 32 complexes × 10 seeds in fp32 is ~210 GB — budget for it.

### Final test comparison

**Do not try to rescore the existing 112,500 poses.** We did not save `z`/`s_trunk`, and re-running the
trunk will not reproduce them — repeating a seed re-rolls the outcome (9n6e_GF went 0.022 → 0.644 on
repeats of one seed). Pairing a saved pose with a freshly computed `z` would score it against the wrong
context.

Instead: generate 10-20 seeds for the test set and run **both** confidence heads in the same fold, old
and new, on the same poses from the same `z`. Perfectly paired, nothing to store.

### Baseline to beat (free-50 pool, measured)

| dataset | argmax-ipTM acceptable | oracle acceptable |
|---|---|---|
| FoldBench | 74% | 92% |
| PXM-2025 | 49% | 79% |

---

## 5. Open questions for the user

- Nothing blocking. Phase 0 is approved and ready to run once the server is back.
- If the phased download turns out to need ~800 GB rather than 459 GB, ask about more space.

## 6. Loose ends, not blocking

- **CDR-L3 check** for `8r4u`, `9gry`, `9gs3`, `9i20` — light-chain-only structures with no heavy chain,
  so CDR-H3 is undefined and they are unchecked at the antibody level. Their antigens were checked.
- **ML-grade filter audit** of the 39 complexes SAbDab2 omits. If resolution is the common cause, the
  training pool is systematically higher quality than our test benchmarks — a train/test distribution
  shift worth stating either way.
- `budget_decomposition.py` has a known **class-conditioning flaw**: classes are assigned from the
  50-seed pool, then 10 vs 50 is compared within class, which conditions on the outcome at 50. The
  class-free ALL rows are sound; the per-class rows need redoing before publication.
