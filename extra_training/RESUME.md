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

## 2. The MSA database build — RUNNING, resumable

**`uniref30_2302` is downloading** into `tools/colabfold/db/` via
`extra_training/scripts/archive/phase1_uniref.sh`, launched in tmux session `msa_p1`, logging to
`tools/colabfold/phase1.log`.

The restart will almost certainly interrupt either the download or the index build that follows.
**Both resume.** After the restart:

```bash
cd /home/ebrahim/tts_abag/tools/colabfold
cp /home/ebrahim/tts_abag/extra_training/scripts/archive/phase1_uniref.sh .   # if missing
tmux new-session -d -s msa_p1 "bash phase1_uniref.sh > phase1.log 2>&1"
```

It skips any download already complete (size-checked, `curl -C -`), and `setup_databases.sh` skips
stages whose `*_READY` marker exists. Safe to re-run as many times as needed.

**Do not** run `setup_databases.sh` directly: its `DOWNLOADS_READY` block fetches uniref30 and the
128 GB envdb together, which would need ~825 GB. The wrapper downloads one database at a time and
pre-creates the other stages' markers.

### What is already in place

| | |
|---|---|
| `tools/mmseqs/bin/mmseqs` | MMseqs2 17-b804f, static |
| `envs/colabfold/bin/colabfold_search` | installed, **both patches applied** |
| `envs/.../colabfold/mmseqs/search.py.orig` | pristine copy, for regenerating the patch |
| `tools/hmmer-3.1b2-*/binaries` + `tools/anarci_hmms` | for ANARCI only; system HMMER 3.4 breaks it |
| `tools/colabfold/db_mini` + `test/` | the 280 MB validation harness, still on disk |
| `extra_training/msa_queries/` | query sets, `training_instances.txt`, manifests |

### The two patches — verify after any venv rebuild

```bash
envs/colabfold/bin/python -c "import colabfold.mmseqs.search as m; print(m._emit_taxid(), m._keep_intermediates())"
# expect: True False
```

If the venv is ever recreated, re-apply `extra_training/phases/phase_d_colabfold_search_staging.patch`.
Without patch A there is no paired MSA; without patch B the staged build cannot work at all.

### Decisions settled (do not revisit)

- **`uniref30_2302`**, not 2103. It is the current default in Protenix's own
  `scripts/colabfold_msa.py`. 2103 is what our benchmark MSAs used, but gwdg serves it at 860 KB/s
  (~24 h for 75 GB) and refuses parallel ranges, against 49 MB/s from opendata.
- **Benchmark MSAs are rebuilt on 2302 too**, in the same pass, so train and test match. The existing
  2103 MSAs are **kept, not overwritten**.
- **Templates are in**, via Protenix's `runner/template_search.py` — `hmmsearch` against
  `pdb_seqres_2022_09_28.fasta`. **No pdb70 and no 92 GB PDB mmCIF mirror.**
- Databases live in `tts_abag`, not `/var/tmp`.
- Do **not** clear `~/.cache/huggingface`: ESMFold2 / ESM-C / ESM3 weights and a gated token.
- CPU cap **64 cores**.

### Disk

Quota was 449 GB of 1024 GB with the download partway. Phase 2 peaks at ~459 GB. 461 GB was freed
earlier by deleting trunk `.npz` dumps; all poses were preserved and verified.

## 3. The pipeline

**`extra_training/MSA_RUNBOOK.md` is authoritative** and executable on a fresh machine. It was
rebuilt against the Protenix codebase after an earlier draft got three things wrong. In short:

1. extract the 4,555 selected SAbDab2 instance cifs
2. `scripts/msa/step1-get_prot_seq.py` -> unique sequences + `seq_to_pdb_index.json`
3. phase 1: `colabfold_search --use-env 0` on uniref30_2302, then delete that database
4. phase 2: `--use-env 1` on colabfold_envdb, resuming from `prof_res`
5. `step3-uniref_add_taxid.py` then `step4-split_msa_to_uniref_and_others.py` -> `pairing.a3m` /
   `non_pairing.a3m`
6. `runner/template_search.py` -> `hmmsearch.a3m`

**The thing most likely to be got wrong:** Protenix does not use ColabFold's pairing path. It searches
each unique sequence once *with taxonomy*, then reconstructs pairing from species. That is why patch A
exists -- `step3` reads `uniref_tax.m8`, and `step4` splits on `>UniRef100`.

### Validated on the mini database

`DEBUG_MINI_DB=1` substitutes a 280 MB `mini_swissprot` for both large databases.

- **staging**: uniref30 physically absent during phase 2, depth roughly doubles
  (1ejo_H 268 -> 531, 1ejo_L 422 -> 865, 1j1p_H 180 -> 403, 1j1p_L 334 -> 728, 1j1p_ag0 110 -> 270),
  both staging log lines present, exit 0
- **taxids**: `uniref_tax.m8` carries real annotations (`101 P84751 10090 Mus musculus`)
- **leakage**: `train.fasta` and `bench.fasta` share 0 sequences

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

- Nothing blocking. Phase 0 is done and passed; the phase-1 download is running and resumes itself.
- Templates for TRAINING are settled. Whether the FoldBench evaluation arm should also use templates is open: PXM folds with `free_templates=True`, FoldBench with `False`, so a head trained with templates would be evaluated without them on FoldBench.

## 6. Loose ends, not blocking

- **CDR-L3 check** for `8r4u`, `9gry`, `9gs3`, `9i20` — light-chain-only structures with no heavy chain,
  so CDR-H3 is undefined and they are unchecked at the antibody level. Their antigens were checked.
- **ML-grade filter audit** of the 39 complexes SAbDab2 omits. If resolution is the common cause, the
  training pool is systematically higher quality than our test benchmarks — a train/test distribution
  shift worth stating either way.
- `budget_decomposition.py` has a known **class-conditioning flaw**: classes are assigned from the
  50-seed pool, then 10 vs 50 is compared within class, which conditions on the outcome at 50. The
  class-free ALL rows are sound; the per-class rows need redoing before publication.
