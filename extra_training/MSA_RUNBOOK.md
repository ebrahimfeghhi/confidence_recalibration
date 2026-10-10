# MSA build runbook — executable on a fresh machine

Builds the MSAs needed to fine-tune Protenix v2's confidence head on our own pose distribution.
Self-contained: every command, every version, every verification checkpoint. Written 2026-10-10.

**Validated.** The staged approach in §4 was tested end to end on a 280 MB mini database before this
was written. Evidence in §7.

---

## 1. What this produces

| output | size | what it is |
|---|---|---|
| `extra_training/msa_queries/{train,bench}.fasta` | 1.2 MB | the query sets, already committed |
| `extra_training/msa_out/train/` | ~25 GB | one a3m per unique training sequence |
| `extra_training/msa_out/bench/` | ~4 GB | same for our benchmark complexes, on the new DB version |

**5,305 unique sequences**: 4,672 from the 557 training clusters (2,886 structures), 633 from our
FoldBench + PXM-2025 benchmarks. Deduplicated by sequence, because an MSA is per sequence.

Databases are **transient** — deleted after each phase. Only the a3m output is kept.

### Why the benchmark set is rebuilt

Our existing benchmark MSAs were built on `uniref30_2103`, which **404s now**. Any new MSA is on
`2302`, so training and test data would differ in database version. Rebuilding both on 2302 removes
that confound, and costs little while the databases are resident.

**Write to a new directory. Do not overwrite the 2103 MSAs** — every number already reported depends
on them.

---

## 2. Requirements

| | |
|---|---|
| disk | **459 GB peak**, ~29 GB permanent. Phase 2 is the peak. |
| RAM | 1 TB on our box. `--db-load-mode 2` wants ~768 GB free; `0` works with less but is slower. |
| cores | **cap at 32.** More has crashed this server. Use `taskset -c 0-31`. |
| GPU | none — this is CPU only. |
| network | ~347 GB of downloads across the three phases. |

Check quota before starting — on our box each user has a 1 TB cap, and `df` does **not** reflect it:

```bash
quota -s          # space column vs limit column
```

Note quotas charge by **file owner**, so another user's files inside your home do not count against
you (we have 95 GB of someone else's data under `/home/ebrahim` that is on their quota, not ours).

---

## 3. One-time setup

### 3.1 MMseqs2

```bash
mkdir -p $REPO/tools && cd $REPO/tools
curl -sL -o mmseqs.tar.gz https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs.tar.gz
./mmseqs/bin/mmseqs version        # expect 17-b804f...
```

### 3.2 colabfold_search

```bash
cd $REPO
python3 -m venv envs/colabfold
envs/colabfold/bin/pip install -q --upgrade pip
envs/colabfold/bin/pip install -q "git+https://github.com/sokrypton/ColabFold"
envs/colabfold/bin/colabfold_search --help | head -3
```

### 3.3 Apply the staging patch — MANDATORY

**Without this the staged build silently cannot work.** As shipped, a `--use-env 0` run destroys both
things the env phase needs: it `mvdb`s `uniref.a3m` into `final.a3m`, and it *always* `rmdb`s
`prof_res` (the uniref profile the env search is driven from). It also validates that every database
is present before any resume logic runs, and its teardown assumes `tmp/` still exists.

```bash
cd $REPO
S=envs/colabfold/lib/python3.12/site-packages/colabfold/mmseqs/search.py
cp "$S" "$S.orig"
patch "$S" < extra_training/patches/colabfold_search_staging.patch
envs/colabfold/bin/python -c "import colabfold.mmseqs.search as m; print(m._keep_intermediates())"
# expect: False   (True once COLABFOLD_KEEP_INTERMEDIATES=1 is exported)
```

The patch adds four things, all gated behind `COLABFOLD_KEEP_INTERMEDIATES=1` so default behaviour is
unchanged:

1. keeps `uniref.a3m` (`cpdb` instead of `mvdb`) so the env phase can resume from it
2. keeps `prof_res` / `prof_res_h`
3. drops uniref30 from the precondition check when `uniref.a3m` already exists
4. makes the `tmp/` teardown tolerate an already-removed directory

### 3.4 Download the SAbDab2 dataset (only if regenerating the query FASTAs)

The FASTAs are committed, so this is optional.

```bash
mkdir -p $REPO/extra_training/data/zenodo_splits && cd $_
curl -sL -o splits_v0.2.0.tar.gz https://zenodo.org/records/22019991/files/splits.tar.gz
tar xzf splits_v0.2.0.tar.gz splits_final/abag_split.csv splits_final/ab_split.csv \
    splits_final/abag_split_sd.csv splits_final/ab_split_sd.csv
```

**Record 22019991 is v0.2.0.** Record 20083995 is the older v0.1.0 — do not use it.

To regenerate the FASTAs (needs the 39 RCSB cifs too, which `exclude_homologs.py` fetches):

```bash
cd $REPO/diffusion_commit_clean_v2
python exclude_homologs.py --out ../extra_training/results/excluded_homolog_clusters.csv
python extract_msa_queries.py
# expect: train.fasta 4672 sequences from 557 clusters; bench.fasta 633
```

---

## 4. The build, three phases

Set these once:

```bash
export REPO=/path/to/tts_abag
export MMSEQS=$REPO/tools/mmseqs/bin/mmseqs
export CF=$REPO/envs/colabfold/bin/colabfold_search
export DB=$REPO/tools/colabfold/db           # databases, transient
export Q=$REPO/extra_training/msa_queries
export OUT=$REPO/extra_training/msa_out
export COLABFOLD_KEEP_INTERMEDIATES=1        # REQUIRED for staging
mkdir -p "$DB" "$OUT/train" "$OUT/bench"
```

`setup_databases.sh` is resumable — it drops `*_READY` marker files — but the downloads themselves are
**not** (`curl -L -o`, no `-C -`), so a killed download restarts from zero.

### Phase 1 — uniref30 (peak 367 GB)

```bash
cd $REPO/tools/colabfold
export PATH=$REPO/tools/mmseqs/bin:$PATH
UNIREF30DB=uniref30_2302 taskset -c 0-31 bash setup_databases.sh "$DB" 2>&1 | tee p1_setup.log

for SET in train bench; do
  taskset -c 0-31 $CF --mmseqs $MMSEQS --threads 32 \
      --use-env 0 --use-templates 0 --db-load-mode 2 \
      $Q/$SET.fasta "$DB" $OUT/$SET 2>&1 | tee p1_search_$SET.log
done
```

**Verify before deleting anything:**

```bash
ls $OUT/train/uniref.a3m.dbtype $OUT/train/prof_res.dbtype    # both must exist
ls $OUT/train/*.a3m | wc -l                                   # expect 4672
```

If `uniref.a3m` or `prof_res` is missing, the patch did not apply. **Stop** — deleting uniref30 now
would force a 107 GB re-download.

```bash
rm -rf "$DB"/uniref30_2302* "$DB"/uniref30_2302.tar.gz "$DB"/UNIREF30_READY
```

### Phase 2 — colabfold_envdb (peak 459 GB)

Resumes from `prof_res`; uniref30 does not need to be on disk.

```bash
cd $REPO/tools/colabfold
CFDB=colabfold_envdb_202108 taskset -c 0-31 bash setup_databases.sh "$DB" 2>&1 | tee p2_setup.log

for SET in train bench; do
  taskset -c 0-31 $CF --mmseqs $MMSEQS --threads 32 \
      --use-env 1 --use-templates 0 --db-load-mode 2 \
      $Q/$SET.fasta "$DB" $OUT/$SET 2>&1 | tee p2_search_$SET.log
done
```

**Verify the staging actually engaged** — these two lines must appear in the log:

```
Staging mode: uniref.a3m exists, not requiring uniref30_2302_db to be present
Skipping uniref30_2302_db search because uniref.a3m already exists
```

and depth must have gone **up** (roughly 2× in our test):

```bash
grep -c '^>' $OUT/train/$(ls $OUT/train | grep '\.a3m$' | head -1)
```

```bash
rm -rf "$DB"/colabfold_envdb_202108* "$DB"/COLABDB_READY
```

### Phase 3 — templates (peak ~136 GB)

**Templates are required.** Do not pass `SKIP_TEMPLATES` — an earlier attempt to skip them was
rejected. The only legitimate use of `SKIP_TEMPLATES` is the §7 smoke test, to avoid pulling the
92.3 GB PDB mirror while validating logic.

```bash
cd $REPO/tools/colabfold
taskset -c 0-31 bash setup_databases.sh "$DB" 2>&1 | tee p3_setup.log
# downloads pdb100_230517 (19.2 GB) and rsyncs PDB mmCIF (92.3 GB, 260,626 files) — slow

for SET in train bench; do
  taskset -c 0-31 $CF --mmseqs $MMSEQS --threads 32 \
      --use-env 1 --use-templates 1 --db-load-mode 2 \
      $Q/$SET.fasta "$DB" $OUT/$SET 2>&1 | tee p3_search_$SET.log
done
rm -rf "$DB"
```

### Phase 4 — convert to Protenix layout

Protenix job JSONs reference three files per chain by absolute path:

```
pairedMsaPath:   .../prep/<pdb>/msa/<n>/pairing.a3m
unpairedMsaPath: .../prep/<pdb>/msa/<n>/non_pairing.a3m
templatesPath:   .../prep/<pdb>/msa/<n>/hmmsearch.a3m
```

See `Protenix/docs/msa_template_pipeline.md`. **Not yet written** — this is the first unbuilt step.

**Known caveat:** pairing is a separate function (`mmseqs_search_pair`) that searches uniref30 itself,
so if paired MSAs are needed they must be produced in **phase 1**, while uniref30 is resident. Our
existing benchmark MSAs do have `pairing.a3m`, so plan for it. Use `--pair-mode unpaired_paired` in
phase 1 rather than discovering this after uniref30 is deleted.

---

## 5. Then what

Fine-tune the confidence head with Protenix's own **FT3** stage — the final confidence-only stage,
unmodified, as the baseline. Full detail in `README.md` and `RESUME.md` in this folder. Summary:

- `train_confidence_only=true`, `alpha_pae=1.0`, `alpha_diffusion=0`, `alpha_distogram=0`, crop 768,
  `diffusion_batch_size=32`, lr 0.0018 cosine, warmup 1000, `max_steps` 20000
- only the confidence head updates; the trunk is frozen three ways
- put SAbDab2's `ab_ag_cluster` in the indices CSV `cluster_id` column — **mandatory**, since cluster
  171 alone is 1,001 structures of one antibody
- step count is the only thing being tuned
- baseline to beat, argmax-ipTM acceptable: FoldBench 74% (oracle 92%), PXM-2025 49% (oracle 79%)

---

## 6. Gotchas that cost time here

**ANARCI silently returns no domains with HMMER 3.4.** It parses hmmscan's text output through
Biopython's `Hmmer3TextParser`, which cannot read 3.4's format — coordinates come back `None` and
every hit is dropped, with no error. Fix: HMMER **3.1b2** via `hmmerpath=`, HMMs re-pressed with it,
and patch `sys.modules["anarci.anarci"].HMM_path`. Only needed for CDR work, not for MSAs.

**`chain_roles.csv` and our local reference cifs disagree on chain naming** — 9n6e's roles name chains
E/F/G while its local cif holds A,B,C,D,H,I,L. Do not cross-reference them to identify antigens; pull
structures from RCSB and type chains with `hmmscan` instead.

**Repeating a seed does not reproduce a fold.** 9n6e_GF went 0.022 → 0.644 on repeats of one seed. So
saved poses cannot be paired with a freshly computed trunk `z` — anything needing both must save them
together in one pass.

**Do not clear `~/.cache/huggingface`** — it holds ESMFold2 / ESM-C / ESM3 weights and a gated auth
token. Protenix weights live in `~/checkpoint/` and are unaffected.

---

## 7. Phase 0 — the validation, reproducible in ~2 minutes

`DEBUG_MINI_DB=1` substitutes a 280 MB `mini_swissprot` for both large databases, so the whole staged
sequence can be exercised cheaply. `SKIP_TEMPLATES` is used **here only**, to avoid the 92 GB PDB
mirror.

```bash
cd $REPO/tools/colabfold
mkdir -p db_mini && touch db_mini/SKIP_TEMPLATES
export PATH=$REPO/tools/mmseqs/bin:$PATH
DEBUG_MINI_DB=1 taskset -c 0-31 bash setup_databases.sh $PWD/db_mini > mini_setup.log 2>&1

# phase 1: hide the env DB, uniref only
mkdir -p stash && mv db_mini/colabfold_envdb_202108_db* stash/
COLABFOLD_KEEP_INTERMEDIATES=1 taskset -c 0-31 $CF --mmseqs $MMSEQS --threads 32 \
    --use-env 0 --use-templates 0 --db-load-mode 0 test/q5.fasta db_mini test/out

# phase 2: restore env, HIDE uniref entirely
mv stash/colabfold_envdb_202108_db* db_mini/ && mv db_mini/uniref30_2302_db* stash/
COLABFOLD_KEEP_INTERMEDIATES=1 taskset -c 0-31 $CF --mmseqs $MMSEQS --threads 32 \
    --use-env 1 --use-templates 0 --db-load-mode 0 test/q5.fasta db_mini test/out
```

Result on this box, with uniref30 physically absent during phase 2 — depth roughly doubles, which is
the env DB contributing as it should:

| query | phase 1 (uniref) | phase 2 (+env) |
|---|---|---|
| 1ejo_H | 268 | 531 |
| 1ejo_L | 422 | 865 |
| 1j1p_H | 180 | 403 |
| 1j1p_L | 334 | 728 |
| 1j1p_ag0 | 110 | 270 |

Both staging log lines appeared and exit status was 0.
