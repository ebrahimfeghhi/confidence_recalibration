# MSA build runbook — executable on a fresh machine

Builds the MSAs and templates needed to fine-tune Protenix v2's confidence head. Every step uses
Protenix's own pipeline scripts; the only original code is the two patches in §3.3 and the staging
wrapper in §4, both of which exist to keep peak disk inside a 1 TB quota.

Written 2026-10-10. Supersedes an earlier draft of this file that had three things wrong: it used a
hand-rolled sequence extraction instead of `scripts/msa/step1`, it planned a 92 GB PDB mirror that is
not needed, and it omitted the taxid patch without which there is no paired MSA.

---

## 1. What this produces

Protenix's training layout (`Protenix/docs/msa_template_pipeline.md`):

```
<root>/common/seq_to_pdb_index.json        sequence -> directory id
<root>/mmcif_msa_template/<id>/pairing.a3m        MSAs paired by taxonomy
<root>/mmcif_msa_template/<id>/non_pairing.a3m    unpaired MSAs
<root>/mmcif_msa_template/<id>/hmmsearch.a3m      templates
```

One directory per **unique sequence**, not per complex.

| | count |
|---|---|
| training clusters | 557 |
| training structures | 2,886 |
| training instance cifs | 4,555 |
| **unique training sequences needing MSAs** | **4,672** |
| benchmark sequences (FoldBench + PXM, deduped) | 633 |

## 2. Database choice: uniref30_2302

Protenix's current wrapper defaults to it — `scripts/colabfold_msa.py`:

```python
db1: str = "uniref30_2302_db"
db2: Optional[str] = None                      # no pdb70; templates do not come from ColabFold
db3: Optional[str] = "colabfold_envdb_202108_db"
```

Our existing benchmark MSAs were built on **2103** (all 131 PXM and all 113 FoldBench `msa.sh` files
say `--db1 uniref30_2103_db`), so training on 2302 would leave the two sides inconsistent. Fix by
rebuilding the 633 benchmark sequences on 2302 as well — they go through the same pass.

**Do not overwrite the existing 2103 MSAs.** Write to a new directory; every number already reported
depends on them.

2103 is still downloadable from `wwwuser.gwdg.de`, but measured at **860 KB/s ≈ 24 h** for 75 GB
against 49 MB/s from `opendata.mmseqs.org`, and the host refuses parallel range requests. Not worth it
when 2302 is the current upstream default anyway.

## 3. Setup

### 3.1 MMseqs2

```bash
mkdir -p $REPO/tools && cd $REPO/tools
curl -sL -o mmseqs.tar.gz https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs.tar.gz && ./mmseqs/bin/mmseqs version
```

### 3.2 colabfold_search

```bash
cd $REPO && python3 -m venv envs/colabfold
envs/colabfold/bin/pip install -q --upgrade pip
envs/colabfold/bin/pip install -q "git+https://github.com/sokrypton/ColabFold"
```

### 3.3 Apply the patches — BOTH MANDATORY

```bash
S=$REPO/envs/colabfold/lib/python3.12/site-packages/colabfold/mmseqs/search.py
cp "$S" "$S.orig"
patch "$S" < $REPO/extra_training/phases/phase_d_colabfold_search_staging.patch
$REPO/envs/colabfold/bin/python -c "import colabfold.mmseqs.search as m; print(m._emit_taxid(), m._keep_intermediates())"
# expect: True False
```

**Patch A — `uniref_tax.m8`.** `scripts/msa/step3-uniref_add_taxid.py` reads "the uniref_tax.m8 from
output of mmseqs(colabfold search)" to write NCBI TaxIDs into the a3m headers, and `step4` then uses
those to build `pairing.a3m`. Upstream ColabFold does not emit it (ColabFold issue #216); the
`convertalis` call is the one Protenix's `step2-get_msa.ipynb` instructs adding. **Without this there
is no paired MSA, and every Protenix job json requires a `pairedMsaPath`.** On by default; disable
with `COLABFOLD_EMIT_TAXID=0`.

**Patch B — staging.** Lets the uniref and env searches run in separate passes so only one large
database is resident. As shipped, a `--use-env 0` run `mvdb`s `uniref.a3m` into `final.a3m` and
*always* `rmdb`s `prof_res` — the profile the env search is driven from — so the second pass has
nothing to resume from. It also validates every database before the resume logic runs, and its
teardown assumes `tmp/` exists. Gated behind `COLABFOLD_KEEP_INTERMEDIATES=1`; default behaviour is
unchanged.

### 3.4 Keep Protenix out of $HOME

`runner/template_search.py` downloads its database to `_HOME_DIR / "search_database"`. Set
`PROTENIX_ROOT_DIR` so nothing lands outside the project:

```bash
export PROTENIX_ROOT_DIR=$REPO/extra_training/protenix_root
mkdir -p $PROTENIX_ROOT_DIR/search_database
```

## 4. The build

```bash
export REPO=/path/to/tts_abag
export MMSEQS=$REPO/tools/mmseqs/bin/mmseqs
export CF=$REPO/envs/colabfold/bin/colabfold_search
export DB=$REPO/tools/colabfold/db
export WORK=$REPO/extra_training/msa_work
export COLABFOLD_KEEP_INTERMEDIATES=1
mkdir -p "$DB" "$WORK"
```

Cap CPU at **64 cores** (`taskset -c 0-63`); more has crashed this server. GPU is not used.

### Step A — the cif set

The 4,555 selected instance cifs come out of the SAbDab2 tarball. `training_instances.txt` is
committed, derived by `phase_a_define_train_set.py` + `phase_c_extract_msa_queries.py`:

```bash
cd $REPO/extra_training/data/zenodo_splits
awk '{print "splits_final/"$0".cif"}' ../../msa_queries/training_instances.txt > /tmp/members.txt
mkdir -p ../training_cifs
tar xzf splits_v0.2.0.tar.gz -T /tmp/members.txt --strip-components=1 -C ../training_cifs
ls ../training_cifs | wc -l        # expect 4555
```

Benchmark cifs are already on disk — `paths.DATASETS[ds].ref(pdb)`.

### Step B — unique sequences and the index map

```bash
cd $REPO/../Protenix
python3 scripts/msa/step1-get_prot_seq.py \
    --cif_dir $REPO/extra_training/data/training_cifs \
    --out_dir $WORK/pdb_seqs
```

Emits `pdb_seq.fasta` (headers `>{pdb_id}_{entity_id}`, deduplicated by sequence),
`seq_to_pdb_index.json`, `seq_to_pdb_id_entity_id.json`, `pdb_index_to_seq.json`.

Run it a second time over the benchmark cifs into `$WORK/pdb_seqs_bench`, then concatenate the two
fastas into one query file so both sets go through an identical search.

### Step C — phase 1, uniref30_2302 (peak ~367 GB)

`bash $REPO/extra_training/shell_scripts/archive/phase1_uniref.sh` (copy it next to `setup_databases.sh` first, since it `cd`s to its own directory) — downloads with `curl -C -` and retries (the upstream
downloader has no resume flag), then pre-creates the other stages' marker files so
`setup_databases.sh` builds uniref30 only, and runs the search:

```bash
taskset -c 0-63 $CF --mmseqs $MMSEQS --threads 64 \
    --use-env 0 --use-templates 0 --db-load-mode 2 \
    $WORK/query.fasta "$DB" $WORK/msa_initial
```

**Verify before deleting anything** — if these are missing, the patches did not apply, and deleting
uniref30 forces a 107 GB re-download:

```bash
ls $WORK/msa_initial/uniref.a3m.dbtype $WORK/msa_initial/prof_res.dbtype $WORK/msa_initial/uniref_tax.m8
head -1 $WORK/msa_initial/uniref_tax.m8     # query, target, taxid, taxname, ...
```

Then `rm -rf "$DB"/uniref30_2302* "$DB"/UNIREF30_READY`.

### Step D — phase 2, colabfold_envdb (peak ~459 GB)

Resumes from `prof_res`; uniref30 need not be on disk.

```bash
CFDB=colabfold_envdb_202108 taskset -c 0-63 bash setup_databases.sh "$DB"
taskset -c 0-63 $CF --mmseqs $MMSEQS --threads 64 \
    --use-env 1 --use-templates 0 --db-load-mode 2 \
    $WORK/query.fasta "$DB" $WORK/msa_initial
```

Both lines must appear in the log, and MSA depth must rise:

```
Staging mode: uniref.a3m exists, not requiring uniref30_2302_db to be present
Skipping uniref30_2302_db search because uniref.a3m already exists
```

Then `rm -rf "$DB"`.

### Step E — taxids and the pairing split

```bash
cd $REPO/../Protenix/scripts/msa
python3 step3-uniref_add_taxid.py          # msa_initial + uniref_tax.m8 -> mmcif_msa_with_taxid
python3 step4-split_msa_to_uniref_and_others.py \
    --input_msa_dir $WORK/mmcif_msa_with_taxid \
    --output_msa_dir $WORK/mmcif_msa \
    --seq_to_pdb_id $WORK/pdb_seqs/seq_to_pdb_id_entity_id.json \
    --seq_to_pdb_index $WORK/pdb_seqs/seq_to_pdb_index.json
```

`step4` splits on `line.startswith(">UniRef100")`: those hits become `pairing.a3m`, everything else
`non_pairing.a3m`. This is why the taxid patch matters — Protenix pairs from taxonomy itself rather
than using ColabFold's pairing path. From their notebook: *"search the MSA containing taxonomy
information only once for the unique sequence, and pair it according to the species information of
each MSA."*

### Step F — templates

```bash
cd $REPO/../Protenix
python3 runner/template_search.py --help     # confirm args, then run over $WORK/mmcif_msa
```

`hmmsearch` against `pdb_seqres_2022_09_28.fasta`, auto-downloaded to
`$PROTENIX_ROOT_DIR/search_database`. Profile-based, so it consumes the a3m files from step E and
writes `hmmsearch.a3m` beside them. **No pdb70 and no PDB mmCIF mirror.**

The 2022-09-28 snapshot is also safe on dates: 0 of 90 PXM complexes were deposited before it, so the
template database cannot contain them. 61 of 98 FoldBench complexes predate it, but FoldBench folds
run with `free_templates=False` so they use no templates anyway. PXM runs with templates on.

### Step G — place the index map

```bash
cp $WORK/pdb_seqs/seq_to_pdb_index.json $PROTENIX_ROOT_DIR/common/
```

## 5. Validation already done

Exercised end to end on a 280 MB mini database (`DEBUG_MINI_DB=1`, which substitutes
`mini_swissprot` for both large databases). `SKIP_TEMPLATES` was used **for that test only**.

**Staging** — uniref30 physically absent during phase 2, depth roughly doubles when env is added:

| query | phase 1 (uniref) | phase 2 (+env) |
|---|---|---|
| 1ejo_H | 268 | 531 |
| 1ejo_L | 422 | 865 |
| 1j1p_H | 180 | 403 |
| 1j1p_L | 334 | 728 |
| 1j1p_ag0 | 110 | 270 |

Both staging log lines appeared; exit 0.

**Taxids** — `uniref_tax.m8` produced with real annotations:

```
101  P84751  10090  Mus musculus
101  P0DOX5  9606   Homo sapiens
```

**Leakage** — `train.fasta` ∩ `bench.fasta` at the sequence level: **0 shared**.

## 6. Gotchas that cost time here

**gwdg is 860 KB/s and refuses parallel ranges.** Use `opendata.mmseqs.org` (49 MB/s).

**The upstream downloader does not resume.** `setup_databases.sh` uses `curl -L -o` with no `-C -`, so
a killed 107 GB download restarts from zero. `phase1_uniref.sh` wraps it with resumable fetches.

**`DOWNLOADS_READY` gates all downloads as one block**, so the script cannot be told to fetch only one
database — hence driving the downloads externally and pre-touching markers.

**ANARCI silently returns no domains with HMMER 3.4.** It parses hmmscan text output through
Biopython's `Hmmer3TextParser`, which cannot read 3.4's format: coordinates come back `None`, every
hit is dropped, no error. Needs HMMER **3.1b2** via `hmmerpath=` plus HMMs re-pressed with it. Affects
CDR work only, not MSAs — `hmmsearch` itself is fine on 3.4.

**`chain_roles.csv` and our local reference cifs disagree on chain naming** — 9n6e's roles name E/F/G
while its cif holds A,B,C,D,H,I,L. Don't cross-reference them to identify antigens.

**Repeating a seed does not reproduce a fold** (9n6e_GF: 0.022 → 0.644 on repeats of one seed), so a
saved pose cannot be paired with a freshly computed trunk `z`.

**Do not clear `~/.cache/huggingface`** — ESMFold2 / ESM-C / ESM3 weights plus a gated auth token live
there. Protenix weights are in `~/checkpoint/`.

**Quotas charge by file owner**, so another user's files inside your home don't count against you; and
`df` does not reflect the quota at all. Use `quota -s`.

## 7. After this

Fine-tune with Protenix's **FT3** stage unmodified — see `README.md` in this folder.
The finetuning procedure itself is deliberately not designed yet; this runbook stops at producing
data.
