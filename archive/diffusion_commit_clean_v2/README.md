# Self-templating from the model's own poses

Protenix gives many poses for one antibody–antigen complex, and its confidence (ipTM) often picks a
wrong one while good poses sit unselected in the same pool. This picks up to 50 structurally different
poses per complex, feeds each back in as a template, refolds, and asks whether that beats spending the
same compute on more free seeds.

**The crystal never informs selection.** Two facts are taken as known input, as a biologist would know
them: which complexes form the benchmark, and which chains are antibody and which antigen. Everything
that concerns binding — which chains interface, the chain naming, the ranking, the similarity, the
donors — comes from the predicted poses. The crystal is read again only to score folds and to report.

## How the code fits together

Four stages, each a script or two. A dataset (pxm2025 or FoldBench) goes through all four.

| stage | what happens | scripts | output |
|:---|:---|:---|:---|
| 0. inputs, once per dataset | 50 free Protenix seeds per complex, every pose scored against the crystal, the chain maps and the antibody/antigen roles | pxm2025: built in earlier phases (below). FoldBench: `foldbench_setup.py` | `free/`, pool csv, `chain_maps.csv`, `chain_roles.csv` |
| 1. build an experiment | choose the donor poses and make a template from each | `build_experiment.py` (runs the 7 steps below) | `queue.tsv`, `templates/` |
| 2. fold | fold every donor with its template, score every pose | `call_folder.py` → `folder.py` → `scorer.py` | `poses/`, `results.csv` |
| 3. compare | free 50 vs free + compute-matched vs self-templating | `compare_arms.py`, `analysis/` | tables, plots |

**Why only FoldBench has a setup script.** pxm2025's inputs were built weeks ago, piece by piece, by
scripts from earlier phases: the chain maps by `diffusion_commit_clean/chain_map.py`, the pool scoring
by scripts now in `archive/confident_decoy_experiments/scripts/` (e.g. `score_pxm2025.py`). They are
finished and verified, so they were not rebuilt. FoldBench's were built in one pass by
`foldbench_setup.py`, which also rescored every pose because the old FoldBench scores were stale.

## Stage 1: building an experiment

`build_experiment.py --dataset <pxm2025|foldbench> --out <exp> [--diff-against <earlier exp>]` runs
these steps in order and checks each against the ones before it.

| step | script | what it does | writes |
|---:|:---|:---|:---|
| 1 | `build_experiment.py` | **The pool.** Lists every pose selection may use: the dataset's first 50 free seeds × 5 samples = 250 poses per complex (pxm2025 seeds 300000–300049, FoldBench 101–150) | `poses.csv` |
| 2 | `anchor_relabel.py` | **Roles.** Marks every chain of every pose antibody or antigen. The only fact taken from the benchmark, and every copy of a molecule gets the same role | (in memory) |
| 3 | `anchor_relabel.py` | **Contacts, per pose.** For each pose separately: which antibody–antigen chain pairs touch (5 Å), and the model's ipTM for each — in that pose's own chain letters | `chain_frame/pose_contacts.csv` |
| 4 | `anchor_relabel.py` | **One naming for all poses.** Protenix may call the same antigen copy B in one pose and C in the next. Picks an anchor pose (highest mean ipTM over its touching pairs) and renames every pose's chains to match it | `chain_frame/chain_maps_anchor.csv` |
| 5 | `pose_interfaces.py` | **The interface list, per complex.** Translates step 3's pairs into step 4's names and keeps each antibody–antigen pair that touches in at least one pose | `interfaces.csv` |
| 6 | `pose_to_pose_dockq.py` | **Similarity.** DockQ of every pose against every other, per interface (GPU) | `pose_dockq_matrices/` |
| 7 | `structural_diversity_filter.py` | **The donors.** Per interface, walk the poses in ipTM order and keep one if it is not similar (DockQ < 0.23) to any already kept; alternate across interfaces; at most 50 per complex | `queue.tsv` |
| 8 | `build_templates.py` | **Templates.** Turns each donor pose into a template Protenix can load | `templates/` |
| 9 | `preflight.py` | **A last check.** Refuses a broken template set before any GPU time | — |

Steps 3 and 5 are easy to confuse. Step 3 looks at each pose on its own, in its own chain letters.
Step 5 works on the whole complex, after step 4 has given all poses the same letters, and decides
which pairs count as interfaces.

With `--diff-against`, folds already scored with the same donor, seed and template chains are carried
into `results.csv`, and only the rest go into `queue_tofold.tsv`. `build_varseed_queue.py` then writes
the variable-seed arm's queue: the same donors, each on one of the exact free seeds `compare_arms.py`
counts as that complex's compute match, so both arms spend the same folds.

## Stages 2 and 3: folding, scoring, reporting

| file | role |
|:---|:---|
| `call_folder.py` | launch one worker per GPU on a queue (tmux session `<run dir>w<gpu>`) |
| `folder.py` | load the model once, fold a queue, one directory per fold, poses kept |
| `scorer.py` | DockQ against the crystal and ipTM, per pose, one csv row each |
| `build_free_queue.py` | queue untemplated folds (the free pool and compute-matched top-ups) |
| `build_varseed_queue.py` | the variable-seed queue: fixed-arm donors on the compute-matched free seeds |
| `score_experiment.py` | did templating help, given the pool — rescued / broken per class |
| `compare_arms.py` | free 50 vs free + compute-matched vs the method, per class |
| `foldbench_setup.py` | build FoldBench's inputs (free-pool links, pool csv, chain maps, roles) and audit them |
| `stage_arms.sh` | per GPU: fixed-seed arm → free top-up → variable-seed arm |
| `watch_arms.sh` | every 30 min: claims and scored rows per arm, checked separately |
| `hold_gpus.py` | hold an idle card so the shared machine does not reclaim it |

Shared: `pool_index.py` (index files, ipTM at given chains), `contacts.py` (5 Å heavy-atom contact).
Tests: `tests/test_folder.py`. Paper analyses: `analysis/`.

## Datasets

Every step takes `--dataset pxm2025` (the default) or `--dataset foldbench`, and `build_experiment.py`
passes it to each step. What a dataset means — its free pool, pool csv, role file, crystal chain maps,
job jsons, reference cifs, scorer and pool seeds — is one `Dataset` record in `paths.DATASETS`; no
script names a benchmark itself. FoldBench's inputs are built once by `foldbench_setup.py` from the
existing 50-seed Protenix run (seeds 101–150, OST-scored); its audit is `setup_audit.md` beside them.

## Experiment layout

```
experimental_data/
  free/                                  the free pose pool; every template's donor comes from here
  free_compute_matched/                  extra free seeds, matched to the method's donor folds
  self_template_fixed_50_seed/           one build of the pipeline, then its folds
    poses.csv  interfaces.csv  queue.tsv  queue_tofold.tsv  results.csv
    chain_frame/  pose_dockq_matrices/  templates/  poses/  claims/  build.log
  self_template_variable_50_seed/        the same donors, one seed each
  protenix/foldbench/                    FoldBench, same arms and names; free/ links into the 50-seed run
    free/  pool.csv  chain_maps.csv  chain_roles.csv  setup_audit.md
```

pxm2025's arms move to `protenix/pxm2025/` once the variable-seed arm has finished (their paths are
baked into running queues and templates until then).

`free/` must never be renamed: every template and queue points at it by absolute path. A template
namespace is named at build time for the same reason — its path is baked into every job json.

## Two things worth knowing before changing anything

**The csv is the to-do list.** `folder.py` skips any `(interface, seed, arm)` already in
`results.csv`. `folder.py --verify` cross-checks it against the poses on disk.

**A fold is identified by (donor pose, seed, template chains), not by its tag.** Tags are labels and
change between builds; carry-over and every row match on the identity.

The audit that produced this design, every change and how it was checked: `AUDIT.md`.
