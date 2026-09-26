# Self-templating from the model's own poses

Protenix gives many poses for one antibody–antigen complex, and its confidence (ipTM) often picks a
wrong one while good poses sit unselected in the same pool. This picks up to 50 structurally different
poses per complex, feeds each back in as a template, refolds, and asks whether that beats spending the
same compute on more free seeds.

**The crystal never informs selection.** Two facts are taken as known input, as a biologist would know
them: which complexes form the benchmark, and which chains are antibody and which antigen. Everything
that concerns binding — which chains interface, the chain naming, the ranking, the similarity, the
donors — comes from the predicted poses. The crystal is read again only to score folds and to report.

## The selection pipeline

One command, `self_template_50seeds_pxm.py --out <exp> [--diff-against <earlier exp>]`, runs every step
before folding and validates each against the one before it.

| step | file | what it decides | from |
|---:|:---|:---|:---|
| 1 | (orchestrator) | the pool | poses on disk, seeds 300000–300049, `legacy-no-provenance` only |
| 2 | `anchor_relabel.py` | antibody/antigen label of every chain, every copy alike | known input (benchmark role file via the crystal chain map) |
| 3 | `anchor_relabel.py` | each pose's antibody–antigen pairs in contact, and their ipTM | the pose and its confidence json |
| 4 | `anchor_relabel.py` | the anchor, and every pose's chains renamed after it | highest mean ipTM over the pairs a pose forms |
| 5 | `pose_interfaces.py` | the interfaces | anchor pairs in contact in ≥ 1 pose |
| 6 | `pose_to_pose_dockq.py` | pose-to-pose DockQ and ipTM per interface | the poses (`dockq_torch.py`, GPU) |
| 7 | `structural_diversity_filter.py` | the donors | ipTM order, keep if DockQ < 0.23 to all kept, round-robin, cap 50 |
| 8 | `build_templates.py` | a template per donor | the donor pose |
| 9 | `preflight.py` | nothing — refuses a bad namespace before any GPU time | |

With `--diff-against`, folds already scored with the same donor, seed and template chains are carried
into `results.csv`, and only the rest go into `queue_tofold.tsv`. `build_varseed_queue.py` then writes
the variable-seed arm's queue: the same donors, each on one of the exact free seeds `compare_arms.py`
counts as that complex's compute match, so both arms spend the same folds.

## Folding, scoring, reporting

| file | role |
|:---|:---|
| `call_folder.py` | launch one worker per GPU on a queue (tmux session `<run dir>w<gpu>`) |
| `folder.py` | load the model once, fold a queue, one directory per fold, poses kept |
| `scorer.py` | DockQ against the crystal and ipTM, per pose, one csv row each |
| `build_free_queue.py` | queue untemplated folds (the free pool and compute-matched top-ups) |
| `build_varseed_queue.py` | the variable-seed queue: fixed-arm donors on the compute-matched free seeds |
| `score_experiment.py` | did templating help, given the pool — rescued / broken per class |
| `compare_arms.py` | free 50 vs free + compute-matched vs the method, per class |
| `stage_arms.sh` | per GPU: fixed-seed arm → free top-up → variable-seed arm |
| `watch_arms.sh` | every 30 min: claims and scored rows per arm, checked separately |
| `hold_gpus.py` | hold an idle card so the shared machine does not reclaim it |

Shared: `pool_index.py` (index files, ipTM at given chains), `contacts.py` (5 Å heavy-atom contact).
Tests: `tests/test_folder.py`. Paper analyses: `analysis/`.

## Experiment layout

```
experimental_data/
  free/                                  the free pose pool; every template's donor comes from here
  free_compute_matched/                  extra free seeds, matched to the method's donor folds
  self_template_fixed_50_seed/           one build of the pipeline, then its folds
    poses.csv  interfaces.csv  queue.tsv  queue_tofold.tsv  results.csv
    chain_frame/  pose_dockq_matrices/  templates/  poses/  claims/  build.log
  self_template_variable_50_seed/        the same donors, one seed each
```

`free/` must never be renamed: every template and queue points at it by absolute path. A template
namespace is named at build time for the same reason — its path is baked into every job json.

## Two things worth knowing before changing anything

**The csv is the to-do list.** `folder.py` skips any `(interface, seed, arm)` already in
`results.csv`. `folder.py --verify` cross-checks it against the poses on disk.

**A fold is identified by (donor pose, seed, template chains), not by its tag.** Tags are labels and
change between builds; carry-over and every row match on the identity.

The audit that produced this design, every change and how it was checked: `AUDIT.md`.
