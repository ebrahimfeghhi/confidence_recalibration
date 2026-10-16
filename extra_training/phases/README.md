# phases/

One script per phase of the confidence-head finetune (Protenix FT3). The phases run A → K and each
leaves its output on disk for the next, so any one can be re-run alone. Letters are historical and a
few are shell scripts rather than Python.

`docs/phases_history.md` holds the chronological record — the audit passes, review rounds and the
derivation of every number. It was this file until 2026-10-16, when it reached 1,094 lines and
stopped being a way to find anything.

## What the stages do

| stage | what it produces |
|---|---|
| **A–B** | the cluster sets: which structures train, which validate, with the leakage filters applied |
| **C–F** | MSAs — queries out, searches run, alignments turned into the three files Protenix reads |
| **G–I** | validation job jsons, templates, and the folds themselves, with their score recovery |
| **J** | the training cache: job cifs, labels, and the pairing of every pose to its label and sample index |
| **K** | the finetune itself — train the confidence head, score it, choose the next learning rate |

## The files

### A–B — define the train and validation sets

| file | what it does |
|---|---|
| `phase_a_define_train_set.py` | Drops SAbDab2 training clusters homologous to benchmark complexes SAbDab2 does not itself list. |
| `phase_b_define_val_clusters.py` | Picks the validation clusters: 50 clusters, about 10% of structures. |
| `phase_b2_propose_val_split.py` | Proposes a 40-cluster post-cutoff split and re-audits the existing 21 under the newer rules. |
| `phase_b3_pick_second_reps.py` | Chooses the second representative for each validation cluster that has one. |

### C–F — MSAs

| file | what it does |
|---|---|
| `phase_c_extract_msa_queries.py` | Writes the FASTA of every unique sequence needing an MSA. |
| `phase_d_search_uniref.sh` | Searches UniRef30 for those queries (with `phase_d_colabfold_search_staging.patch`). |
| `phase_e_search_envdb.sh` | Searches the environmental database for the same queries. |
| `phase_f_pair_and_split.py` | Turns the alignments into the three files Protenix actually reads. |

### G–I — validation jobs, templates, folding

| file | what it does |
|---|---|
| `phase_g_build_val_jobs.py` | Writes one Protenix job json per validation structure. |
| `phase_h_search_templates.py` | Finds structural templates for every validation chain, using Protenix's own search. |
| `phase_i_fold_validation.sh` | Runs the validation folds across the cards, claim-based and resumable. |
| `phase_i_inputs.py` | The pre-GPU input check, so a bad input is caught before a card is taken. |
| `phase_i_check.py` | Phase I's verification and score recovery after folding. |

### J — the training cache

| file | what it does |
|---|---|
| `phase_j0_select_structures.py` | Freezes the structure list for the FT3 training cache. |
| `phase_j_prepare_training_data.py` | Turns the selected structures into Protenix's training format. |
| `phase_j_dataset.py` | Instantiates Protenix's training dataset over our prepared data. |
| `phase_j_job_cifs.py` | Writes one mmCIF per training job: the antibody and its polypeptide antigen, nothing else. |
| `phase_j_build_cache.py` | Pairs every cached pose with its label and its sample index — the file the whole cache rests on. |
| `phase_j_replay.py` | Asks whether a cached unit gives the FT3 loss the uncached path would give. This is GATE 2. |
| `phase_j_labels_and_audit.sh` | Runs the label build and its audit chain end to end. |
| `phase_j_check_contacts.py` | Checks geometrically whether each selected antigen chain touches the selected antibody. |
| `phase_j_queue_now.py` | Queues the folds that can run now — the ones whose query is not changing. |
| `phase_j_stage_complete.py` | Asks whether a fold stage is really finished: trunk, poses and every expected score row. |

#### J — labels: audit, verify, repair

| file | what it does |
|---|---|
| `phase_j_label_audit.py` | Audits our RCSB-derived labels against SAbDab2's curated coordinates, independently. |
| `phase_j_label_audit_selftest.py` | Falsifiability harness for the audit: damages labels on purpose and fails if the damage goes unseen. |
| `phase_j_verify_labels.py` | Re-checks every cached label on disk against its job cif and names the structures needing a rebuild. |
| `phase_j_domain_audit.py` | Asks whether every folded query corresponds to a real stretch of its deposited chain. |
| `phase_j_disposition_mismatches.py` | Dispositions the audit's coordinate mismatches with OST DockQ, imposing no chain map. |
| `phase_j_drop_superseded_poses.py` | Removes the pose directories no trunk was computed from, restoring one directory per seed. |
| `phase_j_merge_results.py` | Merges a refold root's `results.csv` into the destination's, replacing rows by key. |

#### J — scoring gaps and refolds

| file | what it does |
|---|---|
| `phase_j_fill_scores.py` | Fills the gaps OST's chain mapper left, by scoring each interface's chain pair alone. |
| `phase_j_rescore_unmapped.py` | Recovers interfaces OST dropped because it left a target chain unmapped. |
| `phase_j_rescore_cache.py` | Rescores a cached fold from its trunk npz and checks it reproduces the stored scores. |
| `phase_j_refold_from_audit.py` | Derives the refold list from the domain audit rather than from a composition guess. |
| `phase_j_refold_list.py` | Turns `job_cifs`' native-gaps report into the committed refold list. |

#### J — corrected queries and MSA remapping

| file | what it does |
|---|---|
| `phase_j_corrected_queries.py` | Writes the corrected-query FASTA and the sequence→directory index both sides share. |
| `phase_j_corrected_jobs.py` | Writes the corrected job jsons for the structures on the refold list. |
| `phase_j_remap_msa.py` | Carries an existing MSA onto a corrected query by inserting pure-gap columns. |
| `phase_j_remap_msa.sh` | Runs that remapping over every corrected query. |
| `phase_j_compare_msa.py` | Compares two a3m files as alignments rather than as statistics. |

### K — the finetune

| file | what it does |
|---|---|
| `phase_k_train.py` | The FT3 finetune: trains the confidence head on our own pose distribution, across three GPUs. |
| `phase_k_train_ddp_selftest.py` | Asks whether multi-GPU training computes the same thing as one GPU, and refuses to pass a no-op. |
| `phase_k_eval.py` | Scores a confidence head on the validation metric the project is judged on. This is GATE 1. |
| `phase_k_eval_sharded.py` | Runs that metric across several GPUs and merges, verified identical to the single-card path. |
| `phase_k_resolve_seed_dir_selftest.py` | Falsifiability harness for the pose-directory resolver: must refuse, not guess. |
| `phase_k_pick_lr.py` | Reads a finished run's loss curve and recommends the next learning rate. |

### Utility

| file | what it does |
|---|---|
| `pick_unfolded_unit.py` | Prints the first `(pdb, seed)` in range with no `run.json`, for the smoke test. |

## The two gates, because nothing downstream means anything without them

**GATE 1** (`phase_k_eval.py`) — the shipped confidence head must reproduce the ipTM already recorded
in `results.csv`, row by row, over all 46,000. It tests the three joins the metric depends on:
interface→chain pair, pose→sample, and `(iface, seed, sample)`→DockQ. If any is wrong the training
curve is uninterpretable, so the pilot does not start. It has caught two real defects.

**GATE 2** (`phase_j_replay.py`) — training on cached trunks must give the same FT3 loss as running
the model live: same label, all four loss terms equal, and the gradient reaching the confidence head
and nothing else. This is the shortcut the whole project rests on.

## Conventions

- Scripts refuse rather than guess. An ambiguous pose directory, an unresolvable chain mapping or a
  label whose fingerprint does not match its trunk stops the run and names what it found.
- A number's provenance travels with it. Labels carry a fingerprint over the cif, bioassembly,
  indices, trunk and builder source, so a stale label cannot be silently trained on.
- Self-tests are separate files ending `_selftest.py` and are built to fail: each damages its input
  on purpose and reports a failure if the damage goes unnoticed.
