# Response to the experimental code review of 2026-10-14

Review commit `d2c43e9`. This response written at `2be5572f`, which lands between the two and
already closes part of organization tip 1.

Every checkable claim in the review was re-derived here rather than accepted. All of them
reproduce. One finding needs a correction that changes its priority, and one finding predicted a
defect found independently the same day.

## Verification of the review's own numbers

| Claim | Reviewer | Re-derived here | Provenance |
|---|---|---|---|
| FoldBench (interface,seed) groups with a top-`pair_iptm` tie across different DockQ | 274 | **274** | `foldbench_baseline_results.csv`, 83,368 rows, 16,674 groups |
| PXM2025, same | 598 | **598** | `pxm2025_baseline_results.csv`, 177,375 rows, 35,475 groups |
| FoldBench groups with 4 samples rather than 5 | 2 | **2** | both are `8b7h` seed 128, interfaces `AC` and `BC`, sample 2 absent |
| Selection cluster 64 share | 77 / 15.4% | **77 / 15.4%** | `cache_train_structures.csv`, 501 structures / 193 clusters, median cluster size 1 |
| `replay` reads the stored `lf_*` loss features | never | **0 references** | `grep -c lf_ phase_j_replay.py` = 0; writer has them |
| Label reuse skips before validation | yes | **confirmed** | `phase_j_build_cache.py`, the `os.path.exists(out)` continue precedes the `try` |
| Completion key omits sample coverage | yes | **confirmed** | `folder.py:623`, `rows_done` is `{(iface, seed, arm)}` |
| Scorer JSON reused on size alone | yes | **confirmed** | `eval/ost_dockq.py:76`, `eval/pxm_dockq.py:41`: `exists() and getsize() > 0` |
| Stale defaults on paths that do not exist | 4 named | **3 of 4 confirmed, 1 already fixed** | `val2_fold`, `val_jobs`, `free_pool` absent from disk, 11 live references; `train_fold` fixed in `2be5572f` |

## Finding 5 needs a correction: the oracle tie-break is in a printer, not in any reported number

The review is right about the code. `run_model/folder.py:462` is `max(v)[1]` over `(iptm, dockq)`
tuples, so an ipTM tie selects the higher ground-truth DockQ. The review is also appropriately
careful: it states the tie counts "do not show that published baseline numbers used this function."

They did not. Two separate selection paths exist and only one is a deliverable.

| Path | Tie rule | Feeds |
|---|---|---|
| `folder.py:462` `report()` | `max(v)[1]` — **oracle** | only `folder.py --report`, a stdout table; no committed artifact, no figure |
| `train_seed_depth_curve.py:156` `three_classes()` | `key=lambda t: (-t[1], t[0])` = `(-iptm, seed)` — **identity** | every seed-depth curve and three-class number reported for FoldBench, PXM and the training set |

The deliverable path already implements the review's recommended fix: a deterministic rule on
candidate identity, with oracle DockQ kept as a separate labelled column (`oracle_dockq`).

Measured, rather than argued — what the oracle rule would have cost had the deliverable used it:

| Table | Mean selected DockQ, identity tie rule | Oracle tie rule | Δ | Acceptable (DockQ ≥ 0.23) |
|---|---|---|---|---|
| FoldBench, n=16,674 groups | 0.3830 | 0.3832 | **+0.0002** | 51.97% → 51.99% |
| PXM2025, n=35,475 groups | 0.2735 | 0.2736 | **+0.0002** | 37.59% → 37.60% |

So: real defect, correctly identified, zero effect on anything reported. Fixing `folder.py:462`
anyway — it is a trap for whoever reads that table next — and the retained-precision point stands
independently, since four-decimal rounding is what manufactures the ties.

## Finding 2 is the most important one, and it predicted a live defect

The review argues the label witnesses establish mutual consistency, not native identity, and states
precisely: the generated job CIF "cannot independently validate an incorrect chain selection used to
construct that same CIF."

That failure was found empirically the same day, from the opposite direction. The first cache smoke
test was run on three structures chosen to exercise known-hard paths:

| Structure | Category | Units built | Outcome |
|---|---|---|---|
| `9g13` | ordinary Fab | 10/10 | clean |
| `9e7l` | identical antigen copies | 10/10 | clean |
| `8h3c` | split single-chain construct | **0/10** | refused: label 121 tokens, fold 238 |

Swept across the whole selection:

| Quantity | n |
|---|---|
| Selected structures | 501 |
| Label does not cover the folded system | **106** |
| of those, split single-chain constructs | 13 |
| of those, ordinary structures | 93 |

Examples: `8h3c` loses a 117-residue chain; `11zw` and `12qd` each lose a 214-residue chain; `7ugq`
loses **both** copies of its 447-residue antigen and keeps only the antibody. The job CIFs are
correct — every chain present, `_pdbx_struct_assembly_gen.asym_id_list` naming all of them. The
chain is lost downstream inside Protenix's `prepare_training_data`, silently; the prep log records
only biotite warnings.

Two things follow. First, the existing checks did catch this, because the count mismatch is coarse
enough to trip the alignment witness — `build_cache` refused and exited non-zero, so no wrong label
was written. Second, the review's point survives that: a chain that is dropped *and* leaves the atom
counts intact would not be caught, which is the case the review constructs. The reproductions it
reports (a reordered label passes; a label reusing one reference atom three times passes; a changed
residue-identity array passes) are accepted.

Accepting the recommendation: record and compare explicit chain, sequence, residue and atom
identities before symmetry permutation; require bijective correspondence over resolved atoms;
validate the permutation against permitted symmetries. And the terminology correction is accepted —
these are consistency checks, not proof of native identity, and the docstrings should say so.

## Finding-by-finding disposition

| # | Review severity | Verdict here | Position |
|---|---|---|---|
| 1 | High | **confirmed** | `replay` constructs both sides from the live dataset and switches only coordinates and the coordinate mask, so a stale cached resolution or frame set cannot be distinguished. This defeats the specific gate the project relies on. Accepted in full, including the `if k in met` silent exclusion, the non-finite rejection, and the deliberate-perturbation test as the acceptance criterion. |
| 2 | High | **confirmed, and independently observed** | See above. Highest priority. |
| 3 | High | **confirmed** | The reuse skip precedes every check, and no input fingerprint exists. With finding 2 pending this is load-bearing: labels written before a chain-selection fix would survive it. Accepted, including rejecting an empty discovery set and verifying against the expected unit set rather than whatever trunks exist. |
| 4 | High | **confirmed** | `rows_done` is keyed `(iface, seed, arm)`, so one scored sample marks the unit done; the watcher counts `run.json`, written before scoring. Accepted. The identity point is important and correct: a same-seed refold is not the same pose, so new scores must not be mixed with an old trunk. |
| 5 | Medium | **confirmed in code, no effect on results** | See the correction above. Measured at +0.0002 DockQ. Fixing the printer; precision retention accepted. |
| 6 | Medium | **confirmed** | Scorer caches are reused on size alone. The two 4-sample groups are one pose — `8b7h` seed 128 sample 2 — missing on both interfaces of the same complex, which is consistent with a single scoring failure rather than random loss, and the table alone does not say why. Accepted, including not defaulting a failed mapping to DockQ 0. |
| 7 | Medium | **confirmed** | Hungarian assignment on `chain_pair_iptm` cannot establish pose identity, and the script says so; presenting its ranking check as evidence about candidate ordering is the overreach. The coordinate-derived identity now exists in `build_cache` (`sample_cifs` / `sample_indices`) and should replace it. The global-`ranking_score`-versus-per-interface-`pair_iptm` mismatch is a real inconsistency with the stated endpoint. |
| 8 | Medium | **confirmed** | The repository SHA does not identify the external Protenix checkout, its local dump hooks, the checkpoint bytes or the resolved configuration. Accepted; must be in place before the final comparison. |

## On the selection population

The cluster-weighting observation is accepted and was not previously stated anywhere: 501
structures over 193 clusters, cluster 64 holding 77 of them (15.4%) against a median cluster size of
1. Capping per (antibody, antigen) pairing was chosen to preserve distinct pairings and does not
make the set cluster-balanced. Uniform structure sampling would therefore spend about one update in
seven on a single cluster. This needs an explicit decision in the sampler rather than an implicit
one, and cluster-level uncertainty for any aggregate claim.

## Not disputed

The review's scope statement is accurate: no CUDA replay, training or inference was run, and
source-established weaknesses are not claims that the generated artifacts are already wrong. That
distinction is maintained here. The two statements this response adds on the other side are that
no reported number used the oracle tie rule, and that 106 of 501 structures currently have a label
that does not cover what was folded.

## Order of work

1. Finding 2 and the 106 structures — find the rule in Protenix's prep that drops a chain. Until
   this is known, the usable population is unknown and no label is worth building at scale.
2. Finding 1 — rebuild `replay` to consume the stored `lf_*` features, with a perturbation test as
   its acceptance criterion. This gates training.
3. Finding 3 — fingerprint label artifacts and verify against an expected unit set, so that fixing
   (1) and (2) actually invalidates what came before.
4. Finding 4 and 6 — per-sample completion and validated scorer-cache reuse.
5. Finding 5 printer, finding 7 identity swap, finding 8 provenance.
6. Organization tips 1 and 2 — the remaining stale defaults, and one source for candidate identity,
   cache schema, completeness and tie rules.
