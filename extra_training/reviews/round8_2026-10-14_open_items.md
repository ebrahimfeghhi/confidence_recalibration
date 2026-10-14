# The open items, closed — 2026-10-14

Round 5's J1/J3 and I1/I2 were the last items on the list. All four are closed, each with a control
that fires on the defect, not just a reading of the code. Two further faults surfaced while proving
them and are fixed here.

## J1 — the chain decomposition of a split single-chain construct

Round 5 measured 9vb4's two auth chains sharing subchain F and one fused 259-residue entity. The
explicit-entity rewrite in `write_job_cif` fixed it; this round verified it across the whole set and
turned the invariant into an assertion the phase runs on every write.

| | |
|---|---|
| job cifs checked | 501 |
| split constructs (one source chain serving two roles) | **18**, not 9 — the copy-index fix made more roles resolve by name (8h3c's `B1`/`B2` are both chain B) |
| a subchain shared by two auth chains | 0 |
| a chain without exactly one subchain | 0 |
| a subchain claimed by two entities | 0 |

**My first check was wrong and reported two false positives.** It called `setup_entities()` after
reading the file, which MERGES entities that share a sequence — exactly what a homodimeric antigen
has. 8dpl and 8f6x each carry their antigen twice, so the merge made them look like they shared an
entity when the file on disk gives each its own (`Cp -> 3`, `Dp -> 4`). The check now reads without
`setup_entities()`. One entity per identical sequence would in fact be correct mmCIF; two auth chains
under one *subchain* is the defect, and that is what is asserted.

Control: forcing 9vb4's chains A and B to share subchain `Ap` is caught —
`shared={'Ap': ['A','B']} dup={'Ap': 2}` — while the file as written passes.

## J3 — the label's sequence frame

Already closed and asserted on every run: the written cif's entity sequences are compared to the job
json phase G produced, read back from disk, mandatory, and expanded per subchain so a homodimer
compares correctly. 501 of 501 pass. A residue at a position its entity sequence does not have is a
separate, also-mandatory check.

## A residue that was being dropped between two alignments

`train_prep/unplaced_residues.csv` had one entry: **8rwf** ag0, 168 of 169 residues placed. The cause
is a convention disagreement that cost a residue twice over.

`gemmi.one_letter_code` (0.7.5) returns `X` for every modified residue. 27 residues over 10 of the 501
structures render that way: PCA 8, SEP 9, TPO 5, MSE 3, DAL 1, MEA 1. **SAbDab2 is not consistent
about them either:**

| structure | SAbDab2's sequence says | the entry's residue |
|---|---|---|
| 8rwf ag0 position 88 | `M` | MSE (selenomethionine, chain B 304) |
| 9p4c H position 0, 9og1 H position 0 | `X` | PCA (pyroglutamate) |

So resolving *every* modified residue to its parent fixed 8rwf and **broke five structures** whose
N-terminal pyroglutamate SAbDab2 spells `X`: the chain said `E`, the query said `X`, the span started
one residue late, and **40 atoms were lost** across the set. Measured, then reverted.

MSE alone is mapped to `M` — the one modified residue sequence tooling normally normalises, and the
only one where the two conventions disagree in the direction that costs a residue. And the mapping
had to live in **one** place: `chain_seqs` spelled MSE as `M` while `write_job_cif`'s per-chain
alignment called `gemmi.one_letter_code` directly and spelled it `X`, so 8rwf aligned to SAbDab2's
sequence at coverage 1.000 and then lost the same residue being positioned in the fold's expected
sequence. Both now call `res_code`.

| | before | after |
|---|---|---|
| structures dropping a residue | 1 (8rwf) | **0 of 501** |
| atoms written across the set | 2,401,908 | 2,401,916 |
| job cifs changed | — | 1 (8rwf, +8 atoms) |

## The label check was too brittle, and 8rwf exposed it

With 8rwf's residue recovered, `phase_j_build_cache.py`'s independent label check rejected it: *34 of
2,153 resolved label atoms are not atoms of the job cif*. **Every one of the 34 was 0.001 Å from a
real atom**, spread over 32 unrelated residues. It was rounding, not a bad label: the check rounded
both sides to three decimals and compared sets, and the label is float32 of a 3-decimal cif value, so
at |x| ~ 80 Å the round-trip error is ~1e-4 and a coordinate of -80.7395 rounds down on one side and
up on the other. It passed on 12qa by luck.

Now nearest-neighbour with a 0.01 Å tolerance — four orders of magnitude below the Ångström-scale
error it exists to catch. Controls, on 8rwf's 2,153-atom label:

| case | required | result |
|---|---|---|
| the label as built | accept | accepted |
| every atom shifted 1 Å | reject | 2,153 of 2,153 too far |
| checked against another structure's cif (12qa) | reject | 2,153 of 2,153 too far |
| pose 1 shifted 0.5 Å | reject | not a permutation of pose 0 |
| one atom of pose 0 moved 5 Å | reject | not a permutation of pose 0 |

## I1 — phase I could report COMPLETE on a job absent from pool.csv

Reproduced here on a synthetic two-job fold tree, which round 5 had not done. The tree is complete and
internally consistent; `pool.csv` names the interfaces of one job only.

| tree | required | result |
|---|---|---|
| both jobs in pool.csv, all poses present | exit 0, COMPLETE | `COMPLETE: every expected unit folded, every pose scored at every interface, every cache replayable` |
| `bbbb` absent from pool.csv | FAIL | exit 1, `FAIL 1 job(s) have no interfaces in pool.csv: ['bbbb']` |
| fully pooled and scored, one pose cif removed | FAIL | exit 1, INCOMPLETE |
| NaN in one trunk cache, `--trunks` | FAIL | exit 1, `unusable or absent trunk caches: 1` |

The positive control matters as much as the rest: without it a checker that always failed would look
like a checker that worked.

## I2 — the cache validator accepted NaN coordinates and bad trunk shapes

Eight controls, each a real cache npz mutated into one defect:

| mutant | `check_npz` returned |
|---|---|
| unmodified | accepted |
| NaN in `ci_coordinate` | `ci_coordinate holds NaN or inf` |
| inf in `ci_s_inputs` | `ci_s_inputs holds NaN or inf` |
| non-square `z_trunk` | `z_trunk shape (625, 312, 256), expected [N, N, c]` |
| `z_trunk` rank 2 | `z_trunk shape (625, 256), expected [N, N, c]` |
| `atom_to_token_idx` out of range | `ranges 0..1000000, outside [0, 624]` |
| `asym_id` one entry short | `has 624 entries, expected 625` |
| `ci_contact_probs` removed | `missing ['ci_contact_probs']` |

## State after this round

`train_prep/job_cifs`, `job_bioassembly` and `job_indices.csv` were rebuilt: 501 structures, 2,762
index rows, resolution 1.05–3.5, max 2,941 tokens, 0 dropped residues. The label build and the replay
both pass on the rebuilt data (8rwf, all five poses: label 0.00e+00 Å from the live permutation, loss
agreement 0.00e+00 on every term, 165 confidence-head tensors with a gradient, 0 outside).

Nothing needs refolding. All of this is label-side; the fold consumes sequences.
