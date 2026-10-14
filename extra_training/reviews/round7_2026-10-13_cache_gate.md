# Phase J's two remaining files, and the two silent zeros they found — 2026-10-13

`phase_j_build_cache.py` and `phase_j_replay.py` are written. The replay passes, and writing it found
two configuration faults that would each have made the finetune a no-op with no error anywhere.

## What the two files do

| file | what it establishes |
|---|---|
| `phase_j_build_cache.py` | pairs every cached pose with its per-pose permuted LABEL and with its SAMPLE INDEX |
| `phase_j_replay.py` | the cached label is the live label, the FT3 loss agrees term by term, and the gradient reaches the confidence head only |

### The sample index comes from coordinates, not scores

`phase_j_rescore_cache.py` matches poses to summaries by `chain_pair_iptm` and says in its own output
why that mapping must not be reused: two distinct poses can carry identical confidence scores
(9fve/400073, samples 3 and 4 agree to five decimals while differing by 89.6 Å). The builder compares
each pose tensor to the atoms of each `_sample_<k>.cif` instead, which is identity, and requires the
match to be unique by a margin: within 0.01 Å of one cif (they carry three decimals) and more than
0.5 Å from every other. Measured on 12qa's ten seeds, the order is a different permutation on every
one — `4 0 3 1 2`, `1 0 4 3 2`, `1 3 0 2 4`, ... — so the index cannot be inferred and had to be
established.

### Two label checks that do not go through the loss

The reviewer's objection to a cached-versus-uncached loss comparison is exact: it can pass while both
paths receive the same wrong label. So the builder carries two checks that bypass the loss entirely.

| check | why it bites |
|---|---|
| every resolved label atom is an atom of the job cif it was built from | catches a label that is internally consistent but describes the wrong molecule — the failure the chain-resolution bugs produced |
| each pose's label is a permutation of its siblings' | the per-pose correction relabels symmetric chains and atoms, so it reorders coordinates and can never move one |

Measured on 12qa, 50 poses: the label coordinate multiset is identical across all 50, and all 3,417
resolved label atoms are exactly the job cif's 3,417 atoms.

A third check lives in the builder because it cannot be inferred: the pose's atom order and the
label's must be the same order. The pose comes from INFERENCE over a job json of sequences, the label
from the TRAINING dataset over a job bioassembly; six arrays pinning atoms to tokens
(`asym_id`, `has_frame`, `atom_to_token_idx`, `atom_to_tokatom_idx`, `distogram_rep_atom_mask`,
`is_ligand`) are required to agree elementwise.

## Fault 1 — the resolution was -1.0 on every structure, so the confidence loss was exactly zero

`phase_j_job_cifs.py` builds each job cif from scratch, and carried none of the entry's experimental
metadata. Protenix's parser looks for `_refine.ls_d_res_high`, `_em_3d_reconstruction.resolution` or
`_reflns.d_resolution_high` and falls through to **-1.0** when none is present
(`protenix/data/core/parser.py:182-208`). Every one of the 2,762 prepared rows read resolution -1.0.

`protenix/model/loss.py:1735-1752` then compares that against `loss.resolution` = [0.1, 4.0], sets
`has_valid_resolution` to 0, and **zeroes pLDDT, PDE, PAE and resolved** — the four terms FT3 trains.
No error, no warning. The replay's first run reported a total loss of `0.000000` with not one term
present.

Fixed by writing SAbDab2's own resolution into the job cif, as the item the method calls for: `_refine`
for the 219 crystallographic entries, `_em_3d_reconstruction` for the 282 cryo-EM ones. All 501 have a
resolution in [1.05, 3.5], so nothing is excluded and nothing is invented. Verified through Protenix's
own parser (7wur 3.5, 8jkf 2.83, 8wnp 3.1) and in the rebuilt indices: 2,762 rows, resolution 1.05 to
3.5, **0 rows outside the loss's window**.

The builder now also refuses a unit whose resolution falls outside that window rather than letting it
train at zero loss, so the failure is counted instead of discovered.

## Fault 2 — `alpha_pae` is 0 in the shipped config, and ipTM is computed from PAE

`configs/configs_base.py:401-403`:

| weight | shipped value | FT3 needs |
|---|---|---|
| `alpha_confidence` | 1e-4 | 1e-4 |
| `alpha_pae` | **0.0** — their own comment reads `# or 1 in finetuning stage 3` | **1** |
| `alpha_except_pae` | 1.0 | 1.0 |

ipTM is built from the PAE logits. At the shipped default the PAE term is multiplied by zero, so
**the one number this project is judged on would not be trained** while pLDDT, PDE and resolved still
moved and the total loss still fell — the finetune would have looked like it was working.

Measured on 12qa/free_500000, pose 0: total `0.000356` = 1e-4 x (2.371 + 1.167 + 0.026), the PAE term
absent. With `alpha_pae = 1`: `0.000618` = 1e-4 x (2.371 + 1.167 + 0.026 + 2.621). The replay now
sets it, prints the four weights, and refuses if any is zero.

## The replay, passing

`12qa/free_500000`, all five poses:

| pose | sample | label max abs dx (Å) | plddt | pde | pae | resolved | total |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | 2.370956 | 1.166762 | 2.621137 | 0.025957 | 0.000618 |
| 1 | 0 | 0 | 2.252503 | 1.239402 | 2.677368 | 0.025003 | 0.000619 |
| 2 | 3 | 0 | 2.487656 | 1.197794 | 2.620165 | 0.026130 | 0.000633 |
| 3 | 1 | 0 | 2.035667 | 1.096213 | 2.644928 | 0.024680 | 0.000580 |
| 4 | 2 | 0 | 1.985708 | 1.087975 | 2.619936 | 0.024786 | 0.000572 |

| | |
|---|---|
| label, max \|stored - live\| coordinate | 0.00e+00 Å |
| coordinate-mask differences | 0 |
| loss, max \|stored - live\| over all five terms | 0.00e+00 |
| confidence-head tensors with a non-zero gradient | 165 |
| parameters outside the confidence head with a gradient | **0** |
| confidence head | 19.8M of 464.4M (4.3%) |

The five poses score differently (0.000572 to 0.000633), which is the mechanism the run exists to
exercise: one seed, one trunk, several poses, a head that has to tell them apart.

## What the replay does NOT establish, stated plainly

The pose is taken from the cache on BOTH sides, and it has to be: our poses are full inference
samples and the fold is not bit-reproducible on this hardware, so no live run can regenerate one.
FT3 proper scores a 20-step mini-rollout; ours scores inference samples, which is the deliberate
deviation this project is built on — the head is deployed on inference poses. So the replay answers
"is the stored label the label the live code derives, and does the loss agree", not "is a cached step
an FT3 step". The latter is false by construction and recorded in `docs/cached_training.md`.

## Costs

| | |
|---|---|
| label build | 0.77-0.91 s/unit, one CPU process -> ~70 min for 5,010 units |
| label size | ~170 KB/unit -> ~850 MB for the full cache |
| replay | ~1 min/unit on one GPU |

## Open

- The job cifs and the prepared bioassemblies were rebuilt twice this evening (chain fixes, then the
  resolution). `train_prep/job_indices.csv` is current as of 12:05 AM ET; anything derived from an
  earlier copy is stale.
- The label build has not been run over the whole cache. It should run as the fold produces units,
  not after, so the two finish together.
