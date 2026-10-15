# build_cache: what fails, why, and what is lost

Companion to `phases/phase_j_build_cache.py`. Written 2026-10-15. Reproduce any row below with
`--only <pdb>`; the guards named here are in that script and in `phases/phase_j_job_cifs.py`.

## Current loss

| | structures | units (x10 seeds) |
|---|---|---|
| selected for the training cache | 501 | 5,010 |
| folded | 500 | 5,000 |
| **labelled and trainable** | **500** | **5,000** |
| lost | 1 | 10 |

`9mx1` is the only loss and cannot be folded at all: 2,941 residues against Protenix's 2,560-token
refusal (`runner/inference.py:399`). Nothing that was folded is unlabelled.

## What the guards are, and why they exist

Two independent checks stand between a pose and a usable label. Neither is a loss function, so
neither can be satisfied by a label that is merely self-consistent.

| guard | what it asserts | what it caught |
|---|---|---|
| fingerprint | a label records the job cif, bioassembly, index rows, trunk and code it was built from, hashed per structure, and is rebuilt when any changes | a label built from the old chain selection surviving a fix |
| atom spot-check | every resolved label atom lies within 0.01 A of an atom of the job cif | `9djw` at 104 A (wrong crystal copy); `8av9`, `8pww`, `7stz`, `8gpu` |

## Resolved, with the measurement that resolved it

| structure | symptom | cause | fix |
|---|---|---|---|
| `9djw` | 2,011 of 2,011 label atoms 104 A out | the antigen was a different crystal copy; a packing contact at 2.55 A over 180 atom pairs with **zero** epitope residues in common with the real interface | `pick_instance_copies`: SAbDab2's instance cif keeps the deposited frame, so the right copy matches at 0.0000 A and the runner-up at 25-33 A. An argmin, not a threshold. Changed 1 of 501 selections |
| `8av9` | 4 of 4,369 atoms, worst 2.44 A | the job cif vetoes 7 residues it cannot place (4 unplaced + 3 misplaced); Protenix's parse reconstructs them from the entity sequence, so the label holds atoms the cif does not | those atoms are marked UNRESOLVED rather than supervised, within a budget derived from the cif's own recorded drops times the largest residue in that file. 40 atoms masked over 10 units |
| `8pww` | 6 of 4,266 atoms, worst 4.05 A | same as `8av9`, 2 recorded drops | same. 60 atoms masked over 10 units |
| `8gpu` | pose 827 tokens vs dataset 429 | SAbDab2's named antigen (chain E) sits 5.02 A from the antibody with 0 contacts, so Protenix's `remove_dissociation` deleted the chain. Entry chain B is the same antigen and contacts at 2.43 A | `rescue_noncontacting_antigen` re-points an antigen that touches NOTHING onto a copy that does. 10 of 10 labels built, no masking needed |

### The bug that hid `7stz` and `8gpu` for so long

`job_chains` returned early -- `if not leftover: return resolved, []` -- whenever every role resolved
by NAME, which skipped all three coordinate pickers for exactly the structures where SAbDab2's
naming was trusted completely. That is where the naming is wrong. `9djw` was only ever fixed because
its names did NOT resolve and it fell through. The named path now runs the contact rescue.

Running all three pickers on the named path was tried and rejected: it changes 24 structures, about
16 cosmetically (the `X1`/`X2` copy index resolving to entry chain `X`, the same chain) and 3
intentionally, but 5 move unexplainably -- `9r09` and `9hng` put the heavy chain on a chain named
`J000`, `9shd` puts the antigen on `H0Z0`. Those 5 are open findings, not shipped behaviour.

## `7stz`, resolved -- and the guard was wrong, not the data

Its antigen is the only one SAbDab2 names, and it does not touch the antibody.

| measurement | value |
|---|---|
| SAbDab2's antigen (entry chain C) to antibody | 5.81 A, 0 contacts |
| its heavy chain to antigen, inside SAbDab2's own instance cif | 18.84 A |
| entry chain D, same antigen, to antibody | **2.41 A, contacts** |
| chain D against the folded antigen sequence | **422 of 422 residues, 100% identity** |
| chain C against the folded antigen sequence | 540 of 540, 100% identity |

So chain D is the SAME protein, with more unresolved density (422 of 590 resolved against C's 540),
and it is the one actually bound. Re-pointing there needs no refold -- the folded sequence is
unchanged at 590 -- but `write_job_cif` refuses it:

    ag1: 80 of 502 observed residues do not align to the fold's sequence (16%, limit 15%)

THE GUARD WAS COUNTING SOLVENT. `502` is chain D's residue count INCLUDING 62 waters, and the 80
"misaligned" residues are exactly those 62 waters plus 18 ligands (glycans, PG4, Ca). None of them
can ever align to a protein sequence and all of them were already being dropped correctly -- only the
RATIO was wrong, which made it a function of how much solvent a chain happens to carry. Chain C
passed the identical artefact only because its 540 polymer residues diluted it to 12%.

The fraction is now computed over POLYMER residues only. Chain D's polymer residues align at
422 of 422, 100%. Result: 7stz's bioassembly rebuilds to 7,848 atoms with all three chains and
dissociation zero -- exactly the 7,848 the pose carries -- and all 10 labels build with no masking.
NO REFOLD was needed: the folded antigen sequence is 590 either way, so only label coordinates moved.

`7stz` is also an anti-hapten antibody -- `agtypes` is `HAPTEN/PROTEIN`, `agchains` is `BC/C` -- so
its true antigen is a small molecule the pipeline excludes as a non-polymer. That is context, not
the reason it fails: chain D is a real contacting protein interface.

## Not failures, recorded so they are not mistaken for failures

11 of 627 selected antigen chains across the 501 have no atom within 5 A of the antibody, every one
of them SAbDab2's own named chain, at 5.02-6.78 A. Only `7stz` and `8gpu` broke anything, because
they are the only two whose structure has no OTHER contacting antigen. In the other 9 the primary
antigen contacts at 2.5-3.0 A and a secondary chain is a bystander in the crystal; their labels
build normally and a non-touching chain simply contributes no interface signal.

Of those 9, four more could be re-pointed to a contacting copy (`9md4` 2.32 A, `9o4o` 2.41 A,
`8vgm` 2.43 A, `8zyx` 2.36 A) and two could not (`9q7b` 5.97 A, `9e3h` 5.23 A -- no copy contacts).
Not done: nothing is blocked, and each re-point is a deviation from SAbDab2's instance definition
that should be a deliberate decision rather than a side effect.

## Also open

- `phase_j_check_contacts.py` re-derives the selection with the OLD picker, so it reports a
  correctly-fixed structure as having a "stale job cif" (`9djw`). It needs `inst_st` passing through.
- `training/fold/pool.csv` lists `8vr9` A-D and `9q7b` D-K/D-L as interfaces while the contact check
  measures 5.91 A and 5.97 A with 0 contacts. Different contact definitions (the pool used OST's
  `contact_d=5.0, contact_mode="aa"`). Affects interface counting for evaluation, not labels. 2 of 1,039.
