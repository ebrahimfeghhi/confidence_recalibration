# Questions for review

Four things I need decided or checked. Everything raised in rounds 1-5 is addressed; these are mine.

## 1. Is the alignment comparison sound? (`fc23b3a2`, `phases/phase_j_compare_msa.py`)

This is the gate on 2,850 folds and I wrote both the tool and its tests, which is the weakest possible
evidence. It compares, per hit shared between two a3m files: the match-column string, the insertion
profile, and malformed rows in either file; matched by accession AND segment, with multiplicity
preserved; reporting GAINED / LOST / CHANGED separately and exiting non-zero on any difference.

It catches the case you constructed -- two hits exchanging gap patterns, with depth, hit ids, taxonomy
and per-column occupancy all bit-identical. **What I have not shown is that it catches everything of
that shape.** Is there a difference class it would still miss?

## 2. `8dpl`: where does the VH end?

Its missing run is native 115-136, `TVSSGTGGSGGGGSGGGGSGGG`. `...WGQGTALTVSS` is the canonical end of
a VH, so the leading `TVSS` looks like domain and the rest like tether. It is classified `unresolved`
and is therefore in NEITHER the refold list nor the linker exclusions -- refolded by nobody until
someone splits it at the domain bound. I do not want to guess, since restoring a tether into an
antibody domain is the failure the classification exists to prevent.

Same question in principle for the three excluded as `linker`: `8iys` 120-134 `AGGGGSGGGGSGGGG`,
`8s6k` 117-125 `SSSGGGGSG`, `8h3c` 119-133 `GGGGSGGGGSGGGGS`. Those look unambiguous to me, but they
were classified by a GS-composition heuristic, not by a domain boundary.

## 3. `7stz` -> `D` and `8gpu` -> `B`: change the copy, or fix the floor?

Both selected a non-contacting antigen copy while a contacting, near-identical one existed:

| structure | selected | contacting alternative |
|---|---|---|
| `7stz` (antibody I/M, 590-res antigen query) | `C`, 540 polymer residues, **5.81 A, 0 atoms within 5 A** | `D`, 422 polymer residues, **2.41 A, 100 atoms within 5 A** |
| `8gpu` (antibody C/D, 398-res antigen query) | `E`, 392 observed, **5.02 A, 0 within 5 A** | `B`, 388 observed, **2.43 A, 105 within 5 A** |

`7stz`'s `D` was rejected by an 85% observed-coverage floor despite carrying the same entity sequence:
missing COORDINATES treated as evidence of a wrong molecule. **Switching the two copies treats the
symptom; changing the floor treats the cause but changes selection everywhere.** I have not swept
whether the floor is silently rejecting correct copies elsewhere, nor checked the other nine
non-contacting chains (5.02-6.78 A) for contacting alternatives. Which should I do?

## 4. Round 3's finding 1 is only half closed

The native-position mapping (`place_by_native_position`) replaced the letter matching, and the
geometric veto catches placements the geometry refuses. What was asked for and NOT built: explicit
chain / sequence / residue / atom identities recorded and compared before symmetry permutation, with
bijective correspondence required over resolved atoms and masked atoms accounted for, plus a
per-residue audit linking source chain and auth residue to output role and position.

So the current checks remain CONSISTENCY checks, as you said -- they establish that the pose and the
label describe the same layout, not that the label is the native molecule. **Is that identity audit a
prerequisite for training, or acceptable to defer with the limitation recorded?** It is real work and
I would rather know before building it than after.

## Not questions, just open

* `e15eadc3`, `38144465`, `fc23b3a2` are unreviewed. Worth saying plainly: every round of my fixes has
  introduced a defect of its own -- the resume offset, a pattern kill that matched my own shell, the
  appender gap, the dict collapse. Four rounds, four.
* whether `tsv2exprofiledb` SUCCEEDS on the real archives is unprovable until it runs.
* the confidence-head training loop does not exist. All of today was making the data correct.
