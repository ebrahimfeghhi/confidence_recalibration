# J2 — the wrong copy of the antigen, 2026-10-13

Review round 5's finding J2: "sequence matching can select the wrong native complex ... With several
identical antigen copies, it can choose an antigen that does not contact the selected antibody."

**Confirmed: it affected 71 of 501 structures.** Root cause was discarding SAbDab2's own chain
assignment in favour of sequence matching. Fixed; 71 -> 6, and no refolding needed.

## What was measured

`phases/phase_j_check_contacts.py` computes the minimum heavy-atom distance between the chosen
antibody chains (roles H, L) and each chosen antigen chain (roles ag*), on the written job cif.

| | |
|---|---|
| antigen chains checked | 627 over 501 structures |
| contacting the antibody within 5 Å | 509 |
| not contacting | 118 |
| **structures with NO contacting antigen at all** | **71 of 501** |
| non-contacting distances | median 20.8 Å; 28 under 10 Å, 28 at 10–20 Å, 62 over 20 Å |
| worst | 94 Å (9o3d) |

## Why

A crystal routinely holds several copies of the complex. The selector picks chains by SEQUENCE with
an alphabetical tie-break, so it can take the antibody from one copy and the antigen from another:
identical sequences, different assembly, no contact. Every sequence check passes and the label
describes a complex that does not exist in that crystal.

The decisive test: for each failing structure, is there ANOTHER chain in the entry with ≥90% identity
to the chosen antigen that DOES contact the chosen antibody? **In all 25 sampled, yes.** So this is
copy selection and nothing else — no structure needs to be dropped.

## A measurement bug found on the way, worth recording

The first version of the checker used `gemmi.NeighborSearch`, which returns SYMMETRY IMAGES while
`cra.atom.pos` gives the ORIGINAL atom's position. A contact made through a crystal neighbour was
therefore reported as the distance to its far-away original. That produced 118 Å for 7udh where the
true separation is 20.8 Å, and several impossible-looking 100+ Å readings; the reported maximum fell
from 157 Å to 94 Å once replaced with brute force. **The headline counts did not change** — 71
structures, 118 chains — but the numbers quoted before the fix were wrong and had already been given
to the user.

Symmetry mates are exactly what must not count here: the question is whether the antigen touches the
antibody in the asymmetric unit.

## The root cause, found after the first write-up

The first fix I wrote was a contact heuristic: re-point a non-contacting antigen at a copy that does
touch. That treats the symptom. The cause is that **SAbDab2's own chain assignment was being thrown
away**.

An INSTANCE is a specific antibody-antigen pairing, and `Hchain` / `Lchain` / `agchains` name the
chains of THAT pairing, so they contact by construction. 32 of 501 structures have names that do not
resolve in the entry (a trailing copy index, numeric antigen chains, two-character names, a `J000`
label), and to rescue those I switched to SEQUENCE resolution **globally** -- taking the other 469
with it. Sequence cannot tell two copies of a complex apart.

The measurement that settles it: for **67 of the 71** failing structures, SAbDab2's own named chains
contact at **2.3-2.8 A**. Textbook interface distances. The names were right the whole time.

This is exactly what round 5's J2 said -- "use the selected instance's chain assignments where
possible" -- and I did not do it.

## The fix

`named_chains()` resolves the instance's own chain names and `spans_for()` aligns each role's
expected sequence to its named chain. Sequence resolution is now the FALLBACK, used only when a name
is absent, duplicated across two roles, or does not carry that role's sequence. 55 of 60 sampled take
the names.

| | sequence-first | names-first |
|---|---|---|
| structures with no contacting antigen | 71 of 501 | **6** |
| antigen chains not contacting | 118 | **19** |
| residues dropped as unplaced | 108 across 17 structures | **2 across 2** |

The drop in unplaced residues is independent evidence: sequence matching had been selecting chains
whose aligned spans carried residues the fold's sequence does not contain.

**Nothing needs refolding.** The fold consumes `train_jobs`, which carries sequences, and every copy
of an antigen has the same sequence. The 5,010-fold training cache running since 6:27 PM PT is
unaffected; this was label-side throughout.

## Open

- **Six structures still have no contacting antigen**: 7stz, 8cz8, 8gpu, 8h3c, 8yj7, 9djw. Three
  exceed 20 Å and want looking at individually rather than by threshold; three sit at 5.0–6.3 Å.
- The remaining chains at 5–10 Å. Some are second copies of a homodimeric antigen adjacent to the one that
  binds (8vgl, 8vgm at 5.9 and 7.0 Å), which is legitimate. Whether a 5 Å heavy-atom cutoff is the
  right threshold, versus a buried-surface or interface-residue criterion, has not been examined.
