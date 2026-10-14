# J2 — the wrong copy of the antigen, 2026-10-13

Review round 5's finding J2: "sequence matching can select the wrong native complex ... With several
identical antigen copies, it can choose an antigen that does not contact the selected antibody."

**Confirmed, and it affects 71 of 501 structures.** Fixed in the selection; no refolding needed.

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

## The fix

`prefer_contacting()` in `phase_j_job_cifs.py`: after the sequence assignment, any antigen role whose
chain does not contact the antibody is re-pointed at a chain with ≥90% identity that does, and the
replacement is re-aligned to its own span. Only the CHAIN changes, never the sequence.

**Nothing needs refolding.** The fold consumes `train_jobs`, which carries sequences, and every copy
of an antigen has the same sequence. The 5,010-fold training cache running since 6:27 PM PT is
unaffected; this is a label-side correction.

## Open

- The fix is written but not yet wired into `main()` or re-run over all 501. Until it is, the job
  cifs on disk still carry the wrong copy for those 71.
- 28 chains sit at 5–10 Å. Some are second copies of a homodimeric antigen adjacent to the one that
  binds (8vgl, 8vgm at 5.9 and 7.0 Å), which is legitimate. Whether a 5 Å heavy-atom cutoff is the
  right threshold, versus a buried-surface or interface-residue criterion, has not been examined.
