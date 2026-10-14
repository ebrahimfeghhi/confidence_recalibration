# J2 — the wrong copy of the antigen, 2026-10-13

Review round 5's finding J2: "sequence matching can select the wrong native complex ... With several
identical antigen copies, it can choose an antigen that does not contact the selected antibody."

**Confirmed: it affected 71 of 501 structures.** Root cause was discarding SAbDab2's own chain
assignment in favour of sequence matching. Fixed in three steps (names-first, then the two bugs in
the round 6 follow-up below); 71 -> 1, and no refolding needed.

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

## Round 6 follow-up: the six, and the two bugs behind them

The reviewer asked for the six remaining structures to be inspected individually rather than filtered
by threshold. Two of the six were further bugs of the same family; the other four were already fine
and only looked otherwise because the checker was reading a stale file.

### The checker was measuring the wrong file

`phase_j_check_contacts.py` recomputed the chain assignment but measured distances **in the job cif
on disk**, which the job-cif writer had produced from the PREVIOUS assignment. The job cif renames
chains `A, B, C...` in role order, so after a resolution fix the roles printed were the new ones and
the distance was the old one: 9djw's antigen moved from chain `4` to chain `x` and the report still
said 37.28 Å. It now measures on the ENTRY coordinates, which hold the same atoms and keep the chain
names, and separately asserts that the job cif on disk was written from the chains being reported
(`write_job_cif` logs the role -> src chain mapping to `unplaced_residues.csv`). The report also
prints the SOURCE chain and whether it came from SAbDab2's name or from sequence — `chain D` named
nothing in the entry and could not be chased.

### Bug 1 — a trailing copy index on a single-chain construct (8h3c)

SAbDab2 numbers the two domains of a single-chain construct `B1`/`B2`, `A1`/`A2`, `D1`/`D2`. Both are
chain `B`, `A`, `D` in the entry, carrying VH and VL on disjoint stretches. Taking the names
literally left both roles unresolved, so sequence matching placed them — and 8h3c has **five copies
of the scFv** (chains A–E, 223 residues each, every one matching both roles at 100% coverage), so VH
and VL could come from different copies. `role_names()` now retries a name without its trailing digit
when the full name is absent and the stripped one is present, and `job_chains()` lets two roles share
one chain when their spans do not overlap. A genuinely numeric antigen chain (`4`, `12`) is untouched
because its full name does resolve.

### Bug 2 — nothing anchored an all-sequence structure to one copy (9djw)

9djw is a 60-chain VNAR assembly whose chains are named `0`–`7`, `A`–`Z`, `a`–`z`, while SAbDab2 names
two-character chains (`AG` for the antibody, `AE` for the antigen). Neither name exists in the entry,
so BOTH roles fell to sequence: the antigen matched exactly one chain and the antibody matched three,
and the pair that came out sat 37.3 Å apart. `pick_contacting_copies()` now fills the
sequence-resolved roles fewest-candidates-first, each taking the candidate closest to what is already
placed, so the single-candidate role anchors the rest. **Roles resolved by name are never moved**, so
this cannot undo the names-first fix. Greedy, not exhaustive: 9dc3 has 60 candidates for each of two
roles, and any contacting copy pair is an equally valid label.

### Where it ended

| | sequence-first | names-first | + round 6 |
|---|---|---|---|
| antigen chains checked | 627 | 627 | 627 |
| contacting within 5 Å | 509 | 608 | **618** |
| chains not contacting | 118 | 19 | **9** |
| structures with NO contacting antigen | 71 of 501 | 4 | **1** |
| chains resolved by sequence rather than name | 1,535 of 1,535 | 57 of 1,535 | **7 of 1,535** |
| residues dropped as unplaced | 108 across 17 | 2 across 2 | **1 across 1** |

All nine remaining non-contacting chains are **SAbDab2's own named chain**, at 5.02–6.78 Å: 8vy9 6.78,
9md4 6.27, 8jgb 6.11, 9q7b 5.97, 9o4o 5.32, 8vgm 5.31, 8zyx 5.25, 9e3h 5.23, 8gpu 5.02. Eight of the
nine are a second antigen chain in a structure whose other antigen contacts. The ninth, **8gpu**, is
the only structure with no antigen inside the cutoff, and it misses by 0.02 Å.

No structure is dropped and nothing needs refolding: the fold consumes sequences, and every copy of an
antigen has the same sequence.

## Still open

- **Whether 5 Å heavy-atom is the right cutoff.** Nine chains now sit 5.02–6.78 Å, and at that range
  a buried-surface-area or interface-residue criterion would likely call most of them contacts.
  Nothing downstream currently reads `contacts`, so this is a question about the audit, not the data.
- **Regression examples.** The reviewer asked for these. Four structures now stand as the cases each
  bug was found on and should be fixtures: 8h3c (copy index on an scFv, 5 copies), 9djw (two-character
  names, 60 chains), 9dc3 (60 candidates for each of two roles), 7wur (names resolve, 2.21 Å) — with
  the resolved role -> chain mapping asserted.
