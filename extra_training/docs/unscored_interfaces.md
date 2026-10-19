# Antibody-antigen interfaces with no DockQ, and what was recovered

Status: 17 of 22 recovered 2026-10-19. **Five remain unscored and are listed below for the
future.** Nothing here is wrong data -- it is missing data, and the gap was silent until measured.

## How the gap was found

`training/fold/pool.contacting.csv` records every antibody-antigen chain pair the contact audit
found in contact at 5 A. Comparing it against `training/fold/results.csv` showed **28 of 1,037
contacting pairs (2.7%), across 19 structures, had no DockQ at all.**

Two of those structures -- `7stz` and `9blx` -- lost *every* interface this way, so they produced
no rows and fell out of the training manifest entirely. That is how the gap surfaced: they were
noticed as three structures with no interfaces (`7stz`, `8gpu`, `9blx`) while building
`training/full_train/structures.csv`.

## Why it happened

`phase_j_fill_scores.py` already documents the mechanism: OST's chemical grouping cannot separate
the homologous heavy and light immunoglobulin domains, so when the whole complex is scored at once
one target chain is left unmapped and **every interface touching it disappears**. `9blx` is the
clean example -- `chem_groups` came back `[['C'], ['D','E']]`, with the antigen E grouped with the
light chain D, and `chain_mapping` resolved only `{'C':'A','D':'B'}`, dropping E.

Scoring each pair in isolation puts one chain per chemical group, so the mapping is unambiguous.
Every one of the 22 targets re-scored this way resolved to a determinate pair (five of them
ambiguous only by symmetry, handled as best-over-symmetry).

An earlier note in this project said these structures had "no contact in the reference, nothing to
fix". That was wrong: it read OST's `dockq_reference_interfaces`, which reflects what OST's chain
mapping could resolve, not what is in contact. `7stz` A-E has **215** contacts at 5 A and `9blx`
C-E has **1,766**.

## Recovered (17 interfaces, 850 rows, merged into results.csv 2026-10-19)

| interface | roles | DockQ range | mapping |
|---|---|---|---|
| 7sk4_AC | ag0-L | 0.028-0.752 | ambiguous copies |
| 7sk4_BC | ag1-L | 0.030-0.882 | 1x1 |
| 7sk8_AC | ag0-L | 0.006-0.876 | ambiguous copies |
| 7sk8_BC | ag1-L | 0.032-0.872 | 1x1 |
| 7stf_AB | H-ag1 | 0.027-0.332 | 1x3, best-over-symmetry |
| 7stf_AD | H-ag0 | 0.007-0.014 | 1x1 |
| 7t82_AD | ag0-H | 0.008-0.032 | 1x1 |
| 7tr4_AD | ag0-H | 0.018-0.195 | 1x2, best-over-symmetry |
| 7tr4_DE | H-ag2 | 0.046-0.796 | 2x1, best-over-symmetry |
| 8eed_DJ | ag0-L | 0.016-0.336 | ambiguous copies |
| 8r9x_AB | ag0-H | 0.005-0.065 | ambiguous copies |
| 9blx_CE | H-ag2 | 0.090-0.453 | 1x1 |
| 9blx_DE | L-ag2 | 0.065-0.560 | 1x1 |
| 9cye_JL | ag1-L | 0.027-0.838 | ambiguous copies |
| 9mpz_BF | ag0-L | 0.013-0.960 | 1x1 |
| 9mpz_CF | ag1-L | 0.018-0.843 | ambiguous copies |
| 9pix_AB | ag0-H | 0.011-0.134 | 1x1 |

`results.csv` went from 1,011 to 1,028 distinct interfaces, 50,500 to 51,350 rows. Backup at
`training/fold/results.csv.bak.20261019-153556`.

**`9blx` is now eligible for the training manifest again** -- it was excluded only because it had
no scored interface.

## STILL UNSCORED -- flagged for the future

These five produced **no rows** and the tool reported no error. It queued 1,300 OST calls, scored
900 assignments and wrote 850 best rows, so roughly 400 calls returned nothing and were dropped
silently. **That silent partial failure is itself a bug worth fixing** -- a tool that loses five of
twenty-two targets without saying so is how a wrong number gets believed.

| interface | roles | mapping reported | contacts at 5 A | note |
|---|---|---|---|---|
| 7stz_AE | ag1-H | 1x1 | 215 | the only interface this structure has; it stays out of the manifest |
| 8dv6_BD | ag1-L | 2x1, best-over-symmetry | yes | |
| 8xcd_AC | ag0-L | 1x1 | yes | |
| 8zh7_AB | ag0-H | 1x1 | yes | |
| 9owf_AF | ag0-L | 2x1, best-over-symmetry | yes | |

Three of the five reported a determinate `1x1` mapping, so the failure is downstream of chain
resolution -- in the OST call itself, not in the mapping. Diagnosing one of them means running a
single OST call by hand and reading what it raises.

## Also still out

- **8gpu** -- genuinely marginal, not a scoring failure. Its antigen's minimum distance to the
  antibody is **5.02 A** (`training/prep/antigen_contacts.csv`), so it has 0 contacts at the 5 A
  cutoff and both its pairs record 0 in `pool_reference_contacts.csv`. Correctly excluded.
- **6 pairs on structures already excluded by `--manifest-skip`**: `9bbn_FH`, `9mx1_AB`,
  `9mx1_AC`, and three on `9dc3` whose chain ids are multi-character (`AB`, `ID`, `GA`, `VA`),
  which the `pdb:IFACE` target format cannot express unambiguously.
- **9dzj seeds 500000/500001** -- a separate, earlier deferral. The cifs exist; only the chain
  mapping and DockQ are missing.

## The structural lesson

The contact audit (`phase_j_check_contacts.py`) *reports* rather than gates. It knew about every
one of these pairs -- they are all in `pool.contacting.csv` -- while `results.csv`, which is what
every metric reads, did not. Two files disagreeing about which interfaces exist, with nothing
comparing them, is what let a 2.7% gap sit unnoticed. The comparison is three lines and should run
whenever either file is rebuilt.
