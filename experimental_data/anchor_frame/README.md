# Reference-free frame: what is in here and how the pieces fit

Everything here exists to answer one question: can donor selection run **without the crystal**? The
selection algorithm is unchanged (sort per interface by `chain_pair_iptm`, keep a pose whose pose-pose
DockQ to everything kept is < 0.23, round-robin across interfaces, cap 50). What changes is where the
two pieces of reference information come from.

| what the crystal used to supply | replacement | status |
|:---|:---|:---|
| which model chain pair IS interface A-B | the complex's highest-mean-ipTM POSE, as a naming anchor | verified: 42/73 donor sets identical, 2/242 interfaces flip coverage |
| which chain pairs ARE interfaces | pairs in contact within 5.0 A in >= 1 pose | recovers 292/292, declares 53 extra |

Evaluation still uses the crystal, and should: GT DockQ and the decoy/solved/nocov labels are
measurements, not inputs to selection.

## Files

| file | contents |
|:---|:---|
| `chain_maps_anchor.csv` | `{anchor_chain: pose_chain}` per pose, from `anchor_relabel.py` |
| `chain_roles_anchor.csv` | the 242 multi-interface reference interfaces, ab/ag in ANCHOR chain names |
| `chain_roles_all.csv` | the above plus the 53 predicted-only pairs (295 rows) |
| `pose_interfaces.csv` | every contacting pair, with `poses_in_contact` and `have_gt` |
| `spurious_pool.csv` | pool-shaped rows for the 53, `pair_iptm` from the confidence json, `dockq` EMPTY |
| `spurious_roles.csv` | roles for the 53 alone |

## Two traps, both hit once

**Matrix filenames collide between frames.** `chain_roles_anchor.csv` names an interface by the
CRYSTAL's `chain_1/chain_2`, while `spurious_roles.csv` names one by its ANCHOR chains. So the
predicted-only pair `8w5i_AD` produces the same filename as the real interface `8w5i_AD`, which is a
different interface. Six of the 53 were silently skipped because the real matrix already sat at that
path -- nothing was corrupted only because `build_matrices` skips an existing file rather than
overwriting it. **Predicted-only matrices therefore live in `../matrices_spur/`, never mixed with
`../matrices_anchor_all/`.** Identity is `(pdb, ab, ag, is_predicted_only)`, and a bare
`pdb_<c1><c2>` string cannot express it.

**`min_poses` was a hidden knob.** `build_matrices` dropped any interface with fewer than 5 contacting
poses -- a rule written for basin clustering, where a matrix over 4 poses means nothing. Once the
interface set came from pose contacts, that rule silently became an interface filter and removed 11 of
the 53. It is now `--min-poses`, and these were built with `--min-poses 1` so nothing is filtered by a
threshold that was never chosen for this purpose.

## Where the 53 are

22 of 123 complexes; 101 are untouched. Concentrated: `9jo7` has 13 (against 10 real), `9k94` 5,
`8w5i` 4, `9quw` 4, `9x1h` 4. The rest have 1-2.

Contacts per predicted-only pair: min 1, **median 107**, max 367 of ~250-375 poses. A pair the model
builds in 107 poses is not a mis-docking artifact -- it is something the model does consistently.
Whether those are model error or interfaces the FoldBench reference omits is **unresolved**, and it
matters: if the latter, they are not spurious and `9jo7`'s 13 are a gap in the benchmark.

## Open

The go/no-go measurement: selection across all 345 interfaces, scored on the 292 evaluable ones,
against 292-way selection. Round-robin gives every interface an equal share of 50 donors, so the 53
extras take budget from the real ones. Dilution is local -- on `9jo7` the budget is more than halved,
on 101 complexes nothing changes -- so the number to look at is coverage on the 22 affected complexes,
not a benchmark-wide average.
