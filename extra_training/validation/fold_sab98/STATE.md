# fold_sab98 — prep is good, the fold is BROKEN. Do not relaunch, do not delete.

**Status 2026-11-24 8:46 PM EST:** stopped. 1,140 units attempted, **0 scored**, `results.csv`
empty. All three workers are dead and the tmux sessions are closed.

**The bug is in our job JSONs, not in the fold.** Every unit died at `run_model/folder.py:305`
with `ArmError: no template count reported -- featurisation did not run`, because
`run_logging/build_sab98_jobs.py` omits **`templatesPath`** from each chain. Compare
`validation/jobs_corrected/22gi-final-updated.json`, which has it. With no `templatesPath`
Protenix skips the template featuriser, never logs `Found N templates`, and the guard fires.

**Two decisions are open and both belong to the owner:**

1. **Templates on or off.** val83 folded WITH templates (7,700 chain-instances at `Found 4
   templates`); the Breadth-over-Depth paper disabled retrieval entirely. Off = comparable to
   their Table S1 Protenix-v2 row; on = comparable to our own val83 poses.
2. **Target unit.** One target per PDB file is wrong — the paper's antibody is `{H, L}`, singular.
   `run_logging/validate_sab98_targets.py` yields **117 instance-level targets over 96
   structures**; see `scratchpad/sab98_targets.json`.

**What is worth keeping here:** 98 job JSONs, `pool.csv` (352 rows), `chain_roles.csv`,
`queue.tsv` (currently the 57 "clean" targets) and `queue_all98.tsv` (the full 1,960-unit queue),
plus 98 hardlinked references in `../refs_sab98/`. All MSAs are already fetched — nothing needs
the network again.
