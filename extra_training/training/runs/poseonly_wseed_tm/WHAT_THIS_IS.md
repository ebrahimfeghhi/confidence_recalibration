# pose-sites-only arm, loss = centred TM error WITHIN a seed

Launched 2026-11-25 12:09 AM EST on card 4, pid 2223084, 50 epochs requested.
Full spec: `training/run_specs/20261125-000728_poseonly_within_seed_tm.md`. Code: commit 5c53d439.

**What is trained:** 18 tensors, 0.066M of 464.5M parameters -- the six pose injection sites and
nothing else (block 0's pretrained pre-stack distance pair, the per-block sites at pairformer
blocks 1-3, and the readout `pre` and `post` sites). `--lean-single` confirmed 0 of the 18 are
gradient-free.

**The only loss** is the per-pair centred TM error, MSE, centred over the 5 diffusion samples of
ONE seed (`--tm-centered --tm-centered-per-seed`). No PAE gradient -- the cross-entropy is
computed as a readout for trajectory comparability and takes none. No ranking loss.

**Why within-seed:** ipTM holds 2.6% of its within-interface variance among a seed's 5 samples,
against DockQ's 49% and iTM's 52%; inside a seed it picks the best of 5 just 25.2% of the time
against a 20% coin flip. Going 1 -> 5 samples per seed raises the oracle 5.7 points (train) and
7.4 (val) while Success@1 does not move.

**Uncropped on purpose:** `--tm-centered` refuses a crop pool, because its d0 would come from the
cropped chain-pair token count while selection reads ipTM over the whole structure. That forces
`--pose-batch 1`; the centring group is still the unit's 5 poses, since pass A sweeps all five to
build the per-pair mean and pass B subtracts it pose by pose.

**Memory:** peak scales with tokens^2 up to the 1250-token checkpoint threshold (1193 tokens ->
69.3 GB) and is bounded above it (1407 -> 50.4 GB). Sampled trunks run to 1745 tokens max, about
77 GB of the 97.4 GB available, so it should hold. If it did OOM, lower
`--ckpt-token-threshold`.

**Card 4 also holds a deliberate 500 MB holder** (pid 2172461). See
`training/runs/DO_NOT_KILL_GPU_HOLDER.md`.
