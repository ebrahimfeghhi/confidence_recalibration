# Steering z_trunk / s_trunk toward a chosen pose (summary, written 2026-09-30, state as of ~13:15)

## What this is
Protenix makes each pose from a **trunk state** (`z_trunk` pair rep, `s_trunk` single rep) followed by a diffusion sampler. For one seed the sampler can land in several
docking arrangements from the same trunk state. The question: can `z` and `s` be **edited directly** so that the sampler reproduces ONE chosen pose every time, and how big an
edit does that take? Nothing in the model weights changes; only `z` and `s` are optimised.

## Method (validated on 8w5q seed 300019, then used for the FoldBench decoy set)
- **Loss** = Protenix's own training loss for the diffusion module, one denoising step, no trunk in the graph: aligned MSE (target rigidly aligned onto the denoised structure, EDM
  weight (sigma^2+16^2)/(16 sigma)^2) + 1.0 x smooth-lDDT, x4 (alpha_diffusion). Noise levels come from the training sampler (sigma = 16 exp(-1.2 + 1.5 n)).
- **Draws per step** = 48 noise draws (Protenix `diffusion_batch_size`), run as micro-batches of 16 with gradient accumulation.
- **Optimiser**: Adam on z and s directly (unconstrained), betas (0.9, 0.95), eps 1e-12, 10-step warmup, grad-clip 10, bf16 autocast. `lr = mult x (0.0018/0.0966) x RMS(entry)` per
  tensor (Protenix training lr over the median weight RMS, applied to z's and s's own scale); mult 0.3 or 1.0.
- **Evaluation**: decode 20 fresh poses with the normal 200-step sampler (different noise seed from training), score DockQ against the crystal, RMSD to the target, ipTM.
- **Target** = a pose the seed already produced (a donor pose). The crystal is used only to score, never to choose targets or donors.

## Code (all under /home/ebrahim/tts_abag)
- `diffusion_commit_clean_v2/steer_reps.py` - the optimiser + helper functions (audited; `steer_reps.py.before_audit` / `.before_recipe` are the earlier versions).
- `diffusion_commit_clean_v2/tests/check_steer_reps.py` - end-to-end checks (all pass): checkpoint load, target lookup, confidence path reproduces saved ipTM/pLDDT, loss, gradients.
- `diffusion_commit_clean_v2/sample_from_trunk.py` - draw extra poses from a saved trunk state.
- `diffusion_commit_clean_v2/decode_and_score.py` - decode + DockQ scatter for a set of checkpoints.
- `gradient_based_commit/` - the FoldBench run: `scripts/steer_donors.py` (worker), `scripts/select_donors.py` (crystal-free leaders, no templates), `scripts/run_decoy_set.py` (controller),
  `recipe_test.sh`, `recipe_report.py`, `chain_overnight.sh`, `RUN.md` (full method record), `controller.log`, `complex_summary/<pdb>.json`.
- `steer_8w5q_300019_full/` - the 8w5q experiments (RUN.md inside).

### Bugs found and fixed in steer_reps.py during the audit
1. Loss compared the denoised structure with the UN-aligned target, but sample_diffusion_training returns x_denoised in a randomly rotated frame -> the loss measured rotation, not pose
   (raw loss 3544 vs aligned 0.013). Now uses Protenix's MSELoss (aligns) + SmoothLDDT.
2. z/s/s_inputs were passed with an extra batch dimension (crash in prepare_cache). 3. Checkpoints were saved as float16 (would quantise away small edits). 4. No check for
LAYERNORM_TYPE=openfold (silent NaN). 5. load_state_dict(strict=False) silently ignored missing weights. 6. Target was chosen by dump index, not by pose (CIF order differs from dump order).

## Results
### 8w5q seed 300019 (502 tokens; 5 original poses, only the top-ranked one near-native)
- Steering to the near-native pose (CIF sample_0, DockQ 0.62/0.67/0.78 on AC/BC/BD): all 20 decoded poses land on it at lr x0.1, x0.3 and x1 (DockQ SD <= 0.007; unedited state: ~half the poses good,
  SD ~0.3). Edit to z 6.1% / 10.6% / 20.3%. ipTM of the decoded poses 0.627 / 0.642 / 0.656 (unedited 0.567; original pose 0.610).
- Steering to a DECOY (CIF sample_1, DockQ 0.011/0.023/0.023): all 20 poses land on it (RMSD to target 0.06-0.11 A), edit 5.0 / 9.0 / 20.3%. ipTM 0.529 / 0.545 / 0.582 (unedited 0.560; original decoy 0.520).
  None of the 20 unedited poses was within 1 A of that decoy.
- Learning rate matters in relative terms: the first attempt (lr 0.01, 4 draws, eps 1e-8) barely moved z (per-entry lr 0.05% of z's scale, per-entry gradients below Adam's eps).

### FoldBench confident-decoy set (22 complexes of the ORIGINAL pool: top-ipTM pose wrong, acceptable pose exists)
- 4 complexes were folded for 50 new seeds (101-150) with full dumps; the new pool does NOT reproduce the original seed-for-seed (8aok: 9/50 seeds identical; 7ucx 25/50; 8fas ~6-7/50).
  Distributions match for 7ucx and 8fas, not for 8aok (good-pose rate 16% vs 44%). 7ucx and 8vui are no longer 'decoy' in the new pool (best DockQ 0.225 and 0.117).
- Donors (crystal-free leaders, cap 50): 8aok 8, 7ucx 5, 8fas 10, 8vui 34. Donor quality vs crystal: 7/18 of the first 18 acceptable (>=0.23) on >=1 interface; one near-native (8fas A03: 0.95/0.97).
- Recipe test (23 donors of 8aok/7ucx/8fas; pass rule fixed beforehand: mean RMSD to target <0.5 A, max <1.0 A, DockQ SD <0.05, |mean DockQ - donor's| <= 0.02):
  200 steps 18/18 (248 s/donor); 100 steps 23/23 (155 s); 50 steps 23/23 (99 s); 25 steps 23/23 (71 s); 10 steps lr x1.0 23/23 (55 s); 10 steps lr x0.3 only 9/23. Chosen: 10 steps, lr x1.0.
- Controller (`run_decoy_set.py`), smallest complex first, all four GPUs when idle: fold 50 seeds -> select donors -> steer -> verify -> delete dumps (full/, trunks/, final_zs.npz).
  Safeguards: rescue pass (donors failing the rule are rerun at 200 steps), escalation ladder 10->25->50->100->200 steps between complexes when a complex's first-pass rate < 0.85, OOM fallback.
- Complexes finished so far (donors / pass / rescued): 8aok 8/8/0 (200 steps), 7ucx 5/5/0 (10), 8fas 10/10/0 (200), 9jbq 29/29/10 (10), 8vui 34/32/7 (25), 8av2 38/38/1 (50), 8slb 50/40/25 (50), 8yxi 50/48/3 (100).
  (updated 15:10) Running: 8oxx at 100 steps. Remaining: 8fbw, 8che, 8ux6, 8ivx, 8oxw, 8the, 8q3j, 8be1, 8bf4, 8udz, 8pn0, 7z2m, 7zau.

## Findings / caveats to keep in mind
- **All** criterion failures (46/192 first-pass, 12/43 after rescue) are on the whole-complex RMSD limits only; none involves a DockQ condition; DockQ of the decoded poses matches the donor's
  within 0.014 in those cases. The RMSD limits were added by the assistant; the user's own stopping criterion was DockQ variance < 0.05. Open question: drop/loosen the RMSD clauses (would make cheap recipes sufficient).
- The 10-step recipe passed 23/23 on small complexes (255-462 tokens) but only 19/29 on 9jbq (573) and 27/34 at 25 steps on 8vui (668): cost/robustness depends on size.
- ipTM after steering: the numbers above are for the DECODED poses (mean over 20). The ipTM of the ORIGINAL donor pose given the optimised z,s (teacher-forced through the confidence head) has NOT been computed;
  the optimised z,s of 8aok, 7ucx, 8fas were deleted by the clean-up stage (only their summaries/decoded poses remain).
- Everything is one complex at a time; conclusions are descriptive, not interpreted.

## Live run: what is running where (snapshot 2026-09-30 15:10; historical; refresh with `gradient_based_commit/scripts/status.sh`)
- One controller process (`run_decoy_set.py`, pid 530518, started via `relaunch_controller.sh` after the ladder patch). It works on ONE complex at a time and uses every allowed GPU that is idle (allowed: 0, 1, 2, 5).
- Current complex: **8oxx** (899 tokens, 50 donors, 50/50 seeds dumped), stage **steering at 100 steps / 48 draws / lr x1.0** (ladder rung 3 after 8slb's first-pass rate 0.50).
- Each GPU runs one `steer_donors.py --pdbs 8oxx --out steer_final ...` worker (~29 GB each); workers claim donors by mkdir, so they take different donors:
  GPU 0 pid 2691191 (was on 8oxx_A17), GPU 1 pid 2691192 (A16), GPU 2 pid 2691193 (A19), GPU 5 pid 2691194 (A18). 16 of 50 first-pass summaries done; no rescues yet.
- Stage order per complex: fold 50 seeds (all 4 GPUs, tmux `fbc_<pdb>w<gpu>`) -> donor selection (CPU, or 1 GPU if >1000 tokens) -> steering (all 4 GPUs) -> rescue (200 steps) -> verify + delete dumps.
- Stop cleanly: `touch gradient_based_commit/STOP` (finishes the current complex, then exits). Logs: `controller.log`, `logs/steer_<pdb>_gpu<N>.log`, `logs/rescue_<pdb>_gpu<N>.log`.

## STOPPED (2026-09-30 15:50)
The sweep was stopped at the user's request (controller, relaunch script and the 4 workers terminated; GPUs 0,1,2,5 idle). Finished and cleaned: 8aok, 7ucx, 8fas, 9jbq, 8vui, 8av2, 8slb, 8yxi.
8oxx was partway through first-pass steering (32/50 donors with a summary; its fold dumps are kept). The 'Live run' section above is a historical snapshot. To resume see gradient_based_commit/docs/RUN.md (last entry).
