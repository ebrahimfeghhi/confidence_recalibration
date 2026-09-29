# 9h1j seed 300057, fp32 — pair-rep scatter baseline (launched 2026-09-28)

**Why:** starting point for "optimize the pair rep so one pose is reproduced consistently (DockQ scatter < 0.05)".
Need a saved z_trunk/s_trunk and the scatter of the 5 diffusion samples decoded from it.

**Method**
- 9h1j (VHH 6C10 + Sequestosome-1), job json = the pool's own (pxm2025), interface 9h1j_AE
- `diffusion_commit_clean_v2/folder.py --dtype fp32 --ncycle 10 --nsample 5`, seed 300057, Protenix's own PDB templates (kind `free`)
- `LAYERNORM_TYPE=openfold` (mandatory), GPU 6, `PROTENIX_DUMP_TRUNK` -> `trunk/` (float32 z_trunk, s_trunk + RNG state)
- outputs: `poses/9h1j/<seed>-<run id>/predictions/`, `results.csv` (DockQ to native per pose), `trunk/*.npz`

**Known caveat:** folds do not reproduce run to run (folder.py docstring). The original bf16 seed 300057 poses
(native DockQ 0.03,0.03,0.03,0.86,0.89) need NOT reappear in fp32. Whatever fp32 gives is reported as is.

**Scatter measures:** (1) DockQ to native per pose (evaluation only); (2) pairwise pose-to-pose DockQ, symmetrised.

## Result (2026-09-28)
- Fold ran 71 s on GPU 6 (measured ~1.7 cores, 4.4 GB RAM). 688 tokens: the entry has 4 nanobody + 4 peptide copies, not one pair.
- Run dir `poses/9h1j/300057-*/`; trunk saved as `z_s_trunk_fp32.npz` (z_trunk 688x688x256, s_trunk 688x384, float32, no NaN, + RNG state).
- DockQ to native (9h1j_AE), samples 0-4: 0.123, 0.058, 0.096, 0.043, 0.104 (ipTM 0.13-0.23). ALL wrong — the bf16 run's
  0.86/0.89 poses did NOT reappear in fp32 (expected: folds don't reproduce).
- Pairwise DockQ between the 5 samples (`pairwise_dockq.py` -> `pairwise_dockq.json`), each of the 4 copy pairs:
  mean 0.20-0.22, min ~0.08, max 0.49-0.51. The four copies agree with each other. Threshold in question: 0.05.
