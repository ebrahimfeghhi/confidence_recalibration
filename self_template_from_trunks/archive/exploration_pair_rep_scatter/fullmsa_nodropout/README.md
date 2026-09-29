# 9fww + 8fxv, self-template, g=1, full MSA, no MC dropout (launched 2026-09-28)

**Why:** user asked to run both interfaces with no MSA subsampling and no dropout, to remove those sources of variation.

**What was found in the code (not assumed):**
- MSA rows are subsampled on every MSA-module call: sample_size ~ U[min_size(test)=1, n] rows, random order (`model/utils.py sample_indices`).
  Override: `PROTENIX_MSA_NROWS=<k>` pins sample_size=min(k,n); `PROTENIX_MSA_STRATEGY=topk` keeps rows in file order. cutoff = 16384.
- The only dropout active at inference is MC dropout on the recycled z: per fold, with prob `mc_dropout_apply_rate` (config 0.4), every
  recycle applies F.dropout(p=0.4). The MSA/pair module Dropout classes are identity in eval mode. No env override existed:
  added `PROTENIX_MC_DROPOUT_APPLY_RATE` to `diffusion_commit_clean_v2/folder.py` build_runner (backup `../folder.py.before_mcrate`; unset = unchanged).

**Settings:** PROTENIX_MSA_NROWS=1000000 (=> all n rows), PROTENIX_MSA_STRATEGY=topk, PROTENIX_MC_DROPOUT_APPLY_RATE=0, PROTENIX_TMPL_GAIN=1.0,
PROTENIX_LOG_MC=1, PROTENIX_LOG_MSA=1 (logged so the settings can be verified from the logs). Same donors, templates, seed 430000, bf16,
ncycle 10, nsample 5 as `../9fww_tmpl_gain/g1` and `../8fxv_tmpl_gain/g1` (the comparison arms).
