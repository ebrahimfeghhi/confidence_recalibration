# pxm2025 free seeds

`results.csv` is every free pose-level score for this benchmark in one table: 177380 rows over 292 interfaces. Scorer `pxm` (paths.py). Columns `pdb, chain1, chain2, seed, sample, dockq, pair_iptm, block, tier`.

## Seed depth by tier

| tier filter | min | median | max |
|---|---|---|---|
| `tier == "pool"` (the published free-50 baseline) | 50 | 50 | 75 |
| `tier != "suspect"` | 100 | 120 | 145 |
| everything | 100 | 120 | 145 |

Rows are seeds per interface. **Reproduce a published number with `tier == "pool"`.** A deeper read is legitimate but has never been the baseline, so if it disagrees with the pool that is a result to explain, not a number to quote.

## Blocks

| block | tier | rows | pose cifs | seeds | where the poses are |
|---|---|---|---|---|---|
| pool | pool | 81670 | 0 | 300000-300094 | deleted |
| top100 | topup | 2305 | 1450 | 300601-300639 | /home/ebrahim/tts_abag/experimental_data/protenix/pxm2025/free_top100 |
| extra10 | topup | 25110 | 0 | 300501-300520 | deleted |
| free100 | topup | 46265 | 895 | 300050-300129 | /home/ebrahim/tts_abag/experimental_data/protenix/pxm2025/free |
| free149 | topup | 22030 | 895 | 300100-300149 | /home/ebrahim/tts_abag/experimental_data/protenix/pxm2025/free |

`suspect` means the settings match but the sampling is uneven -- those seeds exist only for complexes someone had experimented on before, and they measured 0.060 worse on mean DockQ than the pool for that reason. Keep them out of any baseline.

Left out deliberately: `pxm2025/free` arms `freebf16r1/r2` and `freefp32r1/r2`, 200 rows that refold pool seeds in two precisions. Every row key is already in `pool_csv`, so pooling them would count a seed twice; they belong to the determinism analysis.

Built by `diffusion_commit_clean_v2/build_free_pool.py`. No pose was moved or deleted.
