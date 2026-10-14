# foldbench free seeds

`results.csv` is every free pose-level score for this benchmark in one table: 113568 rows over 160 interfaces. Scorer `ost` (paths.py). Columns `pdb, chain1, chain2, seed, sample, dockq, pair_iptm, block, tier`.

## Seed depth by tier

| tier filter | min | median | max |
|---|---|---|---|
| `tier == "pool"` (the published free-50 baseline) | 50 | 50 | 50 |
| `tier != "suspect"` | 59 | 68 | 100 |
| everything | 100 | 110 | 559 |

Rows are seeds per interface. **Reproduce a published number with `tier == "pool"`.** A deeper read is legitimate but has never been the baseline, so if it disagrees with the pool that is a result to explain, not a number to quote.

## Blocks

| block | tier | rows | pose cifs | seeds | where the poses are |
|---|---|---|---|---|---|
| pool | pool | 39998 | 0 | 101-150 | deleted |
| top100 | topup | 12285 | 7770 | 600000-600039 | /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_top100 |
| extra10 | topup | 7995 | 15 | 501-510 | /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_extra10 |
| matched | suspect | 21895 | 1750 | 151-200 | /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free |
| other_seeds | suspect | 31395 | 4215 | 0-2036 (6 blocks) | /home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_other_seeds |

`suspect` means the settings match but the sampling is uneven -- those seeds exist only for complexes someone had experimented on before, and they measured 0.060 worse on mean DockQ than the pool for that reason. Keep them out of any baseline.

Left out deliberately: `pxm2025/free` arms `freebf16r1/r2` and `freefp32r1/r2`, 200 rows that refold pool seeds in two precisions. Every row key is already in `pool_csv`, so pooling them would count a seed twice; they belong to the determinism analysis.

Built by `diffusion_commit_clean_v2/build_free_pool.py`. No pose was moved or deleted.
