# experimental_data — layout

Generated map of what is on disk. Most of it is NOT in git: 150 GB of poses (`.cif`), 79 GB of
pose-to-pose DockQ matrices (`.npz`), 12 GB of MSAs (`.a3m`) and 12 GB of per-pose confidence
json. What IS committed: queues, selections, scores, roles and diffs — enough to read the
experiment without rerunning it.

| directory | size | files | poses (.cif) | matrices (.npz) | results.csv |
|:---|---:|---:|---:|---:|:---|
| `anchor_50/` | 1.3 GB | 10,894 | 1,648 | 0 | yes |
| `anchor_frame/` | 3 MB | 7 | 0 | 0 | - |
| `archive/` | 177.2 GB | 1,283,559 | 133,239 | 4,986 | - |
| `free/` | 25.2 GB | 85,133 | 34,805 | 0 | yes |
| `free_100seed/` | 0 MB | 2 | 0 | 0 | - |
| `free_matched/` | 5.2 GB | 21,413 | 6,295 | 0 | yes |
| `inmem/` | 4.6 GB | 23,204 | 8,146 | 0 | - |
| `k50_per_interface/` | 5.5 GB | 44,239 | 6,934 | 0 | yes |
| `matrices/` | 94 MB | 298 | 0 | 292 | - |
| `matrices_anchor_all/` | 76 MB | 254 | 0 | 242 | - |
| `matrices_spur/` | 7 MB | 57 | 0 | 53 | - |
| `method/` | 6.2 GB | 29,220 | 7,880 | 0 | yes |
| `overnight/` | 10.9 GB | 85,291 | 33,263 | 0 | - |
| `periodic/` | 680 MB | 3,032 | 1,502 | 0 | yes |
| `pxm2025/` | 7.9 GB | 36,385 | 131 | 0 | - |
| `reffree_50/` | 1.0 GB | 8,979 | 1,281 | 0 | yes |
| `seedvar/` | 1.6 GB | 14,416 | 2,425 | 0 | yes |
| `self_template/` | 477 MB | 2,358 | 620 | 0 | yes |
| `selftmpl_50/` | 5.1 GB | 43,825 | 6,619 | 328 | yes |
| `single/` | 281 MB | 2,641 | 377 | 0 | yes |
| `top5/` | 3.4 GB | 8,349 | 3,007 | 2 | yes |
| `top_50_complex/` | 2.4 GB | 21,955 | 3,313 | 0 | yes |
