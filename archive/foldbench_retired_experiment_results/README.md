# Retired experiment directories — 2026-10-14

Fifteen finished experiment directories under `experimental_data/protenix/foldbench/` were deleted on
user instruction: 21 GB, 36,135 poses. Their **scored results** (every `.csv` and `.md` they held,
114 files, 17 MB) were copied here first, under their original directory names, so the write-ups that
cite them keep their provenance.

| deleted | size | poses | cited elsewhere |
|---|---|---|---|
| `recycle_sweep` | 15 GB | 33,190 | 0 |
| `self_template_fixed_50_seed` | 2.8 GB | 0 | 39 |
| `confornet_xchain` | 1.7 GB | 2,930 | 0 |
| `recycle_ladder_8zp2` | 579 MB | 15 | 0 |
| `resume_8vyl_s101` | 522 MB | 0 | 2 |
| `selftmpl_mask_on_nc10` | 41 MB | 0 | 14 |
| `selftmpl_mask_on` | 28 MB | 0 | 23 |
| `self_template_all24_7zoz` | 14 MB | 0 | 2 |
| `self_template_siblings_7zoz` | 9.0 MB | 0 | 2 |
| `self_template_family_7zoz` | 8.4 MB | 0 | 1 |
| `pose_z_rescore` | 8.1 MB | 0 | 1 |
| `self_template_msa1_broken` | 3.1 MB | 0 | 2 |
| `seed_scatter_50` | 1.8 MB | 0 | 9 |
| `self_template_donor_seed_7zoz` | 372 KB | 0 | 2 |
| `rerun_8zp2` | 116 KB | 0 | 0 |

**The poses are gone for good — folds do not reproduce on this hardware.** 33,190 of the 36,135 were
`recycle_sweep`'s, which nothing cited; the rest were `confornet_xchain`'s 2,930 and 15 in
`recycle_ladder_8zp2`.

Write-ups whose stated data source was one of these still read correctly against the tables here:
`experimental_results/self_templating_summary.md` names `self_template_donor_seed_7zoz/`,
`self_template_siblings_7zoz/` and `self_template_all24_7zoz/`; `arms_compare.md`,
`arms_compare_oracle.md` and `arm_statistics_pxm2025.md` name `self_template_fixed_50_seed`.

None of the fifteen was read by any live code path: no reference from `extra_training/`, `run_model/`,
`paths.py`, or any current analysis.
