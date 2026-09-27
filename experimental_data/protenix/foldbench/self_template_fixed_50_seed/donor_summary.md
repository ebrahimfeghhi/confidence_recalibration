# Donors: foldbench, `self_template_fixed_50_seed`

Produced by `analysis/donor_summary.py`. Donor DockQ is against the crystal at each benchmark interface (evaluation only; selection never reads it).

## Donors per complex

| complexes | donors | median | mean | 1-5 | 6-10 | 11-20 | 21-30 | 31-49 | 50 (cap) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 104 | 3209 | 32.500 | 30.9 | 15 | 8 | 13 | 14 | 10 | 44 |

benchmark complexes with no donor: 0

## Donor quality per class (benchmark interfaces)

| class | interfaces | any acceptable donor | median best donor DockQ | mean donor DockQ |
|:---|---:|---:|---:|---:|
| decoy | 28 | 26/28 | 0.412 | 0.097 |
| solved | 119 | 119/119 | 0.799 | 0.342 |
| nocov | 13 | 0/13 | 0.067 | 0.019 |

donor-interface pairs without a crystal score: 0
