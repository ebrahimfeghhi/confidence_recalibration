# Donors: pxm2025, `self_template_fixed_50_seed`

Produced by `analysis/donor_summary.py`. Donor DockQ is against the crystal at each benchmark interface (evaluation only; selection never reads it).

## Donors per complex

| complexes | donors | median | mean | 1-5 | 6-10 | 11-20 | 21-30 | 31-49 | 50 (cap) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 3524 | 31 | 28.7 | 24 | 9 | 16 | 10 | 25 | 39 |

benchmark complexes with no donor: 0

## Donors per class (benchmark interfaces)

Each interface is counted with its complex's donor count: a template is a whole complex.

| class | interfaces | median donors | mean donors | at the cap (50) | under 10 |
|:---|---:|---:|---:|---:|---:|
| decoy | 89 | 50 | 40.7 | 50 | 5 |
| solved | 142 | 13.5 | 20.8 | 28 | 53 |
| nocov | 61 | 50 | 46.2 | 45 | 1 |

## Donor quality per class (benchmark interfaces)

| class | interfaces | any acceptable donor | median best donor DockQ | mean donor DockQ |
|:---|---:|---:|---:|---:|
| decoy | 89 | 69/89 | 0.369 | 0.065 |
| solved | 142 | 142/142 | 0.786 | 0.417 |
| nocov | 61 | 0/61 | 0.057 | 0.019 |

donor-interface pairs without a crystal score: 0
